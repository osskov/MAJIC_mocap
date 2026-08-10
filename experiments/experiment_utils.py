"""
Shared engine for every "run the filters over subjects/activities" script in this
repo — the vanilla full-grid pipeline (experiments/benchmark_experiment.py) and every
experiment under experiments/ (noise sensitivity, threshold sensitivity, oracle
ablation, drift observability). Anything that's physics, IO, or orchestration and
would otherwise be copy-pasted or reached into via private imports lives here.

All filesystem paths come from paths.py: `data/` is read-only source data and every
generated artifact lands under `results/`, with a provenance manifest sidecar.
"""
import os
from pathlib import Path
os.environ["DISABLE_TQDM"] = "True"
import re
import time
import multiprocessing
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from rich.live import Live
from rich.table import Table

import paths
from paths import DATA_DIR, RESULTS_DIR, raw_trial_dir, ensure_parent, write_manifest
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from src.RelativeFilterPlus import RelativeFilter

# ==============================================================================
# CONFIGURATION
# ==============================================================================

JOINTS = {
    'Lumbar': ('pelvis_imu', 'torso_imu'),
    'R_Hip':  ('pelvis_imu', 'femur_r_imu'),
    'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
    'R_Ankle':('tibia_r_imu', 'calcn_r_imu'),
    'L_Hip':  ('pelvis_imu', 'femur_l_imu'),
    'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
    'L_Ankle':('tibia_l_imu', 'calcn_l_imu'),
}

SUBJECTS = [f'{i:02d}' for i in range(1, 12)]
ACTIVITIES = ['walking', 'complexTasks']

# Gravity as an accelerometer reads it, expressed in the mocap world frame.
#
# Two conventions are baked in here and both matter:
#   * Axis   — this dataset's world frame is Y-UP (the .trc mocap frame). Measured
#              across all 11 subjects x 151 sensor-trials, the mean world-frame
#              accelerometer vector is [+0.016, +9.812, -0.003] (sd 0.035).
#   * Sign   — POSITIVE, i.e. the specific-force convention: a stationary sensor
#              reads +9.81 along the up axis, not -9.81. Note this is the opposite
#              sign to the `acc_from_gravity=[0, 0, -9.81]` used by the synthetic
#              fixture generators in src/toolchest/PlateTrial.py, which live in
#              their own self-consistent synthetic world and are test-only.
#
# Getting either wrong is silent: it does not raise, it just corrupts the EKF's
# world reference and both acc oracles. Verified against this dataset by
# test/TestGravityConvention.py; if you point the pipeline at different data, run
# check_gravity_convention() below against it before trusting any output.
EXPECTED_GRAVITY = np.array([0.0, 9.81, 0.0])

# Default filter tuning. Hoisted out of _run_relative_filter's signature so the
# values can be recorded in every output's provenance manifest — their meaning
# depends on whether RelativeFilter normalizes its vector measurements, which is
# not visible from here.
DEFAULT_GYRO_STD = 0.0045
DEFAULT_ACC_STD = 0.037
DEFAULT_MAG_STD = 0.03
DEFAULT_MAG_ADAPT_THRESHOLD = 150.0


def pipeline_constants() -> Dict[str, Any]:
    """The physical/tuning constants in force, for provenance manifests."""
    return {
        'expected_gravity': EXPECTED_GRAVITY.tolist(),
        'gyro_std': DEFAULT_GYRO_STD,
        'acc_std': DEFAULT_ACC_STD,
        'mag_std': DEFAULT_MAG_STD,
        'mag_adapt_threshold': DEFAULT_MAG_ADAPT_THRESHOLD,
    }

METHODS = {
    'marker':      {'kind': 'marker'},
    'mag_on':      {'kind': 'filter', 'project': True,  'mag_mode': 'on'},
    'mag_off':     {'kind': 'filter', 'project': True,  'mag_mode': 'off'},
    'mag_adapt':   {'kind': 'filter', 'project': True,  'mag_mode': 'adapt'},
    'ekf':         {'kind': 'ekf'},
}

# ==============================================================================
# Method name resolution
# ==============================================================================
# A method name is a base (one of METHODS) plus optional suffixes, in this fixed
# order: an observability threshold override (mag_adapt only), then an accelerometer
# oracle flag, then a magnetometer oracle flag. Examples:
#   'mag_adapt_th50.00'                    -> base mag_adapt, threshold=50.0
#   'ekf_perfect_acc_perfect_mag'           -> base ekf, both oracle
#   'mag_on_real_acc_perfect_mag'           -> base mag_on, mag oracle only
#   'mag_off_perfect_acc'                   -> base mag_off, acc oracle only (mag_off
#                                              always zeroes mag regardless, so a mag
#                                              suffix would be a no-op and is omitted)

_METHOD_SUFFIX_RE = re.compile(
    r'^(?P<base>.+?)'
    r'(?:_th(?P<threshold>[\d.]+))?'
    r'(?:_(?P<acc_src>real|perfect)_acc)?'
    r'(?:_(?P<mag_src>real|perfect)_mag)?$'
)


def resolve_method_spec(method: str) -> Dict[str, Any]:
    """Parses a method name into a spec dict: the base METHODS entry plus
    'acc_source' / 'mag_source' ('real' or 'perfect', default 'real') and, for
    mag_adapt, an overridden 'mag_adapt_threshold' if a _th<value> suffix is present."""
    m = _METHOD_SUFFIX_RE.match(method)
    base = m.group('base')
    if base not in METHODS:
        raise ValueError(f"Unknown method '{method}' (base '{base}' not recognized). Allowed bases: {list(METHODS)}")
    spec = dict(METHODS[base])
    spec['acc_source'] = m.group('acc_src') or 'real'
    spec['mag_source'] = m.group('mag_src') or 'real'
    if m.group('threshold') is not None:
        spec['mag_adapt_threshold'] = float(m.group('threshold'))
    return spec

# ==============================================================================
# Raw data loading
# ==============================================================================

def load_raw_data(subject: str, activity: str) -> Dict[str, PlateTrial]:
    return PlateTrial.from_folder(raw_trial_dir(subject, activity))

# ==============================================================================
# Gravity convention check
# ==============================================================================

def measure_world_frame_gravity(plates: Dict[str, PlateTrial]) -> np.ndarray:
    """Mean accelerometer reading rotated into the mocap world frame, averaged over
    every sensor and sample in the trial.

    Over a full trial the subject starts and ends at rest and linear accelerations
    average out, so this converges on gravity as the accelerometers report it —
    which is exactly what EXPECTED_GRAVITY is supposed to be. Measured spread
    across this dataset is 0.035 m/s^2, so it is a sharp check, not a fuzzy one.
    """
    world_accs = [
        np.einsum('nij,nj->ni', plate.world_trace.rotations, plate.imu_trace.acc).mean(axis=0)
        for plate in plates.values()
    ]
    return np.mean(world_accs, axis=0)


def check_gravity_convention(plates: Dict[str, PlateTrial], tol: float = 0.5,
                             context: str = "") -> np.ndarray:
    """Verifies EXPECTED_GRAVITY against a loaded trial, raising on a mismatch.

    Not called by the pipeline — the convention is settled for this dataset and
    covered by test/TestGravityConvention.py. Kept for the tests and for the case
    where the pipeline is pointed at new data, where an axis or sign mismatch would
    otherwise pass silently and corrupt the EKF world reference and both acc oracles.
    """
    measured = measure_world_frame_gravity(plates)
    deviation = float(np.linalg.norm(measured - EXPECTED_GRAVITY))
    if deviation > tol:
        where = f" for {context}" if context else ""
        raise ValueError(
            f"Gravity convention mismatch{where}: EXPECTED_GRAVITY is "
            f"{EXPECTED_GRAVITY.tolist()} but the data's mean world-frame accelerometer "
            f"reads [{measured[0]:+.3f}, {measured[1]:+.3f}, {measured[2]:+.3f}] "
            f"(off by {deviation:.3f} m/s^2, tolerance {tol}).\n"
            f"Either this dataset uses a different world-frame axis or sign convention, "
            f"or EXPECTED_GRAVITY in experiments/experiment_utils.py is wrong. Set it to the measured "
            f"vector above if the dataset is correct."
        )
    return measured

# ==============================================================================
# Physics: oracle ("perfect" acc/mag) helpers
# ==============================================================================

def _compute_expected_mag_field(plate_trials: List[PlateTrial]) -> np.ndarray:
    """Median world-frame magnetic field across all pelvis-mounted IMU readings."""
    all_global_mags = [
        (plate.world_trace.rotations @ plate.imu_trace.mag[..., None])[..., 0]
        for plate in plate_trials if 'pelvis' in plate.name
    ]
    return np.median(np.concatenate(all_global_mags, axis=0), axis=0)


def _setup_ekf_ground_plate_(plate_trials: List[PlateTrial]) -> PlateTrial:
    """Precomputes global expected gravity and magnetic field to build a virtual parent ground plate."""
    base_plate = plate_trials[0]
    expected_mag = _compute_expected_mag_field(plate_trials)

    ground_plate = base_plate.copy()
    ground_plate.name = "ground"
    ground_plate.world_trace.rotations = np.tile(np.eye(3), (len(base_plate), 1, 1))
    ground_plate.imu_trace.gyro = np.zeros_like(base_plate.imu_trace.gyro)
    ground_plate.imu_trace.acc = np.tile(EXPECTED_GRAVITY, (len(base_plate), 1))
    ground_plate.imu_trace.mag = np.tile(expected_mag, (len(base_plate), 1))
    return ground_plate


def _compute_perfect_segment_acc(plate: PlateTrial, gravity: Optional[np.ndarray] = None) -> np.ndarray:
    """Oracle acc for a lone segment (EKF path): gravity rotated into the segment's
    ground-truth orientation, with no linear-acceleration component.

    gravity defaults to EXPECTED_GRAVITY, resolved at call time rather than bound as
    a default argument — a default would freeze the value at import and silently
    ignore any override, which makes the constant untestable."""
    gravity = EXPECTED_GRAVITY if gravity is None else gravity
    return np.einsum('nji,j->ni', plate.world_trace.rotations, gravity)


def _compute_perfect_mag(plate: PlateTrial, expected_mag: np.ndarray) -> np.ndarray:
    """Oracle mag: the expected (median) global field rotated into the plate's
    ground-truth orientation."""
    return np.einsum('nji,j->ni', plate.world_trace.rotations, expected_mag)


def _compute_perfect_joint_acc(parent_trial: PlateTrial, child_trial: PlateTrial,
                                gravity: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Oracle acc for a joint pair (relative-filter path): the true linear
    acceleration of the shared joint center (averaged from both segments' offset
    estimates) plus gravity, rotated into each segment's own ground-truth
    orientation. Unlike the EKF oracle acc, this includes the linear-acceleration
    term, since the joint center itself translates through space.

    gravity is resolved at call time (see _compute_perfect_segment_acc)."""
    gravity = EXPECTED_GRAVITY if gravity is None else gravity
    parent_offset, child_offset, _ = parent_trial.world_trace.get_joint_center(child_trial.world_trace)

    joint_pos_parent = parent_trial.world_trace.positions + np.einsum(
        'nij,j->ni', parent_trial.world_trace.rotations, parent_offset)
    joint_pos_child = child_trial.world_trace.positions + np.einsum(
        'nij,j->ni', child_trial.world_trace.rotations, child_offset)
    joint_pos = 0.5 * (joint_pos_parent + joint_pos_child)

    # Rotations are unused by finite_difference_world_frame_accelerations; reuse
    # the parent's purely as a placeholder to satisfy the WorldTrace constructor.
    joint_trace = WorldTrace(parent_trial.world_trace.timestamps, joint_pos, parent_trial.world_trace.rotations)
    global_acc = joint_trace.finite_difference_world_frame_accelerations(acc_from_gravity=gravity)

    acc_parent = np.einsum('nji,nj->ni', parent_trial.world_trace.rotations, global_acc)
    acc_child = np.einsum('nji,nj->ni', child_trial.world_trace.rotations, global_acc)
    return acc_parent, acc_child

# ==============================================================================
# Physics: relative filter
# ==============================================================================

def _calculate_observability_metric_(parent_trial: PlateTrial, child_trial: PlateTrial) -> np.ndarray:
    """Computes the observability metric between parent and child IMU traces."""
    da_parent = np.diff(parent_trial.imu_trace.acc, axis=0) + np.cross(parent_trial.imu_trace.gyro[1:], parent_trial.imu_trace.acc[1:])
    da_child = np.diff(child_trial.imu_trace.acc, axis=0) + np.cross(child_trial.imu_trace.gyro[1:], child_trial.imu_trace.acc[1:])
    o_parent = np.cross(parent_trial.imu_trace.acc[1:], da_parent)
    o_child = np.cross(child_trial.imu_trace.acc[1:], da_child)
    observability_metric = np.minimum(np.linalg.norm(o_parent, axis=1),
                                      np.linalg.norm(o_child, axis=1))
    return np.concatenate(([0.0], observability_metric))


def _run_relative_filter(parent_trial: PlateTrial,
                         child_trial: PlateTrial,
                         project: bool,
                         mag_mode: str,
                         acc_override_parent: Optional[np.ndarray] = None,
                         acc_override_child: Optional[np.ndarray] = None,
                         mag_override_parent: Optional[np.ndarray] = None,
                         mag_override_child: Optional[np.ndarray] = None,
                         gyro_std_parent: float = DEFAULT_GYRO_STD,
                         acc_std_parent: float = DEFAULT_ACC_STD,
                         mag_std_parent: float = DEFAULT_MAG_STD,
                         gyro_std_child: float = DEFAULT_GYRO_STD,
                         acc_std_child: float = DEFAULT_ACC_STD,
                         mag_std_child: float = DEFAULT_MAG_STD,
                         mag_adapt_threshold: float = DEFAULT_MAG_ADAPT_THRESHOLD,
                         return_observability: bool = False):
    """Estimates joint orientations between parent and child trials using specified filter configurations.

    acc/mag_override_* replace the trial's real IMU reading before anything else
    runs (used for oracle acc/mag ablations). Providing an acc override implies the
    override is already the correctly-projected acceleration at the point of
    interest, so `project` is forced off in that case — otherwise the rigid-body
    projection would be double-applied.

    If return_observability, also returns the o^J observability metric (see
    _calculate_observability_metric_), computed post-projection regardless of
    mag_mode — used by experiments/drift_observability.py to relate drift to
    observability independent of whether the magnetometer was used."""
    parent_trial = parent_trial.copy()
    child_trial = child_trial.copy()

    if acc_override_parent is not None:
        parent_trial.imu_trace.acc = acc_override_parent
    if acc_override_child is not None:
        child_trial.imu_trace.acc = acc_override_child
    if mag_override_parent is not None:
        parent_trial.imu_trace.mag = mag_override_parent
    if mag_override_child is not None:
        child_trial.imu_trace.mag = mag_override_child

    # Save (possibly overridden) accelerometer readings before any projection is applied
    acc_p_raw = parent_trial.imu_trace.acc.copy()
    acc_c_raw = child_trial.imu_trace.acc.copy()

    do_project = project and acc_override_parent is None and acc_override_child is None

    # 1. IMU projection
    parent_offset, child_offset = None, None
    if do_project:
        parent_offset, child_offset, error = parent_trial.world_trace.get_joint_center(child_trial.world_trace)
        if np.mean(np.linalg.norm(error, axis=1)) > 0.05:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Warning: High joint center error ({np.mean(np.linalg.norm(error, axis=1))} m) "
                      f"between {parent_trial.name} and {child_trial.name}. Check marker placement.")
        parent_trial.imu_trace = parent_trial.project_imu_trace(parent_offset)
        child_trial.imu_trace = child_trial.project_imu_trace(child_offset)

    # 2. Magnetometer modifications
    if mag_mode == 'adapt':
        obs = _calculate_observability_metric_(parent_trial, child_trial)
        high_idx = obs > mag_adapt_threshold
        parent_trial.imu_trace.mag[high_idx] = 0.0
        child_trial.imu_trace.mag[high_idx] = 0.0
    elif mag_mode == 'off':
        parent_trial.imu_trace.mag = np.zeros_like(parent_trial.imu_trace.mag)
        child_trial.imu_trace.mag = np.zeros_like(child_trial.imu_trace.mag)
    elif mag_mode != 'on':
        raise ValueError(f"Unknown mag_mode '{mag_mode}' specified.")

    # 3. Filter execution
    joint_filter = RelativeFilter(
        gyro_std_parent=np.ones(3) * gyro_std_parent,
        gyro_std_child=np.ones(3) * gyro_std_child,
        vector_sensor_stds_parent=[np.ones(3) * acc_std_parent, np.ones(3) * mag_std_parent],
        vector_sensor_stds_child=[np.ones(3) * acc_std_child, np.ones(3) * mag_std_child],
        r_parent=parent_offset,
        r_child=child_offset
    )
    joint_filter.set_qs(Rotation.from_matrix(parent_trial.world_trace.rotations[0]), Rotation.from_matrix(child_trial.world_trace.rotations[0]))
    dt = np.mean(parent_trial.imu_trace.timestamps[1:] - parent_trial.imu_trace.timestamps[:-1])

    # Main update loop
    N = len(parent_trial)
    R_pc = np.empty((N, 3, 3), dtype=np.float64)

    for t in range(N):
        joint_filter.update(
            parent_trial.imu_trace.gyro[t], child_trial.imu_trace.gyro[t],
            [parent_trial.imu_trace.acc[t], parent_trial.imu_trace.mag[t]],
            [child_trial.imu_trace.acc[t], child_trial.imu_trace.mag[t]], dt,
            acc_p_raw=acc_p_raw[t],
            acc_c_raw=acc_c_raw[t]
        )
        R_pc[t] = joint_filter.get_R_pc()

    if return_observability:
        return R_pc, _calculate_observability_metric_(parent_trial, child_trial)
    return R_pc

# ==============================================================================
# Joint angles per method kind
# ==============================================================================

def _joint_angles_from_marker(plates: Dict[str, PlateTrial]) -> pd.DataFrame:
    all_joint_data = []
    any_plate = next(iter(plates.values()))
    timestamps = any_plate.imu_trace.timestamps

    for joint_name, (parent, child) in JOINTS.items():
        if parent not in plates or child not in plates:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Skipping joint {joint_name}: parent '{parent}' or child '{child}' not in loaded plates.")
            continue
        parent_plate = plates[parent]
        child_plate = plates[child]
        R_joint = np.einsum('tji,tjk->tik', parent_plate.world_trace.rotations, child_plate.world_trace.rotations)
        rotvec = Rotation.from_matrix(R_joint).as_rotvec()

        df = pd.DataFrame({
            'timestamp': timestamps,
            'joint_name': joint_name,
            'rx': rotvec[:, 0],
            'ry': rotvec[:, 1],
            'rz': rotvec[:, 2],
        })
        all_joint_data.append(df)

    return pd.concat(all_joint_data, ignore_index=True) if all_joint_data else pd.DataFrame()


def _joint_angles_from_filter(plates: Dict[str, PlateTrial], project: bool, mag_mode: str,
                               acc_source: str = 'real', mag_source: str = 'real',
                               mag_adapt_threshold: float = DEFAULT_MAG_ADAPT_THRESHOLD) -> pd.DataFrame:
    all_joint_data = []
    any_plate = next(iter(plates.values()))
    timestamps = any_plate.imu_trace.timestamps
    expected_mag = _compute_expected_mag_field(list(plates.values())) if mag_source == 'perfect' else None

    for joint_name, (parent, child) in JOINTS.items():
        if parent not in plates or child not in plates:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Skipping joint {joint_name}: parent '{parent}' or child '{child}' not in loaded plates.")
            continue
        parent_plate = plates[parent]
        child_plate = plates[child]

        acc_override_parent = acc_override_child = None
        if acc_source == 'perfect':
            acc_override_parent, acc_override_child = _compute_perfect_joint_acc(parent_plate, child_plate)
        elif acc_source != 'real':
            raise ValueError(f"Unknown acc_source '{acc_source}'")

        mag_override_parent = mag_override_child = None
        if mag_source == 'perfect':
            mag_override_parent = _compute_perfect_mag(parent_plate, expected_mag)
            mag_override_child = _compute_perfect_mag(child_plate, expected_mag)
        elif mag_source != 'real':
            raise ValueError(f"Unknown mag_source '{mag_source}'")

        R_pc = _run_relative_filter(
            parent_plate, child_plate, project=project, mag_mode=mag_mode,
            acc_override_parent=acc_override_parent, acc_override_child=acc_override_child,
            mag_override_parent=mag_override_parent, mag_override_child=mag_override_child,
            mag_adapt_threshold=mag_adapt_threshold
        )
        rotvec = Rotation.from_matrix(R_pc).as_rotvec()

        df = pd.DataFrame({
            'timestamp': timestamps,
            'joint_name': joint_name,
            'rx': rotvec[:, 0],
            'ry': rotvec[:, 1],
            'rz': rotvec[:, 2],
        })
        all_joint_data.append(df)

    return pd.concat(all_joint_data, ignore_index=True) if all_joint_data else pd.DataFrame()


def _joint_angles_from_ekf(plates: Dict[str, PlateTrial], acc_source: str = 'real', mag_source: str = 'real') -> pd.DataFrame:
    plate_trials = list(plates.values())
    ground_plate = _setup_ekf_ground_plate_(plate_trials)
    expected_mag = _compute_expected_mag_field(plate_trials) if mag_source == 'perfect' else None

    segment_orientations = {}
    for plate_name, plate in plates.items():
        acc_override = _compute_perfect_segment_acc(plate) if acc_source == 'perfect' else None
        mag_override = _compute_perfect_mag(plate, expected_mag) if mag_source == 'perfect' else None
        segment_orientations[plate_name] = _run_relative_filter(
            ground_plate, plate, project=False, mag_mode='on',
            acc_override_child=acc_override, mag_override_child=mag_override
        )

    all_joint_data = []
    timestamps = plate_trials[0].imu_trace.timestamps

    for joint_name, (parent, child) in JOINTS.items():
        if parent not in segment_orientations or child not in segment_orientations:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Skipping joint {joint_name}: parent '{parent}' or child '{child}' not in segment orientations.")
            continue
        R_parent = np.array(segment_orientations[parent])
        R_child = np.array(segment_orientations[child])
        R_joint = np.einsum('tji,tjk->tik', R_parent, R_child)
        rotvec = Rotation.from_matrix(R_joint).as_rotvec()

        df = pd.DataFrame({
            'timestamp': timestamps,
            'joint_name': joint_name,
            'rx': rotvec[:, 0],
            'ry': rotvec[:, 1],
            'rz': rotvec[:, 2],
        })
        all_joint_data.append(df)

    return pd.concat(all_joint_data, ignore_index=True) if all_joint_data else pd.DataFrame()


def compute_joint_angles(plates: Dict[str, PlateTrial], method: str) -> pd.DataFrame:
    spec = resolve_method_spec(method)
    if spec['kind'] == 'marker':
        return _joint_angles_from_marker(plates)
    if spec['kind'] == 'ekf':
        return _joint_angles_from_ekf(plates, acc_source=spec['acc_source'], mag_source=spec['mag_source'])
    return _joint_angles_from_filter(
        plates,
        project=spec['project'],
        mag_mode=spec['mag_mode'],
        acc_source=spec['acc_source'],
        mag_source=spec['mag_source'],
        mag_adapt_threshold=spec.get('mag_adapt_threshold', DEFAULT_MAG_ADAPT_THRESHOLD)
    )

# ==============================================================================
# Joint-angle intermediate save/load
# ==============================================================================

joint_angles_path = paths.joint_angles_path


def save_joint_angles(df: pd.DataFrame, subject: str, activity: str, method: str):
    path = ensure_parent(joint_angles_path(subject, activity, method))
    df.to_parquet(path, engine='pyarrow')
    spec = resolve_method_spec(method)
    write_manifest(
        path,
        constants=pipeline_constants(),
        subject=f"Subject{subject}", activity=activity, method=method, method_spec=spec,
        source=str(raw_trial_dir(subject, activity).relative_to(paths.REPO_ROOT)),
        n_rows=len(df),
    )


def load_joint_angles(subject: str, activity: str, method: str) -> Optional[pd.DataFrame]:
    path = joint_angles_path(subject, activity, method)
    return pd.read_parquet(path, engine='pyarrow') if path.exists() else None


def load_all_joint_angles(subjects: List[str], activities: List[str], methods: List[str]) -> pd.DataFrame:
    frames = []
    for subject in subjects:
        for activity in activities:
            for method in methods:
                df = load_joint_angles(subject, activity, method)
                if df is None:
                    if os.environ.get("DISABLE_TQDM") != "True":
                        print(f"Warning: missing joint angles for Subject{subject}/{activity}/{method} — skipping")
                    continue
                frames.append(df.assign(subject=f"Subject{subject}", trial_type=activity, method=method))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ==============================================================================
# Statistics
# ==============================================================================

def compute_error_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates summary statistics using fast wide-format C-aggregations."""
    if df.empty:
        return pd.DataFrame()

    marker_df = df[df['method'] == 'marker']
    imu_df = df[df['method'] != 'marker']

    if marker_df.empty or imu_df.empty:
        return pd.DataFrame()

    join_cols = ['subject', 'trial_type', 'joint_name', 'timestamp']
    merged_df = pd.merge(imu_df, marker_df, on=join_cols, suffixes=('_imu', '_marker'))

    if merged_df.empty:
        return pd.DataFrame()

    rotvec_imu = merged_df[['rx_imu', 'ry_imu', 'rz_imu']].to_numpy()
    rotvec_marker = merged_df[['rx_marker', 'ry_marker', 'rz_marker']].to_numpy()

    r_imu = Rotation.from_rotvec(rotvec_imu)
    r_marker = Rotation.from_rotvec(rotvec_marker)
    r_error = r_imu * r_marker.inv()
    rotvec_error = r_error.as_rotvec()

    merged_df['X'] = rotvec_error[:, 0]
    merged_df['Y'] = rotvec_error[:, 1]
    merged_df['Z'] = rotvec_error[:, 2]
    merged_df['MAG'] = np.linalg.norm(rotvec_error, axis=1)

    merged_df = merged_df.rename(columns={'method_imu': 'method'})
    group_cols = ['trial_type', 'method', 'joint_name', 'subject']
    target_cols = ['MAG', 'X', 'Y', 'Z']

    # Pre-calculate squared & absolute columns for vectorized MAE/RMSE
    for col in target_cols:
        merged_df[f'{col}_sq'] = merged_df[col] ** 2
        merged_df[f'{col}_abs'] = merged_df[col].abs()

    # Fast built-in aggregations across wide format
    grouped = merged_df.groupby(group_cols)

    means = grouped[target_cols].mean()
    stds = grouped[target_cols].std()
    mins = grouped[target_cols].min()
    maxs = grouped[target_cols].max()
    medians = grouped[target_cols].median()
    q25s = grouped[target_cols].quantile(0.25)
    q75s = grouped[target_cols].quantile(0.75)
    maes = grouped[[f'{c}_abs' for c in target_cols]].mean().rename(columns=lambda c: c.replace('_abs', ''))
    rmses = np.sqrt(grouped[[f'{c}_sq' for c in target_cols]].mean()).rename(columns=lambda c: c.replace('_sq', ''))

    # MAD: median absolute deviation from median
    mads = grouped[target_cols].apply(lambda g: (g - g.median()).abs().median())

    # Build multi-index summary and melt at the very end
    summary_list = []
    metric_map = {
        'mean_rad': means, 'std_rad': stds, 'rmse_rad': rmses,
        'mae_rad': maes, 'mad_rad': mads, 'min_rad': mins,
        'q25_rad': q25s, 'median_rad': medians, 'q75_rad': q75s, 'max_rad': maxs
    }

    for metric_name, metric_df in metric_map.items():
        melted = metric_df.reset_index().melt(
            id_vars=group_cols, value_vars=target_cols, var_name='axis', value_name=metric_name
        )
        summary_list.append(melted.set_index(group_cols + ['axis']))

    summary_df = pd.concat(summary_list, axis=1).reset_index()
    return summary_df


def save_statistics(df: pd.DataFrame, name: str, **manifest_extra: Any) -> Path:
    """Saves a summary-statistics DataFrame to results/statistics/<name>_statistics.parquet."""
    path = ensure_parent(paths.statistics_path(name))
    df.to_parquet(path, engine='pyarrow')
    write_manifest(
        path, constants=pipeline_constants(), experiment=name,
        methods=sorted(df['method'].unique().tolist()) if 'method' in df.columns else None,
        subjects=sorted(df['subject'].unique().tolist()) if 'subject' in df.columns else None,
        n_rows=len(df), **manifest_extra,
    )
    return path


def load_statistics(name: str) -> Optional[pd.DataFrame]:
    path = paths.statistics_path(name)
    return pd.read_parquet(path, engine='pyarrow') if path.exists() else None

# ==============================================================================
# Parallel runner with live status table
# ==============================================================================
# Every experiment shares this shape: a set of "rows" (e.g. subject/activity pairs,
# or joints) each processed in its own worker process, with a fixed set of "stage"
# columns (e.g. methods, noise combos, thresholds) whose status is tracked live.
# worker_fn runs ALL stages for one row in a single process (so it can load raw data
# once and reuse it across stages) and returns whatever payload the caller wants
# collected (typically a DataFrame, or None).

_STATUS_COLORS = {"Pending": "white", "Running": "yellow", "Success": "green", "Skipped": "yellow"}


def _render_grid_table(title: str, row_keys: List[Any], row_labels: List[str],
                        stage_labels: List[str], shared_state: Dict) -> Table:
    table = Table(title=f"[bold magenta]{title}[/bold magenta]", show_header=True,
                  header_style="bold cyan", border_style="bold blue")
    for label in row_labels:
        table.add_column(label, style="bold white", justify="center")
    for stage in stage_labels:
        table.add_column(stage, justify="center")

    for row_key in row_keys:
        row = list(row_key) if isinstance(row_key, tuple) else [row_key]
        for stage in stage_labels:
            status = shared_state.get((row_key, stage), "Pending")
            t = shared_state.get((row_key, f"{stage}_time"), None)
            suffix = f" [dim]({t:.1f}s)[/dim]" if t is not None else ""
            color = _STATUS_COLORS.get(status, "red")
            row.append(f"[bold {color}]■[/bold {color}]{suffix}")
        table.add_row(*row)
    return table


def run_tracked_grid(row_keys: List[Any], row_labels: List[str], stage_labels: List[str],
                      worker_fn: Callable[[Any, List[str], Dict], Any], workers: int,
                      title: str = "MAJIC MOCAP PIPELINE STATUS",
                      per_cell: bool = False) -> Tuple[Dict, Dict[Any, Any]]:
    """Runs `worker_fn(row_key, stage_labels, shared_state)` across a
    ProcessPoolExecutor pool, rendering a live status table with one row per
    row_key and one status column per stage. `worker_fn` should write
    shared_state[(row_key, stage)] = "Running"/"Success"/"Skipped"/"Failed (...)"
    (and optionally shared_state[(row_key, f'{stage}_time')]) for each stage it's
    given, and return whatever payload should be collected.

    By default (per_cell=False), one process handles ALL stages for a given
    row_key sequentially — worker_fn receives the full stage_labels list. This is
    the right choice when stages share expensive setup (e.g. raw data loaded once
    and reused across every method for that subject/activity). Returns
    {row_key: worker_fn's return value}.

    per_cell=True instead submits one process PER (row_key, stage) pair —
    worker_fn receives a single-element stage list each call. Use this when
    stages are independently expensive and don't share setup worth amortizing
    (e.g. noise-sensitivity combos, where reloading raw data per combo is
    negligible next to the combo's own filter pass): it exposes every cell to the
    worker pool individually instead of serializing a row's stages behind one
    process. Returns {(row_key, stage): worker_fn's return value}.
    """
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    for row_key in row_keys:
        for stage in stage_labels:
            shared_state[(row_key, stage)] = "Pending"

    with Live(_render_grid_table(title, row_keys, row_labels, stage_labels, shared_state),
              refresh_per_second=4) as live:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            if per_cell:
                futures = {executor.submit(worker_fn, row_key, [stage], shared_state): (row_key, stage)
                           for row_key in row_keys for stage in stage_labels}
            else:
                futures = {executor.submit(worker_fn, row_key, stage_labels, shared_state): row_key
                           for row_key in row_keys}

            while any(not f.done() for f in futures):
                time.sleep(0.25)
                live.update(_render_grid_table(title, row_keys, row_labels, stage_labels, shared_state))

            results = {key: future.result() for future, key in futures.items()}

            live.update(_render_grid_table(title, row_keys, row_labels, stage_labels, shared_state))

    return dict(shared_state), results

# ==============================================================================
# Method-name-driven grid workers
# ==============================================================================
# Shared by experiments/benchmark_experiment.py (the vanilla method list) and any
# experiment that sweeps method *names* rather than filter-tuning parameters
# (e.g. experiments/threshold_sensitivity.py's mag_adapt_th<value> sweep,
# experiments/oracle_ablation.py's acc/mag oracle combos). Each is meant to be
# passed straight to run_tracked_grid as the worker_fn, with stage_labels = the
# method names to run for that call.

def generate_joint_angles_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict) -> None:
    """stage_labels = ['load'] + method names. Loads raw data once, then computes
    and saves joint angles for every method, reusing the loaded data across methods."""
    subject, activity = row_key
    methods = stage_labels[1:]

    t_start = time.time()
    shared_state[(row_key, 'load')] = "Running"
    try:
        plates = load_raw_data(subject, activity)
        shared_state[(row_key, 'load_time')] = time.time() - t_start
        shared_state[(row_key, 'load')] = "Success"
    except Exception as e:
        shared_state[(row_key, 'load')] = f"Failed ({e})"
        for method in methods:
            shared_state[(row_key, method)] = "Failed"
        return None

    for method in methods:
        t_method = time.time()
        shared_state[(row_key, method)] = "Running"
        try:
            df = compute_joint_angles(plates, method)
            if df is not None and not df.empty:
                save_joint_angles(df, subject, activity, method)
                shared_state[(row_key, f"{method}_time")] = time.time() - t_method
                shared_state[(row_key, method)] = "Success"
            else:
                shared_state[(row_key, method)] = "Skipped"
        except Exception as e:
            shared_state[(row_key, method)] = f"Failed ({e})"
    return None


def compute_stats_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                          methods: List[str], stats_name: str) -> None:
    """Single-stage worker (stage_labels should be a single name, e.g. ['stats']).
    Reads back whatever per-method parquets generate_joint_angles_worker managed to
    save for this row and computes error stats against 'marker'. Naturally
    fails/no-ops if the load or every method failed there, since no parquet files
    exist to read.

    `stats_name` namespaces the output under results/statistics/per_subject/<stats_name>/.
    It is required rather than defaulted: benchmark, oracle-ablation and
    threshold-sweep runs each produce per-subject stats over a different method
    set, and a shared default filename meant whichever ran last silently won."""
    subject, activity = row_key
    stage = stage_labels[0]

    t_start = time.time()
    shared_state[(row_key, stage)] = "Running"
    try:
        frames = []
        for method in methods:
            df = load_joint_angles(subject, activity, method)
            if df is not None:
                frames.append(df.assign(subject=f"Subject{subject}", trial_type=activity, method=method))

        if not frames:
            shared_state[(row_key, stage)] = "Failed"
            return None

        all_df = pd.concat(frames, ignore_index=True)
        stats_df = compute_error_stats(all_df)
        if not stats_df.empty:
            stats_path = ensure_parent(paths.per_subject_statistics_path(stats_name, subject, activity))
            stats_df.to_parquet(stats_path, engine='pyarrow')
            write_manifest(stats_path, constants=pipeline_constants(), experiment=stats_name,
                           subject=f"Subject{subject}", activity=activity, methods=methods)

        shared_state[(row_key, f"{stage}_time")] = time.time() - t_start
        shared_state[(row_key, stage)] = "Success"
        return None
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
        return None
