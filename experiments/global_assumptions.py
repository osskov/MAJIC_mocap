"""
How well the two GLOBAL SENSOR ASSUMPTIONS hold in this data, split by whether the sensor
was moving or standing still.

Every orientation filter in this repo leans on two claims about what a body-worn IMU reads:

    ACCELEROMETER   a = g                 the accelerometer measures gravity and nothing else
    MAGNETOMETER    m = m_earth           the magnetometer measures one constant field,
                                          the same vector everywhere on the body

Both are false in motion and only approximately true at rest. This experiment measures HOW
false, per sensor, per subject, per trial, and — the part that is new here — separately over
the STATIC and NON-STATIC stretches of every recording, because that split is what decides
whether a filter should gate its measurement updates in time rather than inflate a constant
noise term. It also measures the intrinsic noise floor each sensor sits at while genuinely
still, which is the number the departures above have to be compared against to mean anything.

Supersedes `experiments/sensor_distributions.py`, whose metrics, field-reference reasoning and
console report this inherits wholesale. What changed in the move, all of which shifts printed
numbers relative to that script:

  * STATIC SEGMENTATION, and every distribution reported over it. See `static_mask`.
  * The intrinsic noise floor is measured for EVERY sensor over its own static stretches, not
    only for the two feet during both-feet pauses. The old restriction existed because the only
    detector available was a whole-body one; a per-sensor one makes a stance-phase foot, a
    hanging arm and a seated thigh all usable.
  * Mocap-dependent quantities (linacc, magdev, the world-frame field) are NaN outside `valid`
    frames instead of being computed against a padded constant pose. The old code rotated real
    readings by a pose the subject was never in and folded the result into the distributions.
  * A subject's global field is the median over per-(trial, sensor) medians rather than over
    pooled samples, so a session's longest recording cannot outvote its others. This matters
    for IMoVE, where one session holds a 30 s static pose beside a 2800 s walk.
  * TWO DATASETS. Every segment map, joint table and sensor list is a `DatasetSpec` rather
    than a module constant, so Al Borno's 8 sensors and IMoVE's 15 (three per thigh and shank)
    run through the same code.

Metrics
-------
    linacc        |a_world - g|, the non-gravity acceleration          m/s^2
    acc_norm_dev  ||a| - |g||, the same departure without mocap        m/s^2
    magdev        |m_world - m_global|, deviation from the subject's field   a.u. (see MAG_UNIT)
    magdev_angle  the same, as a DIRECTION error                       deg
    obs_min       o^J at the joint center, what mag_adapt gates on     (m/s^2)(m/s^3)

`linacc` and `magdev` need mocap rotations and so exist only on valid frames; `acc_norm_dev`
and `mag_norm` are reference-free and exist everywhere. Both kinds are reported because the
reference-free pair is what a filter could measure for itself at runtime while the mocap pair
is the ground truth it would be trying to infer — and because the reference-free version
systematically UNDERSTATES the departure (acceleration perpendicular to gravity adds in
quadrature, so a segment swinging horizontally reads |a| ~ |g| throughout).

Regimes
-------
Every distribution and scalar is reported four times, in a `regime` column:

    all           every sample
    static        the sensor's own still stretches (per-sensor, gyro-derived)
    nonstatic     the complement of `static`
    body_static   stretches where EVERY sensor on the body is simultaneously still

`static` and `body_static` are different measurements, not two strengths of one. A foot in
stance is `static` while the subject walks; only a standing or seated pause is `body_static`.
The noise floor is reported over both, and the gap between them is the vibration and impact
energy a stance-phase foot still carries.

Outputs, all under results/experiments/global_assumptions/<dataset>/:

    <subject>/subject_field.parquet          the subject/session global field (one row)
    <subject>/<trial>/trial_field.parquet    per-(trial, sensor) median world-frame field
    <subject>/<trial>/segment_samples.parquet  per-sample, per-sensor metrics + regime masks
    <subject>/<trial>/joint_samples.parquet    per-sample, per-joint o^J + regime masks
    <subject>/<trial>/intervals.parquet        labeled static / posture intervals
    <subject>/<trial>/sensor_stats.parquet     per-(sensor, regime) scalars incl. noise floor
    <subject>/<trial>/joint_stats.parquet      per-(joint, regime) observability + field consistency

plus the pooled quantile summary at
results/statistics/global_assumptions_<dataset>_statistics.parquet, which is what the console
report and the paper text quote.

The per-sample tables are what make the figures cheap to re-tune: plotting reads these back
instead of reloading and re-projecting every trial.

    python -m experiments.global_assumptions --dataset alborno
    python -m experiments.global_assumptions --dataset imove --subjects s13 s14
    python -m experiments.global_assumptions --dataset alborno --report-only
    python -m experiments.global_assumptions --dataset imove --only-tables sensor_stats

THE PARQUET IS THE INTERFACE. Trials are enumerated from results/trials/<dataset>/ — the
build tree — and read through `experiment_utils.load_trial`, which refuses a stale artifact
rather than parsing from source. Build first:

    python -m experiments.build_trials --dataset <name>
"""
import argparse
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import paths
from experiments.experiment_utils import (DEFAULT_ACC_STD, DEFAULT_MAG_ADAPT_THRESHOLD,
                                          EXPECTED_GRAVITY, JOINTS, load_trial,
                                          pipeline_constants, project_pair_to_joint_center,
                                          run_tracked_grid, segment_observability)
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial

EXPERIMENT_NAME = "global_assumptions"
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

# ==============================================================================
# DATASETS
# ==============================================================================
# Everything that differs between the two source datasets lives in a DatasetSpec, so the
# physics below never asks which one it is looking at. Adding a third dataset means adding a
# spec, a reader in src/toolchest/building/, and nothing else.


@dataclass(frozen=True)
class DatasetSpec:
    """Which sensors a dataset has, what to call them, and which pairs form a joint.

    `segment_sensor` maps a DISPLAY name to a plate name and is ordered proximal to distal.
    The order is not cosmetic: the central finding of this experiment is that both assumptions
    degrade monotonically down each limb, so every figure sorts by it and it is declared once
    here rather than per figure.

    `joints` is the pair table o^J and the field-consistency measures run over. It is a
    property of the dataset because IMoVE has no torso sensor, hence no lumbar joint, and
    three sensors per thigh and shank of which only the Mid one is bolted to the marker
    cluster (see building/imove_mocap.RIGID_SENSOR_OFFSET_MM) and therefore the only one whose
    joint-center projection rests on a hardware-fixed offset rather than a per-trial fit.
    """
    name: str
    segment_sensor: Dict[str, str]
    joints: Dict[str, Tuple[str, str]]
    pelvis_sensor: str
    foot_sensors: Tuple[str, ...]
    subject_label: str          # how a subject id is printed; '{}' leaves it verbatim
    field_reference: Dict[str, str]   # joint -> 'parent' | 'child'; see FIELD_REFERENCE below


# Which sensor of each pair supplies the LOCAL FIELD REFERENCE in `joint_field_consistency`,
# i.e. which one plays the role the global field is being compared against. Default is the
# kinematic parent, which for every limb joint is also the more proximal and magnetically
# cleaner sensor.
#
# The lumbar is the one pair where the kinematic chain and the field-quality ordering disagree.
# The chain is rooted at the pelvis, so the pelvis is the lumbar's parent — but the TORSO is the
# cleanest magnetometer on the body (trial-mean mag_norm_std 0.062 vs the pelvis's 0.094, and
# cos-sim to the global field 0.997+), while the pelvis sits lower, nearer whatever ferrous
# structure is in the floor. Rooting the chain at the pelvis is a kinematic convention with no
# magnetic meaning, so the lumbar reference is flipped to the torso. That keeps every row of the
# table asking one question: does the CLEANER neighbour beat the global field as a reference for
# the more distorted one?
#
# This is a fixed, anatomically motivated direction declared up front — NOT chosen per trial from
# whichever direction came out positive, which would manufacture the conclusion. Both directions
# are computed and stored for every joint regardless (var_reduction_parent_ref /
# var_reduction_child_ref), so the choice is auditable and reversible from the saved table.
#
# It is a POPULATION-level claim, and one trial in this dataset violates it: in Subject10/walking
# the torso is the noisier of the pair, and that is the one trial where the flipped lumbar comes
# out negative. The rule is not adjusted for it — a per-trial direction would be the cherry-pick
# this constant exists to avoid — but a lumbar number quoted for that trial should carry the
# caveat. IMoVE has no lumbar, so its map is empty and every joint uses the parent.
ALBORNO_FIELD_REFERENCE = {'Lumbar': 'child'}

ALBORNO = DatasetSpec(
    name='alborno',
    segment_sensor={
        'Torso': 'torso_imu',
        'Pelvis': 'pelvis_imu',
        'Femur R': 'femur_r_imu',
        'Tibia R': 'tibia_r_imu',
        'Calcn R': 'calcn_r_imu',
        'Femur L': 'femur_l_imu',
        'Tibia L': 'tibia_l_imu',
        'Calcn L': 'calcn_l_imu',
    },
    joints=dict(JOINTS),
    pelvis_sensor='pelvis_imu',
    foot_sensors=('calcn_l_imu', 'calcn_r_imu'),
    subject_label='Subject{}',
    field_reference=ALBORNO_FIELD_REFERENCE,
)

# IMoVE carries three sensors on each thigh and shank — High, Mid and Low — against a single
# 4-marker cluster. They are separate rows here rather than being averaged into one "thigh",
# because the whole question this experiment asks is how the two assumptions vary with WHERE on
# the body a sensor sits, and a 15 cm slide down the same segment is exactly that variation
# measured at a finer grain than Al Borno can offer.
IMOVE = DatasetSpec(
    name='imove',
    segment_sensor={
        'Pelvis': 'PELVIS_M',
        'Thigh R High': 'THIGH_R_H',
        'Thigh R Mid': 'THIGH_R_M',
        'Thigh R Low': 'THIGH_R_L',
        'Shank R High': 'SHANK_R_H',
        'Shank R Mid': 'SHANK_R_M',
        'Shank R Low': 'SHANK_R_L',
        'Foot R': 'FOOT_R_M',
        'Thigh L High': 'THIGH_L_H',
        'Thigh L Mid': 'THIGH_L_M',
        'Thigh L Low': 'THIGH_L_L',
        'Shank L High': 'SHANK_L_H',
        'Shank L Mid': 'SHANK_L_M',
        'Shank L Low': 'SHANK_L_L',
        'Foot L': 'FOOT_L_M',
    },
    # Mid sensors only. The High and Low placements are taped on and their cluster offsets are
    # fitted per trial (building/imove_mocap.NOMINAL_SENSOR_OFFSET_MM), so a joint-center
    # projection built on them carries that fit's error into o^J; the Mid sensors share one
    # bolted bracket per segment and their offset is hardware. No lumbar: there is no torso
    # sensor in this dataset.
    joints={
        'R_Hip': ('PELVIS_M', 'THIGH_R_M'),
        'R_Knee': ('THIGH_R_M', 'SHANK_R_M'),
        'R_Ankle': ('SHANK_R_M', 'FOOT_R_M'),
        'L_Hip': ('PELVIS_M', 'THIGH_L_M'),
        'L_Knee': ('THIGH_L_M', 'SHANK_L_M'),
        'L_Ankle': ('SHANK_L_M', 'FOOT_L_M'),
    },
    pelvis_sensor='PELVIS_M',
    foot_sensors=('FOOT_L_M', 'FOOT_R_M'),
    subject_label='{}',
    field_reference={},
)

DATASETS: Dict[str, DatasetSpec] = {ALBORNO.name: ALBORNO, IMOVE.name: IMOVE}


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}. Registered: {sorted(DATASETS)}.") from None


def enumerate_trials(dataset: str) -> List[Tuple[str, str]]:
    """Every (subject, trial) with a BUILT parquet under results/trials/<dataset>/.

    Enumerated from the build tree rather than from the source data or from a constant, for
    the same reason `load_trial` refuses to parse from source: the parquet is the interface,
    so what this experiment can analyse is exactly what has been built. A subject missing an
    activity, or a dataset half-built, shows up here as a shorter list instead of as a run of
    failures — and `--check` can then say what is outstanding without loading anything.
    """
    root = paths.TRIALS_DIR / dataset
    if not root.is_dir():
        return []
    return sorted((subject_dir.name, artifact.stem)
                  for subject_dir in root.iterdir() if subject_dir.is_dir()
                  for artifact in subject_dir.glob('*.parquet'))


def subjects_of(row_keys: Sequence[Tuple[str, str]]) -> List[str]:
    """Distinct subjects in a trial list, in first-seen order."""
    seen: Dict[str, None] = {}
    for subject, _ in row_keys:
        seen.setdefault(subject, None)
    return list(seen)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# The up axis of the mocap world frame, taken from the gravity vector rather than written down
# again: pelvis height and the world-frame gravity convention have to agree, and hard-coding 1
# here would let them drift apart silently if EXPECTED_GRAVITY ever changed. Both datasets are
# Y-up with the positive specific-force convention — measured, not assumed: IMoVE's mean
# world-frame accelerometer vector is [0.072, 9.810, -0.006], Al Borno's [0.016, 9.812, -0.003].
HEIGHT_AXIS = int(np.argmax(np.abs(EXPECTED_GRAVITY)))
GRAVITY_MAGNITUDE = float(np.linalg.norm(EXPECTED_GRAVITY))

# --- Static detection ------------------------------------------------------------------
# A sensor is STATIC over any stretch where its gyroscope stays below STATIC_GYRO_MAX for the
# whole of a STATIC_WINDOW_S window, for at least MIN_STATIC_S.
#
# GYRO ONLY, and deliberately so. This experiment measures how far the accelerometer departs
# from gravity and how far the magnetometer departs from a constant, so a static detector that
# consulted either one would define its answer into existence — "the accelerometer reads
# gravity during the stretches we selected for reading gravity". The gyroscope is the one
# channel not under test, and rotation is what actually breaks both assumptions: a rotating
# sensor sweeps its acc and mag vectors through the body frame whether or not it translates.
#
# The blind spot is pure translation, which no gyro can see. That is measured rather than
# waved away: `sensor_stats` carries the mocap linear and angular speed observed during each
# sensor's static stretches (mocap_speed_*), and across both datasets those come out at 0.5-5
# mm/s — at or below marker reconstruction noise. The one exception is the Al Borno feet during
# walking stance at ~23 mm/s, which is 0.23 mm of marker travel per frame and reads as
# reconstruction noise rather than motion.
#
# Thresholds: 0.05 rad/s is 2.9 deg/s, about 10x the measured gyroscope noise floor, and a
# rolling MAXIMUM rather than a rolling standard deviation, so a stretch turning steadily and
# slowly is correctly called moving instead of "consistently the same speed". The 0.25 s window
# is short enough to resolve a walking stance phase (~0.4 s at self-selected speed); a 1 s
# window finds none of them.
STATIC_WINDOW_S = 0.25
STATIC_GYRO_MAX = 0.05      # rad/s
MIN_STATIC_S = 0.10

# --- Whole-body static -------------------------------------------------------------------
# Every sensor static at once. Held to a longer minimum and trimmed at both ends, because this
# is what the cleanest noise-floor estimate is measured over and the rolling window's own edges
# (where it straddles the transition into motion) should not be counted as noise.
BODY_STATIC_MIN_S = 1.0
BODY_STATIC_MARGIN_S = 0.25

# --- Observability smoothing (distribution summary and figures only) ---------------------
# o^J is a norm of a cross product of a finite difference: non-negative, and full of sharp
# impulsive spikes at every footfall. The spikes are real, but they dominate a KDE and a box
# plot to the point where the body of the distribution is invisible, so the per-sample series
# stored here is low-pass filtered. The filter is applied HERE rather than in the plotting
# layer so that the figures and the quoted quantiles describe the same numbers, and so the
# cutoff lands in the provenance manifest. The RAW series is kept alongside it (obs_min_raw),
# because the gating duty cycle in section 6 has to be computed on what the filter would
# actually see.
OBS_FILTER_CUTOFF_HZ = 25.0
OBS_FILTER_ORDER = 4
OBS_WINSORIZE_PCT = 99.5

# --- Posture labeling (Al Borno complexTasks; degrades gracefully elsewhere) --------------
# Coarser than the static detector and answering a different question: not "was this sensor
# moving" but "what was the subject doing". Kept because the sitting bout is the case that
# motivates gating on observability — it collapses o^J while leaving magnetic distortion
# untouched — and the time-series figures shade against it.
QUIET_LINACC_THRESHOLD = 0.5  # m/s^2, rolling std of pelvis linear acceleration
QUIET_WINDOW_S = 1.0
SIT_HEIGHT_DROP_M = 0.25  # pelvis drop below its trial-typical upright height that counts as sitting
MIN_SITTING_S = 10.0      # sitting bouts run long relative to the rest of the task cycle
MIN_STANDING_S = 2.0      # standing bouts are brief in this protocol

# Sample stride for the PER-SAMPLE tables only. Every scalar in sensor_stats and joint_stats is
# computed on the full unstrided record; this thins only what is written out for the pooled
# quantiles and the figures. At 1 the two datasets together write ~36 M sample-rows.
SAMPLE_STRIDE = 1

REGIMES = ('all', 'static', 'nonstatic', 'body_static')
# The regimes a noise floor is meaningful over. Measuring the "noise" of a moving sensor would
# report its motion.
NOISE_REGIMES = ('static', 'body_static')

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# These files are Xsens exports, whose magnetometer channels are NOT in microtesla: they are
# normalized at calibration so that a nominal local Earth field reads 1.0 (measured |mag|
# medians across both datasets' sensors run 0.4-1.1, and the filter's own mag_std default is on
# that same scale). Every magnetic quantity here is therefore in those arbitrary units and the
# labels say so. Multiply by ~50 uT for a physical scale.
MAG_UNIT = 'a.u.'
METRIC_UNITS = {
    'linacc': 'm/s^2',
    'acc_norm_dev': 'm/s^2',
    'magdev': MAG_UNIT,
    'magdev_angle': 'deg',
    'mag_norm_dev': MAG_UNIT,
    'obs_min': '(m/s^2)(m/s^3)',
}
# Which per-sample column each summarized metric reads, and which table it lives in.
#
# Two of the four segment metrics need a mocap ROTATION (linacc, magdev, magdev_angle) and two
# do not (acc_norm_dev, mag_norm_dev). That split matters more than it looks: the static
# stretches of the Al Borno WALKING trials sit almost entirely outside the mocap window — the
# subject stood for ~430 s before the cameras rolled — so on those trials the mocap-referenced
# metrics have no static samples to report and the reference-free pair is the whole static/moving
# comparison there. The n_valid columns in sensor_stats say which trials that applies to.
SEGMENT_METRICS = ('linacc', 'acc_norm_dev', 'magdev', 'magdev_angle', 'mag_norm_dev')
MOCAP_METRICS = ('linacc', 'magdev', 'magdev_angle')
JOINT_METRICS = ('obs_min',)

TRIAL_TABLES = ('trial_field', 'segment_samples', 'joint_samples', 'intervals',
                'sensor_stats', 'joint_stats')
# trial_field is stage one and every other table depends on the subject field it feeds, so it
# is not offered to --only-tables as something to skip independently.
RECOMPUTABLE_TABLES = tuple(t for t in TRIAL_TABLES if t != 'trial_field')


def analysis_constants(dataset: str) -> Dict[str, object]:
    """Pipeline constants plus this analysis's own thresholds, for provenance manifests."""
    return {
        **pipeline_constants(),
        'dataset': dataset,
        'height_axis': HEIGHT_AXIS,
        'static_window_s': STATIC_WINDOW_S,
        'static_gyro_max_rad_s': STATIC_GYRO_MAX,
        'min_static_s': MIN_STATIC_S,
        'body_static_min_s': BODY_STATIC_MIN_S,
        'body_static_margin_s': BODY_STATIC_MARGIN_S,
        'obs_filter_cutoff_hz': OBS_FILTER_CUTOFF_HZ,
        'obs_filter_order': OBS_FILTER_ORDER,
        'obs_winsorize_pct': OBS_WINSORIZE_PCT,
        'quiet_linacc_threshold': QUIET_LINACC_THRESHOLD,
        'sit_height_drop_m': SIT_HEIGHT_DROP_M,
        'min_sitting_s': MIN_SITTING_S,
        'min_standing_s': MIN_STANDING_S,
        'sample_stride': SAMPLE_STRIDE,
    }


def segment_joint_roles(spec: DatasetSpec) -> Dict[str, List[Tuple[str, str]]]:
    """{segment: [(joint, 'parent'|'child'), ...]} — which joints border each segment, and on
    which side of each.

    Derived from the spec's joint table rather than written out, because a hand-maintained copy
    is a table that can silently disagree with the joint definitions the rest of the pipeline
    uses. Segments carrying no joint (IMoVE's High and Low placements) come back with an empty
    list, which is the correct answer for them.
    """
    sensor_segment = {sensor: segment for segment, sensor in spec.segment_sensor.items()}
    roles: Dict[str, List[Tuple[str, str]]] = {segment: [] for segment in spec.segment_sensor}
    for joint, (parent_sensor, child_sensor) in spec.joints.items():
        for sensor, role in ((parent_sensor, 'parent'), (child_sensor, 'child')):
            if sensor in sensor_segment:
                roles[sensor_segment[sensor]].append((joint, role))
    return roles

# ==============================================================================
# Paths / IO
# ==============================================================================

def dataset_dir(dataset: str) -> Path:
    return EXPERIMENT_DIR / dataset


def trial_table_path(dataset: str, subject: str, trial: str, table: str) -> Path:
    return dataset_dir(dataset) / subject / trial / f"{table}.parquet"


def subject_field_path(dataset: str, subject: str) -> Path:
    return dataset_dir(dataset) / subject / "subject_field.parquet"


def statistics_path(dataset: str) -> Path:
    return paths.statistics_path(f"{EXPERIMENT_NAME}_{dataset}")


def _save(df: pd.DataFrame, path: Path, dataset: str, **manifest_extra) -> None:
    df.to_parquet(paths.ensure_parent(path), engine='pyarrow', index=False)
    paths.write_manifest(path, constants=analysis_constants(dataset), experiment=EXPERIMENT_NAME,
                         n_rows=len(df), **manifest_extra)


def load_trial_table(dataset: str, table: str,
                     row_keys: Optional[Sequence[Tuple[str, str]]] = None,
                     columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Concatenates one per-trial table across trials, adding `subject` and `trial` columns.

    Missing trials are skipped silently — a partial run (`--subjects s13`) is a legitimate
    state, and the caller reports what it found. An unknown table name raises instead of
    returning an empty frame, which would be indistinguishable from "the experiment has not
    been run yet".

    `columns` pushes a projection down into the parquet reader, which is not an optimisation
    to skip on the sample tables: IMoVE's segment_samples is ~24 M rows, and reading all
    fourteen columns to summarize five of them costs several hundred megabytes of resident
    memory for nothing. `subject` and `trial` are always added and need not be listed.
    """
    if table not in TRIAL_TABLES:
        raise ValueError(f"Unknown table '{table}'; expected one of {TRIAL_TABLES}")
    row_keys = enumerate_trials(dataset) if row_keys is None else row_keys
    columns = list(columns) if columns is not None else None
    frames = []
    for subject, trial in row_keys:
        path = trial_table_path(dataset, subject, trial, table)
        if not path.exists():
            continue
        frame = pd.read_parquet(path, engine='pyarrow', columns=columns)
        frames.append(frame.assign(subject=subject, trial=trial))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Categorical after the concat, not before: these repeat across millions of rows and a
    # per-frame category dtype would have to be unified anyway.
    for column in ('subject', 'trial'):
        out[column] = out[column].astype('category')
    out['dataset'] = dataset
    return out


def load_subject_fields(dataset: str, subjects: Optional[Sequence[str]] = None) -> pd.DataFrame:
    subjects = subjects_of(enumerate_trials(dataset)) if subjects is None else subjects
    frames = [pd.read_parquet(subject_field_path(dataset, s), engine='pyarrow')
              for s in subjects if subject_field_path(dataset, s).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_subject_field(dataset: str, subject: str) -> Optional[np.ndarray]:
    """One subject's global field vector, or None if stage one has not run for them."""
    path = subject_field_path(dataset, subject)
    if not path.exists():
        return None
    row = pd.read_parquet(path, engine='pyarrow').iloc[0]
    return np.array([row['world_mag_x'], row['world_mag_y'], row['world_mag_z']], dtype=float)

# ==============================================================================
# Stage one: the subject's global magnetic field
# ==============================================================================

def trial_field_table(plates: Dict[str, PlateTrial], spec: DatasetSpec) -> pd.DataFrame:
    """One row per sensor: the median world-frame magnetic field it read in this trial.

    Valid frames only. It is the mocap ROTATION that puts a body-frame reading into the world
    frame, so outside `valid` the world trace holds a constant padded pose and rotating a real
    measurement by it puts the vector somewhere it never was. Since alignment stopped trimming,
    those padded stretches can outnumber the measured ones.

    Median rather than mean because local ferrous distortion is one-sided and heavy-tailed: a
    mean would be pulled toward whichever part of the room the subject spent longest near.
    """
    sensor_segment = {sensor: segment for segment, sensor in spec.segment_sensor.items()}
    rows = []
    for sensor, plate in sorted(plates.items()):
        valid = np.asarray(plate.valid)
        if not valid.any():
            continue
        world_mag = plate.get_imu_trace_in_global_frame().mag[valid]
        median = np.median(world_mag, axis=0)
        rows.append({
            'sensor': sensor,
            'segment': sensor_segment.get(sensor, sensor),
            'world_mag_x': float(median[0]),
            'world_mag_y': float(median[1]),
            'world_mag_z': float(median[2]),
            'mag_norm_median': float(np.median(np.linalg.norm(plate.imu_trace.mag, axis=1))),
            'n_valid': int(valid.sum()),
            'n_samples': int(len(plate)),
        })
    return pd.DataFrame(rows)


def subject_field_from_trials(trial_fields: pd.DataFrame) -> np.ndarray:
    """A subject's global field: the median over every (trial, sensor) median it has.

    Two levels of median, on purpose. Pooling raw samples instead — what the predecessor did —
    weights a subject's field by recording length, and IMoVE sessions hold a 30 s static pose
    beside a 2800 s walk, so the walk would set the reference for both. Taking each (trial,
    sensor) as one vote makes the reference a property of the subject's magnetic environment
    rather than of which recording ran longest.

    Pooled over all SENSORS as well as all trials so that no segment is the privileged
    reference when segments are then compared against this field. That contrasts deliberately
    with `experiment_utils._compute_expected_mag_field`, which is torso-only: that one defines
    the EKF's mag oracle, where a single clean reference sensor is the point; here the
    reference has to be neutral between segments, since comparing segments to each other is the
    whole measurement.
    """
    vectors = trial_fields[['world_mag_x', 'world_mag_y', 'world_mag_z']].to_numpy(dtype=float)
    return np.median(vectors, axis=0)


def subject_field_table(subject: str, trial_fields: pd.DataFrame) -> pd.DataFrame:
    field = subject_field_from_trials(trial_fields)
    return pd.DataFrame([{
        'subject': subject,
        'world_mag_x': field[0], 'world_mag_y': field[1], 'world_mag_z': field[2],
        'world_mag_norm': float(np.linalg.norm(field)),
        'n_trials': int(trial_fields['trial'].nunique()) if 'trial' in trial_fields else 1,
        'n_sensors': int(trial_fields['sensor'].nunique()),
        'trials': ",".join(sorted(trial_fields['trial'].unique())) if 'trial' in trial_fields else '',
    }])

# ==============================================================================
# Static segmentation
# ==============================================================================

def mask_to_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True runs of `mask`, as half-open (start, end) sample indices."""
    diffs = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return list(zip(np.where(diffs == 1)[0], np.where(diffs == -1)[0]))


def intervals_to_mask(intervals: Sequence[Tuple[int, int]], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in intervals:
        mask[start:end] = True
    return mask


def drop_short_runs(mask: np.ndarray, min_samples: int) -> np.ndarray:
    """`mask` with every True run shorter than `min_samples` cleared."""
    if min_samples <= 1:
        return mask
    return intervals_to_mask([(s, e) for s, e in mask_to_intervals(mask)
                              if (e - s) >= min_samples], len(mask))


def static_mask(imu_trace: IMUTrace) -> np.ndarray:
    """Per-sample: was this sensor still?

    True wherever |gyro| stays under STATIC_GYRO_MAX across a centered STATIC_WINDOW_S window,
    for at least MIN_STATIC_S. See the STATIC_* constants for why the gyroscope alone decides
    this and what that misses.

    `min_periods=1` at the window edges rather than NaN, which is not a detail: the longest
    genuinely still stretches in both datasets are the pauses before the first task and after
    the last one, and those sit at the very start and end of the record. Requiring a full
    window there would discard exactly the samples the noise floor most wants.
    """
    fs = imu_trace.get_sample_frequency()
    window = max(int(round(STATIC_WINDOW_S * fs)), 2)
    speed = np.linalg.norm(imu_trace.gyro, axis=1)
    peak = pd.Series(speed).rolling(window=window, center=True, min_periods=1).max().to_numpy()
    return drop_short_runs(peak < STATIC_GYRO_MAX, max(int(round(MIN_STATIC_S * fs)), 1))


def body_static_mask(masks: Dict[str, np.ndarray], fs: float) -> np.ndarray:
    """Per-sample: was EVERY sensor on the body still at once?

    Not the same measurement as any one sensor's `static`, and both are reported. One foot can
    stand still while the subject shifts weight on the other, and a sensor that is merely
    slowly moving reads accelerations far above its own noise floor — so a per-sensor mask is
    the right window for "does the gravity assumption hold for THIS sensor right now", while
    only the intersection is the right window for an intrinsic noise floor.

    Each surviving run is trimmed by BODY_STATIC_MARGIN_S at both ends so the rolling-window
    detector's own edges, where the window straddles the transition into motion, are not
    measured as noise.
    """
    if not masks:
        return np.zeros(0, dtype=bool)
    n = min(len(mask) for mask in masks.values())
    combined = np.logical_and.reduce([mask[:n] for mask in masks.values()])
    margin = int(round(BODY_STATIC_MARGIN_S * fs))
    minimum = max(int(round(BODY_STATIC_MIN_S * fs)), 1)
    trimmed = [(start + margin, end - margin) for start, end in mask_to_intervals(combined)
               if (end - start) >= minimum + 2 * margin]
    return intervals_to_mask(trimmed, n)


def label_posture_intervals(plates: Dict[str, PlateTrial], spec: DatasetSpec, fs: float
                            ) -> Dict[str, List[Tuple[int, int]]]:
    """Splits a trial into 'sitting' / 'standing' / 'ambulation' intervals from the pelvis.

    Al Borno's complexTasks protocol cycles through sitting, standing, stair ascent/descent,
    side-stepping, walking and running. Every ambulation task involves repeated pelvis impacts,
    so a quiet pelvis isolates sitting+standing from the rest. Sitting vs. standing is then
    split by pelvis height on the up axis: sitting drops the pelvis well below its
    trial-typical upright height, standing does not. A walking trial has no sitting and this
    returns an almost-all-ambulation labeling, which is the correct answer for it.

    "Quiet" is measured from pelvis LINEAR ACCELERATION (gravity removed), not gyro. The gyro
    version this replaced badly undercounted both sitting and standing, because postural sway
    and fidgeting while stationary produce more angular than translational noise: it fragmented
    a single ~19 s seated bout into a ~10 s "sitting" chunk plus a fidgety tail that came out
    labeled as ambulation. Linear acceleration separates the two cases far more sharply —
    stationary stays under ~0.5 m/s^2 even with fidgeting, while every footstep of real
    ambulation is a multi-m/s^2 transient.

    This is a POSTURE labeling and NOT the static/non-static split the rest of this module
    reports over. It describes what the subject was doing; `static_mask` describes what one
    sensor was doing, which is the finer and less circular of the two (this one reads the
    accelerometer, one of the two channels under test). Both are written out because they
    answer different questions, and the sitting bout in particular is the case that motivates
    gating on observability.

    Known limitation: a single quiet run is labeled from its MEAN height, so sitting and
    standing are only separated when the movement between them interrupts the run.
    """
    pelvis = plates.get(spec.pelvis_sensor)
    if pelvis is None:
        return {'sitting': [], 'standing': [], 'ambulation': []}

    height = pelvis.world_trace.positions[:, HEIGHT_AXIS]
    linacc = np.linalg.norm(pelvis.get_imu_trace_in_global_frame().acc - EXPECTED_GRAVITY, axis=1)
    valid = np.asarray(pelvis.valid)
    # Outside `valid` the pose is padded, so both the height and the world-frame acceleration
    # are fictions there; treat those samples as unlabelable rather than as quiet-and-upright.
    linacc = np.where(valid, linacc, np.nan)

    window = max(int(round(QUIET_WINDOW_S * fs)), 2)
    rolling_std = pd.Series(linacc).rolling(window=window, center=True).std().to_numpy()
    is_quiet = (rolling_std < QUIET_LINACC_THRESHOLD) & (~np.isnan(rolling_std))

    baseline_height = float(np.median(height[valid])) if valid.any() else float(np.median(height))
    sitting, standing = [], []
    for start, end in mask_to_intervals(is_quiet):
        if (end - start) < int(MIN_STANDING_S * fs):
            continue
        if (baseline_height - float(height[start:end].mean())) > SIT_HEIGHT_DROP_M:
            if (end - start) >= int(MIN_SITTING_S * fs):
                sitting.append((start, end))
        else:
            standing.append((start, end))

    # Everything else splits three ways, not two. A stretch with no mocap has no height and no
    # world-frame acceleration, so it cannot be called sitting, standing OR ambulation — and
    # folding it into ambulation is not harmless: the Al Borno walking trials open with a ~430 s
    # standing pause before the cameras start, which the predecessor labelled as ambulation and
    # shaded as such under every time series.
    quiet = intervals_to_mask(sitting + standing, len(height))
    return {'sitting': sitting, 'standing': standing,
            'ambulation': mask_to_intervals(~quiet & valid),
            'unlabeled': mask_to_intervals(~quiet & ~valid)}


def intervals_table(posture: Dict[str, List[Tuple[int, int]]],
                    static_by_sensor: Dict[str, np.ndarray],
                    body_static: np.ndarray, timestamps: np.ndarray) -> pd.DataFrame:
    """Every labeled interval in one trial, on the trial's single timeline.

    One clock throughout. Non-destructive alignment means the plates carry the full inertial
    record, so a static interval and a sitting interval index the same timestamps — they used
    to index a trimmed and an untrimmed array respectively, which put shaded regions tens of
    seconds out of place.
    """
    def at(index: int) -> float:
        return float(timestamps[min(max(index, 0), len(timestamps) - 1)])

    rows = []
    for label, intervals in posture.items():
        for start, end in intervals:
            rows.append({'label': label, 'sensor': '', 'start_index': start, 'end_index': end,
                         'start_time': at(start), 'end_time': at(end)})
    for sensor, mask in sorted(static_by_sensor.items()):
        for start, end in mask_to_intervals(mask):
            rows.append({'label': 'static', 'sensor': sensor, 'start_index': start, 'end_index': end,
                         'start_time': at(start), 'end_time': at(end)})
    for start, end in mask_to_intervals(body_static):
        rows.append({'label': 'body_static', 'sensor': '', 'start_index': start, 'end_index': end,
                     'start_time': at(start), 'end_time': at(end)})

    df = pd.DataFrame(rows, columns=['label', 'sensor', 'start_index', 'end_index',
                                     'start_time', 'end_time'])
    df['duration_s'] = df['end_time'] - df['start_time']
    return df

# ==============================================================================
# Per-sample quantities
# ==============================================================================

def smooth_observability(observability: np.ndarray, fs: float,
                         cutoff: float = OBS_FILTER_CUTOFF_HZ, order: int = OBS_FILTER_ORDER,
                         winsorize_pct: float = OBS_WINSORIZE_PCT) -> np.ndarray:
    """Zero-lag low-pass filter for o^J (see OBS_FILTER_CUTOFF_HZ for why it is filtered).

    Two guards, both because o^J is a non-negative spiky signal rather than a smooth one: the
    top percentile is capped BEFORE filtering, since feeding impulses straight into filtfilt
    makes it ring and overshoot negative just after each spike, and the output is clamped at 0
    afterwards so nothing downstream sees a negative observability.

    Returns the input unchanged when the cutoff is at or above Nyquist. IMoVE's 40 Hz sessions
    have a 20 Hz Nyquist against this 25 Hz cutoff, so the filter is not merely a no-op there —
    `butter` would raise on a normalized cutoff above 1.
    """
    nyquist = fs / 2.0
    if len(observability) < 3 * (order + 1) or cutoff >= nyquist:
        return np.maximum(observability, 0.0)
    capped = np.minimum(observability, np.percentile(observability, winsorize_pct))
    b, a = butter(order, cutoff / nyquist, btype='low')
    return np.maximum(filtfilt(b, a, capped), 0.0)


def joint_center_observability(parent_plate: PlateTrial, child_plate: PlateTrial
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """(parent, child) per-sample observability at the shared joint center.

    Both plates are projected to the joint center first, exactly as the filter does before
    gating on o^J, so these are the same numbers mag_adapt thresholds. Returned as a pair
    rather than reduced with min(): a joint's own observability is the minimum of the two, but
    the per-SEGMENT figures need to know which of the two sensors was the limiting one.
    """
    parent_proj, child_proj = project_pair_to_joint_center(parent_plate, child_plate)
    return (segment_observability(parent_proj.imu_trace),
            segment_observability(child_proj.imu_trace))


def segment_samples(plates: Dict[str, PlateTrial], spec: DatasetSpec, global_field: np.ndarray,
                    static_by_sensor: Dict[str, np.ndarray], body_static: np.ndarray,
                    stride: int = SAMPLE_STRIDE) -> pd.DataFrame:
    """Per-sample, per-sensor: both assumptions' departure, the raw signal norms, and the
    regime masks that split them.

    `linacc` removes gravity in the WORLD frame (|a_world - g|), which is what makes it
    comparable across segments: subtracting a constant in each sensor's own body frame would
    leave every segment's own orientation in the result. `magdev` is measured against the
    subject's global field, so it is a distortion magnitude, not a field magnitude, and
    `magdev_angle` is the same disagreement expressed as a direction error — which is what an
    orientation filter actually consumes, and the form in which a magnetometer's magnitude
    error is free and its direction error is not.

    Both of those need the mocap rotation and are therefore NaN outside `valid`. The
    predecessor computed them everywhere, which meant rotating real readings by the constant
    padded pose the world trace holds outside the mocap window and folding the result into
    every distribution — for Subject01 that was 43656 of 103981 frames.

    `acc_norm_dev`, `mag_norm_dev` and the three raw norms need no rotation and are kept over
    the FULL record. They are the reference-free half of the same question: a sensor reading
    gravity alone has |acc| = |g| whatever its orientation, and one reading a single constant
    field has |mag| = |m_global| whatever its orientation, so any departure is real. Unlike
    `linacc` and `magdev` they are things a filter could measure for itself at runtime — and,
    for the trials whose still stretches fall outside the mocap window, they are the ONLY
    static-regime numbers available.

    `mag_norm_dev` carries a caveat `acc_norm_dev` does not: |g| is known exactly while
    |m_global| is estimated from these same sensors, so a per-sensor calibration gain error
    lands in this column as if it were distortion. `mag_norm_std` is the gain-free companion —
    the variability of |mag|, which a constant gain cannot move — and the two should be read
    together.
    """
    sensor_segment = {sensor: segment for segment, sensor in spec.segment_sensor.items()}
    global_norm = float(np.linalg.norm(global_field))
    blocks = []
    for segment, sensor in spec.segment_sensor.items():
        plate = plates.get(sensor)
        if plate is None:
            continue
        local = plate.imu_trace
        world = plate.get_imu_trace_in_global_frame()
        valid = np.asarray(plate.valid)
        n = len(plate)

        linacc = np.linalg.norm(world.acc - EXPECTED_GRAVITY, axis=1)
        mag_error = world.mag - global_field
        magdev = np.linalg.norm(mag_error, axis=1)
        magdev_angle = angle_between_deg(world.mag, np.broadcast_to(global_field, world.mag.shape))
        mocap_only = np.where(valid, 1.0, np.nan)

        static = static_by_sensor.get(sensor, np.zeros(n, dtype=bool))[:n]
        body = body_static[:n] if len(body_static) >= n else np.zeros(n, dtype=bool)

        block = pd.DataFrame({
            'timestamp': local.timestamps.astype(np.float64),
            'sensor': sensor,
            'segment': sensor_segment.get(sensor, sensor),
            'valid': valid,
            'static': static,
            'body_static': body,
            'linacc': (linacc * mocap_only).astype(np.float32),
            'magdev': (magdev * mocap_only).astype(np.float32),
            'magdev_angle': (magdev_angle * mocap_only).astype(np.float32),
            'acc_norm': np.linalg.norm(local.acc, axis=1).astype(np.float32),
            'gyro_norm': np.linalg.norm(local.gyro, axis=1).astype(np.float32),
            'mag_norm': np.linalg.norm(local.mag, axis=1).astype(np.float32),
        })
        block['acc_norm_dev'] = np.abs(block['acc_norm'] - GRAVITY_MAGNITUDE).astype(np.float32)
        block['mag_norm_dev'] = np.abs(block['mag_norm'] - global_norm).astype(np.float32)
        blocks.append(block.iloc[::stride] if stride > 1 else block)

    if not blocks:
        return pd.DataFrame()
    out = pd.concat(blocks, ignore_index=True)
    out['sensor'] = out['sensor'].astype('category')
    out['segment'] = out['segment'].astype('category')
    return out


def joint_samples(plates: Dict[str, PlateTrial], spec: DatasetSpec, fs: float,
                  static_by_sensor: Dict[str, np.ndarray], body_static: np.ndarray,
                  stride: int = SAMPLE_STRIDE) -> pd.DataFrame:
    """Per-sample, per-joint observability at the joint center, for both of the joint's
    sensors, with the regime masks that split it.

    `obs_min` is the joint's own o^J: a joint's relative orientation is only as observable as
    the less informative of its two accelerometers, so this is a minimum and not a sum or a
    mean. It is stored both smoothed (see `smooth_observability`) and raw — the smoothed series
    is what the figures and quantiles use, the raw one is what a causal filter would actually
    threshold, so the gating duty cycle has to be computed on it.

    A joint counts as `static` only when BOTH its sensors are, which is the same minimum
    argument applied to the regime: a knee with a planted shank and a swinging thigh is not a
    static knee.
    """
    rows = []
    for joint, (parent_sensor, child_sensor) in spec.joints.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        parent_plate, child_plate = plates[parent_sensor], plates[child_sensor]
        obs_parent, obs_child = joint_center_observability(parent_plate, child_plate)
        n = min(len(obs_parent), len(obs_child))
        obs_parent, obs_child = obs_parent[:n], obs_child[:n]
        raw_min = np.minimum(obs_parent, obs_child)

        zeros = np.zeros(n, dtype=bool)
        static = (static_by_sensor.get(parent_sensor, zeros)[:n]
                  & static_by_sensor.get(child_sensor, zeros)[:n])
        body = body_static[:n] if len(body_static) >= n else zeros
        valid = np.asarray(parent_plate.valid)[:n] & np.asarray(child_plate.valid)[:n]

        block = pd.DataFrame({
            'timestamp': parent_plate.imu_trace.timestamps[:n].astype(np.float64),
            'joint': joint,
            'valid': valid,
            'static': static,
            'body_static': body,
            'obs_parent': smooth_observability(obs_parent, fs).astype(np.float32),
            'obs_child': smooth_observability(obs_child, fs).astype(np.float32),
            'obs_min': smooth_observability(raw_min, fs).astype(np.float32),
            'obs_min_raw': raw_min.astype(np.float32),
        })
        rows.append(block.iloc[::stride] if stride > 1 else block)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out['joint'] = out['joint'].astype('category')
    return out

# ==============================================================================
# Per-sensor scalars: assumption departure and the intrinsic noise floor
# ==============================================================================

def regime_masks(static: np.ndarray, body_static: np.ndarray, n: int) -> Dict[str, np.ndarray]:
    """The four regime masks for one sensor or joint, all length `n`."""
    static = static[:n] if len(static) >= n else np.zeros(n, dtype=bool)
    body = body_static[:n] if len(body_static) >= n else np.zeros(n, dtype=bool)
    return {'all': np.ones(n, dtype=bool), 'static': static,
            'nonstatic': ~static, 'body_static': body}


def interval_noise(trace: IMUTrace, mask: np.ndarray) -> Dict[str, float]:
    """Per-axis noise floor for one sensor over one regime: the length-weighted mean of the
    per-axis standard deviation WITHIN each static interval.

    Within each interval, not over the pooled mask, and that is the whole estimator. A sensor
    sitting still reads a constant gravity and field vector whose value depends on the
    orientation it happens to be in, so two separate pauses in two different poses have
    genuinely different means. Pooling them first would measure the difference between those
    poses — tens of degrees of orientation change, i.e. metres per second squared — and call it
    sensor noise. Taking the std inside each interval and then averaging removes every
    between-interval offset and leaves only what varies while the sensor is not moving.

    Length-weighted so a 30 s pause counts for more than a 0.1 s stance phase, and
    `n_noise_samples` is carried alongside so that pooling ACROSS trials can weight the same
    way. Intervals shorter than two samples contribute nothing: a standard deviation needs at
    least two.
    """
    result: Dict[str, float] = {'n_noise_samples': 0, 'n_noise_intervals': 0}
    for modality in ('gyro', 'acc', 'mag'):
        for axis in 'xyz':
            result[f'{modality}_noise_{axis}'] = np.nan

    weighted = {modality: np.zeros(3) for modality in ('gyro', 'acc', 'mag')}
    total, count = 0, 0
    for start, end in mask_to_intervals(mask):
        end = min(end, len(trace))
        if end - start < 2:
            continue
        length = end - start
        total += length
        count += 1
        for modality in weighted:
            weighted[modality] += np.std(getattr(trace, modality)[start:end], axis=0) * length
    if total == 0:
        return result

    result['n_noise_samples'] = int(total)
    result['n_noise_intervals'] = int(count)
    for modality, value in weighted.items():
        for i, axis in enumerate('xyz'):
            result[f'{modality}_noise_{axis}'] = float(value[i] / total)
    return result


def mocap_motion_check(plate: PlateTrial, mask: np.ndarray) -> Dict[str, float]:
    """What the MOCAP says the sensor was doing over a mask the GYRO called static.

    The static detector reads only the gyroscope (see the STATIC_* constants), which is what
    keeps it from defining the accelerometer's answer into existence — and what leaves it blind
    to pure translation. This is the independent check on that blind spot, and it is reported
    rather than asserted: if these speeds are not small, the static regime is not static and
    every number computed over it is wrong.

    Valid frames only, since it is the mocap being consulted. NaN when the regime has no valid
    frame at all, which is the normal case for the pauses that sit outside the mocap window.
    """
    result = {'mocap_speed_median_mm_s': np.nan, 'mocap_speed_p95_mm_s': np.nan,
              'mocap_angular_speed_median_deg_s': np.nan, 'mocap_angular_speed_p95_deg_s': np.nan,
              'n_mocap_check': 0}
    world = plate.world_trace
    n = min(len(mask), len(world) - 1)
    if n <= 1:
        return result

    dt = np.diff(world.timestamps)[:n, None]
    speed = np.linalg.norm(np.diff(world.positions, axis=0)[:n] / dt, axis=1)
    # Relative rotation between consecutive frames, as an angle. trace(R) = 1 + 2 cos(theta).
    relative = np.einsum('nij,nkj->nik', world.rotations[1:n + 1], world.rotations[:n])
    cos_theta = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angular = np.degrees(np.arccos(cos_theta)) / dt[:, 0]

    valid = np.asarray(plate.valid)
    selected = mask[:n] & valid[:n] & valid[1:n + 1]
    if selected.sum() < 2:
        return result
    result['n_mocap_check'] = int(selected.sum())
    result['mocap_speed_median_mm_s'] = float(np.median(speed[selected]) * 1000.0)
    result['mocap_speed_p95_mm_s'] = float(np.percentile(speed[selected], 95) * 1000.0)
    result['mocap_angular_speed_median_deg_s'] = float(np.median(angular[selected]))
    result['mocap_angular_speed_p95_deg_s'] = float(np.percentile(angular[selected], 95))
    return result


def sensor_stats(plates: Dict[str, PlateTrial], spec: DatasetSpec, global_field: np.ndarray,
                 static_by_sensor: Dict[str, np.ndarray], body_static: np.ndarray,
                 fs: float) -> pd.DataFrame:
    """Per-(sensor, regime) scalars: how far each assumption fails, and the noise floor it
    fails against.

    Computed on the FULL unstrided record, unlike the per-sample tables, so every number quoted
    in the report is exact rather than estimated off a subsample.

    Two spans within each regime, on purpose. Anything needing a mocap ROTATION — linacc,
    magdev, the world-frame field — is restricted to VALID frames, because outside them the
    world trace holds a constant padded pose and rotating a real reading by it puts the vector
    somewhere it never was. The reference-free magnitudes (|acc|, |mag|, and the noise floor)
    are taken over the whole record, because they need no orientation and because the longest
    genuinely still stretches sit outside the mocap window: recovering those is what the
    non-destructive alignment was for, and holding them to `valid` would throw the best noise
    data away. `n_samples` and `n_valid` say which span each column had.
    """
    global_norm = float(np.linalg.norm(global_field))
    rows = []
    for segment, sensor in spec.segment_sensor.items():
        plate = plates.get(sensor)
        if plate is None:
            continue
        local, world = plate.imu_trace, plate.get_imu_trace_in_global_frame()
        valid = np.asarray(plate.valid)
        n = len(plate)

        acc_norm = np.linalg.norm(local.acc, axis=1)
        mag_norm = np.linalg.norm(local.mag, axis=1)
        linacc = np.linalg.norm(world.acc - EXPECTED_GRAVITY, axis=1)
        magdev = np.linalg.norm(world.mag - global_field, axis=1)
        magdev_angle = angle_between_deg(world.mag, np.broadcast_to(global_field, world.mag.shape))

        for regime, mask in regime_masks(static_by_sensor.get(sensor, np.zeros(n, dtype=bool)),
                                         body_static, n).items():
            if not mask.any():
                continue
            mocap = mask & valid
            # The noise floor and the mocap cross-check are only meaningful over a regime that
            # claims to be still; over 'all' or 'nonstatic' they would report the subject's
            # motion under a column named "noise". An empty mask fills the columns with NaN,
            # which keeps the schema identical across regimes so the table stays rectangular.
            noise_mask = mask if regime in NOISE_REGIMES else np.zeros(n, dtype=bool)
            rows.append({
                'sensor': sensor, 'segment': segment, 'regime': regime,
                'n_samples': int(mask.sum()), 'n_valid': int(mocap.sum()),
                'duration_s': float(mask.sum() / fs),
                'coverage': float(mask.sum() / n),
                # Reference-free, whole record.
                'acc_norm_median': float(np.median(acc_norm[mask])),
                'acc_norm_std': float(np.std(acc_norm[mask])),
                'acc_norm_dev_rms': float(np.sqrt(np.mean((acc_norm[mask] - GRAVITY_MAGNITUDE) ** 2))),
                'mag_norm_median': float(np.median(mag_norm[mask])),
                'mag_norm_std': float(np.std(mag_norm[mask])),
                'mag_norm_dev_rms': float(np.sqrt(np.mean((mag_norm[mask] - global_norm) ** 2))),
                # Mocap-referenced, valid frames only.
                **_mocap_referenced(linacc, magdev, magdev_angle, world.mag, mocap),
                **interval_noise(local, noise_mask),
                **mocap_motion_check(plate, noise_mask),
            })
    return pd.DataFrame(rows)


def _mocap_referenced(linacc: np.ndarray, magdev: np.ndarray, magdev_angle: np.ndarray,
                      world_mag: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """The half of `sensor_stats` that needs a mocap rotation, or NaNs where there is none."""
    keys = ('linacc_rms', 'linacc_median', 'linacc_p95', 'magdev_rms', 'magdev_median',
            'magdev_p95', 'magdev_angle_median', 'magdev_angle_p95',
            'world_mag_x', 'world_mag_y', 'world_mag_z')
    if mask.sum() < 2:
        return {key: np.nan for key in keys}
    return {
        'linacc_rms': float(np.sqrt(np.mean(linacc[mask] ** 2))),
        'linacc_median': float(np.median(linacc[mask])),
        'linacc_p95': float(np.percentile(linacc[mask], 95)),
        'magdev_rms': float(np.sqrt(np.mean(magdev[mask] ** 2))),
        'magdev_median': float(np.median(magdev[mask])),
        'magdev_p95': float(np.percentile(magdev[mask], 95)),
        'magdev_angle_median': float(np.median(magdev_angle[mask])),
        'magdev_angle_p95': float(np.percentile(magdev_angle[mask], 95)),
        'world_mag_x': float(np.median(world_mag[mask, 0])),
        'world_mag_y': float(np.median(world_mag[mask, 1])),
        'world_mag_z': float(np.median(world_mag[mask, 2])),
    }


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between corresponding rows of two (N, 3) arrays, in degrees."""
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cosine = np.clip(np.sum(a * b, axis=1) / norms, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

# ==============================================================================
# Per-joint scalars: observability and local field consistency
# ==============================================================================

def joint_stats(plates: Dict[str, PlateTrial], spec: DatasetSpec, global_field: np.ndarray,
                static_by_sensor: Dict[str, np.ndarray], body_static: np.ndarray,
                fs: float) -> pd.DataFrame:
    """Per-(joint, regime): observability, the local-field consistency test, and the
    accelerometer residual the relative filter actually forms.

    All three on the full unstrided record, and all three split by regime, because the regime
    split is where each of them says something a pooled number hides — o^J collapses when a
    joint stops moving, which is exactly when the magnetometer is most trustworthy and least
    needed to be gated.
    """
    rows = []
    for joint, (parent_sensor, child_sensor) in spec.joints.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        parent_plate, child_plate = plates[parent_sensor], plates[child_sensor]
        parent_proj, child_proj = project_pair_to_joint_center(parent_plate, child_plate)

        obs_parent = segment_observability(parent_proj.imu_trace)
        obs_child = segment_observability(child_proj.imu_trace)
        n = min(len(obs_parent), len(obs_child))
        obs_min = np.minimum(obs_parent[:n], obs_child[:n])

        parent_mag = parent_plate.get_imu_trace_in_global_frame().mag[:n]
        child_mag = child_plate.get_imu_trace_in_global_frame().mag[:n]
        projected_acc = (parent_proj.get_imu_trace_in_global_frame().acc[:n],
                         child_proj.get_imu_trace_in_global_frame().acc[:n])
        raw_acc = (parent_plate.get_imu_trace_in_global_frame().acc[:n],
                   child_plate.get_imu_trace_in_global_frame().acc[:n])
        valid = np.asarray(parent_plate.valid)[:n] & np.asarray(child_plate.valid)[:n]

        zeros = np.zeros(n, dtype=bool)
        joint_static = (static_by_sensor.get(parent_sensor, zeros)[:n]
                        & static_by_sensor.get(child_sensor, zeros)[:n])

        for regime, mask in regime_masks(joint_static, body_static, n).items():
            if not mask.any():
                continue
            mocap = mask & valid
            row = {
                'joint': joint, 'parent_sensor': parent_sensor, 'child_sensor': child_sensor,
                'regime': regime, 'n_samples': int(mask.sum()), 'n_valid': int(mocap.sum()),
                'duration_s': float(mask.sum() / fs),
                'obs_min_median': float(np.median(obs_min[mask])),
                'obs_min_mean': float(np.mean(obs_min[mask])),
                'obs_parent_median': float(np.median(obs_parent[:n][mask])),
                'obs_child_median': float(np.median(obs_child[:n][mask])),
                # What fraction of this regime the default gate would call unobservable, on the
                # RAW metric — the smoothed one is a display convenience and is not what a
                # causal filter thresholds.
                'frac_below_gate': float(np.mean(obs_min[mask] < DEFAULT_MAG_ADAPT_THRESHOLD)),
                'gate_threshold': DEFAULT_MAG_ADAPT_THRESHOLD,
            }
            for quantile in QUANTILES:
                row[f"obs_min_p{int(round(quantile * 100)):02d}"] = float(
                    np.quantile(obs_min[mask], quantile))
            row.update(field_consistency(joint, spec, parent_sensor, child_sensor,
                                         parent_mag, child_mag, mocap))
            row.update(acc_residual(projected_acc, raw_acc, mocap))
            rows.append(row)
    return pd.DataFrame(rows)


def field_consistency(joint: str, spec: DatasetSpec, parent_sensor: str, child_sensor: str,
                      parent_mag: np.ndarray, child_mag: np.ndarray,
                      mask: np.ndarray) -> Dict[str, object]:
    """Per-joint test of the premise behind estimating the field locally: is one sensor's field
    better predicted by its NEIGHBOUR across the joint than by the subject's global field?

    Which sensor of the pair is the reference and which is the target is set by the dataset's
    `field_reference` (default: the kinematic parent references the child). Both directions are
    computed either way, in var_reduction_parent_ref and var_reduction_child_ref, so nothing
    about the choice is baked in irreversibly.

    Two views of the same comparison, both in the world frame:
      * cosine similarity of the target's field against each candidate reference — a
        direction-only measure, which is what an orientation filter actually consumes. The
        target-vs-reference similarity is symmetric, so it is the one number here that the
        direction choice cannot affect;
      * var_reduction = 1 - Var(target - reference) / Var(target - global), the fraction of the
        target's residual variance that using the neighbour instead of the global field removes.

    Read var_reduction with its algebra in mind. The global field is a constant, so
    Var(target - global) = Var(target), and the whole thing collapses to

        var_reduction = [2 Cov(t,r) - Var(r)] / Var(t) = 2 rho sqrt(k) - k,
        where k = Var(reference)/Var(target) and rho = corr(target, reference)

    which is positive iff Var(reference)/Var(target) < 4 rho^2. THE SIGN IS SET BY THE VARIANCE
    RATIO BETWEEN THE TWO SENSORS, not by whether the neighbour carries information about the
    target's field. A negative value does not mean the neighbour is uninformative: it means
    substituting its field verbatim (unit gain) imports more of the reference's own fluctuation
    than it cancels of the target's. Fitting a gain instead recovers a large positive reduction
    on exactly those trials, which is what `optimal_gain` and `var_reduction_best` report.

    The variance framing also discards any CONSTANT offset between the two sensors, which is
    not free for a relative-orientation filter — a fixed field disagreement is a fixed
    orientation error. The cos-sim columns are what cover that side of it.

    Note this is a test of the premise (nearby sensors share a local field the global reference
    misses), not a simulation of the method: the filter estimates the field at the joint center
    rather than copying the parent's reading. Under the `static` regime, where neither sensor
    moves, both variances collapse toward the noise floor and every ratio here becomes a
    statement about noise rather than about field structure — which is why the report quotes
    these from the `all` regime and shows the others only as a contrast.
    """
    if mask.sum() < 3:
        return {}
    parent_mag, child_mag = parent_mag[mask], child_mag[mask]
    reference_role = spec.field_reference.get(joint, 'parent')
    if reference_role == 'parent':
        reference_mag, target_mag = parent_mag, child_mag
        reference_sensor, target_sensor = parent_sensor, child_sensor
    else:
        reference_mag, target_mag = child_mag, parent_mag
        reference_sensor, target_sensor = child_sensor, parent_sensor

    global_direction = np.broadcast_to(np.median(target_mag, axis=0), target_mag.shape)
    return {
        'reference_role': reference_role,
        'reference_sensor': reference_sensor,
        'target_sensor': target_sensor,
        'cos_sim_target_global': float(np.mean(_cosine_similarity(target_mag, global_direction))),
        # Symmetric in the pair, so unaffected by the reference direction.
        'cos_sim_target_reference': float(np.mean(_cosine_similarity(target_mag, reference_mag))),
        'corr_target_reference': _pair_correlation(target_mag, reference_mag),
        'variance_ratio_k': _variance_ratio(target_mag, reference_mag),
        'optimal_gain': _optimal_gain(target_mag, reference_mag),
        # The residual the relative filter actually forms for its mag measurement, and so the
        # scale its mag noise block should carry. RelativeFilter's get_h computes exactly
        # R_wp @ m_p - R_wc @ m_c and drives it to zero, so the RMS of that quantity is the
        # size of the disagreement it is calling noise. Compare against DEFAULT_MAG_STD.
        'mag_residual_rms': float(np.sqrt(np.mean(np.sum((parent_mag - child_mag) ** 2, axis=1)))),
        'mag_residual_angle': float(np.mean(angle_between_deg(parent_mag, child_mag))),
        # Global field = 0 on this scale by construction; see _best_var_reduction.
        'var_reduction_global': 0.0,
        'var_reduction': _var_reduction(target_mag, reference_mag),
        'var_reduction_best': _best_var_reduction(target_mag, reference_mag),
        'var_reduction_parent_ref': _var_reduction(child_mag, parent_mag),
        'var_reduction_child_ref': _var_reduction(parent_mag, child_mag),
    }


def acc_residual(projected: Tuple[np.ndarray, np.ndarray], raw: Tuple[np.ndarray, np.ndarray],
                 mask: np.ndarray) -> Dict[str, float]:
    """RMS of the ACC residual the relative filter actually forms at the joint center:
    |R_wp a_p - R_wc a_c|, the accelerometer twin of mag_residual_rms.

    Deliberately not the same quantity as experiments/acceleration_projection.py's err_proj, and
    the difference is the point. err_proj is each sensor's projection error against the mocap
    joint center — the right measure of whether the projection physics works. This is the
    DISAGREEMENT BETWEEN THE TWO SENSORS, which is what RelativeFilter.get_h drives to zero and
    therefore what its acc noise block has to cover. The two sensors' projection errors are not
    independent — they share one joint-center fit, one mocap alignment and one segment pair — so
    whatever they share CANCELS in the difference and setting acc_std from err_proj would badly
    over-inflate it.

    Gravity cancels identically in the difference, so this needs no gravity constant and no
    double-differentiated mocap position: only the ground-truth rotations that bring both
    readings into a common frame.

    `acc_residual_rms_raw` is the same residual WITHOUT the projection, so the pair says whether
    projecting to the joint center actually reduces the disagreement the filter has to absorb.
    """
    if mask.sum() < 2:
        return {'acc_residual_rms': np.nan, 'acc_residual_rms_raw': np.nan}

    def rms(pair):
        residual = pair[0][mask] - pair[1][mask]
        return float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))

    return {'acc_residual_rms': rms(projected), 'acc_residual_rms_raw': rms(raw)}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def _var_reduction(target: np.ndarray, reference: np.ndarray) -> float:
    """1 - Var(target - reference) / Var(target), summed over the world-frame components.

    Var(target - global) reduces to Var(target) because the global field is a constant, so no
    global field needs passing in — which also makes it obvious that this measures only the
    TIME-VARYING part of the disagreement. See `field_consistency` for how to read the sign."""
    var_target = float(np.sum(np.var(target, axis=0)))
    if var_target <= 0:
        return np.nan
    return 1.0 - float(np.sum(np.var(target - reference, axis=0))) / var_target


def _variance_ratio(target: np.ndarray, reference: np.ndarray) -> float:
    """k = Var(reference)/Var(target), component-summed. The quantity every reading of
    var_reduction turns on: k < 4*rho^2 decides its sign, and k = rho^2 is where substituting
    the reference verbatim happens to be the optimal thing to do."""
    var_target = float(np.sum(np.var(target, axis=0)))
    var_reference = float(np.sum(np.var(reference, axis=0)))
    return np.nan if var_target <= 0 else var_reference / var_target


def _optimal_gain(target: np.ndarray, reference: np.ndarray) -> float:
    """The gain a* minimizing Var(target - a*reference), i.e. rho/sqrt(k).

    Reported because it says how far unit substitution is from the best this pair can do, in
    units anyone can picture: a* = 0.5 means the reference's fluctuation is twice as large as
    the target's own and should be halved before use, and the cost of not halving it is exactly
    (rho - sqrt(k))^2 of the target's variance."""
    k = _variance_ratio(target, reference)
    correlation = _pair_correlation(target, reference)
    if np.isnan(k) or np.isnan(correlation) or k <= 0:
        return np.nan
    return float(correlation / np.sqrt(k))


def _best_var_reduction(target: np.ndarray, reference: np.ndarray) -> float:
    """The largest variance reduction the reference sensor could give: rho^2.

    Same scale and same zero as var_reduction, which is the point of reporting it. Minimizing
    Var(target - a*reference) over the gain a gives a residual of Var(target)(1 - rho^2), so the
    achievable reduction is rho^2 while var_reduction is what the same reference delivers with a
    forced to 1. The two therefore bracket the reference: rho^2 is what the pair COULD support,
    var_reduction is what substitution actually gets.

    This is also the form in which the correlation becomes reportable at all. Correlation
    against the global field is undefined — the global field is a constant, so its variance is
    zero and the ratio is 0/0 — which leaves a bare rho column with no baseline arm to compare
    to. On this scale the global field is exactly 0, so all three quantities sit on one axis
    with a real origin. Symmetric in the pair, so unlike var_reduction it cannot be moved by the
    reference direction choice."""
    correlation = _pair_correlation(target, reference)
    return np.nan if np.isnan(correlation) else float(correlation ** 2)


def _pair_correlation(target: np.ndarray, reference: np.ndarray) -> float:
    """Correlation between the two sensors' world-frame fields, POOLED over the components:

        sum_i Cov(t_i, r_i) / sqrt(sum_i Var(t_i) * sum_i Var(r_i))

    Not the mean of the three per-axis correlations, for two reasons. It is variance-weighted,
    which matters because an axis carrying almost no field variation contributes a per-axis
    correlation that is mostly noise against noise. And it is the rho for which
    var_reduction = 2 rho sqrt(k) - k holds exactly, which is the identity the whole sign
    discussion rests on.

    Blind to two things by construction, both of which the cos-sim columns cover instead: it is
    mean-centered, so a constant field disagreement between the sensors is invisible (and that
    one is NOT free — for a relative-orientation filter it is a constant orientation error), and
    it is scale-free, so an amplitude mismatch is invisible. High correlation therefore does not
    mean the two sensors see the same field, only that their fields move together."""
    var_target = float(np.sum(np.var(target, axis=0)))
    var_reference = float(np.sum(np.var(reference, axis=0)))
    if var_target <= 0 or var_reference <= 0:
        return np.nan
    # Population covariance (ddof=0), matching np.var above rather than np.cov's ddof=1 default:
    # rho and var_reduction are meant to satisfy var_reduction = 2*rho*sqrt(k) - k exactly, and
    # mixing the two conventions breaks that identity in the fourth decimal.
    centered = (target - target.mean(axis=0)) * (reference - reference.mean(axis=0))
    return float(np.sum(centered.mean(axis=0))) / np.sqrt(var_target * var_reference)

# ==============================================================================
# Per-trial driver / grid workers
# ==============================================================================

def trial_masks(plates: Dict[str, PlateTrial], fs: float
                ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Every sensor's static mask, plus the whole-body one built from their intersection."""
    static_by_sensor = {sensor: static_mask(plate.imu_trace) for sensor, plate in plates.items()}
    return static_by_sensor, body_static_mask(static_by_sensor, fs)


def compute_trial(plates: Dict[str, PlateTrial], spec: DatasetSpec, global_field: np.ndarray,
                  tables: Sequence[str] = RECOMPUTABLE_TABLES,
                  stride: int = SAMPLE_STRIDE) -> Dict[str, pd.DataFrame]:
    """Everything this experiment computes for one trial, as the per-trial tables.

    `tables` restricts the work to a subset (see --only-tables). Anything only one table needs
    is skipped when that table is not wanted, which is what makes a targeted rerun cheap: the
    two joint tables carry the rigid-body projection and dominate the runtime.
    """
    wanted = set(tables)
    reference = plates.get(spec.pelvis_sensor) or next(iter(plates.values()))
    fs = reference.imu_trace.get_sample_frequency()
    timestamps = reference.imu_trace.timestamps
    static_by_sensor, body_static = trial_masks(plates, fs)

    builders = {
        'segment_samples': lambda: segment_samples(plates, spec, global_field, static_by_sensor,
                                                   body_static, stride),
        'joint_samples': lambda: joint_samples(plates, spec, fs, static_by_sensor, body_static,
                                               stride),
        'intervals': lambda: intervals_table(label_posture_intervals(plates, spec, fs),
                                             static_by_sensor, body_static, timestamps),
        'sensor_stats': lambda: sensor_stats(plates, spec, global_field, static_by_sensor,
                                             body_static, fs),
        'joint_stats': lambda: joint_stats(plates, spec, global_field, static_by_sensor,
                                           body_static, fs),
    }
    return {table: build() for table, build in builders.items() if table in wanted}


def _field_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                  dataset: str = ALBORNO.name) -> None:
    """Stage one, one process per trial: the per-(trial, sensor) median world-frame field.

    Split from the main pass because the subject's global field is pooled over ALL of that
    subject's trials and every magdev in every trial is measured against it. Doing it in one
    pass would mean holding a whole session in memory at once — for IMoVE that is up to twelve
    trials including a 2800 s walk — or using a per-trial reference, and a reference that shifts
    between a subject's trials is not a reference.
    """
    subject, trial = row_key
    spec = get_dataset(dataset)
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()
    try:
        plates = _load_spec_plates(subject, trial, dataset, spec)
        table = trial_field_table(plates, spec)
        if table.empty:
            shared_state[(row_key, stage)] = "Skipped"
            return None
        _save(table, trial_table_path(dataset, subject, trial, 'trial_field'), dataset,
              subject=subject, trial=trial, table='trial_field')
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success"
    except Exception as e:  # a corrupt or unbuilt trial should not sink the session
        shared_state[(row_key, stage)] = f"Failed ({e})"
    return None


def _trial_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                  dataset: str = ALBORNO.name, tables: Sequence[str] = RECOMPUTABLE_TABLES,
                  stride: int = SAMPLE_STRIDE) -> None:
    """Stage two, one process per trial: every table, against the subject field stage one fed."""
    subject, trial = row_key
    spec = get_dataset(dataset)
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()
    try:
        global_field = read_subject_field(dataset, subject)
        if global_field is None:
            shared_state[(row_key, stage)] = "Failed (no subject field)"
            return None
        plates = _load_spec_plates(subject, trial, dataset, spec)
        for table, df in compute_trial(plates, spec, global_field, tables, stride).items():
            if df.empty:
                continue
            _save(df, trial_table_path(dataset, subject, trial, table), dataset,
                  subject=subject, trial=trial, table=table)
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success"
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
    return None


def _load_spec_plates(subject: str, trial: str, dataset: str,
                      spec: DatasetSpec) -> Dict[str, PlateTrial]:
    """The trial's plates, narrowed to the sensors this dataset's spec names.

    Narrowing matters for the whole-body static mask: it is an AND over every plate, so a
    sensor the spec does not analyse would still be able to veto every body-static interval in
    the trial. Raises if the trial has none of the spec's sensors, which means the spec and the
    build disagree and is not something to paper over with an empty table.
    """
    plates = load_trial(subject, trial, dataset=dataset)
    selected = {sensor: plate for sensor, plate in plates.items()
                if sensor in set(spec.segment_sensor.values())}
    if not selected:
        raise ValueError(f"{dataset}/{subject}/{trial}: none of the spec's sensors "
                         f"{sorted(spec.segment_sensor.values())} are in this trial "
                         f"({sorted(plates)}).")
    return selected


def aggregate_subject_fields(dataset: str, row_keys: Sequence[Tuple[str, str]]) -> List[str]:
    """Rolls stage one's per-trial tables up into one global field per subject. Returns the
    subjects it could write a field for."""
    written = []
    for subject in subjects_of(row_keys):
        frames = []
        for other_subject, trial in row_keys:
            if other_subject != subject:
                continue
            path = trial_table_path(dataset, subject, trial, 'trial_field')
            if path.exists():
                frames.append(pd.read_parquet(path, engine='pyarrow').assign(trial=trial))
        if not frames:
            continue
        trial_fields = pd.concat(frames, ignore_index=True)
        _save(subject_field_table(subject, trial_fields), subject_field_path(dataset, subject),
              dataset, subject=subject)
        written.append(subject)
    return written

# ==============================================================================
# Pooled summary
# ==============================================================================

SUMMARY_COLUMNS = (['dataset', 'subject', 'trial', 'regime', 'metric', 'unit', 'group_kind',
                    'group', 'n_samples', 'mean', 'std', 'min']
                   + [f"p{int(round(q * 100)):02d}" for q in QUANTILES] + ['max'])


def _describe(df: pd.DataFrame, value_col: str, keys: List[str]) -> pd.DataFrame:
    grouped = df.groupby(keys, observed=True)[value_col]
    stats = grouped.agg(n_samples='count', mean='mean', std='std', min='min', max='max')
    quantiles = grouped.quantile(QUANTILES).unstack()
    quantiles.columns = [f"p{int(round(q * 100)):02d}" for q in quantiles.columns]
    return stats.join(quantiles).reset_index()


def summarize(dataset: str, segment_df: pd.DataFrame, joint_df: pd.DataFrame) -> pd.DataFrame:
    """Tidy quantile table for every metric x regime x grouping, WITH MARGINS: rows where
    `subject` or `trial` is the literal string 'all' are the pooled version of the rows above
    them.

    Marginal rows are computed by re-aggregating the samples, not by averaging the per-trial
    quantiles — a mean of medians is not a median, and trials differ in length by two orders of
    magnitude in IMoVE. The three splits the report and the figures need are all here: per
    trial (subject + trial), per subject (subject, trials pooled), and pooled over everything.

    Distributions are summarized by quantiles rather than mean +- std because every metric here
    is strongly right-skewed (impulsive footfalls, one-sided magnetic distortion), so a standard
    deviation implies a symmetry that is not there. `n_samples` counts non-NaN values, so the
    mocap-referenced metrics report the valid-frame count rather than the row count.
    """
    sources: List[Tuple[pd.DataFrame, str, Tuple[str, ...]]] = []
    if not segment_df.empty:
        sources.append((segment_df, 'segment',
                        tuple(m for m in SEGMENT_METRICS if m in segment_df.columns)))
    if not joint_df.empty:
        sources.append((joint_df, 'joint',
                        tuple(m for m in JOINT_METRICS if m in joint_df.columns)))

    frames = []
    for df, group_kind, metrics in sources:
        for regime, mask in regime_row_masks(df).items():
            # Sliced once per regime and reused across every metric and every margin, rather
            # than building a four-copy long-form frame first. On the pooled IMoVE sample table
            # that copy is ~100 M rows and the peak resident set is what decides whether this
            # runs at all.
            block = df[mask] if mask is not None else df
            if block.empty:
                continue
            for metric in metrics:
                for by_subject, by_trial in [(True, True), (True, False), (False, False)]:
                    keys = (['subject'] if by_subject else []) + (['trial'] if by_trial else [])
                    keys += [group_kind]
                    part = _describe(block, metric, keys).rename(columns={group_kind: 'group'})
                    if not by_subject:
                        part['subject'] = 'all'
                    if not by_trial:
                        part['trial'] = 'all'
                    part['dataset'] = dataset
                    part['regime'] = regime
                    part['metric'] = metric
                    part['unit'] = METRIC_UNITS[metric]
                    part['group_kind'] = group_kind
                    frames.append(part)
    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    for column in ('subject', 'trial', 'regime', 'metric', 'group_kind', 'group'):
        out[column] = out[column].astype(str)
    return out[SUMMARY_COLUMNS]


def regime_row_masks(df: pd.DataFrame) -> Dict[str, Optional[np.ndarray]]:
    """{regime: boolean row selector}, with None standing for "every row".

    The sample tables carry `static` and `body_static` as boolean columns because that is the
    compact way to store them; the summary wants them as a categorical dimension it can group
    by. The regimes deliberately OVERLAP — `body_static` is a subset of `static`, and `all`
    contains everything — so this is a set of selectors rather than a partition.
    """
    static = df['static'].to_numpy()
    return {'all': None, 'static': static, 'nonstatic': ~static,
            'body_static': df['body_static'].to_numpy()}

# ==============================================================================
# Console report
# ==============================================================================

def _header(number: int, title: str, subtitle: str) -> None:
    print("\n" + "=" * 96)
    print(f"{number}. {title}")
    print(f"   ({subtitle})")
    print("=" * 96)


def _pooled(summary: pd.DataFrame, metric: str, regime: str, group: str) -> Optional[pd.Series]:
    rows = summary[(summary['subject'] == 'all') & (summary['trial'] == 'all')
                   & (summary['metric'] == metric) & (summary['regime'] == regime)
                   & (summary['group'] == group)]
    return None if rows.empty else rows.iloc[0]


def _weighted_noise(stats: pd.DataFrame, modality: str) -> Optional[np.ndarray]:
    """Length-weighted per-axis noise floor over a set of sensor-trial rows."""
    rows = stats[stats['n_noise_samples'] > 0]
    if rows.empty:
        return None
    weights = rows['n_noise_samples'].to_numpy(dtype=float)
    axes = np.array([rows[f'{modality}_noise_{axis}'].to_numpy(dtype=float) for axis in 'xyz'])
    with np.errstate(invalid='ignore'):
        return np.nansum(axes * weights, axis=1) / np.nansum(~np.isnan(axes) * weights, axis=1)


def report_headline(spec: DatasetSpec, summary: pd.DataFrame, sensor_df: pd.DataFrame) -> None:
    _header(1, "THE TWO GLOBAL ASSUMPTIONS, STATIC vs MOVING",
            "pooled over every sensor, subject and trial; median [IQR]")
    if summary.empty:
        print("No per-sample data found.")
        return
    print(f"  {'metric':<16}{'regime':<14}{'median':>12}{'IQR':>24}{'p95':>12}{'n':>16}")
    for metric in SEGMENT_METRICS + JOINT_METRICS:
        unit = METRIC_UNITS[metric]
        for regime in REGIMES:
            group_kind = 'joint' if metric in JOINT_METRICS else 'segment'
            groups = (list(spec.joints) if group_kind == 'joint' else list(spec.segment_sensor))
            rows = summary[(summary['subject'] == 'all') & (summary['trial'] == 'all')
                           & (summary['metric'] == metric) & (summary['regime'] == regime)
                           & (summary['group'].isin(groups))]
            if rows.empty:
                continue
            # Pool the per-group rows by sample count: these are quantiles, so a straight mean
            # of medians is wrong, but a count-weighted mean of medians is the closest honest
            # single number without re-reading every sample table.
            weights = rows['n_samples'].to_numpy(dtype=float)
            if weights.sum() == 0:
                continue

            def wmean(column: str) -> float:
                return float(np.nansum(rows[column].to_numpy(dtype=float) * weights) / weights.sum())

            iqr = f"[{wmean('p25'):.3g}, {wmean('p75'):.3g}]"
            print(f"  {metric if regime == 'all' else '':<16}{regime:<14}{wmean('p50'):>12.3g}"
                  f"{iqr:>24}{wmean('p95'):>12.3g}{int(weights.sum()):>16,}")
        print()
    print("linacc and magdev are what the ACC=GRAVITY and MAG=CONSTANT assumptions cost, in the")
    print("sensors' own units. Read the static/nonstatic pair, not either alone: the assumptions")
    print("are not uniformly wrong, they are wrong WHEN THE SENSOR MOVES, which is an argument")
    print("for gating a measurement update in time rather than for a larger constant noise term.")


def report_static_coverage(spec: DatasetSpec, sensor_df: pd.DataFrame) -> None:
    _header(2, "STATIC COVERAGE AND DETECTOR VALIDATION",
            "how much of each recording each sensor was still for, and what mocap says it was "
            "doing then")
    if sensor_df.empty:
        print("No sensor stats found.")
        return
    static = sensor_df[sensor_df['regime'] == 'static']
    body = sensor_df[sensor_df['regime'] == 'body_static']
    if static.empty:
        print("No static stretches detected in any trial.")
        return

    print(f"  {'segment':<16}{'static %':>10}{'body-static %':>15}{'mocap |v|':>14}"
          f"{'mocap |w|':>14}{'trials':>9}")
    print(f"  {'':<16}{'':>10}{'':>15}{'(mm/s, med)':>14}{'(deg/s, med)':>14}{'':>9}")
    for segment in [s for s in spec.segment_sensor if s in set(static['segment'])]:
        rows = static[static['segment'] == segment]
        body_rows = body[body['segment'] == segment]
        body_pct = 100 * body_rows['coverage'].mean() if not body_rows.empty else 0.0
        print(f"  {segment:<16}{100 * rows['coverage'].mean():>10.1f}{body_pct:>15.1f}"
              f"{rows['mocap_speed_median_mm_s'].median():>14.1f}"
              f"{rows['mocap_angular_speed_median_deg_s'].median():>14.2f}{len(rows):>9}")

    speeds = static['mocap_speed_p95_mm_s'].dropna()
    if not speeds.empty:
        print(f"\nThe static detector reads the GYROSCOPE ONLY, so that it cannot define the "
              f"accelerometer's\nand magnetometer's answers into existence — see the STATIC_* "
              f"constants. Its blind spot is\npure translation, and the two mocap columns are "
              f"the independent check on it: across every\nsensor-trial the p95 mocap speed "
              f"during detected-static samples is {speeds.median():.1f} mm/s (median over\n"
              f"{len(speeds)} sensor-trials, worst {speeds.max():.1f} mm/s), at or below marker "
              f"reconstruction noise.")

    # How much of the static time the mocap-referenced metrics can actually be computed over.
    # This is not a footnote on these datasets: the Al Borno walking trials open with a long
    # standing pause BEFORE the cameras start, so their static regime is real, large, and
    # entirely invisible to any quantity needing a rotation.
    covered = static['n_valid'].sum() / max(static['n_samples'].sum(), 1)
    per_trial = (static.groupby(['subject', 'trial'], observed=True)[['n_samples', 'n_valid']]
                 .sum())
    dark = per_trial[per_trial['n_valid'] < 0.01 * per_trial['n_samples']]
    print(f"\nMOCAP COVERAGE OF THE STATIC REGIME: {100 * covered:.1f}% of static samples fall "
          f"inside the mocap\nwindow. The mocap-referenced metrics (linacc, magdev, "
          f"magdev_angle) exist only there; the\nreference-free pair (acc_norm_dev, "
          f"mag_norm_dev) is computed over the full record and is what\ncarries the "
          f"static/moving comparison on the rest.")
    if not dark.empty:
        listed = ", ".join(f"{s}/{t}" for s, t in list(dark.index)[:6])
        more = f" (+{len(dark) - 6} more)" if len(dark) > 6 else ""
        print(f"  {len(dark)} trial(s) have effectively NO static sample with mocap: {listed}{more}")


def report_acceleration(spec: DatasetSpec, sensor_df: pd.DataFrame) -> None:
    _header(3, "ACCELEROMETER: HOW FAR THE READING DEPARTS FROM GRAVITY ALONE",
            "per-trial values averaged across trials, proximal to distal")
    if sensor_df.empty:
        print("No sensor stats found.")
        return
    print(f"  {'segment':<16}{'regime':<13}{'RMS |a-g|':>12}{'median':>10}{'p95':>10}"
          f"{'||a|-|g||':>11}{'per-axis':>10}{'vs acc_std':>12}")
    print(f"  {'':<16}{'':<13}{'(m/s^2)':>12}{'(m/s^2)':>10}{'(m/s^2)':>10}{'(m/s^2)':>11}"
          f"{'(m/s^2)':>10}{'':>12}")
    for segment in [s for s in spec.segment_sensor if s in set(sensor_df['segment'])]:
        for regime in ('all', 'static', 'nonstatic'):
            rows = sensor_df[(sensor_df['segment'] == segment) & (sensor_df['regime'] == regime)]
            if rows.empty:
                continue
            means = rows[['linacc_rms', 'linacc_median', 'linacc_p95', 'acc_norm_dev_rms']].mean()
            per_axis = means['linacc_rms'] / np.sqrt(3)
            print(f"  {segment if regime == 'all' else '':<16}{regime:<13}"
                  f"{means['linacc_rms']:>12.3f}{means['linacc_median']:>10.3f}"
                  f"{means['linacc_p95']:>10.3f}{means['acc_norm_dev_rms']:>11.3f}"
                  f"{per_axis:>10.3f}{per_axis / DEFAULT_ACC_STD:>11.0f}x")
        print()
    print("The first three columns need mocap and are blank wherever a regime has no valid "
          "frame; the\nfourth, ||a|-|g||, needs none and is always populated. It is also the "
          "weaker measure by\nconstruction — acceleration perpendicular to gravity adds in "
          "quadrature, so a segment\nswinging horizontally reads |a| ~ |g| throughout while its "
          "non-gravity component is large.\nRead it as a floor on the departure, not an "
          "estimate of it.\n")

    ratios = (sensor_df[sensor_df['regime'] == 'nonstatic'].groupby('segment')['linacc_rms'].mean()
              / sensor_df[sensor_df['regime'] == 'static'].groupby('segment')['linacc_rms'].mean())
    ratios = ratios.dropna().sort_values()
    if not ratios.empty:
        print(f"Moving/static ratio of RMS |a-g| ranges {ratios.iloc[0]:.0f}x ({ratios.index[0]}) to "
              f"{ratios.iloc[-1]:.0f}x ({ratios.index[-1]}).\nThe last column is what matters for "
              f"tuning: an orientation filter treats the accelerometer as a\ngravity reference, so "
              f"the non-gravity component is measurement ERROR, and comparing it against\nthe "
              f"acc_std={DEFAULT_ACC_STD} the filter is tuned with says how far off that tuning is. "
              f"A single constant\ncannot describe both rows of any segment at once — far too small "
              f"while moving and far too\nlarge while still — which is the case for gating in time.")
    print(f"\nNote this is the PER-SENSOR departure from gravity, which is not what the RELATIVE "
          f"filter has\nto absorb: that filter's acc residual is the DIFFERENCE of two sensors "
          f"projected to a shared\njoint center, and the two projections' errors largely cancel. "
          f"See acc_residual_rms in\njoint_stats for the pairwise number, which is the one to tune "
          f"acc_std from.")


def report_magnetometer(spec: DatasetSpec, sensor_df: pd.DataFrame) -> None:
    _header(4, "MAGNETOMETER: HOW FAR THE FIELD DEPARTS FROM ONE GLOBAL CONSTANT",
            "deviation from the subject's own global field, per-trial values averaged across "
            "trials")
    if sensor_df.empty:
        print("No sensor stats found.")
        return
    print(f"  {'segment':<16}{'regime':<13}{'RMS dev':>11}{'median':>10}{'angle':>10}"
          f"{'angle p95':>11}{'||m|-|M||':>11}{'std |m|':>10}")
    print(f"  {'':<16}{'':<13}{f'({MAG_UNIT})':>11}{f'({MAG_UNIT})':>10}{'(deg)':>10}"
          f"{'(deg)':>11}{f'({MAG_UNIT})':>11}{f'({MAG_UNIT})':>10}")
    for segment in [s for s in spec.segment_sensor if s in set(sensor_df['segment'])]:
        for regime in ('all', 'static', 'nonstatic'):
            rows = sensor_df[(sensor_df['segment'] == segment) & (sensor_df['regime'] == regime)]
            if rows.empty:
                continue
            means = rows[['magdev_rms', 'magdev_median', 'magdev_angle_median',
                          'magdev_angle_p95', 'mag_norm_dev_rms', 'mag_norm_std']].mean()
            print(f"  {segment if regime == 'all' else '':<16}{regime:<13}"
                  f"{means['magdev_rms']:>11.3f}{means['magdev_median']:>10.3f}"
                  f"{means['magdev_angle_median']:>10.1f}{means['magdev_angle_p95']:>11.1f}"
                  f"{means['mag_norm_dev_rms']:>11.4f}{means['mag_norm_std']:>10.4f}")
        print()

    overall = sensor_df[sensor_df['regime'] == 'all'].groupby('segment')['magdev_rms'].mean()
    overall = overall.dropna().sort_values()
    if not overall.empty:
        print(f"-> CLEANEST field: {overall.index[0]} ({overall.iloc[0]:.3f} {MAG_UNIT})")
        print(f"-> WORST field   : {overall.index[-1]} ({overall.iloc[-1]:.3f} {MAG_UNIT})")
    print(f"\nThe angle columns are the ones an orientation filter pays for. A magnetometer is "
          f"used as a\nDIRECTION reference, so an error in |m| costs nothing while an error in "
          f"its direction is a\nheading error of the same size. Note how little the static and "
          f"moving rows differ here\ncompared to the accelerometer: magnetic distortion is a "
          f"property of WHERE the sensor is, not\nof whether it is moving, so unlike the "
          f"gravity assumption it cannot be recovered by waiting.\nThat asymmetry is the reason "
          f"the two assumptions need different treatment.")


def report_noise_floor(spec: DatasetSpec, sensor_df: pd.DataFrame) -> None:
    _header(5, "INTRINSIC SENSOR NOISE FLOOR DURING STATIC PERIODS",
            "per-axis std within each static interval, length-weighted across intervals and "
            "trials")
    if sensor_df.empty or 'n_noise_samples' not in sensor_df:
        print("No sensor stats found.")
        return
    units = {'gyro': 'rad/s', 'acc': 'm/s^2', 'mag': MAG_UNIT}
    for regime, blurb in (('body_static', 'whole body still — the cleanest estimate'),
                          ('static', 'this sensor still, the rest of the body free to move')):
        rows = sensor_df[(sensor_df['regime'] == regime) & (sensor_df['n_noise_samples'] > 0)]
        if rows.empty:
            print(f"\n{regime}: no usable intervals.")
            continue
        total = int(rows['n_noise_samples'].sum())
        print(f"\n{regime} ({blurb}) — {len(rows)} sensor-trials, {total:,} samples:")
        for modality, unit in units.items():
            pooled = _weighted_noise(rows, modality)
            if pooled is None or np.isnan(pooled).all():
                continue
            print(f"  {modality.capitalize():<5} ({unit:<6}): mean {np.nanmean(pooled):.6f} | "
                  f"[x={pooled[0]:.5f}, y={pooled[1]:.5f}, z={pooled[2]:.5f}]")

    print(f"\n  {'segment':<16}{'gyro (rad/s)':>15}{'acc (m/s^2)':>15}{f'mag ({MAG_UNIT})':>15}"
          f"{'samples':>12}")
    body = sensor_df[(sensor_df['regime'] == 'body_static') & (sensor_df['n_noise_samples'] > 0)]
    for segment in [s for s in spec.segment_sensor if s in set(body['segment'])]:
        rows = body[body['segment'] == segment]
        cells = []
        for modality in ('gyro', 'acc', 'mag'):
            pooled = _weighted_noise(rows, modality)
            cells.append(np.nanmean(pooled) if pooled is not None else np.nan)
        print(f"  {segment:<16}{cells[0]:>15.6f}{cells[1]:>15.5f}{cells[2]:>15.5f}"
              f"{int(rows['n_noise_samples'].sum()):>12,}")

    constants = pipeline_constants()
    print(f"\nThe filter is tuned with gyro_std={constants['gyro_std']}, "
          f"acc_std={constants['acc_std']}, mag_std={constants['mag_std']}.")
    body_rows = sensor_df[(sensor_df['regime'] == 'body_static') & (sensor_df['n_noise_samples'] > 0)]
    static_rows = sensor_df[(sensor_df['regime'] == 'static') & (sensor_df['n_noise_samples'] > 0)]
    for modality in ('gyro', 'acc', 'mag'):
        clean = _weighted_noise(body_rows, modality)
        loose = _weighted_noise(static_rows, modality)
        if clean is None or loose is None:
            continue
        ratio = np.nanmean(loose) / np.nanmean(clean)
        print(f"  {modality:<5}: per-sensor-static is {ratio:.1f}x the whole-body-static floor.")
    print("\nThat ratio is not an error bar — it is signal. A sensor can be rotationally still "
          "while the\nbody around it is not: a foot in stance carries footfall transients from "
          "the other leg and\nthe ground, and a seated thigh carries postural sway. The "
          "whole-body number is the sensor's\nown floor; the per-sensor one is what a filter "
          "would actually see during a 'quiet' window it\ndetected for itself.")


def report_observability(spec: DatasetSpec, joint_df: pd.DataFrame) -> None:
    _header(6, "JOINT OBSERVABILITY o^J, OVERALL AND BY REGIME",
            "min(parent, child) at the joint center — the quantity mag_adapt gates on")
    if joint_df.empty:
        print("No joint stats found.")
        return
    gate = joint_df['gate_threshold'].iloc[0]
    print(f"  {'joint':<10}{'regime':<13}{'median':>11}{'p05':>10}{'p95':>10}"
          f"{'parent':>11}{'child':>11}{'% < gate':>10}")
    for joint in [j for j in spec.joints if j in set(joint_df['joint'])]:
        for regime in REGIMES:
            rows = joint_df[(joint_df['joint'] == joint) & (joint_df['regime'] == regime)]
            if rows.empty:
                continue
            means = rows[['obs_min_median', 'obs_min_p05', 'obs_min_p95', 'obs_parent_median',
                          'obs_child_median', 'frac_below_gate']].mean()
            print(f"  {joint if regime == 'all' else '':<10}{regime:<13}"
                  f"{means['obs_min_median']:>11.1f}{means['obs_min_p05']:>10.1f}"
                  f"{means['obs_min_p95']:>10.1f}{means['obs_parent_median']:>11.1f}"
                  f"{means['obs_child_median']:>11.1f}{100 * means['frac_below_gate']:>9.1f}%")
        print()

    overall = joint_df[joint_df['regime'] == 'all'].groupby('joint')['obs_min_median'].mean()
    limiting = joint_df[joint_df['regime'] == 'all'].groupby('joint')[
        ['obs_parent_median', 'obs_child_median']].mean()
    print(f"o^J is gated at {gate:g} by default (experiment_utils.DEFAULT_MAG_ADAPT_THRESHOLD), so "
          f"the last\ncolumn is the fraction of each regime the magnetometer would be distrusted "
          f"over. Note the\nstatic rows: a joint that is not moving has almost no observability, "
          f"which is correct — a\nstationary accelerometer carries no information about "
          f"orientation beyond gravity — and it is\nalso when the magnetometer is at its most "
          f"trustworthy, since the field it reads is not being\nswept through a distorted region. "
          f"A gate on o^J alone therefore distrusts the magnetometer\nprecisely when it is safest "
          f"to use.")
    if not overall.empty:
        print(f"\n-> LEAST observable joint: {overall.idxmin()} (median o^J {overall.min():.1f})")
        print(f"-> MOST observable joint : {overall.idxmax()} (median o^J {overall.max():.1f})")
    for joint in limiting.index:
        parent, child = limiting.loc[joint, 'obs_parent_median'], limiting.loc[joint, 'obs_child_median']
        if min(parent, child) <= 0:
            continue
        side = 'parent' if parent < child else 'child'
        print(f"   {joint:<9} limited by its {side} sensor ({min(parent, child):.0f} vs "
              f"{max(parent, child):.0f})")


def report_field_consistency(spec: DatasetSpec, joint_df: pd.DataFrame) -> None:
    _header(7, "LOCAL FIELD CONSISTENCY: NEIGHBOUR SENSOR vs. GLOBAL FIELD",
            "world-frame fields over the 'all' regime; one value per trial, summarized across "
            "trials")
    rows_all = joint_df[joint_df['regime'] == 'all'] if not joint_df.empty else joint_df
    if rows_all.empty or 'var_reduction' not in rows_all.columns:
        print("No joint field-consistency stats found.")
        return
    print(f"{'Joint':<10}{'Reference':>14}{'CosSim tgt-global':>19}{'CosSim tgt-ref':>16}"
          f"{'Var reduction':>15}{'[min, max]':>18}{'n<0':>5}{'best':>8}{'n':>5}")
    flipped = []
    for joint in [j for j in spec.joints if j in set(rows_all['joint'])]:
        rows = rows_all[rows_all['joint'] == joint]
        reduction = rows['var_reduction'].dropna()
        if reduction.empty:
            continue
        role = rows['reference_role'].iloc[0]
        reference = str(rows['reference_sensor'].iloc[0]).replace('_imu', '')
        if role != 'parent':
            flipped.append((joint, reference, str(rows['target_sensor'].iloc[0]).replace('_imu', '')))
        print(f"{joint:<10}{reference + ('*' if role != 'parent' else ''):>14}"
              f"{rows['cos_sim_target_global'].median():>19.3f}"
              f"{rows['cos_sim_target_reference'].median():>16.3f}"
              f"{reduction.median():>15.3f}"
              f"{f'[{reduction.min():.2f}, {reduction.max():.2f}]':>18}"
              f"{int((reduction < 0).sum()):>5}{rows['var_reduction_best'].median():>8.3f}"
              f"{len(rows):>5}")

    print("\nWhat this compares: for a filter that needs the field at a sensor, is it better to "
          "assume ONE\nCONSTANT field everywhere on the body, or to take the field from the "
          "sensor on the ADJACENT\nsegment? The constant scores exactly 0 by construction — "
          "Var(target - global) = Var(target) for\nany constant global — so the neighbour wins "
          "wherever that column is positive.\n"
          "  Var reduction  what substituting the neighbour's field verbatim achieves. Equals\n"
          "                 2*rho*sqrt(k) - k for k = Var(reference)/Var(target), so its SIGN\n"
          "                 tracks the two sensors' variance ratio, not how informative the\n"
          "                 neighbour is — read `field_consistency` before quoting one.\n"
          "  best           rho^2, the ceiling if the neighbour's field were optimally "
          "rescaled.\n                 The shortfall against it is exactly (rho - sqrt(k))^2, and "
          "optimal_gain in\n                 the saved table says which way to rescale. Symmetric "
          "in the pair, so\n                 unlike Var reduction it is unaffected by the "
          "reference direction.")
    for joint, reference, target in flipped:
        print(f"* {joint}: reference reversed to {reference} (target {target}) by the dataset's "
              f"field_reference —\n  {reference} is the cleaner sensor of the pair, where for "
              f"every other joint the kinematic parent\n  already is. Both directions are in the "
              f"saved table.")


def report_residuals(spec: DatasetSpec, joint_df: pd.DataFrame) -> None:
    _header(8, "WHAT THE RELATIVE FILTER ACTUALLY HAS TO ABSORB",
            "the between-sensor residuals RelativeFilter.get_h drives to zero, by regime")
    if joint_df.empty or 'acc_residual_rms' not in joint_df.columns:
        print("No joint residual stats found.")
        return
    constants = pipeline_constants()
    print(f"  {'joint':<10}{'regime':<13}{'acc resid':>12}{'unprojected':>13}{'vs acc_std':>12}"
          f"{'mag resid':>12}{'mag angle':>11}{'vs mag_std':>12}")
    print(f"  {'':<10}{'':<13}{'(m/s^2)':>12}{'(m/s^2)':>13}{'':>12}{f'({MAG_UNIT})':>12}"
          f"{'(deg)':>11}{'':>12}")
    for joint in [j for j in spec.joints if j in set(joint_df['joint'])]:
        for regime in ('all', 'static', 'nonstatic'):
            rows = joint_df[(joint_df['joint'] == joint) & (joint_df['regime'] == regime)]
            if rows.empty:
                continue
            means = rows[['acc_residual_rms', 'acc_residual_rms_raw', 'mag_residual_rms',
                          'mag_residual_angle']].mean()
            acc_axis = means['acc_residual_rms'] / np.sqrt(3)
            mag_axis = means['mag_residual_rms'] / np.sqrt(3)
            print(f"  {joint if regime == 'all' else '':<10}{regime:<13}"
                  f"{means['acc_residual_rms']:>12.3f}{means['acc_residual_rms_raw']:>13.3f}"
                  f"{acc_axis / constants['acc_std']:>11.0f}x{means['mag_residual_rms']:>12.4f}"
                  f"{means['mag_residual_angle']:>11.2f}"
                  f"{mag_axis / constants['mag_std']:>11.1f}x")
        print()
    print("These are the quantities to tune acc_std and mag_std from — not the per-sensor "
          "departures in\nsections 3 and 4. The relative filter never compares one sensor "
          "against a world reference; it\ncompares the two sensors of a pair against each other, "
          "and whatever error they share cancels\nin the difference. The 'unprojected' column is "
          "the same residual without the joint-center\nprojection, so the pair says whether "
          "projecting reduces the disagreement the filter absorbs.")


def report_subject_spread(spec: DatasetSpec, summary: pd.DataFrame,
                          fields_df: pd.DataFrame) -> None:
    _header(9, "PER-SUBJECT SPREAD",
            "median of each metric per subject, pooled over that subject's trials and sensors")
    if summary.empty:
        print("No summary found.")
        return
    per_subject = summary[(summary['trial'] == 'all') & (summary['subject'] != 'all')
                          & (summary['regime'] == 'all')]
    if per_subject.empty:
        print("No per-subject rows in the summary.")
        return
    pivot = (per_subject.groupby(['subject', 'metric'])
             .apply(lambda g: np.average(g['p50'], weights=g['n_samples'].clip(lower=1)),
                    include_groups=False)
             .unstack())
    metrics = [m for m in SEGMENT_METRICS + JOINT_METRICS if m in pivot.columns]
    print("  " + f"{'subject':<12}" + "".join(f"{m:>16}" for m in metrics)
          + f"{'|field|':>10}")
    field_by_subject = (fields_df.set_index('subject')['world_mag_norm'].to_dict()
                        if not fields_df.empty else {})
    for subject in pivot.index:
        cells = "".join(f"{pivot.loc[subject, m]:>16.3g}" for m in metrics)
        norm = field_by_subject.get(subject, np.nan)
        print(f"  {subject:<12}{cells}{norm:>10.3f}")
    print(f"\n  {'spread':<12}" + "".join(
        f"{pivot[m].max() / max(pivot[m].min(), 1e-12):>15.1f}x" for m in metrics))
    if not fields_df.empty:
        vectors = fields_df[['world_mag_x', 'world_mag_y', 'world_mag_z']].to_numpy()
        mean, std = vectors.mean(axis=0), vectors.std(axis=0)
        print(f"\nGlobal field across {len(fields_df)} subjects, all in {MAG_UNIT}:")
        print(f"  Mean            : [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
        print(f"  Std across subj : [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
        print(f"  Std of magnitude: {np.std(np.linalg.norm(vectors, axis=1)):.4f}")
        print("\nA field that varies this much BETWEEN subjects, measured in the same room, is "
              "already an\nargument against a single hard-coded world field: the reference has to "
              "be estimated per\nsession whatever else a filter does.")


def report_trial_spread(spec: DatasetSpec, summary: pd.DataFrame, top: int = 12) -> None:
    _header(10, "PER-TRIAL EXTREMES",
            f"the {top} trials where each assumption fails hardest, pooled over their sensors")
    per_trial = summary[(summary['subject'] != 'all') & (summary['trial'] != 'all')
                        & (summary['regime'] == 'all')]
    if per_trial.empty:
        print("No per-trial rows in the summary.")
        return
    for metric in ('linacc', 'magdev'):
        rows = per_trial[per_trial['metric'] == metric]
        if rows.empty:
            continue
        ranked = (rows.groupby(['subject', 'trial'])
                  .apply(lambda g: np.average(g['p50'], weights=g['n_samples'].clip(lower=1)),
                         include_groups=False)
                  .sort_values(ascending=False))
        unit = METRIC_UNITS[metric]
        print(f"\n{metric} ({unit}), median over the trial's sensors "
              f"({len(ranked)} trials):")
        for (subject, trial), value in ranked.head(top).items():
            print(f"  {subject:<10}{trial:<28}{value:>10.3f}")
        # The quietest few only when they are not already listed above; with a handful of
        # trials the head covers everything and a tail would print them twice.
        if len(ranked) > top + 3:
            print(f"  {'...':<10}{'':<28}")
            for (subject, trial), value in ranked.tail(3).items():
                print(f"  {subject:<10}{trial:<28}{value:>10.3f}")


def print_report(spec: DatasetSpec, summary: pd.DataFrame, sensor_df: pd.DataFrame,
                 joint_df: pd.DataFrame, fields_df: pd.DataFrame) -> None:
    report_headline(spec, summary, sensor_df)
    report_static_coverage(spec, sensor_df)
    report_acceleration(spec, sensor_df)
    report_magnetometer(spec, sensor_df)
    report_noise_floor(spec, sensor_df)
    report_observability(spec, joint_df)
    report_field_consistency(spec, joint_df)
    report_residuals(spec, joint_df)
    report_subject_spread(spec, summary, fields_df)
    report_trial_spread(spec, summary)

# ==============================================================================
# CLI / ORCHESTRATOR
# ==============================================================================

def select_trials(dataset: str, subjects: Optional[List[str]],
                  trials: Optional[List[str]]) -> List[Tuple[str, str]]:
    """The built trials matching the filters, or a ValueError naming what does not exist.

    Filters are checked against what is on disk rather than against a constant, so a typo is
    caught against reality and a subject with only one of two activities needs no special case.
    """
    row_keys = enumerate_trials(dataset)
    if not row_keys:
        raise ValueError(f"No built trials under {paths.TRIALS_DIR / dataset}. "
                         f"Run: python -m experiments.build_trials --dataset {dataset}")
    if subjects:
        unknown = sorted(set(subjects) - {s for s, _ in row_keys})
        if unknown:
            raise ValueError(f"No such subject(s) built in {dataset}: {unknown}")
        row_keys = [(s, t) for s, t in row_keys if s in subjects]
    if trials:
        unknown = sorted(set(trials) - {t for _, t in row_keys})
        if unknown:
            raise ValueError(f"No such trial(s) built in {dataset}: {unknown}")
        row_keys = [(s, t) for s, t in row_keys if t in trials]
    return row_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default=ALBORNO.name, choices=sorted(DATASETS),
                        help="Which built dataset to analyse.")
    parser.add_argument('--subjects', nargs='+', default=None,
                        help="Restrict to these subject/session ids (default: everything built).")
    parser.add_argument('--trials', nargs='+', default=None,
                        help="Restrict to these trial names (default: everything built).")
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--stride', type=int, default=SAMPLE_STRIDE,
                        help="Keep every Nth sample in the PER-SAMPLE tables. Scalars in "
                             "sensor_stats and joint_stats are always computed on the full "
                             "record, so this trades figure resolution and pooled-quantile "
                             "precision against disk, and nothing else.")
    parser.add_argument('--only-tables', nargs='+', choices=RECOMPUTABLE_TABLES,
                        default=list(RECOMPUTABLE_TABLES), metavar='TABLE',
                        help="Recompute only these per-trial tables, leaving the others on disk "
                             f"untouched (one or more of: {', '.join(RECOMPUTABLE_TABLES)}). "
                             "Loading the trials is unavoidable, but the two joint tables carry "
                             "the rigid-body projection and dominate the runtime.")
    parser.add_argument('--skip-fields', action='store_true',
                        help="Reuse the subject global fields already on disk instead of "
                             "recomputing stage one. Safe only when the trial set has not "
                             "changed, since a subject's field is pooled over all of their "
                             "trials.")
    parser.add_argument('--report-only', action='store_true',
                        help="Skip recomputation and rebuild the summary + report from the "
                             "per-trial tables already on disk. The pooled summary always covers "
                             "every trial present on disk, not just those passed to --subjects, "
                             "so a partial run does not silently narrow it.")
    args = parser.parse_args()

    try:
        row_keys = select_trials(args.dataset, args.subjects, args.trials)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    spec = get_dataset(args.dataset)

    if not args.report_only:
        if not args.skip_fields:
            print(f"Stage 1/2: per-trial magnetic field over {len(row_keys)} trials...")
            run_tracked_grid(row_keys, ['Subject', 'Trial'], ['field'],
                             partial(_field_worker, dataset=args.dataset),
                             args.workers, title=f"GLOBAL ASSUMPTIONS — FIELD ({args.dataset})")
            written = aggregate_subject_fields(args.dataset, row_keys)
            print(f"Wrote a global field for {len(written)} subject(s).")

        if set(args.only_tables) != set(RECOMPUTABLE_TABLES):
            print(f"Recomputing only {', '.join(args.only_tables)}; every other per-trial table "
                  f"is left as it is on disk.")
        print(f"Stage 2/2: per-trial tables over {len(row_keys)} trials...")
        run_tracked_grid(row_keys, ['Subject', 'Trial'], ['tables'],
                         partial(_trial_worker, dataset=args.dataset, tables=args.only_tables,
                                 stride=args.stride),
                         args.workers, title=f"GLOBAL ASSUMPTIONS — TABLES ({args.dataset})")

    print("\nLoading per-trial tables...")
    # Only the columns the summary groups by or aggregates. The sample tables run to tens of
    # millions of rows across both datasets, and reading the timestamps and raw norms here
    # would cost hundreds of megabytes to throw away.
    segment_df = load_trial_table(args.dataset, 'segment_samples',
                                  columns=['segment', 'static', 'body_static', *SEGMENT_METRICS])
    joint_sample_df = load_trial_table(args.dataset, 'joint_samples',
                                       columns=['joint', 'static', 'body_static', *JOINT_METRICS])
    sensor_df = load_trial_table(args.dataset, 'sensor_stats')
    joint_stat_df = load_trial_table(args.dataset, 'joint_stats')
    fields_df = load_subject_fields(args.dataset)

    found = set()
    for df in (segment_df, joint_sample_df, sensor_df):
        if not df.empty:
            found |= {(str(s), str(t)) for s, t
                      in df[['subject', 'trial']].drop_duplicates().to_numpy()}
    if not found:
        print(f"No results found under {dataset_dir(args.dataset)}. Run without --report-only "
              f"first.")
        return 1
    print(f"Found {len(found)} trial(s) across {len({s for s, _ in found})} subject(s).")

    summary = summarize(args.dataset, segment_df, joint_sample_df)
    del segment_df, joint_sample_df
    if not summary.empty:
        path = paths.ensure_parent(statistics_path(args.dataset))
        summary.to_parquet(path, engine='pyarrow', index=False)
        paths.write_manifest(path, constants=analysis_constants(args.dataset),
                             experiment=EXPERIMENT_NAME, n_rows=len(summary),
                             subjects=sorted({s for s, _ in found}),
                             trials=sorted({t for _, t in found}))
        print(f"Saved summary to {path}")

    print_report(spec, summary, sensor_df, joint_stat_df, fields_df)
    print(f"\nPer-trial tables under {dataset_dir(args.dataset)}")
    print(f"Figures: python -m plotting.global_assumptions --dataset {args.dataset}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
