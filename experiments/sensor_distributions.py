"""
Sensor characterization for this dataset: what each body-worn IMU actually measures,
and how observable each joint is, across every subject and activity.

This is the computation behind plotting/sensor_distributions.py. It answers the questions
the sensor-model argument needs, all of which are properties of the DATA rather than of
any filter, so none of it requires running the EKF:

  1. Magnetic distortion by segment - how far the field measured on each segment deviates
     from that subject's global field, and how variable each sensor's field magnitude is.
  2. Linear acceleration by segment - what the accelerometers see once gravity is removed.
  3. Joint observability - o^J (segment_observability, projected to the joint center) in
     practice, per joint and per segment. This is the quantity mag_adapt gates on.
  4. Intrinsic sensor noise - the per-axis noise floor, measured on ground-anchored feet
     during genuinely stationary periods.
  5. Local field consistency - whether one sensor's field is better predicted by its NEIGHBOUR
     across the joint than by the global field, which is the premise behind estimating the field
     at the joint center. Which of the pair references which is set by FIELD_REFERENCE; read
     joint_mag_consistency before quoting var_reduction, since its sign turns on the two
     sensors' variance ratio rather than on how coupled they are.

Supersedes the root-level generate_sensor_stats.py. Items 4 and 5 and the console report
are that script's, and (1)-(3) are the sample-level distributions that used to live in
scratch/. Three things changed deliberately in the move, all of which shift printed
numbers relative to the old script:

  * Observability comes from experiment_utils.segment_observability, on traces projected
    to the joint center, instead of a local copy of the formula. The local copy was
    missing the /dt in its finite difference (see that docstring), so it was a different
    metric that happened to have the same name.
  * Sitting is detected from pelvis LINEAR ACCELERATION rather than pelvis gyro (see
    label_activity_intervals). Gyro-based detection fragmented single seated bouts,
    because postural sway while seated produces more angular than translational noise.
  * A subject's "global" field is the median world-frame field over all of that subject's
    sensors AND both activities (expected_mag_field), not one trial or one sensor. The
    contrast with experiment_utils._compute_expected_mag_field, which is torso-only, is
    deliberate: that one defines the EKF's mag oracle, where a single reference sensor is
    the point; here the reference has to be neutral between segments, since comparing
    segments to each other is the whole measurement.

Outputs, all under results/experiments/sensor_distributions/:

    Subject<NN>/expected_mag_field.parquet     subject-level global field (one row)
    Subject<NN>/<activity>/segment_samples.parquet   per-sample, per-segment linacc/magdev
    Subject<NN>/<activity>/joint_samples.parquet     per-sample, per-joint o^J (parent/child)
    Subject<NN>/<activity>/foot_samples.parquet      per-sample foot signal norms, RAW time base
    Subject<NN>/<activity>/intervals.parquet         labeled sitting/standing/ambulation/stationary
    Subject<NN>/<activity>/sensor_stats.parquet      per-sensor scalars (distortion, noise floor)
    Subject<NN>/<activity>/joint_stats.parquet       per-joint field-consistency scalars

plus the pooled quantile summary at
results/statistics/sensor_distributions_statistics.parquet, which is what the console
report and the paper text quote.

The per-sample tables are what make the figures cheap to re-tune: plotting reads these back
instead of reloading and re-projecting every trial (19 of them in this dataset — three
subjects have no complexTasks recording).
"""
import argparse
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import paths
from experiments.experiment_utils import (ACTIVITIES, DEFAULT_ACC_STD, EXPECTED_GRAVITY,
                                          JOINTS, SUBJECTS,
                                          load_raw_data, pipeline_constants,
                                          project_pair_to_joint_center, run_tracked_grid,
                                          segment_observability)
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial

EXPERIMENT_NAME = "sensor_distributions"
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Display name -> sensor, ordered proximal to distal down each limb. The order is what
# every figure sorts by, and the whole point of most of these plots is that distortion,
# acceleration and observability all vary systematically WITH that ordering, so it is
# defined once here rather than per figure.
SEGMENT_SENSOR = {
    'Torso': 'torso_imu',
    'Pelvis': 'pelvis_imu',
    'Femur R': 'femur_r_imu',
    'Tibia R': 'tibia_r_imu',
    'Calcn R': 'calcn_r_imu',
    'Femur L': 'femur_l_imu',
    'Tibia L': 'tibia_l_imu',
    'Calcn L': 'calcn_l_imu',
}
SEGMENT_ORDER = list(SEGMENT_SENSOR)
SENSOR_SEGMENT = {sensor: segment for segment, sensor in SEGMENT_SENSOR.items()}

# The up axis of the mocap world frame, taken from the gravity vector rather than written
# down again: pelvis height and the world-frame gravity convention have to agree, and
# hard-coding 1 here would let them drift apart silently if EXPECTED_GRAVITY ever changed.
HEIGHT_AXIS = int(np.argmax(np.abs(EXPECTED_GRAVITY)))

# --- Observability smoothing (visualization / distribution summary only) -------------
# o^J is a norm of a cross product of a finite difference: non-negative, and full of
# sharp impulsive spikes at every footfall. The spikes are real, but they dominate a KDE
# and a box plot to the point where the body of the distribution is invisible, so the
# per-sample series stored here is low-pass filtered. The filter is applied HERE rather
# than in the plotting layer so that the figures and the quoted quantiles describe the
# same numbers, and so the cutoff lands in the provenance manifest.
OBS_FILTER_CUTOFF_HZ = 25.0
OBS_FILTER_ORDER = 4
OBS_WINSORIZE_PCT = 99.5

# --- Sitting / standing / ambulation labeling ----------------------------------------
QUIET_LINACC_THRESHOLD = 0.5  # m/s^2, rolling std of pelvis linear acceleration
QUIET_WINDOW_S = 1.0
SIT_HEIGHT_DROP_M = 0.25  # pelvis drop below its trial-typical upright height that counts as sitting
MIN_SITTING_S = 10.0  # sitting bouts run long relative to the rest of the task cycle
MIN_STANDING_S = 2.0  # standing bouts are brief in this protocol

# --- Ground-anchored stationary detection (intrinsic noise floor) ---------------------
STATIONARY_GYRO_STD_THRESHOLD = 0.05  # rad/s, rolling std of foot gyro norm
STATIONARY_WINDOW_S = 1.0
MIN_STATIONARY_S = 20.0  # long enough that a per-axis std is a noise estimate, not a sample of motion
STATIONARY_MARGIN_S = 3.0  # trimmed off each end, so detector edge effects are not measured as noise
FOOT_SENSORS = ('calcn_l_imu', 'calcn_r_imu')

# Which sensor of each pair supplies the LOCAL FIELD REFERENCE in joint_mag_consistency, i.e.
# which one plays the role the global field is being compared against. Default is the kinematic
# parent, which for every limb joint is also the more proximal and magnetically cleaner sensor.
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
# the torso is the noisier of the pair (mag_norm_std ratio torso/pelvis 1.72, against 0.48-0.98 in
# the other 18 trials), and that is the one trial where the flipped lumbar comes out negative
# (-1.13, where the unflipped direction gives +0.17). The rule is not adjusted for it — a
# per-trial direction would be the cherry-pick this constant exists to avoid — but a lumbar number
# quoted for that trial should carry the caveat.
FIELD_REFERENCE = {'Lumbar': 'child'}  # joint -> 'parent' (default) or 'child'

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# These files are Xsens exports, whose magnetometer channels are NOT in microtesla: they are
# normalized at calibration so that a nominal local Earth field reads 1.0 (measured |mag|
# medians across this dataset's sensors run 0.4-1.1, and the filter's own mag_std default of
# 0.03 is on that same scale). Every magnetic quantity here is therefore in those arbitrary
# units and the labels say so — the script this replaced printed "uT" throughout, which
# overstated every magnetic number by roughly the local field strength. Multiply by ~50 uT
# for a physical scale.
MAG_UNIT = 'a.u.'
METRIC_UNITS = {'linacc': 'm/s^2', 'magdev': MAG_UNIT, 'obs_min': '(m/s^2)(m/s^3)'}

TRIAL_TABLES = ('segment_samples', 'joint_samples', 'foot_samples',
                'intervals', 'sensor_stats', 'joint_stats')


def segment_joint_roles() -> Dict[str, List[Tuple[str, str]]]:
    """{segment: [(joint, 'parent'|'child'), ...]} - which joints border each segment, and
    on which side of each.

    Derived from JOINTS rather than written out, because a hand-maintained copy is a table
    that can silently disagree with the joint definitions the rest of the pipeline uses.
    Every segment borders one or two joints; the pelvis borders three (lumbar + both hips).
    """
    roles: Dict[str, List[Tuple[str, str]]] = {segment: [] for segment in SEGMENT_ORDER}
    for joint, (parent_sensor, child_sensor) in JOINTS.items():
        for sensor, role in ((parent_sensor, 'parent'), (child_sensor, 'child')):
            if sensor in SENSOR_SEGMENT:
                roles[SENSOR_SEGMENT[sensor]].append((joint, role))
    return roles


def analysis_constants() -> Dict[str, object]:
    """Pipeline constants plus this analysis's own thresholds, for provenance manifests."""
    return {
        **pipeline_constants(),
        'height_axis': HEIGHT_AXIS,
        'obs_filter_cutoff_hz': OBS_FILTER_CUTOFF_HZ,
        'obs_filter_order': OBS_FILTER_ORDER,
        'obs_winsorize_pct': OBS_WINSORIZE_PCT,
        'quiet_linacc_threshold': QUIET_LINACC_THRESHOLD,
        'sit_height_drop_m': SIT_HEIGHT_DROP_M,
        'min_sitting_s': MIN_SITTING_S,
        'min_standing_s': MIN_STANDING_S,
        'stationary_gyro_std_threshold': STATIONARY_GYRO_STD_THRESHOLD,
        'min_stationary_s': MIN_STATIONARY_S,
        'stationary_margin_s': STATIONARY_MARGIN_S,
    }

# ==============================================================================
# Paths / IO
# ==============================================================================

def trial_table_path(subject: str, activity: str, table: str) -> Path:
    return EXPERIMENT_DIR / f"Subject{subject}" / activity / f"{table}.parquet"


def expected_mag_field_path(subject: str) -> Path:
    return EXPERIMENT_DIR / f"Subject{subject}" / "expected_mag_field.parquet"


def _save(df: pd.DataFrame, path: Path, **manifest_extra) -> None:
    df.to_parquet(paths.ensure_parent(path), engine='pyarrow', index=False)
    paths.write_manifest(path, constants=analysis_constants(), experiment=EXPERIMENT_NAME,
                         n_rows=len(df), **manifest_extra)


def load_trial_table(table: str, subjects: Optional[List[str]] = None,
                     activities: Optional[List[str]] = None) -> pd.DataFrame:
    """Concatenates one per-trial table across subjects/activities, adding `subject` and
    `activity` columns. Missing trials are skipped silently - a partial run
    (`--subjects 06`) is a legitimate state, and the caller reports what it found. An unknown
    table name raises instead of returning an empty frame, which would be indistinguishable
    from "the experiment has not been run yet"."""
    if table not in TRIAL_TABLES:
        raise ValueError(f"Unknown table '{table}'; expected one of {TRIAL_TABLES}")
    subjects = SUBJECTS if subjects is None else subjects
    activities = ACTIVITIES if activities is None else activities
    frames = []
    for subject in subjects:
        for activity in activities:
            path = trial_table_path(subject, activity, table)
            if not path.exists():
                continue
            frames.append(pd.read_parquet(path, engine='pyarrow')
                          .assign(subject=f"Subject{subject}", activity=activity))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_expected_mag_fields(subjects: Optional[List[str]] = None) -> pd.DataFrame:
    subjects = SUBJECTS if subjects is None else subjects
    frames = [pd.read_parquet(expected_mag_field_path(s), engine='pyarrow')
              for s in subjects if expected_mag_field_path(s).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ==============================================================================
# Per-sample quantities
# ==============================================================================

def expected_mag_field(plates_by_activity: Dict[str, Dict[str, PlateTrial]]) -> np.ndarray:
    """One subject's global magnetic field: the median world-frame field over every sensor
    and every sample the subject has, across both activities.

    Median, not mean, because local ferrous distortion is one-sided and heavy-tailed - a
    mean would be pulled toward whichever segment sat closest to metal. Pooled over all
    sensors so that no segment is the privileged reference when segments are then compared
    against this field (see the module docstring on _compute_expected_mag_field)."""
    world_mags = [plate.get_imu_trace_in_global_frame().mag
                  for plates in plates_by_activity.values() for plate in plates.values()]
    return np.median(np.concatenate(world_mags, axis=0), axis=0)


def smooth_observability(observability: np.ndarray, fs: float,
                         cutoff: float = OBS_FILTER_CUTOFF_HZ, order: int = OBS_FILTER_ORDER,
                         winsorize_pct: float = OBS_WINSORIZE_PCT) -> np.ndarray:
    """Zero-lag low-pass filter for o^J (see OBS_FILTER_CUTOFF_HZ for why it is filtered).

    Two guards, both because o^J is a non-negative spiky signal rather than a smooth one:
    the top percentile is capped BEFORE filtering, since feeding impulses straight into
    filtfilt makes it ring and overshoot negative just after each spike, and the output is
    clamped at 0 afterwards so nothing downstream sees a negative observability."""
    capped = np.minimum(observability, np.percentile(observability, winsorize_pct))
    b, a = butter(order, cutoff / (fs / 2.0), btype='low')
    return np.maximum(filtfilt(b, a, capped), 0.0)


def joint_center_observability(parent_plate: PlateTrial, child_plate: PlateTrial
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """(parent, child) per-sample observability at the shared joint center.

    Both plates are projected to the joint center first, exactly as the filter does before
    gating on o^J, so these are the same numbers mag_adapt thresholds. Returned as a pair
    rather than reduced with min(): a joint's own observability is the minimum of the two
    (_calculate_observability_metric_), but the per-SEGMENT figures need to know which of
    the two sensors was the limiting one."""
    parent_proj, child_proj = project_pair_to_joint_center(parent_plate, child_plate)
    return (segment_observability(parent_proj.imu_trace),
            segment_observability(child_proj.imu_trace))


def segment_samples(plates: Dict[str, PlateTrial], expected_mag: np.ndarray) -> pd.DataFrame:
    """Per-sample, per-segment: world-frame linear acceleration and magnetic deviation
    magnitudes, plus the raw body-frame signal norms.

    linacc removes gravity in the WORLD frame (|a_world - g|), which is what makes it
    comparable across segments: subtracting a constant in each sensor's own body frame
    would leave every segment's own orientation in the result. magdev is measured against
    the subject's global field (expected_mag_field), so it is a distortion magnitude, not
    a field magnitude. The three raw norms are kept because the interval-detection
    diagnostics plot exactly those, and storing them here is what keeps the plotting layer
    from having to reload raw data."""
    rows = []
    for segment, sensor in SEGMENT_SENSOR.items():
        if sensor not in plates:
            continue
        plate = plates[sensor]
        world = plate.get_imu_trace_in_global_frame()
        local = plate.imu_trace
        rows.append(pd.DataFrame({
            'timestamp': local.timestamps.astype(np.float64),
            'segment': segment,
            'linacc': np.linalg.norm(world.acc - EXPECTED_GRAVITY, axis=1).astype(np.float32),
            'magdev': np.linalg.norm(world.mag - expected_mag, axis=1).astype(np.float32),
            'acc_norm': np.linalg.norm(local.acc, axis=1).astype(np.float32),
            'gyro_norm': np.linalg.norm(local.gyro, axis=1).astype(np.float32),
            'mag_norm': np.linalg.norm(local.mag, axis=1).astype(np.float32),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def joint_samples(plates: Dict[str, PlateTrial], fs: float) -> pd.DataFrame:
    """Per-sample, per-joint observability at the joint center, for both of the joint's
    sensors, smoothed (see smooth_observability). obs_min is the joint's own o^J."""
    rows = []
    for joint, (parent_sensor, child_sensor) in JOINTS.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        obs_parent, obs_child = joint_center_observability(plates[parent_sensor], plates[child_sensor])
        obs_parent = smooth_observability(obs_parent, fs)
        obs_child = smooth_observability(obs_child, fs)
        rows.append(pd.DataFrame({
            'timestamp': plates[parent_sensor].imu_trace.timestamps.astype(np.float64),
            'joint': joint,
            'obs_parent': obs_parent.astype(np.float32),
            'obs_child': obs_child.astype(np.float32),
            'obs_min': np.minimum(obs_parent, obs_child).astype(np.float32),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ==============================================================================
# Interval labeling
# ==============================================================================

def mask_to_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True runs of `mask`, as half-open (start, end) sample indices."""
    diffs = np.diff(mask.astype(int), prepend=0, append=0)
    return list(zip(np.where(diffs == 1)[0], np.where(diffs == -1)[0]))


def label_activity_intervals(plates: Dict[str, PlateTrial], fs: float
                             ) -> Dict[str, List[Tuple[int, int]]]:
    """Splits a trial into 'sitting' / 'standing' / 'ambulation' intervals from the pelvis.

    Protocol: each complexTasks trial cycles through sitting, standing, stair
    ascent/descent, side-stepping, walking and running. Every ambulation task involves
    repeated pelvis impacts, so a quiet pelvis isolates sitting+standing from the rest.
    Sitting vs. standing is then split by pelvis height (world_trace.positions on the up
    axis): sitting drops the pelvis well below its trial-typical upright height, standing
    does not. A walking trial has no sitting and this returns an almost-all-ambulation
    labeling, which is the correct answer for it.

    "Quiet" is measured from pelvis LINEAR ACCELERATION (gravity removed), not gyro. The
    gyro version this replaced badly undercounted both sitting and standing, because
    postural sway and fidgeting while stationary produce more angular than translational
    noise: it fragmented a single ~19s seated bout into a ~10s "sitting" chunk plus a
    fidgety tail that came out labeled as ambulation, and it missed the standing bout
    after standing up almost entirely. Linear acceleration separates the two cases far
    more sharply - stationary stays under ~0.5 m/s^2 even with fidgeting, while every
    footstep of real ambulation is a multi-m/s^2 transient.

    Known limitation: a single quiet run is labeled from its MEAN height, so sitting and
    standing are only separated when the movement between them interrupts the run. Standing up
    from a chair does interrupt it (that transient is what the linear-acceleration threshold
    is good at catching), which is why real trials come out with interleaved sitting and
    standing bouts — but a posture change with no measurable pelvis acceleration would be
    labeled as whichever posture dominates the run.

    This is a coarser, plot-facing relative of drift_observability's
    detect_quiet_sitting_segments, which also requires low pelvis vertical velocity and
    low gyro. That one defines the segments a drift rate is fit over, so it errs toward
    purity; this one labels a whole trial for shading and stratification, so it errs
    toward covering the bout end to end.
    """
    pelvis = plates['pelvis_imu']
    height = pelvis.world_trace.positions[:, HEIGHT_AXIS]
    linacc = np.linalg.norm(pelvis.get_imu_trace_in_global_frame().acc - EXPECTED_GRAVITY, axis=1)

    window = max(int(QUIET_WINDOW_S * fs), 2)
    rolling_std = pd.Series(linacc).rolling(window=window, center=True).std().to_numpy()
    is_quiet = (rolling_std < QUIET_LINACC_THRESHOLD) & (~np.isnan(rolling_std))

    baseline_height = np.median(height)
    sitting, standing = [], []
    for start, end in mask_to_intervals(is_quiet):
        if (end - start) < int(MIN_STANDING_S * fs):
            continue
        if (baseline_height - height[start:end].mean()) > SIT_HEIGHT_DROP_M:
            if (end - start) >= int(MIN_SITTING_S * fs):
                sitting.append((start, end))
        else:
            standing.append((start, end))

    quiet_mask = np.zeros(len(height), dtype=bool)
    for start, end in sitting + standing:
        quiet_mask[start:end] = True
    return {'sitting': sitting, 'standing': standing, 'ambulation': mask_to_intervals(~quiet_mask)}


def find_foot_stationary_intervals(raw_imus: Dict[str, IMUTrace]) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Intervals where BOTH feet are completely stationary, for the intrinsic noise floor.

    Both feet, because one foot can sit still while the subject shifts weight on the other
    - and a sensor that is merely slowly moving reads accelerations far above its own noise
    floor, which would silently inflate the estimate. Each surviving interval is trimmed by
    STATIONARY_MARGIN_S at both ends so that the rolling-std detector's own edges (where
    the window straddles motion) are not measured as noise.

    Operates on RAW, untrimmed IMUTraces, not PlateTrials: PlateTrial.from_folder syncs to
    the mocap .trc and trims to the overlap, and the long anchored-foot pauses this needs
    usually sit before or after the captured mocap window. Returns ([], []) when the trial
    has no long-enough pause, which is a normal outcome for walking trials.
    """
    foot_sensors = [s for s in FOOT_SENSORS if s in raw_imus]
    if not foot_sensors:
        return [], []

    fs = raw_imus[foot_sensors[0]].get_sample_frequency()
    window = max(int(STATIONARY_WINDOW_S * fs), 2)
    n = min(len(raw_imus[s]) for s in foot_sensors)

    is_stationary = np.ones(n, dtype=bool)
    for sensor in foot_sensors:
        gyro_norm = np.linalg.norm(raw_imus[sensor].gyro[:n], axis=1)
        rolling_std = pd.Series(gyro_norm).rolling(window=window, center=True).std().to_numpy()
        is_stationary &= (rolling_std < STATIONARY_GYRO_STD_THRESHOLD) & (~np.isnan(rolling_std))

    min_samples = int(MIN_STATIONARY_S * fs)
    margin = int(STATIONARY_MARGIN_S * fs)
    intervals = [(start + margin, end - margin)
                 for start, end in mask_to_intervals(is_stationary)
                 if (end - start) >= min_samples]
    return intervals, foot_sensors


def intervals_table(activity_intervals: Dict[str, List[Tuple[int, int]]], timestamps: np.ndarray,
                    stationary_intervals: List[Tuple[int, int]],
                    raw_timestamps: Optional[np.ndarray]) -> pd.DataFrame:
    """All labeled intervals for one trial, in one table.

    `time_base` distinguishes the two clocks in play: 'synced' indices/times refer to the
    mocap-synchronized PlateTrial (segment_samples, joint_samples), 'raw' ones to the
    untrimmed IMU files (foot_samples). Mixing them up would place a shaded region tens of
    seconds off, so the column is mandatory rather than implied by the label.
    """
    rows = []
    for label, intervals in activity_intervals.items():
        for start, end in intervals:
            rows.append({'label': label, 'time_base': 'synced', 'start_index': start, 'end_index': end,
                         'start_time': timestamps[start], 'end_time': timestamps[min(end, len(timestamps) - 1)]})
    if raw_timestamps is not None:
        for start, end in stationary_intervals:
            rows.append({'label': 'foot_stationary', 'time_base': 'raw', 'start_index': start, 'end_index': end,
                         'start_time': raw_timestamps[start],
                         'end_time': raw_timestamps[min(end, len(raw_timestamps) - 1)]})
    df = pd.DataFrame(rows, columns=['label', 'time_base', 'start_index', 'end_index',
                                     'start_time', 'end_time'])
    df['duration_s'] = df['end_time'] - df['start_time']
    return df

# ==============================================================================
# Per-sensor and per-joint scalars
# ==============================================================================

def foot_samples(raw_imus: Dict[str, IMUTrace], foot_sensors: List[str]) -> pd.DataFrame:
    """Raw-time-base signal norms for the foot sensors, so the stationary-detection
    diagnostic figure can be drawn without reloading the raw IMU files."""
    rows = []
    for sensor in foot_sensors:
        trace = raw_imus[sensor]
        rows.append(pd.DataFrame({
            'timestamp': trace.timestamps.astype(np.float64),
            'sensor': sensor,
            'acc_norm': np.linalg.norm(trace.acc, axis=1).astype(np.float32),
            'gyro_norm': np.linalg.norm(trace.gyro, axis=1).astype(np.float32),
            'mag_norm': np.linalg.norm(trace.mag, axis=1).astype(np.float32),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sensor_stats(plates: Dict[str, PlateTrial], raw_imus: Optional[Dict[str, IMUTrace]],
                 stationary_intervals: List[Tuple[int, int]]) -> pd.DataFrame:
    """Per-sensor scalars for one trial: field-magnitude variability, non-gravity acceleration,
    median world-frame field, and (feet only) the intrinsic per-axis noise floor.

    mag_norm_std is the std of |mag| over the whole trial. It is deliberately the
    magnitude's variability rather than a deviation from a reference: |mag| is
    orientation-independent, so any variation in it is distortion or sensor noise and
    nothing else, which makes it the one distortion measure that needs no field reference
    at all.

    acc_norm_std is the same construction for the accelerometer, and the same argument applies:
    a sensor reading gravity alone has |acc| = |g| whatever its orientation, so the variability
    of |acc| is a reference-free measure of how much the reading is NOT gravity.

    linacc_rms is the direct version of that, available here because this dataset has mocap
    orientations: RMS |a_world - g|, the actual non-gravity acceleration. Both are reported
    because they answer slightly different questions - acc_norm_std is what a filter could
    measure for itself at runtime, linacc_rms is the ground truth it would be trying to infer -
    and because acc_norm_std systematically UNDERSTATES the departure. Linear acceleration
    perpendicular to gravity barely changes |acc| (it adds in quadrature), so a segment swinging
    horizontally can read |acc| ~ |g| throughout while its non-gravity component is large.

    The noise columns are length-weighted means of the per-interval per-axis stds, and
    n_stationary_samples is carried alongside so that pooling ACROSS trials can weight the
    same way. They are NaN for every non-foot sensor: only the feet are ground-anchored,
    and a torso sensor during a seated pause is still tracking postural sway, not sitting
    at its noise floor.
    """
    rows = []
    for sensor, plate in plates.items():
        world = plate.get_imu_trace_in_global_frame()
        mag_norm = np.linalg.norm(plate.imu_trace.mag, axis=1)
        acc_norm = np.linalg.norm(plate.imu_trace.acc, axis=1)
        linacc = np.linalg.norm(world.acc - EXPECTED_GRAVITY, axis=1)
        row = {
            'sensor': sensor,
            'segment': SENSOR_SEGMENT.get(sensor, sensor),
            'mag_norm_std': float(np.std(mag_norm)),
            'mag_norm_median': float(np.median(mag_norm)),
            'acc_norm_std': float(np.std(acc_norm)),
            'acc_norm_median': float(np.median(acc_norm)),
            'linacc_rms': float(np.sqrt(np.mean(linacc ** 2))),
            'linacc_median': float(np.median(linacc)),
            'world_mag_x': float(np.median(world.mag[:, 0])),
            'world_mag_y': float(np.median(world.mag[:, 1])),
            'world_mag_z': float(np.median(world.mag[:, 2])),
            'n_stationary_samples': 0,
        }
        for modality in ('gyro', 'acc', 'mag'):
            for axis in 'xyz':
                row[f'{modality}_noise_{axis}'] = np.nan
        rows.append(row)

    by_sensor = {row['sensor']: row for row in rows}
    if raw_imus is not None and stationary_intervals:
        # Feet only, and deliberately so: the intervals were defined by both feet being
        # still, which says nothing about the torso or thighs. Filling these columns for
        # every sensor would look like more data and be a measurement of postural sway.
        for sensor in FOOT_SENSORS:
            if sensor not in by_sensor or sensor not in raw_imus:
                continue
            row = by_sensor[sensor]
            trace = raw_imus[sensor]
            weighted = {modality: np.zeros(3) for modality in ('gyro', 'acc', 'mag')}
            total = 0
            for start, end in stationary_intervals:
                start, end = min(start, len(trace)), min(end, len(trace))
                if end <= start:
                    continue
                length = end - start
                total += length
                for modality in ('gyro', 'acc', 'mag'):
                    weighted[modality] += np.std(getattr(trace, modality)[start:end], axis=0) * length
            if total == 0:
                continue
            row['n_stationary_samples'] = total
            for modality, value in weighted.items():
                for i, axis in enumerate('xyz'):
                    row[f'{modality}_noise_{axis}'] = float(value[i] / total)

    return pd.DataFrame(rows)


def joint_mag_consistency(plates: Dict[str, PlateTrial], expected_mag: np.ndarray) -> pd.DataFrame:
    """Per-joint test of MAJIC's premise: is one sensor's field better predicted by its
    NEIGHBOUR across the joint than by the subject's global field?

    Which sensor of the pair is the reference and which is the target is set by
    FIELD_REFERENCE (default: the kinematic parent references the child; the lumbar is
    reversed, since there the torso is the clean sensor and the pelvis the distorted one).
    Both directions are computed either way, in var_reduction_parent_ref and
    var_reduction_child_ref, so nothing about the choice is baked in irreversibly.

    Two views of the same comparison, both in the world frame:
      * cosine similarity of the target's field against each candidate reference - a
        direction-only measure, which is what an orientation filter actually consumes.
        Note the target-vs-reference similarity is symmetric, so it is the one number here
        that the direction choice cannot affect;
      * var_reduction = 1 - Var(target - reference) / Var(target - global), the fraction of
        the target's residual variance that using the neighbour instead of the global field
        removes.

    Read var_reduction with its algebra in mind. The global field is a constant, so
    Var(target - global) = Var(target), and the whole thing collapses to

        var_reduction = [2 Cov(t,r) - Var(r)] / Var(t) = 2 rho sqrt(k) - k,
        where k = Var(reference)/Var(target) and rho = corr(target, reference)

    which is positive iff Var(reference)/Var(target) < 4 rho^2. THE SIGN IS SET BY THE
    VARIANCE RATIO BETWEEN THE TWO SENSORS, not by whether the neighbour carries information
    about the target's field. Two consequences worth knowing before quoting a number:

      * A negative value does not mean the neighbour is uninformative. It means substituting
        its field verbatim (unit gain) imports more of the reference's own fluctuation than it
        cancels of the target's. Fitting a gain instead recovers a large positive reduction on
        exactly those trials - the lumbar direction this module does NOT use scores -0.88 on
        Subject03 unit-gain and +0.81 with a fitted gain, off the same samples.
      * This asymmetry is exactly why FIELD_REFERENCE reverses the lumbar. Referencing the
        clean torso to the distorted pelvis gives k ~ 0.2 and a reduction in line with the
        limb joints; referencing the pelvis to the torso gives k ~ 1.4-4.6 and a sign that
        flips trial to trial. Same samples, same rho: only the roles differ.

    The variance framing also discards any CONSTANT offset between the two sensors, which is
    not free for a relative-orientation filter - a fixed field disagreement is a fixed
    orientation error. The cos-sim columns are what cover that side of it.

    Note this is a test of the premise (nearby sensors share a local field the global
    reference misses), not a simulation of the method: the filter estimates the field at the
    joint center rather than copying the parent's reading.
    """
    rows = []
    for joint, (parent_sensor, child_sensor) in JOINTS.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        parent_mag = plates[parent_sensor].get_imu_trace_in_global_frame().mag
        child_mag = plates[child_sensor].get_imu_trace_in_global_frame().mag
        n = min(len(parent_mag), len(child_mag))
        parent_mag, child_mag = parent_mag[:n], child_mag[:n]
        global_mag = np.tile(expected_mag, (n, 1))

        reference_role = FIELD_REFERENCE.get(joint, 'parent')
        reference_mag, target_mag = ((parent_mag, child_mag) if reference_role == 'parent'
                                     else (child_mag, parent_mag))
        reference_sensor, target_sensor = ((parent_sensor, child_sensor) if reference_role == 'parent'
                                          else (child_sensor, parent_sensor))

        rows.append({
            'joint': joint,
            'reference_role': reference_role,
            'reference_sensor': reference_sensor,
            'target_sensor': target_sensor,
            'cos_sim_target_global': float(np.mean(_cosine_similarity(target_mag, global_mag))),
            # Symmetric in the pair, so unaffected by the reference direction.
            'cos_sim_target_reference': float(np.mean(_cosine_similarity(target_mag, reference_mag))),
            'corr_target_reference': _pair_correlation(target_mag, reference_mag),
            # k and the gain that would be optimal for it. Stored because every reading of
            # var_reduction routes through them: the sign is set by k vs 4*rho^2, the shortfall
            # against the rho^2 ceiling is exactly (rho - sqrt(k))^2, and unit substitution is
            # optimal only where optimal_gain happens to be 1.
            'variance_ratio_k': _variance_ratio(target_mag, reference_mag),
            'optimal_gain': _optimal_gain(target_mag, reference_mag),
            # The residual the relative filter actually forms for its mag measurement, and so the
            # scale its mag noise block should carry. RelativeFilter's get_h computes exactly
            # R_wp @ m_p - R_wc @ m_c and drives it to zero, so the RMS of that quantity is the
            # size of the disagreement it is calling noise. Compare against DEFAULT_MAG_STD.
            'mag_residual_rms': float(np.sqrt(np.mean(
                np.sum((parent_mag[:n] - child_mag[:n]) ** 2, axis=1)))),
            'mag_residual_angle_deg': float(np.degrees(np.mean(np.arccos(
                np.clip(_cosine_similarity(parent_mag[:n], child_mag[:n]), -1.0, 1.0))))),
            # Global field = 0 on this scale by construction; see _best_var_reduction.
            'var_reduction_global': 0.0,
            'var_reduction': _var_reduction(target_mag, reference_mag),
            'var_reduction_best': _best_var_reduction(target_mag, reference_mag),
            'var_reduction_parent_ref': _var_reduction(child_mag, parent_mag),
            'var_reduction_child_ref': _var_reduction(parent_mag, child_mag),
        })
    return pd.DataFrame(rows, columns=[
        'joint', 'reference_role', 'reference_sensor', 'target_sensor', 'cos_sim_target_global',
        'cos_sim_target_reference', 'corr_target_reference', 'variance_ratio_k', 'optimal_gain',
        'mag_residual_rms', 'mag_residual_angle_deg', 'var_reduction_global', 'var_reduction',
        'var_reduction_best', 'var_reduction_parent_ref', 'var_reduction_child_ref'])


def joint_acc_residual(plates: Dict[str, PlateTrial]) -> pd.DataFrame:
    """Per-joint RMS of the ACC residual the relative filter actually forms, projected to the
    joint center: |R_wp a_p - R_wc a_c|, the accelerometer twin of mag_residual_rms.

    This is deliberately not the same quantity as experiments/acceleration_projection.py's
    err_proj, and the difference is the whole point. err_proj is each sensor's projection error
    against the mocap joint center — the right measure of whether the projection physics works.
    This is the DISAGREEMENT BETWEEN THE TWO SENSORS, which is what RelativeFilter.get_h drives
    to zero and therefore what its acc noise block has to cover.

    The two can differ by a lot in either direction, because the two sensors' projection errors
    are not independent: they share one joint-center fit, one mocap alignment, and one segment
    pair. Whatever they share CANCELS in the difference, so if the error is largely common-mode
    the filter sees far less than 2x the per-sensor variance and setting acc_std from err_proj
    would badly over-inflate it. Whatever is independent adds. Which case holds is measured
    rather than assumed — compare acc_residual_rms against sqrt(2) * err_proj.

    Gravity cancels identically in the difference, so this needs no gravity constant and no
    double-differentiated mocap position: only the ground-truth rotations that bring both
    readings into a common frame. That makes it a much better conditioned quantity than any
    error measured against a mocap-derived acceleration.

    acc_residual_rms_raw is the same residual WITHOUT the projection, so the pair says whether
    projecting to the joint center actually reduces the disagreement the filter has to absorb.
    """
    rows = []
    for joint, (parent_sensor, child_sensor) in JOINTS.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        parent_proj, child_proj = project_pair_to_joint_center(plates[parent_sensor],
                                                               plates[child_sensor])
        projected = [parent_proj.get_imu_trace_in_global_frame().acc,
                     child_proj.get_imu_trace_in_global_frame().acc]
        raw = [plates[parent_sensor].get_imu_trace_in_global_frame().acc,
               plates[child_sensor].get_imu_trace_in_global_frame().acc]
        n = min(len(projected[0]), len(projected[1]))

        def rms(pair):
            residual = pair[0][:n] - pair[1][:n]
            return float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))

        rows.append({'joint': joint, 'acc_residual_rms': rms(projected),
                     'acc_residual_rms_raw': rms(raw)})
    return pd.DataFrame(rows, columns=['joint', 'acc_residual_rms', 'acc_residual_rms_raw'])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def _var_reduction(target: np.ndarray, reference: np.ndarray) -> float:
    """1 - Var(target - reference) / Var(target), summed over the world-frame components.

    Var(target - global) reduces to Var(target) because the global field is a constant, so no
    global field needs passing in — which also makes it obvious that this measures only the
    TIME-VARYING part of the disagreement. See joint_mag_consistency for how to read the sign."""
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
    Var(target - a*reference) over the gain a gives a* = Cov/Var(reference) and a residual of
    Var(target)(1 - rho^2), so the achievable reduction is rho^2 while var_reduction is what
    the same reference delivers with a forced to 1. The two therefore bracket the reference:
    rho^2 is what the pair COULD support, var_reduction is what substitution actually gets.

    This is also the form in which the correlation becomes reportable at all. Correlation
    against the global field is undefined - the global field is a constant, so its variance is
    zero and the ratio is 0/0 - which leaves a bare rho column with no baseline arm to compare
    to. On this scale the global field is exactly 0, since Var(target - global) = Var(target)
    for any constant global, so all three quantities sit on one axis with a real origin.

    Symmetric in the pair (rho is), so like corr and unlike var_reduction it cannot be moved by
    the FIELD_REFERENCE direction choice.
    """
    correlation = _pair_correlation(target, reference)
    return np.nan if np.isnan(correlation) else float(correlation ** 2)


def _pair_correlation(target: np.ndarray, reference: np.ndarray) -> float:
    """Correlation between the two sensors' world-frame fields, POOLED over the components:

        sum_i Cov(t_i, r_i) / sqrt(sum_i Var(t_i) * sum_i Var(r_i))

    Not the mean of the three per-axis correlations, for two reasons. It is variance-weighted,
    which matters because an axis carrying almost no field variation contributes a per-axis
    correlation that is mostly noise against noise - at the lumbar, x and z correlate at 0.36
    and 0.14 while y, which carries the variation, correlates at 0.93, so a flat mean reads
    0.47 against this 0.65. And it is the rho for which var_reduction = 2 rho sqrt(k) - k holds
    exactly, which is the identity the whole sign discussion rests on.

    Reported alongside var_reduction because it, not var_reduction, is the direction-independent
    evidence for the premise: it is symmetric in the pair, so FIELD_REFERENCE cannot move it,
    and var_reduction can go negative on a pair whose fields correlate at 0.86.

    Blind to two things by construction, both of which the cos-sim columns cover instead: it is
    mean-centered, so a constant field disagreement between the sensors is invisible (and that
    one is NOT free - for a relative-orientation filter it is a constant orientation error), and
    it is scale-free, so an amplitude mismatch is invisible. High correlation therefore does not
    mean the two sensors see the same field, only that their fields move together."""
    var_target = float(np.sum(np.var(target, axis=0)))
    var_reference = float(np.sum(np.var(reference, axis=0)))
    if var_target <= 0 or var_reference <= 0:
        return np.nan
    # Population covariance (ddof=0), matching np.var above rather than np.cov's ddof=1 default:
    # rho and var_reduction are meant to satisfy var_reduction = 2*rho*sqrt(k) - k exactly, and
    # mixing the two conventions breaks that identity in the fourth decimal.
    centered_product = (target - target.mean(axis=0)) * (reference - reference.mean(axis=0))
    covariance = float(np.sum(centered_product.mean(axis=0)))
    return covariance / np.sqrt(var_target * var_reference)

# ==============================================================================
# Per-trial driver / grid worker
# ==============================================================================

def compute_trial(subject: str, activity: str, plates: Dict[str, PlateTrial],
                  expected_mag: np.ndarray,
                  tables: Sequence[str] = TRIAL_TABLES) -> Dict[str, pd.DataFrame]:
    """Everything this experiment computes for one subject/activity, as the six per-trial tables.

    `tables` restricts the work to a subset (see --only-tables). Anything only one table needs is
    skipped when that table is not wanted, which is what makes a targeted rerun cheap: the two
    per-sample tables carry the rigid-body projection and dominate the runtime, and the raw
    untrimmed IMU load only exists for the foot/noise tables.

    Raw (untrimmed) IMUs are loaded here in addition to the synced plates — see
    find_foot_stationary_intervals for why the noise floor cannot use the synced traces.
    """
    wanted = set(tables)
    fs = plates['pelvis_imu'].imu_trace.get_sample_frequency()
    timestamps = plates['pelvis_imu'].imu_trace.timestamps

    raw_imus, raw_timestamps = None, None
    stationary, foot_sensor_names = [], []
    imu_folder = paths.raw_trial_dir(subject, activity) / "imu data"
    if wanted & {'foot_samples', 'intervals', 'sensor_stats'} and imu_folder.exists():
        raw_imus = IMUTrace.from_folder(imu_folder)
        stationary, foot_sensor_names = find_foot_stationary_intervals(raw_imus)
        if foot_sensor_names:
            raw_timestamps = raw_imus[foot_sensor_names[0]].timestamps

    computed = {
        'segment_samples': lambda: segment_samples(plates, expected_mag),
        'joint_samples': lambda: joint_samples(plates, fs),
        'foot_samples': lambda: foot_samples(raw_imus, foot_sensor_names) if raw_imus else pd.DataFrame(),
        'intervals': lambda: intervals_table(label_activity_intervals(plates, fs), timestamps,
                                            stationary, raw_timestamps),
        'sensor_stats': lambda: sensor_stats(plates, raw_imus, stationary),
        'joint_stats': lambda: joint_mag_consistency(plates, expected_mag).merge(
            joint_acc_residual(plates), on='joint', how='outer'),
    }
    return {table: build() for table, build in computed.items() if table in wanted}


def _subject_worker(row_key: str, stage_labels: List[str], shared_state: Dict,
                    activities: List[str], tables: Sequence[str] = TRIAL_TABLES) -> None:
    """One process per SUBJECT, not per trial: a subject's global magnetic field is the
    median over all of that subject's data (expected_mag_field), so both activities have
    to be in hand before either trial's magdev can be computed. Splitting by trial would
    mean either loading every subject twice or using a per-trial field reference, and a
    reference that shifts between a subject's two trials is not a reference."""
    subject = row_key
    stage_labels = list(stage_labels)
    load_stage, activity_stages = stage_labels[0], stage_labels[1:]

    t_start = time.time()
    shared_state[(row_key, load_stage)] = "Running"
    plates_by_activity = {}
    for activity in activities:
        if not paths.raw_trial_dir(subject, activity).exists():
            continue
        try:
            plates = load_raw_data(subject, activity)
        except Exception as e:  # a corrupt or incomplete trial should not sink the subject
            shared_state[(row_key, activity)] = f"Failed ({e})"
            continue
        if 'pelvis_imu' not in plates:
            shared_state[(row_key, activity)] = "Skipped"
            continue
        plates_by_activity[activity] = plates

    if not plates_by_activity:
        shared_state[(row_key, load_stage)] = "Failed (no trials)"
        for activity in activity_stages:
            shared_state[(row_key, activity)] = "Skipped"
        return None

    field = expected_mag_field(plates_by_activity)
    _save(pd.DataFrame([{
        'subject': f"Subject{subject}",
        'world_mag_x': field[0], 'world_mag_y': field[1], 'world_mag_z': field[2],
        'world_mag_norm': float(np.linalg.norm(field)),
        'activities': ",".join(sorted(plates_by_activity)),
        'n_sensors': len(next(iter(plates_by_activity.values()))),
    }]), expected_mag_field_path(subject), subject=f"Subject{subject}")
    shared_state[(row_key, f"{load_stage}_time")] = time.time() - t_start
    shared_state[(row_key, load_stage)] = "Success"

    for activity in activity_stages:
        if activity not in plates_by_activity:
            if shared_state.get((row_key, activity), "Pending") == "Pending":
                shared_state[(row_key, activity)] = "Skipped"
            continue
        t_activity = time.time()
        shared_state[(row_key, activity)] = "Running"
        try:
            computed = compute_trial(subject, activity, plates_by_activity[activity],
                                     field, tables=tables)
            for table, df in computed.items():
                if df.empty:
                    continue
                _save(df, trial_table_path(subject, activity, table),
                      subject=f"Subject{subject}", activity=activity, table=table)
            shared_state[(row_key, f"{activity}_time")] = time.time() - t_activity
            shared_state[(row_key, activity)] = "Success"
        except Exception as e:
            shared_state[(row_key, activity)] = f"Failed ({e})"
    return None

# ==============================================================================
# Pooled summary
# ==============================================================================

def _describe(df: pd.DataFrame, value_col: str, keys: List[str]) -> pd.DataFrame:
    grouped = df.groupby(keys, observed=True)[value_col]
    stats = grouped.agg(n_samples='size', mean='mean', std='std', min='min', max='max')
    quantiles = grouped.quantile(QUANTILES).unstack()
    quantiles.columns = [f"p{int(round(q * 100)):02d}" for q in quantiles.columns]
    return stats.join(quantiles).reset_index()


SUMMARY_COLUMNS = (['subject', 'activity', 'metric', 'unit', 'group_kind', 'group',
                    'n_samples', 'mean', 'std', 'min']
                   + [f"p{int(round(q * 100)):02d}" for q in QUANTILES] + ['max'])


def summarize_distributions(segment_df: pd.DataFrame, joint_df: pd.DataFrame) -> pd.DataFrame:
    """Tidy quantile table for every metric x grouping, WITH MARGINS: rows where `subject`
    or `activity` is the literal string 'all' are the pooled version of the rows above
    them. Marginal rows are computed by re-aggregating the samples, not by averaging the
    per-trial quantiles - a mean of medians is not a median, and trials differ in length.

    Distributions are summarized by quantiles rather than mean +- std because all three of
    these metrics are strongly right-skewed (impulsive footfalls, one-sided magnetic
    distortion), so a standard deviation implies a symmetry that is not there.
    """
    specs = []
    if not segment_df.empty:
        specs += [(segment_df, 'linacc', 'segment'), (segment_df, 'magdev', 'segment')]
    if not joint_df.empty:
        specs += [(joint_df, 'obs_min', 'joint')]

    frames = []
    for df, metric, group_kind in specs:
        for by_subject, by_activity in [(True, True), (True, False), (False, True), (False, False)]:
            keys = (['subject'] if by_subject else []) + (['activity'] if by_activity else []) + [group_kind]
            part = _describe(df, metric, keys).rename(columns={group_kind: 'group'})
            if not by_subject:
                part['subject'] = 'all'
            if not by_activity:
                part['activity'] = 'all'
            part['metric'] = metric
            part['unit'] = METRIC_UNITS[metric]
            part['group_kind'] = group_kind
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.concat(frames, ignore_index=True)[SUMMARY_COLUMNS]

# ==============================================================================
# Console report
# ==============================================================================

def _header(number: int, title: str, subtitle: str) -> None:
    print("\n" + "=" * 88)
    print(f"{number}. {title}")
    print(f"   ({subtitle})")
    print("=" * 88)


def report_distributions(summary: pd.DataFrame) -> None:
    _header(1, "SENSOR DISTRIBUTIONS BY SEGMENT / JOINT",
            "pooled over all subjects and activities; median [IQR], p90")
    pooled = summary[(summary['subject'] == 'all') & (summary['activity'] == 'all')]
    if pooled.empty:
        print("No per-sample data found.")
        return
    for metric, frame in pooled.groupby('metric', sort=False):
        unit = frame['unit'].iloc[0]
        order = SEGMENT_ORDER if frame['group_kind'].iloc[0] == 'segment' else list(JOINTS)
        print(f"\n{metric} ({unit}):")
        print(f"  {'group':<12}{'median':>10}{'IQR':>22}{'p90':>10}{'p99':>10}{'n':>12}")
        for group in [g for g in order if g in set(frame['group'])]:
            row = frame[frame['group'] == group].iloc[0]
            iqr = f"[{row['p25']:.2f}, {row['p75']:.2f}]"
            print(f"  {group:<12}{row['p50']:>10.2f}{iqr:>22}{row['p90']:>10.2f}"
                  f"{row['p99']:>10.2f}{int(row['n_samples']):>12,}")


def report_magnetic_distortion(sensor_df: pd.DataFrame) -> None:
    _header(2, "MAGNETIC FIELD DISTORTION PER SENSOR",
            "std of the mag vector MAGNITUDE over the whole trial, averaged across trials")
    if sensor_df.empty:
        print("No sensor stats found.")
        return
    means = sensor_df.groupby('sensor')['mag_norm_std'].mean().sort_values()
    for sensor, value in means.items():
        print(f"{sensor:<20}: {value:.4f} {MAG_UNIT}")
    print(f"\n-> LOWEST distortion : {means.index[0]} ({means.iloc[0]:.4f} {MAG_UNIT})")
    print(f"-> HIGHEST distortion: {means.index[-1]} ({means.iloc[-1]:.4f} {MAG_UNIT})")


def report_acceleration_departure(sensor_df: pd.DataFrame) -> None:
    """The accelerometer counterpart to report_magnetic_distortion: how far each sensor's
    reading departs from gravity alone.

    Ordered proximal to distal rather than by value, because that ordering IS the finding here
    (the magnetic section sorts by value and comes out in nearly the same order for the same
    reason). The last column is the one that matters for tuning: an orientation filter treats
    the accelerometer as a gravity reference, so the non-gravity component is measurement ERROR,
    and comparing it against the acc_std the filter is tuned with says how far off that tuning
    is. Per-axis converts the 3-vector RMS to the per-axis scale acc_std is expressed in.
    """
    _header(3, "NON-GRAVITY ACCELERATION PER SENSOR",
            "how far each accelerometer departs from reading gravity alone, "
            "per-trial values averaged across trials")
    if sensor_df.empty:
        print("No sensor stats found.")
        return

    means = sensor_df.groupby('segment')[['acc_norm_std', 'linacc_rms', 'linacc_median']].mean()
    print(f"  {'segment':<10}{'std |acc|':>12}{'RMS |a-g|':>12}{'median |a-g|':>14}"
          f"{'per-axis':>10}{'vs acc_std':>12}")
    print(f"  {'':<10}{'(m/s^2)':>12}{'(m/s^2)':>12}{'(m/s^2)':>14}{'(m/s^2)':>10}{'':>12}")
    for segment in [s for s in SEGMENT_ORDER if s in means.index]:
        row = means.loc[segment]
        per_axis = row['linacc_rms'] / np.sqrt(3)
        print(f"  {segment:<10}{row['acc_norm_std']:>12.2f}{row['linacc_rms']:>12.2f}"
              f"{row['linacc_median']:>14.2f}{per_axis:>10.2f}"
              f"{per_axis / DEFAULT_ACC_STD:>11.0f}x")

    ordered = means['linacc_rms'].sort_values()
    print(f"\n-> LOWEST non-gravity acceleration : {ordered.index[0]} ({ordered.iloc[0]:.2f} m/s^2)")
    print(f"-> HIGHEST non-gravity acceleration: {ordered.index[-1]} ({ordered.iloc[-1]:.2f} m/s^2)")

    # Read the two middle columns together, not either one alone: they disagree about the feet,
    # and the disagreement is the useful part.
    ratio = (means['linacc_rms'] / means['linacc_median']).sort_values()
    worst = ratio.index[-1]
    print(f"\nRMS/median ranges from {ratio.iloc[0]:.1f}x ({ratio.index[0]}) to {ratio.iloc[-1]:.1f}x "
          f"({worst}), so these\ndistributions are not merely skewed but differently shaped by "
          f"segment. The distal segments are\nthe extreme case: {worst} has the highest RMS of any "
          f"segment while its MEDIAN is among the lowest,\nbecause a foot alternates between flat "
          f"stance — genuinely gravity-only, the best accelerometer\nreference on the body — and "
          f"footfall impacts of tens of m/s^2. A single per-segment acc_std\ntherefore describes "
          f"the foot badly in both directions at once: far too small during impact and\nfar too "
          f"large during stance. That is an argument for gating the accelerometer in TIME (which "
          f"is\nwhat o^J does) rather than for a bigger constant, and it does not apply to the "
          f"magnetic side,\nwhere distortion is a property of location rather than of gait phase.")
    # Computed against the measured floor rather than a written-down one: DEFAULT_ACC_STD is
    # actively being retuned, and a hardcoded "acc_std is about 2x the noise" sentence here went
    # false the moment it moved.
    anchored = sensor_df[sensor_df['n_stationary_samples'] > 0]
    if not anchored.empty:
        weights = anchored['n_stationary_samples'].to_numpy()
        axes = np.array([anchored[f'acc_noise_{axis}'].to_numpy() for axis in 'xyz'])
        floor = float(((axes * weights).sum(axis=1) / weights.sum()).mean())
        relation = (f"{DEFAULT_ACC_STD / floor:.1f}x that floor" if DEFAULT_ACC_STD >= floor
                    else f"{floor / DEFAULT_ACC_STD:.1f}x BELOW that floor")
        print(f"\nThe filter is tuned with acc_std={DEFAULT_ACC_STD}, against a measured "
              f"accelerometer noise floor of\n{floor:.4f} m/s^2 (section 5) — i.e. {relation}. "
              f"Either way the quantity it actually has to\nabsorb is the non-gravity acceleration "
              f"above, orders of magnitude larger than both and growing\nsteadily toward the distal "
              f"segments.")
    print(f"\nNote this is the PER-SENSOR departure from gravity, which is not the same as what the "
          f"relative\nfilter has to absorb: that filter's acc residual is the DIFFERENCE of two "
          f"sensors projected to a\nshared joint center, and the two projections' errors largely "
          f"cancel (measured at 12-47% of what\nindependent errors would give). See "
          f"joint_acc_residual for the pairwise number, which is the one\nto tune acc_std from.")


def report_world_field(fields_df: pd.DataFrame) -> None:
    _header(4, "GLOBAL MAGNETIC FIELD VARIATION ACROSS SUBJECTS",
            "median world-frame field per subject, over all that subject's sensors and trials")
    if fields_df.empty:
        print("No per-subject field estimates found.")
        return
    for _, row in fields_df.sort_values('subject').iterrows():
        print(f"{row['subject']}: [{row['world_mag_x']:>7.2f}, {row['world_mag_y']:>7.2f}, "
              f"{row['world_mag_z']:>7.2f}] {MAG_UNIT} (norm {row['world_mag_norm']:.2f})")
    vectors = fields_df[['world_mag_x', 'world_mag_y', 'world_mag_z']].to_numpy()
    mean, std = vectors.mean(axis=0), vectors.std(axis=0)
    print(f"\nAcross {len(fields_df)} subjects (all values in {MAG_UNIT}):")
    print(f"  Mean            : [{mean[0]:.2f}, {mean[1]:.2f}, {mean[2]:.2f}]")
    print(f"  Std across subj : [{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}]")
    print(f"  Std of magnitude: {np.std(np.linalg.norm(vectors, axis=1)):.4f}")


def report_sensor_noise(sensor_df: pd.DataFrame) -> None:
    _header(5, "INTRINSIC SENSOR NOISE FLOOR",
            "per-axis std of ground-anchored foot sensors during stationary periods, "
            "length-weighted")
    anchored = sensor_df[sensor_df['n_stationary_samples'] > 0] if not sensor_df.empty else sensor_df
    if anchored.empty:
        print("No stationary foot periods found - no noise estimate available.")
        return
    weights = anchored['n_stationary_samples'].to_numpy()
    print(f"Pooled over {len(anchored)} sensor-trials, "
          f"{weights.sum():,} stationary samples:")
    units = {'gyro': 'rad/s', 'acc': 'm/s^2', 'mag': MAG_UNIT}
    for modality, unit in units.items():
        axes = np.array([anchored[f'{modality}_noise_{axis}'].to_numpy() for axis in 'xyz'])
        pooled = (axes * weights).sum(axis=1) / weights.sum()
        print(f"  {modality.capitalize():<5} noise std ({unit:<6}): {pooled.mean():.6f} | "
              f"[x={pooled[0]:.5f}, y={pooled[1]:.5f}, z={pooled[2]:.5f}]")
    print(f"\nFor reference, the filter is tuned with gyro_std={pipeline_constants()['gyro_std']}, "
          f"acc_std={pipeline_constants()['acc_std']}, mag_std={pipeline_constants()['mag_std']}.")


def report_joint_mag_consistency(joint_df: pd.DataFrame) -> None:
    _header(6, "MAGNETIC FIELD LOCAL CONSISTENCY: NEIGHBOUR SENSOR vs. GLOBAL FIELD",
            "world-frame fields; one value per trial, summarized across trials")
    if joint_df.empty:
        print("No joint consistency stats found.")
        return
    # Median leads and the range follows, because var_reduction is not symmetric across trials
    # and a mean over a sign-flipping spread reads as a small consistent effect when the data
    # shows no consistent effect at all.
    print(f"{'Joint':<10}{'Reference':>14}{'CosSim tgt-global':>19}{'CosSim tgt-ref':>16}"
          f"{'Var reduction':>15}{'[min, max]':>18}{'n<0':>5}{'best possible':>15}{'n':>4}")
    grouped = joint_df.groupby('joint')
    flipped = []
    for joint in [j for j in JOINTS if j in grouped.groups]:
        rows = grouped.get_group(joint)
        reduction = rows['var_reduction']
        role = rows['reference_role'].iloc[0]
        reference = rows['reference_sensor'].iloc[0].replace('_imu', '')
        if role != 'parent':
            flipped.append((joint, reference, rows['target_sensor'].iloc[0].replace('_imu', '')))
        print(f"{joint:<10}{reference + ('*' if role != 'parent' else ''):>14}"
              f"{rows['cos_sim_target_global'].median():>19.3f}"
              f"{rows['cos_sim_target_reference'].median():>16.3f}"
              f"{reduction.median():>15.3f}"
              f"{f'[{reduction.min():.2f}, {reduction.max():.2f}]':>18}"
              f"{int((reduction < 0).sum()):>5}{rows['var_reduction_best'].median():>15.3f}"
              f"{len(rows):>4}")

    print("\nWhat this table compares: for a filter that needs the field at a sensor, is it better to\n"
          "assume ONE CONSTANT field everywhere on the body, or to take the field from the sensor on\n"
          "the ADJACENT segment? Var reduction answers it — the constant scores 0 and the neighbour\n"
          "scores what is printed, so the neighbour is better wherever that column is positive.\n"
          "\nBoth reduction columns are the fraction of the TARGET sensor's field variance that\n"
          "using its neighbour removes, on a scale where THE CONSTANT GLOBAL FIELD IS EXACTLY 0 —\n"
          "that is definitional, not measured, since a constant cannot track anything that varies:\n"
          "Var(target - global) = Var(target) for any constant global. The flip side is that this\n"
          "column is blind to a constant OFFSET error, which the CosSim columns cover instead.\n"
          "  Var reduction  what substituting the neighbour's field verbatim achieves. Equals\n"
          "                 2*rho*sqrt(k) - k for k = Var(reference)/Var(target), so its SIGN\n"
          "                 tracks the two sensors' variance ratio, not how informative the\n"
          "                 neighbour is — read joint_mag_consistency before quoting one.\n"
          "  best possible  NOT part of the constant-vs-neighbour comparison. It answers the\n"
          "                 follow-up: is Var reduction limited by how much the neighbour knows,\n"
          "                 or by how crudely it is being used? rho^2, the ceiling if the\n"
          "                 neighbour's field were optimally rescaled; the shortfall against it is\n"
          "                 exactly (rho - sqrt(k))^2, and optimal_gain in the saved table says\n"
          "                 which way to rescale (~2 at knee and ankle, where the distal segment\n"
          "                 fluctuates several times more than its parent, so using the parent\n"
          "                 verbatim under-corrects).\n"
          "                 Reported as rho^2 rather than as rho because a correlation against\n"
          "                 the constant global field is undefined (zero variance), so rho\n"
          "                 alone has no baseline arm to compare against; rho^2 shares this\n"
          "                 column's scale and origin. Symmetric in the pair, so unlike Var\n"
          "                 reduction it is unaffected by the reference direction.")
    for joint, reference, target in flipped:
        print(f"* {joint}: reference reversed to {reference} (target {target}) by FIELD_REFERENCE — "
              f"{reference} is the cleaner sensor of the pair,\n"
              f"  where for every other joint the kinematic parent already is. Both directions are in "
              f"the saved table.")


def print_report(summary: pd.DataFrame, sensor_df: pd.DataFrame, fields_df: pd.DataFrame,
                 joint_df: pd.DataFrame) -> None:
    report_distributions(summary)
    report_magnetic_distortion(sensor_df)
    report_acceleration_departure(sensor_df)
    report_world_field(fields_df)
    report_sensor_noise(sensor_df)
    report_joint_mag_consistency(joint_df)

# ==============================================================================
# CLI / ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--activities', nargs='+', default=ACTIVITIES)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--only-tables', nargs='+', choices=TRIAL_TABLES, default=list(TRIAL_TABLES),
                        metavar='TABLE',
                        help="Recompute only these per-trial tables, leaving the others on disk "
                             f"untouched (one or more of: {', '.join(TRIAL_TABLES)}). Loading the "
                             "trials is unavoidable, but the two per-sample tables carry the "
                             "rigid-body projection and dominate the rest of the runtime, so "
                             "iterating on a derived table (e.g. --only-tables joint_stats) is much "
                             "cheaper than a full rerun.")
    parser.add_argument('--report-only', action='store_true',
                        help="Skip recomputation and rebuild the summary + report from the per-trial "
                             "tables already on disk. Note that the pooled summary always covers "
                             "every subject/activity present on disk, not just those passed to "
                             "--subjects, so a partial run does not silently narrow it.")
    args = parser.parse_args()

    if not args.report_only:
        stage_labels = ['load'] + args.activities
        if set(args.only_tables) != set(TRIAL_TABLES):
            print(f"Recomputing only {', '.join(args.only_tables)}; every other per-trial table is "
                  f"left as it is on disk.")
        run_tracked_grid(args.subjects, ['Subject'], stage_labels,
                         partial(_subject_worker, activities=args.activities,
                                 tables=args.only_tables),
                         args.workers, title="SENSOR DISTRIBUTIONS")

    print("\nLoading per-trial tables...")
    segment_df = load_trial_table('segment_samples')
    joint_sample_df = load_trial_table('joint_samples')
    sensor_df = load_trial_table('sensor_stats')
    joint_stat_df = load_trial_table('joint_stats')
    fields_df = load_expected_mag_fields()

    trials = set()
    for df in (segment_df, joint_sample_df, sensor_df):
        if not df.empty:
            trials |= set(map(tuple, df[['subject', 'activity']].drop_duplicates().to_numpy()))
    if not trials:
        print(f"No results found under {EXPERIMENT_DIR}. Run without --report-only first.")
        return
    print(f"Found {len(trials)} trial(s) across {len({s for s, _ in trials})} subject(s).")

    summary = summarize_distributions(segment_df, joint_sample_df)
    if not summary.empty:
        path = paths.ensure_parent(paths.statistics_path(EXPERIMENT_NAME))
        summary.to_parquet(path, engine='pyarrow', index=False)
        paths.write_manifest(path, constants=analysis_constants(), experiment=EXPERIMENT_NAME,
                             n_rows=len(summary),
                             subjects=sorted({s for s, _ in trials}),
                             activities=sorted({a for _, a in trials}))
        print(f"Saved summary to {path}")

    print_report(summary, sensor_df, fields_df, joint_stat_df)
    print(f"\nPer-trial tables under {EXPERIMENT_DIR}")
    print("Figures: python -m plotting.sensor_distributions")


if __name__ == '__main__':
    main()
