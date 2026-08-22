"""
Where on the segment you tape the IMU, and what it costs the joint angle.

IMoVE is the only dataset here that can answer this. It carries THREE sensors on each thigh and
each shank — High, Mid and Low, 7-19 cm apart against one marker cluster — so the same trial,
the same subject and the same motion can be scored nine different ways at the knee and three at
the hip and the ankle. Every other dataset offers one placement per segment and can only report
what it happened to get.

    python -m experiments.sensor_placement --dataset imove
    python -m experiments.sensor_placement --dataset imove --subjects s13 --trials t1_walking_001
    python -m experiments.sensor_placement --dataset imove --report-only

THE MEASUREMENT IS A CONTROLLED SUBSTITUTION
============================================
One trial, one joint, one filter, one reference construction — the ONLY thing that changes
between two cells is which sensor on the same rigid segment the filter was handed. The subject,
the motion, the marker cluster, the sample rate and the tuning are all held fixed by
construction, which is what makes a difference between cells attributable to placement rather
than to anything else.

Two structural facts make that substitution clean, and both were verified rather than assumed
(see test/TestSensorPlacement.py):

  THE SEGMENT POSE IS SHARED. `imove_mocap._pair_world_traces` hands the same WorldTrace to all
  three sensors on a segment, so the marker-derived segment orientation is bit-identical across
  placements. Angular velocity is position-invariant on a rigid body, so the three sensors also
  see the same |omega| — measured, and reported as `gyro_rms_spread_deg_s`, which is a CONTROL:
  it must stay near zero, and if it does not, the three plates did not come from one segment.

  WHAT DIFFERS IS THE LEVER ARM. Each placement gets its own mocap origin (`shift_world_origin`
  onto its own IMU) and therefore its own vector to the joint centre. Measured on
  s13/t1_walking_001, the thigh's three offsets to the knee are 257 / 159 / 93 mm going down the
  segment, off one identical 6.0 mm joint-centre fit residual — the residual belongs to the
  segment, the arm to the placement.

WHAT DOES NOT CARRY ACROSS PLACEMENTS, AND WHY THE SCORE STILL DOES
===================================================================
`align_world_to_imu` estimates a SEPARATE sensor-to-segment rotation for each placement, so the
three plates' world rotations differ by a constant — 12-19 deg on s13's right leg, with a
per-sample spread of 0.000 deg, i.e. exactly constant. Consequences, in order of how easy they
are to get wrong:

  * The joint ANGLE is expressed in a different frame for every placement. Comparing the raw
    rotvec components across placements is meaningless, so this file never does: the only
    quantity compared across cells is the SCALAR angular error against that cell's own reference.
  * That is a fair comparison, because each cell's reference is built from the same two sensor
    frames its estimate lives in, so the constant cancels.
  * But a mis-estimated mounting rotation does NOT cancel: a constant frame error E conjugates
    into the error series as E S(t)^T S(0) E^T S(0)^T S(t), which grows with how far the segment
    has rotated away from t=0. So part of any placement's penalty can be its own alignment fit
    rather than its position. That is a confound, it is measurable, and `align_residual_fraction`
    (the build's own gyro-alignment residual, from the trial manifest) carries it per sensor into
    every table. Section 6 of the report regresses it out rather than leaving it as a caveat.

THE ARMS, AND THE PREDICTION EACH ONE TESTS
===========================================
Six: three projections crossed with the magnetometer on and off.

    projection='none'       the accelerometer is used where it sits, which is what a filter with
                            no joint model does. The lever-arm term alpha x r + omega x (omega x r)
                            is then pure measurement error, and it grows with |r|, so this arm is
                            where placement should hurt MOST.
    projection='mocap'      MAJIC's projection with the joint centre from the markers. This is
                            the best case: the offset is as good as this dataset can make it.
    projection='inertial'   the same projection with the joint centre estimated from gyro and
                            accelerometer alone (Seel's objective, solved as in
                            experiments/inertial_joint_center.py, cold start). This is the
                            DEPLOYABLE case, and it is the one a practitioner's answer depends on.

The prediction the file exists to test: if placement sensitivity is the lever arm, then the
projection removes it, and the residual placement effect under 'mocap' is a floor set by
everything else (soft tissue, alignment, the field). If placement sensitivity survives the
projection, the lever arm was not the mechanism and the recommendation cannot be "project and
place it anywhere".

The inertial arm is what stops that from being a laboratory answer. The joint centre has to be
estimated from the very sensor whose position is in question, and a distal placement makes that
estimate worse at the same time as it makes the projection matter more — so the two effects
compound, and only this arm sees them together.

THE SIGN FLIP IS THE FALSIFICATION TEST
=======================================
A sensor's distance to the joint centre is not a property of the sensor. THIGH_R_H is 207 mm from
the hip and 257 mm from the knee; THIGH_R_L is 366 mm from the hip and 93 mm from the knee. So a
lever-arm mechanism predicts that the RANKING OF THE THREE PLACEMENTS REVERSES between the two
joints a segment spans, while every rival explanation — that sensor is noisier, that tape job was
sloppier, that placement has more soft tissue over it — predicts the same ranking at both. The
same six numbers therefore settle it, with no extra data and no model. `report_sign_flip` scores
the reversal per segment and per subject.

WHAT IT FOUND, over 21 subjects and 231 trials
==============================================
Quoted here so the file can be read for its result before it is read for its method. Every
number is the median over trials of a quantity computed WITHIN a trial and joint; `rms_deg`.

    placement spread (worst - best cell)    none 41.7   inertial 13.1   mocap 10.3  deg
    error against lever arm, Spearman       none  1.00  inertial  0.50  mocap  0.50
    error against lever arm, slope          none 16.0   inertial  2.5   mocap  1.9  deg/100 mm
    ranking reverses between the two
      joints a segment spans               none   66%  inertial   36%  mocap   28%
      (geometry control: 100%)

So placement is worth more than most of the method choices this repository benchmarks, the lever
arm is why, and the projection is most of the fix. What it does not fix — the residual ~10 deg —
correlates with the sensor-to-segment alignment residual (rho 0.38) rather than with position,
which points at the taped sensors' mounting estimate and not at where they were taped.

The mismatch penalty is nil: placement-matched and mismatched pairs differ by at most 0.4 deg and
never significantly, so both lever arms matter and their agreement does not.

WHAT IS DELIBERATELY NOT VARIED
===============================
Filter tuning. DEFAULT_*_STD is inherited unchanged from the rest of the pipeline, and it is known
to over-trust the accelerometer by orders of magnitude relative to the projected-acc mismatch
(see the ekf-acc-overtrust finding). That inflates every unprojected number in here. It is left
alone on purpose: retuning per placement would make the arms incomparable, and the question asked
here is what placement costs THIS pipeline, not what the best achievable pipeline would be. It
does mean the absolute degrees below are large; the ratios between placements are the result.

Trials with one sensor per segment are skipped rather than run. IMoVE's five long-walk sessions
carry only the Mid sensor of each cluster segment (plus feet and pelvis), so they offer no
placement contrast at all and every cell would be the same cell. They are counted and named at
startup rather than silently absent.

OUTPUTS, all under results/experiments/sensor_placement/<dataset>/<subject>/<trial>/

    placement_stats.parquet     one row per (joint, placement pair, arm): the error scalars
    placement_geometry.parquet  one row per (joint, placement pair): lever arms, the inertial
                                offset error, sensor height, alignment residual, offset provenance
    placement_samples.parquet   strided per-sample error, for the figures

plus the pooled quantile summary at
results/statistics/sensor_placement_<dataset>_statistics.parquet.

THE PARQUET IS THE INTERFACE — trials come from results/trials/<dataset>/ through
`experiment_utils.load_trial`, which refuses a stale artifact. Build first:

    python -m experiments.build_trials --dataset imove

Figures: python -m plotting.sensor_placement --dataset imove
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
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

import paths
from experiments.acceleration_projection import (PLACEMENT_TOKENS, chordal_mean,
                                                 same_segment_groups)
from experiments.experiment_utils import (_run_relative_filter, load_trial,
                                          measure_world_frame_gravity, pipeline_constants,
                                          run_tracked_grid, run_window)
from experiments.global_assumptions import (DATASETS, DatasetSpec, build_name, enumerate_trials,
                                            get_dataset, orphaned_trials, subjects_of)
from experiments.inertial_joint_center import DEFAULT_CONFIG as INERTIAL_CONFIG
from experiments.inertial_joint_center import fit as inertial_fit
from experiments.inertial_joint_center import sample_mask as inertial_sample_mask
from experiments.joint_center import MIN_FIT_FRAMES
from src.toolchest.PlateTrial import PlateTrial

EXPERIMENT_NAME = "sensor_placement"
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# The placement label given to a segment that carries exactly one sensor — the pelvis and the
# feet. Not 'Mid': their plate names end '_M' because the reader needed one suffix, but there is
# no High or Low beside them and calling it Mid would invite a reader to compare that cell
# against a thigh's Mid as though the two meant the same thing. 'Only' says the substitution this
# file is built on was not available on that side of the joint.
SINGLE_PLACEMENT = 'Only'

# projection -> how the joint centre used by the projection is obtained. 'none' does not project.
# Ordered worst-informed to best-informed, which is also the order every figure and table uses.
PROJECTIONS = ('none', 'inertial', 'mocap')

MAG_MODES = ('on', 'off')

# The full arm set, as (projection, mag_mode). Named centrally so the tables, the report and the
# plotting module cannot drift apart on what an arm is called.
ARMS: Tuple[Tuple[str, str], ...] = tuple((projection, mag)
                                          for projection in PROJECTIONS for mag in MAG_MODES)

# The arm every "how much does placement cost" headline is quoted from, and the one the paper's
# method corresponds to: projected against a marker joint centre, magnetometer on.
PRIMARY_ARM = ('mocap', 'on')

# The arm the DEPLOYMENT answer is quoted from. Kept separate from PRIMARY_ARM because the two
# genuinely disagree and reporting one number would hide which world it belongs to.
DEPLOYED_ARM = ('inertial', 'on')

# Samples discarded from the head of the record before the 'steady' scalars, in seconds.
#
# The filter is seeded from the marker orientation at t=0 with `init_orientation_std` = 0.1 deg,
# so the burn-in that used to dominate pooled RMSE (P = eye(6), ~57 deg/axis) is gone and this is
# no longer load-bearing. It is kept, and reported beside the full-record scalars rather than
# instead of them, because the fix lives in a filter constant that a future tuning change could
# undo silently — and because a placement with a large lever arm has a larger initial
# accelerometer disagreement, so if a startup transient ever came back it would come back
# UNEVENLY ACROSS THE CELLS THIS FILE COMPARES. `steady_minus_full_deg` is that check, per cell.
BURN_IN_S = 20.0

# Window over which a constant rotation is removed to separate slow heading offset from fast
# tracking error. 5 s is a few gait cycles: long enough that the removal cannot absorb the
# within-stride error the metric is meant to keep, short enough to track a drift that takes tens
# of seconds to develop. The same 5 s the mag_on/mag_off window analysis used, so the two
# decompositions can be read against each other.
DETREND_WINDOW_S = 5.0

# A cell needs this many scoreable frames to be worth a row. Below it the quantiles are noise and
# the detrending has fewer windows than parameters.
MIN_SCORED_FRAMES = 200

# Storage decimation for placement_samples ONLY; every scalar in placement_stats is computed on
# the full scoreable record. 25 at 40 Hz is 1.6 Hz, which is a shape for a time-series panel and
# not an analysis rate — the panels this feeds show minute-scale error trajectories.
SAMPLE_STRIDE = 25

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

TRIAL_TABLES = ('placement_stats', 'placement_geometry', 'placement_samples')

# The error scalars, and what each one is FOR. Kept as a table because the summary, the report and
# the plotting module all need the same list and the same units.
METRICS = {
    'rms_deg': 'full record, the number every other experiment here quotes',
    'median_deg': 'full record, robust to a single bad stretch',
    'p95_deg': 'full record, the tail that a clinician notices',
    'steady_rms_deg': f'after the first {BURN_IN_S:g} s',
    'detrended_rms_deg': f'constant rotation removed per {DETREND_WINDOW_S:g} s window',
    'offset_deg': 'the slow part: sqrt(rms^2 - detrended^2)',
}


def analysis_constants(dataset: str) -> Dict[str, object]:
    """Everything that can change a number in this experiment's outputs, for the manifest."""
    return {**pipeline_constants(), 'dataset': dataset,
            'projections': list(PROJECTIONS), 'mag_modes': list(MAG_MODES),
            'burn_in_s': BURN_IN_S, 'detrend_window_s': DETREND_WINDOW_S,
            'min_scored_frames': MIN_SCORED_FRAMES, 'min_fit_frames': MIN_FIT_FRAMES,
            'sample_stride': SAMPLE_STRIDE,
            'inertial_config': list(INERTIAL_CONFIG)}

# ==============================================================================
# The placement grid
# ==============================================================================


@dataclass(frozen=True)
class PlacementPair:
    """One cell of the substitution grid: a joint, and which sensor stands for each segment."""
    joint: str                  # anatomical, e.g. 'R_Knee' — NOT the spec's variant key
    parent_sensor: str
    child_sensor: str
    parent_placement: str       # 'High' | 'Mid' | 'Low' | SINGLE_PLACEMENT
    child_placement: str

    @property
    def pair(self) -> str:
        """'High-Low', for a label and for grouping."""
        return f"{self.parent_placement}-{self.child_placement}"

    @property
    def matched(self) -> bool:
        """Both sensors at the same height, which is what a careful two-IMU setup would do.

        True by definition where one side of the joint has no choice: the pelvis and the feet
        carry one sensor, so 'Only-High' is as matched as that joint can be, and calling it
        mismatched would put every hip and ankle cell in the mismatched bucket and make the
        matched/mismatched contrast a knee-vs-everything-else contrast instead.
        """
        return (self.parent_placement == self.child_placement
                or SINGLE_PLACEMENT in (self.parent_placement, self.child_placement))

    def key(self) -> Dict[str, object]:
        return {'joint': self.joint, 'pair': self.pair,
                'parent_sensor': self.parent_sensor, 'child_sensor': self.child_sensor,
                'parent_placement': self.parent_placement,
                'child_placement': self.child_placement, 'matched': self.matched}


def placement_options(spec: DatasetSpec) -> Dict[str, List[Tuple[str, str]]]:
    """{sensor: [(placement, sensor on the same segment), ...]}, ordered proximal to distal.

    A sensor whose segment carries only it maps to a single (SINGLE_PLACEMENT, itself) entry, so
    every caller can iterate the same structure without asking which segments are special.

    Built off `same_segment_groups`, which reads the spec's DISPLAY names rather than IMoVE's
    private '<SEGMENT>_<H|M|L>' plate convention. That is the whole reason this file works on a
    spec rather than on the plate names: a future dataset that labels placements the same way
    needs no change here, and a dataset with one sensor per segment produces an empty grid and is
    refused at the CLI instead of silently reporting a self-comparison as a placement effect.
    """
    groups = same_segment_groups(spec)
    options: Dict[str, List[Tuple[str, str]]] = {}
    for display, sensor in spec.segment_sensor.items():
        head, _, tail = display.rpartition(' ')
        segment = head if tail in PLACEMENT_TOKENS else display
        options[sensor] = groups.get(segment, [(SINGLE_PLACEMENT, sensor)])
    return options


def placement_pairs(spec: DatasetSpec,
                    plates: Optional[Dict[str, PlateTrial]] = None) -> List[PlacementPair]:
    """Every (joint, parent placement, child placement) cell, as the FULL cross product.

    Nine cells at a knee, three at a hip or an ankle. That is the deliberate difference from
    `global_assumptions._imove_joints`, which pairs placement-MATCHED sensors only and says so:
    for the question it asks, a mismatched pair "answers a question nobody asked". Here the
    mismatched cells ARE a question — a real subject taped by a real technician is not guaranteed
    to have both sensors at the same height, and whether that matters beyond the two lever arms
    it implies is section 5.

    Restricted to `spec.primary_joints`, the anatomical joints, because the spec's placement
    VARIANT keys ('R_Knee_H') are a different spelling of cells this grid already contains and
    would double-count them.

    `plates` narrows to what a given trial actually has; omitted, the full grid is returned, which
    is what the pair-table test and the CLI's --check both want.
    """
    options = placement_options(spec)
    pairs: List[PlacementPair] = []
    for joint in spec.primary_joints:
        parent_sensor, child_sensor = spec.joints[joint]
        for parent_placement, parent in options.get(parent_sensor, []):
            for child_placement, child in options.get(child_sensor, []):
                if plates is not None and (parent not in plates or child not in plates):
                    continue
                pairs.append(PlacementPair(joint, parent, child,
                                           parent_placement, child_placement))
    return pairs


def has_placement_contrast(pairs: Sequence[PlacementPair]) -> bool:
    """Does this grid actually substitute anything, or is every joint a single cell?

    False for IMoVE's long walks, which carry only the Mid sensor of each cluster segment. Every
    number this experiment produces on such a trial would be a one-cell "spread" of exactly zero,
    which is not a measurement of anything — so the trial is skipped, named and counted rather
    than contributing rows that look like a finding.
    """
    per_joint: Dict[str, int] = {}
    for pair in pairs:
        per_joint[pair.joint] = per_joint.get(pair.joint, 0) + 1
    return any(count > 1 for count in per_joint.values())

# ==============================================================================
# Storage
# ==============================================================================


def dataset_dir(dataset: str) -> Path:
    return EXPERIMENT_DIR / dataset


def trial_table_path(dataset: str, subject: str, trial: str, table: str) -> Path:
    return dataset_dir(dataset) / subject / trial / f"{table}.parquet"


def statistics_path(dataset: str) -> Path:
    return paths.statistics_path(f"{EXPERIMENT_NAME}_{dataset}")


def _save(df: pd.DataFrame, path: Path, dataset: str, **manifest_extra) -> None:
    paths.ensure_parent(path)
    df.to_parquet(path, engine='pyarrow', index=False)
    paths.write_manifest(path, constants=analysis_constants(dataset),
                         experiment=EXPERIMENT_NAME, n_rows=len(df), **manifest_extra)


def load_trial_table(dataset: str, table: str,
                     row_keys: Optional[Sequence[Tuple[str, str]]] = None) -> pd.DataFrame:
    """Every per-trial `table` under this dataset, concatenated, with subject/trial columns.

    Missing files are skipped rather than raising: a trial can legitimately have no rows (one
    placement per segment, or no joint with enough valid frames), and the pooled report is over
    whatever succeeded.
    """
    keys = list(row_keys) if row_keys is not None else enumerate_trials(build_name(dataset))
    frames = []
    for subject, trial in keys:
        path = trial_table_path(dataset, subject, trial, table)
        if not path.exists():
            continue
        frame = pd.read_parquet(path, engine='pyarrow')
        if frame.empty:
            continue
        frames.append(frame.assign(subject=subject, trial=trial))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ==============================================================================
# Error metrics
# ==============================================================================


def error_series(estimate: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(error angle in degrees, the error rotation stack) for two (N, 3, 3) joint-rotation series.

    The error rotation is reference^T estimate — a rotation in the CHILD SENSOR's frame, which is
    why only its angle is ever compared across placements (see the module docstring). It is
    returned alongside the angle because the detrending needs the rotation, not the scalar: the
    mean of a series of angles is not the angle of their mean rotation.
    """
    error = np.einsum('tji,tjk->tik', reference, estimate)
    angle = np.degrees(np.linalg.norm(Rotation.from_matrix(error).as_rotvec(), axis=1))
    return angle, error


def detrended_deg(error: np.ndarray, index: np.ndarray, fs: float,
                  window_s: float = DETREND_WINDOW_S) -> np.ndarray:
    """Angular error with the best constant rotation removed, window by window.

    Splits `index` — the scoreable samples, in order — into consecutive blocks of `window_s`,
    removes each block's chordal mean rotation, and returns what is left. What survives is the
    error that moves WITHIN a window; what was removed is a slow heading or tilt offset.

    The split is on position within `index`, not on wall-clock time, so a window never spans a
    gap in the mocap: two samples either side of a 40 s dropout would otherwise be averaged into
    one "constant" offset that neither of them had.
    """
    if len(index) == 0:
        return np.array([])
    block_size = max(3, int(round(window_s * fs)))
    parts = []
    for start in range(0, len(index), block_size):
        block = error[index[start:start + block_size]]
        if len(block) < 3:
            continue
        mean = chordal_mean(block)
        deviation = np.linalg.norm(
            Rotation.from_matrix(np.einsum('ij,njk->nik', mean.T, block)).as_rotvec(), axis=1)
        parts.append(np.degrees(deviation))
    return np.concatenate(parts) if parts else np.array([])


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2))) if len(values) else np.nan


def score(angle: np.ndarray, error: np.ndarray, index: np.ndarray, timestamps: np.ndarray,
          fs: float) -> Dict[str, float]:
    """Every scalar in METRICS, for one cell, over the scoreable samples `index`."""
    scored = angle[index]
    steady = index[timestamps[index] >= timestamps[index[0]] + BURN_IN_S]
    detrended = detrended_deg(error, index, fs)

    rms = _rms(scored)
    detrended_rms = _rms(detrended)
    return {
        'n_scored': int(len(index)),
        'scored_s': float(len(index) / fs),
        'rms_deg': rms,
        'median_deg': float(np.median(scored)),
        'p95_deg': float(np.percentile(scored, 95)),
        'steady_rms_deg': _rms(angle[steady]) if len(steady) >= MIN_SCORED_FRAMES else np.nan,
        'detrended_rms_deg': detrended_rms,
        # The slow component, by Pythagoras on the two rms values. Clipped at zero because the
        # detrended series is a per-window residual and nothing forbids it from exceeding the
        # total by a hair on a cell whose windows are few; a negative "offset" would be reported
        # as a NaN-free number that cannot exist.
        'offset_deg': float(np.sqrt(max(rms ** 2 - detrended_rms ** 2, 0.0)))
        if np.isfinite(rms) and np.isfinite(detrended_rms) else np.nan,
    }

# ==============================================================================
# Geometry: what differs between two cells, measured
# ==============================================================================


def _up_vector(plates: Dict[str, PlateTrial]) -> np.ndarray:
    """The world frame's up axis, from the accelerometers rather than from a constant.

    `measure_world_frame_gravity` averages the world-frame accelerometer reading over the trial,
    which is +g up in the specific-force convention. Measured rather than declared because the
    datasets disagree — Al Borno and IMoVE's Motive export are Y-up, the biplane tree is Z-up —
    and a hardcoded axis would silently turn `sensor_height_m` into a horizontal coordinate.
    """
    gravity = measure_world_frame_gravity(plates)
    norm = np.linalg.norm(gravity)
    return gravity / norm if norm > 0 else np.array([0.0, 1.0, 0.0])


def _plate_diagnostics(dataset: str, subject: str, trial: str) -> Dict[str, Dict[str, object]]:
    """The build's own per-plate record for this trial, or {} if it has no manifest.

    Read rather than recomputed. `sensor_offset_used_fallback` and `residual_fraction` are
    DECISIONS the build made and wrote down — whether this sensor's cluster-to-IMU offset was
    fitted or defaulted, and how much of its gyro the sensor-to-segment rotation failed to
    explain — and recomputing them here would be a second implementation of the build's logic
    that nobody diffs against the first.
    """
    manifest = paths.read_manifest(paths.cached_trial_path(build_name(dataset), subject, trial))
    return ((manifest or {}).get('diagnostics') or {}).get('plates') or {}


def sensor_rows(plates: Dict[str, PlateTrial], diagnostics: Dict[str, Dict[str, object]],
                up: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Per-sensor covariates: height, excitation, and the two build-quality flags.

    `gyro_rms_deg_s` is the CONTROL described in the module docstring. Angular velocity is
    position-invariant on a rigid body, so three placements on one segment must read the same
    value; a spread here means the plates are not on one segment and every lever-arm statement
    below it is void.
    """
    rows = {}
    for name, plate in plates.items():
        valid = np.asarray(plate.valid, dtype=bool)
        positions = np.asarray(plate.world_trace.positions)
        record = diagnostics.get(name, {})
        rows[name] = {
            'sensor_height_m': float(np.mean(positions[valid] @ up)) if valid.any() else np.nan,
            'gyro_rms_deg_s': float(np.degrees(
                np.sqrt(np.mean(np.sum(np.asarray(plate.imu_trace.gyro) ** 2, axis=1))))),
            'acc_rms': float(np.sqrt(np.mean(
                np.sum(np.asarray(plate.imu_trace.acc) ** 2, axis=1)))),
            'align_residual_fraction': float(record.get('residual_fraction', np.nan)),
            'offset_fallback': bool(record.get('sensor_offset_used_fallback', 0.0)),
            'offset_mm': float(np.linalg.norm(record['sensor_offset_mm']))
            if record.get('sensor_offset_mm') is not None else np.nan,
        }
    return rows


def segment_gyro_spread(sensors: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """{segment: max - min of |omega| rms over the sensors mounted on it}, in deg/s.

    THE CONTROL the whole experiment rests on. Angular velocity is position-invariant on a rigid
    body, so three placements 7-19 cm apart on one thigh must read the same |omega| whatever their
    lever arms are. A spread here is not a small imperfection to note — it means the three plates
    are not describing one rigid segment, and every lever-arm statement in this file would be
    attributing to position something that is not position.

    Measured across the placements of ONE segment, which is the only comparison in which the
    invariance holds. The parent-to-child spread across a joint is a different number entirely (a
    thigh and a shank genuinely do rotate at different rates) and reading it as this control is
    the mistake this function exists to make impossible.
    """
    per_segment: Dict[str, List[float]] = {}
    for name, row in sensors.items():
        per_segment.setdefault(_segment_of(name), []).append(row['gyro_rms_deg_s'])
    return {segment: float(max(values) - min(values)) if len(values) > 1 else 0.0
            for segment, values in per_segment.items()}


def pair_geometry(pair: PlacementPair, plates: Dict[str, PlateTrial], valid: np.ndarray,
                  sensors: Dict[str, Dict[str, float]], offsets: Dict[str, np.ndarray],
                  gyro_spread: Dict[str, float]) -> Dict[str, object]:
    """One row of placement_geometry: the lever arms and everything that could confound them."""
    parent, child = plates[pair.parent_sensor], plates[pair.child_sensor]
    parent_sensor, child_sensor = sensors[pair.parent_sensor], sensors[pair.child_sensor]

    row: Dict[str, object] = {
        **pair.key(),
        'lever_parent_mm': float(np.linalg.norm(offsets['parent']) * 1000.0),
        'lever_child_mm': float(np.linalg.norm(offsets['child']) * 1000.0),
        'joint_fit_residual_mm': float(offsets['residual_mm']),
        'n_valid': int(valid.sum()),
    }
    # The two arms enter the projection independently, so both a sum and a max are wanted: the
    # sum is what a "total lever arm across the joint" story predicts, the max is what a
    # weakest-link story predicts. Reported side by side rather than choosing, since which one
    # explains the error better is itself a finding (section 2).
    row['lever_sum_mm'] = row['lever_parent_mm'] + row['lever_child_mm']
    row['lever_max_mm'] = max(row['lever_parent_mm'], row['lever_child_mm'])

    for role, sensor, plate in (('parent', parent_sensor, parent), ('child', child_sensor, child)):
        for column, value in sensor.items():
            row[f'{role}_{column}'] = value
    # Carried per pair rather than per segment so the control travels with every cell it could
    # invalidate; both sides are named because a cell is only as good as the worse of them.
    row['parent_segment_gyro_spread_deg_s'] = gyro_spread.get(
        _segment_of(pair.parent_sensor), np.nan)
    row['child_segment_gyro_spread_deg_s'] = gyro_spread.get(
        _segment_of(pair.child_sensor), np.nan)

    inertial = offsets.get('inertial')
    row['inertial_converged'] = bool(inertial is not None and inertial['converged'])
    row['inertial_error_parent_mm'] = (float(np.linalg.norm(
        inertial['x'][:3] - offsets['parent']) * 1000.0) if inertial is not None else np.nan)
    row['inertial_error_child_mm'] = (float(np.linalg.norm(
        inertial['x'][3:] - offsets['child']) * 1000.0) if inertial is not None else np.nan)
    row['inertial_lever_parent_mm'] = (float(np.linalg.norm(inertial['x'][:3]) * 1000.0)
                                       if inertial is not None else np.nan)
    row['inertial_lever_child_mm'] = (float(np.linalg.norm(inertial['x'][3:]) * 1000.0)
                                      if inertial is not None else np.nan)
    return row

# ==============================================================================
# One cell
# ==============================================================================


def pair_offsets(pair: PlacementPair, plates: Dict[str, PlateTrial], valid: np.ndarray,
                 want_inertial: bool) -> Optional[Dict[str, object]]:
    """The joint centre for one cell, from the markers and (optionally) from the IMUs alone.

    Fitted per CELL, not per joint. That is the point of the experiment: the offset from
    THIGH_R_H to the knee is not the offset from THIGH_R_L to the knee, and a shared per-joint
    fit would hand every placement the same lever arm and erase the effect being measured.

    None when the marker fit has too few valid frames, which takes the whole cell out rather than
    letting a projection run against an offset nothing supports.
    """
    parent, child = plates[pair.parent_sensor], plates[pair.child_sensor]
    if valid.sum() < MIN_FIT_FRAMES:
        return None
    r_parent, r_child, residual = parent.world_trace.get_joint_center(
        child.world_trace, valid, min_frames=MIN_FIT_FRAMES)
    offsets: Dict[str, object] = {
        'parent': r_parent, 'child': r_child,
        'residual_mm': float(np.nanmean(np.linalg.norm(residual, axis=1)) * 1000.0),
    }
    if want_inertial:
        # Cold start from zero, exactly as `inertial_joint_center.inertial_fits` does, because a
        # warm start from the mocap answer is not an estimate a deployed system could make. The
        # mask is the estimator's own excitation selection, not this file's scoring mask: it needs
        # no mocap, and restricting it to valid frames would flatter it.
        mask = inertial_sample_mask(parent, child)
        offsets['inertial'] = (inertial_fit(parent, child, mask, np.zeros(6), INERTIAL_CONFIG)
                               if mask.sum() >= MIN_FIT_FRAMES else None)
    return offsets


def _projected(plate: PlateTrial, offset: np.ndarray) -> PlateTrial:
    """A copy of `plate` whose accelerometer has been transported to `offset`.

    Done here rather than by `_run_relative_filter(project=True)` so the offset is an ARGUMENT:
    that path refits the marker joint centre internally and there is no way to hand it the
    inertial estimate. With the marker offsets this reproduces that path exactly, which
    test/TestSensorPlacement.py asserts rather than assumes.
    """
    projected = plate.copy()
    projected.imu_trace = plate.project_imu_trace(offset)
    return projected


def run_arm(pair: PlacementPair, plates: Dict[str, PlateTrial], offsets: Dict[str, object],
            projection: str, mag: str) -> Optional[np.ndarray]:
    """The filter's joint-rotation estimate for one cell and one arm, or None if unavailable."""
    parent, child = plates[pair.parent_sensor], plates[pair.child_sensor]
    if projection == 'none':
        pass
    elif projection == 'mocap':
        parent = _projected(parent, offsets['parent'])
        child = _projected(child, offsets['child'])
    elif projection == 'inertial':
        inertial = offsets.get('inertial')
        if inertial is None:
            return None
        parent = _projected(parent, inertial['x'][:3])
        child = _projected(child, inertial['x'][3:])
    else:
        raise ValueError(f"Unknown projection '{projection}'")
    # project=False throughout: the transport above has already happened, and letting the filter
    # project again would apply the lever-arm term twice.
    return _run_relative_filter(parent, child, project=False, mag_mode=mag)

# ==============================================================================
# One trial
# ==============================================================================


def compute_trial(plates: Dict[str, PlateTrial], spec: DatasetSpec, dataset: str, subject: str,
                  trial: str, tables: Sequence[str] = TRIAL_TABLES,
                  stride: int = SAMPLE_STRIDE) -> Dict[str, pd.DataFrame]:
    """Every table for one trial: the whole placement grid crossed with every arm."""
    wanted = set(tables)
    pairs = placement_pairs(spec, plates)
    if not has_placement_contrast(pairs):
        return {}

    window = run_window(plates)
    plates = {name: plate[window] for name, plate in plates.items()}
    any_plate = next(iter(plates.values()))
    timestamps = np.asarray(any_plate.imu_trace.timestamps)
    fs = 1.0 / float(np.mean(np.diff(timestamps)))
    up = _up_vector(plates)
    sensors = sensor_rows(plates, _plate_diagnostics(dataset, subject, trial), up)
    gyro_spread = segment_gyro_spread(sensors)
    want_inertial = 'inertial' in {projection for projection, _ in ARMS}

    stats: List[Dict[str, object]] = []
    geometry: List[Dict[str, object]] = []
    samples: List[pd.DataFrame] = []

    for pair in pairs:
        parent, child = plates[pair.parent_sensor], plates[pair.child_sensor]
        # A joint angle inherits the worse of its two segments' validity: one corrupt plate takes
        # the frame out of every cell it participates in. Same rule as
        # `experiment_utils._joint_valid`, restated here because the pair table is this file's
        # own rather than JOINTS'.
        valid = np.asarray(parent.valid, dtype=bool) & np.asarray(child.valid, dtype=bool)
        index = np.flatnonzero(valid)
        if len(index) < MIN_SCORED_FRAMES:
            continue
        offsets = pair_offsets(pair, plates, valid, want_inertial)
        if offsets is None:
            continue
        if 'placement_geometry' in wanted:
            geometry.append(pair_geometry(pair, plates, valid, sensors, offsets, gyro_spread))
        if not ({'placement_stats', 'placement_samples'} & wanted):
            continue

        reference = np.einsum('tji,tjk->tik', parent.world_trace.rotations,
                              child.world_trace.rotations)
        for projection, mag in ARMS:
            estimate = run_arm(pair, plates, offsets, projection, mag)
            if estimate is None:
                continue
            angle, error = error_series(estimate, reference)
            row = {**pair.key(), 'projection': projection, 'mag': mag,
                   **score(angle, error, index, timestamps, fs)}
            row['steady_minus_full_deg'] = row['steady_rms_deg'] - row['rms_deg']
            stats.append(row)
            if 'placement_samples' in wanted:
                keep = index[::stride]
                samples.append(pd.DataFrame({
                    'joint': pair.joint, 'pair': pair.pair, 'projection': projection, 'mag': mag,
                    'timestamp': timestamps[keep], 'error_deg': angle[keep]}))

    out: Dict[str, pd.DataFrame] = {}
    if 'placement_stats' in wanted and stats:
        out['placement_stats'] = pd.DataFrame(stats)
    if 'placement_geometry' in wanted and geometry:
        out['placement_geometry'] = pd.DataFrame(geometry)
    if 'placement_samples' in wanted and samples:
        out['placement_samples'] = pd.concat(samples, ignore_index=True)
    return out


def _load_spec_plates(subject: str, trial: str, dataset: str,
                      spec: DatasetSpec) -> Dict[str, PlateTrial]:
    """The trial's plates, narrowed to the sensors this dataset's spec names."""
    plates = load_trial(subject, trial, dataset=build_name(dataset))
    selected = {sensor: plate for sensor, plate in plates.items()
                if sensor in set(spec.segment_sensor.values())}
    if not selected:
        raise ValueError(f"{dataset}/{subject}/{trial}: none of the spec's sensors are in this "
                         f"trial ({sorted(plates)}).")
    return selected


def _trial_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                  dataset: str = 'imove', tables: Sequence[str] = TRIAL_TABLES,
                  stride: int = SAMPLE_STRIDE) -> None:
    """One process per trial: the whole grid, written to that trial's directory."""
    subject, trial = row_key
    spec = get_dataset(dataset)
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()
    try:
        plates = _load_spec_plates(subject, trial, dataset, spec)
        computed = compute_trial(plates, spec, dataset, subject, trial, tables, stride)
        if not computed:
            # Not a failure: a trial with one sensor per segment has nothing to substitute. Said
            # in the grid so the count of skipped trials is visible rather than inferred from a
            # short summary.
            shared_state[(row_key, stage)] = "Skipped"
            return None
        for table, frame in computed.items():
            _save(frame, trial_table_path(dataset, subject, trial, table), dataset,
                  subject=subject, trial=trial, table=table)
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success"
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
    return None

# ==============================================================================
# Pooled summary
# ==============================================================================

SUMMARY_KEYS = ['dataset', 'joint', 'pair', 'projection', 'mag', 'metric']


def summarize(dataset: str, stats: pd.DataFrame) -> pd.DataFrame:
    """Pooled quantiles per (joint, placement pair, arm, metric), over every trial.

    Pooled over TRIALS, each of which contributes one scalar per cell, rather than over samples:
    a 60 s trial and a 300 s trial would otherwise weight the cohort by their length, and the
    substitution being measured is per trial.
    """
    if stats.empty:
        return pd.DataFrame()
    rows = []
    for metric in METRICS:
        if metric not in stats.columns:
            continue
        grouped = stats.groupby(['joint', 'pair', 'projection', 'mag'], observed=True)[metric]
        described = grouped.agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
        quantiles = grouped.quantile(QUANTILES).unstack()
        quantiles.columns = [f"p{int(round(q * 100)):02d}" for q in QUANTILES]
        merged = described.merge(quantiles.reset_index(),
                                 on=['joint', 'pair', 'projection', 'mag'])
        merged.insert(0, 'dataset', dataset)
        merged.insert(len(merged.columns), 'metric', metric)
        merged['unit'] = 'deg'
        rows.append(merged.rename(columns={'count': 'n_trials'}))
    return pd.concat(rows, ignore_index=True)

# ==============================================================================
# Report
# ==============================================================================


def _header(number: object, title: str, subtitle: str) -> None:
    print(f"\n{'=' * 96}\n{number}. {title}\n   {subtitle}\n{'=' * 96}")


def _indent(table: pd.DataFrame, pad: str = '     ') -> str:
    """A DataFrame printed under a report heading, every line inside the section's margin."""
    return "\n".join(pad + line for line in table.to_string().split("\n"))


def _arm(stats: pd.DataFrame, arm: Tuple[str, str]) -> pd.DataFrame:
    projection, mag = arm
    return stats[(stats.projection == projection) & (stats.mag == mag)]


def _arm_label(arm: Tuple[str, str]) -> str:
    projection, mag = arm
    return f"{projection} projection, mag {mag}"


def _cell_index(frame: pd.DataFrame) -> List[str]:
    """The columns identifying one TRIAL x JOINT, which is the unit every paired test blocks on."""
    return ['subject', 'trial', 'joint']


def _paired(frame: pd.DataFrame, metric: str, group_column: str, left: str, right: str
            ) -> Tuple[int, float, float]:
    """(n, median of right - left, Wilcoxon p) over cells present in both groups.

    Paired on (subject, trial, joint) because the trials differ enormously — a 30 s sit-to-stand
    and a 10 minute walk are not exchangeable — and an unpaired comparison of two placements would
    mostly measure which trials happened to have both.
    """
    index = _cell_index(frame)
    pivot = frame.pivot_table(index=index, columns=group_column, values=metric, observed=True)
    if left not in pivot.columns or right not in pivot.columns:
        return 0, np.nan, np.nan
    both = pivot[[left, right]].dropna()
    if len(both) < 2:
        return len(both), np.nan, np.nan
    difference = both[right] - both[left]
    if np.allclose(difference, 0.0):
        return len(both), 0.0, 1.0
    return len(both), float(np.median(difference)), float(wilcoxon(difference).pvalue)


def _spread_per_cell(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (subject, trial, joint, arm): the best and worst placement and the gap between them.

    The gap is computed WITHIN a cell before anything is pooled. Pooling first and then taking a
    max-minus-min over cohort medians is a different and much smaller number — it averages away
    the trials where placement mattered — and it is the mistake this function exists to prevent.
    """
    index = _cell_index(frame) + ['projection', 'mag']
    grouped = frame.groupby(index, observed=True)[metric]
    out = grouped.agg(['min', 'max', 'median', 'count']).reset_index()
    out['spread_deg'] = out['max'] - out['min']
    out['ratio'] = out['max'] / out['min'].replace(0.0, np.nan)
    best = frame.loc[grouped.idxmin()].set_index(index)['pair']
    worst = frame.loc[grouped.idxmax()].set_index(index)['pair']
    out = out.merge(best.rename('best_pair'), left_on=index, right_index=True)
    out = out.merge(worst.rename('worst_pair'), left_on=index, right_index=True)
    return out[out['count'] > 1]


def report_headline(spec: DatasetSpec, stats: pd.DataFrame, metric: str) -> None:
    _header(1, "WHAT PLACEMENT COSTS",
            f"{metric}: best against worst placement, WITHIN each trial x joint, then pooled.")
    spread = _spread_per_cell(stats, metric)
    if spread.empty:
        print("   No cell has more than one placement. Nothing to compare.")
        return
    for projection, mag in ARMS:
        arm = spread[(spread.projection == projection) & (spread.mag == mag)]
        if arm.empty:
            continue
        print(f"\n   {_arm_label((projection, mag))}")
        table = arm.groupby('joint', observed=True).agg(
            n=('spread_deg', 'size'), best=('min', 'median'), worst=('max', 'median'),
            spread=('spread_deg', 'median'), ratio=('ratio', 'median')).round(2)
        # WHICH placement won, not just by how much — the actionable half of the answer, and it
        # is not the same placement at every joint, which is the finding underneath it.
        for column, source in (('best_at', 'best_pair'), ('worst_at', 'worst_pair')):
            modal = arm.groupby('joint', observed=True)[source].agg(
                lambda values: f"{values.mode().iloc[0]} ({values.eq(values.mode().iloc[0]).mean():.0%})")
            table[column] = modal
        table = table.loc[[j for j in spec.primary_joints if j in table.index]]
        print(_indent(table))
    print("\n   'spread' is the median over trials of (worst placement - best placement) in the "
          "\n   same trial, so it is what one subject would see by moving the sensor, not a "
          "\n   difference of cohort averages.")


def report_lever_arm(stats: pd.DataFrame, geometry: pd.DataFrame, metric: str) -> None:
    _header(2, "IS IT THE LEVER ARM?",
            "Error against distance from sensor to joint centre, within each trial x joint.")
    merged = stats.merge(geometry, on=['subject', 'trial', 'joint', 'pair'], suffixes=('', '_g'))
    if merged.empty:
        print("   No geometry rows matched the stats rows.")
        return
    print(f"\n   {'arm':32s} {'n cells':>8s} {'rho(sum)':>9s} {'rho(max)':>9s} "
          f"{'deg per 100 mm':>15s}")
    for projection, mag in ARMS:
        arm = merged[(merged.projection == projection) & (merged.mag == mag)]
        if len(arm) < 10:
            continue
        # Spearman WITHIN each trial x joint, then pooled: across joints the lever arm and the
        # error both vary for reasons that have nothing to do with placement (a hip is not an
        # ankle), and a pooled correlation would mostly measure that.
        rho_sum, rho_max, slopes, n = [], [], [], 0
        for _, cell in arm.groupby(_cell_index(arm), observed=True):
            if len(cell) < 3 or cell[metric].isna().all():
                continue
            n += 1
            rho_sum.append(cell['lever_sum_mm'].corr(cell[metric], method='spearman'))
            rho_max.append(cell['lever_max_mm'].corr(cell[metric], method='spearman'))
            fit = np.polyfit(cell['lever_sum_mm'], cell[metric], 1)
            slopes.append(fit[0] * 100.0)
        if not n:
            continue
        print(f"   {_arm_label((projection, mag)):32s} {n:8d} {np.nanmedian(rho_sum):9.2f} "
              f"{np.nanmedian(rho_max):9.2f} {np.nanmedian(slopes):15.2f}")
    print("\n   A projection that works removes the lever arm from the measurement, so rho should "
          "\n   fall toward zero from 'none' to 'mocap'. It is the CHANGE across the rows that is "
          "\n   the result; the absolute rho of an unprojected arm is not a surprise.")


def report_sign_flip(stats: pd.DataFrame, geometry: pd.DataFrame, metric: str) -> None:
    _header(3, "THE FALSIFICATION TEST: DOES THE RANKING REVERSE?",
            "A thigh sensor is far from one joint exactly when it is near the other.")
    merged = stats.merge(geometry, on=['subject', 'trial', 'joint', 'pair'], suffixes=('', '_g'))
    if merged.empty:
        print("   Nothing to test.")
        return
    print(f"\n   {'arm':32s} {'segments':>9s} {'reversed':>9s} {'rate':>7s} {'lever check':>12s}")
    for projection, mag in ARMS:
        arm = merged[(merged.projection == projection) & (merged.mag == mag)]
        reversed_count, total, lever_reversed = 0, 0, 0
        for _, cell in arm.groupby(['subject', 'trial'], observed=True):
            for _, proximal, distal in _segment_roles(cell):
                ranking = _placement_ranking(proximal, metric)
                distal_ranking = _placement_ranking(distal, metric)
                if ranking is None or distal_ranking is None:
                    continue
                total += 1
                reversed_count += int(_is_reversed(ranking, distal_ranking))
                # The same test on the LEVER ARM itself, which is geometry and not a measurement.
                # It should reverse essentially always; a rate below ~100% here means the segment
                # roles were assembled wrongly and the error rate beside it means nothing.
                lever = _placement_ranking(proximal, 'lever_child_mm')
                distal_lever = _placement_ranking(distal, 'lever_parent_mm')
                lever_reversed += int(lever is not None and distal_lever is not None
                                      and _is_reversed(lever, distal_lever))
        if total:
            print(f"   {_arm_label((projection, mag)):32s} {total:9d} {reversed_count:9d} "
                  f"{reversed_count / total:7.0%} {lever_reversed / total:12.0%}")
    print("\n   A lever-arm mechanism predicts a high reversal rate on the unprojected arm: the "
          "\n   best placement for the hip should be the worst for the knee. A sensor-quality "
          "\n   mechanism predicts ~0%, because a bad sensor is bad at both ends. The last "
          "\n   column is the same test on the geometry alone and is the control for both.")


def _is_reversed(first: Sequence[str], second: Sequence[str]) -> bool:
    """Do two best-to-worst rankings disagree at BOTH ends?

    Both ends rather than any disagreement: the middle placement swapping with an end is one
    noisy comparison, while the best becoming the worst is the reversal the lever-arm mechanism
    predicts and is not reachable by a single rank error.
    """
    return first[0] != second[0] and first[-1] != second[-1] and first[0] == second[-1]


def _segment_roles(cell: pd.DataFrame) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """(segment, rows where it is the DISTAL sensor, rows where it is the PROXIMAL one).

    A thigh is the child of the hip and the parent of the knee; a shank is the child of the knee
    and the parent of the ankle. So the same segment's three placements are ranked twice against
    two different joints, and the joint sits at the opposite end of the segment each time — which
    is exactly the substitution the sign-flip test needs.

    Derived from the sensor names in the rows rather than from an anatomy table, so it needs no
    per-dataset configuration beyond the pair grid that produced them. Both frames carry a
    `placement` column naming THIS segment's sensor, whichever role it played.
    """
    parent_segment = cell.parent_sensor.map(_segment_of)
    child_segment = cell.child_sensor.map(_segment_of)
    out = []
    for segment in sorted(set(parent_segment) & set(child_segment)):
        as_child = cell[child_segment == segment]
        as_parent = cell[parent_segment == segment]
        if as_child.empty or as_parent.empty:
            continue
        out.append((segment,
                    as_child.assign(placement=as_child.child_placement),
                    as_parent.assign(placement=as_parent.parent_placement)))
    return out


def _segment_of(sensor: str) -> str:
    """'THIGH_R_H' -> 'THIGH_R'; a sensor with no placement suffix maps to itself."""
    head, _, tail = str(sensor).rpartition('_')
    return head if head and tail in ('H', 'M', 'L') else str(sensor)


def _placement_ranking(rows: pd.DataFrame, metric: str) -> Optional[List[str]]:
    """Placements of `rows` ordered best (lowest error) to worst, or None if fewer than three.

    Averaged over the OTHER side's placements first, so the ranking is of this segment's sensor
    and not of whichever pairing it happened to appear in.
    """
    if 'placement' not in rows.columns or rows.empty:
        return None
    means = rows.groupby('placement', observed=True)[metric].mean().dropna()
    if len(means) < 3:
        return None
    return list(means.sort_values().index)


def report_projection(spec: DatasetSpec, stats: pd.DataFrame, metric: str) -> None:
    _header(4, "DOES THE PROJECTION MAKE PLACEMENT MOOT?",
            "The placement spread under each projection, and what it costs to have to estimate "
            "the joint centre.")
    spread = _spread_per_cell(stats, metric)
    if spread.empty:
        print("   Nothing to compare.")
        return
    for mag in MAG_MODES:
        arm = spread[spread.mag == mag]
        if arm.empty:
            continue
        print(f"\n   magnetometer {mag}")
        table = arm.pivot_table(index='joint', columns='projection', values='spread_deg',
                                aggfunc='median', observed=True)
        table = table[[p for p in PROJECTIONS if p in table.columns]].round(2)
        table = table.loc[[j for j in spec.primary_joints if j in table.index]]
        print(_indent(table))
        for reference, candidate in (('none', 'mocap'), ('none', 'inertial'),
                                     ('inertial', 'mocap')):
            n, delta, p = _paired(arm.rename(columns={'spread_deg': metric}), metric,
                                  'projection', reference, candidate)
            if n:
                print(f"     spread {candidate} - {reference}: {delta:+.2f} deg "
                      f"(n={n}, Wilcoxon p={p:.2g})")
    print("\n   'mocap' is the ceiling: the joint centre is as good as markers can make it. "
          "\n   'inertial' is what a deployed system can actually reach, and the gap between "
          "\n   them is the part of MAJIC's placement-independence that depends on mocap.")


def report_matched(stats: pd.DataFrame, metric: str) -> None:
    _header(5, "DOES IT MATTER THAT THE TWO SENSORS AGREE?",
            "Matched placements (High-High) against mismatched (High-Low), knees only.")
    knees = stats[stats.pair.str.count('-').eq(1)
                  & ~stats.pair.str.contains(SINGLE_PLACEMENT, regex=False)]
    if knees.empty:
        print("   No joint has a choice of placement on both sides.")
        return
    print(f"\n   {'arm':32s} {'n cells':>8s} {'matched':>9s} {'mismatched':>11s} {'delta':>8s} "
          f"{'p':>9s}")
    for projection, mag in ARMS:
        arm = knees[(knees.projection == projection) & (knees.mag == mag)]
        if arm.empty:
            continue
        cell = arm.groupby(_cell_index(arm) + ['matched'], observed=True)[metric].mean()
        cell = cell.unstack('matched').dropna()
        if cell.empty or True not in cell.columns or False not in cell.columns:
            continue
        difference = cell[False] - cell[True]
        p = wilcoxon(difference).pvalue if not np.allclose(difference, 0) else 1.0
        print(f"   {_arm_label((projection, mag)):32s} {len(cell):8d} {cell[True].median():9.2f} "
              f"{cell[False].median():11.2f} {difference.median():+8.2f} {p:9.2g}")
    print("\n   Both sensors' lever arms already enter the projection separately, so a mismatch "
          "\n   penalty BEYOND them would mean something the pairwise geometry does not capture.")


def report_magnetometer(stats: pd.DataFrame, geometry: pd.DataFrame, metric: str) -> None:
    _header(6, "THE MAGNETOMETER AND THE FLOOR",
            "The lab's magnetic distortion grows toward the floor, so a Low sensor should pay "
            "more for using it.")
    merged = stats.merge(geometry, on=['subject', 'trial', 'joint', 'pair'], suffixes=('', '_g'))
    if merged.empty:
        print("   Nothing to compare.")
        return
    index = _cell_index(merged) + ['pair', 'projection']
    pivot = merged.pivot_table(index=index, columns='mag', values=metric, observed=True).dropna()
    if pivot.empty or 'on' not in pivot.columns or 'off' not in pivot.columns:
        print("   Both magnetometer arms are needed and only one is present.")
        return
    benefit = (pivot['off'] - pivot['on']).rename('mag_benefit_deg').reset_index()
    heights = merged.groupby(index, observed=True)[
        ['parent_sensor_height_m', 'child_sensor_height_m']].mean().reset_index()
    benefit = benefit.merge(heights, on=index)
    benefit['mean_height_m'] = benefit[['parent_sensor_height_m',
                                        'child_sensor_height_m']].mean(axis=1)
    print(f"\n   {'projection':>12s} {'n cells':>8s} {'mag benefit deg':>16s} "
          f"{'rho(benefit, height)':>21s}")
    for projection in PROJECTIONS:
        rows = benefit[benefit.projection == projection]
        if len(rows) < 10:
            continue
        rho = rows['mean_height_m'].corr(rows['mag_benefit_deg'], method='spearman')
        print(f"   {projection:>12s} {len(rows):8d} {rows['mag_benefit_deg'].median():16.2f} "
              f"{rho:21.2f}")
    print("\n   Positive benefit means the magnetometer helped. A positive correlation with "
          "\n   height means it helped MORE the further the sensor sat from the floor, which is "
          "\n   the sign the floor-source finding predicts.")


def report_confounds(stats: pd.DataFrame, geometry: pd.DataFrame, metric: str) -> None:
    _header(7, "WHAT ELSE DIFFERS BETWEEN THESE CELLS",
            "Every alternative explanation for a placement effect, measured rather than argued.")
    if geometry.empty:
        print("   No geometry table.")
        return
    control = pd.concat([geometry['parent_segment_gyro_spread_deg_s'],
                         geometry['child_segment_gyro_spread_deg_s']]).dropna()
    reference = pd.concat([geometry['parent_gyro_rms_deg_s'],
                           geometry['child_gyro_rms_deg_s']]).dropna()
    print(f"\n   POSITION-INVARIANCE CONTROL. The placements of ONE segment must read the same "
          f"\n   |omega| however far apart they are. Spread within a segment: median "
          f"{control.median():.2f} deg/s,\n   p99 {control.quantile(0.99):.2f} deg/s, against a "
          f"signal of {reference.median():.0f} deg/s — "
          f"{control.median() / reference.median():.1%} of it.")

    print("\n   THE TAPED SENSORS. Mid is bolted to its marker cluster; High and Low are taped, "
          "\n   so their cluster-to-IMU offset is fitted per trial and sometimes falls back.")
    for role in ('parent', 'child'):
        column = f'{role}_offset_fallback'
        if column not in geometry.columns:
            continue
        rate = geometry.groupby(f'{role}_placement', observed=True)[column].mean()
        print(f"     {role} fallback rate by placement: "
              + ", ".join(f"{k} {v:.1%}" for k, v in rate.items()))

    print("\n   THE ALIGNMENT RESIDUAL. A mis-estimated sensor-to-segment rotation inflates a "
          "\n   placement's error without its position having anything to do with it.")
    merged = stats.merge(geometry, on=['subject', 'trial', 'joint', 'pair'], suffixes=('', '_g'))
    for projection, mag in (PRIMARY_ARM, ('none', 'on')):
        arm = merged[(merged.projection == projection) & (merged.mag == mag)]
        if arm.empty:
            continue
        worst = arm[['parent_align_residual_fraction',
                     'child_align_residual_fraction']].max(axis=1)
        rho = worst.corr(arm[metric], method='spearman')
        print(f"     {_arm_label((projection, mag)):32s} rho(worst residual fraction, "
              f"{metric}) = {rho:.2f}")
    for role in ('parent', 'child'):
        column = f'{role}_align_residual_fraction'
        if column in geometry.columns:
            by_placement = geometry.groupby(f'{role}_placement', observed=True)[column].median()
            print(f"     median {role} residual fraction by placement: "
                  + ", ".join(f"{k} {v:.3f}" for k, v in by_placement.items()))

    print("\n   THE INERTIAL JOINT CENTRE. How far the IMU-only offset lands from the marker one, "
          "\n   by placement — the estimator the deployable arm depends on.")
    for role in ('parent', 'child'):
        error_column, placement_column = f'inertial_error_{role}_mm', f'{role}_placement'
        if error_column not in geometry.columns:
            continue
        table = geometry.groupby(placement_column, observed=True)[error_column].median()
        print(f"     median {role} error: " + ", ".join(f"{k} {v:.0f} mm" for k, v in
                                                        table.items()))


def report_verdict(stats: pd.DataFrame, geometry: pd.DataFrame, metric: str) -> None:
    _header(8, "VERDICT", "What to tell someone about to tape an IMU to a limb.")
    spread = _spread_per_cell(stats, metric)
    if spread.empty:
        print("   Nothing to conclude.")
        return
    for arm in (('none', 'on'), DEPLOYED_ARM, PRIMARY_ARM):
        rows = spread[(spread.projection == arm[0]) & (spread.mag == arm[1])]
        if rows.empty:
            continue
        print(f"\n   {_arm_label(arm)}")
        print(f"     placement spread   {rows['spread_deg'].median():6.2f} deg median, "
              f"{rows['spread_deg'].quantile(0.95):6.2f} deg p95")
        print(f"     worst / best       {rows['ratio'].median():6.2f}x median")
        best = rows['best_pair'].value_counts(normalize=True)
        worst = rows['worst_pair'].value_counts(normalize=True)
        print(f"     best placement     {best.index[0]} in {best.iloc[0]:.0%} of cells "
              f"(pooled over joints, which do not agree — see section 1)")
        print(f"     worst placement    {worst.index[0]} in {worst.iloc[0]:.0%} of cells")

    primary = _arm(stats, PRIMARY_ARM)
    deployed = _arm(stats, DEPLOYED_ARM)
    if not primary.empty and not deployed.empty:
        merged = primary.merge(deployed, on=_cell_index(primary) + ['pair'],
                               suffixes=('_mocap', '_inertial'))
        if not merged.empty:
            gap = merged[f'{metric}_inertial'] - merged[f'{metric}_mocap']
            # Median AND tail, because they say different things and only one of them is the
            # answer. The typical cell barely notices which joint centre it was given; the tail
            # is the cells where the IMU-only fit failed, and those are the distal placements —
            # so the cost of losing the markers is not a level shift, it is a widening.
            print(f"\n   Cost of not having markers for the joint centre, per cell: "
                  f"{gap.median():+.2f} deg median, {gap.quantile(0.90):+.2f} deg p90, "
                  f"{gap.max():+.2f} deg worst,\n   over {len(merged)} cells. Worse in "
                  f"{(gap > 0).mean():.0%} of them.")


def print_report(spec: DatasetSpec, stats: pd.DataFrame, geometry: pd.DataFrame,
                 metric: str) -> None:
    print(f"\n{'#' * 96}")
    print(f"# SENSOR PLACEMENT — {spec.name} — {stats.subject.nunique()} subject(s), "
          f"{stats.groupby(['subject', 'trial'], observed=True).ngroups} trial(s), "
          f"{len(stats):,} cells")
    print(f"# metric: {metric} — {METRICS.get(metric, '')}")
    print(f"{'#' * 96}")
    report_headline(spec, stats, metric)
    report_lever_arm(stats, geometry, metric)
    report_sign_flip(stats, geometry, metric)
    report_projection(spec, stats, metric)
    report_matched(stats, metric)
    report_magnetometer(stats, geometry, metric)
    report_confounds(stats, geometry, metric)
    report_verdict(stats, geometry, metric)

# ==============================================================================
# CLI
# ==============================================================================


def select_trials(dataset: str, subjects: Optional[List[str]],
                  trials: Optional[List[str]]) -> List[Tuple[str, str]]:
    """The built trials matching the filters, or a ValueError naming what does not exist."""
    row_keys = enumerate_trials(build_name(dataset))
    if not row_keys:
        raise ValueError(f"No built trials under {paths.TRIALS_DIR / build_name(dataset)}. "
                         f"Run: python -m experiments.build_trials --dataset "
                         f"{build_name(dataset)}")
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
    parser.add_argument('--dataset', default='imove', choices=sorted(DATASETS))
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--trials', nargs='+', default=None)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--stride', type=int, default=SAMPLE_STRIDE,
                        help="Keep every Nth sample in placement_samples. Every scalar in "
                             "placement_stats is computed on the full scoreable record.")
    parser.add_argument('--metric', default='rms_deg', choices=sorted(METRICS),
                        help="Which error scalar the console report is built from. Every metric "
                             "is written to the tables regardless.")
    parser.add_argument('--only-tables', nargs='+', choices=TRIAL_TABLES,
                        default=list(TRIAL_TABLES), metavar='TABLE')
    parser.add_argument('--report-only', action='store_true',
                        help="Rebuild the summary and report from what is already on disk.")
    args = parser.parse_args()

    spec = get_dataset(args.dataset)
    grid = placement_pairs(spec)
    if not has_placement_contrast(grid):
        print(f"Error: {args.dataset} carries one sensor per segment, so there is no placement "
              f"to vary.\nThis experiment substitutes one sensor for another on the SAME segment "
              f"and scores the\ndifference; with nothing to substitute, every cell would be the "
              f"same cell. IMoVE is the\nonly dataset here with High/Mid/Low placements.")
        return 1

    try:
        row_keys = select_trials(args.dataset, args.subjects, args.trials)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    orphans = orphaned_trials(build_name(args.dataset))
    if orphans:
        names = ", ".join(f"{s}/{t}" for s, t in orphans[:3])
        print(f"Skipping {len(orphans)} built parquet(s) the source no longer enumerates "
              f"({names}{', …' if len(orphans) > 3 else ''}).")

    if not args.report_only:
        print(f"Scoring {len(grid)} placement cells x {len(ARMS)} arms over {len(row_keys)} "
              f"trials...")
        state, _ = run_tracked_grid(
            row_keys, ['Subject', 'Trial'], ['placement'],
            partial(_trial_worker, dataset=args.dataset, tables=args.only_tables,
                    stride=args.stride),
            args.workers, title=f"SENSOR PLACEMENT — {args.dataset}")
        skipped = [key for (key, stage), value in state.items()
                   if stage == 'placement' and value == 'Skipped']
        failures = {key: value for (key, stage), value in state.items()
                    if stage == 'placement' and isinstance(value, str)
                    and value.startswith('Failed')}
        if skipped:
            names = ", ".join(f"{s}/{t}" for s, t in sorted(skipped)[:3])
            print(f"\n{len(skipped)} trial(s) carry one sensor per segment and were skipped "
                  f"({names}{', …' if len(skipped) > 3 else ''}).")
        if failures:
            reasons: Dict[str, int] = {}
            for message in failures.values():
                reasons[message[:70]] = reasons.get(message[:70], 0) + 1
            print(f"\n{len(failures)} of {len(row_keys)} trials FAILED:")
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
                print(f"  {count:4d} x {reason}")
            if len(failures) + len(skipped) == len(row_keys):
                print("\nEvery trial failed or was skipped, so nothing was written. Anything "
                      "below would describe a PREVIOUS run.")
                return 1

    print("\nLoading per-trial tables...")
    stats = load_trial_table(args.dataset, 'placement_stats', row_keys)
    if stats.empty:
        # Two different situations, and telling them apart matters: "nothing has been run" wants
        # a re-run, while "everything you selected has one sensor per segment" wants a different
        # --trials. Saying "run without --report-only first" for the second is advice that
        # cannot work, and it is the message a --trials t12_longwalk_001 run used to get.
        if not args.report_only:
            print(f"\nNothing to report: every selected trial was skipped or failed, so no "
                  f"placement_stats\nwas written. A skipped trial carries one sensor per "
                  f"segment and has no placement to vary.")
        else:
            print(f"\nNo placement_stats under {dataset_dir(args.dataset)} for the "
                  f"{len(row_keys)} selected trial(s).\nRun without --report-only first, or "
                  f"widen --subjects/--trials.")
        return 1
    geometry = load_trial_table(args.dataset, 'placement_geometry', row_keys)
    print(f"Found {stats.groupby(['subject', 'trial'], observed=True).ngroups} trial(s) across "
          f"{stats.subject.nunique()} subject(s), {len(stats):,} cells.")

    summary = summarize(args.dataset, stats)
    if not summary.empty:
        path = paths.ensure_parent(statistics_path(args.dataset))
        summary.to_parquet(path, engine='pyarrow', index=False)
        paths.write_manifest(path, constants=analysis_constants(args.dataset),
                             experiment=EXPERIMENT_NAME, n_rows=len(summary),
                             subjects=subjects_of(row_keys))
        print(f"Saved summary to {path}")

    print_report(spec, stats, geometry, args.metric)
    print(f"\nPer-trial tables under {dataset_dir(args.dataset)}")
    print(f"Figures: python -m plotting.sensor_placement --dataset {args.dataset}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
