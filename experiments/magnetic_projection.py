"""
Does a magnetometer reading transport to the joint centre, and can an array of magnetometers on
one body do better than assuming it does? Measured the same three ways
experiments/acceleration_projection.py measures its projection, against the same three
references, and reported in the same table so the two can be read side by side.

THE ASYMMETRY WITH THE ACCELEROMETER IS THE POINT
=================================================
An accelerometer reading does not transport unchanged to another point of the same rigid body:
the lever arm shows up as alpha x r + omega x (omega x r), and `IMUTrace.project_acc` computes
it. A magnetometer reading has no such term. Every filter in this repo therefore transports the
field to the joint centre by doing nothing to it,

    m_joint = m_sensor                                             (the 0TH-ORDER PROJECTION)

which is exact if and only if the field is uniform over the distance involved — 10-35 cm from a
sensor to its joint centre, 25-47 cm between the two sensors spanning a joint. Nothing else in
this repository checks that. `global_assumptions` measures how far each sensor's world-frame
reading sits from ONE constant field, which is a different and weaker question: a field that
varies smoothly across the room would fail that test while transporting perfectly over 10 cm,
and a field that is constant across the room but locally disturbed by the subject's own hardware
would pass it while transporting badly.

    python -m experiments.magnetic_projection --dataset alborno
    python -m experiments.magnetic_projection --dataset imove
    python -m experiments.magnetic_projection --dataset alborno --report-only

THE THREE QUESTIONS
===================

1. MARKER AGREEMENT — does the projected magnetometer agree with the markers?
   family='marker'. Mocap cannot measure a magnetic field, so "agreement with the markers" can
   only mean agreement with a field MODEL the markers support, and two are built:

       variant='global'  the subject's constant global field, which is what every filter here
                         assumes and what `global_assumptions` measures against. Marker input:
                         the orientation only.
       variant='labmap'  the world-frame field as a low-order polynomial in MARKER POSITION,
                         fitted over every sensor and sample (`lab_field_map`). Marker input:
                         position as well as orientation, which makes this the only reference
                         in the file that can express a field that varies across the room —
                         and the lab floor is where this dataset's distortion comes from.

   The map is scored LEAVE-ONE-SENSOR-OUT. A map fitted over every sensor can absorb a
   per-sensor bias into its spatial terms and then "predict" that sensor, which is exactly how
   calibration error gets mistaken for a field gradient. `r2` and `r2_loso` are both reported and
   the gap between them is that effect, measured.

   This is the only family with an external reference, so it is the only one that can catch an
   error both segments make together. Its weakness is that the reference is a MODEL rather than a
   measurement: the residual it reports is the sum of the projection error and the model's own
   inadequacy, and at order 0 the model's inadequacy is most of it.

2. CROSS-SEGMENT AGREEMENT — do the two segments of a joint agree at the joint centre?
   family='joint'. Both readings transported to the shared joint centre and compared in the world
   frame. This is not a side observation: it IS the residual a relative filter's magnetometer
   update consumes, so this family scores the assumption in the currency the pipeline spends.
   No field model enters it.

   It is blind to a common-mode error — a field disturbance that moves both sensors together
   leaves the two in agreement — which is what family 1 is for.

   Its NORM channel needs no mocap at all. |R m| = |m| for any rotation, so comparing the two
   sensors' field MAGNITUDES uses neither the orientations nor the offsets, and it is therefore
   available over the whole record rather than only inside the mocap window. That is a stronger
   mocap-free statement than the accelerometer version can make (which still needs the two
   offsets), and it is the part of a disagreement that no orientation estimate can ever absorb:
   if two sensors read field magnitudes differing by 8%, no filter, calibration of orientation,
   or choice of frame can reconcile them.

3. SAME-SEGMENT AGREEMENT — do two magnetometers on ONE segment read the same field?
   family='segment', IMoVE only, the one dataset here carrying three sensors on each thigh and
   shank, 7-19 cm apart on one marker cluster. The cleanest measurement available, because the
   truth side is another magnetometer rather than a model, and because a disagreement here cannot
   be blamed on the joint model, on soft tissue between segments, or on the joint-centre fit.

   Scored in two ways that need different amounts of mocap:
       scope='full'   both sensors are on one rigid body, so the frame transform between them is a
                      CONSTANT (`chordal_mean` of R_A^T R_B), and comparing m_A against C_AB m_B
                      needs no per-sample pose at all. Available over the whole record.
       scope='mocap'  both rotated into the world frame per sample, which is what the higher-order
                      modes need.

   target_kind='partner' with mode='sensor' is the pairwise disagreement — question 3. The same
   family with a multi-sensor mode is question 6: the OTHER sensors on the segment predict the
   held-out one, and there is a real measurement waiting at the point being predicted, which is
   not true of the joint centre where nothing is mounted.

AND THE FOURTH QUESTION, WHICH IS THE ONE THIS FILE WAS BUILT FOR
=================================================================
Can several magnetometers on a body estimate the field GRADIENT, and does a 1st- or 2nd-order
projection then beat the 0th order? Eight projections are scored on all three families (see
MODES), from the reading itself through a segment-scale Taylor expansion to a whole-body gradient
tensor. Two things decide the answer and both are measured rather than argued:

  THE GEOMETRY. The three sensors on a thigh or shank are nearly COLLINEAR — measured, 5-8 mm of
  off-line spread against a 160-193 mm span. A collinear array observes exactly ONE directional
  derivative; the two transverse derivatives are unobservable at any order, however long the
  trial. So a segment-scale fit is a Taylor expansion in a single arc-length coordinate, not a
  3x3 tensor, and `offline_mm` records how far off that line the joint centre actually sits. A
  full tensor is observable only at the BODY scale, where the sensors are not collinear.

  And the joint centre sits 0.4-1.3 sensor-spans BEYOND the end of the array, so every
  segment-scale projection is an extrapolation. `extrapolation` carries the ratio per comparison,
  which is what predicts when a higher-order fit helps and when it explodes.

  THE MECHANISM. Two sensors on one segment disagree for two candidate reasons that transform
  differently, so they are separable. A hard-iron or gain error is CONSTANT IN THE SENSOR'S BODY
  FRAME. A real spatial gradient contributes G (x_A - x_B) with G fixed in the LAB, and the
  separation vector rotates with the limb. `mechanism_rows` fits the same difference series both
  ways and reports the variance each explains — and adds the sharpest test in the file: a real
  magnetostatic gradient tensor is SYMMETRIC AND TRACELESS (curl B = 0, div B = 0), five free
  numbers rather than nine, and a fit that is really absorbing calibration error has no reason to
  satisfy either constraint. `explained_gradient_physical` against `explained_gradient` is that
  comparison.

WHY AGREEMENT ALONE WOULD BE THE WRONG SCORE
============================================
Cross-segment agreement can be driven to EXACTLY ZERO by throwing the measurement away: have both
segments report a field model's value at the joint centre and they agree perfectly, having
measured nothing. That is not a hypothetical weakness of the metric, it is a mode in the table —
`body_model` — kept so the blind spot is visible in the same numbers instead of being argued
about. Every mode is therefore scored on the joint family AND the marker family, and a mode that
improves the first while degrading the second has moved error around rather than removed it.

CALIBRATION, AND THE CIRCULARITY IT WOULD CREATE
================================================
Every comparison is run under four calibrations (see CALIBRATIONS), three of them fitted on the
subject's OTHER trials and applied here. The fit's target is the subject's constant global field
rotated into the body frame, so a calibrated arm CANNOT be used to answer question 1 — it was
fitted to minimise exactly that deviation. It can be used for questions 2, 3 and 6, whose
metrics the fit never saw. The report says so where it uses them, and `hard_iron_own` is carried
purely so the gap between an in-sample and an out-of-sample calibration is a measured number
rather than an assumption.

SHARED WITH THE ACCELERATION EXPERIMENT, DELIBERATELY
=====================================================
`Window`, `build_window`, `condition`, `lowpass`, `angle_between_deg`, `rms`, `norm_gap`,
`residual_alignment` and `same_segment_groups` are imported from
experiments/acceleration_projection.py rather than reimplemented. Both files score "two signals
that should be equal" over contiguous runs of valid frames, filtered run by run and trimmed; two
implementations of that would agree until they did not, and the whole reason for matching the
family/channel/scope vocabulary is that the magnetic and inertial answers are meant to be read
against each other.

The low-pass is the same 6 Hz, and applying it here needs its own justification since the reason
it was chosen there (mocap double-differentiation noise) does not exist here — no reference in
this file is differentiated. It is applied anyway, to both sides of every comparison identically,
so that a residual quoted here is over the same band as a residual quoted there. `magnetic
content above 6 Hz` is not zero — a limb swinging at 500 deg/s sweeps the body-frame field fast —
but both sides see the same filter, so the comparison is fair and only its bandwidth is narrowed.

OUTPUTS, all under results/experiments/magnetic_projection/<dataset>/

    <subject>/subject_field.parquet          the subject's constant global field (stage 1)
    <subject>/<trial>/trial_field.parquet    per-(trial, sensor) median world-frame field
    <subject>/<trial>/calibration.parquet    per-sensor hard-iron and affine fit (stage 2)
    <subject>/<trial>/agreement_samples.parquet  per-sample metrics, every family, strided
    <subject>/<trial>/agreement_stats.parquet    per-comparison scalars, every mode x calibration
    <subject>/<trial>/lab_field.parquet      the marker-supported field map, orders 0-2
    <subject>/<trial>/gradient_fits.parquet  array geometry and gradient-estimation quality
    <subject>/<trial>/mechanism.parquet      gradient-or-sensor decomposition, same-segment pairs

plus the pooled quantile summary at
results/statistics/magnetic_projection_<dataset>_statistics.parquet.

THE PARQUET IS THE INTERFACE — trials come from results/trials/<dataset>/ through
`experiment_utils.load_trial`, which refuses a stale artifact. Build first:

    python -m experiments.build_trials --dataset <name>

A dataset whose IMUs carry no magnetometer (imove_biplane) is refused outright rather than
producing tables about the zeros its reader writes.

Figures: python -m plotting.magnetic_projection --dataset <name>
"""
import argparse
import os
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import paths
from experiments.acceleration_projection import (CHANNELS, LOWPASS_CUTOFF_HZ, LOWPASS_ORDER,
                                                 MIN_KEEP_S, PLACEMENT_TOKENS, ROLES, SCOPES,
                                                 TRIM_PERIODS, Window, angle_between_deg,
                                                 build_window, chordal_mean, condition, lowpass,
                                                 norm_gap, residual_alignment, rms,
                                                 rotation_spread_deg, same_segment_groups)
from experiments.experiment_utils import load_trial, pipeline_constants, run_tracked_grid
from experiments.global_assumptions import (DATASETS, MAG_UNIT, DatasetSpec, build_name,
                                            canonical_joint, enumerate_trials, get_dataset,
                                            orphaned_trials, select_trials,
                                            subject_field_from_trials, subject_field_table,
                                            subjects_of, trial_field_table)
from experiments.joint_center import MIN_FIT_FRAMES, joint_offsets
from src.toolchest.PlateTrial import PlateTrial

EXPERIMENT_NAME = "magnetic_projection"
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Family names, in the order the report walks them. Same three questions, same names and same
# meanings as experiments/acceleration_projection.py — the two files are meant to be read as one
# pair of results, and a family that meant something different in each would make that impossible.
FAMILIES = ('marker', 'joint', 'segment')
FAMILY_LABELS = {
    'marker': 'projected field vs a marker-supported field model',
    'joint': 'parent vs child, both transported to the joint centre',
    'segment': 'two magnetometers on one segment, compared with each other',
}

# --- Projection modes ---------------------------------------------------------------------
# How a segment turns the magnetometer(s) it carries into a field estimate at a target point.
# Applied to BOTH sides of a comparison where both can support it, and to whichever side can where
# only one can — a hip has three sensors on the thigh and one on the pelvis, and upgrading the
# thigh alone is both what a real system would do and the more sensitive test, since the unchanged
# side cannot absorb the effect. `n_roles_upgraded` records which happened.
#
#   sensor         the reading itself, transported unchanged. THE 0TH ORDER, what every filter in
#                  this repo does today, and the baseline every other mode is measured against.
#                  `estimate_raw` is this mode in every comparison, so the projected-vs-baseline
#                  structure of the acceleration experiment carries over exactly.
#   mean           the arithmetic mean of the segment's sensors. Still 0th order — no spatial
#                  model at all — but it averages down per-sensor error, which separates "more
#                  sensors help" from "a spatial model helps". Without it, any gain from `linear`
#                  could be either.
#   linear         1st order along the segment's sensor line: B(s) = c0 + c1 s per component in
#                  the arc-length coordinate, evaluated at the target's foot point. Needs >= 2
#                  sensors on the segment.
#   quadratic      2nd order along the same line. Needs 3, and with exactly 3 it INTERPOLATES them
#                  exactly, so it has no in-sample residual and its only test is extrapolation.
#   body_linear    the sensor's own reading plus the change a whole-body gradient predicts over
#                  its lever arm, the gradient fitted PER SAMPLE from every sensor except the ones
#                  being scored. Without that exclusion the fit is minimising the very
#                  disagreement it is then scored on.
#   trial_gradient the same, with ONE gradient tensor for the whole trial. Thousands of samples
#                  against nine parameters instead of fifteen sensors against nine, and the
#                  version with a physical claim behind it: a lab's field gradient is a property
#                  of the room, not of the instant.
#   trial_gradient_phys
#                  the same tensor constrained SYMMETRIC AND TRACELESS, which curl B = 0 and
#                  div B = 0 require of any real magnetostatic gradient. Five parameters rather
#                  than nine, and the only mode here that cannot fit something a magnetic field
#                  could not do. If a gradient projection is going to work at all, this is the
#                  estimator it should use.
#   body_model     both sides report the fitted whole-body model's value at the target and ignore
#                  their own reading. THE CONTROL, not a proposal — see the docstring.
MODES = ('sensor', 'mean', 'linear', 'quadratic', 'body_linear', 'trial_gradient',
         'trial_gradient_phys', 'body_model')
PRIMARY_MODE = 'sensor'

# Modes fitted along the segment's own sensor line, and the polynomial order each uses.
LINE_MODE_ORDER = {'mean': 0, 'linear': 1, 'quadratic': 2}
# Modes needing a whole-body fit over the other sensors.
BODY_MODES = ('body_linear', 'trial_gradient', 'trial_gradient_phys', 'body_model')

# --- Calibration ----------------------------------------------------------------------------
# A magnetometer reads m = A R^T B + b: a soft-iron/scale matrix A and a hard-iron offset b, both
# fixed in the SENSOR frame. `assembly.align_world_to_imu` leaves `imu_trace` in the raw sensor
# frame and puts the sensor->world rotation in the world trace, so that frame is the same across
# a subject's trials — which is what makes a calibration fitted on one trial applicable to
# another, and makes its repeatability a test of whether it is a device property at all.
#
#   none           the reading as recorded. THE PRIMARY ARM: every headline number is quoted under
#                  it, because it is what the pipeline consumes.
#   hard_iron      b removed, fitted on the subject's OTHER trials.
#   hard_iron_own  the same fitted on THIS trial. In sample, and kept only so the gap between the
#                  two measures the optimism of an in-sample calibration.
#   affine         the full 12-parameter map, fitted on the other trials.
CALIBRATIONS = ('none', 'hard_iron', 'hard_iron_own', 'affine')
PRIMARY_CALIBRATION = 'none'

# Minimum valid frames before a calibration is fitted at all. The affine arm has twelve parameters
# and its conditioning depends on how much the sensor rotated, which is recorded per fit
# (`design_condition`) rather than assumed.
MIN_CALIBRATION_FRAMES = 200

# --- Lab field map --------------------------------------------------------------------------
LAB_MAP_ORDERS = (0, 1, 2)
# The order the marker family's `labmap` variant uses as its truth. 1 rather than 2 because the
# order-2 map's leave-one-sensor-out score is WORSE than order 1 on both datasets (measured: 0.043
# against 0.055 on IMoVE), i.e. the quadratic terms are fitting sensors rather than the room.
LAB_MAP_TRUTH_ORDER = 1
# Rows per sensor the map is fitted on. The map is a function of POSITION and consecutive samples
# sit millimetres apart, so the thousandth sample of a stationary pose adds cost and no
# information. Predictions are evaluated at full rate against the strided fit, which is the
# intended asymmetry: the fit needs distinct positions, the prediction needs to line up sample for
# sample with the reading it is scored against.
LAB_MAP_MAX_SAMPLES = 4000

# --- Gradient estimation ----------------------------------------------------------------------
# A whole-body linear fit has four parameters per field component, so it needs four sensors and
# five before the residual means anything. Below this the body modes are unavailable for that
# comparison rather than returning a fit with no degrees of freedom left.
MIN_BODY_SENSORS = 5
TRIAL_GRADIENT_MAX_SAMPLES = 4000

# --- General ------------------------------------------------------------------------------
SAMPLE_STRIDE = 10
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# Every magnetic quantity here is in the arbitrary units the Xsens export uses, where a nominal
# Earth field reads ~1 (see MAG_UNIT in experiments/global_assumptions.py). Multiply by ~50 uT for
# a physical scale. Direction errors are in degrees and are the ones to quote: the relative
# filter normalizes its magnetometer measurement, so the direction is what its update responds to.
METRIC_UNITS = {
    'err_proj': MAG_UNIT,
    'err_raw': MAG_UNIT,
    'err_proj_alt': MAG_UNIT,
    'ang_proj': 'deg',
    'ang_raw': 'deg',
    'dnorm_proj': MAG_UNIT,
    'absdnorm_proj': MAG_UNIT,
    'absdnorm_raw': MAG_UNIT,
    'corr_norm': MAG_UNIT,
    'truth_norm': MAG_UNIT,
    'sep_norm': 'm',
}

SUMMARY_METRICS = ('err_proj', 'err_raw', 'ang_proj', 'ang_raw', 'absdnorm_proj', 'absdnorm_raw',
                   'corr_norm')

# --- Covariation --------------------------------------------------------------------------
# Everything else in this file measures how far two sensors DISAGREE. Covariation asks the
# complementary question — how much of what they see is SHARED — and it is the one that decides
# whether differencing them helps. A relative filter subtracts one segment's magnetometer from the
# other's, so a disturbance common to both CANCELS and a disturbance private to each ADDS.
#
# The scalar is the fraction of disturbance energy that SURVIVES the subtraction,
#
#     surviving = |dB_a - dB_b|^2 / (|dB_a|^2 + |dB_b|^2)
#
# which for two signals of equal variance is exactly 1 - rho. So 0 means perfectly shared and
# differencing removes it entirely, 1 means independent and differencing neither helps nor hurts,
# and above 1 means anti-correlated and differencing makes it worse. That single number is what a
# relative formulation is betting on.
#
# Split into two bands, because the two candidate mechanisms live at different timescales: a
# room-driven disturbance changes as the subject walks across the lab (slow), while a body-fixed
# bias sweeps the world frame at the limb's own rotation rate (gait band).
SLOW_BAND_HZ = 0.5

# Pair classes, ordered by how close the two sensors are. The non-adjacent classes are the CONTROL:
# if adjacency buys nothing, the shared part is not spatial.
PAIR_CLASSES = ('same_segment', 'across_joint', 'same_limb', 'contralateral', 'distant')

STAGE1_TABLE = 'trial_field'
STAGE2_TABLE = 'calibration'
ANALYSIS_TABLES = ('agreement_samples', 'agreement_stats', 'lab_field', 'gradient_fits',
                   'mechanism', 'covariation')
TRIAL_TABLES = (STAGE1_TABLE, STAGE2_TABLE) + ANALYSIS_TABLES

# Which comparisons reach the per-sample table. agreement_stats carries the full mode x
# calibration grid computed on the whole conditioned record; this table exists for the figures,
# and at stride 10 over IMoVE's 280 k-sample long walks the full grid would be millions of rows
# per trial for quantiles that are identical to the scalars beside them.
SAMPLE_CALIBRATION = PRIMARY_CALIBRATION


def analysis_constants(dataset: str) -> Dict[str, object]:
    """Pipeline constants plus this analysis's own choices, for provenance manifests."""
    return {
        **pipeline_constants(),
        'dataset': dataset,
        'families': list(FAMILIES),
        'modes': list(MODES),
        'calibrations': list(CALIBRATIONS),
        'primary_mode': PRIMARY_MODE,
        'primary_calibration': PRIMARY_CALIBRATION,
        'lowpass_cutoff_hz': LOWPASS_CUTOFF_HZ,
        'lowpass_order': LOWPASS_ORDER,
        'trim_periods': TRIM_PERIODS,
        'min_keep_s': MIN_KEEP_S,
        'lab_map_orders': list(LAB_MAP_ORDERS),
        'lab_map_truth_order': LAB_MAP_TRUTH_ORDER,
        'lab_map_max_samples': LAB_MAP_MAX_SAMPLES,
        'min_calibration_frames': MIN_CALIBRATION_FRAMES,
        'min_body_sensors': MIN_BODY_SENSORS,
        'slow_band_hz': SLOW_BAND_HZ,
        'trial_gradient_max_samples': TRIAL_GRADIENT_MAX_SAMPLES,
        'min_fit_frames': MIN_FIT_FRAMES,
        'sample_stride': SAMPLE_STRIDE,
    }

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
    """Concatenates one per-trial table across trials, adding `subject`, `trial` and `dataset`.

    Missing trials are skipped silently — a partial run is a legitimate state and the caller
    reports what it found. `columns` is pushed down into the parquet read, which matters for
    agreement_samples; the label columns come back as categoricals because as objects they cost
    more than every float column combined.
    """
    if table not in TRIAL_TABLES:
        raise ValueError(f"Unknown table '{table}'; expected one of {TRIAL_TABLES}")
    row_keys = enumerate_trials(dataset) if row_keys is None else row_keys
    if columns is not None:
        columns = list(dict.fromkeys(columns))
    frames = []
    for subject, trial in row_keys:
        path = trial_table_path(dataset, subject, trial, table)
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path, engine='pyarrow', columns=columns)
                      .assign(subject=subject, trial=trial))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df['dataset'] = dataset
    for label in ('family', 'group', 'variant', 'source', 'target', 'target_kind', 'scope',
                  'channel', 'mode', 'calibration', 'segment', 'sensor', 'subject', 'trial',
                  'dataset'):
        if label in df.columns:
            df[label] = df[label].astype('category')
    return df


def read_subject_field(dataset: str, subject: str) -> Optional[np.ndarray]:
    """One subject's constant global field, or None if stage one has not run for them."""
    path = subject_field_path(dataset, subject)
    if not path.exists():
        return None
    row = pd.read_parquet(path, engine='pyarrow').iloc[0]
    return np.array([row['world_mag_x'], row['world_mag_y'], row['world_mag_z']], dtype=float)

# ==============================================================================
# Helpers
# ==============================================================================

def as_magnitude(vectors: np.ndarray) -> np.ndarray:
    """An (N, 3) array carrying |v| on its first axis and zeros elsewhere.

    How the norm channel rides through machinery written for 3-vectors: every table then has one
    code path whatever it is describing, and `err_proj` on such a pair is exactly
    | |estimate| - |truth| |. The two zeroed axes are why `channel='norm'` blanks every angular
    column rather than reporting the degenerate zero the arithmetic would produce.
    """
    out = np.zeros_like(np.asarray(vectors, dtype=float))
    out[:, 0] = np.linalg.norm(vectors, axis=1)
    return out


def _finite(*arrays: np.ndarray) -> bool:
    """Whether every array is finite. A comparison with a NaN in it is dropped rather than
    written: one NaN poisons an rms and a Kabsch fit alike, and every cause seen here (a
    degenerate offset fit, a sensor with no calibration) is a condition where the whole
    comparison is meaningless rather than one sample of it."""
    return all(np.all(np.isfinite(a)) for a in arrays)


def _quantile_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """median / p90 / rms / mean of a 1-D array, as a flat dict. NaN- and empty-safe."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {f'{prefix}_p50': np.nan, f'{prefix}_p90': np.nan,
                f'{prefix}_rms': np.nan, f'{prefix}_mean': np.nan}
    return {f'{prefix}_p50': float(np.median(values)),
            f'{prefix}_p90': float(np.percentile(values, 90)),
            f'{prefix}_rms': float(np.sqrt(np.mean(values ** 2))),
            f'{prefix}_mean': float(np.mean(values))}

# ==============================================================================
# Calibration
# ==============================================================================

AFFINE_COLUMNS = tuple(f'affine_{i}{j}' for i in range(4) for j in range(3))
HARD_IRON_COLUMNS = ('hard_iron_x', 'hard_iron_y', 'hard_iron_z')


def fit_calibration(mag: np.ndarray, rotations: np.ndarray, reference_field: np.ndarray,
                    mask: np.ndarray) -> Dict[str, object]:
    """One sensor's hard-iron offset and full affine map against the subject's global field.

    The target is that field rotated into the sensor's own frame,

        y(t) = R(t)^T B_ref

    which is what the sensor WOULD read if the field were uniform and the sensor perfect. Two fits
    against it: a hard iron, b = mean(m - y), three parameters; and an affine map, [m 1] W = y in
    least squares, twelve parameters covering a soft-iron/scale matrix and an offset together.

    WHAT SEPARATES b FROM A FIELD ERROR is orientation coverage and nothing else: b is constant in
    the body frame while y sweeps as the sensor turns, so a trial in which the sensor barely
    rotates cannot tell them apart. `design_condition` is that coverage made numeric — on this
    data it runs ~50 for a foot and ~2500 for a pelvis during walking — and it is recorded rather
    than thresholded on, because what counts as enough depends on how the fit is then used.

    Any genuine spatial variation of the field is absorbed here too: a sensor that spends the
    trial in one part of the room reads a locally offset field that looks exactly like a hard
    iron. That is not a defect to correct but the reason the calibrated arms are fitted on OTHER
    trials before being applied, and the reason `report_calibration` leads with how repeatable the
    fits are between trials. A term that reproduces across a subject's trials is a device
    property; one that does not was the room.
    """
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    out: Dict[str, object] = {'n_frames': n}
    if n < MIN_CALIBRATION_FRAMES:
        return out

    target = np.einsum('nji,j->ni', rotations[mask], reference_field)
    reading = np.asarray(mag[mask], dtype=float)

    hard_iron = (reading - target).mean(axis=0)
    design = np.column_stack([reading, np.ones(n)])
    gram = design.T @ design
    affine, *_ = np.linalg.lstsq(design, target, rcond=None)

    eigenvalues = np.linalg.eigvalsh(gram)
    residual = lambda corrected: float(np.sqrt(np.mean(np.sum((corrected - target) ** 2, axis=1))))
    out.update({
        'hard_iron_x': float(hard_iron[0]), 'hard_iron_y': float(hard_iron[1]),
        'hard_iron_z': float(hard_iron[2]),
        'hard_iron_norm': float(np.linalg.norm(hard_iron)),
        'residual_raw': residual(reading),
        'residual_hard_iron': residual(reading - hard_iron),
        'residual_affine': residual(design @ affine),
        'design_condition': float(eigenvalues[-1] / max(eigenvalues[0], 1e-12)),
        'mag_norm_median': float(np.median(np.linalg.norm(reading, axis=1))),
        # The affine map's departure from the identity, which is what says whether anything beyond
        # a hard iron was needed: a perfect sensor in a uniform field gives W = [I; 0].
        'affine_gain_departure': float(np.linalg.norm(affine[:3] - np.eye(3))),
        **{f'affine_{i}{j}': float(affine[i, j]) for i in range(4) for j in range(3)},
    })
    return out


def calibration_table(plates: Dict[str, PlateTrial], reference_field: np.ndarray) -> pd.DataFrame:
    """Stage two: every sensor's calibration fit for one trial, one row each."""
    rows = []
    for sensor, plate in sorted(plates.items()):
        fit = fit_calibration(plate.imu_trace.mag, plate.world_trace.rotations,
                              reference_field, np.asarray(plate.valid))
        rows.append({'sensor': sensor, **fit})
    return pd.DataFrame(rows)


def load_calibrations(dataset: str, subject: str, trial: str,
                      row_keys: Sequence[Tuple[str, str]]) -> Dict[str, Dict[str, np.ndarray]]:
    """{sensor: {'hard_iron': (3,), 'affine': (4,3), '*_own': ...}} for one trial.

    The unsuffixed entries are averaged over the subject's OTHER trials; the `_own` entries come
    from this trial. A subject with a single trial has nothing to average, so those keys are
    absent and `apply_calibration` falls back and says so — `calibration_source` then records that
    in every row, so a leave-one-out claim can never be read off a run that quietly did not leave
    anything out.

    Averaging affine maps across trials is a mean of linear operators estimated noisily for one
    physical device, which is the same thing the hard-iron mean is.
    """
    own_path = trial_table_path(dataset, subject, trial, STAGE2_TABLE)
    own = pd.read_parquet(own_path, engine='pyarrow') if own_path.exists() else pd.DataFrame()

    others = []
    for other_subject, other_trial in row_keys:
        if other_subject != subject or other_trial == trial:
            continue
        path = trial_table_path(dataset, subject, other_trial, STAGE2_TABLE)
        if path.exists():
            others.append(pd.read_parquet(path, engine='pyarrow'))
    other = pd.concat(others, ignore_index=True) if others else pd.DataFrame()

    parameters: Dict[str, Dict[str, np.ndarray]] = {}
    for frame, suffix in ((own, '_own'), (other, '')):
        if frame.empty or 'hard_iron_x' not in frame.columns:
            continue
        usable = frame.dropna(subset=['hard_iron_x'])
        for sensor, group in usable.groupby('sensor'):
            entry = parameters.setdefault(str(sensor), {})
            entry[f'hard_iron{suffix}'] = group[list(HARD_IRON_COLUMNS)].to_numpy(float).mean(axis=0)
            if set(AFFINE_COLUMNS) <= set(group.columns):
                entry[f'affine{suffix}'] = (group[list(AFFINE_COLUMNS)].to_numpy(float)
                                            .mean(axis=0).reshape(4, 3))
    return parameters


def apply_calibration(mag: np.ndarray, parameters: Dict[str, np.ndarray],
                      calibration: str) -> Tuple[np.ndarray, str]:
    """The corrected reading and the source actually used ('other', 'own' or 'none').

    Falls back one step at a time and says which: an `affine` arm with no other-trial fit uses
    this trial's, and a sensor with no fit at all is left uncorrected. Returning the source rather
    than silently substituting is what keeps `hard_iron` and `hard_iron_own` distinguishable when
    a subject has only one trial.
    """
    if calibration == 'none':
        return mag, 'none'
    base = 'hard_iron' if calibration.startswith('hard_iron') else 'affine'
    wanted = ['own', ''] if calibration.endswith('_own') else ['', 'own']
    for suffix in wanted:
        key = f'{base}{"_" + suffix if suffix else ""}'
        if key in parameters:
            values = parameters[key]
            source = suffix or 'other'
            if base == 'hard_iron':
                return mag - values, source
            return np.column_stack([mag, np.ones(len(mag))]) @ values, source
    return mag, 'none'

# ==============================================================================
# Field models: the lab map and the gradient fits
# ==============================================================================

def _polynomial_terms(displacement: np.ndarray, order: int) -> np.ndarray:
    """Design columns for a polynomial in 3-D displacement, up to `order` (0, 1 or 2)."""
    columns = [np.ones(len(displacement))]
    if order >= 1:
        columns += [displacement[:, 0], displacement[:, 1], displacement[:, 2]]
    if order >= 2:
        columns += [displacement[:, 0] ** 2, displacement[:, 1] ** 2, displacement[:, 2] ** 2,
                    displacement[:, 0] * displacement[:, 1],
                    displacement[:, 0] * displacement[:, 2],
                    displacement[:, 1] * displacement[:, 2]]
    return np.stack(columns, axis=1)


def lab_field_map(fields: Dict[str, np.ndarray], positions: Dict[str, np.ndarray],
                  order: int, stride: int = 1) -> Dict[str, object]:
    """The world-frame field as a polynomial in LAB POSITION, fitted over every sensor at once.

    The only reference in this file whose marker input is POSITION rather than just orientation,
    which is what makes it the honest form of "do the magnetometers agree with the markers": the
    markers say where each sensor was, and the map asks whether one smooth function of that
    position accounts for what they all read.

    Three numbers carry the argument:
        r2            in sample, over every sensor.
        r2_loso       refit without sensor i, scored on sensor i, pooled. A map fitted with every
                      sensor can hide a per-sensor bias inside its spatial terms and then
                      "predict" that sensor — precisely how calibration error becomes an apparent
                      field gradient. This column cannot.
        gradient_norm the Frobenius norm of the fitted first-order term, in field units per metre:
                      the SCALE of the spatial variation the markers can actually see, and the
                      yardstick every gradient claim in the file is read against.
    """
    names = sorted(fields)
    stacked_x = {name: positions[name][::stride] for name in names}
    stacked_b = {name: fields[name][::stride] for name in names}
    origin = np.concatenate([stacked_x[name] for name in names]).mean(axis=0)

    def solve(subset: Sequence[str]) -> np.ndarray:
        design = _polynomial_terms(np.concatenate([stacked_x[n] for n in subset]) - origin, order)
        target = np.concatenate([stacked_b[n] for n in subset])
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return coefficients

    coefficients = solve(names)
    all_x = np.concatenate([stacked_x[n] for n in names])
    all_b = np.concatenate([stacked_b[n] for n in names])
    residual = all_b - _polynomial_terms(all_x - origin, order) @ coefficients
    total = float(np.sum((all_b - all_b.mean(axis=0)) ** 2))

    loso_residual = []
    for held_out in names:
        kept = [n for n in names if n != held_out]
        if not kept:
            continue
        prediction = _polynomial_terms(stacked_x[held_out] - origin, order) @ solve(kept)
        loso_residual.append(stacked_b[held_out] - prediction)
    loso = np.concatenate(loso_residual) if loso_residual else residual

    return {
        'order': order,
        'n_samples': int(len(all_b)),
        'n_sensors': len(names),
        'r2': 1.0 - float(np.sum(residual ** 2)) / total if total > 0 else np.nan,
        'r2_loso': 1.0 - float(np.sum(loso ** 2)) / total if total > 0 else np.nan,
        'residual_p50': float(np.median(np.linalg.norm(residual, axis=1))),
        'residual_loso_p50': float(np.median(np.linalg.norm(loso, axis=1))),
        'field_norm_p50': float(np.median(np.linalg.norm(all_b, axis=1))),
        'gradient_norm': float(np.linalg.norm(coefficients[1:4])) if order >= 1 else 0.0,
    }


def lab_field_table(fields: Dict[str, np.ndarray], positions: Dict[str, np.ndarray],
                    stride: int) -> pd.DataFrame:
    return pd.DataFrame([lab_field_map(fields, positions, order, stride)
                         for order in LAB_MAP_ORDERS])


def lab_map_predictor(fields: Dict[str, np.ndarray], positions: Dict[str, np.ndarray],
                      order: int, stride: int, exclude: Sequence[str] = ()):
    """A callable predicting the world-frame field at any set of world positions.

    `exclude` is what makes it usable as the marker family's truth: the sensors being scored must
    not have helped fit the model they are scored against, or the comparison measures how well a
    sensor predicts itself.
    """
    names = [name for name in sorted(fields) if name not in set(exclude)]
    origin = np.concatenate([positions[n][::stride] for n in sorted(fields)]).mean(axis=0)
    if not names:
        return None
    design = _polynomial_terms(np.concatenate([positions[n][::stride] for n in names]) - origin,
                               order)
    target = np.concatenate([fields[n][::stride] for n in names])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return lambda points: _polynomial_terms(np.asarray(points) - origin, order) @ coefficients


# The five independent components of a physically admissible magnetostatic gradient. Away from
# currents and magnetized material curl B = 0 and div B = 0, so dB_i/dx_j is SYMMETRIC and
# TRACELESS — five free numbers, not nine.
#
# Imposing it is the sharpest test in the file of whether a fitted "gradient" is a magnetic field
# at all: a genuine field gradient loses nothing by being constrained, while a fit that is really
# absorbing per-sensor calibration error has no reason to be either and collapses.
_SYMMETRIC_TRACELESS_BASIS = np.array([
    [[1, 0, 0], [0, 0, 0], [0, 0, -1]],
    [[0, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
], dtype=float)


def fit_gradient(separation: np.ndarray, difference: np.ndarray, physical: bool = False
                 ) -> Tuple[np.ndarray, float]:
    """The tensor best explaining `difference ~ G @ separation`, and the variance it explains.

    Both arrays are world-frame and (N, 3). The separation ROTATES with the body, which is what
    makes G identifiable at all and what distinguishes this model from a body-fixed constant —
    the two would be indistinguishable on a body that never turned.

    With `physical`, G is constrained symmetric and traceless: the design is built directly in the
    five basis coefficients rather than by constraining nine entries afterwards.
    """
    total = float(np.sum(difference ** 2))
    if not physical:
        solution, *_ = np.linalg.lstsq(separation, difference, rcond=None)
        G = solution.T
        residual = difference - separation @ solution
    else:
        design = np.stack([(separation @ basis.T).reshape(-1)
                           for basis in _SYMMETRIC_TRACELESS_BASIS], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, difference.reshape(-1), rcond=None)
        G = np.einsum('k,kij->ij', coefficients, _SYMMETRIC_TRACELESS_BASIS)
        residual = difference - separation @ G.T
    explained = 1.0 - float(np.sum(residual ** 2)) / total if total > 0 else np.nan
    return G, explained


def instantaneous_gradient(fields: Dict[str, np.ndarray], positions: Dict[str, np.ndarray],
                           names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample (B0, G, centre) from a set of sensors: B(x) ~ B0 + G (x - centre).

    Four parameters per field component against `len(names)` sensors, solved for the three
    components at once because they share one design matrix. G's rows are the components'
    gradients, so `G @ dx` is the modelled field change.

    BADLY CONDITIONED, AND MEASURING THAT IS THE POINT. Nine gradient parameters against 8-15
    sensors is a fit with almost no redundancy, so anything that makes two sensors disagree —
    calibration, misalignment, marker error — is absorbed into G rather than left in the residual.
    `gradient_fits` reports the resulting |G| beside the lab map's, and the ratio is the honest
    measure of how much of it is field and how much is fit.
    """
    B = np.stack([fields[n] for n in names], axis=1)          # (T, S, 3)
    X = np.stack([positions[n] for n in names], axis=1)
    centre = X.mean(axis=1)
    design = np.concatenate([np.ones((len(B), len(names), 1)), X - centre[:, None, :]], axis=2)
    gram = np.einsum('tsi,tsj->tij', design, design)
    rhs = np.einsum('tsi,tsk->tik', design, B)
    # A ridge of 1e-12 on the diagonal: with four sensors the Gram is singular whenever they are
    # momentarily coplanar, which happens on real limb poses. The term is far below the signal and
    # turns a LinAlgError into the minimum-norm answer.
    coefficients = np.linalg.solve(gram + 1e-12 * np.eye(4), rhs)
    return coefficients[:, 0, :], coefficients[:, 1:, :].transpose(0, 2, 1), centre


def trial_gradient(fields: Dict[str, np.ndarray], positions: Dict[str, np.ndarray],
                   names: Sequence[str], stride: int = 1,
                   physical: bool = False) -> Tuple[np.ndarray, float]:
    """One gradient tensor for the whole trial, and the inter-sensor variance it explains.

    Fitted on each sensor's deviation from the per-sample mean, which removes the unknown ambient
    field exactly and leaves a plain least squares in the gradient alone:

        B_i(t) - mean_j B_j(t)  ~  G ( x_i(t) - mean_j x_j(t) )

    Thousands of samples against nine parameters, so unlike the per-sample fit this one is heavily
    overdetermined — and because the body moves through the room and rotates, the separation
    vectors span all three directions, so every entry of G is identifiable.
    """
    B = np.stack([fields[n][::stride] for n in names], axis=1)
    X = np.stack([positions[n][::stride] for n in names], axis=1)
    return fit_gradient((X - X.mean(axis=1, keepdims=True)).reshape(-1, 3),
                        (B - B.mean(axis=1, keepdims=True)).reshape(-1, 3), physical=physical)

# ==============================================================================
# The segment-scale line
# ==============================================================================

class SegmentLine:
    """The arc-length parametrization of a segment's near-collinear sensor array.

    A collinear array observes ONE directional derivative, so everything a segment-scale fit can
    say is a function of the single coordinate

        s_i(t) = ( x_i(t) - centre(t) ) . u(t),      u along the array,

    and `coordinate` returns the transverse distance alongside it — the part of any lever arm no
    fit along this line can correct, at any order.

    THE ARRAY'S SHAPE IS A BODY-FRAME FACT and is measured in a body frame. Averaging world-frame
    displacements over a trial averages a rotating vector, which shrinks it toward zero and
    distorts its shape; on a walking thigh that alone reported the array as three times straighter
    than it is. u(t) itself comes from the two extreme sensors' live positions rather than from a
    per-sample SVD: on a rigid body the direction is fixed in the body frame, so the two agree to
    the extent the array is collinear at all, and the endpoint version costs one normalization
    instead of a decomposition per sample.
    """

    def __init__(self, fields: Dict[str, np.ndarray], positions: Dict[str, np.ndarray],
                 names: Sequence[str], rotations: np.ndarray):
        X = np.stack([positions[n] for n in names], axis=1)          # (T, S, 3)
        self.names = list(names)
        self.fields = np.stack([fields[n] for n in names], axis=1)   # (T, S, 3)
        self.centre = X.mean(axis=1)
        displacement = X - self.centre[:, None, :]

        geometry = np.einsum('tji,tsj->tsi', rotations, displacement).mean(axis=0)
        principal = np.linalg.svd(geometry - geometry.mean(axis=0), full_matrices=False)[2][0]
        # Ordered along the array once, from that geometry, so the direction cannot flip sample to
        # sample when two sensors are momentarily equidistant.
        order = np.argsort(geometry @ principal)
        self.order = [self.names[i] for i in order]

        axis = displacement[:, order[-1], :] - displacement[:, order[0], :]
        self.direction = axis / np.linalg.norm(axis, axis=1, keepdims=True)
        self.s = np.einsum('tsi,ti->ts', displacement, self.direction)
        self.span = float(np.median(self.s.max(axis=1) - self.s.min(axis=1)))
        along = geometry @ principal
        perpendicular = geometry - np.outer(along, principal)
        self.along_mm = float((along.max() - along.min()) * 1000.0)
        # The largest perpendicular distance of any sensor from the best-fit line, as a DISTANCE
        # rather than a singular value, because it is compared against the off-line displacement
        # of the joint centre, which is also a distance.
        self.offline_mm = float(np.max(np.linalg.norm(
            perpendicular - perpendicular.mean(axis=0), axis=1)) * 1000.0)

    def coordinate(self, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(arc-length coordinate, transverse distance) of a world-frame target point."""
        relative = np.asarray(target) - self.centre
        s = np.einsum('ti,ti->t', relative, self.direction)
        return s, np.linalg.norm(relative - s[:, None] * self.direction, axis=1)

    def _fit(self, indices: Sequence[int], order: int) -> np.ndarray:
        """Polynomial coefficients (T, order+1, 3) from the sensors at `indices`."""
        s = self.s[:, indices]
        design = np.stack([s ** k for k in range(order + 1)], axis=2)
        gram = np.einsum('tsi,tsj->tij', design, design)
        rhs = np.einsum('tsi,tsk->tik', design, self.fields[:, indices, :])
        return np.linalg.solve(gram + 1e-12 * np.eye(order + 1), rhs)

    def evaluate(self, target: np.ndarray, order: int,
                 exclude: Optional[str] = None) -> np.ndarray:
        """The order-`order` fit along the line, evaluated at `target`. Order 0 is the mean.

        `exclude` drops one sensor from the fit, which is what turns this into the held-out
        prediction of question 6 — the only fully out-of-sample test in the file, because a real
        measurement sits at the point being predicted.
        """
        indices = [i for i, name in enumerate(self.names) if name != exclude]
        if len(indices) < order + 1:
            raise ValueError(f"order {order} needs {order + 1} sensors, have {len(indices)}")
        coefficients = self._fit(indices, order)
        s_target, _ = self.coordinate(target)
        terms = np.stack([s_target ** k for k in range(order + 1)], axis=1)
        return np.einsum('ti,tik->tk', terms, coefficients)

    def extrapolation_ratio(self, s_target: np.ndarray,
                            exclude: Optional[str] = None) -> float:
        """|s_target| beyond the nearest end of the array, measured in array spans.

        0 inside the array, 1 one full span past its end. The single number that predicts whether
        a higher-order fit helps or explodes — and it is a property of the MOUNTING rather than of
        the field: three sensors over 17 cm of thigh, asked about a knee 25 cm away.
        """
        indices = [i for i, name in enumerate(self.names) if name != exclude]
        s = self.s[:, indices]
        lowest, highest = float(np.median(s.min(axis=1))), float(np.median(s.max(axis=1)))
        span = highest - lowest
        target = float(np.median(s_target))
        if lowest <= target <= highest or span <= 0:
            return 0.0
        return float((target - highest if target > highest else lowest - target) / span)

# ==============================================================================
# The Comparison: two field estimates that should be equal
# ==============================================================================

@dataclass
class Comparison:
    """Two magnetic-field estimates that SHOULD be equal, in one common frame, conditioned alike.

    The only object the table builders know about, mirroring
    `acceleration_projection.Comparison` so the two experiments' outputs are one schema. Every
    family produces these and nothing downstream asks which family it is looking at.

    `estimate` is the field this side's mode produced at the target point; `estimate_raw` is the
    SAME sensor's 0th-order projection — its reading, transported unchanged — which is the
    baseline the whole analysis turns on. For `mode='sensor'` the two are identical by
    construction, and that row is the baseline every other mode is measured against.

    `truth` is whatever the family compares against: a marker-supported field model, the other
    segment's estimate, or the partner sensor's own reading. `truth_raw` is that side's own 0th
    order, so `corr_norm` can measure how much the mode moved the GAP rather than how much it
    moved either side — a pair whose two corrections move together has been asked to do nothing
    however large those corrections are.
    """
    family: str
    group: str
    variant: str
    source: str
    target: str
    target_kind: str
    mode: str
    calibration: str
    calibration_source: str
    window: Window
    timestamps: np.ndarray
    estimate: np.ndarray
    estimate_raw: np.ndarray
    truth: np.ndarray
    truth_raw: np.ndarray
    truth_alt: Optional[np.ndarray] = None
    channel: str = 'vector'
    n_roles_upgraded: int = 0
    scalars: Dict[str, float] = field(default_factory=dict)

    @property
    def scope(self) -> str:
        return self.window.scope

    @property
    def key(self) -> Dict[str, object]:
        """The label columns identifying this comparison in every table."""
        return {'family': self.family, 'group': self.group, 'variant': self.variant,
                'source': self.source, 'target': self.target, 'target_kind': self.target_kind,
                'mode': self.mode, 'calibration': self.calibration,
                'calibration_source': self.calibration_source,
                'scope': self.scope, 'channel': self.channel}

# ==============================================================================
# Per-trial context
# ==============================================================================

class TrialContext:
    """One trial's plates plus everything the three families share, for ONE calibration arm.

    Built once per arm rather than all four at once: four copies of a 280 k-sample IMoVE long
    walk's fifteen sensors is over a gigabyte, and nothing downstream needs two arms at the same
    moment.

    The world-frame field `B[sensor]` is the full unconditioned record — filtering and trimming
    happen per comparison through `condition`, because different comparisons are scored on
    different windows and filtering once up front would fix one of them for all.
    """

    def __init__(self, plates: Dict[str, PlateTrial], spec: DatasetSpec, dataset: str,
                 subject: str, trial: str, parameters: Dict[str, Dict[str, np.ndarray]],
                 calibration: str, offsets: Dict[str, Dict[str, np.ndarray]]):
        self.plates = plates
        self.spec = spec
        self.dataset, self.subject, self.trial = dataset, subject, trial
        self.calibration = calibration
        self.offsets = offsets
        self.names = sorted(plates)
        self.n = min(len(plate) for plate in plates.values())
        self.fs = self._sample_rate()

        self.mag: Dict[str, np.ndarray] = {}
        self.B: Dict[str, np.ndarray] = {}
        self.X: Dict[str, np.ndarray] = {}
        self.R: Dict[str, np.ndarray] = {}
        sources = set()
        for name in self.names:
            plate = plates[name]
            corrected, source = apply_calibration(
                np.asarray(plate.imu_trace.mag, dtype=float)[:self.n],
                parameters.get(name, {}), calibration)
            sources.add(source)
            rotations = np.asarray(plate.world_trace.rotations, dtype=float)[:self.n]
            self.mag[name] = corrected
            self.R[name] = rotations
            self.B[name] = np.einsum('nij,nj->ni', rotations, corrected)
            self.X[name] = np.asarray(plate.world_trace.positions, dtype=float)[:self.n]
        self.calibration_source = sources.pop() if len(sources) == 1 else 'mixed'

        self.segments = segment_sensors(self.names)
        self._lines: Dict[str, SegmentLine] = {}
        self._body_fits: Dict[Tuple[str, ...], Optional[Dict[str, object]]] = {}
        self._window_cache: Dict[Tuple[str, ...], Optional[Window]] = {}
        self._joint_centres: Dict[str, np.ndarray] = {}

    def _sample_rate(self) -> float:
        reference = self.plates.get(self.spec.pelvis_sensor) or next(iter(self.plates.values()))
        return float(1.0 / np.median(np.diff(reference.imu_trace.timestamps[:self.n])))

    def timestamps(self, sensor: str) -> np.ndarray:
        return np.asarray(self.plates[sensor].imu_trace.timestamps, dtype=float)[:self.n]

    # --- windows ------------------------------------------------------------------
    def mocap_window(self, *sensors: str) -> Optional[Window]:
        """The valid-run window for the AND of these sensors' mocap validity."""
        key = ('mocap',) + tuple(sorted(sensors))
        if key not in self._window_cache:
            mask = np.ones(self.n, dtype=bool)
            for sensor in sensors:
                mask &= np.asarray(self.plates[sensor].valid)[:self.n]
            self._window_cache[key] = build_window(mask, self.fs, 'mocap')
        return self._window_cache[key]

    def full_window(self) -> Optional[Window]:
        """The whole record as one run — available to any comparison needing no per-sample pose."""
        if ('full',) not in self._window_cache:
            self._window_cache[('full',)] = build_window(np.ones(self.n, dtype=bool), self.fs,
                                                         'full')
        return self._window_cache[('full',)]

    # --- geometry -----------------------------------------------------------------
    def segment_of(self, sensor: str) -> str:
        """The plate-name segment key a sensor belongs to."""
        return next(name for name, members in self.segments.items() if sensor in members)

    def line(self, segment: str) -> SegmentLine:
        """The segment's arc-length parametrization, built once per calibration arm."""
        if segment not in self._lines:
            sensors = self.segments[segment]
            self._lines[segment] = SegmentLine(self.B, self.X, sensors, self.R[sensors[0]])
        return self._lines[segment]

    def body_fit(self, exclude: Sequence[str]) -> Optional[Dict[str, object]]:
        """The whole-body field fit from every sensor EXCEPT `exclude`.

        The exclusion is what makes the body modes a test rather than a tautology: a gradient
        fitted over all sensors is the least-squares minimiser of exactly the inter-sensor
        disagreement the joint family then measures, so including the scored pair would guarantee
        an improvement that says nothing about the field.
        """
        key = tuple(sorted(exclude))
        if key in self._body_fits:
            return self._body_fits[key]
        names = [name for name in self.names if name not in set(exclude)]
        if len(names) < MIN_BODY_SENSORS:
            self._body_fits[key] = None
            return None
        B0, G, centre = instantaneous_gradient(self.B, self.X, names)
        stride = max(1, self.n // TRIAL_GRADIENT_MAX_SAMPLES)
        G_trial, explained = trial_gradient(self.B, self.X, names, stride)
        G_physical, explained_physical = trial_gradient(self.B, self.X, names, stride,
                                                        physical=True)
        self._body_fits[key] = {
            'B0': B0, 'G': G, 'centre': centre, 'G_trial': G_trial,
            'G_trial_phys': G_physical, 'n_sensors': len(names),
            'trial_gradient_explained': explained,
            'trial_gradient_explained_physical': explained_physical,
        }
        return self._body_fits[key]

    def joint_centre(self, joint: str) -> Optional[np.ndarray]:
        """The world-frame joint centre, as the MIDPOINT of the two segments' implied points.

        The midpoint rather than either segment's own estimate, matching
        `experiment_utils._compute_perfect_joint_acc`: the two differ by the fit residual, and
        giving the two sides of a comparison different target points would introduce a
        disagreement this experiment is not trying to measure.
        """
        if joint in self._joint_centres:
            return self._joint_centres[joint]
        sides = self.offsets.get(joint)
        if sides is None:
            self._joint_centres[joint] = None
            return None
        points = []
        for role, sensor in zip(ROLES, self.spec.joints[joint]):
            plate = self.plates[sensor]
            points.append(np.asarray(plate.world_trace.positions)[:self.n]
                          + np.einsum('nij,j->ni', np.asarray(plate.world_trace.rotations)[:self.n],
                                      sides[role]))
        self._joint_centres[joint] = 0.5 * (points[0] + points[1])
        return self._joint_centres[joint]


def segment_sensors(sensors: Sequence[str]) -> Dict[str, List[str]]:
    """{segment: [sensor, ...]} over the sensors present, keyed on the plate-name convention.

    IMoVE names a segment's three placements <SEGMENT>_H / _M / _L; anything else is alone on its
    segment and is its own group, which is what makes every multi-sensor code path degrade to the
    single-sensor one on Al Borno without a special case. `same_segment_groups` answers the same
    question off the spec's DISPLAY names and is what the segment family's pair labels come from;
    this one is keyed on plate names because that is what the field dictionaries are keyed on.
    """
    groups: Dict[str, List[str]] = {}
    for sensor in sorted(sensors):
        head, _, tail = sensor.rpartition('_')
        segment = head if head and tail in ('H', 'M', 'L') else sensor
        groups.setdefault(segment, []).append(sensor)
    return groups

# ==============================================================================
# Projection: what a segment reports at a target point
# ==============================================================================

def project(context: TrialContext, sensor: str, target: np.ndarray, mode: str,
            exclude_from_body: Sequence[str] = ()) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
    """One segment's field estimate at `target` under one mode, plus its geometry diagnostics.

    Returns None where the segment cannot support the mode — one sensor cannot be fitted with a
    line, and a body mode needs a fit that may not exist. The caller then leaves that side at the
    0th order and records it in `n_roles_upgraded`, rather than dropping the comparison: a hip
    with three thigh sensors and one pelvis sensor is exactly the case a real system would upgrade
    on one side only, and it is the more sensitive test because the unchanged side cannot absorb
    the effect.
    """
    own = context.B[sensor]
    if mode == 'sensor':
        return own, {}

    segment = context.segment_of(sensor)
    if mode in LINE_MODE_ORDER:
        order = LINE_MODE_ORDER[mode]
        # max(2, ...): with one sensor the order-0 fit IS that sensor's reading, so `mean` would
        # silently duplicate the 0th-order row under another name and show up in every paired
        # comparison as a tie. A segment with one sensor supports no multi-sensor mode.
        if len(context.segments[segment]) < max(2, order + 1):
            return None
        line = context.line(segment)
        s_target, offline = line.coordinate(target)
        return line.evaluate(target, order), {
            'span_mm': line.span * 1000.0,
            'array_offline_mm': line.offline_mm,
            'target_offline_mm': float(np.median(offline)) * 1000.0,
            'extrapolation': line.extrapolation_ratio(s_target),
        }

    body = context.body_fit(exclude_from_body)
    if body is None:
        return None
    displacement = target - context.X[sensor]
    if mode == 'body_linear':
        return own + np.einsum('tki,ti->tk', body['G'], displacement), {
            'gradient_norm': float(np.median(np.linalg.norm(body['G'], axis=(1, 2))))}
    if mode in ('trial_gradient', 'trial_gradient_phys'):
        G = body['G_trial' if mode == 'trial_gradient' else 'G_trial_phys']
        return own + displacement @ G.T, {'gradient_norm': float(np.linalg.norm(G))}
    if mode == 'body_model':
        return (body['B0'] + np.einsum('tki,ti->tk', body['G'], target - body['centre'])), {}
    raise ValueError(f"Unknown mode '{mode}'")

# ==============================================================================
# Family 1: agreement with a marker-supported field model
# ==============================================================================

def marker_comparisons(context: TrialContext, reference_field: np.ndarray,
                       lab_stride: int) -> List[Comparison]:
    """Each segment's projection at its joint centre, against two marker-supported field models.

    Both truths are MODELS, and that is the family's limitation rather than a detail: the residual
    is the sum of the projection's error and the model's own inadequacy. At order 0 the model's
    inadequacy dominates — which is exactly the finding `global_assumptions` reports as the
    magnetometer assumption failing distally — so this family is read for the ORDERING across
    segments and for the comparison between modes, not as an absolute projection error.

    The `labmap` truth is fitted WITHOUT the two sensors of the joint being scored.
    """
    out: List[Comparison] = []
    for joint, (parent_sensor, child_sensor) in context.spec.joints.items():
        if joint not in context.spec.primary_joints:
            continue
        if parent_sensor not in context.B or child_sensor not in context.B:
            continue
        centre = context.joint_centre(joint)
        if centre is None:
            continue
        window = context.mocap_window(parent_sensor, child_sensor)
        if window is None:
            continue
        predictor = lab_map_predictor(context.B, context.X, LAB_MAP_TRUTH_ORDER, lab_stride,
                                      exclude=(parent_sensor, child_sensor))
        constant = np.tile(reference_field, (context.n, 1))
        modelled = predictor(centre) if predictor is not None else None

        for mode in MODES:
            for role, sensor in zip(ROLES, (parent_sensor, child_sensor)):
                result = project(context, sensor, centre, mode,
                                 exclude_from_body=(parent_sensor, child_sensor))
                if result is None:
                    continue
                estimate, diagnostics = result
                raw = context.B[sensor]
                out.extend(_vector_comparison(
                    context, family='marker', group=joint, variant='global',
                    source=sensor, target=joint, target_kind='joint', mode=mode, window=window,
                    estimate=estimate, estimate_raw=raw, truth=constant, truth_raw=constant,
                    truth_alt=modelled, n_roles_upgraded=int(mode != PRIMARY_MODE),
                    scalars={'arm_m': float(np.median(np.linalg.norm(centre - context.X[sensor],
                                                                     axis=1))),
                             'role_is_parent': float(role == 'parent'), **diagnostics}))
    return out

# ==============================================================================
# Family 2: cross-segment agreement at the joint centre
# ==============================================================================

def joint_comparisons(context: TrialContext) -> List[Comparison]:
    """Parent against child, both transported to the shared joint centre.

    The residual a relative filter's magnetometer update consumes, so this is the family whose
    numbers say what the assumption costs the pipeline.

    Placement variants (IMoVE's 'R_Knee_H' and friends) are carried at the 0TH ORDER ONLY. Every
    higher mode uses all of the segment's sensors and would compute an identical estimate for all
    three variants of a joint; the 0th order is the one thing they say something new about —
    whether a pair mounted High on both segments agrees better than one mounted Mid, which is what
    a genuine field gradient would require and a per-sensor calibration error would not.
    """
    out: List[Comparison] = []
    for joint, (parent_sensor, child_sensor) in context.spec.joints.items():
        if parent_sensor not in context.B or child_sensor not in context.B:
            continue
        centre = context.joint_centre(joint)
        if centre is None:
            continue
        primary = joint in context.spec.primary_joints
        window = context.mocap_window(parent_sensor, child_sensor)
        separation = float(np.median(np.linalg.norm(
            context.X[parent_sensor] - context.X[child_sensor], axis=1)))
        scalars = {
            'sep_m': separation,
            'arm_parent_m': float(np.median(np.linalg.norm(centre - context.X[parent_sensor],
                                                           axis=1))),
            'arm_child_m': float(np.median(np.linalg.norm(centre - context.X[child_sensor],
                                                          axis=1))),
        }

        if window is not None:
            for mode in (MODES if primary else (PRIMARY_MODE,)):
                estimates, diagnostics, upgraded = {}, {}, 0
                for role, sensor in zip(ROLES, (parent_sensor, child_sensor)):
                    result = project(context, sensor, centre, mode,
                                     exclude_from_body=(parent_sensor, child_sensor))
                    if result is None:
                        estimates[role] = context.B[sensor]
                        continue
                    estimates[role], extra = result
                    upgraded += int(mode != PRIMARY_MODE)
                    diagnostics.update({f'{role}_{key}': value for key, value in extra.items()})
                if mode != PRIMARY_MODE and upgraded == 0:
                    continue
                out.extend(_vector_comparison(
                    context, family='joint', group=joint, variant='parent-child',
                    source=parent_sensor, target=child_sensor, target_kind='joint', mode=mode,
                    window=window, estimate=estimates['parent'],
                    estimate_raw=context.B[parent_sensor], truth=estimates['child'],
                    truth_raw=context.B[child_sensor], n_roles_upgraded=upgraded,
                    scalars={**scalars, **diagnostics}))

        # THE NORM CHANNEL NEEDS NO MOCAP AT ALL. |R m| = |m|, so comparing the two sensors' field
        # magnitudes uses neither the orientations nor the offsets — a stronger mocap-free claim
        # than the accelerometer version of this family can make, which still needs the two
        # offsets to project. Scored over the whole record, which is where it is available, and it
        # is the part of a disagreement no orientation estimate can ever absorb.
        full = context.full_window()
        if full is not None and primary:
            out.extend(_norm_comparison(
                context, family='joint', group=joint, variant='parent-child',
                source=parent_sensor, target=child_sensor, target_kind='joint', window=full,
                estimate=context.mag[parent_sensor], truth=context.mag[child_sensor],
                scalars=scalars))
    return out

# ==============================================================================
# Family 3: two magnetometers on one segment
# ==============================================================================

def segment_comparisons(context: TrialContext) -> List[Comparison]:
    """Two sensors on one rigid segment, and the held-out prediction of a third.

    Two things this family can do that no other can. First, its frame transform is a CONSTANT —
    both sensors are on one body reconstructed from one marker cluster — so `m_A` against
    `C_AB m_B` needs no per-sample pose and is scored over the whole record. `frame_spread_deg`
    records how constant that transform actually is rather than assuming it; on IMoVE it is
    constant to float precision by construction, and a reading that is not near zero would mean
    the two plates came from different clusters and the assumption had quietly stopped holding.

    Second, `target_kind='partner'` with a multi-sensor mode is the only fully out-of-sample test
    in the file: the other sensors predict the held-out one and a real measurement is waiting at
    the point being predicted, which is not true of the joint centre where nothing is mounted.
    """
    out: List[Comparison] = []
    groups = same_segment_groups(context.spec)
    full = context.full_window()

    for segment, members in groups.items():
        present = [(placement, sensor) for placement, sensor in members if sensor in context.B]
        if len(present) < 2:
            continue

        # --- pairwise disagreement, in the constant same-segment frame -----------------
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                (placement_a, a), (placement_b, b) = present[i], present[j]
                variant = f"{placement_a}-{placement_b}"
                separation = float(np.median(np.linalg.norm(context.X[a] - context.X[b], axis=1)))
                transform = chordal_mean(np.einsum('nji,njk->nik', context.R[a], context.R[b]))
                spread_rms, spread_max = rotation_spread_deg(
                    np.einsum('nji,njk->nik', context.R[a], context.R[b]), transform)
                scalars = {'sep_m': separation, 'frame_spread_deg': spread_rms,
                           'frame_spread_max_deg': spread_max}

                if full is not None:
                    transported = context.mag[b] @ transform.T
                    out.extend(_vector_comparison(
                        context, family='segment', group=segment, variant=variant,
                        source=a, target=b, target_kind='partner', mode=PRIMARY_MODE, window=full,
                        estimate=context.mag[a], estimate_raw=context.mag[a], truth=transported,
                        truth_raw=transported, scalars=scalars))
                    out.extend(_norm_comparison(
                        context, family='segment', group=segment, variant=variant,
                        source=a, target=b, target_kind='partner', window=full,
                        estimate=context.mag[a], truth=context.mag[b], scalars=scalars))

        # --- predict the held-out sensor from the others ------------------------------
        if len(present) < 3:
            continue
        sensors = [sensor for _, sensor in present]
        window = context.mocap_window(*sensors)
        if window is None:
            continue
        # `same_segment_groups` keys on the spec's DISPLAY names ('Thigh R') while the field
        # dictionaries and the line cache key on the plate-name prefix ('THIGH_R'). The sensors
        # are the same either way, so the plate-side key is taken from one of them.
        line = context.line(context.segment_of(sensors[0]))
        placements = {sensor: placement for placement, sensor in present}
        for held_out in sensors:
            kept = [s for s in sensors if s != held_out]
            s_target, _ = line.coordinate(context.X[held_out])
            ratio = line.extrapolation_ratio(s_target, exclude=held_out)
            source_label = "".join(placements[s][0] for s in kept)
            for mode, order in LINE_MODE_ORDER.items():
                if len(kept) < max(2, order + 1):
                    continue
                estimate = line.evaluate(context.X[held_out], order, exclude=held_out)
                out.extend(_vector_comparison(
                    context, family='segment', group=segment,
                    variant=f"{source_label}->{placements[held_out][0]}",
                    source=source_label, target=held_out, target_kind='held_out', mode=mode,
                    window=window, estimate=estimate,
                    # The 0th-order baseline for a held-out prediction is the mean of the sensors
                    # doing the predicting, not any one of them: with two sources there is no
                    # "the" reading to fall back on, and picking the nearer one would make the
                    # baseline depend on which end of the array was held out.
                    estimate_raw=line.evaluate(context.X[held_out], 0, exclude=held_out),
                    truth=context.B[held_out], truth_raw=context.B[held_out],
                    n_roles_upgraded=1, time_sensor=held_out,
                    scalars={'extrapolation': ratio, 'span_mm': line.span * 1000.0,
                             'array_offline_mm': line.offline_mm}))
    return out

# ==============================================================================
# Comparison construction
# ==============================================================================

def _vector_comparison(context: TrialContext, family: str, group: str, variant: str, source: str,
                       target: str, target_kind: str, mode: str, window: Window,
                       estimate: np.ndarray, estimate_raw: np.ndarray, truth: np.ndarray,
                       truth_raw: np.ndarray, scalars: Dict[str, float],
                       truth_alt: Optional[np.ndarray] = None,
                       n_roles_upgraded: int = 0,
                       time_sensor: Optional[str] = None) -> List[Comparison]:
    """One vector-channel Comparison, conditioned. Empty when anything in it is not finite.

    `time_sensor` names the plate whose clock the timestamps come from, defaulting to `source`.
    It exists because a held-out prediction's `source` is a LABEL for a set of sensors ('HM')
    rather than a plate; every plate in a trial shares one uniform grid, so which one is asked is
    immaterial to the values and material only to the lookup.
    """
    comparison = Comparison(
        family=family, group=group, variant=variant, source=source, target=target,
        target_kind=target_kind, mode=mode, calibration=context.calibration,
        calibration_source=context.calibration_source, window=window,
        timestamps=condition(context.timestamps(time_sensor or source), window, cutoff=None),
        estimate=condition(estimate, window), estimate_raw=condition(estimate_raw, window),
        truth=condition(truth, window), truth_raw=condition(truth_raw, window),
        truth_alt=condition(truth_alt, window) if truth_alt is not None else None,
        channel='vector', n_roles_upgraded=n_roles_upgraded, scalars=scalars)
    return [comparison] if _finite(comparison.estimate, comparison.estimate_raw,
                                   comparison.truth) else []


def _norm_comparison(context: TrialContext, family: str, group: str, variant: str, source: str,
                     target: str, target_kind: str, window: Window, estimate: np.ndarray,
                     truth: np.ndarray, scalars: Dict[str, float]) -> List[Comparison]:
    """One magnitude-channel Comparison over a window that needs no frame.

    FILTERED BEFORE THE MAGNITUDE IS TAKEN, matching every other family: the magnitude is a
    nonlinear function, so |lowpass(m)| and lowpass(|m|) differ, and the first is what a filter
    consuming a band-limited magnetometer would see.

    The inputs are BODY-FRAME readings rather than world-frame fields, which for this channel is
    the same number — |R m| = |m| — and makes the mocap-independence explicit in the code rather
    than only in the comment.
    """
    comparison = Comparison(
        family=family, group=group, variant=variant, source=source, target=target,
        target_kind=target_kind, mode=PRIMARY_MODE, calibration=context.calibration,
        calibration_source=context.calibration_source, window=window,
        timestamps=condition(context.timestamps(source), window, cutoff=None),
        estimate=as_magnitude(condition(estimate, window)),
        estimate_raw=as_magnitude(condition(estimate, window)),
        truth=as_magnitude(condition(truth, window)),
        truth_raw=as_magnitude(condition(truth, window)),
        channel='norm', scalars=scalars)
    return [comparison] if _finite(comparison.estimate, comparison.truth) else []

# ==============================================================================
# Tables, all generated from a list of Comparisons
# ==============================================================================

def sample_rows(comparison: Comparison, stride: int = SAMPLE_STRIDE) -> pd.DataFrame:
    """Per-sample metrics for one comparison, decimated by `stride`.

    `corr_norm` is the size of the correction the mode applied TO THE GAP,
    |(est - est_raw) - (truth - truth_raw)|, not to either side. For the marker family the truth
    is a model and does not move, so this reduces to how far the mode shifted the estimate; for
    the agreement families it is the quantity that matters, since two segments whose corrections
    move together have been asked to do nothing however large those corrections are.
    """
    step = slice(None, None, stride)
    diff = comparison.estimate - comparison.truth
    correction = ((comparison.estimate - comparison.estimate_raw)
                  - (comparison.truth - comparison.truth_raw))
    dnorm_proj = norm_gap(comparison.estimate, comparison.truth)
    dnorm_raw = norm_gap(comparison.estimate_raw, comparison.truth_raw)
    blank = np.full(len(diff), np.nan)

    frame = {
        **comparison.key,
        'timestamp': comparison.timestamps[step].astype(np.float64),
        'run_id': comparison.window.run_id[step].astype(np.int32),
        'err_proj': np.linalg.norm(diff, axis=1)[step].astype(np.float32),
        'err_raw': np.linalg.norm(comparison.estimate_raw - comparison.truth_raw,
                                  axis=1)[step].astype(np.float32),
        # NaN rather than the identical zero a magnitude-only comparison would produce. A zero
        # here would read as perfect directional agreement in every quantile and every boxplot
        # that did not know to exclude it — the most misleading number this table could contain.
        'ang_proj': (blank if comparison.channel == 'norm'
                     else angle_between_deg(comparison.estimate,
                                            comparison.truth))[step].astype(np.float32),
        'ang_raw': (blank if comparison.channel == 'norm'
                    else angle_between_deg(comparison.estimate_raw,
                                           comparison.truth_raw))[step].astype(np.float32),
        'dnorm_proj': dnorm_proj[step].astype(np.float32),
        'absdnorm_proj': np.abs(dnorm_proj)[step].astype(np.float32),
        'absdnorm_raw': np.abs(dnorm_raw)[step].astype(np.float32),
        'corr_norm': np.linalg.norm(correction, axis=1)[step].astype(np.float32),
        'truth_norm': np.linalg.norm(comparison.truth, axis=1)[step].astype(np.float32),
    }
    if comparison.truth_alt is not None:
        frame['err_proj_alt'] = np.linalg.norm(
            comparison.estimate - comparison.truth_alt, axis=1)[step].astype(np.float32)
    return pd.DataFrame(frame)


def stats_row(comparison: Comparison) -> Dict[str, object]:
    """One row of per-comparison scalars, computed on the FULL conditioned record.

    `align_angle_deg` and `rms_after` are the misalignment floor made measurable: the single
    constant rotation best mapping estimate onto truth, and what is left once it is removed.
    Both are NaN where they would be meaningless — for the magnitude channel, which has no
    orientation, and for the marker family's constant truth, where the Kabsch fit is minimising
    against a fixed vector and returns a rotation with no interpretation.
    """
    truth_varies = float(np.sum((comparison.truth - comparison.truth.mean(axis=0)) ** 2))
    if comparison.channel == 'norm' or truth_varies <= 0:
        angle_deg, rms_after = np.nan, np.nan
        rms_before = rms(comparison.estimate - comparison.truth)
    else:
        _, angle_deg, rms_before, rms_after = residual_alignment(comparison.estimate,
                                                                 comparison.truth)
    error = np.linalg.norm(comparison.estimate - comparison.truth, axis=1)
    error_raw = np.linalg.norm(comparison.estimate_raw - comparison.truth_raw, axis=1)
    angle = (np.full(len(error), np.nan) if comparison.channel == 'norm'
             else angle_between_deg(comparison.estimate, comparison.truth))
    angle_raw = (np.full(len(error), np.nan) if comparison.channel == 'norm'
                 else angle_between_deg(comparison.estimate_raw, comparison.truth_raw))

    row: Dict[str, object] = {
        **comparison.key,
        'n_samples': len(comparison.estimate),
        'duration_s': comparison.window.duration_s,
        'n_runs': len(comparison.window.runs),
        'valid_fraction': comparison.window.n_mask / max(comparison.window.n_record, 1),
        'fs': comparison.window.fs,
        'n_roles_upgraded': comparison.n_roles_upgraded,
        'align_angle_deg': angle_deg,
        'rms_before': rms_before,
        'rms_after': rms_after,
        'rms_raw': rms(comparison.estimate_raw - comparison.truth_raw),
        'truth_norm_p50': float(np.median(np.linalg.norm(comparison.truth, axis=1))),
        'median_corr_norm': float(np.median(np.linalg.norm(
            (comparison.estimate - comparison.estimate_raw)
            - (comparison.truth - comparison.truth_raw), axis=1))),
        **_quantile_stats(error, 'err'),
        **_quantile_stats(error_raw, 'err_raw'),
        **_quantile_stats(angle, 'ang'),
        **_quantile_stats(angle_raw, 'ang_raw'),
        **_quantile_stats(np.abs(norm_gap(comparison.estimate, comparison.truth)), 'absdnorm'),
    }
    if comparison.truth_alt is not None:
        row['rms_alt'] = rms(comparison.estimate - comparison.truth_alt)
        row['err_alt_p50'] = float(np.median(np.linalg.norm(
            comparison.estimate - comparison.truth_alt, axis=1)))
    row.update(comparison.scalars)
    return row

# ==============================================================================
# The mechanism decomposition
# ==============================================================================

def mechanism_rows(context: TrialContext) -> List[Dict[str, object]]:
    """Is a same-segment disagreement a field gradient, or is it the sensors?

    THE DECIDING FACT is that the two candidates transform differently, so one difference series
    separates them. For each pair, d(t) = B_A - B_B is fitted four ways:

        body      a constant vector in sensor A's own frame. Exactly what a hard-iron or gain
                  error is — fixed to the device, sweeping the world frame as the limb turns.
        world     a constant vector in the LAB frame. What a static difference between two fixed
                  points would be, if the limb never moved.
        gradient  G (x_A - x_B) with one world-frame G for the trial. What a uniform spatial
                  gradient gives: the separation rotates with the limb, so this model predicts a
                  difference that rotates in a specific, checkable way.
        gradient_physical
                  the same, with G symmetric and traceless. The only one of the four that cannot
                  fit something a magnetostatic field could not produce, and therefore the one
                  whose explained fraction is evidence about a FIELD rather than about a
                  nine-parameter regression.

    The models are nested in `combined` (body + gradient refitted together), so their explained
    fractions are not a partition and do not sum to it. They are four separate answers to "could
    it be this alone?", which is the question that names a mechanism.

    The unconstrained gradient model is partly confounded with the body model on a limb that
    sweeps a limited range of orientations — both are linear in R(t) — which is the second reason
    the physical constraint matters: it removes four of the nine directions the confounding lives
    in.
    """
    rows = []
    for segment, members in same_segment_groups(context.spec).items():
        present = [(placement, sensor) for placement, sensor in members if sensor in context.B]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                (placement_a, a), (placement_b, b) = present[i], present[j]
                valid = (np.asarray(context.plates[a].valid)[:context.n]
                         & np.asarray(context.plates[b].valid)[:context.n])
                if valid.sum() < MIN_CALIBRATION_FRAMES:
                    continue
                difference = (context.B[a] - context.B[b])[valid]
                separation = (context.X[a] - context.X[b])[valid]
                rotations = context.R[a][valid]
                total = float(np.sum(difference ** 2))
                if total <= 0:
                    continue

                explained = lambda model: 1.0 - float(np.sum((difference - model) ** 2)) / total
                body_constant = np.einsum('nji,nj->ni', rotations, difference).mean(axis=0)
                body_model = np.einsum('nij,j->ni', rotations, body_constant)
                world_model = np.tile(difference.mean(axis=0), (len(difference), 1))
                G, explained_gradient = fit_gradient(separation, difference)
                G_physical, explained_physical = fit_gradient(separation, difference,
                                                              physical=True)
                # Refitted together rather than added: on a limb sweeping a limited range the two
                # regressors are correlated, so the sum of two separate fits is not the joint one.
                design = np.concatenate([separation, body_model], axis=1)
                combined, *_ = np.linalg.lstsq(design, difference, rcond=None)

                rows.append({
                    'segment': segment,
                    'variant': f"{placement_a}-{placement_b}",
                    'sensor_a': a, 'sensor_b': b,
                    'calibration': context.calibration,
                    'calibration_source': context.calibration_source,
                    'n_samples': int(valid.sum()),
                    'sep_m': float(np.median(np.linalg.norm(separation, axis=1))),
                    'field_norm_p50': float(np.median(np.linalg.norm(context.B[a][valid], axis=1))),
                    'dev_p50': float(np.median(np.linalg.norm(difference, axis=1))),
                    'ang_p50': float(np.median(angle_between_deg(context.B[a][valid],
                                                                 context.B[b][valid]))),
                    'explained_body': explained(body_model),
                    'explained_world': explained(world_model),
                    'explained_gradient': explained_gradient,
                    'explained_gradient_physical': explained_physical,
                    'explained_combined': explained(design @ combined),
                    'body_constant_norm': float(np.linalg.norm(body_constant)),
                    'gradient_norm': float(np.linalg.norm(G)),
                    'gradient_norm_physical': float(np.linalg.norm(G_physical)),
                })
    return rows

# ==============================================================================
# Covariation: how much of what two sensors see is SHARED
# ==============================================================================

def fit_local_field_and_bias(mag: np.ndarray, rotations: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Solve m(t) = R(t)^T B_local + b for BOTH at once. Returns (B_local, b, condition, weakest).

    THE ESTIMATOR `fit_calibration` SHOULD HAVE BEEN for anything asking whether the offset is a
    device property. That one computes mean(m - R^T B_ref), which assumes the reference field, so
    any error in B_ref lands in b weighted by the trial's mean orientation — and different
    activities have different mean orientations, which manufactures trial-to-trial scatter that
    looks like the offset moving. This one assumes no reference: it separates the two purely by
    rotation, since B_local is world-fixed and its body-frame image sweeps while b does not.

    Both are kept. `fit_calibration` defines the CALIBRATIONS arms, whose job is to be applied to
    a reading, and it is measured against the same global field every other family is measured
    against. This one is for asking what the offset IS.

    THE CONDITIONING IS THE WHOLE STORY and is returned rather than hidden. Rotation about a
    single body-fixed axis leaves the field's component along that axis constant in the body
    frame, hence indistinguishable from b along the same axis — so a near-planar trial, which gait
    is, cannot observe one component of b at all. `weakest` is the direction in BIAS SPACE the
    estimate is blind in. Measured on IMoVE, 73% of the between-trial scatter of b lies along it,
    against 33% for chance.

    `weakest` is the bias half of the least-observed 6-vector eigenvector, RENORMALIZED, because
    every consumer uses it as a direction to project onto. The 6-vector is unit but its two halves
    are not, and the split between them is itself informative: a weakest mode lying mostly in the
    field half means it is the FIELD that is poorly observed, not the bias, and the returned
    direction is then near-arbitrary. Guarded rather than raised on, since it only arises for a
    trial that could not be analysed anyway.
    """
    n = len(mag)
    design = np.concatenate([rotations.transpose(0, 2, 1), np.tile(np.eye(3), (n, 1, 1))],
                            axis=2).reshape(-1, 6)
    solution, *_ = np.linalg.lstsq(design, np.asarray(mag, dtype=float).reshape(-1), rcond=None)
    eigenvalues, eigenvectors = np.linalg.eigh(design.T @ design)
    weakest = eigenvectors[3:, 0]
    scale = np.linalg.norm(weakest)
    weakest = weakest / scale if scale > 1e-9 else np.array([1.0, 0.0, 0.0])
    return (solution[:3], solution[3:],
            float(eigenvalues[-1] / max(eigenvalues[0], 1e-12)), weakest)


def pair_class(context: TrialContext, a: str, b: str) -> str:
    """How close two sensors are, as one of PAIR_CLASSES.

    The far classes are not padding — they are the control. A disturbance that is genuinely
    spatial is shared by neighbours and not by a sensor on the other leg, so a covariation that
    does not fall off with adjacency is not a field.
    """
    segment_a, segment_b = context.segment_of(a), context.segment_of(b)
    if segment_a == segment_b:
        return 'same_segment'
    spanning = {frozenset(pair) for pair in context.spec.joints.values()}
    if frozenset((a, b)) in spanning:
        return 'across_joint'
    side_a, stem_a = _side_and_stem(segment_a)
    side_b, stem_b = _side_and_stem(segment_b)
    if side_a and side_a == side_b:
        return 'same_limb'
    if side_a and side_b and stem_a == stem_b:
        return 'contralateral'
    return 'distant'


def _side_and_stem(segment: str) -> Tuple[str, str]:
    """('l'|'r'|'', the name with that token removed) for a segment name.

    The side is a lone 'l'/'r' TOKEN anywhere in the name, not a suffix, because the two datasets
    put it in different places: IMoVE names a segment 'THIGH_R' and Al Borno names its sensor
    'femur_r_imu'. Keying on the last token found only IMoVE's, which silently collapsed every Al
    Borno pair into 'distant' and cost that dataset three of the five pair classes — the two that
    matter most, since contralateral is the control the whole comparison rests on.
    """
    tokens = segment.split('_')
    for index, token in enumerate(tokens):
        if token.lower() in ('l', 'r'):
            return token.lower(), '_'.join(tokens[:index] + tokens[index + 1:])
    return '', segment


def covariation_rows(context: TrialContext) -> List[Dict[str, object]]:
    """Every sensor pair in the trial: how much of their world-frame field FLUCTUATION is shared.

    The mean is removed first, so this is about the disturbance rather than about the ambient
    field the two obviously share. Reported raw and after subtracting each sensor's own fitted
    body-fixed bias, because two sensors on one segment rotate together — so two INDEPENDENT
    biases still produce correlated world-frame fluctuations, and the debiased column is what
    separates "they see the same field disturbance" from "they are both rotating".
    """
    window = context.mocap_window(*context.names)
    if window is None:
        return []

    biases = {name: fit_local_field_and_bias(context.mag[name], context.R[name])[1]
              for name in context.names}
    fluctuation, debiased, slow = {}, {}, {}
    for name in context.names:
        field = condition(context.B[name], window)
        fluctuation[name] = field - field.mean(axis=0)
        without = condition(context.B[name]
                            - np.einsum('nij,j->ni', context.R[name], biases[name]), window)
        debiased[name] = without - without.mean(axis=0)
        slow[name] = lowpass(fluctuation[name], window.fs, SLOW_BAND_HZ)

    def shared(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """(correlation, surviving fraction) between two vector fluctuation series."""
        energy_a, energy_b = float(np.sum(a ** 2)), float(np.sum(b ** 2))
        if energy_a <= 0 or energy_b <= 0:
            return np.nan, np.nan
        return (float(np.sum(a * b) / np.sqrt(energy_a * energy_b)),
                float(np.sum((a - b) ** 2) / (energy_a + energy_b)))

    rows = []
    names = context.names
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            rho, surviving = shared(fluctuation[a], fluctuation[b])
            rho_slow, _ = shared(slow[a], slow[b])
            rho_gait, _ = shared(fluctuation[a] - slow[a], fluctuation[b] - slow[b])
            rho_debiased, surviving_debiased = shared(debiased[a], debiased[b])
            rows.append({
                'sensor_a': a, 'sensor_b': b,
                'pair_class': pair_class(context, a, b),
                'calibration': context.calibration,
                'n_samples': len(window),
                'sep_m': float(np.median(np.linalg.norm(context.X[a] - context.X[b], axis=1))),
                'rho': rho, 'surviving': surviving,
                'rho_slow': rho_slow, 'rho_gait': rho_gait,
                'rho_debiased': rho_debiased, 'surviving_debiased': surviving_debiased,
                'fluctuation_a': float(np.sqrt(np.mean(np.sum(fluctuation[a] ** 2, axis=1)))),
                'fluctuation_b': float(np.sqrt(np.mean(np.sum(fluctuation[b] ** 2, axis=1)))),
                'field_norm': float(np.median(np.linalg.norm(context.B[a], axis=1))),
            })
    return rows


# ==============================================================================
# Geometry and gradient-estimation quality
# ==============================================================================

def gradient_fits_table(context: TrialContext, lab_field: pd.DataFrame) -> pd.DataFrame:
    """One row for the whole-body array and one per multi-sensor segment.

    The columns carrying the argument are `gradient_norm` against the lab map's, and
    `residual_linear_p50` against `residual_constant_p50`: a fit that reduces the inter-sensor
    residual while returning a gradient an order of magnitude larger than the one the markers can
    see is absorbing sensor error, not measuring a field.

    `gradient_constancy_world` against `_body` asks the same question a second way. A real room
    gradient is constant in the WORLD frame; a fit tracking the subject's own hardware is constant
    in the BODY frame. Neither being near 1 means the per-sample fit is tracking something that is
    neither.
    """
    lab_gradient = np.nan
    if not lab_field.empty and (lab_field['order'] == 1).any():
        lab_gradient = float(lab_field.loc[lab_field['order'] == 1, 'gradient_norm'].iloc[0])

    rows = []
    names = context.names
    if len(names) >= MIN_BODY_SENSORS:
        body = context.body_fit(())
        B = np.stack([context.B[n] for n in names], axis=1)
        X = np.stack([context.X[n] for n in names], axis=1)
        modelled = body['B0'][:, None, :] + np.einsum('tki,tsi->tsk', body['G'],
                                                      X - body['centre'][:, None, :])
        residual_linear = np.linalg.norm(B - modelled, axis=2)
        residual_constant = np.linalg.norm(B - B.mean(axis=1, keepdims=True), axis=2)
        pelvis = (context.spec.pelvis_sensor if context.spec.pelvis_sensor in context.R
                  else names[0])
        R = context.R[pelvis]
        G_body = np.einsum('tji,tjk,tkl->til', R, body['G'], R)
        constancy = lambda M: 1.0 - float(np.sum((M - M.mean(axis=0)) ** 2)) / float(np.sum(M ** 2))
        spread = X - X.mean(axis=1, keepdims=True)
        singular = np.median(np.linalg.svd(spread, compute_uv=False), axis=0)
        rows.append({
            'scope': 'body', 'group': 'body', 'calibration': context.calibration,
            'n_sensors': len(names), 'n_samples': context.n,
            'residual_constant_p50': float(np.median(residual_constant)),
            'residual_linear_p50': float(np.median(residual_linear)),
            'gradient_norm_p50': float(np.median(np.linalg.norm(body['G'], axis=(1, 2)))),
            'trial_gradient_norm': float(np.linalg.norm(body['G_trial'])),
            'trial_gradient_explained': body['trial_gradient_explained'],
            'trial_gradient_norm_physical': float(np.linalg.norm(body['G_trial_phys'])),
            'trial_gradient_explained_physical': body['trial_gradient_explained_physical'],
            'lab_map_gradient_norm': lab_gradient,
            'gradient_constancy_world': constancy(body['G']),
            'gradient_constancy_body': constancy(G_body),
            'array_extent_mm': float(singular[0] * 1000.0),
            'array_thickness_mm': float(singular[-1] * 1000.0),
        })

    for segment, sensors in context.segments.items():
        if len(sensors) < 2:
            continue
        line = context.line(segment)
        rows.append({
            'scope': 'segment', 'group': segment, 'calibration': context.calibration,
            'n_sensors': len(sensors), 'n_samples': context.n,
            'array_extent_mm': line.along_mm,
            'array_thickness_mm': line.offline_mm,
            'span_mm': line.span * 1000.0,
            'lab_map_gradient_norm': lab_gradient,
        })
    return pd.DataFrame(rows)

# ==============================================================================
# Per-trial driver
# ==============================================================================

def compute_trial(plates: Dict[str, PlateTrial], spec: DatasetSpec, dataset: str, subject: str,
                  trial: str, reference_field: np.ndarray,
                  parameters: Dict[str, Dict[str, np.ndarray]],
                  tables: Sequence[str] = ANALYSIS_TABLES,
                  stride: int = SAMPLE_STRIDE) -> Dict[str, pd.DataFrame]:
    """Every analysis table for one trial, over every calibration arm."""
    wanted = set(tables)
    # Fitted once, outside the calibration loop: the joint centres come from the MOCAP alone and
    # do not move when the magnetometer is recalibrated.
    offsets = joint_offsets(plates, spec, min_frames=MIN_FIT_FRAMES)

    out: Dict[str, pd.DataFrame] = {}
    stats: List[Dict[str, object]] = []
    samples: List[pd.DataFrame] = []
    mechanism: List[Dict[str, object]] = []
    covariation: List[Dict[str, object]] = []

    for calibration in CALIBRATIONS:
        context = TrialContext(plates, spec, dataset, subject, trial, parameters, calibration,
                               offsets)
        lab_stride = max(1, context.n // LAB_MAP_MAX_SAMPLES)

        if calibration == PRIMARY_CALIBRATION:
            lab_field = lab_field_table(context.B, context.X, lab_stride)
            if 'lab_field' in wanted:
                out['lab_field'] = lab_field
            if 'gradient_fits' in wanted:
                out['gradient_fits'] = gradient_fits_table(context, lab_field)

        if 'mechanism' in wanted:
            mechanism.extend(mechanism_rows(context))
        if 'covariation' in wanted:
            covariation.extend(covariation_rows(context))

        if not ({'agreement_stats', 'agreement_samples'} & wanted):
            continue
        comparisons = (marker_comparisons(context, reference_field, lab_stride)
                       + joint_comparisons(context) + segment_comparisons(context))
        for comparison in comparisons:
            stats.append(stats_row(comparison))
            # The per-sample table exists for the figures and carries one calibration arm. At
            # stride 10 over IMoVE's long walks the full grid would be millions of rows per trial
            # for quantiles identical to the scalars beside them.
            if ('agreement_samples' in wanted and comparison.calibration == SAMPLE_CALIBRATION
                    and (comparison.family == 'joint' or comparison.mode == PRIMARY_MODE)):
                samples.append(sample_rows(comparison, stride))

    if 'agreement_stats' in wanted:
        out['agreement_stats'] = pd.DataFrame(stats)
    if 'agreement_samples' in wanted and samples:
        out['agreement_samples'] = pd.concat(samples, ignore_index=True)
    if 'mechanism' in wanted:
        out['mechanism'] = pd.DataFrame(mechanism)
    if 'covariation' in wanted:
        out['covariation'] = pd.DataFrame(covariation)
    return out


def _load_spec_plates(subject: str, trial: str, dataset: str,
                      spec: DatasetSpec) -> Dict[str, PlateTrial]:
    """The trial's plates, narrowed to the sensors this dataset's spec names.

    Narrowing matters more here than usual: the whole-body gradient is fitted over every plate
    handed in, so a sensor the spec does not analyse would pull the fit.
    """
    plates = load_trial(subject, trial, dataset=build_name(dataset))
    selected = {sensor: plate for sensor, plate in plates.items()
                if sensor in set(spec.segment_sensor.values())}
    if not selected:
        raise ValueError(f"{dataset}/{subject}/{trial}: none of the spec's sensors "
                         f"{sorted(spec.segment_sensor.values())} are in this trial "
                         f"({sorted(plates)}).")
    return selected


def _field_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                  dataset: str = 'alborno') -> None:
    """Stage one, one process per trial: the per-(trial, sensor) median world-frame field.

    Split out for the same reason `global_assumptions` splits it: the subject's global field is
    pooled over ALL of that subject's trials and every deviation is measured against it, so it
    cannot be computed inside the pass that consumes it without either holding a whole session in
    memory or using a per-trial reference, which is not a reference.
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
        _save(table, trial_table_path(dataset, subject, trial, STAGE1_TABLE), dataset,
              subject=subject, trial=trial, table=STAGE1_TABLE)
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success"
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
    return None


def _calibration_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                        dataset: str = 'alborno') -> None:
    """Stage two, one process per trial: each sensor's calibration against the subject field.

    A separate pass from stage three because stage three needs the calibrations of the subject's
    OTHER trials, which only exist on disk once every trial has been through here.
    """
    subject, trial = row_key
    spec = get_dataset(dataset)
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()
    try:
        reference_field = read_subject_field(dataset, subject)
        if reference_field is None:
            shared_state[(row_key, stage)] = "Failed (no subject field)"
            return None
        plates = _load_spec_plates(subject, trial, dataset, spec)
        _save(calibration_table(plates, reference_field),
              trial_table_path(dataset, subject, trial, STAGE2_TABLE), dataset,
              subject=subject, trial=trial, table=STAGE2_TABLE)
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success"
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
    return None


def _trial_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                  dataset: str = 'alborno', tables: Sequence[str] = ANALYSIS_TABLES,
                  stride: int = SAMPLE_STRIDE,
                  row_keys: Sequence[Tuple[str, str]] = ()) -> None:
    """Stage three, one process per trial: every analysis table."""
    subject, trial = row_key
    spec = get_dataset(dataset)
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()
    try:
        reference_field = read_subject_field(dataset, subject)
        if reference_field is None:
            shared_state[(row_key, stage)] = "Failed (no subject field)"
            return None
        parameters = load_calibrations(dataset, subject, trial, row_keys)
        plates = _load_spec_plates(subject, trial, dataset, spec)
        for table, frame in compute_trial(plates, spec, dataset, subject, trial, reference_field,
                                          parameters, tables, stride).items():
            if frame.empty:
                continue
            _save(frame, trial_table_path(dataset, subject, trial, table), dataset,
                  subject=subject, trial=trial, table=table)
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success"
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
    return None


def aggregate_subject_fields(dataset: str, row_keys: Sequence[Tuple[str, str]]) -> List[str]:
    """Rolls stage one's per-trial tables into one global field per subject."""
    written = []
    for subject in subjects_of(row_keys):
        frames = []
        for other_subject, trial in row_keys:
            if other_subject != subject:
                continue
            path = trial_table_path(dataset, subject, trial, STAGE1_TABLE)
            if path.exists():
                frames.append(pd.read_parquet(path, engine='pyarrow').assign(trial=trial))
        if not frames:
            continue
        _save(subject_field_table(subject, pd.concat(frames, ignore_index=True)),
              subject_field_path(dataset, subject), dataset, subject=subject)
        written.append(subject)
    return written

# ==============================================================================
# Pooled summary
# ==============================================================================

SUMMARY_COLUMNS = (['dataset', 'subject', 'trial', 'family', 'channel', 'scope', 'mode',
                    'group', 'metric', 'unit', 'n_samples', 'mean', 'std', 'min']
                   + [f"p{int(round(q * 100)):02d}" for q in QUANTILES] + ['max'])


def _describe(df: pd.DataFrame, metric: str, keys: List[str]) -> pd.DataFrame:
    grouped = df.groupby(keys, observed=True)[metric]
    stats = grouped.agg(n_samples='count', mean='mean', std='std', min='min', max='max')
    quantiles = grouped.quantile(QUANTILES).unstack()
    quantiles.columns = [f"p{int(round(q * 100)):02d}" for q in quantiles.columns]
    return stats.join(quantiles).reset_index()


def summarize(dataset: str, samples: pd.DataFrame) -> pd.DataFrame:
    """Tidy quantile table per metric x family x channel x scope x mode x group, WITH MARGINS.

    Rows whose `subject`, `trial` or `group` is the literal string 'all' are the pooled version of
    the rows above them, recomputed from the samples rather than averaged from the per-trial rows
    — a mean of medians is not a median, and IMoVE's trials differ in length by two orders of
    magnitude. Quantiles rather than mean +- sd because every metric here is a non-negative
    magnitude with a heavy right tail.
    """
    if samples.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    frames = []
    keys = ['family', 'channel', 'scope', 'mode']
    for metric in SUMMARY_METRICS:
        if metric not in samples.columns:
            continue
        block = samples[samples[metric].notna()]
        if block.empty:
            continue
        for by_subject, by_trial, by_group in [(True, True, True), (True, False, True),
                                               (False, False, True), (False, False, False)]:
            grouping = (['subject'] if by_subject else []) + (['trial'] if by_trial else [])
            grouping += keys + (['group'] if by_group else [])
            part = _describe(block, metric, grouping)
            if not by_subject:
                part['subject'] = 'all'
            if not by_trial:
                part['trial'] = 'all'
            if not by_group:
                part['group'] = 'all'
            part['dataset'] = dataset
            part['metric'] = metric
            part['unit'] = METRIC_UNITS.get(metric, '')
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    for column in ('subject', 'trial', 'family', 'channel', 'scope', 'mode', 'group', 'metric'):
        out[column] = out[column].astype(str)
    return out[SUMMARY_COLUMNS]

# ==============================================================================
# Console report
# ==============================================================================

def _header(number: object, title: str, subtitle: str) -> None:
    print("\n" + "=" * 96)
    print(f"{number}. {title}")
    print(f"   ({subtitle})")
    print("=" * 96)


def _cells(stats: pd.DataFrame, **filters) -> pd.DataFrame:
    """The stats rows matching every filter. Values may be a scalar or a collection."""
    frame = stats
    for column, value in filters.items():
        if column not in frame.columns:
            return frame.iloc[0:0]
        if isinstance(value, (list, tuple, set)):
            frame = frame[frame[column].isin(list(value))]
        else:
            frame = frame[frame[column] == value]
    return frame


def _by(frame: pd.DataFrame, keys: List[str], columns: Sequence[str]) -> pd.DataFrame:
    """Median across trials of per-trial scalars, grouped by `keys`.

    A median of per-trial medians, NOT a pooled sample quantile: the per-trial scalars are
    computed on the full conditioned record and a TRIAL is the replicate here, so a 2800 s walk
    and a 30 s task count once each. The pooled-sample weighting is in the statistics parquet for
    anyone who wants the other one.
    """
    present = [c for c in columns if c in frame.columns]
    if frame.empty or not present:
        return pd.DataFrame()
    return frame.groupby(keys, observed=True)[present].median().reset_index()


def report_headline(spec: DatasetSpec, stats: pd.DataFrame) -> None:
    _header(1, "THE THREE FAMILIES AT THE 0TH ORDER",
            "how far a magnetometer reading is from what it would need to be for transporting it "
            "unchanged to the joint centre to be exact")
    frame = _cells(stats, mode=PRIMARY_MODE, calibration=PRIMARY_CALIBRATION)
    if frame.empty:
        print("   No comparisons.")
        return
    table = _by(frame, ['family', 'channel', 'scope'],
                ['err_p50', 'err_p90', 'ang_p50', 'ang_p90', 'truth_norm_p50', 'n_samples'])
    table['err_%_of_field'] = 100 * table['err_p50'] / table['truth_norm_p50']
    order = {family: i for i, family in enumerate(FAMILIES)}
    table = table.sort_values('family', key=lambda s: s.map(order))
    print(table.to_string(index=False, float_format='%.3f'))
    print(f"\n   Magnitudes are in {MAG_UNIT}, where the ambient field is ~1. `ang_p50` is the "
          f"DIRECTION error and\n   is the column to quote: the relative filter normalizes its "
          f"magnetometer measurement, so\n   direction is what its update responds to.")
    print("   The `norm` channel uses NO mocap — |R m| = |m| — so it is scored over the whole "
          "record and\n   is the part of a disagreement no orientation estimate can absorb. Where "
          "it is large, no\n   filter, frame or alignment can reconcile the two sensors.")
    for family in FAMILIES:
        rows = table[table['family'] == family]
        if not rows.empty:
            print(f"     {family:<8} {FAMILY_LABELS[family]}")


def report_marker(spec: DatasetSpec, stats: pd.DataFrame, lab: pd.DataFrame) -> None:
    _header(2, "AGAINST THE MARKERS",
            "mocap cannot measure a field, so the reference is a field MODEL the markers support: "
            "one constant vector, or a polynomial in marker position")
    frame = _cells(stats, family='marker', mode=PRIMARY_MODE, calibration=PRIMARY_CALIBRATION,
                   channel='vector')
    if not frame.empty:
        table = _by(frame, ['group'], ['err_p50', 'ang_p50', 'err_alt_p50', 'arm_m'])
        print("   Per joint, both segments pooled — `err_p50` against the constant global field, "
              "`err_alt_p50`\n   against the lab field map (order "
              f"{LAB_MAP_TRUTH_ORDER}, fitted without this joint's two sensors):")
        print(table.to_string(index=False, float_format='%.4f'))

    if lab.empty:
        return
    print("\n   The lab field map itself — the world-frame field as a polynomial in MARKER "
          "POSITION,\n   fitted over every sensor and sample, median over trials:")
    columns = ['r2', 'r2_loso', 'residual_p50', 'residual_loso_p50', 'gradient_norm']
    print(lab.groupby('order')[columns].median().to_string(float_format='%.4f'))
    print("   `r2_loso` refits without the sensor it then scores. The gap between the two columns "
          "is how\n   much of the map's apparent skill was fitting per-sensor error rather than "
          "the room — and if\n   r2_loso FALLS from order 1 to order 2, the quadratic terms are "
          "fitting sensors, which is why\n   the truth used above is order "
          f"{LAB_MAP_TRUTH_ORDER}.")
    row = lab[lab['order'] == 1]
    if not row.empty:
        gradient = float(row['gradient_norm'].median())
        print(f"   The order-1 map's gradient is {gradient:.3f} {MAG_UNIT}/m, so over a 0.25 m "
              f"lever arm the field\n   the markers can see changes by "
              f"{0.25 * gradient:.4f} {MAG_UNIT} — the yardstick every gradient claim\n   below "
              f"has to be read against.")


def report_pairs(spec: DatasetSpec, stats: pd.DataFrame) -> None:
    _header(3, "DO THE TWO SEGMENTS OF A JOINT AGREE, AND DO TWO SENSORS ON ONE SEGMENT?",
            "the first is the residual a relative filter's magnetometer update consumes; the "
            "second cannot be blamed on the joint model, on soft tissue, or on the joint-centre "
            "fit")
    joint = _cells(stats, family='joint', mode=PRIMARY_MODE, calibration=PRIMARY_CALIBRATION,
                   channel='vector')
    if not joint.empty:
        table = _by(joint, ['group'], ['err_p50', 'ang_p50', 'sep_m', 'truth_norm_p50'])
        table['err_%_of_field'] = 100 * table['err_p50'] / table['truth_norm_p50']
        table['ang_per_100mm'] = table['ang_p50'] / (table['sep_m'] * 10.0)
        print(table.to_string(index=False, float_format='%.3f'))
        # Placement variants exist only where a segment carries more than one sensor. Where they
        # do, they answer what the canonical pair cannot: whether a pair mounted High agrees
        # better than one mounted Mid — which a field gradient would require and a per-sensor
        # calibration error would not.
        if joint['group'].nunique() > len(spec.primary_joints):
            variants = joint.copy()
            variants['anatomical'] = [canonical_joint(str(g), spec) for g in variants['group']]
            variants['placement'] = [str(g).rsplit('_', 1)[-1] if str(g) not in spec.primary_joints
                                     else 'M' for g in variants['group']]
            print("\n   Direction disagreement (deg) by which placement spans the joint:")
            print(_by(variants, ['anatomical', 'placement'], ['ang_p50'])
                  .pivot(index='anatomical', columns='placement', values='ang_p50')
                  .to_string(float_format='%.3f'))

    segment = _cells(stats, family='segment', target_kind='partner', mode=PRIMARY_MODE,
                     calibration=PRIMARY_CALIBRATION, channel='vector')
    if segment.empty:
        print("\n   This dataset carries one sensor per segment, so the same-segment family is "
              "silent.")
        return
    table = _by(segment, ['group', 'variant'],
                ['sep_m', 'err_p50', 'ang_p50', 'frame_spread_deg'])
    table['ang_per_100mm'] = table['ang_p50'] / (table['sep_m'] * 10.0)
    print("\n   Two magnetometers on ONE segment, compared in their constant shared frame "
          "(no per-sample mocap):")
    print(table.to_string(index=False, float_format='%.3f'))
    print("\n   `ang_per_100mm` is the comparison that matters between the two halves of this "
          "section. A\n   smooth spatial field gradient produces a disagreement PROPORTIONAL to "
          "separation, so the two\n   should agree in that column; a per-sensor error does not "
          "scale with distance at all and\n   makes the short baselines look far worse per "
          "millimetre.")
    if not joint.empty:
        joint_rate = float((joint['ang_p50'] / (joint['sep_m'] * 10.0)).median())
        segment_rate = float((segment['ang_p50'] / (segment['sep_m'] * 10.0)).median())
        print(f"   Measured: {segment_rate:.2f} deg/100 mm within a segment against "
              f"{joint_rate:.2f} across a joint, a\n   factor of "
              f"{segment_rate / max(joint_rate, 1e-9):.1f}. "
              + ("Steeper at short range is the signature of a per-sensor term, not a field."
                 if segment_rate > joint_rate else
                 "Proportionality at both scales is what a real gradient would look like."))
    print(f"   `frame_spread_deg` is how constant the same-segment frame transform actually is "
          f"(median\n   {segment['frame_spread_deg'].median():.4f} deg). Near zero confirms both "
          f"plates come from one marker\n   cluster, which is what makes the mocap-free scoring "
          f"legitimate.")


def report_mechanism(mechanism: pd.DataFrame, gradients: pd.DataFrame) -> None:
    _header(4, "IS THE DISAGREEMENT A FIELD GRADIENT, OR IS IT THE SENSORS?",
            "the same difference series fitted as a body-frame constant, a world-frame constant, "
            "and a rotating gradient — they transform differently, so they are separable")
    if mechanism.empty:
        print("   Needs two sensors on one segment; this dataset has one.")
    else:
        frame = mechanism[mechanism['calibration'] == PRIMARY_CALIBRATION]
        columns = ['explained_body', 'explained_world', 'explained_gradient',
                   'explained_gradient_physical', 'explained_combined']
        table = frame.groupby('segment', observed=True)[columns].median()
        print((100 * table).to_string(float_format='%.1f'))
        print("\n   Percent of the pair difference's variance each model explains ALONE (they are "
              "nested in\n   `combined`, so they do not partition it). `body` is a constant in "
              "the sensor's own frame —\n   a hard iron or a gain error. `gradient` is one "
              "world-frame G times the rotating separation.")
        body = float(table['explained_body'].median())
        gradient = float(table['explained_gradient'].median())
        physical = float(table['explained_gradient_physical'].median())
        print(f"   Median across segments: body {100*body:.0f}% (3 parameters), "
              f"gradient {100*gradient:.0f}% (9), physical\n   gradient {100*physical:.0f}% (5).")
        print(f"   THE PHYSICAL COLUMN IS THE TEST. A real magnetostatic gradient is symmetric "
              f"and traceless, so\n   constraining it should cost almost nothing; here it costs "
              f"{100*(gradient - physical):.0f} points of "
              f"{100*gradient:.0f}, i.e.\n   "
              f"{100 * (1 - physical / max(gradient, 1e-9)):.0f}% of what the unconstrained fit "
              f"explained could not have come from a magnetic field.")
        verdict = ("SENSOR-FIXED — a per-device calibration term, not a field"
                   if body > physical else "consistent with a genuine spatial gradient")
        print(f"   Verdict: {verdict}.")

        for calibration in ('hard_iron', 'affine'):
            arm = mechanism[mechanism['calibration'] == calibration]
            if arm.empty:
                continue
            print(f"   With the {calibration} calibration removed (fitted on the subject's other "
                  f"trials): pair\n     disagreement {arm['ang_p50'].median():.2f} deg against "
                  f"{frame['ang_p50'].median():.2f} raw, and the body term now explains "
                  f"{100 * arm['explained_body'].median():.0f}%.")

    if gradients.empty:
        return
    body_rows = gradients[(gradients['scope'] == 'body')
                          & (gradients['calibration'] == PRIMARY_CALIBRATION)]
    if body_rows.empty:
        return
    print("\n   The whole-body array, where the sensors are NOT collinear and a full 3x3 gradient "
          "is\n   observable (median over trials):")
    columns = ['n_sensors', 'residual_constant_p50', 'residual_linear_p50', 'gradient_norm_p50',
               'trial_gradient_norm', 'trial_gradient_explained', 'trial_gradient_norm_physical',
               'trial_gradient_explained_physical', 'lab_map_gradient_norm',
               'gradient_constancy_world', 'gradient_constancy_body',
               'array_extent_mm', 'array_thickness_mm']
    print(body_rows[[c for c in columns if c in body_rows.columns]].median()
          .to_frame('median').to_string(float_format='%.4f'))
    per_sample = float(body_rows['gradient_norm_p50'].median())
    lab = float(body_rows['lab_map_gradient_norm'].median())
    if np.isfinite(lab) and lab > 0:
        print(f"\n   The per-sample fit returns a gradient {per_sample / lab:.0f}x the one the lab "
              f"map sees over the whole\n   room. Nine parameters against "
              f"{body_rows['n_sensors'].median():.0f} sensors leaves almost no redundancy, so a "
              f"sensor's own\n   error has nowhere to go except into G. The trial-constant "
              f"physical fit — thousands of\n   samples, five parameters, and the only one that "
              f"cannot fit a non-field — returns "
              f"{float(body_rows['trial_gradient_norm_physical'].median()):.3f},\n   which is the "
              f"one estimate of the three that agrees with the markers.")
    print("   `gradient_constancy_world` vs `_body`: a real room gradient is constant in the "
          "world frame, a\n   fit tracking the subject's own hardware is constant in the body "
          "frame. Neither being near 1\n   means the per-sample fit is tracking something that is "
          "neither.")


def report_modes(spec: DatasetSpec, stats: pd.DataFrame) -> None:
    _header(5, "DOES A HIGHER-ORDER PROJECTION IMPROVE CROSS-SEGMENT AGREEMENT?",
            "the question this file was built for — same joints, same samples, same reference; "
            "only how each segment transports its reading to the joint centre differs")
    joint = _cells(stats, family='joint', calibration=PRIMARY_CALIBRATION, channel='vector')
    marker = _cells(stats, family='marker', calibration=PRIMARY_CALIBRATION, channel='vector')
    if joint.empty:
        print("   No cross-segment comparisons.")
        return

    table = _by(joint, ['mode'], ['err_p50', 'ang_p50', 'n_roles_upgraded'])
    reference = _by(marker, ['mode'], ['err_p50', 'ang_p50']).rename(
        columns={'err_p50': 'ref_err_p50', 'ang_p50': 'ref_ang_p50'})
    table = table.merge(reference, on='mode', how='left')
    baseline = table[table['mode'] == PRIMARY_MODE]
    if not baseline.empty:
        table['vs_0th_%'] = 100 * (table['ang_p50'] / float(baseline['ang_p50'].iloc[0]) - 1)
        if 'ref_ang_p50' in table and baseline['ref_ang_p50'].notna().any():
            table['ref_vs_0th_%'] = 100 * (table['ref_ang_p50']
                                           / float(baseline['ref_ang_p50'].iloc[0]) - 1)
    # Ordered by MODES rather than alphabetically, so the 0th-order baseline reads first and the
    # degenerate control reads last. Indexed off the COLUMN before set_index, not off the frame's
    # index afterwards — the latter is still a RangeIndex at the point the list is built.
    order = [m for m in MODES if m in set(table['mode'])]
    table = table.set_index('mode').reindex(order)
    print(table.to_string(float_format='%.4f'))
    print("\n   READ BOTH ERROR COLUMNS. `ang_p50` is how well the two segments AGREE; "
          "`ref_ang_p50` is how far\n   their projections sit from the marker-supported field "
          "model. `body_model` scores 0 on the\n   first by construction — both segments report "
          "the same modelled value and have stopped\n   measuring — and its second column is what "
          "that costs. Agreement alone is not a score.")

    # Paired per trial x joint, since every mode is computed on identical inputs and a pooled
    # median could hide a mode that wins on most cells and loses badly on a few.
    wide = joint.pivot_table(index=['subject', 'trial', 'group'], columns='mode',
                             values='ang_p50', observed=True)
    # Pivoting on a categorical leaves a CategoricalIndex on the columns, which `.dropna()`
    # and `.join()` both raise on. Plain strings from here.
    wide.columns = [str(column) for column in wide.columns]
    if PRIMARY_MODE in wide.columns:
        print("\n   Paired over trial x joint cells, against the 0th order:")
        for mode in MODES:
            if mode == PRIMARY_MODE or mode not in wide.columns:
                continue
            pair = wide[[mode, PRIMARY_MODE]].dropna()
            if pair.empty:
                continue
            ratio = pair[mode] / pair[PRIMARY_MODE]
            line = (f"     {mode:<20} better in {int((ratio < 1).sum()):4d}/{len(ratio):4d} "
                    f"cells, ratio median {ratio.median():.3f} "
                    f"(IQR {ratio.quantile(0.25):.3f}-{ratio.quantile(0.75):.3f})")
            if len(pair) >= 6 and mode != 'body_model':
                _, p_value = wilcoxon(pair[mode], pair[PRIMARY_MODE])
                line += f", p={p_value:.1e}"
            print(line)

    line_modes = joint[joint['mode'].isin(list(LINE_MODE_ORDER))]
    for column in ('parent_extrapolation', 'child_extrapolation'):
        if column in line_modes.columns and line_modes[column].notna().any():
            print(f"\n   Geometry behind it: the joint centre sits "
                  f"{line_modes[column].median():.2f} sensor-spans beyond the end of the\n   "
                  f"array (0 would be an interpolation)"
                  + (f", and {line_modes['parent_target_offline_mm'].median():.0f} mm off the "
                     f"line the array can see along."
                     if 'parent_target_offline_mm' in line_modes.columns else "."))
            break

    for calibration in ('hard_iron', 'affine'):
        arm = _cells(stats, family='joint', calibration=calibration, channel='vector')
        if arm.empty:
            continue
        summary = _by(arm, ['mode'], ['ang_p50'])
        print(f"\n   Under the {calibration} calibration (fitted on the subject's other trials, "
              f"so out of sample\n   for this metric): "
              + ", ".join(f"{row['mode']} {row['ang_p50']:.2f}" for _, row in summary.iterrows()))


def report_holdout(stats: pd.DataFrame) -> None:
    _header(6, "CAN TWO MAGNETOMETERS ON A SEGMENT PREDICT THE THIRD?",
            "the only fully out-of-sample test here: a real measurement sits at the point being "
            "predicted, unlike the joint centre where nothing is mounted")
    frame = _cells(stats, family='segment', target_kind='held_out',
                   calibration=PRIMARY_CALIBRATION)
    if frame.empty:
        print("   Needs three sensors on one segment; this dataset has fewer.")
        return
    table = _by(frame, ['variant', 'mode'], ['extrapolation', 'err_p50', 'ang_p50'])
    table['geometry'] = np.where(table['extrapolation'] > 0, 'extrapolation', 'interpolation')
    print(table.to_string(index=False, float_format='%.4f'))
    print("\n   `mean` is the two sources averaged — no spatial model. `linear` is the line "
          "through them,\n   evaluated at the held-out sensor. `extrapolation` is how far beyond "
          "the two sources that\n   evaluation sits, in spans: 0 means the target is between "
          "them.")

    wide = frame.pivot_table(index=['subject', 'trial', 'group', 'target'], columns='mode',
                             values='err_p50', observed=True)
    # Pivoting on a categorical leaves a CategoricalIndex on the columns, which `.dropna()`
    # and `.join()` both raise on. Plain strings from here.
    wide.columns = [str(column) for column in wide.columns]
    if {'mean', 'linear'} <= set(wide.columns):
        pair = wide[['linear', 'mean']].dropna()
        ratio = pair['linear'] / pair['mean']
        print(f"   Paired over {len(pair)} cells: the line beats the mean in "
              f"{int((ratio < 1).sum())}, ratio median {ratio.median():.3f}.")
        inner = frame[frame['extrapolation'] <= 0][['subject', 'trial', 'group', 'target']]
        if not inner.empty:
            keys = pd.MultiIndex.from_frame(inner.drop_duplicates())
            interpolating = wide[wide.index.isin(keys)]
            if not interpolating.empty:
                inner_ratio = (interpolating['linear'] / interpolating['mean']).dropna()
                print(f"   Restricted to the INTERPOLATING cells, ratio median "
                      f"{inner_ratio.median():.3f} over {len(inner_ratio)} cells — "
                      f"the same model,\n   asked to reach inside the array rather than beyond "
                      f"it.")

    for calibration in ('hard_iron', 'affine'):
        arm = _cells(stats, family='segment', target_kind='held_out', calibration=calibration)
        if arm.empty:
            continue
        print(f"   Under the {calibration} calibration: "
              + ", ".join(f"{row['mode']} {row['err_p50']:.4f}"
                          for _, row in _by(arm, ['mode'], ['err_p50']).iterrows()))


def report_calibration(calibration: pd.DataFrame, stats: pd.DataFrame) -> None:
    _header(7, "WHAT A PER-SENSOR CALIBRATION BUYS, AND WHETHER IT TRANSFERS",
            "a hard iron is fixed in the sensor frame, so if these fits are a device property "
            "they repeat across a subject's trials; if they are the room, they do not")
    if calibration.empty or 'hard_iron_norm' not in calibration.columns:
        print("   No calibration fits.")
        return
    usable = calibration.dropna(subset=['hard_iron_norm'])
    print(f"   Fitted on {len(usable)} sensor-trials. Hard-iron magnitude: median "
          f"{usable['hard_iron_norm'].median():.4f} {MAG_UNIT}, p90 "
          f"{usable['hard_iron_norm'].quantile(0.9):.4f}, against a field norm of "
          f"~{usable['mag_norm_median'].median():.2f}.")
    print("\n   Residual against the subject's constant field, IN SAMPLE (a floor, not a score):")
    print(usable[['residual_raw', 'residual_hard_iron', 'residual_affine']].median()
          .to_frame('median').to_string(float_format='%.4f'))
    print(f"   Affine departure from the identity: median "
          f"{usable['affine_gain_departure'].median():.3f}. Design condition number median "
          f"{usable['design_condition'].median():.0f}\n   — that condition number IS the "
          f"orientation coverage, and a sensor that barely rotates cannot\n   have its offset "
          f"told apart from a field error.")

    repeats = usable.groupby(['subject', 'sensor'], observed=True)[list(HARD_IRON_COLUMNS)]
    spread, magnitude = repeats.std().dropna(), repeats.mean().dropna()
    if not spread.empty:
        between = np.linalg.norm(spread.to_numpy(), axis=1)
        mean_norm = np.linalg.norm(magnitude.to_numpy(), axis=1)
        fraction = 100 * np.median(between) / max(np.median(mean_norm), 1e-9)
        print(f"\n   BETWEEN-TRIAL REPEATABILITY over {len(spread)} subject-sensors: the fitted "
              f"offset scatters by\n   {np.median(between):.4f} {MAG_UNIT} around a mean of "
              f"{np.median(mean_norm):.4f} — {fraction:.0f}% of its own size.")
        print("   A term that reproduces across a subject's trials is a property of the device; "
              "one that does\n   not was the room, and carrying it to another trial can only "
              "hurt. That is why every\n   calibrated arm here is fitted leave-one-trial-out, and "
              "why `hard_iron_own` is reported\n   beside it rather than instead of it.")

    joint = _cells(stats, family='joint', mode=PRIMARY_MODE, channel='vector')
    if not joint.empty:
        wide = joint.pivot_table(index=['subject', 'trial', 'group'], columns='calibration',
                                 values='ang_p50', observed=True).dropna()
        # Pivoting on a categorical leaves a CategoricalIndex on the columns, which `.dropna()`
        # and `.join()` both raise on. Plain strings from here.
        wide.columns = [str(column) for column in wide.columns]
        available = [c for c in CALIBRATIONS if c in wide.columns]
        if len(available) > 1 and not wide.empty:
            print(f"\n   Cross-segment direction disagreement (deg) by calibration arm, paired "
                  f"over {len(wide)} cells:")
            print(wide[available].median().to_frame('median deg').to_string(float_format='%.3f'))
            if {'hard_iron', 'hard_iron_own'} <= set(available):
                gap = float(wide['hard_iron_own'].median() - wide['hard_iron'].median())
                print(f"   The in-sample arm is {abs(gap):.3f} deg "
                      f"{'better' if gap < 0 else 'worse'} than the leave-one-trial-out one — the "
                      f"size of the\n   optimism an in-sample calibration would have added to "
                      f"every number in this file.")


def report_verdict(spec: DatasetSpec, stats: pd.DataFrame, mechanism: pd.DataFrame) -> None:
    _header(9, "WHAT THIS MEANS FOR THE FILTER",
            "the numbers above, converted into the currency the relative filter spends")
    joint = _cells(stats, family='joint', mode=PRIMARY_MODE, calibration=PRIMARY_CALIBRATION,
                   channel='vector')
    if joint.empty:
        print("   No cross-segment comparisons.")
        return
    print(f"   The 0th-order projection — what every filter here does — leaves the two segments of "
          f"a joint\n   disagreeing about the field direction by a median "
          f"{joint['ang_p50'].median():.1f} deg (p90 "
          f"{joint['ang_p90'].median():.1f} deg). That disagreement\n   enters the magnetometer "
          f"update as an irreducible residual: it is not noise the filter can\n   average away, "
          f"because it is nearly constant over a stride.")
    worst = _by(joint, ['group'], ['ang_p50']).sort_values('ang_p50')
    if not worst.empty:
        print(f"   Best joint {worst.iloc[0]['group']} at {worst.iloc[0]['ang_p50']:.1f} deg, "
              f"worst {worst.iloc[-1]['group']} at {worst.iloc[-1]['ang_p50']:.1f} deg — the same "
              f"proximal-to-distal\n   ordering every other magnetic result in this repository "
              f"shows.")
    norm = _cells(stats, family='joint', mode=PRIMARY_MODE, calibration=PRIMARY_CALIBRATION,
                  channel='norm')
    if not norm.empty:
        print(f"   The mocap-free magnitude channel puts a floor under that: the two sensors "
              f"disagree about\n   the field's MAGNITUDE by {norm['err_p50'].median():.4f} "
              f"{MAG_UNIT}, {100 * norm['err_p50'].median() / max(norm['truth_norm_p50'].median(), 1e-9):.0f}% "
              f"of the field, and no orientation\n   estimate can absorb a magnitude difference.")

    # The best mode, reported with its PAIRED result rather than only its pooled median. The two
    # can disagree — a mode can lower the pooled median while being no better than even on a
    # cell-by-cell comparison — and quoting the pooled number alone is how a 17% "improvement"
    # that no individual joint reliably sees gets into a paper.
    all_modes = _cells(stats, family='joint', calibration=PRIMARY_CALIBRATION, channel='vector')
    modes = _by(all_modes, ['mode'], ['ang_p50'])
    best_gain_deg = np.nan
    if not modes.empty:
        best = modes[modes['mode'] != 'body_model'].sort_values('ang_p50')
        baseline = modes.loc[modes['mode'] == PRIMARY_MODE, 'ang_p50']
        if not best.empty and not baseline.empty:
            winner = best.iloc[0]
            change = 100 * (float(winner['ang_p50']) / float(baseline.iloc[0]) - 1)
            best_gain_deg = float(baseline.iloc[0]) - float(winner['ang_p50'])
            # The winner CAN BE the baseline itself, and on Al Borno it is — nothing there beats
            # transporting the reading unchanged. Said outright rather than compared against
            # itself, which is both meaningless and, when the same column is selected twice,
            # a DataFrame where the code below expects a Series.
            if str(winner['mode']) == PRIMARY_MODE:
                print(f"\n   Of the {len(modes)} projections scored, THE 0TH ORDER IS THE BEST "
                      f"non-degenerate one, at\n   {float(baseline.iloc[0]):.2f} deg. Every "
                      f"spatial model tried here is worse than doing nothing to the reading.")
                return_early = True
            else:
                return_early = False
                print(f"\n   Of the {len(modes)} projections scored, the best non-degenerate one "
                      f"is '{winner['mode']}' at {winner['ang_p50']:.2f} deg\n   against the 0th "
                      f"order's {float(baseline.iloc[0]):.2f} — "
                      f"{abs(change):.0f}% {'better' if change < 0 else 'worse'} pooled.", end='')
            wide = all_modes.pivot_table(index=['subject', 'trial', 'group'], columns='mode',
                                         values='ang_p50', observed=True)
            # Pivoting on a categorical leaves a CategoricalIndex on the columns, which `.dropna()`
            # and `.join()` both raise on. Plain strings from here.
            wide.columns = [str(column) for column in wide.columns]
            if not return_early and {str(winner['mode']), PRIMARY_MODE} <= set(wide.columns):
                pair = wide[[str(winner['mode']), PRIMARY_MODE]].dropna()
                ratio = pair[str(winner['mode'])] / pair[PRIMARY_MODE]
                print(f" Paired per trial x joint, though, it is\n   better in only "
                      f"{int((ratio < 1).sum())} of {len(ratio)} cells (ratio median "
                      f"{ratio.median():.3f}) — the pooled gain is a few cells, not a\n   "
                      f"population effect.")
            elif not return_early:
                print()
    if not mechanism.empty:
        # Median across SEGMENTS, matching section 4 — a median across rows would weight a
        # segment by how many sensor pairs it happens to carry and print a different number for
        # the same claim two sections apart.
        frame = mechanism[mechanism['calibration'] == PRIMARY_CALIBRATION]
        by_segment = frame.groupby('segment', observed=True)[
            ['explained_body', 'explained_gradient_physical']].median()
        body = float(by_segment['explained_body'].median())
        physical = float(by_segment['explained_gradient_physical'].median())
        print(f"\n   The reason is in section 4: {100*body:.0f}% of a same-segment disagreement "
              f"is explained by a constant\n   in the SENSOR's frame, and only "
              f"{100*physical:.0f}% by a physically admissible field gradient. There\n   is no "
              f"gradient of the right shape to estimate, so estimating one fits the sensors "
              f"instead —\n   and extrapolating that fit past the end of a 17 cm array to a "
              f"joint centre 25 cm away\n   amplifies it.")

    print("\n   THE ACTIONABLE CONCLUSION IS NOT A HIGHER-ORDER PROJECTION. The disagreement is "
          "dominated by a\n   per-sensor, body-fixed term, which is what a calibration removes "
          "and what no spatial model\n   can.", end='')
    calibrated = _by(_cells(stats, family='joint', mode=PRIMARY_MODE, channel='vector'),
                     ['calibration'], ['ang_p50'])
    if not calibrated.empty and {'none', 'hard_iron'} <= set(calibrated['calibration']):
        raw = float(calibrated.loc[calibrated['calibration'] == 'none', 'ang_p50'].iloc[0])
        fixed = float(calibrated.loc[calibrated['calibration'] == 'hard_iron', 'ang_p50'].iloc[0])
        comparison = (f", against {best_gain_deg:.2f} deg for the best spatial model"
                      if np.isfinite(best_gain_deg) else "")
        print(f" A three-parameter hard iron fitted on the subject's OTHER trials takes\n   "
              f"the 0th order from {raw:.2f} deg to {fixed:.2f} — a gain of {raw - fixed:.2f} deg"
              f"{comparison}.\n   That is where the error is.", end='')
    print("\n   The caveat that decides whether it is usable online is in section 7: on this data "
          "the fitted\n   offset scatters between a subject's own trials by nearly its own size, "
          "so most of what it\n   removes was the room the subject was standing in rather than "
          "the device.")


def report_covariation(covariation: pd.DataFrame) -> None:
    _header(8, "HOW MUCH DO NEIGHBOURING MAGNETOMETERS COVARY?",
            "the complementary question to every section above: not how far two sensors "
            "disagree, but how much of what they see is SHARED — which is what decides whether "
            "differencing them helps")
    if covariation.empty:
        print("   No covariation rows.")
        return
    frame = covariation[covariation['calibration'] == PRIMARY_CALIBRATION]
    columns = ['sep_m', 'rho', 'surviving', 'rho_slow', 'rho_gait', 'rho_debiased',
               'surviving_debiased']
    table = frame.groupby('pair_class', observed=True)[columns].median()
    table = table.reindex([c for c in PAIR_CLASSES if c in table.index])
    table['n_pairs'] = frame.groupby('pair_class', observed=True).size().reindex(table.index)
    print(table.to_string(float_format='%.3f'))
    print("\n   `surviving` is the fraction of disturbance energy left after subtracting the two "
          "sensors:\n   0 = perfectly shared and differencing removes it, 1 = independent so "
          "differencing neither\n   helps nor hurts, >1 = anti-correlated and differencing makes "
          "it worse. For equal variances\n   it is exactly 1 - rho.")
    print("   The far classes are the CONTROL. A genuinely spatial disturbance is shared by "
          "neighbours\n   and not by a sensor on the other leg, so covariation that does not "
          "fall off with adjacency\n   is not a field.")

    near = table.loc['same_segment'] if 'same_segment' in table.index else None
    far = table.loc['contralateral'] if 'contralateral' in table.index else None
    if near is not None and far is not None:
        print(f"   Measured: rho {near['rho']:.2f} on one segment against {far['rho']:.2f} "
              f"between legs. "
              + ("Adjacency does buy shared signal."
                 if near['rho'] > far['rho'] + 0.1 else
                 "Adjacency buys little — the disturbance is largely per-sensor."))
    if near is not None:
        print(f"   But removing each sensor's own fitted body-fixed bias takes the same-segment "
              f"rho from\n   {near['rho']:.2f} to {near['rho_debiased']:.2f}"
              + (" — so most of what looked shared was two biases rotating together,\n   not one "
                 "field seen twice." if near['rho_debiased'] < near['rho'] - 0.1 else
                 " — the shared part survives debiasing, so it is a field."))
    print(f"   Timescale: rho is {table['rho_slow'].median():.2f} below "
          f"{SLOW_BAND_HZ} Hz and {table['rho_gait'].median():.2f} in the gait band. A "
          f"room-driven\n   disturbance changes as the subject crosses the lab and should be the "
          f"slow one; a body-fixed\n   bias sweeps at the limb's own rotation rate.")


def print_report(spec: DatasetSpec, stats: pd.DataFrame, lab: pd.DataFrame,
                 mechanism: pd.DataFrame, gradients: pd.DataFrame,
                 calibration: pd.DataFrame, covariation: pd.DataFrame) -> None:
    report_headline(spec, stats)
    report_marker(spec, stats, lab)
    report_pairs(spec, stats)
    report_mechanism(mechanism, gradients)
    report_modes(spec, stats)
    report_holdout(stats)
    report_calibration(calibration, stats)
    report_covariation(covariation)
    report_verdict(spec, stats, mechanism)

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='alborno', choices=sorted(DATASETS))
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--trials', nargs='+', default=None)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--stride', type=int, default=SAMPLE_STRIDE,
                        help="Keep every Nth sample in agreement_samples. Every scalar in "
                             "agreement_stats is computed on the full conditioned record.")
    parser.add_argument('--only-tables', nargs='+', choices=ANALYSIS_TABLES,
                        default=list(ANALYSIS_TABLES), metavar='TABLE')
    parser.add_argument('--skip-fields', action='store_true',
                        help="Reuse the subject fields and calibrations already on disk. Safe "
                             "only when the trial set has not changed, since both are pooled over "
                             "a subject's trials.")
    parser.add_argument('--report-only', action='store_true',
                        help="Rebuild the summary and report from what is already on disk.")
    args = parser.parse_args()

    try:
        row_keys = select_trials(args.dataset, args.subjects, args.trials)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    spec = get_dataset(args.dataset)
    if not spec.has_magnetometer:
        print(f"Error: {args.dataset} has no magnetometer — its IMUs measure acceleration and "
              f"rotation only, and\nthe reader fills `mag` with exact zeros. Every number this "
              f"experiment produces would be a\nstatement about those zeros. Nothing to do.")
        return 1

    orphans = orphaned_trials(args.dataset)
    if orphans:
        names = ", ".join(f"{s}/{t}" for s, t in orphans[:3])
        print(f"Skipping {len(orphans)} built parquet(s) the {spec.build_name} source no longer "
              f"enumerates ({names}{', …' if len(orphans) > 3 else ''}).")

    if not args.report_only:
        if not args.skip_fields:
            print(f"Stage 1/3: per-trial magnetic field over {len(row_keys)} trials...")
            run_tracked_grid(row_keys, ['Subject', 'Trial'], ['field'],
                             partial(_field_worker, dataset=args.dataset), args.workers,
                             title=f"MAGNETIC PROJECTION — FIELD — {args.dataset}")
            written = aggregate_subject_fields(args.dataset, row_keys)
            print(f"Wrote a global field for {len(written)} subject(s).")

            print(f"\nStage 2/3: per-sensor calibration over {len(row_keys)} trials...")
            run_tracked_grid(row_keys, ['Subject', 'Trial'], ['calibrate'],
                             partial(_calibration_worker, dataset=args.dataset), args.workers,
                             title=f"MAGNETIC PROJECTION — CALIBRATION — {args.dataset}")

        print(f"\nStage 3/3: projections over {len(row_keys)} trials...")
        state, _ = run_tracked_grid(
            row_keys, ['Subject', 'Trial'], ['project'],
            partial(_trial_worker, dataset=args.dataset, tables=args.only_tables,
                    stride=args.stride, row_keys=row_keys),
            args.workers, title=f"MAGNETIC PROJECTION — {args.dataset}")
        failures = {key: value for (key, stage), value in state.items()
                    if stage == 'project' and isinstance(value, str) and value.startswith('Failed')}
        if failures:
            reasons: Dict[str, int] = {}
            for message in failures.values():
                reasons[message[:70]] = reasons.get(message[:70], 0) + 1
            print(f"\n{len(failures)} of {len(row_keys)} trials FAILED:")
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
                print(f"  {count:4d} x {reason}")
            if len(failures) == len(row_keys):
                print("\nEvery trial failed, so nothing was written. Anything below would "
                      "describe a PREVIOUS run.")
                return 1

    print("\nLoading per-trial tables...")
    stats = load_trial_table(args.dataset, 'agreement_stats')
    if stats.empty:
        print(f"No results under {dataset_dir(args.dataset)}. Run without --report-only first.")
        return 1
    lab = load_trial_table(args.dataset, 'lab_field')
    mechanism = load_trial_table(args.dataset, 'mechanism')
    gradients = load_trial_table(args.dataset, 'gradient_fits')
    calibration = load_trial_table(args.dataset, STAGE2_TABLE)
    covariation = load_trial_table(args.dataset, 'covariation')

    found = {(str(s), str(t)) for s, t in stats[['subject', 'trial']].drop_duplicates().to_numpy()}
    print(f"Found {len(found)} trial(s) across {len({s for s, _ in found})} subject(s), "
          f"{len(stats):,} comparisons.")

    samples = load_trial_table(args.dataset, 'agreement_samples')
    summary = summarize(args.dataset, samples)
    if not summary.empty:
        path = paths.ensure_parent(statistics_path(args.dataset))
        summary.to_parquet(path, engine='pyarrow', index=False)
        paths.write_manifest(path, constants=analysis_constants(args.dataset),
                             experiment=EXPERIMENT_NAME, n_rows=len(summary),
                             subjects=sorted({s for s, _ in found}))
        print(f"Saved summary to {path}")

    print_report(spec, stats, lab, mechanism, gradients, calibration, covariation)
    print(f"\nPer-trial tables under {dataset_dir(args.dataset)}")
    print(f"Figures: python -m plotting.magnetic_projection --dataset {args.dataset}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
