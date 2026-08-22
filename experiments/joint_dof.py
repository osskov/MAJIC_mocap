"""
How well is a measured joint approximated by a low-DOF mechanism?

Every joint-axis method in this repo — `PlateTrial.find_biaxial_joint_axes`, hinge-axis
identification, the Reuben knee coupling IMoveLab enforces every sample — assumes the joint
IS some particular mechanism. That assumption has a cost, it is paid whether or not anyone
measures it, and it is an ERROR FLOOR: no filter can recover what the mechanism itself
cannot represent. This module measures the floor directly from the reference kinematics,
with no filter and no IMU orientation estimate anywhere in the path.

THE TWO QUESTIONS, each a difference between two rungs of the ladder below:

    hinge -> universal   how much does the missing SECOND AXIS cost?
    hinge -> coupling    does Reuben's published curvature help, or is it the wrong shape?

WHAT THIS FILE CANNOT ANSWER, stated up front because the omission is deliberate and someone
will otherwise read it as an answer. An earlier version also fitted a FREE curve — the same
model with c(q) a cubic B-spline instead of Reuben's polynomials — as an envelope on what any
1-DOF nonlinear model could achieve. It has been removed. Without it, a coupling residual of
15 deg cannot be split into "Reuben has the wrong shape for this knee" and "no 1-DOF curve of
any shape would have done better", because there is nothing measuring the second. The
hinge-to-coupling drop is still exactly Reuben's own contribution; only the ceiling is gone.

Sibling of `experiments/joint_center.py`, which asks the same kind of question about the
joint's POSITION (is there a fixed centre?) where this one asks about its ORIENTATION (is
there a fixed axis?). Same shape: per-trial parquets, a per-dataset statistics table, a
console report.

    python -m experiments.joint_dof --dataset alborno
    python -m experiments.joint_dof --dataset imove_biplane
    python -m experiments.joint_dof --dataset imove --joints R_Knee L_Knee
    python -m experiments.joint_dof --dataset alborno --report-only
    python -m experiments.joint_dof --validate

THE LADDER OF MODELS
--------------------
All are fitted to the same reference relative rotation R_pc(t) = R_parent(t)^T R_child(t),
and they NEST, so their residuals are monotone and each drop is what the extra freedom
buys. Per-sample freedom is what makes two models comparable; structure parameters are
fitted once for the whole trial and are charged for separately, by cross-validation.

  0-DOF  weld        R_pc(t) = R_0.
         Not a joint anyone proposes — it is the DENOMINATOR. Every other model's error is
         also reported as a fraction of this one, because a joint that barely moved is
         fitted well by everything and that must be visible rather than flattering.

  1-DOF  hinge       R_pc(t) = exp(theta(t) [u]) R_0.
         One axis, fixed in BOTH segments. Structure: u (2) + R_0 (3).

  1-DOF  coupling    R_pc(t) = R_0 exp([ q(t) u_c + P f(q(t)) ]),  f PUBLISHED.
         THE 1-DOF NONLINEAR KNEE, and the model this file exists to test. Still one angle
         per sample, exactly as the hinge, but the two non-sagittal channels are set to the
         Reuben et al. (1986) polynomials of flexion rather than to zero — the coupling
         IMoveLab enforces every sample at gain 0.9, which at 100 Hz is replacement rather
         than regularization. Knee-only, by definition.

         Structure: u_c (2) + R_0 (3) + which direction in u_c's normal plane is adduction
         (1) — the hinge's five plus one, and that one is a per-subject calibration any
         implementation has to establish. The CURVE is published, not fitted, which is what
         makes the residual a statement about Reuben and makes the HINGE the right thing to
         read it against: same per-sample freedom, one pinning the off-axis channels to zero
         and the other to a published curve.

         A LINEAR coupling is not a new kind of joint — it is a hinge about a TILTED axis,
         and the hinge's axis is already free, so it absorbs the linear part for nothing.
         Reuben's rotation channel runs at 0.3695 deg/deg, i.e. a 20 deg tilt, taken for
         free. What separates the two models is therefore only the CURVATURE of the
         polynomials over the flexion range the trial visited. That is the whole budget.

         Fitted at its best — flexion sense, plate handedness and channel orientation are all
         searched, since dropping the published channels into an arbitrary basis unaligned
         scored the same curve at 23.2 deg instead of 15.1. See `fit_knee_coupling`.

  2-DOF  universal   R_pc(t) = exp(theta1(t) [u]) R_0 exp(theta2(t) [v]).
         u fixed in the PARENT frame, v fixed in the CHILD frame, R_0 carrying the angle
         between them. Same kinematics as `PlateTrial.generate_2dof_plate`, with the constant
         factors absorbed into R_0. Structure: u (2) + v (2) + R_0 (3).

  3-DOF  spherical   R_pc(t) = anything.
         Zero by construction, not by measurement: a ball joint constrains relative position,
         not relative orientation. Carried explicitly so the table reads as a ladder.

WHAT IS NEW HERE relative to the exploratory version this replaces
------------------------------------------------------------------
  * ALL THREE DATA SOURCES, through one `DatasetSpec` path: Al Borno (7 joints, marker
    plates), IMoVE mocap (6 joints x 3 sensor placements, marker plates) and IMoVE biplane
    (one knee per trial, measured TWICE — by Vicon marker clusters and by biplane
    fluoroscopy bone poses).

  * MARKERS VERSUS BIPLANE, which is the question the biplane half exists to answer. On
    those trials both references see the same knee over the same frames, so
    `reference_agreement` fits the constant frame change between them and reports what is
    left. That residual is soft-tissue artifact plus marker error, and it is the NOISE FLOOR
    every model residual on a marker-based dataset has to be read against — a 5 deg hinge
    residual means something different if the markers themselves are good to 1 deg than if
    they are good to 4.

  * CROSS-VALIDATED ERROR, blocked in time. The coupling fits one more structure parameter
    than the hinge, so a bare in-sample comparison tilts toward it by construction. Every model
    is scored out-of-fold on contiguous blocks, and the report leads with that number.

  * THE `valid` MASK, per contiguous run. Outside `valid` a WorldTrace holds a constant
    padded pose; the exploratory version fitted straight through those, which on the biplane
    trials would mean fitting 60,000 identical frames and 135 real ones. Low-pass filtering
    is likewise applied per run, so a filter never smears across a gap.

  * POOLED FITS. A biplane trial is 0.4-0.6 s of one hop: enough to fit seven structure
    parameters numerically, nowhere near enough to have swept the joint. Trials of the same
    subject/side/source are pooled into one fit — legitimate because the BioStamps and the
    marker clusters stay on for the whole session and the bone frames are anatomy — and the
    per-trial fits are kept beside them so the pooling assumption is testable rather than
    assumed.

  * DETAILED STATISTICS: the full quantile spine, the residual's own principal directions
    (is the error one channel or all three?), error as a function of joint angle, the raw
    residual of the filtered fit (how much was marker jitter), excitation and conditioning,
    convergence and identifiability flags, and leave-one-subject-out transfer of a single
    proposed axis.

CAVEATS THAT SURVIVE ALL OF THAT
--------------------------------
  * These are SENSOR-PLATE frames on the marker datasets, not anatomical ones. The fitted
    axes are directions in each subject's own plate frame and the plates are re-strapped per
    subject, so axis VECTORS are not comparable across subjects and the report never pools
    them without saying so. The RESIDUALS and the CARRYING ANGLE are mounting-invariant and
    are what the cross-subject tables use. The biplane source is the exception: its frames
    are bone-fixed, so its axes are anatomical and do pool.

  * Plate-to-bone motion is inside the marker residuals. That is the right convention for
    this repo — a joint-axis constraint in a filter applies to the IMU frames, not to the
    bones — but it means a marker hinge residual is not a statement about the femur and the
    tibia. The biplane source is what separates the two.

  * A joint that barely moves is fitted well by everything. ROM and the 0-DOF reference are
    reported beside every residual so that case is visible.

Outputs, all under results/experiments/joint_dof/<dataset>/:

    <subject>/<trial>/joint_fits.parquet       one row per (joint, source, model)
    <subject>/<trial>/error_samples.parquet    strided per-sample error, for the figures
    <subject>/<trial>/error_curve.parquet      error binned by joint angle
    <subject>/<trial>/coupling_shape.parquet   the fitted coupling curve, in degrees
    <subject>/<trial>/reference_agreement.parquet   markers vs biplane (biplane only)

plus <dataset>/pooled_fits.parquet, <dataset>/generic_axes.parquet and
results/statistics/joint_dof_<dataset>_statistics.parquet.
"""
import argparse
import os

os.environ.setdefault("DISABLE_TQDM", "True")

# ONE BLAS THREAD PER PROCESS, set before numpy is imported because the thread pools are sized
# at import time and cannot be resized afterwards.
#
# This module is already parallel over trials, so a BLAS that also spawns a thread per core
# oversubscribes the machine by the worker count squared. The linear algebra here is a stream of
# tiny operations — 3x3 solves, a (2400, 20) least squares — every one of which is far below the
# size where threading pays, so the threads contend and win nothing. Measured on Al Borno with
# ten workers: 7 minutes per trial before, 1.5 after, for identical output.
for _threads_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_threads_var, "1")

import time
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

import paths
from experiments.experiment_utils import load_trial, pipeline_constants, run_tracked_grid
from experiments.global_assumptions import (DATASETS as MAGNETIC_DATASETS, DatasetSpec,
                                            canonical_joint, enumerate_trials, subjects_of)
from src import joint_constraints
from src.toolchest import so3
from src.toolchest.PlateTrial import PlateTrial

EXPERIMENT_NAME = "joint_dof"
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Marker-plate rotations carry jitter well above human joint bandwidth, and that jitter lands
# almost entirely in the MINOR axes — which is exactly where the 1-vs-2-DOF decision is made.
# 6 Hz is the usual gait-analysis kinematics cutoff. It is not taken on faith: every fit also
# reports its residual re-scored against the UNFILTERED rotations (`*_raw_rms_deg`), so the
# amount of each residual that is jitter is visible in the same row.
LOWPASS_HZ = 6.0
LOWPASS_ORDER = 4

# Shortest contiguous run of valid frames worth KEEPING. This is a filtering floor and nothing
# more: filtfilt needs more samples than its own pad length, 3*(2*order+1) = 27 for order 4, and
# 40 leaves margin. It is NOT the floor for reporting a fit — that is MIN_FIT_SAMPLES, and
# conflating the two cost the whole marker-versus-biplane comparison on its first run. Set to 60
# on the reasoning that a shorter run is a pose rather than a motion, it silently discarded every
# biplane fluoroscopy window under 60 frames, which is most of them: those windows are 0.2-0.5 s
# at 250 Hz and they are the gold-standard half of this analysis.
MIN_RUN_SAMPLES = 40

# Total valid samples before a joint's MODEL FITS are reported. A 55-frame window determines
# seven structure parameters numerically and nothing kinematically, because the knee never left
# a narrow band of flexion, so a per-trial biplane fit is refused here and `pooled_fits` is where
# the biplane answer actually comes from. Every per-trial row carries `n_samples` so a reader can
# see which regime they are in.
MIN_FIT_SAMPLES = 100

# Samples before the two reference systems are COMPARED. Far lower than MIN_FIT_SAMPLES, and
# legitimately so: the comparison fits six constant parameters, not a joint model, and it is the
# one measurement that must survive on the shortest biplane windows because it is what those
# windows exist to provide. 40 samples against 6 parameters is a comfortable ratio.
MIN_AGREEMENT_SAMPLES = 40

# Samples used for the OUTER structure search. The inner per-sample angle solve and all
# reported errors run on the scoring set, which is far larger; this only bounds the cost of a
# 5-to-7 parameter search, and those are pinned down by a few hundred well-spread samples.
SUBSAMPLE_TARGET = 800

# Cap on samples the errors are computed over, applied as a uniform stride. Without it a
# 450k-sample IMoVE trial spends 2 s per model per fold inside the angle solve, which across
# 18 sensor pairs and 4 models and 4 folds is ten minutes for one trial. At 40k samples an RMS
# is estimated to far better than the two decimals anything is reported to. `--full-score`
# turns it off for a spot check.
SCORE_MAX_SAMPLES = 40_000

# Inner Gauss-Newton passes for the per-sample joint angles. Fixed rather than
# tolerance-based on purpose: the outer optimizer's finite-difference Jacobian needs the
# objective to be a deterministic function of the structure parameters, and an early-exit
# tolerance makes the iteration count jump between neighbouring evaluations.
INNER_GN_ITERS = 10


# Cap on outer objective evaluations per fit, so a badly conditioned joint cannot stall a
# 380-trial run. Convergence is recorded per fit rather than assumed.
MAX_OUTER_NFEV = 400
# The coupling's search only REFINES an axis and neutral pose already fitted by the hinge,
# with the coefficients solved exactly inside, so it needs far fewer evaluations.
MAX_CURVE_NFEV = 80


# Points at which the model curve is scanned to seed each sample's joint angle. See
# `curve_angle_seed`. GENEROUS ON PURPOSE, and the number was measured rather than guessed: on
# a synthetic joint the fit reaches 0.09 deg internally, a 48-point scan scores it at 0.37 and a
# 200-point scan at 0.09. The Gauss-Newton passes cannot rescue a seed in the wrong basin —
# 10, 20 and 40 of them all sat at 0.37 — so the resolution of this grid IS the accuracy floor
# of every coupling number in the module. The scan is one matrix multiply, ~30 ms at 256 points
# over 40k samples, which is a few percent of a joint's runtime.
# How far past the observed angle range the published coupling's seed scan reaches, as a
# fraction of that range. The projection that brackets it is biased when the curve bends, so the
# bracket is padded rather than trusted tight.
CURVE_SEED_SPAN_PAD = 0.3

# How far below zero flexion the fitted coupling may run before the report flags it. The
# polynomials are clipped there, so past this the model being scored is a constant in the
# non-sagittal channels rather than Reuben's curve. A few degrees is ordinary calibration
# slack; more than this and the residual is not about the published coupling.
COUPLING_CLIP_WARN_DEG = 5.0

# How much the published coupling must beat the hinge by before the report calls it a help.
# Below this the two models are the same model for practical purposes, and a fraction of a
# degree of in-sample advantage from one extra structure parameter is not a finding.
COUPLING_MATERIAL_GAIN_DEG = 0.5

# How far outside the swept flexion range the angle solve may look, each side. Wide enough that
# the seeded anatomical zero being a few degrees off cannot clip real motion, far short of the
# 2 pi that would let a sample jump onto another turn of the periodic clipped region.
COUPLING_FLEXION_MARGIN_RAD = np.deg2rad(30.0)

# Ratio of pooled to per-trial residual past which the report stops presenting both as equals
# and works out which one to believe. 3x is well clear of the spread a sound pooled fit shows
# (the marker sources sit at 1.7-2.1x) and well below the 13-16x the biplane half reaches.
POOLING_INFLATION_WARN = 3.0

# Trials a subject needs before the scatter of their per-trial fitted axes means anything. Two
# axes always agree to within their own separation, which says nothing about identifiability.
MIN_AXIS_SCATTER_TRIALS = 4

CURVE_SEED_GRID = 256


# Cross-validation. Contiguous blocks, interleaved across folds rather than split in half:
# a half-split of a trial that changes task partway through tests transfer between tasks,
# which is a different question and a much harder one. Interleaving keeps train and test
# covering the same session while still leaving whole blocks out, so neighbouring samples do
# not leak across the split.
CV_FOLDS = 3
CV_BLOCKS_PER_FOLD = 4

# Range of motion is a robust percentile spread, not min-to-max: a single dropped marker frame
# would otherwise set the ROM of every minor axis.
ROM_PERCENTILES = (2.5, 97.5)

# Sample stride for the per-sample error table only. Every scalar in joint_fits is computed on
# the scoring set, which SCORE_MAX_SAMPLES alone bounds.
SAMPLE_STRIDE = 25

# Bins for `error_curve`: is the hinge bad everywhere, or only past 60 deg of flexion? Binned
# on the fitted 1-DOF angle, which is comparable across models because they nest.
ANGLE_BINS = 18

# Below this carrying angle the 2-DOF fit is valid but its GEOMETRY is not interpretable. The
# two angles are separated only by the Jacobian columns [N^T u, v], whose conditioning is set
# by the angle between the axes; as they approach parallel, (theta1 + d, theta2 - d) leaves
# the model nearly unchanged, so the pair has a near-null direction. The total error stays
# meaningful — it is an achieved residual either way — but the carrying angle and the
# individual angle ranges drift along that null direction. Observed on the hips.
GEOMETRY_IDENTIFIABILITY_MIN_CARRYING_DEG = 30.0

# How closely a mounting-invariant geometric quantity must agree across subjects before it is
# called a property of the joint rather than of one recording. Loose on purpose: real
# inter-subject variation in joint axes is a few degrees to ~15, so anything inside this band
# is plausibly one population value and anything outside it is the fit chasing whichever
# motion the trial happened to contain.
CARRYING_ANGLE_AGREEMENT_DEG = 20.0

# Fraction of subjects whose 2-DOF geometry must be identified before the cohort carrying
# angle is quoted at all. Not 100%: with a full cohort a single badly conditioned fit should
# not veto an otherwise clean result.
MIN_IDENTIFIED_FRACTION = 0.7

# Which DOF sits inboard (parent-fixed u) versus outboard (child-fixed v) is a real structural
# choice, and at the knee the data cannot make it: the flexion axis is nearly fixed in BOTH
# segments, so both assignments fit to within noise. Letting a multi-start pick the winner
# therefore picks arbitrarily, and different subjects landed on different assignments, which
# silently destroyed cross-subject comparability — pooled axis dispersion 85 deg p90 against
# ~17 within an assignment. So the assignment is fixed by CONVENTION: the dominant DOF is
# always u. Set False only to reproduce the free-assignment behaviour, whose axes do not pool.
CANONICAL_ASSIGNMENT_ONLY = True

MODELS = ('weld', 'hinge', 'universal', 'spherical')

# Per-sample freedom, which is what makes two models' residuals comparable, and the count of
# structure parameters, which is what cross-validation charges for. The coupling's structure
# count is filled in per fit because it depends on the knot count.
# `knee_coupling` is the PUBLISHED 1-DOF nonlinear knee (Reuben et al. 1986, as IMoveLab
# enforces it). It is not in MODELS because it exists only at the knee, but everywhere else it
# is an ordinary rung: one angle per sample, five fitted structure parameters — the same five
# the hinge fits — and a curve that is published rather than fitted.
MODEL_DOF = {'weld': 0, 'hinge': 1, 'knee_coupling': 1, 'universal': 2, 'spherical': 3}
MODEL_STRUCTURE_PARAMS = {'weld': 3, 'hinge': 5, 'knee_coupling': 6, 'universal': 7,
                          'spherical': 0}

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

TRIAL_TABLES = ('joint_fits', 'error_samples', 'error_curve', 'coupling_shape',
                'reference_agreement')

AXIS_COMPONENTS = ('x', 'y', 'z')


# Reuben et al. (1986) knee coupling. Defined in src/joint_constraints.py, next to the SO(3)
# primitives, because src/RelativeFilterPlus.py evaluates the same curve to use the coupling as
# a filter measurement — this module measures how wrong the curve is, that one acts on it, and
# a second transcription of the coefficients is exactly the kind of drift neither would catch.
# Scored only when --knee-coupling is passed; see `knee_coupling_curve`.
REUBEN_ADDUCTION_COEFFS = joint_constraints.REUBEN_ADDUCTION_COEFFS
REUBEN_ROTATION_COEFFS = joint_constraints.REUBEN_ROTATION_COEFFS


def analysis_constants(dataset: str) -> Dict[str, object]:
    return {
        **pipeline_constants(),
        'dataset': dataset,
        'lowpass_hz': LOWPASS_HZ,
        'lowpass_order': LOWPASS_ORDER,
        'min_run_samples': MIN_RUN_SAMPLES,
        'min_fit_samples': MIN_FIT_SAMPLES,
        'min_agreement_samples': MIN_AGREEMENT_SAMPLES,
        'subsample_target': SUBSAMPLE_TARGET,
        'score_max_samples': SCORE_MAX_SAMPLES,
        'inner_gn_iters': INNER_GN_ITERS,
        'curve_seed_grid': CURVE_SEED_GRID,
        'curve_seed_span_pad': CURVE_SEED_SPAN_PAD,
        'coupling_clip_warn_deg': COUPLING_CLIP_WARN_DEG,
        'coupling_material_gain_deg': COUPLING_MATERIAL_GAIN_DEG,
        'pooling_inflation_warn': POOLING_INFLATION_WARN,
        'cv_folds': CV_FOLDS,
        'cv_blocks_per_fold': CV_BLOCKS_PER_FOLD,
        'rom_percentiles': list(ROM_PERCENTILES),
        'sample_stride': SAMPLE_STRIDE,
        'canonical_assignment_only': CANONICAL_ASSIGNMENT_ONLY,
        'geometry_identifiability_min_carrying_deg': GEOMETRY_IDENTIFIABILITY_MIN_CARRYING_DEG,
    }


# ==============================================================================
# DATASETS
# ==============================================================================
# The biplane spec lives HERE rather than in global_assumptions.DATASETS, which is otherwise
# this repo's dataset registry, and the reason is that the registry carries a contract this
# dataset cannot meet. Every spec in it is required to name a pelvis sensor and a pair of foot
# sensors (TestGlobalAssumptions asserts it), and the experiment that owns the registry is
# built on magnetometer physics. A biplane trial has four sensors — two BioStamps on one
# thigh and one shank, each seen by two reference systems — no pelvis, no feet, and no
# magnetometer at all. Registering it centrally would offer it as a `--dataset` choice to
# experiments that cannot analyse it, so it is declared where the experiment that can
# analyse it lives.
#
# The '__source' suffix on each joint key is what carries the marker-versus-biplane
# comparison through the ordinary per-joint machinery: `split_source` splits it back out into
# a joint and a reference, and every other dataset's joints come back as source 'markers'.

SOURCE_SEPARATOR = '__'
MARKER_SOURCE = 'markers'

_BIPLANE_SIDES = {'R': 'right', 'L': 'left'}
_BIPLANE_SOURCES = ('vicon', 'biplane')


def _biplane_segment_sensor() -> Dict[str, str]:
    return {f'{"Femur" if bone == "thigh" else "Tibia"} {side[0]} ({source})':
            f'lateral_{bone}_{full}{SOURCE_SEPARATOR}{source}'
            for side, full in _BIPLANE_SIDES.items()
            for bone in ('thigh', 'shank')
            for source in _BIPLANE_SOURCES}


def _biplane_joints() -> Dict[str, Tuple[str, str]]:
    """One knee per side per reference source.

    A biplane trial contains ONE knee — the side is in the trial name, LSDrop2 being a left
    single-leg drop — so the other side's entries are simply absent from the plate dict and
    skipped, exactly as a missing sensor is on any other dataset. Both are declared because a
    dataset spec describes the dataset, not one trial of it.
    """
    return {f'{side}_Knee{SOURCE_SEPARATOR}{source}':
            (f'lateral_thigh_{full}{SOURCE_SEPARATOR}{source}',
             f'lateral_shank_{full}{SOURCE_SEPARATOR}{source}')
            for side, full in _BIPLANE_SIDES.items()
            for source in _BIPLANE_SOURCES}


BIPLANE = DatasetSpec(
    name='imove_biplane',
    segment_sensor=_biplane_segment_sensor(),
    joints=_biplane_joints(),
    primary_joints=('R_Knee', 'L_Knee'),
    # No pelvis and no feet. Named as empty rather than faked, so any caller that needs them
    # gets nothing and says so instead of silently analysing the wrong sensor.
    pelvis_sensor='',
    foot_sensors=(),
    subject_label='{}',
    field_reference={},
)

DATASETS: Dict[str, DatasetSpec] = {**MAGNETIC_DATASETS, BIPLANE.name: BIPLANE}


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}. Registered: {sorted(DATASETS)}.") from None


def split_source(joint_key: str) -> Tuple[str, str]:
    """'R_Knee__biplane' -> ('R_Knee', 'biplane'); 'R_Knee_H' -> ('R_Knee_H', 'markers')."""
    if SOURCE_SEPARATOR in joint_key:
        joint, source = joint_key.split(SOURCE_SEPARATOR, 1)
        return joint, source
    return joint_key, MARKER_SOURCE


def joint_identity(joint_key: str, spec: DatasetSpec) -> Dict[str, str]:
    """The three names a joint row is keyed on: the raw spec key, the anatomical joint, the
    reference source, and the placement variant if the dataset has any.

    Kept in one place because three datasets encode three different things into the key —
    IMoVE a sensor placement, biplane a reference system, Al Borno neither — and every table
    and every report groups on the anatomical joint underneath.
    """
    joint, source = split_source(joint_key)
    canonical = canonical_joint(joint, spec)
    return {'joint_key': joint_key, 'joint': canonical, 'source': source,
            'placement': joint[len(canonical) + 1:] if joint != canonical else 'M'}


def has_multiple_sources(spec: DatasetSpec) -> bool:
    return len({split_source(key)[1] for key in spec.joints}) > 1


def selected_joints(spec: DatasetSpec, joints: Optional[Sequence[str]] = None,
                    placements: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, str]]:
    """{joint key: identity} for the joints a run is asked to fit.

    TWO independent filters, because IMoVE needs both and they are not the same question.
    `joints` selects an anatomical joint — R_Knee — and keeps every sensor placement on it.
    `placements` selects where on the segment the sensor sat: 'M' alone is the six bolted Mid
    pairs, which is the six-joint dataset most readers have in mind, while the full table is
    eighteen pairs and four times the runtime. Neither filter exists on the other two datasets,
    where every joint comes back as placement 'M' and the flags are no-ops.
    """
    wanted_joints = set(joints) if joints else None
    wanted_placements = set(placements) if placements else None
    selected = {}
    for joint_key in spec.joints:
        identity = joint_identity(joint_key, spec)
        if wanted_joints and identity['joint'] not in wanted_joints:
            continue
        if wanted_placements and identity['placement'] not in wanted_placements:
            continue
        selected[joint_key] = identity
    return selected


# ==============================================================================
# PATHS / IO
# ==============================================================================

def dataset_dir(dataset: str) -> Path:
    return EXPERIMENT_DIR / dataset


def trial_table_path(dataset: str, subject: str, trial: str, table: str) -> Path:
    return dataset_dir(dataset) / subject / trial / f"{table}.parquet"


def statistics_path(dataset: str) -> Path:
    return paths.statistics_path(f"{EXPERIMENT_NAME}_{dataset}")


def _save(df: pd.DataFrame, path: Path, dataset: str, **manifest_extra) -> None:
    df.to_parquet(paths.ensure_parent(path), engine='pyarrow', index=False)
    paths.write_manifest(path, constants=analysis_constants(dataset), experiment=EXPERIMENT_NAME,
                         n_rows=len(df), **manifest_extra)


def load_trial_table(dataset: str, table: str,
                     row_keys: Optional[Sequence[Tuple[str, str]]] = None,
                     columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Concatenates one per-trial table across trials, adding `subject` and `trial`."""
    if table not in TRIAL_TABLES:
        raise ValueError(f"Unknown table '{table}'; expected one of {TRIAL_TABLES}")
    row_keys = enumerate_trials(dataset) if row_keys is None else row_keys
    frames = []
    for subject, trial in row_keys:
        path = trial_table_path(dataset, subject, trial, table)
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path, engine='pyarrow',
                                      columns=list(columns) if columns else None)
                      .assign(subject=subject, trial=trial))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out['dataset'] = dataset
    return out


# ==============================================================================
# GEOMETRY HELPERS
# ==============================================================================

def _rodrigues(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """exp(theta[t] * [axis]) as a (T, 3, 3) stack, for a single FIXED axis.

    Closed form: the two powers of K are formed once and only the scalar coefficients vary
    with t. Used instead of Rotation.from_rotvec because this sits in the innermost loop of a
    nested optimization, where scipy's quaternion round-trip and per-call validation cost
    ~50x this.
    """
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    sin = np.sin(theta)[:, None, None]
    one_minus_cos = (1.0 - np.cos(theta))[:, None, None]
    return np.eye(3) + sin * K + one_minus_cos * (K @ K)


# The SO(3) primitives now live in src/toolchest/so3.py, because RelativeFilterPlus needs the
# same formulas to USE these models as filter measurements and src/ cannot import experiments/.
# Re-exported under the private names this module has always used, so every call site and every
# `jd._log_matrix` reference in test/TestJointDof.py keeps working against one implementation.
_skew = so3.skew
_exp_matrix = so3.exp_matrix
_right_jacobian = so3.right_jacobian
_log_matrix = so3.log_matrix
_transpose_multiply = so3.transpose_multiply
_tangent_basis = so3.tangent_basis


def _principal_axes(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending, clipped at 0) and matching unit eigenvectors as columns."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    return np.clip(eigenvalues[order], 0.0, None), eigenvectors[:, order]


def _rom_deg(x: np.ndarray) -> float:
    if len(x) == 0:
        return float('nan')
    lo, hi = np.percentile(x, ROM_PERCENTILES)
    return float((hi - lo) * RAD2DEG)


def report_angle(theta: np.ndarray) -> np.ndarray:
    """A fitted joint angle put on a footing that is comparable across trials.

    The raw angle out of the solver is not, for two reasons, and both were visible in the
    figures before this existed.

    IT WRAPS. theta and theta + 2 pi describe the same rotation, so the solver has no reason to
    prefer either and on two of Al Borno's nineteen trials it returned both — which made the
    percentile spread of a knee that moved 68 deg read as 339, in `theta1_rom_deg` and on every
    angle axis. Shifting each sample into the 2 pi window centred on the angle's own CIRCULAR
    mean removes the ambiguity without touching the model, because for the hinge the two are
    literally the same rotation.

    ITS ZERO IS ARBITRARY. The zero is wherever the fitted neutral pose landed, which is per
    trial: across those same nineteen trials one knee's median angle ran from -130 to +139 deg.
    Pooling raw angles therefore smeared a 68 deg range across 300. Re-centring on the trial's
    own median makes the pooled axis mean "how far from this trial's typical posture", which is
    a real quantity, rather than a sum of nineteen unrelated offsets.

    WHAT THIS CANNOT FIX is the SIGN. A joint axis is an undirected line, so u and -u are the
    same axis and negate theta with it; nothing short of an anatomical frame decides which way
    is flexion. Every angle axis in this module is therefore signed arbitrarily per trial, and
    the captions say so.
    """
    if len(theta) == 0:
        return theta
    mean = float(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))
    wrapped = mean + (theta - mean + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped - float(np.median(wrapped))


def axial_mean(axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean direction of a set of UNDIRECTED axes, plus each one's angle from it.

    A joint axis is a LINE, so u and -u are the same axis and an arithmetic mean would cancel
    to nothing given arbitrary signs. The principal eigenvector of sum(u u^T) is the
    sign-agnostic equivalent, and the returned angles are folded into [0, 90] to match.
    """
    axes = np.atleast_2d(np.asarray(axes, dtype=float))
    _, eigenvectors = _principal_axes(axes.T @ axes)
    mean_axis = eigenvectors[:, 0]
    angles = np.degrees(np.arccos(np.clip(np.abs(axes @ mean_axis), 0.0, 1.0)))
    return mean_axis, angles


# ==============================================================================
# THE SERIES: valid runs, filtering, scoring and cross-validation index sets
# ==============================================================================

@dataclass
class JointSeries:
    """One joint's relative rotation over the frames where BOTH plates are valid.

    `R` is low-passed and `R_raw` is not, sample for sample, so any fit can be re-scored
    against the unfiltered signal without refitting — which is how the jitter floor in every
    row is computed. `run_id` marks contiguous valid runs, which the filter respects and the
    cross-validation blocks never straddle.
    """
    R: np.ndarray
    R_raw: np.ndarray
    timestamps: np.ndarray
    run_id: np.ndarray
    fs: float
    n_valid_total: int
    n_runs: int


def valid_runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Contiguous [start, stop) runs of True at least `min_len` long."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def _lowpass_run(R: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """Low-passes a rotation series in the log space of its own chordal mean.

    Filtering around the mean rather than around identity keeps every sample far from the
    +-pi wrap where the log map is discontinuous, so no unwrapping is needed. The assumption
    is asserted rather than trusted: a surviving marker-swap flip puts a sample 180 deg from
    its mean and trips this instead of quietly corrupting the fit. (Those flips are detected
    and repaired upstream in WorldTrace.repair_reconstruction_glitches; this is the guard
    that it happened.)
    """
    rot = Rotation.from_matrix(R)
    ref = rot.mean()
    v = (ref.inv() * rot).as_rotvec()

    max_dev = float(np.max(np.linalg.norm(v, axis=1))) * RAD2DEG
    if max_dev >= 150.0:
        raise ValueError(f"rotation deviates {max_dev:.0f} deg from its own mean; log-space "
                         f"filtering is unsafe this close to the pi wrap")

    nyquist = 0.5 * fs
    if cutoff_hz >= nyquist:
        return R
    b, a = butter(LOWPASS_ORDER, cutoff_hz / nyquist, btype='low')
    return (ref * Rotation.from_rotvec(filtfilt(b, a, v, axis=0))).as_matrix()


def joint_series(parent: PlateTrial, child: PlateTrial, cutoff_hz: float = LOWPASS_HZ,
                 min_run: int = MIN_RUN_SAMPLES) -> Optional[JointSeries]:
    """R_pc(t) over the valid frames of both plates, low-passed per contiguous run.

    Returns None when nothing survives the run-length floor, which is the honest answer for a
    joint whose reference never held for long enough to be a motion.
    """
    n = min(len(parent), len(child))
    valid = np.asarray(parent.valid)[:n] & np.asarray(child.valid)[:n]
    runs = valid_runs(valid, min_run)
    if not runs:
        return None

    timestamps = np.asarray(parent.imu_trace.timestamps[:n], dtype=float)
    rotations_parent = np.asarray(parent.world_trace.rotations[:n], dtype=np.float64)
    rotations_child = np.asarray(child.world_trace.rotations[:n], dtype=np.float64)

    # The sample rate is taken from the WHOLE record, not from a run. A run may be short
    # enough that its own diff is dominated by timestamp quantization, and the grid is uniform
    # by construction — every reader puts a trial on one common grid before it is cached.
    fs = float(1.0 / np.median(np.diff(timestamps)))

    raw_parts, filt_parts, time_parts, run_parts = [], [], [], []
    for index, (start, stop) in enumerate(runs):
        block = _transpose_multiply(rotations_parent[start:stop], rotations_child[start:stop])
        raw_parts.append(block)
        filt_parts.append(_lowpass_run(block, fs, cutoff_hz))
        time_parts.append(timestamps[start:stop])
        run_parts.append(np.full(stop - start, index, dtype=np.int32))

    return JointSeries(R=np.ascontiguousarray(np.concatenate(filt_parts)),
                       R_raw=np.ascontiguousarray(np.concatenate(raw_parts)),
                       timestamps=np.concatenate(time_parts),
                       run_id=np.concatenate(run_parts),
                       fs=fs,
                       n_valid_total=int(valid.sum()),
                       n_runs=len(runs))


def concatenate_series(parts: Sequence[JointSeries]) -> JointSeries:
    """Stacks several trials' series into one, renumbering runs so none collide.

    This is what makes the pooled fit possible: a subject's biplane trials are 0.4 s each and
    only their union sweeps enough of the joint to determine an axis.
    """
    offsets, run_parts, cursor = [], [], 0
    for part in parts:
        run_parts.append(part.run_id + cursor)
        cursor += part.n_runs
        offsets.append(part)
    return JointSeries(
        R=np.ascontiguousarray(np.concatenate([p.R for p in offsets])),
        R_raw=np.ascontiguousarray(np.concatenate([p.R_raw for p in offsets])),
        timestamps=np.concatenate([p.timestamps for p in offsets]),
        run_id=np.concatenate(run_parts),
        fs=float(np.median([p.fs for p in offsets])),
        n_valid_total=int(sum(p.n_valid_total for p in offsets)),
        n_runs=cursor)


def scoring_index(n: int, cap: int) -> np.ndarray:
    """A uniform stride down to at most `cap` samples. See SCORE_MAX_SAMPLES."""
    if cap <= 0 or n <= cap:
        return np.arange(n)
    return np.arange(0, n, int(np.ceil(n / cap)))


def fit_index(index: np.ndarray, target: int = SUBSAMPLE_TARGET) -> np.ndarray:
    """A further stride, for the OUTER structure search only."""
    if len(index) <= target:
        return index
    return index[::int(np.ceil(len(index) / target))]


def cv_folds(run_id: np.ndarray, folds: int = CV_FOLDS,
             blocks_per_fold: int = CV_BLOCKS_PER_FOLD) -> List[np.ndarray]:
    """Fold membership as a list of boolean test masks over contiguous, interleaved blocks.

    Blocks never straddle a run boundary, because the two sides of a gap are minutes apart in
    the biplane trials and treating them as one contiguous stretch would put a test sample
    immediately beside its own training neighbour.
    """
    if folds < 2:
        return []
    n = len(run_id)
    block = np.zeros(n, dtype=np.int64)
    cursor = 0
    for run in np.unique(run_id):
        rows = np.flatnonzero(run_id == run)
        n_blocks = max(1, min(len(rows), folds * blocks_per_fold))
        block[rows] = cursor + np.minimum((np.arange(len(rows)) * n_blocks) // len(rows),
                                          n_blocks - 1)
        cursor += n_blocks
    masks = [(block % folds) == f for f in range(folds)]
    # A fold that captured nothing, or everything, is not a fold. Happens when a joint has
    # fewer samples than folds*blocks_per_fold, which the biplane trials brush against.
    return [m for m in masks if 0 < int(m.sum()) < n]


# ==============================================================================
# THE MODELS
# ==============================================================================
# A model is a plain dict carrying 'kind' plus its structure parameters, so it round-trips
# into a parquet row and back and can be scored on data it was not fitted on. `fit_model`
# produces one, `score_model` evaluates one, and nothing else needs to know the difference
# between a hinge and a curved 1-DOF joint.

def _model_residual(target: np.ndarray, u: np.ndarray, R_0: np.ndarray, theta1: np.ndarray,
                    v: Optional[np.ndarray], theta2: Optional[np.ndarray]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """(residual rotvec, N^T u) for the chain N = exp(theta1[u]) R_0 exp(theta2[v])."""
    model = _rodrigues(u, theta1) @ R_0
    if v is not None:
        model = model @ _rodrigues(v, theta2)
    return _log_matrix(_transpose_multiply(model, target)), np.einsum('tji,j->ti', model, u)


def _solve_chain_angles(target: np.ndarray, u: np.ndarray, R_0: np.ndarray,
                        theta1: np.ndarray, v: Optional[np.ndarray],
                        theta2: Optional[np.ndarray], iters: int = INNER_GN_ITERS
                        ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    r"""Geodesic-optimal joint angles for a fixed hinge or universal structure.

    Model N(t) = exp(theta1 [u]) R_0 exp(theta2 [v]), residual r = log(N^T target).

    The Jacobian falls out of how each angle perturbs N:
        theta1 + d  ->  N' = exp(d [u]) N   =>  N'^T target = exp(-d [N^T u]) Delta
        theta2 + d  ->  N' = N exp(d [v])   =>  N'^T target = exp(-d [v])     Delta
    so to first order r_new = r - d1 (N^T u) - d2 v, and the step solves [N^T u, v] d = r in
    least squares. Passing v=None drops the second column and gives the hinge, which is why
    the two share this routine: the models nest, and so do their solvers.

    The angles are therefore the true GEODESIC optimum for the given structure, not a linear
    projection into a tangent plane — which matters at the 60-90 deg ranges these joints
    actually visit, where the two differ by degrees.
    """
    for _ in range(iters):
        residual, jac_1 = _model_residual(target, u, R_0, theta1, v, theta2)
        if v is None:
            # One column, unit norm, so the least-squares step is just the projection.
            theta1 = theta1 + np.einsum('ti,ti->t', jac_1, residual)
            continue
        jac = np.stack([jac_1, np.broadcast_to(v, jac_1.shape)], axis=2)  # (T, 3, 2)
        # Ridge term: u and v can become near-parallel mid-search, which makes the 2x2 normal
        # matrix singular. 1e-9 is far below any real step and only bites there.
        normal = np.einsum('tik,til->tkl', jac, jac) + 1e-9 * np.eye(2)
        rhs = np.einsum('tik,ti->tk', jac, residual)
        # rhs needs a trailing axis: numpy's batched solve reads a bare (T, 2) as a single
        # 2-column right-hand side rather than T separate 2-vectors.
        step = np.linalg.solve(normal, rhs[..., None])[..., 0]
        theta1 = theta1 + step[:, 0]
        theta2 = theta2 + step[:, 1]

    residual, _ = _model_residual(target, u, R_0, theta1, v, theta2)
    return theta1, theta2, residual


# ------------------------------------------------------------------------------
# The 1-DOF NONLINEAR CURVE — the published knee coupling
# ------------------------------------------------------------------------------
# One angle per sample, but the two channels perpendicular to the axis follow a fixed
# nonlinear function of that angle instead of staying at zero:
#
#     R_pc(q) = R_0 exp([ q u_c + P c(q) ]),   P an orthonormal basis of u_c's normal plane
#
# Fixing the component of the rotation vector along u_c to be exactly q is the gauge that makes
# q well defined, and it makes the HINGE the exact special case c == 0 — so the two models nest
# and the drop between them is exactly what the curvature bought.
#
# `c` is Reuben's pair of quartics and is NOT fitted. An earlier version also fitted a free
# cubic B-spline here, as an envelope on what any 1-DOF nonlinear model could achieve; that has
# been removed at the user's direction, so this file now measures the published coupling only.
# The consequence is worth stating where it will be read: a coupling residual can no longer be
# separated into "the wrong curve" and "no curve would have helped".


# (c, dc/dq) of the two published channels, evaluated exactly. Defined in
# src/joint_constraints.py so the filter's coupling measurement evaluates the identical curve;
# see that module for why the projection-onto-a-spline route was abandoned (0.84 deg of RMS on
# a synthetic knee whose true residual is zero) and why flexion is clipped at zero.
#
# A held-out fold reaching past the training fold's extension is the reason the clip matters
# HERE; the filter has a second reason, which is that it evaluates the curve at its own
# estimated flexion and that estimate can start on the wrong side of zero.
_reuben_design = joint_constraints.reuben_design


def _curve_nu(q: np.ndarray, u_c: np.ndarray, basis_P: np.ndarray, coeffs: np.ndarray,
              basis: np.ndarray, derivative: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(nu(q), dnu/dq) for nu = q u_c + P c(q). `coeffs` is the 2x2 channel frame map."""
    nu = q[:, None] * u_c + (basis @ coeffs) @ basis_P.T
    dnu = u_c + (derivative @ coeffs) @ basis_P.T
    return nu, dnu


def curve_angle_seed(target: np.ndarray, u_c: np.ndarray, R_0: np.ndarray, sigma: float,
                     span: Tuple[float, float], coeffs: np.ndarray,
                     n_grid: int = CURVE_SEED_GRID) -> np.ndarray:
    r"""The nearest point on the model curve to each sample, found by exhaustive scan.

    NOT an optimization — a scan, and that is the point. Seeding the angle by projecting the
    sample onto the axis, as the hinge legitimately can, ASSUMES the curve is close to a
    geodesic. Once the curve bends, that seed lands some samples on the wrong stretch of it and
    the Gauss-Newton passes then converge neatly to a local minimum tens of degrees away. The
    failure is silent and one-sided: it inflates the error of exactly the samples where the
    coupling matters most, so the coupling looks worse than the hinge it contains.

    The scan is cheap because geodesic distance is monotone in the trace, and the trace of
    C(q)^T R is the flat dot product of the two matrices — so the whole T x n_grid distance
    table is one matrix multiply and the largest entry per row is the seed.
    """
    grid = np.linspace(span[0], span[1], n_grid)
    basis, derivative = _reuben_design(grid, sigma)
    nu, _ = _curve_nu(grid, u_c, _tangent_basis(u_c), coeffs, basis, derivative)
    curve = R_0 @ _exp_matrix(nu)
    traces = target.reshape(len(target), 9) @ curve.reshape(n_grid, 9).T
    return grid[np.argmax(traces, axis=1)]


def solve_curve_angles(target: np.ndarray, u_c: np.ndarray, R_0: np.ndarray, sigma: float,
                       coeffs: np.ndarray, q: Optional[np.ndarray] = None,
                       span: Optional[Tuple[float, float]] = None,
                       inner_iters: int = INNER_GN_ITERS
                       ) -> Tuple[np.ndarray, np.ndarray]:
    r"""Gauss-Newton for the per-sample joint angle q(t) against a FIXED curve.

    Model  N(q) = R_0 exp([nu(q)]),  residual r = log(N^T target).

    Perturbing the rotation vector by dnu sends exp([nu]) -> exp([nu]) exp([J_r(nu) dnu]), so
    N^T target -> exp([-J_r dnu]) Delta and r_new = r - J_r(nu) dnu to first order. With
    dnu = (dnu/dq) dq the column is g = J_r(nu) (u_c + P c'(q)) and the step is the projection
    (g.r)/(g.g) — the hinge's update with a curved generator, reducing to it exactly when
    c == 0.

    Only the angle is solved: the curve is published, so there are no coefficients to alternate
    onto and the outer loop the free-spline version needed is gone with it.
    """
    basis_P = _tangent_basis(u_c)
    if q is None:
        if span is None:
            # The angle range to scan, in THIS model's gauge: project each sample's displacement
            # from the neutral pose onto the axis. As a per-sample seed that projection is
            # unsound once the curve bends — which is what `curve_angle_seed` is for — but as a
            # bracket for the scan it is right, and it moves with R_0 as the search does.
            v_log = _log_matrix(_transpose_multiply(
                np.broadcast_to(R_0, (len(target), 3, 3)), target))
            projected = v_log @ u_c
            lo, hi = float(projected.min()), float(projected.max())
            pad = CURVE_SEED_SPAN_PAD * (hi - lo) + 1e-3
            span = (lo - pad, hi + pad)
        q = curve_angle_seed(target, u_c, R_0, sigma, span, coeffs)

    for _ in range(inner_iters):
        basis, derivative = _reuben_design(q, sigma)
        nu, dnu = _curve_nu(q, u_c, basis_P, coeffs, basis, derivative)
        jac_right = _right_jacobian(nu)
        residual = _log_matrix(_transpose_multiply(R_0 @ _exp_matrix(nu), target))
        g = np.einsum('tij,tj->ti', jac_right, dnu)
        q = q + (np.einsum('ti,ti->t', g, residual)
                 / np.maximum(np.einsum('ti,ti->t', g, g), 1e-12))

    basis, derivative = _reuben_design(q, sigma)
    nu, _ = _curve_nu(q, u_c, basis_P, coeffs, basis, derivative)
    return q, _log_matrix(_transpose_multiply(R_0 @ _exp_matrix(nu), target))



# ------------------------------------------------------------------------------
# Fitting
# ------------------------------------------------------------------------------

def pca_initialization(R: np.ndarray) -> Dict[str, object]:
    """Displacement PCA of the relative rotation: the seed for every fit, and the ROM table.

    The rotations are logged about their own chordal mean, so the principal directions are of
    the DISPLACEMENT from neutral rather than of the absolute orientation. PC1 is the joint's
    dominant axis and is what seeds u; PC2 seeds v.
    """
    rot = Rotation.from_matrix(R)
    ref = rot.mean()
    v_log = (ref.inv() * rot).as_rotvec()
    eigenvalues, eigenvectors = _principal_axes(np.cov(v_log.T) if len(v_log) > 1
                                                else np.zeros((3, 3)))
    total = float(eigenvalues.sum())
    scores = v_log @ eigenvectors
    return {
        'ref': ref,
        'ref_matrix': ref.as_matrix(),
        'eigenvectors': eigenvectors,
        'eigenvalues': eigenvalues,
        'scores': scores,
        'roms': np.array([_rom_deg(scores[:, k]) for k in range(3)]),
        'var_frac1': float(eigenvalues[0] / total) if total > 0 else float('nan'),
        'var_frac12': float(eigenvalues[:2].sum() / total) if total > 0 else float('nan'),
        # u lives in the PARENT frame (it left-multiplies), so PC1 is rotated out of the
        # neutral child frame that v_log lives in. v is a CHILD-frame axis and is left alone.
        'u0': ref.as_matrix() @ eigenvectors[:, 0],
        'v0': eigenvectors[:, 1],
    }


def fit_weld(R: np.ndarray) -> Dict[str, object]:
    """The best constant relative rotation — the chordal mean. No per-sample freedom."""
    return {'kind': 'weld', 'R_0': Rotation.from_matrix(R).mean().as_matrix()}


def fit_chain(R: np.ndarray, n_dof: int, u0: np.ndarray, R0_init: np.ndarray,
              theta1_init: np.ndarray, v0: Optional[np.ndarray] = None,
              theta2_init: Optional[np.ndarray] = None, free_axes: bool = True,
              max_nfev: int = MAX_OUTER_NFEV) -> Dict[str, object]:
    """Fits the hinge (n_dof=1) or universal (n_dof=2) model on the samples it is given.

    Axes are parameterized as tangent-plane offsets from their initial guesses, which keeps
    the search free of the gimbal singularity a spherical parameterization would introduce.

    `free_axes=False` holds the axes at exactly u0 (and v0) and fits only R_0. That is the
    generalization test for a PROPOSED axis: the axis is the transferable claim, whereas R_0
    is the neutral relative pose, which legitimately differs per subject through both anatomy
    and sensor mounting and which any real method calibrates per subject anyway.
    """
    basis_u = _tangent_basis(u0)
    basis_v = _tangent_basis(v0) if v0 is not None else None
    R0_matrix = np.asarray(R0_init, dtype=float)

    def unpack(params: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        if not free_axes:
            return (u0, None if n_dof == 1 else v0,
                    R0_matrix @ Rotation.from_rotvec(params[0:3]).as_matrix())
        u = u0 + basis_u @ params[0:2]
        u = u / np.linalg.norm(u)
        offset = 2 if n_dof == 1 else 4
        R_0 = R0_matrix @ Rotation.from_rotvec(params[offset:offset + 3]).as_matrix()
        if n_dof == 1:
            return u, None, R_0
        v = v0 + basis_v @ params[2:4]
        return u, v / np.linalg.norm(v), R_0

    def residuals(params: np.ndarray) -> np.ndarray:
        u, v, R_0 = unpack(params)
        _, _, res = _solve_chain_angles(R, u, R_0, theta1_init.copy(), v,
                                        None if theta2_init is None else theta2_init.copy())
        return res.ravel()

    n_params = (5 if n_dof == 1 else 7) if free_axes else 3
    solution = least_squares(residuals, np.zeros(n_params), method='trf',
                             xtol=1e-8, ftol=1e-8, max_nfev=max_nfev)
    u, v, R_0 = unpack(solution.x)
    model: Dict[str, object] = {
        'kind': 'hinge' if n_dof == 1 else 'universal',
        'axis_parent': u, 'R_0': R_0,
        'nfev': int(solution.nfev), 'converged': bool(solution.status > 0),
    }
    if n_dof == 2:
        model['axis_child'] = v
        # Both axes in the parent frame at the neutral pose, where child <- parent is R_0.
        # Folded into [0, 90] because a hinge axis is an undirected line, so 105 deg and
        # 75 deg are the same geometry.
        model['carrying_angle_deg'] = float(np.degrees(np.arccos(
            abs(float(np.clip(np.dot(u, R_0 @ v), -1.0, 1.0))))))
    return model


def score_model(model: Dict[str, object], R: np.ndarray, q_init: Optional[np.ndarray] = None
                ) -> Dict[str, np.ndarray]:
    """Per-sample geodesic error in degrees for a fitted model on ANY samples.

    The per-sample joint angles are always re-solved — they are the model's per-sample freedom,
    not part of its structure — so scoring a fitted structure on a held-out fold is exactly
    what cross-validation needs and no coefficient is refitted while doing it.
    """
    kind = model['kind']
    if kind == 'spherical':
        return {'error_deg': np.zeros(len(R)), 'theta1': np.zeros(len(R)),
                'theta2': np.zeros(len(R))}
    if kind == 'weld':
        residual = _log_matrix(_transpose_multiply(
            np.broadcast_to(np.asarray(model['R_0']), (len(R), 3, 3)), R))
        return {'error_deg': np.linalg.norm(residual, axis=1) * RAD2DEG,
                'theta1': np.zeros(len(R)), 'theta2': np.zeros(len(R))}

    R_0 = np.asarray(model['R_0'], dtype=float)

    if kind == 'coupling':
        # Seeded by scanning the curve rather than by projection, which is only valid while the
        # curve is nearly a geodesic; see `curve_angle_seed`. A caller-supplied seed is honoured
        # for the cross-check in the tests.
        q, residual = solve_curve_angles(
            R, np.asarray(model['axis_child']), R_0, float(model['coupling_flexion_sign']),
            np.asarray(model['coeffs']), None if q_init is None else q_init.copy(),
            span=model.get('q_span'))
        return {'error_deg': np.linalg.norm(residual, axis=1) * RAD2DEG,
                'theta1': q, 'theta2': np.zeros(len(R))}

    if q_init is None:
        # Seed by projecting the displacement from neutral onto the axis that drives it,
        # expressed in the frame the displacement lives in. Sound for the chain models: their
        # curves ARE geodesics, so the projection is the exact answer up to curvature of the
        # group itself.
        v_log = _log_matrix(_transpose_multiply(np.broadcast_to(R_0, (len(R), 3, 3)), R))
        q_init = v_log @ (R_0.T @ np.asarray(model['axis_parent']))

    v = np.asarray(model['axis_child']) if kind == 'universal' else None
    theta2 = np.zeros(len(R)) if v is not None else None
    theta1, theta2, residual = _solve_chain_angles(
        R, np.asarray(model['axis_parent']), R_0, q_init.copy(), v, theta2)
    return {'error_deg': np.linalg.norm(residual, axis=1) * RAD2DEG,
            'theta1': theta1,
            'theta2': theta2 if theta2 is not None else np.zeros(len(R))}


def fit_all_models(R_fit: np.ndarray, init: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """Every model in the ladder, fitted on `R_fit`, seeded from one shared PCA.

    The 2-DOF assignment is fixed by convention rather than multi-started; see
    CANONICAL_ASSIGNMENT_ONLY for the measurement that forced that.
    """
    scores = init['scores']
    models = {'weld': fit_weld(R_fit), 'spherical': {'kind': 'spherical'}}

    models['hinge'] = fit_chain(R_fit, 1, init['u0'], init['ref_matrix'],
                                scores[:, 0].copy())

    candidates = [fit_chain(R_fit, 2, init['u0'], init['ref_matrix'], scores[:, 0].copy(),
                            v0=init['eigenvectors'][:, 1], theta2_init=scores[:, 1].copy())]
    if not CANONICAL_ASSIGNMENT_ONLY:
        candidates.append(fit_chain(
            R_fit, 2, init['ref_matrix'] @ init['eigenvectors'][:, 1], init['ref_matrix'],
            scores[:, 1].copy(), v0=init['eigenvectors'][:, 0], theta2_init=scores[:, 0].copy()))
    models['universal'] = min(candidates, key=lambda m: float(np.sum(
        score_model(m, R_fit)['error_deg'] ** 2)))

    return models


# ==============================================================================
# STATISTICS
# ==============================================================================

def error_stats(error_deg: np.ndarray, prefix: str = '') -> Dict[str, float]:
    """The quantile spine, on the same grid as every other summary in this repo."""
    if len(error_deg) == 0:
        return {}
    stats = {
        f'{prefix}rms_deg': float(np.sqrt(np.mean(error_deg ** 2))),
        f'{prefix}mean_deg': float(np.mean(error_deg)),
        f'{prefix}max_deg': float(np.max(error_deg)),
    }
    for quantile in QUANTILES:
        stats[f'{prefix}p{int(round(quantile * 100)):02d}_deg'] = float(
            np.quantile(error_deg, quantile))
    return stats


def residual_anisotropy(residual_rotvec: np.ndarray, axis_parent: Optional[np.ndarray],
                        R_0: Optional[np.ndarray]) -> Dict[str, float]:
    """Is what the model missed one channel or all three?

    A hinge residual concentrated on a single direction is a missing SECOND AXIS — the joint
    has a DOF the model does not — and adding one will fix it. A residual spread evenly over
    all three directions is noise, soft tissue, or a joint that simply is not a mechanism, and
    no amount of extra structure will help. The two cases have the same RMS and completely
    different implications, and nothing else in the table separates them.

    `along_frac` additionally asks whether the leftover sits ALONG the fitted axis, which it
    cannot if the angle solve converged — the solve drives exactly that component to zero — so
    a nonzero value is a convergence check rather than a measurement.
    """
    if len(residual_rotvec) < 2:
        return {}
    second_moment = residual_rotvec.T @ residual_rotvec / len(residual_rotvec)
    eigenvalues, eigenvectors = _principal_axes(second_moment)
    total = float(eigenvalues.sum())
    out = {
        'resid_frac1': float(eigenvalues[0] / total) if total > 0 else float('nan'),
        'resid_frac12': float(eigenvalues[:2].sum() / total) if total > 0 else float('nan'),
    }
    if axis_parent is not None and R_0 is not None:
        axis_child = np.asarray(R_0, dtype=float).T @ np.asarray(axis_parent, dtype=float)
        along = float(axis_child @ second_moment @ axis_child)
        out['resid_along_axis_frac'] = along / total if total > 0 else float('nan')
        out['resid_principal_from_axis_deg'] = float(np.degrees(np.arccos(np.clip(
            abs(float(eigenvectors[:, 0] @ axis_child)), 0.0, 1.0))))
    return out


def excitation(R: np.ndarray) -> Dict[str, float]:
    r"""A HINGE TEST THAT INVOLVES NO FITTING, from the singular values of sum_t R_pc(t).

    Write S = sum_t R_pc(t), with singular values scaled by N so they lie in [0, 1]. For a
    hinge R_pc(t) = exp(theta_t [u]) R_0, so S = (sum_t exp(theta_t [u])) R_0 and the left
    factor is N u u^T + (sum cos theta) (I - u u^T) + (sum sin theta) [u]_x. Its singular
    values are therefore exactly

        sigma = { N,  r,  r },      r = | sum_t exp(i theta_t) | <= N

    and R_0 being orthogonal leaves them unchanged. Two readings fall out, and getting them the
    right way round matters — the first version of this function had them swapped:

      * `excitation` = 1 - sigma_min/N is 0 for a WELDED pair and grows with any motion at all.
        It is the "did this joint move" number, and it is why a residual is meaningless without
        it: a joint that did not move is fitted perfectly by every model on the ladder.

      * `hinge_deficiency` = 1 - sigma_max/N is 0 for a pure hinge — and also for a weld, which
        is a hinge that never moved. It is nonzero exactly when the relative rotation has NO
        common invariant direction, i.e. when the pair genuinely needs more than one axis.

    `hinge_deficiency` is the useful one, because it answers the module's central question by a
    single SVD with nothing fitted. Its agreement with the fitted hinge residual sitting beside
    it in the same row is an independent check on the whole optimizer.
    """
    if len(R) < 2:
        return {}
    sigma = np.linalg.svd(R.sum(axis=0), compute_uv=False) / len(R)
    return {'excitation': float(1.0 - sigma[2]),
            'hinge_deficiency': float(1.0 - sigma[0]),
            'excitation_mid': float(1.0 - sigma[1])}


def coupling_curve(model: Dict[str, object], q_grid: np.ndarray) -> np.ndarray:
    """The 1-DOF model's own rotation at each value of its joint angle."""
    u_c = np.asarray(model['axis_child'], dtype=float)
    basis, derivative = _reuben_design(q_grid, float(model['coupling_flexion_sign']))
    nu, _ = _curve_nu(q_grid, u_c, _tangent_basis(u_c),
                      np.asarray(model['coeffs'], dtype=float), basis, derivative)
    return np.asarray(model['R_0'], dtype=float) @ _exp_matrix(nu)


def coupling_geodesic_departure(model: Dict[str, object], q: np.ndarray, n_grid: int = 64
                              ) -> Optional[Dict[str, np.ndarray]]:
    r"""How far the fitted 1-DOF curve strays from the best GEODESIC through it.

    THE GAUGE PROBLEM THIS EXISTS TO SOLVE. It is tempting to report the curve's coefficients
    themselves as the coupling, and the first version of this file did. They are not a
    property of the joint: the outer search is free to move R_0 and u_c, and moving R_0 slides
    curvature into and out of the coefficient vector at will. Measured on a synthetic joint
    with a known 8 deg quadratic coupling, two fits with identical 0.008 deg residuals
    reported coefficient departures of 8.6 deg and 44.8 deg. Nothing derived from the
    coefficients directly can be quoted.

    What IS invariant is the CURVE the model traces in SO(3), and the honest measure of its
    curvature is its distance from the nearest one-parameter subgroup — that is, from the best
    hinge. So the model is sampled over the angle range the joint actually visited, a hinge is
    fitted to those rotations alone, and the leftover is reported. That leftover is by
    construction what a free-axis hinge could NOT have absorbed, which is exactly the quantity
    the ladder's hinge-to-coupling drop is paid for.

    The leftover is also returned resolved into the hinge's own neutral frame, which is what
    makes it comparable to a published coupling: `along` is the component on the hinge axis
    (zero to solver precision — the angle solve drives it there) and `perp1`, `perp2` are the
    two off-axis channels. Their split between the two perpendicular directions is arbitrary
    up to a rotation of that plane, which no amount of care can fix without an anatomical
    frame; their MAGNITUDE is not.
    """
    lo, hi = np.percentile(q, ROM_PERCENTILES)
    if not np.isfinite(lo) or hi - lo < 1e-6:
        return None
    grid = np.linspace(lo, hi, n_grid)
    curve = coupling_curve(model, grid)

    init = pca_initialization(curve)
    hinge = fit_chain(curve, 1, init['u0'], init['ref_matrix'], init['scores'][:, 0].copy())
    scored = score_model(hinge, curve)
    theta = scored['theta1']
    residual = _residual_rotvec(hinge, curve, scored)

    # The residual is in the model's body frame at each theta; rotating by exp(theta [u_c])
    # carries it back to the hinge's NEUTRAL child frame, where the same physical direction
    # has the same coordinates at every angle and the two channels can be plotted as curves.
    u_child = np.asarray(hinge['R_0']).T @ np.asarray(hinge['axis_parent'])
    neutral = np.einsum('tij,tj->ti', _rodrigues(u_child, theta), residual)
    perpendicular = _tangent_basis(u_child)
    return {
        # The MODEL's own joint angle, not the inner hinge fit's theta. They agree up to an
        # offset when the inner fit is well posed, and when it is not — a nearly straight curve
        # leaves its axis barely determined — theta runs away while the departure it is used to
        # transport stays correct. Reporting theta put four of Al Borno's seven joints on a
        # +-180 deg axis when no joint moved past 70.
        'angle_deg': grid * RAD2DEG,
        'along_deg': (neutral @ u_child) * RAD2DEG,
        'perp_deg': (neutral @ perpendicular) * RAD2DEG,
        'magnitude_deg': np.linalg.norm(neutral, axis=1) * RAD2DEG,
    }


def coupling_curve_stats(model: Dict[str, object], q: np.ndarray) -> Dict[str, float]:
    """Invariant curvature of the fitted 1-DOF curve. See `coupling_geodesic_departure`."""
    departure = coupling_geodesic_departure(model, q)
    lo, hi = np.percentile(q, ROM_PERCENTILES)
    stats = {'curvature_range_deg': float((hi - lo) * RAD2DEG)}
    if departure is None:
        return stats
    return {
        **stats,
        'curvature_max_deg': float(np.max(departure['magnitude_deg'])),
        'curvature_rms_deg': float(np.sqrt(np.mean(departure['magnitude_deg'] ** 2))),
        # A convergence check on the inner hinge fit, not a measurement: the angle solve drives
        # the on-axis component to zero, so anything but ~0 means that fit did not converge.
        'curvature_along_axis_deg': float(np.max(np.abs(departure['along_deg']))),
    }


# ==============================================================================
# THE PER-JOINT FIT
# ==============================================================================

@dataclass
class JointResult:
    """Everything one (joint, source) produced: a row per model, plus the sample-level tables."""
    rows: List[Dict[str, object]]
    samples: pd.DataFrame
    curve: pd.DataFrame
    coupling: pd.DataFrame
    models: Dict[str, Dict[str, object]]


def analyze_joint(series: JointSeries, identity: Dict[str, str],
                  folds: int = CV_FOLDS, stride: int = SAMPLE_STRIDE,
                  score_cap: int = SCORE_MAX_SAMPLES,
                  knee_coupling: bool = True) -> Optional[JointResult]:
    """Fits and scores the whole ladder on one joint's series."""
    score_idx = scoring_index(len(series.R), score_cap)
    if len(score_idx) < MIN_FIT_SAMPLES:
        return None
    R_score = np.ascontiguousarray(series.R[score_idx])
    R_score_raw = np.ascontiguousarray(series.R_raw[score_idx])
    run_score = series.run_id[score_idx]

    init = pca_initialization(R_score)
    R_fit = np.ascontiguousarray(series.R[fit_index(score_idx)])
    init_fit = pca_initialization(R_fit)
    models = fit_all_models(R_fit, init_fit)

    shared = {
        **identity,
        'n_samples': int(len(series.R)),
        'n_scored': int(len(score_idx)),
        'n_valid_frames': int(series.n_valid_total),
        'n_runs': int(series.n_runs),
        'fs_hz': float(series.fs),
        'duration_s': float(len(series.R) / series.fs),
        'rom1_deg': float(init['roms'][0]), 'rom2_deg': float(init['roms'][1]),
        'rom3_deg': float(init['roms'][2]),
        'var_frac1': init['var_frac1'], 'var_frac12': init['var_frac12'],
        **excitation(R_score),
    }

    # The published coupling joins the ladder as an ordinary rung rather than as an appendix,
    # so it picks up the raw re-score, the residual anisotropy, the per-sample error columns and
    # the error-against-angle curve that every other model gets. It needs the hinge's
    # fitted axis and neutral pose, which is why it is built here and not inside
    # `fit_all_models`; and it needs the RAW hinge angle, which the loop below records.
    score_coupling = knee_coupling and identity['joint'].endswith('Knee')
    cv_by_model = cross_validate(R_score, cv_folds(run_score, folds), score_coupling)
    if score_coupling:
        models['knee_coupling'] = fit_knee_coupling(models['hinge'], R_fit)

    rows, sample_columns, angle_column = [], {}, None
    # Each curve model's own joint angle, kept so `_coupling_shape` can set each curve's range
    # from the angles THAT model visited. It used to be handed the hinge's re-centred angle,
    # which is a different gauge and a different zero.
    curve_angles: Dict[str, np.ndarray] = {}
    for name in [*MODELS, *(['knee_coupling'] if score_coupling else [])]:
        model = models[name]
        scored = score_model(model, R_score)
        error = scored['error_deg']

        row: Dict[str, object] = {
            **shared, 'model': name, 'n_dof': MODEL_DOF[name],
            'n_structure_params': int(model.get('n_structure_params',
                                                MODEL_STRUCTURE_PARAMS[name])),
            **error_stats(error),
            'converged': bool(model.get('converged', True)),
            'nfev': int(model.get('nfev', 0)),
        }

        # Re-scored against the UNFILTERED rotations, with no refit. The gap between the two
        # is the marker jitter the low-pass removed, and it is the floor this model's residual
        # cannot go below on raw data however good the mechanism is.
        row.update(error_stats(score_model(model, R_score_raw)['error_deg'], prefix='raw_'))

        if name != 'spherical':
            R_0 = model.get('R_0')
            residual_rotvec = (
                _log_matrix(_transpose_multiply(
                    np.broadcast_to(np.asarray(R_0), (len(R_score), 3, 3)), R_score))
                if name == 'weld' else _residual_rotvec(model, R_score, scored))
            row.update(residual_anisotropy(residual_rotvec, model.get('axis_parent'), R_0))

        for key, prefix in (('axis_parent', 'u'), ('axis_child', 'v')):
            if key in model:
                for i, component in enumerate(AXIS_COMPONENTS):
                    row[f'{prefix}_{component}'] = float(np.asarray(model[key])[i])
        if 'carrying_angle_deg' in model:
            row['carrying_angle_deg'] = float(model['carrying_angle_deg'])
            row['geometry_identified'] = bool(
                model['carrying_angle_deg'] >= GEOMETRY_IDENTIFIABILITY_MIN_CARRYING_DEG)
        # The spherical joint is excluded even though its DOF count says otherwise: it has no
        # fitted angles at all (its residual is zero by construction), so `score_model` returns
        # zeros and a 0.0 range in the table would read as "this joint did not move".
        if name != 'spherical':
            if MODEL_DOF[name] >= 1:
                row['theta1_rom_deg'] = _rom_deg(report_angle(scored['theta1']))
            if MODEL_DOF[name] >= 2:
                row['theta2_rom_deg'] = _rom_deg(report_angle(scored['theta2']))
        # Curvature for BOTH 1-DOF nonlinear models, measured the same invariant way, so the
        # measured the same gauge-invariant way that the hinge-to-coupling drop is, so the two
        # numbers can be read against each other.
        if name == 'knee_coupling':
            row.update(coupling_curve_stats(model, scored['theta1']))
            curve_angles[name] = scored['theta1']
        for key in ('coupling_flexion_sign', 'coupling_handedness',
                    'coupling_channel_angle_deg', 'coupling_min_flexion_deg'):
            if key in model:
                row[key] = float(model[key])

        row.update(cv_by_model.get(name, {}))
        rows.append(row)

        sample_columns[f'error_{name}_deg'] = error.astype(np.float32)
        if name == 'hinge':
            # TWO angles, and they are not interchangeable. `angle_column` is the REPORTING
            # angle — unwrapped and re-centred — which is what every sample-level table is
            # binned on so the bins mean the same thing across trials. `hinge_angle` is the raw
            # solver output, which is the only one the published coupling can be scored in:
            # that scoring evaluates the published polynomials, whose argument is flexion in
            # the model's own gauge, so a re-centred angle would shift the whole curve.
            hinge_angle = scored['theta1']
            angle_column = report_angle(hinge_angle)

    # Every model's error as a fraction of the weld's, which is the only denominator that
    # makes a 2 deg residual on a joint that moved 8 deg read differently from a 2 deg
    # residual on a joint that moved 90. Applied last so every row has it, including the
    # published coupling's.
    weld_rms = next(r['rms_deg'] for r in rows if r['model'] == 'weld')
    for row in rows:
        row['rms_frac_of_weld'] = (float(row['rms_deg'] / weld_rms) if weld_rms > 0
                                   else float('nan'))

    keep = np.arange(0, len(score_idx), max(1, stride))
    samples = pd.DataFrame({
        **identity,
        'sample': score_idx[keep],
        'time_s': series.timestamps[score_idx][keep].astype(np.float32),
        'run': run_score[keep],
        'angle_deg': (angle_column[keep] * RAD2DEG).astype(np.float32),
        **{name: column[keep] for name, column in sample_columns.items()},
    })

    return JointResult(rows=rows, samples=samples,
                       curve=_error_curve(identity, angle_column, sample_columns),
                       coupling=_coupling_shape(
                           identity, {name: models[name] for name in curve_angles}, curve_angles),
                       models=models)


def _residual_rotvec(model: Dict[str, object], R: np.ndarray,
                     scored: Dict[str, np.ndarray]) -> np.ndarray:
    """The residual rotation vectors behind an already-computed error, for the anisotropy."""
    R_0 = np.asarray(model['R_0'], dtype=float)
    if model['kind'] == 'coupling':
        u_c = np.asarray(model['axis_child'])
        basis, derivative = _reuben_design(scored['theta1'],
                                           float(model['coupling_flexion_sign']))
        nu, _ = _curve_nu(scored['theta1'], u_c, _tangent_basis(u_c),
                          np.asarray(model['coeffs']), basis, derivative)
        return _log_matrix(_transpose_multiply(R_0 @ _exp_matrix(nu), R))
    v = np.asarray(model['axis_child']) if model['kind'] == 'universal' else None
    residual, _ = _model_residual(R, np.asarray(model['axis_parent']), R_0, scored['theta1'],
                                  v, scored['theta2'] if v is not None else None)
    return residual


def cross_validate(R_score: np.ndarray, fold_masks: List[np.ndarray],
                   knee_coupling: bool = False
                   ) -> Dict[str, Dict[str, float]]:
    """Out-of-fold error for EVERY model, refitting each one's structure on each training split.

    Without this the ladder is not a fair comparison. The coupling carries one more structure
    parameters than the hinge, so its in-sample residual is smaller by construction and the
    in-sample drop measures capacity rather than truth. Every model is refitted from scratch on
    each training split and scored on the held-out one, so the numbers are comparable across
    models of different complexity, and the report leads with them.

    One pass over the folds for the whole ladder, not one per model. The models share their
    seed and their optimizer, so fitting them together costs what fitting the most expensive
    one costs; doing it per model repeated the shared work four times over and turned a
    two-second joint into thirty.

    The spherical model is exempt: it has no structure to overfit and its error is identically
    zero on any split.
    """
    if not fold_masks:
        return {}
    collected: Dict[str, List[np.ndarray]] = {}
    for test in fold_masks:
        train_rows = np.flatnonzero(~test)
        test_rows = np.flatnonzero(test)
        if len(train_rows) < MIN_FIT_SAMPLES or len(test_rows) < 10:
            continue
        R_train = np.ascontiguousarray(R_score[fit_index(train_rows)])
        R_test = np.ascontiguousarray(R_score[test_rows])
        try:
            fitted = fit_all_models(R_train, pca_initialization(R_train))
        except (ValueError, np.linalg.LinAlgError):
            continue
        for name, model in fitted.items():
            if name == 'spherical':
                continue
            collected.setdefault(name, []).append(score_model(model, R_test)['error_deg'])

        # The published coupling is cross-validated too, and it has to be: it appears in the
        # same table as the fitted models, and a model with no CV number simply vanishes from
        # the cross-validated figure rather than showing up as the worst bar. Its CURVE is not
        # refitted per fold — that is what makes it the published model — but the five structure
        # parameters it does fit are, exactly as the hinge's are.
        if knee_coupling:
            coupling = fit_knee_coupling(fitted['hinge'], R_train)
            collected.setdefault('knee_coupling', []).append(
                score_model(coupling, R_test)['error_deg'])

    return {name: {'cv_folds_used': len(errors),
                   **error_stats(np.concatenate(errors), prefix='cv_')}
            for name, errors in collected.items() if errors}


def _error_curve(identity: Dict[str, str], angle: np.ndarray,
                 errors: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Each model's error binned by the joint's 1-DOF angle.

    A hinge that is fine through mid-range and fails past 60 deg of flexion is a different
    finding from one that is uniformly mediocre, and it is the one that decides whether a
    hinge constraint is safe for walking but not for stair descent. Binned on the HINGE angle
    for every model, so the bins mean the same thing across the row.
    """
    if len(angle) == 0:
        return pd.DataFrame()
    degrees = angle * RAD2DEG
    lo, hi = np.percentile(degrees, (0.5, 99.5))
    if hi - lo < 1e-6:
        return pd.DataFrame()
    edges = np.linspace(lo, hi, ANGLE_BINS + 1)
    which = np.clip(np.digitize(degrees, edges) - 1, 0, ANGLE_BINS - 1)
    rows = []
    for b in range(ANGLE_BINS):
        rows_in_bin = which == b
        if rows_in_bin.sum() < 5:
            continue
        row = {**identity, 'bin': b,
               'angle_lo_deg': float(edges[b]), 'angle_hi_deg': float(edges[b + 1]),
               'angle_mid_deg': float(0.5 * (edges[b] + edges[b + 1])),
               'n': int(rows_in_bin.sum())}
        for name, error in errors.items():
            model = name.replace('error_', '').replace('_deg', '')
            row[f'{model}_rms_deg'] = float(np.sqrt(np.mean(error[rows_in_bin] ** 2)))
            row[f'{model}_p95_deg'] = float(np.quantile(error[rows_in_bin], 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def _coupling_shape(identity: Dict[str, str], models: Dict[str, Dict[str, object]],
                  angles: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Each 1-DOF nonlinear curve's departure from the best geodesic through it.

    LONG FORM, one block per curve, keyed by `model`. Both 1-DOF nonlinear models land here —
    only the published Reuben coupling now that the free curve has been removed, but kept in
    long form keyed by `model` so the figure and the table do not have to change if another
    published coupling is ever added beside it.

    Off-axis rotation in degrees against joint angle in degrees, which is the form a published
    coupling is quoted in. It is the departure from the best HINGE rather than from the model's
    own neutral, because only the former is a property of the joint — see
    `coupling_geodesic_departure` for the measurement that forced that — and because it is the
    right comparison anyway: the linear part of any coupling is a tilted hinge, which a
    free-axis hinge already gets for nothing.

    The two channels are anatomical only when the source is, which means the biplane rows and
    not the marker ones.
    """
    blocks = []
    for name, model in models.items():
        angle = angles.get(name)
        if angle is None or len(angle) == 0:
            continue
        departure = coupling_geodesic_departure(model, angle)
        if departure is None:
            continue
        blocks.append(pd.DataFrame({
            **identity, 'model': name,
            # Re-centred on the trial's own median for the same reason every other angle axis
            # is; `coupling_geodesic_departure` returns the model's raw gauge.
            'angle_deg': departure['angle_deg'] - float(np.median(departure['angle_deg'])),
            'coupling1_deg': departure['perp_deg'][:, 0],
            'coupling2_deg': departure['perp_deg'][:, 1],
            'coupling_magnitude_deg': departure['magnitude_deg'],
            'along_axis_deg': departure['along_deg'],
        }))
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def knee_coupling_curve(flexion_rad: np.ndarray) -> np.ndarray:
    """(adduction, internal rotation) in RADIANS from the Reuben polynomials.

    The coupling a shipped method enforces every sample, transcribed as published. Its shape
    is fixed here; only where it sits on the joint is fitted.
    """
    x = flexion_rad * RAD2DEG
    powers = np.stack([x, x ** 2, x ** 3, x ** 4], axis=-1)
    return np.stack([powers @ REUBEN_ADDUCTION_COEFFS,
                     powers @ REUBEN_ROTATION_COEFFS], axis=-1) * DEG2RAD


def fit_knee_coupling(hinge: Dict[str, object], R: np.ndarray) -> Dict[str, object]:
    r"""The Reuben knee coupling, fitted to one joint's reference rotations.

    THIS IS THE 1-DOF NONLINEAR JOINT the file exists to test — Reuben et al. (1986), the
    coupling IMoveLab enforces every sample at gain 0.9, which at 100 Hz is replacement rather
    than regularization. It carries one angle per sample, exactly as the hinge does, and
    replaces the two non-sagittal channels with a fixed polynomial of that angle rather than
    with zero. So the HINGE is the model to read it against: same per-sample freedom, one
    pinning the off-axis channels to zero and the other to a published curve, and the drop
    between them is what Reuben's curvature bought.

    WHAT CURVATURE CAN AND CANNOT BUY, because the drop is small and the reason is structural
    rather than a defect. A coupling LINEAR in the joint angle is not a new kind of joint — it
    is a hinge about a TILTED axis. Reuben's rotation channel runs at 0.3695 deg per deg, i.e.
    a hinge tilted arctan(0.3695) ~ 20 deg off medio-lateral, and the hinge's axis is already
    free, so it absorbs that entire linear part for nothing. Both models here fit u freely, so
    what separates them is only the CURVATURE of the polynomials over the flexion range the
    trial actually visited. That is the whole budget the coupling has.

    FITTED AT ITS BEST, deliberately, so the residual bounds the published model rather than
    describing one calibration of it. Five continuous parameters are fitted — the axis (2), the
    neutral pose (3) — plus the channel orientation (1), and three discrete choices are
    searched:

      * sigma, the sense of flexion. IMoveLab negates it before evaluating the polynomials, and
        a joint axis is undirected, so which end of the range is extension is not known here.
      * the handedness of the plate frame in the plane perpendicular to the axis.
      * phi, WHICH DIRECTION IN THAT PLANE IS ADDUCTION. Not optional and not absorbed by R_0:
        the polynomials deliver adduction and internal rotation in ANATOMICAL axes while
        `_tangent_basis` returns an arbitrary basis of the plane, and rotating the channel frame
        conjugates exp(nu), which leaves a trailing exp(phi u_c) in the CHILD frame — the
        plate's frame, not a free one. Dropping phi scored the published curve in whatever
        orientation the basis happened to return: 23.2 deg against 15.1 on Subject01's right
        knee, larger than every effect this module reports.

    WHERE ANATOMICAL EXTENSION SITS is carried by R_0 and is deliberately NOT a separate
    parameter. An earlier version fitted one and it was an exact duplicate: rotating R_0 about
    u_c shifts every solved q by the same amount, because u_c commutes with itself, so
    (R_0 -> R_0 exp(delta u_c), offset -> offset + sigma delta) leaves the model pointwise
    identical. That one-dimensional null direction behaved as null directions do — the optimizer
    slid along it until the offset hit its bound and stalled there, 12.75 deg against a 12.75
    deg bound, R_0 175 deg from its true value, and a synthetic knee obeying the coupling
    exactly scored 0.83 deg instead of 0.

    What is NOT fitted is the CURVE, which is the point: its shape is published, so the residual
    is a statement about Reuben rather than about a curve fitted to this knee.
    """
    hinge_R0 = np.asarray(hinge['R_0'], dtype=float)
    u_child_init = hinge_R0.T @ np.asarray(hinge['axis_parent'], dtype=float)
    basis_u = _tangent_basis(u_child_init)
    basis_P = _tangent_basis(u_child_init)
    # The hinge's own joint angle, which is the gauge every seed below is computed in.
    q_hinge = (_log_matrix(_transpose_multiply(
        np.broadcast_to(hinge_R0, (len(R), 3, 3)), R)) @ u_child_init)

    best = None
    for sigma in (1.0, -1.0):
        # SEED R_0 AT ANATOMICAL FULL EXTENSION, which is the seed that actually decides whether
        # this fit succeeds. The hinge centres its angle wherever the solver left it, so its R_0
        # is the pose mid-range; the coupling needs R_0 at ZERO FLEXION, because q is fed
        # straight into polynomials that are only defined for q >= 0. On a synthetic knee the
        # two differed by 88.8 deg — nearly the whole flexion range — and the optimizer, started
        # at the hinge's R_0, converged (nfev 31, converged=True) to a local minimum at 0.17 deg
        # in one frame and 0.77 in a remounted copy of the SAME joint, where the answer is 0.
        #
        # Rotating R_0 about u_c shifts every q by the same amount, since u_c commutes with
        # itself, so the shift that puts the extreme of the range at zero is available exactly.
        # It is a SEED, not a constraint: R_0 stays fully free, and this only decides where the
        # search starts. Taken at a robust percentile rather than the raw extreme so one dropped
        # marker frame cannot set it.
        extension = float(np.percentile(q_hinge, ROM_PERCENTILES[0] if sigma > 0
                                        else ROM_PERCENTILES[1]))
        R0_init = hinge_R0 @ _rodrigues(u_child_init, np.array([extension]))[0]

        # phi's seed, computed in that shifted gauge so the published channels are evaluated at
        # the flexion the model will actually see. phi is periodic and nonconvex, and starting it
        # at zero made the fit depend on how the arbitrary perpendicular basis happened to fall.
        # The seed is the orthogonal Procrustes map between the published channels and the
        # observed off-axis displacement — one 2x2 SVD.
        nu_observed = _log_matrix(_transpose_multiply(
            np.broadcast_to(R0_init, (len(R), 3, 3)), R))
        channels = _reuben_design(nu_observed @ u_child_init, sigma)[0]
        u_svd, _, vt = np.linalg.svd((nu_observed @ basis_P).T @ channels)
        for handedness in (1.0, -1.0):
            # Forced to the requested determinant, so both plate chiralities get a seed of
            # their own rather than both starting from whichever one the SVD preferred.
            aligned = u_svd @ np.diag([1.0, handedness * np.linalg.det(u_svd @ vt)]) @ vt
            rotation = aligned @ np.diag([1.0, handedness])
            phi_init = float(np.arctan2(rotation[1, 0], rotation[0, 0]))

            def build_coeffs(phi: float, handedness=handedness) -> np.ndarray:
                """The O(2) map from published channels into the perpendicular plane's basis."""
                cos, sin = np.cos(phi), np.sin(phi)
                return np.array([[cos, -sin], [sin, cos]]) @ np.diag([1.0, handedness])

            # THE BRANCH THE ANGLE IS ALLOWED TO LIVE ON. Below zero flexion the polynomials
            # are clipped to a constant, which makes the model EXACTLY A HINGE there — and a
            # hinge is periodic in q, so on those samples q is only determined modulo 2 pi. Left
            # to a bracket derived from the wrapped logarithm, the seed scan then picked whichever
            # branch scored better PER SAMPLE, which is a silent per-sample choice between Reuben
            # and a plain hinge rather than the published model. Measured on Al Borno S02's right
            # knee it put 56% of samples a full turn out, over a 467 deg span, and across all
            # three datasets it hit ~25% of knee fits and roughly doubled their apparent gain.
            #
            # So the scan is confined to one physical window: from a little below the seeded
            # extension to a little past the range the hinge swept. Anatomical zero is at 0 by
            # construction of the seed, so this is a statement about knees, not about the fit.
            span = (-COUPLING_FLEXION_MARGIN_RAD,
                    float(np.ptp(np.percentile(q_hinge, ROM_PERCENTILES)))
                    + COUPLING_FLEXION_MARGIN_RAD)

            def residuals(params: np.ndarray, sigma=sigma, R0_init=R0_init,
                          span=span) -> np.ndarray:
                u_c = u_child_init + basis_u @ params[0:2]
                u_c = u_c / np.linalg.norm(u_c)
                R_0 = R0_init @ Rotation.from_rotvec(params[2:5]).as_matrix()
                # Seeded by the scan, whose bracket is derived from the data in this model's own
                # angle gauge. Handing it the hinge's angles instead put every seed in the wrong
                # gauge and a synthetic knee the model represents exactly fitted to 0.77 deg
                # rather than 0.
                _, res = solve_curve_angles(R, u_c, R_0, sigma,
                                            build_coeffs(float(params[5])), span=span)
                return res.ravel()

            solution = least_squares(residuals, np.r_[np.zeros(5), phi_init], method='trf',
                                     xtol=1e-8, ftol=1e-8, max_nfev=MAX_CURVE_NFEV)
            u_c = u_child_init + basis_u @ solution.x[0:2]
            u_c = u_c / np.linalg.norm(u_c)
            R_0 = R0_init @ Rotation.from_rotvec(solution.x[2:5]).as_matrix()
            model = {
                'kind': 'coupling', 'axis_child': u_c, 'axis_parent': R_0 @ u_c, 'R_0': R_0,
                'coeffs': build_coeffs(float(solution.x[5])),
                # SIX structure parameters, not six plus the curve. One more than the hinge's
                # five, and that one — which way the channels point — is a per-subject
                # calibration any real implementation has to establish.
                'n_structure_params': 6,
                'converged': bool(solution.status > 0), 'nfev': int(solution.nfev),
                'coupling_flexion_sign': sigma,
                'coupling_handedness': handedness,
                'coupling_channel_angle_deg': float(solution.x[5]) * RAD2DEG,
                # Carried so held-out scoring resolves q on the SAME branch the fit used. Without
                # it a fold could legitimately re-solve onto a different turn and be scored under
                # a different model than the one being cross-validated.
                'q_span': span,
            }
            scored = score_model(model, R)
            rms = float(np.sqrt(np.mean(scored['error_deg'] ** 2)))
            # How far below anatomical extension the model ran, as a DIAGNOSTIC rather than a
            # parameter. The polynomials are defined only for non-negative flexion and are
            # clipped below it, so a strongly negative value means part of the trial was scored
            # against a constant — the residual is then optimistic about a model that was not
            # being used.
            model['coupling_min_flexion_deg'] = float(
                np.percentile(sigma * scored['theta1'], ROM_PERCENTILES[0]) * RAD2DEG)
            if best is None or rms < best[0]:
                best = (rms, model)
    return best[1]



# ==============================================================================
# MARKERS VERSUS BIPLANE
# ==============================================================================

def reference_agreement(plates: Dict[str, PlateTrial], spec: DatasetSpec,
                        cutoff_hz: float = LOWPASS_HZ) -> pd.DataFrame:
    """How far apart the two reference systems are about the SAME joint over the SAME frames.

    This is the number every model residual in this file has to be read against, and it exists
    only on the biplane dataset, where a knee is measured twice: by Vicon marker clusters
    taped to the skin, and by biplane fluoroscopy solving the bone poses directly.

    The two disagree by a CONSTANT frame change plus whatever moves. A marker cluster's frame
    is wherever the template landed on the skin and a bone frame is anatomy, so
    R_pc_biplane = Q_p^T R_pc_vicon Q_c holds exactly if the skin does not move relative to the
    bone. Q_p and Q_c are fitted — six parameters over hundreds of samples — and what is left
    is soft-tissue artifact plus marker error plus whatever the fluoroscopy got wrong.

    Reported as `agreement_rms_deg`. A marker-based hinge residual smaller than this is not
    measuring the joint; it is measuring the markers.
    """
    rows = []
    pairs: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for joint_key, sensors in spec.joints.items():
        joint, source = split_source(joint_key)
        pairs.setdefault(joint, {})[source] = sensors

    for joint, by_source in sorted(pairs.items()):
        if len(by_source) < 2:
            continue
        for reference, comparison in ((a, b) for a in by_source for b in by_source if a < b):
            series = {}
            for source in (reference, comparison):
                parent, child = by_source[source]
                if parent not in plates or child not in plates:
                    break
                built = joint_series(plates[parent], plates[child], cutoff_hz)
                if built is None:
                    break
                series[source] = built
            if len(series) < 2:
                continue

            # Compared frame for frame. Both series are indexed by the ORIGINAL sample number
            # through their valid runs, so they are intersected on those rather than assumed
            # to be the same length — the biplane window is a tenth of the Vicon one.
            a, b = series[reference], series[comparison]
            common, index_a, index_b = np.intersect1d(
                np.round(a.timestamps, 6), np.round(b.timestamps, 6), return_indices=True)
            if len(common) < MIN_AGREEMENT_SAMPLES:
                rows.append({'joint': joint, 'reference': reference, 'comparison': comparison,
                             'n_common': int(len(common)), 'agreement_rms_deg': float('nan')})
                continue

            R_a = np.ascontiguousarray(a.R[index_a])
            R_b = np.ascontiguousarray(b.R[index_b])
            fitted = _fit_constant_frames(R_a, R_b)
            rows.append({
                'joint': joint, 'reference': reference, 'comparison': comparison,
                'n_common': int(len(common)),
                'duration_s': float(len(common) / a.fs),
                **error_stats(fitted['error_deg'], prefix='agreement_'),
                'frame_parent_deg': float(np.degrees(np.linalg.norm(fitted['rotvec_parent']))),
                'frame_child_deg': float(np.degrees(np.linalg.norm(fitted['rotvec_child']))),
                # Before any frame change is fitted — the raw disagreement, which is dominated
                # by the constant and is reported so the fitted number is visibly a fit.
                'agreement_uncorrected_rms_deg': float(np.sqrt(np.mean(
                    (np.linalg.norm(_log_matrix(_transpose_multiply(R_a, R_b)), axis=1)
                     * RAD2DEG) ** 2))),
                'converged': fitted['converged'],
            })
    return pd.DataFrame(rows)


def _fit_constant_frames(R_a: np.ndarray, R_b: np.ndarray) -> Dict[str, object]:
    """Fits Q_p, Q_c minimizing || log( (Q_p^T R_a Q_c)^T R_b ) ||, six parameters."""
    def residuals(params: np.ndarray) -> np.ndarray:
        Q_p = Rotation.from_rotvec(params[0:3]).as_matrix()
        Q_c = Rotation.from_rotvec(params[3:6]).as_matrix()
        model = np.einsum('ji,tjk,kl->til', Q_p, R_a, Q_c)
        return _log_matrix(_transpose_multiply(model, R_b)).ravel()

    solution = least_squares(residuals, np.zeros(6), method='trf', xtol=1e-10, ftol=1e-10,
                             max_nfev=MAX_OUTER_NFEV)
    error = np.linalg.norm(solution.fun.reshape(-1, 3), axis=1) * RAD2DEG
    return {'error_deg': error, 'rotvec_parent': solution.x[0:3],
            'rotvec_child': solution.x[3:6], 'converged': bool(solution.status > 0)}


# ==============================================================================
# PER-TRIAL DRIVER
# ==============================================================================

def compute_trial(plates: Dict[str, PlateTrial], spec: DatasetSpec,
                  tables: Sequence[str] = TRIAL_TABLES, folds: int = CV_FOLDS,
                  stride: int = SAMPLE_STRIDE, score_cap: int = SCORE_MAX_SAMPLES,
                  cutoff_hz: float = LOWPASS_HZ, joints: Optional[Sequence[str]] = None,
                  placements: Optional[Sequence[str]] = None,
                  knee_coupling: bool = True) -> Dict[str, pd.DataFrame]:
    """Every table for one trial. Joints whose plates are absent are skipped, not failed."""
    wanted = set(tables)
    fits, samples, curves, shapes = [], [], [], []

    for joint_key, identity in selected_joints(spec, joints, placements).items():
        parent_sensor, child_sensor = spec.joints[joint_key]
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        try:
            series = joint_series(plates[parent_sensor], plates[child_sensor], cutoff_hz)
        except ValueError:
            # A plate carrying a discontinuity WorldTrace could not repair honestly costs its
            # own joint, not the trial. Subject08's calcn_l is the standing example.
            continue
        if series is None:
            continue
        result = analyze_joint(series, identity, folds, stride, score_cap, knee_coupling)
        if result is None:
            continue
        fits.extend(result.rows)
        samples.append(result.samples)
        curves.append(result.curve)
        shapes.append(result.coupling)

    out: Dict[str, pd.DataFrame] = {}
    if 'joint_fits' in wanted:
        out['joint_fits'] = pd.DataFrame(fits)
    if 'error_samples' in wanted and samples:
        out['error_samples'] = pd.concat(samples, ignore_index=True)
    if 'error_curve' in wanted and curves:
        out['error_curve'] = pd.concat([c for c in curves if not c.empty], ignore_index=True) \
            if any(not c.empty for c in curves) else pd.DataFrame()
    if 'coupling_shape' in wanted and shapes:
        out['coupling_shape'] = pd.concat([f for f in shapes if not f.empty], ignore_index=True) \
            if any(not f.empty for f in shapes) else pd.DataFrame()
    if 'reference_agreement' in wanted and has_multiple_sources(spec):
        out['reference_agreement'] = reference_agreement(plates, spec, cutoff_hz)
    return out


def _load_spec_plates(subject: str, trial: str, dataset: str,
                      spec: DatasetSpec) -> Dict[str, PlateTrial]:
    plates = load_trial(subject, trial, dataset=dataset)
    wanted = set(spec.segment_sensor.values())
    selected = {sensor: plate for sensor, plate in plates.items() if sensor in wanted}
    if not selected:
        raise ValueError(f"{dataset}/{subject}/{trial}: none of the spec's sensors are present "
                         f"(trial has {sorted(plates)}).")
    return selected


def _trial_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                  dataset: str = 'alborno', tables: Sequence[str] = TRIAL_TABLES,
                  folds: int = CV_FOLDS, stride: int = SAMPLE_STRIDE,
                  score_cap: int = SCORE_MAX_SAMPLES,
                  cutoff_hz: float = LOWPASS_HZ,
                  joints: Optional[Sequence[str]] = None,
                  placements: Optional[Sequence[str]] = None,
                  knee_coupling: bool = True) -> None:
    subject, trial = row_key
    spec = get_dataset(dataset)
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()
    try:
        plates = _load_spec_plates(subject, trial, dataset, spec)
        produced = compute_trial(plates, spec, tables, folds, stride, score_cap,
                                 cutoff_hz, joints, placements, knee_coupling)
        for table, frame in produced.items():
            if frame.empty:
                continue
            _save(frame, trial_table_path(dataset, subject, trial, table), dataset,
                  subject=subject, trial=trial, table=table)
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success" if produced else "Skipped (no joint)"
    except Exception as exc:                                    # noqa: BLE001 — reported, not swallowed
        shared_state[(row_key, stage)] = f"Failed ({type(exc).__name__}: {exc})"
    return None


# ==============================================================================
# POOLED FITS
# ==============================================================================

def pooled_group(dataset: str, subject: str, trials: Sequence[str]) -> str:
    """The unit a pooled fit is legitimate over.

    A pooled fit assumes the sensor mounting and the reference frames did not move between the
    trials it pools, so the grouping has to follow the SESSION, not the subject. Al Borno and
    IMoVE mocap put one session per subject. The biplane trials name theirs — '12/Test1/B/...'
    — and blocks within one Test are one capture, so the group is the subject and the test.
    """
    if dataset != BIPLANE.name:
        return subject
    session = trials[0].split('/')[0] if trials else ''
    return f"{subject}/{session}"


def group_trials(dataset: str, row_keys: Sequence[Tuple[str, str]]
                 ) -> Dict[str, List[Tuple[str, str]]]:
    """{session: [(subject, trial), ...]} — the units a pooled fit may span."""
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for subject, trial in row_keys:
        groups.setdefault(pooled_group(dataset, subject, [trial]), []).append((subject, trial))
    return {group: sorted(members) for group, members in sorted(groups.items())}


def group_series(dataset: str, spec: DatasetSpec, members: Sequence[Tuple[str, str]],
                 cutoff_hz: float = LOWPASS_HZ, joints: Optional[Sequence[str]] = None,
                 placements: Optional[Sequence[str]] = None,
                 ) -> Dict[str, Tuple[JointSeries, int]]:
    """{joint key: (series concatenated over the session, how many trials contributed)}."""
    per_joint: Dict[str, List[JointSeries]] = {}
    for subject, trial in members:
        try:
            plates = _load_spec_plates(subject, trial, dataset, spec)
        except Exception:                                       # noqa: BLE001
            continue
        for joint_key in selected_joints(spec, joints, placements):
            parent_sensor, child_sensor = spec.joints[joint_key]
            if parent_sensor not in plates or child_sensor not in plates:
                continue
            try:
                series = joint_series(plates[parent_sensor], plates[child_sensor], cutoff_hz)
            except ValueError:
                continue
            if series is not None:
                per_joint.setdefault(joint_key, []).append(series)
    return {joint_key: (concatenate_series(parts), len(parts))
            for joint_key, parts in per_joint.items()}


def pooled_group_fit(group: str, members: Sequence[Tuple[str, str]], dataset: str,
                     spec: DatasetSpec, folds: int, score_cap: int,
                     cutoff_hz: float, joints: Optional[Sequence[str]],
                     placements: Optional[Sequence[str]] = None,
                     knee_coupling: bool = True) -> List[Dict[str, object]]:
    """Every model, fitted once over one whole session. See `pooled_fits`."""
    rows: List[Dict[str, object]] = []
    for joint_key, (merged, n_trials) in group_series(dataset, spec, members, cutoff_hz,
                                                      joints, placements).items():
        result = analyze_joint(merged, joint_identity(joint_key, spec), folds, SAMPLE_STRIDE,
                               score_cap, knee_coupling)
        if result is None:
            continue
        for row in result.rows:
            rows.append({**row, 'group': group, 'subject': members[0][0],
                         'n_trials': n_trials})
    return rows


def _pooled_worker(row_key: str, stage_labels: List[str], shared_state: Dict,
                   members_by_group: Dict[str, List[Tuple[str, str]]] = None,
                   dataset: str = 'alborno', folds: int = CV_FOLDS,
                   score_cap: int = SCORE_MAX_SAMPLES,
                   cutoff_hz: float = LOWPASS_HZ,
                   joints: Optional[Sequence[str]] = None,
                   placements: Optional[Sequence[str]] = None,
                   knee_coupling: bool = True) -> List[Dict[str, object]]:
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()
    try:
        rows = pooled_group_fit(row_key, members_by_group[row_key], dataset,
                                get_dataset(dataset), folds, score_cap,
                                cutoff_hz, joints, placements, knee_coupling)
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success" if rows else "Skipped (no joint)"
        return rows
    except Exception as exc:                                    # noqa: BLE001
        shared_state[(row_key, stage)] = f"Failed ({type(exc).__name__}: {exc})"
        return []


def pooled_fits(dataset: str, spec: DatasetSpec, row_keys: Sequence[Tuple[str, str]],
                folds: int = CV_FOLDS, score_cap: int = SCORE_MAX_SAMPLES,
                cutoff_hz: float = LOWPASS_HZ,
                joints: Optional[Sequence[str]] = None,
                placements: Optional[Sequence[str]] = None,
                workers: int = 1, knee_coupling: bool = True) -> pd.DataFrame:
    """One fit per (session, joint, source) over ALL that session's trials at once.

    NECESSARY, not merely tidier, for the biplane half. A biplane trial is 0.4-0.6 s of a
    single hop or drop landing: seven structure parameters are numerically determined by it and
    kinematically determined by nothing, because the knee never left a narrow band of flexion.
    Pooling a session's twenty-odd trials is what produces a sweep wide enough to identify an
    axis, and it is legitimate because the BioStamps and the marker clusters stay on the
    subject for the whole session and the bone frames are anatomy.

    It is also the honest way to fit the marker datasets, where the per-trial answers are
    already good and the question becomes whether ONE model serves the whole session. The
    per-trial fits stay on disk beside these, so the pooling assumption is checkable: if a
    pooled residual is far above the per-trial ones, the structure moved between trials — and
    `report_pooling` puts the two side by side for exactly that reason.
    """
    groups = group_trials(dataset, row_keys)
    if not groups:
        return pd.DataFrame()
    _, results = run_tracked_grid(
        list(groups), ['Session'], ['pool'],
        partial(_pooled_worker, members_by_group=groups, dataset=dataset, folds=folds,
                score_cap=score_cap, cutoff_hz=cutoff_hz,
                joints=joints, placements=placements, knee_coupling=knee_coupling),
        max(1, workers), title=f"JOINT DOF (pooled per session) — {dataset}")
    rows = [row for group_rows in results.values() for row in (group_rows or [])]
    return pd.DataFrame(rows)


# ==============================================================================
# ONE AXIS FOR EVERYONE
# ==============================================================================

def generic_axes(dataset: str, spec: DatasetSpec, pooled: pd.DataFrame,
                 row_keys: Sequence[Tuple[str, str]], cutoff_hz: float = LOWPASS_HZ,
                 score_cap: int = SCORE_MAX_SAMPLES,
                 joints: Optional[Sequence[str]] = None,
                 placements: Optional[Sequence[str]] = None,
                 progress: bool = True) -> pd.DataFrame:
    """What a SINGLE published axis per joint costs, leave-one-subject-out.

    For each subject the axis is the axial mean over the OTHER subjects, so the number is a
    genuine generalization error rather than a fit re-scored on its own data. Only R_0 is
    refitted per subject: the axis is the transferable claim, whereas the neutral relative pose
    differs per subject through both anatomy and mounting and any real method calibrates it.

    READ THE DISPERSION FIRST. On the marker datasets these axes live in each subject's own
    plate frame and the plates are re-strapped per subject, so a single proposed axis is only
    meaningful if the mounting is repeatable — which is exactly what the dispersion column
    measures. On the biplane source the frames are bone-fixed and the axis is anatomical, so
    the same table means something much stronger there.
    """
    if pooled.empty:
        return pd.DataFrame()

    hinges = pooled[(pooled['model'] == 'hinge') & pooled['u_x'].notna()]
    if hinges.empty:
        return pd.DataFrame()

    by_subject: Dict[str, List[str]] = {}
    for subject, trial in row_keys:
        by_subject.setdefault(subject, []).append(trial)

    records = []
    for (joint, source), group in hinges.groupby(['joint', 'source'], observed=True):
        if joints and joint not in set(joints):
            continue
        if len(group) < 3:
            continue
        axes = group[[f'u_{c}' for c in AXIS_COMPONENTS]].to_numpy(dtype=float)
        cohort_axis, cohort_angles = axial_mean(axes)
        for position, (_, row) in enumerate(group.iterrows()):
            others = np.delete(axes, position, axis=0)
            loo_axis, _ = axial_mean(others)
            records.append({
                'dataset': dataset, 'joint': joint, 'source': source,
                'group': row['group'], 'subject': row['subject'],
                'own_rms_deg': float(row['rms_deg']),
                'own_cv_rms_deg': float(row.get('cv_rms_deg', np.nan)),
                'axis_from_cohort_deg': float(cohort_angles[position]),
                'loo_axis_x': loo_axis[0], 'loo_axis_y': loo_axis[1], 'loo_axis_z': loo_axis[2],
                'cohort_axis_x': cohort_axis[0], 'cohort_axis_y': cohort_axis[1],
                'cohort_axis_z': cohort_axis[2],
                'cohort_dispersion_median_deg': float(np.median(cohort_angles)),
                'cohort_dispersion_p90_deg': float(np.percentile(cohort_angles, 90)),
                'n_cohort': len(group),
            })

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)

    # Re-scoring each leave-one-out axis on that session's own data needs the data back, so it
    # is done session by session with one load pass rather than one per row.
    all_groups = group_trials(dataset, row_keys)
    generic_rms: Dict[Tuple[str, str, str], float] = {}
    for group_name, group_rows in frame.groupby('group'):
        wanted = {(row.joint, row.source) for row in group_rows.itertuples()}
        for joint_key, (merged, _) in group_series(dataset, spec, all_groups[group_name],
                                                   cutoff_hz, joints, placements).items():
            identity = joint_identity(joint_key, spec)
            if (identity['joint'], identity['source']) not in wanted:
                continue
            match = group_rows[(group_rows['joint'] == identity['joint'])
                               & (group_rows['source'] == identity['source'])]
            if match.empty:
                continue

            R = np.ascontiguousarray(merged.R[scoring_index(len(merged.R), score_cap)])
            axis = match.iloc[0][['loo_axis_x', 'loo_axis_y',
                                  'loo_axis_z']].to_numpy(dtype=float)
            init = pca_initialization(R)
            # Seeded by projecting the displacement from neutral onto the proposed axis, pulled
            # back into the frame the displacement lives in. u is a parent-frame axis; the
            # displacement is in the neutral child frame.
            theta_init = init['scores'] @ (init['eigenvectors'].T @ init['ref_matrix'].T @ axis)
            inner = fit_index(np.arange(len(R)))
            fitted = fit_chain(np.ascontiguousarray(R[inner]), 1, axis, init['ref_matrix'],
                               theta_init[inner].copy(), free_axes=False)
            generic_rms[(group_name, identity['joint'], identity['source'])] = float(
                np.sqrt(np.mean(score_model(fitted, R)['error_deg'] ** 2)))
        if progress:
            print(f"  generic-axis: {group_name} done", flush=True)

    frame['generic_rms_deg'] = [
        generic_rms.get((row.group, row.joint, row.source), np.nan)
        for row in frame.itertuples()]
    frame['generic_penalty_deg'] = frame['generic_rms_deg'] - frame['own_rms_deg']
    return frame


# ==============================================================================
# SUMMARY
# ==============================================================================

def summarize(dataset: str, fits: pd.DataFrame) -> pd.DataFrame:
    """Cohort quantiles per (joint, source, model), on the standard spine."""
    if fits.empty:
        return pd.DataFrame()
    metrics = [c for c in ('rms_deg', 'p95_deg', 'cv_rms_deg', 'cv_p95_deg', 'raw_rms_deg',
                           'rms_frac_of_weld', 'max_deg') if c in fits.columns]
    rows = []
    for (joint, source, model), group in fits.groupby(['joint', 'source', 'model'],
                                                      observed=True):
        row: Dict[str, object] = {
            'dataset': dataset, 'joint': joint, 'source': source, 'model': model,
            'n_trials': int(len(group)),
            'n_subjects': int(group['subject'].nunique()) if 'subject' in group else 0,
            'total_samples': int(group['n_samples'].sum()),
        }
        for metric in metrics:
            values = group[metric].dropna()
            if values.empty:
                continue
            row[f'{metric}_median'] = float(values.median())
            row[f'{metric}_iqr'] = float(values.quantile(0.75) - values.quantile(0.25))
            row[f'{metric}_min'] = float(values.min())
            row[f'{metric}_max'] = float(values.max())
        for extra in ('carrying_angle_deg', 'curvature_max_deg', 'theta1_rom_deg',
                      'rom1_deg', 'excitation', 'hinge_deficiency', 'resid_frac1'):
            if extra in group.columns and group[extra].notna().any():
                row[f'{extra}_median'] = float(group[extra].median())
        if 'converged' in group.columns:
            row['converged_frac'] = float(group['converged'].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def n_trials(fits: pd.DataFrame) -> int:
    """Distinct (subject, trial) pairs.

    Not `fits['trial'].nunique()`. Al Borno names every subject's trials 'walking' and
    'complexTasks', so counting the name alone reported 19 trials as 2 — in the report header
    and in every figure's provenance stamp.
    """
    if 'trial' not in fits.columns:
        return 0
    keys = ['subject', 'trial'] if 'subject' in fits.columns else ['trial']
    return int(len(fits[keys].drop_duplicates()))


def model_ladder(fits: pd.DataFrame, metric: str = 'cv_rms_deg') -> pd.DataFrame:
    """Median error per model, wide by joint — the table the whole file exists to produce.

    Falls back to the in-sample metric when cross-validation could not run (too few samples,
    which on the biplane per-trial rows is most of them), and says which it used.
    """
    if fits.empty:
        return pd.DataFrame()
    column = metric if metric in fits.columns and fits[metric].notna().any() else 'rms_deg'
    table = fits.pivot_table(index=['joint', 'source'], columns='model', values=column,
                             aggfunc='median', observed=True)
    ordered = [m for m in (*MODELS, 'knee_coupling') if m in table.columns]
    table = table[ordered]
    table.attrs['metric'] = column
    return table


# ==============================================================================
# REPORT
# ==============================================================================

def _header(number: int, title: str, subtitle: str = "") -> None:
    print(f"\n{'=' * 100}")
    print(f"{number}. {title}")
    if subtitle:
        print(f"   {subtitle}")
    print('=' * 100)


def _fmt(value: object, width: int = 7, places: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ' ' * (width - 1) + '-'
    return f"{float(value):{width}.{places}f}"


def report_ladder(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(1, "THE LADDER — what each degree of freedom buys",
            "Geodesic error in degrees, median over trials. Models nest, so errors are "
            "monotone\n   in DOF and each drop is the value of the extra freedom.")
    table = model_ladder(fits)
    if table.empty:
        print("   No fits.")
        return
    metric = table.attrs.get('metric', 'rms_deg')
    print(f"\n   Metric: {metric} "
          f"({'cross-validated, out-of-fold' if metric.startswith('cv_') else 'IN-SAMPLE — '
             'cross-validation did not run, so the coupling is flattered by its extra structure'})\n")
    print(table.round(2).to_string())

    if 'hinge' in table.columns and 'knee_coupling' in table.columns:
        print("\n   What the published CURVATURE buys (hinge -> coupling, one DOF either way):")
        gain = (table['hinge'] - table['knee_coupling'])
        for key, value in gain.sort_values(ascending=False).items():
            joint, source = key
            share = value / table.loc[key, 'hinge'] if table.loc[key, 'hinge'] > 0 else np.nan
            print(f"      {joint:<10} {source:<9} {_fmt(value)} deg  ({share:5.1%} of the hinge "
                  f"residual)")
        print("   A coupling LINEAR in the joint angle is a tilted hinge and the hinge's axis is")
        print("   already free, so this column is curvature and nothing else.")

    if 'hinge' in table.columns and 'universal' in table.columns:
        print("\n   What the SECOND AXIS buys (hinge -> universal):")
        gain = (table['hinge'] - table['universal'])
        for key, value in gain.sort_values(ascending=False).items():
            joint, source = key
            share = value / table.loc[key, 'hinge'] if table.loc[key, 'hinge'] > 0 else np.nan
            print(f"      {joint:<10} {source:<9} {_fmt(value)} deg  ({share:5.1%})")


def report_context(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(2, "IS THE RESIDUAL SMALL, OR DID THE JOINT NOT MOVE?",
            "Every residual above divided by the 0-DOF (weld) residual, plus how far the "
            "joint\n   swept. A joint that barely moved is fitted well by everything.")
    hinge = fits[fits['model'] == 'hinge']
    if hinge.empty:
        print("   No hinge fits.")
        return
    columns = ['rms_frac_of_weld', 'rom1_deg', 'rom2_deg', 'rom3_deg', 'var_frac1',
               'excitation', 'hinge_deficiency']
    available = [c for c in columns if c in hinge.columns]
    table = hinge.groupby(['joint', 'source'], observed=True)[available].median()
    print()
    print(table.round(3).to_string())
    print("\n   excitation        0 when the pair never moved relative to each other. A residual")
    print("                     is meaningless without it: a joint that did not move is fitted")
    print("                     perfectly by every model on the ladder.")
    print("   hinge_deficiency  0 for a PURE hinge (and for a weld). Nonzero exactly when the")
    print("                     relative rotation has no common invariant direction. One SVD,")
    print("                     nothing fitted — an independent check on the hinge column above.")


def report_anisotropy(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(3, "WHAT THE HINGE MISSED — one channel, or all three?",
            "resid_frac1 is the share of the leftover on its own dominant direction. Near 1 "
            "means\n   a missing SECOND AXIS, which a 2-DOF model will fix; near 1/3 means "
            "noise or soft\n   tissue, which nothing structural will.")
    hinge = fits[fits['model'] == 'hinge']
    columns = [c for c in ('resid_frac1', 'resid_frac12', 'resid_principal_from_axis_deg',
                           'resid_along_axis_frac') if c in hinge.columns]
    if hinge.empty or not columns:
        print("   Not available.")
        return
    print()
    print(hinge.groupby(['joint', 'source'], observed=True)[columns].median().round(3).to_string())
    if 'resid_along_axis_frac' in columns:
        worst = float(hinge['resid_along_axis_frac'].max())
        print(f"\n   Largest share of residual left ALONG the fitted axis: {worst:.2e}. The "
              f"angle solve\n   drives exactly that component to zero, so this is a "
              f"convergence check, not a finding.")


def report_jitter(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(4, f"HOW MUCH OF THE RESIDUAL WAS MARKER JITTER",
            f"Each fitted model re-scored, without refitting, against the UNFILTERED "
            f"rotations.\n   The gap is what the {LOWPASS_HZ:g} Hz low-pass removed, and it is "
            f"the floor a residual on raw\n   data cannot go below however good the mechanism "
            f"is.")
    if 'raw_rms_deg' not in fits.columns:
        print("   Not available.")
        return
    rows = []
    for (joint, source, model), group in fits.groupby(['joint', 'source', 'model'],
                                                      observed=True):
        if model in ('spherical',):
            continue
        rows.append({'joint': joint, 'source': source, 'model': model,
                     'filtered_rms': group['rms_deg'].median(),
                     'raw_rms': group['raw_rms_deg'].median(),
                     'jitter_deg': float(np.sqrt(max(
                         group['raw_rms_deg'].median() ** 2 - group['rms_deg'].median() ** 2,
                         0.0)))})
    frame = pd.DataFrame(rows)
    print()
    print(frame.pivot_table(index=['joint', 'source'], columns='model',
                            values='jitter_deg', observed=True).round(2).to_string())
    print("\n   Read as: the jitter added in quadrature to the filtered residual. It is nearly")
    print("   model-independent by construction, and a column that is NOT nearly constant "
          "means\n   the low-pass removed something that model was using.")


def report_geometry(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(5, "THE FITTED 2-DOF GEOMETRY, and whether it reproduces",
            "Only MOUNTING-INVARIANT quantities are pooled. Remounting the parent sensor by a "
            "\n   constant Q sends u -> Q'u and R_0 -> Q'R_0, so the angle between u and R_0 v "
            "is\n   unchanged — but the axis VECTORS are in each subject's own frame and are "
            "not\n   comparable. They are deliberately absent from this table.")
    universal = fits[fits['model'] == 'universal']
    if universal.empty or 'carrying_angle_deg' not in universal.columns:
        print("   No 2-DOF fits.")
        return
    print(f"\n   {'joint':<10} {'source':<9} {'verdict':<17} {'identified':<12} "
          f"{'carrying angle':<34} why\n")
    for (joint, source), group in universal.groupby(['joint', 'source'], observed=True):
        identified = group[group.get('geometry_identified', pd.Series(True, index=group.index))]
        n_identified, n_total = len(identified), len(group)
        fraction = n_identified / n_total if n_total else 0.0
        angles = identified['carrying_angle_deg']
        iqr = (float(angles.quantile(0.75) - angles.quantile(0.25)) if n_identified >= 2
               else float('nan'))
        summary = (f"median {angles.median():5.1f}, IQR {iqr:4.1f}, "
                   f"range {angles.min():.0f}-{angles.max():.0f} deg" if n_identified >= 2
                   else "-")

        # The reason is always printed, never only on failure. A row reading
        # "NOT REPRODUCIBLE (19/19 identified)" with no further text invites the reading that
        # the identification count was the problem, when in that case it was the spread.
        if n_identified < 2:
            verdict, why = 'NOT REPRODUCIBLE', 'fewer than two subjects identified'
        elif fraction < MIN_IDENTIFIED_FRACTION:
            verdict = 'NOT REPRODUCIBLE'
            why = (f"only {fraction:.0%} of trials identified, under the "
                   f"{MIN_IDENTIFIED_FRACTION:.0%} bar")
        elif iqr > CARRYING_ANGLE_AGREEMENT_DEG:
            verdict = 'NOT REPRODUCIBLE'
            why = f"spread: IQR {iqr:.1f} deg over the {CARRYING_ANGLE_AGREEMENT_DEG:g} deg bar"
        else:
            verdict = 'reproducible'
            why = f"IQR {iqr:.1f} deg within the {CARRYING_ANGLE_AGREEMENT_DEG:g} deg bar"
        print(f"   {joint:<10} {source:<9} {verdict:<17} {n_identified:>4}/{n_total:<7} "
              f"{summary:<34} {why}")

    print(f"\n   'identified' means a carrying angle of at least "
          f"{GEOMETRY_IDENTIFIABILITY_MIN_CARRYING_DEG:g} deg. Below that the two axes are\n"
          f"   near parallel, the two angles have a near-null direction, and the geometry — "
          f"though\n   not the residual — is meaningless.")


def report_coupling(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(6, "THE 1-DOF NONLINEAR KNEE — does the published coupling hold?",
            "The Reuben coupling (Reuben et al. 1986, as IMoveLab enforces it every sample at "
            "\n   alpha=0.9) against the HINGE it has to beat: the same one-angle joint with the "
            "\n   off-axis channels pinned to zero instead of to a published curve. The 2-DOF "
            "column\n   is there for scale — it is what a second free axis buys on the same joint.")
    coupling = fits[fits['model'] == 'knee_coupling']
    if coupling.empty:
        print("   Not fitted (no knees in this selection, or --no-knee-coupling).")
        return

    metric = ('cv_rms_deg' if 'cv_rms_deg' in fits.columns
              and fits['cv_rms_deg'].notna().any() else 'rms_deg')
    by_model = fits.pivot_table(index=['joint', 'source'], columns='model', values=metric,
                                aggfunc='median', observed=True)
    print(f"\n   Error in degrees ({metric}), median over trials:\n")
    print(f"   {'joint':<9} {'source':<9} {'weld':>7} {'hinge':>7} {'coupling':>9} {'2-DOF':>7}"
          f"   {'verdict':<25} coupling vs hinge")
    for key in by_model.index:
        reuben = by_model.loc[key].get('knee_coupling', np.nan)
        if pd.isna(reuben):
            continue
        weld = by_model.loc[key].get('weld', np.nan)
        hinge = by_model.loc[key].get('hinge', np.nan)
        two_dof = by_model.loc[key].get('universal', np.nan)

        # A coupling WORSE than a plain hinge is not merely imperfect — it is the wrong
        # curvature, and enforcing it costs more than assuming no coupling at all.
        gain = hinge - reuben
        if not np.isfinite(gain):
            verdict = 'not comparable'
        elif gain < -COUPLING_MATERIAL_GAIN_DEG:
            verdict = 'WORSE THAN A PLAIN HINGE'
        elif gain < COUPLING_MATERIAL_GAIN_DEG:
            verdict = 'no better than a hinge'
        else:
            verdict = 'helps'
        share = (f"{gain:+.2f} deg ({gain / hinge:+.0%})" if np.isfinite(hinge) and hinge > 1e-9
                 else f"{gain:+.2f} deg")
        print(f"   {key[0]:<9} {key[1]:<9} {_fmt(weld)} {_fmt(hinge)} {_fmt(reuben, 9)} "
              f"{_fmt(two_dof)}   {verdict:<25} {share}")

    print(f"\n   'helps' means beating the hinge by more than {COUPLING_MATERIAL_GAIN_DEG:g} deg. "
          f"Read the gain against the WELD\n   column, not against zero: a joint that barely "
          f"moved is fitted well by everything.")
    print("\n   TWO THINGS A GAIN HERE DOES NOT PROVE.")
    print("   (1) That the knee obeys REUBEN. Over a limited flexion range the published "
          "quartics are\n       not far from a generic quadratic, and with the axis, neutral pose "
          "and channel\n       direction all fitted they absorb a good part of a coupling they do "
          "not describe. On a\n       synthetic joint curved the WRONG way Reuben still took the "
          "hinge's 1.46 deg down to\n       0.47; on one that truly obeys it, it goes to 0. Only "
          "the second is a match.")
    print("   (2) That no OTHER 1-DOF curve would have done better. That needs a free curve as an"
          "\n       envelope, which this module does not fit, so a coupling that fails to beat the"
          "\n       hinge cannot be told apart from a knee with no usable 1-DOF curvature at all.")
    print("\n   HOW CURVED THE PUBLISHED CURVE IS over the flexion each trial actually visited —")
    print("   distance from the best GEODESIC through it, which is the only gauge-invariant way")
    print("   to state it and the right one anyway: a coupling LINEAR in the joint angle is a")
    print("   tilted hinge, and the hinge's axis is already free, so only the curvature below")
    print("   was ever worth extra structure.\n")
    columns = [c for c in ('curvature_max_deg', 'curvature_rms_deg', 'curvature_range_deg',
                           'rms_deg') if c in coupling.columns]
    if columns:
        print(coupling.groupby(['joint', 'source'],
                               observed=True)[columns].median().round(2).to_string())

    # THE CALIBRATION THE COUPLING NEEDED, and whether it was the same one every time. None of
    # this is scored above — the residual already reflects it — but all of it decides whether
    # the residual is a statement about the published model or about one search landing oddly.
    print("\n   WHAT THE FIT HAD TO CHOOSE to place the published curve, and whether it chose")
    print("   the same way for every trial. The curve's shape is fixed, but WHERE it sits is")
    print("   not: which end of the range is extension (sign), which chirality the plate frame")
    print("   has in the plane perpendicular to the axis (hand), and which direction in that")
    print("   plane is adduction (channel). All three are per-subject calibration in any real")
    print("   implementation. Disagreement WITHIN a joint is the warning — it means the search")
    print("   is landing in different places on different trials, and the median above is then")
    print("   a median over two different models rather than over one.\n")
    print(f"   {'joint':<10} {'source':<9} {'n':>4}  {'sign':<11} {'hand':<11} "
          f"{'channel deg':<20} min flexion deg")
    for (joint, source), rows in coupling.groupby(['joint', 'source'], observed=True):
        if rows.empty:
            continue

        def agreement(column: str, rows=rows) -> str:
            if column not in rows.columns or rows[column].isna().all():
                return '-'
            counts = rows[column].value_counts()
            return f"{counts.index[0]:+.0f} ({counts.iloc[0] / counts.sum():.0%})"

        channel = (rows['coupling_channel_angle_deg'].dropna()
                   if 'coupling_channel_angle_deg' in rows.columns else pd.Series(dtype=float))
        # Folded into [-90, 90]: the channel angle enters through an O(2) map whose reflection
        # branch is searched separately, so phi and phi+180 name the same fitted geometry and an
        # unfolded spread would read as disagreement where there is none.
        folded = ((channel + 90.0) % 180.0) - 90.0 if len(channel) else channel
        channel_text = (
            f"med {folded.median():+6.1f} IQR {folded.quantile(.75) - folded.quantile(.25):5.1f}"
            if len(folded) >= 2 else (f"{folded.iloc[0]:+6.1f}" if len(folded) else '-'))

        # How far below anatomical extension the model ran. The polynomials are defined only for
        # non-negative flexion and are clipped below it, so a strongly negative value means part
        # of the trial was scored against a constant rather than against Reuben's curve.
        flexion = (rows['coupling_min_flexion_deg'].dropna()
                   if 'coupling_min_flexion_deg' in rows.columns else pd.Series(dtype=float))
        flexion_text = f"{flexion.median():+6.1f}" if len(flexion) else '-'
        if len(flexion) and flexion.median() < -COUPLING_CLIP_WARN_DEG:
            flexion_text += '  (!) clipped'

        print(f"   {joint:<10} {source:<9} {len(rows):>4}  "
              f"{agreement('coupling_flexion_sign'):<11} "
              f"{agreement('coupling_handedness'):<11} {channel_text:<20} {flexion_text}")

    print(f"\n   (!) marks a joint whose median trial ran more than {COUPLING_CLIP_WARN_DEG:g} deg "
          f"below zero flexion,\n   where the polynomials are clipped to their value at zero and "
          f"the model being scored\n   is no longer the published one.")



def report_reference(spec: DatasetSpec, agreement: pd.DataFrame, fits: pd.DataFrame) -> None:
    _header(7, "MARKERS VERSUS BIPLANE — the noise floor under everything above",
            "The same knee over the same frames, measured by skin-mounted marker clusters and "
            "by\n   biplane fluoroscopy. The constant frame change between them is fitted "
            "away; what is\n   left is soft-tissue artifact plus marker error.")
    if agreement.empty:
        print("   Not available for this dataset (it has one reference system).")
        return
    columns = [c for c in ('n_common', 'agreement_uncorrected_rms_deg', 'agreement_rms_deg',
                           'agreement_p95_deg', 'frame_parent_deg', 'frame_child_deg')
               if c in agreement.columns]
    table = agreement.groupby(['joint', 'reference', 'comparison'],
                              observed=True)[columns].median()
    print()
    print(table.round(2).to_string())
    print(f"\n   n = {len(agreement)} trial-joint comparisons.")

    floor = float(agreement['agreement_rms_deg'].median())
    print(f"\n   THE FLOOR: {floor:.2f} deg RMS. Read every marker-based residual above "
          f"against it.")
    hinge = fits[(fits['model'] == 'hinge')]
    if not hinge.empty and 'source' in hinge.columns:
        for source, group in hinge.groupby('source', observed=True):
            value = float(group['rms_deg'].median())
            verdict = ("BELOW the reference disagreement — this residual is not measuring the "
                       "joint" if value < floor else
                       f"{value / floor:.1f}x the reference disagreement")
            print(f"      hinge residual, {source:<9} {_fmt(value)} deg   {verdict}")


def report_pooling(spec: DatasetSpec, fits: pd.DataFrame, pooled: pd.DataFrame) -> None:
    _header(8, "DOES ONE MODEL SERVE A WHOLE SESSION?",
            "A pooled fit over all a session's trials, against the per-trial fits it pools. A "
            "\n   pooled residual far above the per-trial median means the structure MOVED "
            "between\n   trials, so the pooled number is the one to distrust — and on the "
            "biplane half the\n   per-trial fits are the ones to distrust, because a 0.5 s hop "
            "does not identify an axis.")
    if pooled.empty:
        print("   No pooled fits.")
        return
    rows = []
    for (joint, source, model), group in pooled.groupby(['joint', 'source', 'model'],
                                                        observed=True):
        per_trial = fits[(fits['joint'] == joint) & (fits['source'] == source)
                         & (fits['model'] == model)]
        rows.append({
            'joint': joint, 'source': source, 'model': model,
            'pooled_rms': float(group['rms_deg'].median()),
            'per_trial_rms': float(per_trial['rms_deg'].median()) if not per_trial.empty
            else np.nan,
            'n_groups': int(len(group)),
            'trials_pooled': float(group['n_trials'].median()) if 'n_trials' in group else np.nan,
            'pooled_rom1': float(group['rom1_deg'].median()),
            'per_trial_rom1': float(per_trial['rom1_deg'].median()) if not per_trial.empty
            else np.nan,
        })
    frame = pd.DataFrame(rows)
    frame['inflation'] = frame['pooled_rms'] / frame['per_trial_rms']
    print()
    print(frame[frame['model'].isin(('hinge', 'knee_coupling', 'universal'))]
          .set_index(['joint', 'source', 'model']).round(2).to_string())

    # WHICH OF THE TWO NUMBERS IS THE UNTRUSTWORTHY ONE, decided from the data rather than left
    # to the reader. A large inflation has two possible causes and they point opposite ways:
    # either the structure really did move between trials (distrust the pooled fit), or the
    # per-trial fits were never identified in the first place and only the pooled one is real
    # (distrust the per-trial fits). Frames per trial separates them — a fit with a few dozen
    # samples against five to seven structure parameters is not a measurement.
    inflated = frame[(frame['model'] == 'hinge')
                     & (frame['inflation'] > POOLING_INFLATION_WARN)]
    if inflated.empty:
        return
    print(f"\n   (!) POOLING DISAGREES WITH THE PER-TRIAL FITS by more than "
          f"{POOLING_INFLATION_WARN:g}x on:\n")
    for _, row in inflated.iterrows():
        per_trial = fits[(fits['joint'] == row['joint']) & (fits['source'] == row['source'])
                         & (fits['model'] == 'hinge')]
        scored = float(per_trial['n_scored'].median()) if 'n_scored' in per_trial else np.nan
        # Within-subject scatter of the per-trial fitted axis. If the per-trial fits were
        # identified, their axes agree; if they scatter, each trial found its own axis and its
        # small residual is a fit to noise, so the pooled number is the one to believe.
        scatter = []
        for _, subject_rows in per_trial.dropna(subset=['u_x']).groupby('subject',
                                                                       observed=True):
            if len(subject_rows) >= MIN_AXIS_SCATTER_TRIALS:
                scatter.append(np.median(
                    axial_mean(subject_rows[['u_x', 'u_y', 'u_z']].to_numpy(float))[1]))
        scatter_text = (f"{np.median(scatter):5.1f} deg" if scatter else "  n/a")
        print(f"      {row['joint']:<8} {row['source']:<9} {row['inflation']:5.1f}x  "
              f"per-trial samples {scored:8,.0f}  within-subject axis scatter {scatter_text}")
    print("\n   Read the two right-hand columns together. Few samples AND a scattered axis mean")
    print("   the per-trial fits are unidentified and their small residuals are overfitting, so")
    print("   the POOLED row is the measurement. Many samples and a tight axis mean the")
    print("   per-trial fits are sound and the pooled row is averaging over structure that")
    print("   genuinely moved. Neither is assumed here; the numbers above decide it.")


def report_generic(spec: DatasetSpec, generic: pd.DataFrame) -> None:
    _header(9, "ONE PUBLISHED AXIS FOR EVERYONE — leave-one-subject-out",
            "The axis is the axial mean over the OTHER subjects (a joint axis is an undirected "
            "\n   line, so the mean is the principal eigenvector of sum(u u^T)). Only the "
            "neutral pose\n   is refitted per subject.")
    if generic.empty:
        print("   Not available — needs at least three subjects with a pooled fit.")
        return
    table = generic.groupby(['joint', 'source'], observed=True).agg(
        n=('subject', 'nunique'),
        dispersion_median_deg=('axis_from_cohort_deg', 'median'),
        dispersion_p90_deg=('axis_from_cohort_deg', lambda s: float(np.percentile(s, 90))),
        own_rms_deg=('own_rms_deg', 'median'),
        generic_rms_deg=('generic_rms_deg', 'median'),
        penalty_deg=('generic_penalty_deg', 'median'))
    print()
    print(table.round(2).to_string())
    print("\n   READ THE DISPERSION FIRST. On a marker dataset these axes are in each subject's")
    print("   own PLATE frame and the sensors are re-strapped per subject, so a large "
          "dispersion\n   means the 'axis' below is an average over mountings and says nothing "
          "anatomical.\n   The biplane source's frames are bone-fixed, so its row is a real "
          "anatomical claim.")


def print_report(spec: DatasetSpec, fits: pd.DataFrame, pooled: pd.DataFrame,
                 agreement: pd.DataFrame, generic: pd.DataFrame) -> None:
    print(f"\n{'#' * 100}")
    print(f"# JOINT DOF — {spec.name}")
    # Counted on the PAIR, not on the trial name. Al Borno's trials are called 'walking'
    # and 'complexTasks' for every subject, so a nunique() over the name alone reported 19
    # trials as 2 — in the report header and in every figure's provenance stamp.
    print(f"# {len(fits)} model fits over {n_trials(fits)} trials, "
          f"{fits['subject'].nunique() if 'subject' in fits else 0} subjects")
    print(f"{'#' * 100}")
    report_ladder(spec, fits)
    report_context(spec, fits)
    report_anisotropy(spec, fits)
    report_jitter(spec, fits)
    report_geometry(spec, fits)
    report_coupling(spec, fits)
    report_reference(spec, agreement, fits)
    report_pooling(spec, fits, pooled)
    report_generic(spec, generic)


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate(duration: float = 60.0, seed: int = 0) -> pd.DataFrame:
    """Fits the ladder to synthetic joints whose DOF count is known by construction.

    Any nonzero residual on a MATCHED model is a bug in this file, not sensor noise. The 1-DOF
    case swings through ~120 deg, so a ~0 hinge error there also shows these are exact rank
    tests rather than small-angle approximations.

    The Reuben case is the one that matters: a knee built to obey the published coupling
    exactly must be fitted to ~0 by the coupling model while costing the hinge real error. If
    the estimator cannot find a coupling that is there by construction, its small gains on real
    data mean nothing.
    """
    # The synthetic joint builders live in test/fixtures.py rather than on PlateTrial, so this
    # import reaches into the test tree. That is deliberate: they exist to make a joint whose
    # DOF count is known by construction, which is a testing job, and duplicating them here to
    # avoid the import would create a second definition of "a hinge" that could drift from the
    # one the PlateTrial tests use.
    from test.fixtures import (generate_1dof_plate, generate_2dof_plate, generate_3dof_plate,
                               generate_random_plate_trial)

    np.random.seed(seed)
    parent = generate_random_plate_trial(duration=duration, fs=100.0, add_noise=False)
    timestamps = np.asarray(parent.imu_trace.timestamps, dtype=float)
    centre_parent, centre_child = np.array([0.1, 0.0, 0.0]), np.array([0.0, -0.2, 0.0])

    def relative(child) -> np.ndarray:
        return _transpose_multiply(np.asarray(parent.world_trace.rotations, dtype=np.float64),
                                   np.asarray(child.world_trace.rotations, dtype=np.float64))

    cases: Dict[str, np.ndarray] = {
        '1dof hinge': relative(generate_1dof_plate(
            parent, centre_parent, centre_child, add_noise=False)),
        # 15 deg of deviation from perpendicular, i.e. axes 75 deg apart. generate_2dof_plate's
        # `carrying_angle` is the deviation, not the angle between the axes; the elbow
        # literature's sense of the term — and the one this file reports — is the latter.
        '2dof universal': relative(generate_2dof_plate(
            parent, centre_parent, centre_child, carrying_angle=np.deg2rad(15.0),
            parent_offset=centre_parent, child_offset=centre_child, add_noise=False)),
        '3dof spherical': relative(generate_3dof_plate(
            parent, centre_parent, centre_child, add_noise=False)),
    }
    cases['1dof curved'] = _synthetic_curved_joint(cases['1dof hinge'])
    cases['1dof reuben'] = _synthetic_reuben_knee(len(timestamps))

    rows = []
    for label, R in cases.items():
        series = JointSeries(R=R, R_raw=R, timestamps=timestamps[:len(R)],
                             run_id=np.zeros(len(R), dtype=np.int32),
                             fs=100.0, n_valid_total=len(R), n_runs=1)
        # Named 'Knee' for BOTH 1-DOF curved cases, so `analyze_joint` fits the coupling to
        # each. The reuben case is the positive control and the curved case is the negative
        # one — a joint that is genuinely curved but not curved Reuben's way — and the negative
        # control only tests anything if the coupling is actually fitted to it. Named for the
        # reuben case alone, it passed on a NaN.
        joint = 'R_Knee' if label in ('1dof reuben', '1dof curved') else label
        result = analyze_joint(series, {'joint_key': joint, 'joint': joint,
                                        'source': 'synthetic', 'placement': 'M'},
                               folds=0)
        by_model = {row['model']: row for row in result.rows}
        coupling = by_model.get('knee_coupling', {})
        rows.append({
            'case': label,
            'rom1_deg': by_model['hinge']['rom1_deg'],
            'weld_rms': by_model['weld']['rms_deg'],
            'hinge_rms': by_model['hinge']['rms_deg'],
            'hinge_p95': by_model['hinge']['p95_deg'],

            'universal_rms': by_model['universal']['rms_deg'],
            'universal_p95': by_model['universal']['p95_deg'],
            'carrying_angle_deg': by_model['universal'].get('carrying_angle_deg'),
            'coupling_rms': coupling.get('rms_deg', np.nan),
            'coupling_p95': coupling.get('p95_deg', np.nan),
            'coupling_curvature_deg': coupling.get('curvature_max_deg', np.nan),
        })
    return pd.DataFrame(rows)


def _synthetic_reuben_knee(n: int, max_flexion_deg: float = 85.0) -> np.ndarray:
    """A knee that obeys the PUBLISHED Reuben coupling exactly, through a knee-like range.

    Flexion is non-negative, like a knee's. The polynomials are quartics with large negative
    coefficients and are zero at zero by construction, so at negative flexion they diverge
    rather than extrapolate — the rotation channel reads +13 deg at 60 deg and -84 deg at -108
    — and a zero-mean profile would build a joint that is not a knee and that no correct
    estimator should fit.
    """
    axis = np.array([0.0, 0.0, 1.0])
    t = np.linspace(0.0, 1.0, n)
    swing = 0.5 * (1.0 - np.cos(2 * np.pi * 3.0 * t)) * (0.6 + 0.4 * t)
    flexion = np.deg2rad(max_flexion_deg) * swing / max(float(swing.max()), 1e-9)
    perp = _tangent_basis(axis)
    nu = flexion[:, None] * axis + knee_coupling_curve(flexion) @ perp.T
    return Rotation.from_rotvec([0.2, -0.3, 0.45]).as_matrix() @ _exp_matrix(nu)


def _synthetic_curved_joint(R_hinge: np.ndarray, amplitude_deg: float = 8.0) -> np.ndarray:
    """A 1-DOF joint whose off-axis rotation is a QUADRATIC function of the joint angle.

    Quadratic on purpose. A linear coupling is a tilted hinge, which the free-axis hinge
    absorbs for nothing, so a linear test case would say nothing. The quadratic term is
    curvature the hinge genuinely cannot reach — and it is NOT Reuben's shape, which is what
    makes this the negative control for the coupling.
    """
    rot = Rotation.from_matrix(R_hinge)
    ref = rot.mean()
    v_log = (ref.inv() * rot).as_rotvec()
    axis = _principal_axes(np.cov(v_log.T))[1][:, 0]
    q = v_log @ axis
    perp = _tangent_basis(axis)
    scale = np.deg2rad(amplitude_deg) / max(float(np.max(q ** 2)), 1e-9)
    nu = q[:, None] * axis + (scale * q ** 2)[:, None] * perp[:, 0]
    return ref.as_matrix() @ _exp_matrix(nu)


def report_validation() -> int:
    frame = validate()
    print(f"\n{'=' * 100}\nVALIDATION on synthetic joints of known DOF (noise-free)\n{'=' * 100}")
    print(frame.round(3).to_string(index=False))

    hinge = frame[frame['case'] == '1dof hinge'].iloc[0]
    curved = frame[frame['case'] == '1dof curved'].iloc[0]
    reuben = frame[frame['case'] == '1dof reuben'].iloc[0]
    universal = frame[frame['case'] == '2dof universal'].iloc[0]
    checks = {
        f"hinge fitted to ~0 at {hinge['rom1_deg']:.0f} deg of range":
            hinge['hinge_p95'] < 0.05,
        "a curved joint costs the hinge real error":
            curved['hinge_rms'] > 1.0,
        # The negative control, and it is a RATIO rather than a threshold because the honest
        # answer turned out to be softer than "Reuben must not help". Over a limited flexion
        # range Reuben's quartics are not far from a generic quadratic, and with the axis, the
        # neutral pose and the channel direction all free the published curve absorbs a good
        # part of a coupling it does not describe: on this joint it takes the hinge's 1.46 deg
        # down to 0.47. What it CANNOT do is what it does on a knee that really obeys it, which
        # is go to zero. So the discriminating quantity is the FRACTION of the hinge's error
        # left behind, and the gap between the two cases is what the check pins.
        #
        # This is a caveat on the real results, not only on the fixture: a knee where the
        # coupling beats the hinge is not thereby shown to obey Reuben. `report_coupling` says
        # so where the verdicts are printed.
        "Reuben leaves far more behind on a joint curved the WRONG way":
            np.isfinite(curved['coupling_rms'])
            and (curved['coupling_rms'] / curved['hinge_rms']
                 > reuben['coupling_rms'] / reuben['hinge_rms'] + 0.2),
        "universal joint fitted to ~0 by the 2-DOF model":
            universal['universal_p95'] < 0.5,
        "universal joint costs the hinge real error":
            universal['hinge_rms'] > 2.0,
        "universal joint's carrying angle recovered near 75 deg":
            abs(universal['carrying_angle_deg'] - 75.0) < 3.0,
        "a knee obeying the PUBLISHED coupling is fitted to ~0 by it":
            reuben['coupling_p95'] < 0.5,
        "the published coupling beats a free hinge when it is true":
            reuben['coupling_rms'] < 0.5 * reuben['hinge_rms'],
        "the coupling's own curvature is recovered, not flattened":
            reuben['coupling_curvature_deg'] > 1.0,
        # The central quantitative point: a free-axis hinge already absorbs the coupling's
        # linear part, which is nearly all of it, so what the coupling can buy is only the
        # curvature. If this ever fails the report's whole framing is wrong.
        "a free hinge axis absorbs almost all of that coupling":
            reuben['hinge_rms'] < 1.5,
    }
    print()
    for description, passed in checks.items():
        print(f"   [{'PASS' if passed else 'FAIL'}] {description}")
    return 0 if all(checks.values()) else 1


# ==============================================================================
# CLI
# ==============================================================================

def select_trials(dataset: str, subjects: Optional[List[str]],
                  trials: Optional[List[str]]) -> List[Tuple[str, str]]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='alborno', choices=sorted(DATASETS))
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--trials', nargs='+', default=None)
    parser.add_argument('--joints', nargs='+', default=None,
                        help="Anatomical joint names to fit, e.g. R_Knee L_Knee. Keeps every "
                             "sensor placement on the joints it names.")
    parser.add_argument('--placements', nargs='+', default=None, metavar='P',
                        help="Sensor placements to fit: M (Mid), H (High), L (Low). Only IMoVE "
                             "has more than one — three sensors per thigh and shank against one "
                             "marker cluster — which makes its joint table eighteen pairs and "
                             "four times the runtime of the six. '--placements M' is the "
                             "six-joint dataset most readers have in mind. No-op elsewhere.")
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--folds', type=int, default=CV_FOLDS,
                        help="Cross-validation folds; 0 disables it and leaves only in-sample "
                             "errors, which are NOT comparable across models of different "
                             "structure size.")
    parser.add_argument('--stride', type=int, default=SAMPLE_STRIDE,
                        help="Keep every Nth sample in error_samples. Every scalar in "
                             "joint_fits is computed on the full scoring set regardless.")
    parser.add_argument('--lowpass-hz', type=float, default=LOWPASS_HZ)
    parser.add_argument('--full-score', action='store_true',
                        help=f"Score on every sample instead of striding down to "
                             f"{SCORE_MAX_SAMPLES:,}. Slow on IMoVE; the difference is well "
                             f"below the reported precision.")
    parser.add_argument('--no-knee-coupling', action='store_true',
                        help="Skip the published Reuben knee coupling — the 1-DOF NONLINEAR "
                             "rung — which is otherwise fitted at every knee. Only worth it to "
                             "save the few seconds per knee it costs.")
    parser.add_argument('--only-tables', nargs='+', choices=TRIAL_TABLES,
                        default=list(TRIAL_TABLES), metavar='TABLE')
    parser.add_argument('--no-pooled', action='store_true',
                        help="Skip the per-session pooled fits and the generic-axis pass, "
                             "which reload every trial.")
    parser.add_argument('--report-only', action='store_true',
                        help="Rebuild the summary and report from what is already on disk.")
    parser.add_argument('--validate', action='store_true',
                        help="Fit the ladder to synthetic joints of known DOF, then exit.")
    args = parser.parse_args()

    pd.set_option('display.width', 220, 'display.max_columns', 60)

    if args.validate:
        return report_validation()

    try:
        row_keys = select_trials(args.dataset, args.subjects, args.trials)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    spec = get_dataset(args.dataset)
    score_cap = 0 if args.full_score else SCORE_MAX_SAMPLES

    if not args.report_only:
        print(f"Fitting joint models over {len(row_keys)} trials...")
        state, _ = run_tracked_grid(
            row_keys, ['Subject', 'Trial'], ['fit'],
            partial(_trial_worker, dataset=args.dataset, tables=args.only_tables,
                    folds=args.folds, stride=args.stride, score_cap=score_cap,
                    cutoff_hz=args.lowpass_hz,
                    joints=args.joints, placements=args.placements,
                    knee_coupling=not args.no_knee_coupling),
            args.workers, title=f"JOINT DOF — {args.dataset}")
        # Say so when the compute pass computed nothing. Without this the run goes on to load
        # whatever per-trial tables happen to be on disk and prints a full report off them,
        # describing a PREVIOUS run's numbers with nothing to indicate it.
        failures = {key: value for (key, stage), value in state.items()
                    if stage == 'fit' and isinstance(value, str) and value.startswith('Failed')}
        if failures:
            reasons: Dict[str, int] = {}
            for message in failures.values():
                reasons[message[:80]] = reasons.get(message[:80], 0) + 1
            print(f"\n{len(failures)} of {len(row_keys)} trials FAILED:")
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
                print(f"  {count:4d} x {reason}")
            if len(failures) == len(row_keys):
                print("\nEvery trial failed, so nothing was written. Anything below would "
                      "describe a\nPREVIOUS run's tables — stopping instead.")
                return 1

    print("\nLoading per-trial tables...")
    fits = load_trial_table(args.dataset, 'joint_fits')
    if fits.empty:
        print(f"No results under {dataset_dir(args.dataset)}. Run without --report-only first.")
        return 1
    found = sorted({(str(s), str(t))
                    for s, t in fits[['subject', 'trial']].drop_duplicates().to_numpy()})
    print(f"Found {len(found)} trial(s) across {len(subjects_of(found))} subject(s).")

    agreement = load_trial_table(args.dataset, 'reference_agreement')

    pooled = pd.DataFrame()
    generic = pd.DataFrame()
    if not args.no_pooled:
        print("\nFitting pooled per-session models...")
        pooled = pooled_fits(args.dataset, spec, found, args.folds, score_cap,
                             args.lowpass_hz, args.joints, args.placements,
                             workers=args.workers,
                             knee_coupling=not args.no_knee_coupling)
        if not pooled.empty:
            path = dataset_dir(args.dataset) / 'pooled_fits.parquet'
            _save(pooled, path, args.dataset, table='pooled_fits')
            print(f"Saved pooled fits to {path}")
            print("\nMeasuring the cost of one axis for everyone...")
            generic = generic_axes(args.dataset, spec, pooled, found, args.lowpass_hz,
                                   score_cap, args.joints, args.placements)
            if not generic.empty:
                path = dataset_dir(args.dataset) / 'generic_axes.parquet'
                _save(generic, path, args.dataset, table='generic_axes')
                print(f"Saved generic-axis transfer to {path}")

    summary = summarize(args.dataset, fits)
    if not summary.empty:
        path = statistics_path(args.dataset)
        _save(summary, path, args.dataset, subjects=subjects_of(found))
        print(f"Saved summary to {path}")

    print_report(spec, fits, pooled, agreement, generic)
    print(f"\nPer-trial tables under {dataset_dir(args.dataset)}")
    print(f"Figures: python -m plotting.joint_dof --dataset {args.dataset}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
