"""
Why a relative correction at the joint center beats correcting each sensor against a global
reference — as geometry, and then as a measurement on this dataset.

This is a property of the DATA and of spherical geometry alone. No filter is run, no EKF is
touched, and nothing here depends on a tuning constant.

The two ways to use a vector field
----------------------------------
An IMU pair spanning a joint measures the same physical field twice, once in each sensor's
body frame. There are two ways to turn that into an orientation measurement, and this repo
contains both:

    ABSOLUTE   Each sensor is compared against a single global reference: the residual is
               R_J m_J - m_G for the parent and R_K m_K - m_G for the child. This is what
               every per-segment AHRS does, and in this repo it is what the `ekf` baseline
               does — experiment_utils._setup_ekf_ground_plate_ builds a virtual parent whose
               magnetometer is a constant world field and whose accelerometer is a constant
               gravity vector, so its residual reduces to exactly the above.

    RELATIVE   The two sensors are compared against EACH OTHER: the residual is
               R_J m_J - R_K m_K, with no global reference anywhere in it. This is what
               RelativeFilter.get_h computes (see its 1DOF vector-sensor block), and it is
               the whole reason the method is called relative.

Both residuals are zero when the field is uniform and the sensors are perfect. Neither is,
so both residuals are nonzero at the TRUE orientation, and each filter responds by rotating
its orientation estimate away from truth until the residual it is looking at vanishes. The
question this experiment answers is how much rotation each residual demands.

The geometry
------------
Rotate each sensor's reading into the world frame with its ground-truth mocap orientation and
normalize, giving unit vectors u_J and u_K on the sphere S^2, and let g be the unit global
reference. Three angles follow, all great-circle distances on S^2:

    theta_rel     = angle(u_J, u_K)     what the relative residual demands
    theta_parent  = angle(u_J, g)       what the absolute residual demands of the parent
    theta_child   = angle(u_K, g)       what the absolute residual demands of the child

Write C_J for the minimum rotation carrying u_J onto g, and C_K likewise for u_K. These are
the world-frame orientation errors the absolute route incurs. The joint angle it reports is
then wrong by

    C_comp = C_J^T C_K,          theta_comp = |rotvec(C_comp)|

because the estimated relative orientation is R_J^T C_J^T C_K R_K, whose error is a
conjugation of C_J^T C_K and therefore has the same angle. Four facts, all verified to
machine precision over 200,000 random configurations and re-verified on every sample of this
dataset (see report_theorem):

  1. GEODESIC TRIANGLE INEQUALITY.  theta_rel <= theta_parent + theta_child.
     Great-circle distance is a metric on S^2, so the direct route from u_J to u_K is never
     longer than the detour through g. The relative correction is therefore never larger than
     the total correction the absolute route must apply across the two segments.

  2. MINIMUM ROTATION PROPERTY.  theta_rel <= theta_comp.
     C_comp carries u_K onto u_J: C_K u_K = g and C_J^T g = u_J. Any rotation carrying one
     unit vector onto another must turn by at least the angle between them, so theta_comp is
     bounded below by theta_rel. This is the sharp statement, and the one that matters for
     joint kinematics: it says the relative residual's demand is the ABSOLUTE LOWER BOUND on
     the relative orientation error that ANY reference-based correction of the same pair can
     achieve. Note the quantifier — it holds pointwise for every g, so it cannot be evaded by
     choosing a better global reference (report_reference_sensitivity measures how much the
     choice moves the gap, which is a different and answerable question).

  3. AN EXACT CLOSED FORM, not just an inequality.
         cos(theta_comp / 2) = cos(theta_rel / 2) cos(psi / 2)
     where psi is the angle of C_comp C_min^T, C_min being the minimum rotation carrying u_K
     onto u_J. C_comp and C_min both carry u_K onto u_J, so their ratio fixes u_J and is a
     rotation ABOUT u_J — i.e. the absolute route's excess is a pure twist about the measured
     field direction, which is precisely the direction a single vector measurement cannot
     observe. Since the twist axis is perpendicular to C_min's axis, the half-angle formula
     for composed rotations collapses to the product above, from which theta_comp >= theta_rel
     follows again with equality iff psi = 0.

  4. THE EXCESS IS A SPHERICAL AREA.  psi is the spherical excess (the signed area) of the
     triangle (u_J, g, u_K) on S^2, wrapped to (-180, 180] degrees. So the penalty for going
     through a global reference is zero exactly when that reference lies on the great circle
     through the two measured directions, and grows with how far off that plane it lies.
     Coplanar configurations give theta_comp = theta_rel exactly, which is the degenerate case
     the inequality in 2 is tight on, and the reason 2 is stated as <= and not <.

Together: correcting both sensors to a global reference does not merely add error, it adds a
specific, geometrically identified error — an out-of-plane twist about the field direction,
of magnitude set by the area of the spherical triangle the two measurements and the reference
enclose. The relative residual has no such term because it has no third point.

What the geometry above is worth on this dataset, which is less than it looks
----------------------------------------------------------------------------
Facts 1-4 are exact and hold on every sample here. Their EFFECT SIZES are wildly different,
and reporting the two inequalities as though they were one result would be misleading:

    theta_sum  - theta_rel     ~ 14 deg at the median   (fact 1: large)
    theta_comp - theta_rel     ~ 0.06 deg at the median (fact 2: real, and negligible)

Fact 3 explains the second number instead of leaving it as a surprise. Expanding the closed
form for small angles gives

    theta_comp ~ sqrt(theta_rel^2 + psi^2),

so the out-of-plane twist adds to the relative error IN QUADRATURE, not linearly. On this data
the triangle (u_J, g, u_K) is a sliver — the two measurements are 5-20 deg apart and the
reference is a similar distance from both — so its area psi is a few tenths of a degree, and a
few tenths added in quadrature to several degrees costs essentially nothing.

The honest reading of fact 2 is therefore a NEGATIVE result about single-field geometry, and it
is worth stating in the paper as one: when two sensors are corrected individually to a shared
global reference, the two absolute errors are largely COMMON MODE and cancel in the relative
orientation. Correcting to a global reference is bad for each segment's absolute orientation
(fact 1, 14 deg) and very nearly free for the joint angle between them (fact 2, 0.06 deg). Any
claim that the relative residual improves joint kinematics has to come from somewhere else,
because for one vector field at a time it does not.

The irreducible part, which is where the joint-angle advantage actually lives
----------------------------------------------------------------------------
It comes from using the two fields TOGETHER. The angle between the accelerometer and
magnetometer vectors,

    beta = angle(a, m),

is a ROTATION INVARIANT: it is measured in the sensor's own body frame and no orientation
estimate enters it, so no orientation estimate can change it. That single observation converts
the argument from one about which rotation to apply into one about which errors cannot be
removed at all. Write beta_J and beta_K for the two sensors and beta_G = angle(g, m_G) for the
global reference pair. Three more facts, verified alongside the first four:

  5. AN INVARIANT MISMATCH IS IRREDUCIBLE, AND SPLITS.  For two unit observation pairs whose
     inter-vector angles differ by Delta beta, the rotation minimizing the summed squared
     alignment error leaves EACH vector misaligned by exactly Delta beta / 2, and no rotation
     does better than Delta beta / 2 on the worse of the two. So Delta beta / 2 is a hard floor
     on the orientation error, not a bound on one particular estimator.

  6. THE FLOORS OBEY THE SAME TRIANGLE INEQUALITY, now on the real line rather than on S^2:
         |beta_J - beta_K|  <=  |beta_J - beta_G| + |beta_G - beta_K|
     The relative residual must reconcile beta_J against beta_K. The absolute residual must
     reconcile each of them against beta_G. Unlike facts 1-4 this survives being turned into a
     statement about joint angles, because the mismatch cannot be shuffled between the two
     segments — there is no rotation that removes it from either.

  7. THE MISMATCH IS WHAT DIFFERS BETWEEN THE TWO ROUTES, and it differs a lot. Both physical
     mechanisms that corrupt beta act almost identically on two sensors a segment apart and
     not at all on the global reference pair: a distorted local field tilts m the same way at
     both, and after projection to the shared joint center the linear acceleration tilts a
     identically at both. So beta_J - beta_K is small while beta_J - beta_G and beta_K - beta_G
     are not (report_invariant reports all three).

This is also the one part of the analysis that does not depend on mocap at all. beta is computed
from raw body-frame readings, so the alignment caveat below — which inflates every angle in
facts 1-4 — cannot touch it.

Why this is not a magnetometer argument
---------------------------------------
Nothing above mentions magnetism. The same three unit vectors and the same four facts apply
to the accelerometer, and the two fields fail the uniform-field assumption for entirely
different physical reasons, which is what makes running both worth the effort:

    MAGNETOMETER   u_J and u_K differ from g because the field is locally distorted by
                   ferrous material in the room and on the body. Distortion is spatially
                   correlated, so two sensors a segment apart see similar fields — small
                   theta_rel — while both differ from the global field — large theta_parent,
                   theta_child. (experiments/sensor_distributions.py measures that spatial
                   correlation directly, as var_reduction.)

    ACCELEROMETER  u_J and u_K differ from gravity because the body accelerates. Here the
                   mechanism is sharper: once both readings are rigid-body projected to the
                   SHARED joint center, both sensors are measuring the specific force at the
                   same physical point, so the linear acceleration is exactly common-mode and
                   cancels from the relative residual, while it survives in full in the
                   absolute one. The `acc_unprojected` field is carried through the whole
                   analysis as the control for that claim: without the projection the two
                   sensors measure different points and the cancellation is only partial.

So the mag and acc arms of the figure are not two versions of one result. They are the same
theorem driven by field inhomogeneity in one case and by rigid-body kinematics in the other.

What the numbers are and are not
--------------------------------
These angles are the orientation error each residual DEMANDS, evaluated at the ground-truth
orientation. They are not a filter's realized error, and this analysis cannot become one:

  * A real filter sees acc and mag together, plus a gyro prediction, and distributes the
    correction across both segments according to its covariance. To first order at equal
    trust the Kalman update takes the minimum-norm state correction consistent with the
    residual, which is what makes theta_rel attained rather than merely a bound — but that is
    an argument about the update, not something measured here.

  * A single vector measurement leaves rotation about its own direction unobserved. theta_rel
    and theta_comp are the components the measurement actually constrains; the unconstrained
    twist is carried by the gyro and by the other sensor.

  * MOCAP ALIGNMENT BIASES THIS AGAINST ITS OWN CONCLUSION, which is worth stating plainly
    because it is the first thing a referee will ask. u_J and u_K are built by rotating with
    PlateTrial's mocap-to-sensor alignment, so a residual misalignment of the parent tilts
    u_J and inflates theta_parent by that tilt — but it inflates theta_rel by the parent's
    tilt AND the child's. The alignment floor therefore hurts the relative angle roughly
    twice as hard as it hurts either absolute angle, so every measured gap here is a LOWER
    bound on the true gap. (The floor itself is measured in
    experiments/acceleration_projection.py, report_alignment.)

  * theta_min_abs = min(theta_parent, theta_child) is reported, and the relative correction is
    NOT guaranteed to beat it: if one sensor happens to sit exactly on the global field then
    its absolute correction is zero while theta_rel is the other sensor's full distortion.
    That case is real and is counted (report_oracle_caveat). It is not an implementable
    method — nothing tells a per-segment filter which of its sensors to trust — so it is
    reported as an oracle, in the same spirit as experiments/oracle_ablation.py.

Outputs, all under results/experiments/relative_vs_absolute/:

    Subject<NN>/<activity>/angle_samples.parquet      per-sample direction angles (facts 1-4),
                                                      every joint x field, PRIMARY reference
                                                      only, decimated by SAMPLE_STRIDE
    Subject<NN>/<activity>/invariant_samples.parquet  per-sample inter-field angles and their
                                                      mismatches (facts 5-7), same decimation
    Subject<NN>/<activity>/traces.parquet             full-rate angles for EXAMPLE_JOINT, every
                                                      field, for the time-series panel
    Subject<NN>/<activity>/joint_stats.parquet        per joint x field x reference scalars: the
                                                      quantiles, the theorem's verification
                                                      residuals, and the win fractions
    Subject<NN>/<activity>/invariant_stats.parquet    the same, per joint x acc source

plus the pooled quantile summary at
results/statistics/relative_vs_absolute_statistics.parquet, which is what the console report
and the figure caption quote. Its `family` column separates the two arguments: 'direction' for
facts 1-4 and 'invariant' for facts 5-7.

Runtime is dominated by IMUTrace.project_acc's per-sample polyfit gyro derivative (the same
cost experiments/acceleration_projection.py pays), so the grid runs one process per subject
with both activities inside it — the magnetometer's global reference is per-subject and spans
both activities, so they cannot be split.
"""
import argparse
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import paths
from experiments.experiment_utils import (ACTIVITIES, EXPECTED_GRAVITY, JOINTS, SUBJECTS,
                                          _compute_expected_mag_field, load_raw_data,
                                          pipeline_constants, project_pair_to_joint_center,
                                          run_tracked_grid)
from experiments.sensor_distributions import expected_mag_field
from src.toolchest.PlateTrial import PlateTrial

EXPERIMENT_NAME = "relative_vs_absolute"
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# The three vector fields compared, and what each one is for. Order matters only in that the
# first two are the paper's claim and the third is the control behind the accelerometer half
# of it.
#
#   mag              body-frame magnetometer, as measured. No projection: a magnetic field is
#                    a property of a POINT IN SPACE, not of a rigid body, so there is no
#                    rigid-body transport law that would move a reading to the joint center.
#                    That is the entire premise of the MAJIC filter — the field at the joint
#                    center is estimated, not derived.
#   acc              accelerometer AFTER rigid-body projection to the shared joint center,
#                    i.e. exactly the signal the relative filter consumes when project=True.
#   acc_unprojected  the raw accelerometer, the control for the projection's role. The two
#                    sensors then measure the specific force at two DIFFERENT points, so the
#                    common-mode cancellation that makes theta_rel small is only partial.
FIELDS = ('mag', 'acc', 'acc_unprojected')

# Global reference variants, per field, primary first.
#
# The primary is what the summary and the figure quote; the alternates exist because the
# obvious objection to this whole analysis is that the global reference was chosen badly, and
# an inequality that holds for every g (fact 2 in the module docstring) deserves to be shown
# holding under more than one. What the choice CAN move is the size of the gap, and
# report_reference_sensitivity reports exactly that.
#
#   subject_median    median world-frame field over every sensor and both activities of the
#                     subject. Median and pooled for the reasons given in
#                     sensor_distributions.expected_mag_field: distortion is one-sided and
#                     heavy-tailed, and pooling keeps any one segment from being privileged.
#   torso_median      median world-frame field over the TORSO sensors of this trial — i.e.
#                     experiment_utils._compute_expected_mag_field, which is literally the
#                     reference the `ekf` baseline's ground plate is built from. Included so
#                     the absolute arm is measured against the reference this repo's own
#                     absolute method actually uses, not a reconstruction of it.
#   expected_gravity  EXPECTED_GRAVITY, what a gravity-referenced AHRS assumes.
#   spherical_mean    the constant unit direction closest to the measurements themselves:
#                     normalize(mean of every u_J and u_K over the trial). This is the best a
#                     constant reference can do in the least-squares-chord sense, so it is the
#                     steelman — if the gap survives against this, no calibration of the global
#                     reference closes it.
REFERENCE_VARIANTS = {
    'mag': ('subject_median', 'torso_median', 'spherical_mean'),
    'acc': ('expected_gravity', 'spherical_mean'),
    'acc_unprojected': ('expected_gravity', 'spherical_mean'),
}
PRIMARY_REFERENCE = {field: variants[0] for field, variants in REFERENCE_VARIANTS.items()}

# Seconds trimmed from both ends of every trial. IMUTrace.project_acc's gyro derivative is a
# sliding-window polynomial fit, so its first and last samples are fit against a truncated
# window; 1.0 s at 100 Hz clears that by two orders of magnitude. Applied to every field, not
# just the projected one, so the three are compared over identical sample sets.
TRIM_S = 1.0

# Stride for the stored per-sample table. Three fields over the same joints means three times
# the rows acceleration_projection stores at the same stride, and every panel drawn from this
# table is a density or a quantile — neither of which resolves anything at 100 Hz that it does
# not resolve at 20 Hz. The per-joint scalars in joint_stats are computed on EVERY sample
# before this decimation, so no reported number is affected by it.
SAMPLE_STRIDE = 5

# Full-rate traces are stored for this joint only, for the time-series panel. The knee is the
# joint where both mechanisms are visible at once: the shank swings hard enough for the
# accelerometer's linear-acceleration term to dominate, and the sensor sits far enough down the
# limb to pick up real field distortion.
EXAMPLE_JOINT = 'R_Knee'

# A unit vector shorter than this is treated as having no direction, and its sample is dropped
# from every angle rather than producing a NaN or an arbitrary axis. Real magnetometer norms
# here are ~1 and real accelerometer norms ~9.8, so this only fires on genuinely degenerate
# samples (a dropout, or a projected accelerometer passing through zero in free fall); the
# fraction it drops is recorded per joint as `degenerate_frac` instead of being silently
# absorbed.
MIN_NORM = {'mag': 1e-6, 'acc': 1e-6, 'acc_unprojected': 1e-6}

# Antiparallel measurement and reference: the shortest-arc quaternion is undefined there (the
# rotation axis is any perpendicular) and psi is undefined with it. Physically unreachable for
# either field on this data, but counted rather than assumed away.
ANTIPARALLEL_COS = -1.0 + 1e-12

# Tolerance the theorem is checked to. This is a floating-point tolerance, not a physical one:
# facts 1-4 are exact identities, so anything above it is a bug in the code that computes them,
# and report_theorem is meant to fail loudly rather than report a small violation rate.
#
# Set from the arithmetic rather than picked round. Every angle here comes out of an arccos, and
# arccos loses half its significant digits at its endpoints: near 0 the derivative is infinite,
# so arccos(1 - eps) ~ sqrt(2 eps), and at eps ~ 1e-16 that is ~1.5e-8 rad = 9e-7 deg of
# irreducible noise on an angle that should be exactly zero. Differences of two such angles
# therefore have a floor near 1e-6 deg — MEASURED, at 1.7e-6 deg over a million near-parallel
# pairs in test/TestRelativeVsAbsolute.py, not estimated. 1e-4 leaves two orders of margin
# above that floor while sitting four orders below the smallest angle this dataset produces
# (pooled median theta_rel is degrees, and the smallest per-cell median is tenths), so it
# cannot mask a violation that means anything.
#
# Tightening this to 1e-6 makes report_theorem flag arccos granularity as a violated theorem,
# which is where this constant started.
THEOREM_TOL_DEG = 1e-4

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# Facts 1-4: the single-field direction geometry. One set per entry in FIELDS.
ANGLE_METRICS = ('theta_rel', 'theta_parent', 'theta_child', 'theta_sum', 'theta_comp',
                 'theta_min_abs', 'psi_abs', 'excess')

# Facts 5-7: the two-field rotation invariant. `beta_*` are the inter-field angles themselves
# and `dip_*` their mismatches; `floor_rel` and `floor_abs` are those mismatches halved, which
# fact 5 makes the irreducible orientation error rather than a bound on it. The halves are
# stored rather than left to the reader because they are the quantity that is comparable to
# every theta_* above, and a factor of two between the two halves of one figure is exactly the
# kind of thing that gets misread.
INVARIANT_METRICS = ('beta_parent', 'beta_child', 'beta_global', 'dip_rel', 'dip_parent',
                     'dip_child', 'dip_sum', 'floor_rel', 'floor_abs')

METRIC_UNITS = {metric: 'deg' for metric in (*ANGLE_METRICS, *INVARIANT_METRICS)}

# `family` distinguishes the two arguments inside one summary table. They answer different
# questions — facts 1-4 bound the rotation a residual demands, facts 5-7 bound the error no
# rotation can remove — and their effect sizes differ by two orders of magnitude, so pooling
# them into one unlabelled set of rows would invite exactly the wrong comparison.
FAMILIES = {'direction': ANGLE_METRICS, 'invariant': INVARIANT_METRICS}

FIELD_LABELS = {'mag': 'Magnetometer', 'acc': 'Accelerometer (projected)',
                'acc_unprojected': 'Accelerometer (unprojected)',
                'acc_projected+mag': 'Acc (projected) + Mag',
                'acc_unprojected+mag': 'Acc (unprojected) + Mag'}

# The two acc sources the invariant is computed for, as `field` values, primary first. Same
# projected-vs-not control as FIELDS, and the projection matters more here than anywhere else:
# beta is where the common-mode cancellation either happens or does not.
INVARIANT_FIELDS = ('acc_projected+mag', 'acc_unprojected+mag')

TRIAL_TABLES = ('angle_samples', 'invariant_samples', 'traces', 'joint_stats',
                'invariant_stats')

ROLES = ('parent', 'child')


def analysis_constants() -> Dict[str, object]:
    """Pipeline constants plus this analysis's own choices, for provenance manifests."""
    return {
        **pipeline_constants(),
        'fields': list(FIELDS),
        'invariant_fields': list(INVARIANT_FIELDS),
        'reference_variants': {field: list(variants)
                               for field, variants in REFERENCE_VARIANTS.items()},
        'primary_reference': dict(PRIMARY_REFERENCE),
        'trim_s': TRIM_S,
        'sample_stride': SAMPLE_STRIDE,
        'example_joint': EXAMPLE_JOINT,
        'theorem_tol_deg': THEOREM_TOL_DEG,
        'min_norm': dict(MIN_NORM),
    }

# ==============================================================================
# Paths / IO
# ==============================================================================

def trial_table_path(subject: str, activity: str, table: str) -> Path:
    return EXPERIMENT_DIR / f"Subject{subject}" / activity / f"{table}.parquet"


def _save(df: pd.DataFrame, path: Path, **manifest_extra) -> None:
    df.to_parquet(paths.ensure_parent(path), engine='pyarrow', index=False)
    paths.write_manifest(path, constants=analysis_constants(), experiment=EXPERIMENT_NAME,
                         n_rows=len(df), **manifest_extra)


def load_trial_table(table: str, subjects: Optional[List[str]] = None,
                     activities: Optional[List[str]] = None,
                     columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Concatenates one per-trial table across subjects/activities, adding `subject` and
    `activity` columns. Missing trials are skipped silently — a partial run
    (`--subjects 06`) is a legitimate state, and the caller reports what it found.

    `columns` is pushed down into the parquet read rather than selected afterwards: the full
    angle_samples table across every trial is millions of rows, and a panel that needs two of
    its columns should not pay for all eight. Label columns are read back as categoricals for
    the same reason — as objects they cost more than every float column combined."""
    subjects = SUBJECTS if subjects is None else subjects
    activities = ACTIVITIES if activities is None else activities
    if columns is not None:
        columns = list(dict.fromkeys(columns))  # de-duplicate, preserve caller's order
    frames = []
    for subject in subjects:
        for activity in activities:
            path = trial_table_path(subject, activity, table)
            if not path.exists():
                continue
            frames.append(pd.read_parquet(path, engine='pyarrow', columns=columns)
                          .assign(subject=f"Subject{subject}", activity=activity))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for label in ('joint', 'field', 'reference', 'subject', 'activity'):
        if label in df.columns:
            df[label] = df[label].astype('category')
    return df

# ==============================================================================
# Spherical geometry
# ==============================================================================
# Everything here is vectorized over samples and works in unit quaternions rather than
# scipy Rotation objects. That is not a micro-optimization: this runs over ~8M samples x 3
# fields x 3 reference variants, and building a Rotation per sample was measured at roughly
# two orders of magnitude slower than the quaternion arithmetic below.
#
# Quaternion convention throughout: (N, 4) arrays laid out as [w, x, y, z].

def normalize_rows(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Row-wise unit vectors of an (N, 3) array, and the norms they came from.

    Zero rows are returned as zeros rather than NaN; the caller masks on the returned norms.
    Dividing first and masking afterwards would raise or warn on every degenerate sample."""
    norms = np.linalg.norm(v, axis=1)
    safe = np.where(norms > 0.0, norms, 1.0)
    return v / safe[:, None], norms


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample angle between two (N, 3) arrays of UNIT vectors, in degrees.

    The dot product is clipped before arccos. That is not defensive padding: these vectors are
    near-parallel most of the time (both fields are dominated by the same physical direction),
    and a normalized dot product for near-parallel vectors lands a few ulp outside [-1, 1]
    routinely, so the unclipped version returns NaN on ordinary data."""
    return np.degrees(np.arccos(np.clip(np.einsum('ni,ni->n', a, b), -1.0, 1.0)))


def shortest_arc_quat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit quaternions of the minimum rotations carrying each unit vector `a` onto `b`.

    q = [1 + a.b, a x b], normalized — the standard shortest-arc form. Its half-angle
    structure falls out directly: |a x b| = sin(theta) and 1 + a.b = 2cos^2(theta/2), so after
    normalization w = cos(theta/2) and the vector part is sin(theta/2) times the unit axis.

    Degenerate only when a.b = -1, where the axis is any perpendicular and the norm collapses
    to zero. Those samples are excluded upstream by ANTIPARALLEL_COS; the guarded divide here
    keeps a stray one from poisoning the array with NaN rather than pretending it cannot
    happen."""
    quat = np.empty((len(a), 4))
    quat[:, 0] = 1.0 + np.einsum('ni,ni->n', a, b)
    quat[:, 1:] = np.cross(a, b)
    norms = np.linalg.norm(quat, axis=1)
    return quat / np.where(norms > 0.0, norms, 1.0)[:, None]


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Row-wise conjugate, i.e. the inverse rotation for unit quaternions."""
    return np.concatenate([q[:, :1], -q[:, 1:]], axis=1)


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product, row-wise: the quaternion whose rotation matrix is M(p) @ M(q).

    Pinned against scipy's Rotation composition in test/TestRelativeVsAbsolute.py rather than
    asserted here. The opposite convention differs only by the sign of the cross term and
    produces angles of an entirely plausible magnitude for the wrong composition, so a value
    check is the only thing that catches it."""
    pw, pv = p[:, :1], p[:, 1:]
    qw, qv = q[:, :1], q[:, 1:]
    return np.concatenate([
        pw * qw - np.einsum('ni,ni->n', pv, qv)[:, None],
        pw * qv + qw * pv + np.cross(pv, qv),
    ], axis=1)


def quat_angle_deg(q: np.ndarray) -> np.ndarray:
    """Rotation angle of each unit quaternion, in degrees, in [0, 180].

    |w| rather than w: q and -q are the same rotation, and taking the absolute value is what
    selects the representative with angle <= 180 instead of returning its 360-complement."""
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(q[:, 0]), 0.0, 1.0)))


def spherical_excess_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Signed spherical excess (equivalently, the signed area) of each triangle a-b-c on the
    unit sphere, in degrees, wrapped to (-180, 180].

    The Van Oosterom-Strackee form:  E = 2 atan2( a . (b x c),  1 + a.b + b.c + c.a ),
    chosen over the law-of-cosines route because it stays accurate for the thin, nearly
    degenerate triangles that dominate here — a and c are usually a few degrees apart, which
    is exactly where a l'Huilier-style formula loses most of its digits.

    The wrap is not cosmetic. A triangle's area can exceed 180 deg, while the twist recovered
    from a rotation is confined to [0, 180], and the two agree only modulo a full turn. The
    identity in fact 3 of the module docstring is stated in terms of the wrapped value, and
    verification against the quaternion route (report_theorem) is what pins it."""
    numerator = np.einsum('ni,ni->n', a, np.cross(b, c))
    denominator = (1.0 + np.einsum('ni,ni->n', a, b)
                   + np.einsum('ni,ni->n', b, c) + np.einsum('ni,ni->n', c, a))
    excess = 2.0 * np.arctan2(numerator, denominator)
    return np.degrees(np.remainder(excess + np.pi, 2.0 * np.pi) - np.pi)


def correction_angles(u_parent: np.ndarray, u_child: np.ndarray, reference: np.ndarray
                      ) -> Dict[str, np.ndarray]:
    """The whole geometry for one joint, one field, one global reference.

    Takes (N, 3) unit measurement directions in the world frame and a single (3,) unit
    reference, and returns the angles defined in the module docstring plus the two residuals
    that verify the theorem. Every value is in degrees; NaN marks a sample excluded as
    degenerate (see MIN_NORM, ANTIPARALLEL_COS), and NaN propagates through the comparisons
    in report_theorem rather than being counted as a violation.

    theta_comp is computed from the composed quaternion DIRECTLY and psi from the spherical
    excess DIRECTLY, so the closed form in fact 3 is a check rather than a definition. Deriving
    one from the other would make the verification circular and it would pass no matter what.
    """
    reference = np.broadcast_to(reference, u_parent.shape)

    theta_rel = angle_between_deg(u_parent, u_child)
    theta_parent = angle_between_deg(u_parent, reference)
    theta_child = angle_between_deg(u_child, reference)

    # The absolute route's two corrections, and the relative orientation error they compose to.
    q_parent = shortest_arc_quat(u_parent, reference)
    q_child = shortest_arc_quat(u_child, reference)
    q_comp = quat_multiply(quat_conjugate(q_parent), q_child)
    theta_comp = quat_angle_deg(q_comp)

    # The minimum rotation carrying u_child onto u_parent, and what the composed correction
    # has left over relative to it: a twist about u_parent, of magnitude psi.
    q_min = shortest_arc_quat(u_child, u_parent)
    psi_quat = quat_angle_deg(quat_multiply(q_comp, quat_conjugate(q_min)))
    psi = spherical_excess_deg(u_parent, reference, u_child)

    half = np.radians(0.5)
    identity_residual = (np.cos(half * theta_comp)
                         - np.cos(half * theta_rel) * np.cos(half * psi))

    return {
        'theta_rel': theta_rel,
        'theta_parent': theta_parent,
        'theta_child': theta_child,
        'theta_sum': theta_parent + theta_child,
        'theta_comp': theta_comp,
        'theta_min_abs': np.minimum(theta_parent, theta_child),
        'psi': psi,
        'psi_abs': np.abs(psi),
        'excess': theta_comp - theta_rel,
        # Verification only, reduced to scalars in joint_stats and never stored per sample.
        '_identity_residual': identity_residual,
        '_psi_residual': np.abs(psi_quat) - np.abs(psi),
    }

# ==============================================================================
# The measured directions
# ==============================================================================

def world_field_directions(parent_plate: PlateTrial, child_plate: PlateTrial, field: str
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u_parent, u_child, valid) for one joint and one field: ground-truth-rotated,
    world-frame, unit measurement directions, and the mask of samples with a direction at all.

    The projection to the joint center is applied HERE for `acc` and not for `mag`, which is
    the asymmetry the whole method rests on: a rigid body transports specific force to another
    of its own points exactly (a + alpha x r + omega x (omega x r)), and transports a magnetic
    field not at all. `acc_unprojected` is the same accelerometer without that transport, kept
    as the control.
    """
    if field == 'acc':
        parent_plate, child_plate = project_pair_to_joint_center(parent_plate, child_plate)

    attribute = 'mag' if field == 'mag' else 'acc'
    parent_world = getattr(parent_plate.get_imu_trace_in_global_frame(), attribute)
    child_world = getattr(child_plate.get_imu_trace_in_global_frame(), attribute)

    u_parent, norm_parent = normalize_rows(parent_world)
    u_child, norm_child = normalize_rows(child_world)
    valid = (norm_parent > MIN_NORM[field]) & (norm_child > MIN_NORM[field])
    return u_parent, u_child, valid


def invariant_angles(parent_plate: PlateTrial, child_plate: PlateTrial,
                     reference_mag: np.ndarray, project: bool
                     ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Facts 5-7 for one joint: the inter-field angle at each sensor, at the global reference
    pair, and the three mismatches. Returns (metrics, valid).

    beta is computed in each sensor's OWN BODY FRAME, from the raw accelerometer and
    magnetometer, with no rotation applied anywhere. That is the whole point — beta is a
    rotation invariant, so this function never touches world_trace.rotations and its output is
    therefore immune to the mocap-alignment floor that inflates every angle in
    correction_angles. The only mocap dependency left is the joint-center offset used by the
    projection, and `project=False` removes even that.

    The global reference pair is (EXPECTED_GRAVITY, reference_mag), so beta_global is a single
    number per trial: the acc-mag angle a per-segment AHRS assumes. Every sensor's deviation
    from it is a mismatch no orientation can remove.
    """
    if project:
        parent_plate, child_plate = project_pair_to_joint_center(parent_plate, child_plate)

    metrics, valid = {}, None
    for role, plate in (('parent', parent_plate), ('child', child_plate)):
        u_acc, norm_acc = normalize_rows(plate.imu_trace.acc)
        u_mag, norm_mag = normalize_rows(plate.imu_trace.mag)
        metrics[f'beta_{role}'] = angle_between_deg(u_acc, u_mag)
        usable = (norm_acc > MIN_NORM['acc']) & (norm_mag > MIN_NORM['mag'])
        valid = usable if valid is None else (valid & usable)

    u_gravity, _ = normalize_rows(EXPECTED_GRAVITY[None, :])
    u_reference, _ = normalize_rows(reference_mag[None, :])
    beta_global = float(angle_between_deg(u_gravity, u_reference)[0])
    metrics['beta_global'] = np.full_like(metrics['beta_parent'], beta_global)

    metrics['dip_rel'] = np.abs(metrics['beta_parent'] - metrics['beta_child'])
    metrics['dip_parent'] = np.abs(metrics['beta_parent'] - beta_global)
    metrics['dip_child'] = np.abs(metrics['beta_child'] - beta_global)
    metrics['dip_sum'] = metrics['dip_parent'] + metrics['dip_child']
    # Fact 5 turns each mismatch into an achievable-and-unbeatable per-vector error of half it.
    metrics['floor_rel'] = 0.5 * metrics['dip_rel']
    metrics['floor_abs'] = 0.5 * np.maximum(metrics['dip_parent'], metrics['dip_child'])
    return metrics, valid


def reference_directions(field: str, u_parent: np.ndarray, u_child: np.ndarray,
                         valid: np.ndarray, subject_field: np.ndarray,
                         torso_field: np.ndarray) -> Dict[str, np.ndarray]:
    """The unit global reference for each variant in REFERENCE_VARIANTS[field].

    `spherical_mean` is computed from the valid samples of THIS joint and field only. That is
    deliberately generous to the absolute route: it lets the reference be re-chosen per joint,
    which no real per-segment AHRS can do (it has one world frame for the whole body), so the
    gap it leaves is a floor rather than an estimate.
    """
    gravity, _ = normalize_rows(EXPECTED_GRAVITY[None, :])
    pooled = np.concatenate([u_parent[valid], u_child[valid]], axis=0)
    spherical_mean = (normalize_rows(pooled.mean(axis=0)[None, :])[0][0]
                      if len(pooled) else np.full(3, np.nan))

    available = {
        'subject_median': normalize_rows(subject_field[None, :])[0][0],
        'torso_median': normalize_rows(torso_field[None, :])[0][0],
        'expected_gravity': gravity[0],
        'spherical_mean': spherical_mean,
    }
    return {variant: available[variant] for variant in REFERENCE_VARIANTS[field]}

# ==============================================================================
# Per-trial tables
# ==============================================================================

def _trim(n_samples: int, fs: float) -> slice:
    """Sample slice with TRIM_S removed from both ends, or everything if the trial is too
    short to trim (a trial shorter than 2*TRIM_S is not a trial, but it should not crash)."""
    margin = int(round(TRIM_S * fs))
    return slice(margin, n_samples - margin) if n_samples > 2 * margin + 1 else slice(None)


def joint_field_angles(plates: Dict[str, PlateTrial], subject_field: np.ndarray,
                       torso_field: np.ndarray, fs: float
                       ) -> Dict[Tuple[str, str, str], Dict[str, np.ndarray]]:
    """{(joint, field, reference): angle arrays} for every joint x field x reference variant
    of one trial, computed on EVERY sample inside the trim.

    Joints missing either sensor are skipped rather than failing the trial: a dropped sensor
    costs that joint, not the other six.
    """
    computed = {}
    for joint, (parent_sensor, child_sensor) in JOINTS.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        for field in FIELDS:
            u_parent, u_child, valid = world_field_directions(
                plates[parent_sensor], plates[child_sensor], field)
            window = _trim(len(u_parent), fs)
            u_parent, u_child, valid = u_parent[window], u_child[window], valid[window]

            references = reference_directions(field, u_parent, u_child, valid,
                                              subject_field, torso_field)
            for variant, reference in references.items():
                if not np.all(np.isfinite(reference)):
                    continue
                # Antiparallel to the reference makes the shortest-arc rotation and psi
                # undefined; excluded here rather than inside correction_angles so the mask
                # is shared by both sensors and both stay on the same sample set.
                usable = valid & (u_parent @ reference > ANTIPARALLEL_COS) \
                              & (u_child @ reference > ANTIPARALLEL_COS)
                angles = correction_angles(u_parent, u_child, reference)
                for key, values in angles.items():
                    angles[key] = np.where(usable, values, np.nan)
                angles['valid'] = usable
                computed[(joint, field, variant)] = angles
    return computed


def joint_invariants(plates: Dict[str, PlateTrial], subject_field: np.ndarray, fs: float
                     ) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    """{(joint, field): invariant arrays} for every joint x acc source, over the same trimmed
    sample window the direction angles use, so the two families are comparable sample for
    sample."""
    computed = {}
    for joint, (parent_sensor, child_sensor) in JOINTS.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        for field in INVARIANT_FIELDS:
            metrics, valid = invariant_angles(plates[parent_sensor], plates[child_sensor],
                                              subject_field, project=field.startswith('acc_projected'))
            window = _trim(len(valid), fs)
            metrics = {key: np.where(valid[window], values[window], np.nan)
                       for key, values in metrics.items()}
            metrics['valid'] = valid[window]
            computed[(joint, field)] = metrics
    return computed


def _samples_frame(computed: Dict[Tuple[str, ...], Dict[str, np.ndarray]],
                   metrics: Sequence[str], keys: Sequence[str]) -> pd.DataFrame:
    """Shared body of the two per-sample tables: decimate by SAMPLE_STRIDE, drop the samples
    with no measurable direction, cast to float32.

    float32 because these are angles in degrees whose interesting range spans 0.01 to 180, so
    seven significant digits is five more than any panel resolves and the table is millions of
    rows long. Degenerate samples are dropped rather than stored as NaN — the *_stats tables
    already record how many there were, and a NaN row in a density panel costs memory to carry
    and then gets filtered anyway."""
    frames = []
    for key, values in computed.items():
        keep = values['valid'][::SAMPLE_STRIDE]
        if not keep.any():
            continue
        frame = dict(zip(keys, key))
        for metric in metrics:
            frame[metric] = values[metric][::SAMPLE_STRIDE][keep].astype(np.float32)
        frames.append(pd.DataFrame(frame))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def invariant_samples_table(computed: Dict[Tuple[str, str], Dict[str, np.ndarray]]
                            ) -> pd.DataFrame:
    """Per-sample inter-field angles and mismatches, decimated by SAMPLE_STRIDE."""
    return _samples_frame(computed, INVARIANT_METRICS, ('joint', 'field'))


def invariant_stats_table(computed: Dict[Tuple[str, str], Dict[str, np.ndarray]]
                          ) -> pd.DataFrame:
    """One row per joint x acc source: quantiles of every invariant metric, the fraction of
    samples on which fact 6 holds, and its worst violation.

    Computed on every sample inside the trim, before the decimation angle_samples applies, so
    this is the table the report and the significance testing quote."""
    rows = []
    for (joint, field), metrics in computed.items():
        valid = metrics['valid']
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        row = {'joint': joint, 'field': field, 'n_samples': n_valid,
               'n_total': int(len(valid)),
               'degenerate_frac': float(1.0 - n_valid / len(valid))}
        for metric in INVARIANT_METRICS:
            values = metrics[metric][valid]
            row[f'{metric}_mean'] = float(np.mean(values))
            for q, value in zip(QUANTILES, np.quantile(values, QUANTILES)):
                row[f'{metric}_p{int(round(q * 100)):02d}'] = float(value)
        # Counted to THEOREM_TOL_DEG, for the reason given in joint_stats_table. Fact 6 is tight
        # exactly when beta_global falls BETWEEN beta_J and beta_K, where dip_sum equals dip_rel
        # identically, and that happens often enough here that a strict <= turns float noise at
        # equality into an apparent violation rate.
        dip_rel, dip_sum = metrics['dip_rel'][valid], metrics['dip_sum'][valid]
        row['frac_rel_below_sum'] = float(np.mean(dip_rel <= dip_sum + THEOREM_TOL_DEG))
        row['max_violation_sum'] = float(np.max(dip_rel - dip_sum))
        # The comparison fact 6 does NOT bound, kept for the same reason theta_min_abs is.
        row['frac_rel_below_min_abs'] = float(np.mean(
            dip_rel <= np.minimum(metrics['dip_parent'][valid], metrics['dip_child'][valid])))
        rows.append(row)
    return pd.DataFrame(rows)


def angle_samples_table(computed: Dict[Tuple[str, str, str], Dict[str, np.ndarray]]
                        ) -> pd.DataFrame:
    """Per-sample direction angles for the PRIMARY reference of each field, decimated by
    SAMPLE_STRIDE. `psi` is carried signed alongside `psi_abs` so the mechanism panel can show
    which side of the plane the reference fell on."""
    primary = {(joint, field): angles
               for (joint, field, variant), angles in computed.items()
               if variant == PRIMARY_REFERENCE[field]}
    return _samples_frame(primary, (*ANGLE_METRICS, 'psi'), ('joint', 'field'))


def traces_table(computed: Dict[Tuple[str, str, str], Dict[str, np.ndarray]],
                 plates: Dict[str, PlateTrial], timestamps: np.ndarray, fs: float,
                 subject_field: np.ndarray, joint: str = EXAMPLE_JOINT) -> pd.DataFrame:
    """Full-rate angles for one joint, every field, primary reference — the time-series panel.

    Full rate rather than strided because this is the one panel that shows the angles AS
    SIGNALS, and the point it makes is temporal: for the accelerometer, theta_parent and
    theta_child swing together with every stride while theta_rel stays flat, which is what
    common-mode cancellation looks like. Decimating that by five would alias the swing.

    `drive` is the physical quantity doing the driving, so the panel can show cause next to
    effect. Its UNIT DEPENDS ON THE FIELD and it is not an angle: m/s^2 for the two
    accelerometer fields (the parent's world-frame linear acceleration, |a_world - g|) and
    magnetometer units for mag (the parent's deviation from the global field,
    |m_world - m_G|). Both are the parent's, not a pair average — this is a qualitative panel
    and averaging the two would blur the very thing it is showing.
    """
    parent_sensor, child_sensor = JOINTS[joint]
    if parent_sensor not in plates or child_sensor not in plates:
        return pd.DataFrame()
    window = _trim(len(timestamps), fs)

    frames = []
    for field in FIELDS:
        key = (joint, field, PRIMARY_REFERENCE[field])
        if key not in computed:
            continue
        angles = computed[key]
        parent_world = plates[parent_sensor].get_imu_trace_in_global_frame()
        drive = (np.linalg.norm(parent_world.mag - subject_field, axis=1) if field == 'mag'
                 else np.linalg.norm(parent_world.acc - EXPECTED_GRAVITY, axis=1))
        frames.append(pd.DataFrame({
            'timestamp': timestamps[window].astype(np.float64),
            'joint': joint,
            'field': field,
            'drive': drive[window].astype(np.float32),
            **{metric: angles[metric].astype(np.float32)
               for metric in ('theta_rel', 'theta_parent', 'theta_child', 'theta_comp')},
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def joint_stats_table(computed: Dict[Tuple[str, str, str], Dict[str, np.ndarray]]
                      ) -> pd.DataFrame:
    """One row per joint x field x reference variant: quantiles of every angle, the win
    fractions, and the theorem's verification residuals.

    Computed on EVERY sample inside the trim, before the decimation that angle_samples
    applies, so this is the table the report quotes and no number in it depends on
    SAMPLE_STRIDE. It is also the table the significance testing blocks on — one cell per
    subject x activity x joint x field — because a Wilcoxon over 8 million correlated samples
    measures the sample count, not the effect.
    """
    rows = []
    for (joint, field, variant), angles in computed.items():
        valid = angles['valid']
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        row = {'joint': joint, 'field': field, 'reference': variant,
               'n_samples': n_valid, 'n_total': int(len(valid)),
               'degenerate_frac': float(1.0 - n_valid / len(valid))}

        for metric in (*ANGLE_METRICS, 'psi'):
            values = angles[metric][valid]
            row[f'{metric}_mean'] = float(np.mean(values))
            for q, value in zip(QUANTILES, np.quantile(values, QUANTILES)):
                row[f'{metric}_p{int(round(q * 100)):02d}'] = float(value)

        # The bounded claims, counted to THEOREM_TOL_DEG rather than to exact <=. The
        # inequalities are tight on coplanar configurations, and this data sits close to that
        # degenerate case constantly, so a strict comparison counts arccos granularity as a
        # violated theorem: measured at 99.9977% "holding" with a worst violation of 3e-14 deg,
        # which reads as a real exception and is not one. The worst violation is reported
        # separately and unrounded, so nothing is hidden by the tolerance.
        theta_rel = angles['theta_rel'][valid]
        row['frac_rel_below_comp'] = float(np.mean(
            theta_rel <= angles['theta_comp'][valid] + THEOREM_TOL_DEG))
        row['frac_rel_below_sum'] = float(np.mean(
            theta_rel <= angles['theta_sum'][valid] + THEOREM_TOL_DEG))
        # The oracle caveat, quantified: how often the relative correction is beaten by the
        # BETTER of the two absolute ones. Not a bounded quantity — see the module docstring.
        row['frac_rel_below_min_abs'] = float(np.mean(theta_rel <= angles['theta_min_abs'][valid]))

        # Theorem verification, as worst-case violations in degrees. Positive means violated.
        row['max_violation_sum'] = float(np.max(theta_rel - angles['theta_sum'][valid]))
        row['max_violation_comp'] = float(np.max(theta_rel - angles['theta_comp'][valid]))
        row['max_identity_residual'] = float(np.max(np.abs(angles['_identity_residual'][valid])))
        row['max_psi_residual'] = float(np.max(np.abs(angles['_psi_residual'][valid])))

        # How well the excess is explained by the spherical area alone, per cell. Spearman
        # rather than Pearson: the closed form is monotone in |psi| but not linear in it.
        excess, psi_abs = angles['excess'][valid], angles['psi_abs'][valid]
        row['spearman_excess_psi'] = (float(spearmanr(excess, psi_abs).statistic)
                                      if n_valid > 2 and np.ptp(psi_abs) > 0 else float('nan'))
        rows.append(row)
    return pd.DataFrame(rows)

# ==============================================================================
# Per-trial driver / grid worker
# ==============================================================================

def compute_trial(plates: Dict[str, PlateTrial], subject_field: np.ndarray
                  ) -> Dict[str, pd.DataFrame]:
    """Everything this experiment computes for one subject/activity, as the three tables."""
    fs = plates['pelvis_imu'].imu_trace.get_sample_frequency()
    timestamps = plates['pelvis_imu'].imu_trace.timestamps
    torso_field = _compute_expected_mag_field(list(plates.values()))

    computed = joint_field_angles(plates, subject_field, torso_field, fs)
    invariants = joint_invariants(plates, subject_field, fs)
    return {
        'angle_samples': angle_samples_table(computed),
        'invariant_samples': invariant_samples_table(invariants),
        'traces': traces_table(computed, plates, timestamps, fs, subject_field),
        'joint_stats': joint_stats_table(computed),
        'invariant_stats': invariant_stats_table(invariants),
    }


def _subject_worker(row_key: str, stage_labels: List[str], shared_state: Dict,
                    activities: Sequence[str]) -> None:
    """One process per subject, both activities inside it — run_tracked_grid(per_cell=False).

    The subject, not the trial, is the unit of work here because the magnetometer's primary
    global reference is the subject's median world field over BOTH activities (see
    sensor_distributions.expected_mag_field). Splitting the activities across processes would
    either compute two different references or require a separate pass to reconcile them, and
    the reference has to be the same in both for a subject's two trials to be comparable.
    """
    subject = row_key
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
            shared_state[(row_key, activity)] = "Skipped (no pelvis)"
            continue
        plates_by_activity[activity] = plates

    if not plates_by_activity:
        shared_state[(row_key, load_stage)] = "Failed (no trials)"
        for activity in activity_stages:
            shared_state[(row_key, activity)] = "Skipped"
        return None

    subject_field = expected_mag_field(plates_by_activity)
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
            for table, df in compute_trial(plates_by_activity[activity], subject_field).items():
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

SUMMARY_COLUMNS = (['family', 'subject', 'activity', 'field', 'metric', 'unit', 'joint',
                    'n_samples', 'mean', 'std', 'min']
                   + [f"p{int(round(q * 100)):02d}" for q in QUANTILES] + ['max'])


def _describe(df: pd.DataFrame, metric: str, keys: List[str]) -> pd.DataFrame:
    grouped = df.groupby(keys, observed=True)[metric]
    stats = grouped.agg(n_samples='size', mean='mean', std='std', min='min', max='max')
    quantiles = grouped.quantile(QUANTILES).unstack()
    quantiles.columns = [f"p{int(round(q * 100)):02d}" for q in quantiles.columns]
    return stats.join(quantiles).reset_index()


def summarize_family(samples: pd.DataFrame, family: str) -> pd.DataFrame:
    """Tidy quantile table per field x metric x joint for one family, WITH MARGINS: rows whose
    `subject`, `activity` or `joint` is the literal string 'all' are the pooled version of the
    rows above them.

    Margins are recomputed from the samples rather than averaged from the per-trial rows — a
    mean of medians is not a median, and trials differ in length. Quantiles rather than
    mean +- sd throughout: every metric here is a non-negative angle with a long tail
    (distortion events for mag, footfalls for acc), so a standard deviation implies a symmetry
    that is not there. It is reported anyway, next to the quantiles, for anyone who wants it.

    The 'direction' family covers the PRIMARY reference only, because that is what
    angle_samples stores. The alternates are summarized from per-trial medians in
    report_reference_sensitivity, which says so where it says it.
    """
    if samples.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    pooled = samples.assign(subject='all', activity='all')
    frames = []
    for metric in FAMILIES[family]:
        if metric not in samples.columns:
            continue
        for source in (samples, pooled):
            frames.append(_describe(source, metric, ['subject', 'activity', 'field', 'joint'])
                          .assign(metric=metric))
            frames.append(_describe(source, metric, ['subject', 'activity', 'field'])
                          .assign(metric=metric, joint='all'))
    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=['subject', 'activity', 'field', 'metric', 'joint'])
    summary['family'] = family
    summary['unit'] = summary['metric'].map(METRIC_UNITS)
    return summary[SUMMARY_COLUMNS]


def summarize(samples: pd.DataFrame, invariants: pd.DataFrame) -> pd.DataFrame:
    """Both families in one table, distinguished by the `family` column (see FAMILIES)."""
    frames = [summarize_family(samples, 'direction'),
              summarize_family(invariants, 'invariant')]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(
        ['family', 'field', 'metric', 'subject', 'activity', 'joint'])

# ==============================================================================
# Console report
# ==============================================================================

def _header(number: int, title: str, subtitle: str) -> None:
    print(f"\n{'=' * 80}\n{number}. {title}\n   {subtitle}\n{'=' * 80}")


def _pooled(summary: pd.DataFrame, field: str, metric: str, joint: str = 'all'
            ) -> Optional[pd.Series]:
    """The pooled (subject='all', activity='all') summary row for one field x metric, or None.

    Not keyed on family: metric names are unique across the two families, so the metric alone
    identifies the row and requiring the caller to also name the family would only be one more
    thing to get wrong.

    Numeric fields are coerced on the way out: the row also carries the family/field/metric/unit
    strings, so without this every quantile read from it is an object and arithmetic on it
    silently produces objects too."""
    rows = summary[(summary['subject'] == 'all') & (summary['activity'] == 'all')
                   & (summary['field'] == field) & (summary['metric'] == metric)
                   & (summary['joint'] == joint)]
    if rows.empty:
        return None
    row = rows.iloc[0].copy()
    numeric = [c for c in row.index
               if c not in ('family', 'subject', 'activity', 'field', 'metric', 'unit', 'joint')]
    row[numeric] = pd.to_numeric(row[numeric], errors='coerce')
    return row


def _primary(stats: pd.DataFrame) -> pd.DataFrame:
    """joint_stats restricted to each field's primary reference."""
    if stats.empty:
        return stats
    keep = stats.apply(lambda r: r['reference'] == PRIMARY_REFERENCE.get(r['field']), axis=1)
    return stats[keep]


def report_theorem(stats: pd.DataFrame) -> None:
    """Facts 1-4 of the module docstring, checked on every sample of every trial.

    These are exact identities, so the only reportable outcome is "violated nowhere, to within
    floating point". A nonzero violation count here is a bug in this file, not a finding about
    the data, and the line that says so is the point of the section."""
    _header(1, "The geometry, verified on every sample",
            "exact identities: anything above the tolerance is a bug, not a result")
    if stats.empty:
        print("   No per-joint statistics on disk.")
        return

    total = int(stats['n_samples'].sum())
    print(f"   {total:,} samples over {len(stats)} joint x field x reference cells "
          f"(every sample inside the trim, before SAMPLE_STRIDE)\n")
    checks = [
        ('theta_rel <= theta_parent + theta_child', 'max_violation_sum', 'deg',
         'geodesic triangle inequality'),
        ('theta_rel <= theta_comp', 'max_violation_comp', 'deg',
         'minimum rotation property'),
        ('cos(theta_comp/2) = cos(theta_rel/2) cos(psi/2)', 'max_identity_residual', '',
         'exact closed form'),
        ('|psi| = |spherical excess(u_J, g, u_K)|', 'max_psi_residual', 'deg',
         'the excess is an area'),
    ]
    print(f"   {'claim':<48s} {'worst residual':>16s}  {'cells over tol':>14s}")
    print(f"   {'-' * 48} {'-' * 16}  {'-' * 14}")
    for claim, column, unit, _ in checks:
        worst = float(stats[column].max())
        over = int((stats[column] > THEOREM_TOL_DEG).sum())
        print(f"   {claim:<48s} {worst:>+11.3e} {unit:<4s} {over:>10d}/{len(stats)}")
    print(f"\n   Tolerance {THEOREM_TOL_DEG:g} deg. Every residual above is floating-point "
          f"noise, which is\n   the only outcome these four can have if the code is right.")

    degenerate = float(stats['degenerate_frac'].max())
    print(f"   Worst per-cell degenerate fraction (no measurable direction): {degenerate:.2e}")


def report_pooled_angles(summary: pd.DataFrame) -> None:
    """The headline numbers: median [IQR] of each angle, per field."""
    _header(2, "How much rotation each residual demands",
            "median [IQR] over every subject, activity, joint and sample, in degrees")
    if summary.empty:
        print("   No summary rows.")
        return

    labels = {'theta_rel': 'RELATIVE  joint error, theta_rel',
              'theta_comp': 'ABSOLUTE  joint error, theta_comp',
              'theta_sum': 'ABSOLUTE  total over both segments, theta_parent + theta_child',
              'theta_parent': '   of which parent, theta_parent',
              'theta_child': '   of which child, theta_child',
              'theta_min_abs': 'ORACLE    better of the two absolute, min(theta_p, theta_c)',
              'psi_abs': 'GEOMETRY  out-of-plane twist, |psi|',
              'excess': 'GEOMETRY  excess, theta_comp - theta_rel'}

    for field in FIELDS:
        rows = {metric: _pooled(summary, field, metric) for metric in labels}
        if rows['theta_rel'] is None:
            continue
        print(f"\n   {FIELD_LABELS[field]}  (reference: {PRIMARY_REFERENCE[field]}, "
              f"n = {int(rows['theta_rel']['n_samples']):,} samples)")
        for metric, label in labels.items():
            row = rows[metric]
            if row is None:
                continue
            print(f"     {label:<58s} {row['p50']:7.2f}  [{row['p25']:6.2f}, {row['p75']:6.2f}]"
                  f"   p95 {row['p95']:7.2f}")
        rel, comp, total = rows['theta_rel'], rows['theta_comp'], rows['theta_sum']
        print(f"     {'-> fact 1 gap, total absolute error (theta_sum - theta_rel)':<58s} "
              f"{total['p50'] - rel['p50']:7.2f} deg")
        print(f"     {'-> fact 2 gap, joint angle (theta_comp - theta_rel)':<58s} "
              f"{comp['p50'] - rel['p50']:7.2f} deg")

    print("\n   READ THE TWO GAPS SEPARATELY. Fact 1 is large and fact 2 is negligible, and "
          "fact 3\n   says why: theta_comp ~ sqrt(theta_rel^2 + psi^2), so the out-of-plane "
          "twist adds in\n   QUADRATURE. With |psi| a few tenths of a degree against a "
          "theta_rel of several, it\n   costs almost nothing. Correcting each sensor to a "
          "global reference is expensive for\n   each segment's absolute orientation and very "
          "nearly free for the joint angle between\n   them, because the two absolute errors "
          "are largely common mode and cancel in the\n   difference. The joint-angle advantage "
          "is in section 9, not here.")


def report_invariant(summary: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Facts 5-7: the part of the error no orientation can remove.

    This is the section the joint-angle claim rests on. Everything above is about which
    rotation a residual demands, and a demanded rotation can cancel between the two segments —
    section 2 shows it very nearly does. An inter-field-angle mismatch cannot cancel, because
    it is invariant to rotation: no orientation estimate for either segment changes it.
    """
    _header(9, "The irreducible part: the inter-field angle",
            "beta = angle(acc, mag), a rotation invariant. Mismatches in it survive any "
            "orientation.")
    if summary.empty:
        print("   No summary rows.")
        return

    labels = {'beta_parent': 'beta at the parent sensor',
              'beta_child': 'beta at the child sensor',
              'beta_global': 'beta of the global reference pair (g, m_G)',
              'dip_rel': 'RELATIVE  mismatch |beta_J - beta_K|',
              'dip_parent': 'ABSOLUTE  mismatch |beta_J - beta_G|',
              'dip_child': 'ABSOLUTE  mismatch |beta_K - beta_G|',
              'dip_sum': 'ABSOLUTE  total, |beta_J - beta_G| + |beta_G - beta_K|',
              'floor_rel': '-> irreducible error, relative route (half of dip_rel)',
              'floor_abs': '-> irreducible error, absolute route (half of the worse)'}

    for field in INVARIANT_FIELDS:
        rows = {metric: _pooled(summary, field, metric) for metric in labels}
        if rows['dip_rel'] is None:
            continue
        print(f"\n   {FIELD_LABELS[field]}  (n = {int(rows['dip_rel']['n_samples']):,} samples)")
        for metric, label in labels.items():
            row = rows[metric]
            if row is None:
                continue
            print(f"     {label:<58s} {row['p50']:7.2f}  [{row['p25']:6.2f}, {row['p75']:6.2f}]"
                  f"   p95 {row['p95']:7.2f}")
        rel, absolute = rows['floor_rel'], rows['floor_abs']
        ratio = absolute['p50'] / rel['p50'] if rel['p50'] > 0 else float('nan')
        print(f"     {'-> absolute route pays':<58s} {absolute['p50'] - rel['p50']:7.2f} deg "
              f"more that cannot be removed ({ratio:.1f}x)")

    if stats.empty:
        return
    print()
    for field in INVARIANT_FIELDS:
        cells = stats[stats['field'] == field]
        if cells.empty:
            continue
        weights = cells['n_samples'].to_numpy()
        holds = np.average(cells['frac_rel_below_sum'].to_numpy(), weights=weights)
        beaten = 1.0 - np.average(cells['frac_rel_below_min_abs'].to_numpy(), weights=weights)
        worst = float(cells['max_violation_sum'].max())
        print(f"   {FIELD_LABELS[field]:<24s} fact 6 holds on {holds:8.4%} of samples "
              f"(worst violation {worst:+.2e} deg);\n   {'':<24s} beaten by the better single "
              f"absolute mismatch on {beaten:6.2%}")


def report_paired_cells(stats: pd.DataFrame) -> None:
    """The blocked paired test, on per-cell medians rather than per-sample values.

    A Wilcoxon over 8 million samples from 22 trials would report the sample count: successive
    samples at 100 Hz are almost perfectly correlated, so its n is fiction and any effect
    clears p < 1e-300. The replicate here is a trial x joint cell, which is the same choice
    plotting/utils.py makes and for the same reason (see its DEFAULT_BLOCK_COLS note). The
    figure's own test blocks one level coarser still, on subject x joint, averaging over side
    and activity."""
    _header(3, "Paired comparison over trial x joint cells",
            "one median per cell; n is cells, not samples")
    primary = _primary(stats)
    if primary.empty:
        print("   No per-joint statistics on disk.")
        return

    for field in FIELDS:
        cells = primary[primary['field'] == field]
        if len(cells) < 3:
            continue
        rel = cells['theta_rel_p50'].to_numpy()
        print(f"\n   {FIELD_LABELS[field]}  (n = {len(cells)} cells)")
        for label, column in (('vs theta_comp (absolute, composed)', 'theta_comp_p50'),
                              ('vs theta_sum  (absolute, both segments)', 'theta_sum_p50'),
                              ('vs min(theta_p, theta_c) (oracle)', 'theta_min_abs_p50')):
            other = cells[column].to_numpy()
            wins = int(np.sum(rel < other))
            try:
                p = float(wilcoxon(rel, other, alternative='less', zero_method='zsplit').pvalue)
            except ValueError:  # all differences zero
                p = 1.0
            print(f"     {label:<42s} relative smaller in {wins:3d}/{len(cells)} cells, "
                  f"median diff {np.median(rel - other):+7.2f} deg, p = {p:.2e}")


def report_by_joint(summary: pd.DataFrame) -> None:
    """Median theta_rel vs theta_comp per joint, so the claim can be checked where it is
    weakest rather than only where it is pooled."""
    _header(4, "By joint", "pooled median theta_rel vs theta_comp, degrees")
    if summary.empty:
        print("   No summary rows.")
        return
    for field in FIELDS:
        print(f"\n   {FIELD_LABELS[field]}")
        print(f"     {'joint':<10s} {'theta_rel':>10s} {'theta_comp':>11s} {'gap':>8s} {'ratio':>7s}")
        for joint in JOINTS:
            rel = _pooled(summary, field, 'theta_rel', joint)
            comp = _pooled(summary, field, 'theta_comp', joint)
            if rel is None or comp is None:
                continue
            ratio = comp['p50'] / rel['p50'] if rel['p50'] > 0 else float('nan')
            print(f"     {joint:<10s} {rel['p50']:10.2f} {comp['p50']:11.2f} "
                  f"{comp['p50'] - rel['p50']:8.2f} {ratio:6.1f}x")


def report_projection_role(summary: pd.DataFrame) -> None:
    """The accelerometer control: what the joint-center projection buys the relative residual.

    This is the section that separates the accelerometer result from the magnetometer one. The
    absolute route cannot benefit from the projection at all — its reference is gravity either
    way — so any change in theta_comp between the two acc fields is second-order, while
    theta_rel should drop substantially once both sensors are measuring the same point."""
    _header(5, "What the joint-center projection does for the accelerometer",
            "theta_rel with and without rigid-body projection to the shared joint center")
    rows = {field: {metric: _pooled(summary, field, metric)
                    for metric in ('theta_rel', 'theta_comp')}
            for field in ('acc', 'acc_unprojected')}
    if any(r['theta_rel'] is None for r in rows.values()):
        print("   Need both acc and acc_unprojected on disk.")
        return
    print(f"     {'field':<28s} {'theta_rel':>10s} {'theta_comp':>11s}")
    for field in ('acc_unprojected', 'acc'):
        print(f"     {FIELD_LABELS[field]:<28s} {rows[field]['theta_rel']['p50']:10.2f} "
              f"{rows[field]['theta_comp']['p50']:11.2f}")
    drop = rows['acc_unprojected']['theta_rel']['p50'] - rows['acc']['theta_rel']['p50']
    print(f"\n   Projection removes {drop:+.2f} deg of median theta_rel. theta_comp barely "
          f"moves, as it must:\n   the absolute route's reference is gravity whether or not "
          f"the reading was transported.")


def report_oracle_caveat(stats: pd.DataFrame) -> None:
    """How often the relative correction loses to the better of the two absolute ones.

    Reported because it is the one comparison in this analysis that is NOT bounded, and
    leaving it out would make the guarantee look broader than it is. It is also not an
    implementable method: nothing tells a per-segment filter which of its two sensors is the
    one sitting closer to the global field."""
    _header(6, "The unbounded comparison, counted",
            "fraction of samples where the relative correction is beaten by the better "
            "single absolute one")
    primary = _primary(stats)
    if primary.empty:
        print("   No per-joint statistics on disk.")
        return
    for field in FIELDS:
        cells = primary[primary['field'] == field]
        if cells.empty:
            continue
        weights = cells['n_samples'].to_numpy()
        beaten = 1.0 - np.average(cells['frac_rel_below_min_abs'].to_numpy(), weights=weights)
        bounded_comp = np.average(cells['frac_rel_below_comp'].to_numpy(), weights=weights)
        bounded_sum = np.average(cells['frac_rel_below_sum'].to_numpy(), weights=weights)
        print(f"   {FIELD_LABELS[field]:<28s} beaten by min(theta_p, theta_c) on "
              f"{beaten:6.2%} of samples")
        print(f"   {'':<28s} bounded claims hold on {bounded_comp:7.3%} (comp), "
              f"{bounded_sum:7.3%} (sum)")


def report_reference_sensitivity(stats: pd.DataFrame) -> None:
    """The gap under every global reference variant, including the per-joint steelman.

    Fact 2 holds pointwise for every reference, so no variant can flip the sign of the gap and
    none does. What varies is the SIZE, and that is what a reader deciding whether the effect
    matters needs. Quoted as the median over trial x joint cells of each cell's median, which
    is stated rather than smuggled: angle_samples only carries the primary reference, so a
    properly pooled quantile is not available for the alternates."""
    _header(7, "Sensitivity to the choice of global reference",
            "median over trial x joint cells of each cell's median, degrees")
    if stats.empty:
        print("   No per-joint statistics on disk.")
        return
    print(f"     {'field':<28s} {'reference':<17s} {'theta_rel':>10s} {'theta_comp':>11s} "
          f"{'gap':>8s} {'cells':>6s}")
    for field in FIELDS:
        for variant in REFERENCE_VARIANTS[field]:
            cells = stats[(stats['field'] == field) & (stats['reference'] == variant)]
            if cells.empty:
                continue
            rel = float(cells['theta_rel_p50'].median())
            comp = float(cells['theta_comp_p50'].median())
            marker = ' *' if variant == PRIMARY_REFERENCE[field] else '  '
            print(f"     {FIELD_LABELS[field]:<28s} {variant + marker:<17s} {rel:10.2f} "
                  f"{comp:11.2f} {comp - rel:8.2f} {len(cells):6d}")
    print("\n   * primary, the one the summary and the figure quote. theta_rel is independent "
          "of the\n   reference by construction, so it repeats down each field's block — that "
          "it does is\n   itself a check that the variants differ only in g.")


def report_mechanism(stats: pd.DataFrame, samples: pd.DataFrame) -> None:
    """Whether the excess really is the spherical area, as a correlation rather than as an
    identity check.

    report_theorem already verified the closed form exactly, which proves the excess is a
    FUNCTION of (theta_rel, psi). This asks the empirical question that does not follow from
    it: over the actual distribution of configurations in this dataset, is |psi| what the
    excess tracks? If the answer were no, the geometry would be correct and irrelevant."""
    _header(8, "Is the out-of-plane twist what drives the excess?",
            "Spearman(excess, |psi|) per cell, and the excess by |psi| decile")
    primary = _primary(stats)
    if not primary.empty:
        for field in FIELDS:
            cells = primary[primary['field'] == field]['spearman_excess_psi'].dropna()
            if cells.empty:
                continue
            print(f"   {FIELD_LABELS[field]:<28s} Spearman median {cells.median():+.3f} "
                  f"[{cells.quantile(0.05):+.3f}, {cells.quantile(0.95):+.3f}] "
                  f"over {len(cells)} cells")
    if samples.empty:
        return
    print()
    for field in FIELDS:
        subset = samples[samples['field'] == field]
        if subset.empty:
            continue
        deciles = pd.qcut(subset['psi_abs'], 10, duplicates='drop')
        grouped = subset.groupby(deciles, observed=True)[['psi_abs', 'excess']].median()
        summary = "  ".join(f"{row.psi_abs:.2f}->{row.excess:.2f}"
                            for row in grouped.itertuples())
        print(f"   {FIELD_LABELS[field]:<28s} |psi| -> excess, by decile (deg):\n"
              f"     {summary}")


def report_invariant_by_joint(summary: pd.DataFrame) -> None:
    """The irreducible floors per joint, for the primary acc source.

    Separate from report_by_joint because this is the comparison the paper's joint-angle claim
    quotes, and burying it under the direction-family table would put a 0.06 deg gap and a
    several-degree one in the same column."""
    _header(10, "The irreducible floors, by joint",
            "half the inter-field-angle mismatch, degrees, projected accelerometer")
    field = INVARIANT_FIELDS[0]
    if summary.empty or _pooled(summary, field, 'floor_rel') is None:
        print("   No invariant summary rows.")
        return
    print(f"     {'joint':<10s} {'relative':>10s} {'absolute':>10s} {'gap':>8s} {'ratio':>7s}")
    for joint in JOINTS:
        rel = _pooled(summary, field, 'floor_rel', joint)
        absolute = _pooled(summary, field, 'floor_abs', joint)
        if rel is None or absolute is None:
            continue
        ratio = absolute['p50'] / rel['p50'] if rel['p50'] > 0 else float('nan')
        print(f"     {joint:<10s} {rel['p50']:10.2f} {absolute['p50']:10.2f} "
              f"{absolute['p50'] - rel['p50']:8.2f} {ratio:6.1f}x")


def print_report(summary: pd.DataFrame, stats: pd.DataFrame, samples: pd.DataFrame,
                 invariant_stats: pd.DataFrame) -> None:
    report_theorem(stats)
    report_pooled_angles(summary)
    report_paired_cells(stats)
    report_by_joint(summary)
    report_projection_role(summary)
    report_oracle_caveat(stats)
    report_reference_sensitivity(stats)
    report_mechanism(stats, samples)
    report_invariant(summary, invariant_stats)
    report_invariant_by_joint(summary)

# ==============================================================================
# CLI / ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--activities', nargs='+', default=ACTIVITIES)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--report-only', action='store_true',
                        help="Skip recomputation and rebuild the summary + report from the "
                             "per-trial tables already on disk. The pooled summary always "
                             "covers every subject/activity present on disk, not just those "
                             "passed to --subjects, so a partial run does not silently "
                             "narrow it.")
    args = parser.parse_args()

    if not args.report_only:
        run_tracked_grid(args.subjects, ['Subject'], ['load'] + list(args.activities),
                         partial(_subject_worker, activities=args.activities),
                         args.workers, title="RELATIVE VS ABSOLUTE CORRECTION")

    print("\nLoading per-trial tables...")
    samples = load_trial_table('angle_samples')
    invariants = load_trial_table('invariant_samples')
    stats = load_trial_table('joint_stats')
    invariant_stats = load_trial_table('invariant_stats')
    if samples.empty:
        print(f"No results found under {EXPERIMENT_DIR}. Run without --report-only first.")
        return

    trials = set(map(tuple, samples[['subject', 'activity']].drop_duplicates().to_numpy()))
    print(f"Found {len(trials)} trial(s) across {len({s for s, _ in trials})} subject(s), "
          f"{len(samples):,} + {len(invariants):,} stored samples.")

    summary = summarize(samples, invariants)
    path = paths.ensure_parent(paths.statistics_path(EXPERIMENT_NAME))
    summary.to_parquet(path, engine='pyarrow', index=False)
    paths.write_manifest(path, constants=analysis_constants(), experiment=EXPERIMENT_NAME,
                         n_rows=len(summary),
                         subjects=sorted({s for s, _ in trials}),
                         activities=sorted({a for _, a in trials}))
    print(f"Saved summary to {path}")

    print_report(summary, stats, samples, invariant_stats)
    print(f"\nPer-trial tables under {EXPERIMENT_DIR}")
    print("Figures: python -m plotting.relative_vs_absolute")


if __name__ == '__main__':
    main()
