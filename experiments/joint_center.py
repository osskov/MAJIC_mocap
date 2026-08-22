"""
How well the joint center is reconstructed, per dataset — the input every projection in this
repo depends on and none of them measures.

Two segments spanning a joint share one physical point. If both poses are rigid and the joint
is a ball joint at a fixed location, then at every sample

    p_parent(t) + R_parent(t) r_parent  =  p_child(t) + R_child(t) r_child                (1)

with r_parent and r_child the constant sensor-to-joint-center offsets in each segment's own
frame. `WorldTrace.get_joint_center` solves (1) in least squares for both offsets at once — over the
frames where both traces are valid, or over any mask this module hands it — and
those six numbers are what `IMUTrace.project_acc` moves the accelerometer along and what the
magnetometer projection would extrapolate the field across. Everything downstream inherits
whatever error is in them.

The OFFSETS are not an artifact — `joint_offsets` recomputes them, at ~17 ms per joint, because
`get_joint_center` now masks by `valid` itself and so every caller gets the same deterministic
answer without a file to keep in sync. What this file writes is the ANALYSIS: how good those
offsets are, how well determined, how repeatable, and what an error in them costs.

    python -m experiments.joint_center --dataset alborno
    python -m experiments.joint_center --dataset imove
    python -m experiments.joint_center --dataset alborno --report-only

THE OBJECTIVE IS CLOSURE, and the report leads with it. What a relative filter needs is that the
two segments construct the SAME point — it compares two estimates of one physical point rather
than locating that point anatomically — so the measure of an offset pair is

    RMS_t | (p_parent + R_parent c_parent) - (p_child + R_child c_child) |

scored OUT OF SAMPLE. Any six numbers can be scored on any trial, so a mocap fit, an anatomical
midpoint and a gyro-only Seel fit all land in one column of millimetres. The headline table does
exactly that; every numbered section below is evidence about some component of it.

Closure is blind to a simultaneous slide of both offsets along the joint axis. That is the right
blindness rather than a defect, because the downstream shares it: two segments constructing the
same point are both computing the acceleration of that one point whether or not it is
anatomically the joint. The metric's null space and the application's null space coincide.

`ambiguity_radii` is what makes that quantitative — the displacement costing one millimetre of
closure, in millimetres, per direction. It reconciles the two ways of scoring an offset error, and
resolves what otherwise looks like a contradiction: two fits 30 mm apart scoring within 2 mm of
each other, because the difference lies along a direction 12 mm wide.

What is measured
----------------
  1. FIT RESIDUAL — exactly the separation of the two segments' implied centres, so it reads in
     metres as "how far apart they think the joint is". NOT sensor noise: it is how badly a
     fixed-centre ball joint describes the pair. Soft tissue, a translating knee, marker
     reconstruction error and residual misalignment all land in it.

  2. AMBIGUITY, in closed form and in millimetres. Stacking (1) over N samples gives a 3N x 6
     design matrix
     M = [R_parent, -R_child]. Because both blocks are orthonormal, its Gram matrix is

         M^T M = [[N I, -S], [-S^T, N I]],        S = sum_t R_parent(t)^T R_child(t)

     whose eigenvalues are exactly N +- sigma_i(S), eigenvectors (u_i, +-v_i) from the SVD of S.
     Verified to 1e-2 in 2e5. Since RSS(x* + d) = RSS(x*) + d^T G d exactly, the displacement
     along eigenvector i costing one millimetre of closure is sqrt(n/lambda_i) mm — the condition
     number in the units the offsets are quoted in. Three consequences the report uses:

       * Conditioning is set entirely by the RELATIVE ROTATION. `excitation` = 1 - sigma_max/N is
         0 when the pair never moves relative to each other, and grows as the joint sweeps SO(3).
       * The worst-determined mode is a SIMULTANEOUS slide of both offsets — the assumed centre
         moves and both frames absorb it, so the residual stays small while the offsets are wrong.
       * A pure hinge has R_rel(t) n = n, so n is a singular vector with sigma = N exactly and the
         offset along the hinge axis is unobservable however long the trial. The classic
         degeneracy, falling out of the algebra rather than asserted.

  3. WHAT THE valid MASK IS WORTH. Outside `valid` the world trace holds a constant padded pose
     — thousands of identical rows the least squares is dragged toward satisfying exactly, and
     40%+ of frames on some Al Borno trials. `get_joint_center` now masks by default; this
     section reproduces the unmasked fit to measure what that was worth (7 mm mean, 90 mm worst).

  4. GENERALIZATION. Six free parameters always reduce a residual, so the in-sample number says
     little alone. Refit on the first half of the valid frames, scored on the second.

  5. BETWEEN-TRIAL STABILITY. Same subject and joint, fitted independently per trial. The offset
     is anatomy plus mounting, so it should be near-constant; the spread is the honest error bar
     on any single trial's fit, and the one to quote.

  6. WHAT AN OFFSET ERROR COSTS, in the currency the next experiment works in:
     |da| <= (|alpha| + |omega|^2)|dr|, with the gain measured per trial rather than nominal.
     Turns "good to 8 mm" into "the projection inherits 0.6 m/s^2 from them".

  7. ONE FIT PER SUBJECT instead of one per trial, and the cross-validation that says whether the
     offsets transfer. They do not: cross/own runs 1.45 on Al Borno, 1.9-5.8 on IMoVE. Whether
     that forbids pooling depends on the cause, which section 8 takes up.

  8. SEGMENT LENGTHS, plus the direct test of whether the sensor moved. A calibration difference
     scales every length by one factor, which the length alone catches. A RE-MOUNT it cannot
     catch, contrary to what this section used to claim: length sees only the differential of the
     two centres' shifts and is second-order blind to the transverse part, so the common component
     is measured directly instead. On Al Borno it is absent — the centres wander independently and
     the mounting itself moves ~4.5 mm — so the between-trial spread is fit error, not re-taping.

The inertial-only estimator — the same six numbers from gyro and accelerometer, no mocap — is
`experiments/inertial_joint_center.py`, which reads this module's fit as its reference.

Outputs, all under results/experiments/joint_center/<dataset>/:

    <subject>/<trial>/joint_fits.parquet     per-joint scalars: residual, conditioning, cost
    <subject>/<trial>/fit_samples.parquet    per-sample residual, strided, for the figures

plus results/statistics/joint_center_<dataset>_statistics.parquet.

Nothing here runs a filter: this measures the mocap and the joint model alone.
"""
import argparse
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import paths
from experiments.experiment_utils import load_trial, pipeline_constants, run_tracked_grid
from experiments.global_assumptions import (DATASETS, DatasetSpec, build_name, canonical_joint,
                                            enumerate_trials, get_dataset, segment_joint_roles,
                                            subjects_of)
from src.toolchest import trial_io
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import MIN_JOINT_CENTER_FRAMES, WorldTrace

EXPERIMENT_NAME = "joint_center"
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Minimum valid frames before this experiment attempts a fit. STRICTER than the library floor in
# WorldTrace (MIN_JOINT_CENTER_FRAMES, currently 50), and deliberately so: that one refuses a fit
# that cannot be done at all, while this one refuses a fit whose numbers should not be reported.
# Everything downstream here — the quantile summary, the holdout split, the per-trial spread —
# wants far more than the minimum that makes the least squares solvable.
MIN_FIT_FRAMES = 500
assert MIN_FIT_FRAMES >= MIN_JOINT_CENTER_FRAMES, "the experiment's bar cannot be below the library's"

# Fraction of the valid frames used to FIT in the holdout check; the rest are scored. Split in
# time rather than at random, deliberately: a random split leaves the two halves sharing nearly
# the same poses, which is not a test of anything, whereas a subject in the second half of a
# trial is often doing a different task.
HOLDOUT_FIT_FRACTION = 0.5

# Sample stride for the per-sample residual table only. Every scalar in joint_fits is computed on
# the full valid record.
SAMPLE_STRIDE = 10


QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

TRIAL_TABLES = ('joint_fits', 'fit_samples', 'normal_equations', 'learning_curve',
                'landmarks')

OFFSET_COLUMNS = ('parent_x', 'parent_y', 'parent_z', 'child_x', 'child_y', 'child_z')
NE_GRAM_COLUMNS = tuple(f"g{i}{j}" for i in range(6) for j in range(6))
NE_RHS_COLUMNS = tuple(f"r{i}" for i in range(6))


def analysis_constants(dataset: str) -> Dict[str, object]:
    return {
        **pipeline_constants(),
        'dataset': dataset,
        'min_fit_frames': MIN_FIT_FRAMES,
        'holdout_fit_fraction': HOLDOUT_FIT_FRACTION,
        'sample_stride': SAMPLE_STRIDE,
    }

# ==============================================================================
# Paths / IO
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
        frames.append(pd.read_parquet(path, engine='pyarrow', columns=list(columns) if columns
                                      else None).assign(subject=subject, trial=trial))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out['dataset'] = dataset
    return out





def joint_offsets(plates: Dict[str, PlateTrial], spec: DatasetSpec,
                  min_frames: int = MIN_FIT_FRAMES) -> Dict[str, Dict[str, np.ndarray]]:
    """{joint: {'parent': r_parent, 'child': r_child}} for a trial, COMPUTED FRESH.

    Recomputed rather than read back from a parquet, and the reason the parquet went away is worth
    recording. It existed because `WorldTrace.get_joint_center` used to fit over padded frames, so
    every caller refitting independently got the wrong answer and a shared artifact was the way to
    ensure one right one. That method now masks by `valid` itself, which puts the decision in one
    place for every caller automatically — and the fit is a deterministic linear least squares at
    ~17 ms per joint, so recomputing is both cheap and bit-identical.

    Joints whose pair is absent, or too poorly covered to fit, are simply missing from the result;
    callers iterate what they get rather than assuming a full set.
    """
    offsets: Dict[str, Dict[str, np.ndarray]] = {}
    for joint, (parent_sensor, child_sensor) in spec.joints.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        parent, child = plates[parent_sensor], plates[child_sensor]
        n = min(len(parent), len(child))
        valid = np.asarray(parent.valid)[:n] & np.asarray(child.valid)[:n]
        if valid.sum() < min_frames:
            continue
        r_parent, r_child, _ = parent.world_trace.get_joint_center(
            child.world_trace, valid, min_frames=min_frames)
        offsets[joint] = {'parent': r_parent, 'child': r_child}
    return offsets


# ==============================================================================
# The fit
# ==============================================================================

def normal_equations(parent: WorldTrace, child: WorldTrace,
                     mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """The 6x6 normal equations for equation (1) over `mask`, as {gram, rhs, sum_bb, n}.

    A COMPLETE SUFFICIENT STATISTIC for the fit: 44 numbers per joint, ADDITIVE across trials,
    and any candidate offset's residual reads back out of them without the samples. So combining
    a subject's trials is a sum rather than a reload, cross-validation is free via
    `residual_rms_from` (RSS(x) = x^T G x - 2 x^T r + sum_bb), and memory is O(1) in trial length
    — which is what makes the subject-level pass possible on IMoVE's 450 k-sample trials.

    Assembled directly rather than from the closed form `fit_conditioning` uses, so the two are
    independent and the test that they agree means something.
    """
    n = min(len(parent), len(child))
    mask = np.ones(n, dtype=bool) if mask is None else np.asarray(mask)[:n]
    # float64 THROUGHOUT, and this path specifically needs it. The trial parquet stores poses as
    # float32 (see trial_io), and forming normal equations SQUARES the condition number — which
    # here is already 100-1000 — so a sum of 10^5 float32 terms loses real precision. Measured on
    # Subject01/walking across all seven joints: the normal-equations solve in float32 is off by
    # up to 2.8 mm against float64, a third of that joint's between-trial spread.
    #
    # The DIRECT least squares in `WorldTrace.get_joint_center` is not affected (0.0000 mm)
    # because LAPACK's QR path is backward-stable and never forms the Gram matrix. So this is a
    # cost of the sufficient-statistic representation, not a latent bug in the shipped fit — and
    # it is worth the cost, because that representation is what makes combining trials free.
    rotations_parent = parent.rotations[:n][mask].astype(np.float64)
    rotations_child = child.rotations[:n][mask].astype(np.float64)
    b = (child.positions[:n] - parent.positions[:n])[mask].astype(np.float64)

    design = np.concatenate([rotations_parent, -rotations_child], axis=2)  # (n, 3, 6)
    return {
        'gram': np.einsum('nij,nik->jk', design, design),
        'rhs': np.einsum('nij,ni->j', design, b),
        'sum_bb': float(np.sum(b ** 2)),
        'n': int(mask.sum()),
    }


def combine_normal_equations(parts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Sum of several trials' normal equations, i.e. the joint fit over all of them."""
    return {'gram': sum(p['gram'] for p in parts), 'rhs': sum(p['rhs'] for p in parts),
            'sum_bb': float(sum(p['sum_bb'] for p in parts)),
            'n': int(sum(p['n'] for p in parts))}


def solve_offsets(equations: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """The six offsets, as one stacked vector. None when the system is singular.

    `lstsq` rather than `solve`: the Gram matrix is near-singular whenever the pair is poorly
    excited (its smallest eigenvalue is N - sigma_max(S), which goes to zero for a rigid pair),
    and a minimum-norm answer there is far better than either a LinAlgError or the enormous
    numbers an exact solve returns.
    """
    if equations['n'] < 2:
        return None
    solution, *_ = np.linalg.lstsq(equations['gram'], equations['rhs'], rcond=None)
    return solution


def ambiguity_radii(equations: Dict[str, np.ndarray], tolerance_mm: float = 1.0
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """How far the offsets may move for `tolerance_mm` of extra closure, per direction.

    THE CONDITION NUMBER IN MILLIMETRES, and the quantity that reconciles the two ways of
    measuring an offset error. Starting from the optimum the residual is exactly quadratic,

        RSS(x* + d) = RSS(x*) + d^T G d

    so closure^2 rises by d^T G d / n. Along the unit eigenvector with eigenvalue lambda a
    displacement r therefore costs r^2 lambda / n, and the displacement that costs exactly
    `tolerance_mm` is

        r = tolerance_mm * sqrt(n / lambda)

    directly in millimetres, because G is built from rotation blocks and is dimensionless.

    This is what explains an observation that otherwise looks contradictory: two fits whose
    offsets differ by 30 mm scoring within 2 mm of each other on closure. The difference lies
    along a direction whose radius is tens of millimetres, so it is nearly free. A direction
    with a large radius is one the data never pinned — and, for the relative use this metric
    serves, one that does not need pinning.

    Returned WORST FIRST (largest radius, smallest eigenvalue), with the matching eigenvectors
    as columns, so `radii[0]` and `vectors[:, 0]` are the ambiguous mode. `eigh` returns
    eigenvalues ASCENDING, and the radius goes as 1/sqrt(lambda), so that order is already
    worst-first and must not be flipped again.
    """
    values, vectors = np.linalg.eigh(equations['gram'])
    n = max(equations['n'], 1)
    radii = tolerance_mm * np.sqrt(n / np.clip(values, 1e-12, None))
    return radii, vectors


def closure_cost(equations: Dict[str, np.ndarray], displacement: np.ndarray) -> float:
    """The closure a displacement from the OPTIMUM would add, in mm, without the samples.

    sqrt(d^T G d / n): the exact quadratic above, and the inverse of `ambiguity_radii`.
    """
    if equations['n'] < 1:
        return float('nan')
    quadratic = float(displacement @ equations['gram'] @ displacement)
    return float(np.sqrt(max(quadratic, 0.0) / equations['n']) * 1000.0)


def residual_rms_from(equations: Dict[str, np.ndarray], offsets: np.ndarray) -> float:
    """RMS residual of `offsets` on the data behind `equations`, in mm, without the samples.

    RSS(x) = x^T G x - 2 x^T r + sum_bb, expanded from sum_t |M_t x - b_t|^2. Clamped at zero
    before the square root: the three terms are large and nearly cancel at the optimum, so
    float error can take the difference a few ULPs negative for a perfectly good fit.
    """
    if equations['n'] < 1 or offsets is None:
        return float('nan')
    rss = (float(offsets @ equations['gram'] @ offsets)
           - 2.0 * float(offsets @ equations['rhs']) + equations['sum_bb'])
    return float(np.sqrt(max(rss, 0.0) / equations['n']) * 1000.0)


def fit_conditioning(parent: WorldTrace, child: WorldTrace, mask: np.ndarray,
                     hinge_alignment: bool = False) -> Dict[str, object]:
    """How well-determined the six offsets are, from the closed form in the module docstring.

    The Gram matrix of the design is [[N I, -S], [-S^T, N I]] with S = sum_t R_p^T R_c, so its
    eigenvalues are N +- sigma_i(S) and nothing has to be assembled at 3N x 6 to know them. That
    is not only cheaper — it is what makes the result interpretable, because it says the
    conditioning depends on the RELATIVE ROTATION alone and not on the positions, the offsets or
    the trial length beyond the count.

    `excitation` = 1 - sigma_max/N in [0, 1]: 0 when the two segments never rotate relative to
    each other, in which case the offsets are formally unresolved and any reported value is
    whatever the least squares happened to land on. `condition_number` is the ratio of the
    extreme eigenvalues, which is the factor by which mocap noise is amplified into the
    worst-determined mode.

    `worst_parent` / `worst_child` are that mode: the offsets slide TOGETHER along these two
    directions with no penalty. For a hinge they are the hinge axis, and `hinge_alignment`
    reports the measured agreement with `get_primary_joint_axis` rather than assuming it.
    """
    n_samples = int(mask.sum())
    result = {'n_fit_frames': n_samples, 'excitation': np.nan, 'condition_number': np.nan,
              'sigma_max_ratio': np.nan, 'hinge_alignment': np.nan}
    for i, axis in enumerate('xyz'):
        result[f'worst_parent_{axis}'] = np.nan
        result[f'worst_child_{axis}'] = np.nan
    if n_samples < MIN_FIT_FRAMES:
        return result

    n = min(len(parent), len(child))
    mask = np.asarray(mask)[:n]
    cross = np.einsum('nji,njk->ik', parent.rotations[:n][mask].astype(np.float64),
                      child.rotations[:n][mask].astype(np.float64))
    left, singular, right_t = np.linalg.svd(cross)

    ratio = float(singular[0] / n_samples)
    result['sigma_max_ratio'] = ratio
    result['excitation'] = float(1.0 - ratio)
    smallest = n_samples - singular[0]
    result['condition_number'] = (float((n_samples + singular[0]) / smallest)
                                  if smallest > 1e-9 else np.inf)

    worst_parent, worst_child = left[:, 0], right_t[0]
    for i, axis in enumerate('xyz'):
        result[f'worst_parent_{axis}'] = float(worst_parent[i])
        result[f'worst_child_{axis}'] = float(worst_child[i])

    # OFF BY DEFAULT: `get_primary_joint_axis` runs its own optimization and costs ~920 ms per
    # joint against the 17 ms of the fit it annotates — 97% of this experiment's runtime for one
    # cross-check column. It has already served its purpose (measured alignment runs 0.34-0.78,
    # i.e. the degeneracy is NOT cleanly the hinge axis), so it is opt-in via --hinge-alignment.
    if not hinge_alignment:
        return result
    try:
        hinge_axis, _ = parent.get_primary_joint_axis(child)
        result['hinge_alignment'] = float(abs(np.dot(worst_parent, hinge_axis)))
    except Exception:
        # get_primary_joint_axis solves its own optimization and can fail on a pair that barely
        # moves — exactly the pairs whose conditioning this function is reporting as bad. A
        # missing cross-check is not a reason to lose the conditioning numbers.
        pass
    return result


def holdout_residual(parent: WorldTrace, child: WorldTrace, mask: np.ndarray
                     ) -> Dict[str, float]:
    """In-sample and held-out residual RMS, refitting on the first half of the valid frames.

    Six free parameters always reduce a residual, so the in-sample number on its own cannot say
    whether the fixed-centre model describes the joint. Split in TIME rather than at random: a
    random split leaves both halves covering the same poses, which tests nothing, while the
    second half of a trial is often a different task.
    """
    result = {'residual_in_sample_mm': np.nan, 'residual_holdout_mm': np.nan}
    indices = np.flatnonzero(mask)
    if len(indices) < 2 * MIN_FIT_FRAMES:
        return result
    split = int(len(indices) * HOLDOUT_FIT_FRACTION)
    fit_mask = np.zeros(len(mask), dtype=bool)
    fit_mask[indices[:split]] = True
    test_mask = np.zeros(len(mask), dtype=bool)
    test_mask[indices[split:]] = True

    _, _, residual = parent.get_joint_center(child, fit_mask, min_frames=1)
    norms = np.linalg.norm(residual, axis=1)
    result['residual_in_sample_mm'] = float(np.sqrt(np.mean(norms[fit_mask] ** 2)) * 1000.0)
    result['residual_holdout_mm'] = float(np.sqrt(np.mean(norms[test_mask] ** 2)) * 1000.0)
    return result



# Samples fed to the optimizer. Six parameters need far fewer than a full trial, and the
# Jacobian is rebuilt on every iteration.
# Excitation gate. Seel's residual carries no information where the segment is not accelerating:
# A = [alpha]x + omega omega^T - |omega|^2 I goes to zero, so those samples contribute rows of
# zeros to the Jacobian and only dilute the fit. Keep the most excited half.



def projection_gain(plate: PlateTrial, mask: np.ndarray) -> Dict[str, float]:
    """How many m/s^2 of projected-acceleration error one metre of offset error buys.

    `project_acc` adds alpha x r + omega x (omega x r), both linear in r, so an offset error dr
    propagates as (alpha x dr + omega x (omega x dr)) and its magnitude is bounded by
    (|alpha| + |omega|^2)|dr|. Reported as the median and p95 of that gain over the trial, from
    the MEASURED omega and its derivative rather than a nominal — the whole point is to turn a
    millimetre figure into the acceleration error the next experiment inherits.

    An upper bound, not an estimate: the two terms are perpendicular to different things and
    partially cancel for a given dr. It is the right side to err on for a budget.
    """
    trace = plate.imu_trace
    n = min(len(mask), len(trace))
    selected = np.asarray(mask)[:n]
    if selected.sum() < 2:
        return {'gain_median': np.nan, 'gain_p95': np.nan}
    omega = trace.gyro[:n][selected]
    alpha = trace._finite_difference_gyros('backward')[:n][selected]
    gain = np.linalg.norm(alpha, axis=1) + np.sum(omega ** 2, axis=1)
    return {'gain_median': float(np.median(gain)), 'gain_p95': float(np.percentile(gain, 95))}

# ==============================================================================
# Precision, convergence, migration, symmetry
# ==============================================================================

def offset_precision(equations: Dict[str, np.ndarray], residual: np.ndarray,
                     mask: np.ndarray) -> Dict[str, float]:
    """Standard errors on the six offsets, from cov = sigma^2 (M^T M)^-1.

    Both ingredients were already computed and never combined: sigma from the residual, M^T M from
    the normal equations. A scalar condition number says the fit is ill-posed; this says WHERE, in
    millimetres per axis.

    THE NAIVE VERSION IS OVER-CONFIDENT and the correction matters. sigma^2 (M^T M)^-1 assumes
    independent errors, but the residual is a smooth function of a walking subject's pose and
    consecutive frames are almost identical — so the effective sample size is far below N. The
    integral autocorrelation time of the residual is measured and the covariance inflated by
    N / N_eff, which is the standard fix and is typically a factor of tens here. Both are reported
    so the size of the correction is visible rather than buried.
    """
    out = {'se_worst_mm': np.nan, 'se_worst_naive_mm': np.nan, 'autocorr_time': np.nan,
           'effective_n': np.nan}
    for column in OFFSET_COLUMNS:
        out[f"se_{column}_mm"] = np.nan
    n = int(mask.sum())
    if n <= 7:
        return out

    norms = np.linalg.norm(residual[mask], axis=1)
    rss = float(np.sum(norms ** 2))
    # 3 scalar equations per frame, 6 parameters.
    sigma_squared = rss / max(3 * n - 6, 1)

    # Integral autocorrelation time of the residual norm: sum the autocorrelation until it first
    # goes non-positive. N_eff = N / (1 + 2*tau).
    centred = norms - norms.mean()
    variance = float(np.dot(centred, centred))
    tau = 0.0
    if variance > 0:
        limit = min(len(centred) // 4, 2000)
        for lag in range(1, limit):
            rho = float(np.dot(centred[:-lag], centred[lag:])) / variance
            if rho <= 0:
                break
            tau += rho
    inflation = 1.0 + 2.0 * tau
    out['autocorr_time'] = float(tau)
    out['effective_n'] = float(n / inflation)

    try:
        covariance = sigma_squared * np.linalg.inv(equations['gram'])
    except np.linalg.LinAlgError:
        return out
    naive = np.sqrt(np.clip(np.diag(covariance), 0, None)) * 1000.0
    corrected = naive * np.sqrt(inflation)
    for column, value in zip(OFFSET_COLUMNS, corrected):
        out[f"se_{column}_mm"] = float(value)
    # Worst direction, which is not any single axis: the largest eigenvalue of the covariance.
    eigenvalues = np.linalg.eigvalsh(covariance)
    out['se_worst_naive_mm'] = float(np.sqrt(max(eigenvalues[-1], 0)) * 1000.0)
    out['se_worst_mm'] = out['se_worst_naive_mm'] * float(np.sqrt(inflation))
    return out


# Frame counts the learning curve fits on, as a geometric ladder. Log-spaced because the
# interesting behaviour is at the short end, where IMoVE's 30-90 s trials live.
LEARNING_LADDER = (200, 500, 1000, 2000, 5000, 10000, 20000, 50000)


def learning_curve(plates: Dict[str, PlateTrial], spec: DatasetSpec) -> pd.DataFrame:
    """How far each prefix of a trial's valid frames lands from the full-trial fit.

    Answers "how much data does a fit need", which nothing else here asks — and which the IMoVE
    result needs, since those trials are short AND single-task and section 4 could not separate
    duration from content.

    Measures CONVERGENCE, not accuracy: the reference is the same trial's full fit, so a curve
    that flattens says more data would not move the answer, NOT that the answer is right. Contiguous
    prefixes rather than random subsets, because that is what a shorter recording would give.
    """
    rows = []
    for joint, (parent_sensor, child_sensor) in spec.joints.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        parent = plates[parent_sensor].world_trace
        child = plates[child_sensor].world_trace
        n = min(len(parent), len(child))
        valid = (np.asarray(plates[parent_sensor].valid)[:n]
                 & np.asarray(plates[child_sensor].valid)[:n])
        indices = np.flatnonzero(valid)
        if len(indices) < MIN_FIT_FRAMES:
            continue
        full = solve_offsets(normal_equations(parent, child, valid))
        if full is None:
            continue
        for count in LEARNING_LADDER:
            if count > len(indices):
                continue
            prefix = np.zeros(n, dtype=bool)
            prefix[indices[:count]] = True
            partial_fit = solve_offsets(normal_equations(parent, child, prefix))
            if partial_fit is None:
                continue
            error = np.linalg.norm((partial_fit - full).reshape(2, 3), axis=1) * 1000.0
            rows.append({'joint': joint, 'n_frames': int(count),
                         'error_parent_mm': float(error[0]),
                         'error_child_mm': float(error[1]),
                         'error_mm': float(error.mean()),
                         'n_valid_total': int(len(indices))})
    return pd.DataFrame(rows)


def centre_migration(parent_plate: PlateTrial, child_plate: PlateTrial,
                     offsets: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, float]:
    """Does the joint centre MOVE with flexion, and by how much per degree?

    The direct test of the fixed-centre assumption, and the one thing sections 1 and 8 cannot do:
    they establish that the model fails, this says it fails AS A FUNCTION OF POSTURE. A knee that
    translates as it flexes shows up as a residual that regresses on the flexion angle; noise does
    not.

    The residual is taken in the PARENT's frame so the migration direction is anatomical, and
    regressed component-wise on the relative rotation angle. `migration_mm_per_deg` is the norm of
    the slope vector; `migration_r2` is how much of the residual that linear term explains — a
    large slope with low R^2 is a coincidence, both together is a translating joint.
    """
    out = {'migration_mm_per_deg': np.nan, 'migration_r2': np.nan, 'flexion_range_deg': np.nan}
    n = min(len(parent_plate), len(child_plate))
    selected = np.asarray(mask)[:n]
    if selected.sum() < MIN_FIT_FRAMES:
        return out

    rotations_parent = parent_plate.world_trace.rotations[:n].astype(np.float64)
    rotations_child = child_plate.world_trace.rotations[:n].astype(np.float64)
    relative = np.einsum('nji,njk->nik', rotations_parent, rotations_child)
    # Rotation angle from the trace, which needs no rotvec conversion per sample.
    trace = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.degrees(np.arccos(trace))[selected]

    separation = (parent_plate.world_trace.positions[:n]
                  - child_plate.world_trace.positions[:n]).astype(np.float64)
    residual = (separation
                + np.einsum('nij,j->ni', rotations_parent, offsets['parent'])
                - np.einsum('nij,j->ni', rotations_child, offsets['child']))
    in_parent = np.einsum('nji,nj->ni', rotations_parent, residual)[selected] * 1000.0

    out['flexion_range_deg'] = float(np.percentile(angle, 95) - np.percentile(angle, 5))
    if out['flexion_range_deg'] < 1.0:
        return out
    design = np.column_stack([angle, np.ones_like(angle)])
    coefficients, *_ = np.linalg.lstsq(design, in_parent, rcond=None)
    predicted = design @ coefficients
    total = float(np.sum((in_parent - in_parent.mean(axis=0)) ** 2))
    out['migration_mm_per_deg'] = float(np.linalg.norm(coefficients[0]))
    out['migration_r2'] = (float(1.0 - np.sum((in_parent - predicted) ** 2) / total)
                           if total > 0 else np.nan)
    return out


def contralateral(name: str) -> Optional[str]:
    """'R_Knee' -> 'L_Knee', 'Thigh R Mid' -> 'Thigh L Mid'. None for a midline name."""
    for a, b in (('R_', 'L_'), ('_R_', '_L_'), (' R ', ' L '), (' R', ' L'), ('_R', '_L')):
        if name.startswith(a) or a in name:
            return name.replace(a, b, 1)
        if name.startswith(b) or b in name:
            return name.replace(b, a, 1)
    return None


# ==============================================================================
# Per-trial driver
# ==============================================================================

def joint_fits(plates: Dict[str, PlateTrial], spec: DatasetSpec,
               hinge_alignment: bool = False) -> pd.DataFrame:
    """One row per joint: the offsets, the residual, the conditioning, and the cost of getting
    it wrong."""
    rows = []
    for joint, (parent_sensor, child_sensor) in spec.joints.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        parent_plate, child_plate = plates[parent_sensor], plates[child_sensor]
        parent, child = parent_plate.world_trace, child_plate.world_trace
        n = min(len(parent), len(child))
        valid = np.asarray(parent_plate.valid)[:n] & np.asarray(child_plate.valid)[:n]

        row = {'joint': joint, 'parent_sensor': parent_sensor, 'child_sensor': child_sensor,
               'n_samples': int(n), 'n_valid': int(valid.sum()),
               'converged': bool(valid.sum() >= MIN_FIT_FRAMES)}
        row.update(fit_conditioning(parent, child, valid, hinge_alignment))

        if not row['converged']:
            # Every column still present, all NaN. A short row would make a downstream merge
            # silently drop the joint instead of surfacing that it was never fitted.
            for column in OFFSET_COLUMNS:
                row[column] = np.nan
            for column in ('parent_norm_mm', 'child_norm_mm', 'residual_rms_mm',
                           'residual_median_mm', 'residual_p95_mm', 'unmasked_shift_mm',
                           'unmasked_residual_rms_mm', 'residual_in_sample_mm',
                           'residual_holdout_mm', 'gain_median', 'gain_p95',
                           'acc_error_from_offset', 'imu_error_parent_mm',
                           'imu_error_child_mm', 'imu_init_spread_mm', 'imu_error_relative',
                           'acc_error_from_offset', 'se_worst_mm', 'se_worst_naive_mm',
                           'autocorr_time', 'effective_n', 'migration_mm_per_deg',
                           'migration_r2', 'flexion_range_deg',
                           *(f"se_{c}_mm" for c in OFFSET_COLUMNS)):
                row[column] = np.nan
            rows.append(row)
            continue

        r_parent, r_child, residual = parent.get_joint_center(child, valid, min_frames=1)
        norms = np.linalg.norm(residual, axis=1)
        for column, value in zip(OFFSET_COLUMNS, np.concatenate([r_parent, r_child])):
            row[column] = float(value)
        row['parent_norm_mm'] = float(np.linalg.norm(r_parent) * 1000.0)
        row['child_norm_mm'] = float(np.linalg.norm(r_child) * 1000.0)
        row['residual_rms_mm'] = float(np.sqrt(np.mean(norms[valid] ** 2)) * 1000.0)
        row['residual_median_mm'] = float(np.median(norms[valid]) * 1000.0)
        row['residual_p95_mm'] = float(np.percentile(norms[valid], 95) * 1000.0)

        # The same fit WITHOUT the valid mask — what get_joint_center returns today. The shift
        # between the two is the size of the padded-pose defect, per trial and per joint.
        # An explicit all-True mask, NOT the default: the default is now `valid`, and the whole
        # point of this line is to reproduce the fit-over-everything behaviour to measure what
        # masking is worth.
        unmasked_parent, unmasked_child, unmasked_residual = parent.get_joint_center(
            child, np.ones(min(len(parent), len(child)), dtype=bool), min_frames=1)
        row['unmasked_shift_mm'] = float(max(np.linalg.norm(unmasked_parent - r_parent),
                                             np.linalg.norm(unmasked_child - r_child)) * 1000.0)
        row['unmasked_residual_rms_mm'] = float(
            np.sqrt(np.mean(np.linalg.norm(unmasked_residual, axis=1)[valid] ** 2)) * 1000.0)

        row.update(holdout_residual(parent, child, valid))
        row.update(projection_gain(parent_plate, valid))
        row.update(offset_precision(normal_equations(parent, child, valid), residual, valid))
        row.update(centre_migration(parent_plate, child_plate,
                                    {'parent': r_parent, 'child': r_child}, valid))
        # The headline translation: the between-trial offset spread is not known until the
        # summary stage, so this uses THIS trial's residual as the stand-in error scale. It is
        # the per-trial version of section 6; the summary recomputes it against the real spread.
        row['acc_error_from_offset'] = row['gain_median'] * row['residual_rms_mm'] / 1000.0
        rows.append(row)
    return pd.DataFrame(rows)


def fit_samples(plates: Dict[str, PlateTrial], spec: DatasetSpec,
                stride: int = SAMPLE_STRIDE) -> pd.DataFrame:
    """Per-sample residual norm for each joint — the separation of the two segments' implied
    joint centers, in millimetres, on the trial's own clock."""
    blocks = []
    for joint, (parent_sensor, child_sensor) in spec.joints.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        parent_plate, child_plate = plates[parent_sensor], plates[child_sensor]
        parent, child = parent_plate.world_trace, child_plate.world_trace
        n = min(len(parent), len(child))
        valid = np.asarray(parent_plate.valid)[:n] & np.asarray(child_plate.valid)[:n]
        if valid.sum() < MIN_FIT_FRAMES:
            continue
        _, _, residual = parent.get_joint_center(child, valid, min_frames=1)
        block = pd.DataFrame({
            'timestamp': parent.timestamps[:n].astype(np.float64),
            'joint': joint,
            'valid': valid,
            'residual_mm': (np.linalg.norm(residual, axis=1) * 1000.0).astype(np.float32),
        })
        blocks.append(block.iloc[::stride] if stride > 1 else block)
    if not blocks:
        return pd.DataFrame()
    out = pd.concat(blocks, ignore_index=True)
    out['joint'] = out['joint'].astype('category')
    return out


def compute_trial(plates: Dict[str, PlateTrial], spec: DatasetSpec,
                  tables: Sequence[str] = TRIAL_TABLES, stride: int = SAMPLE_STRIDE,
                  hinge_alignment: bool = False, dataset: str = 'alborno',
                  subject: str = '', trial: str = '') -> Dict[str, pd.DataFrame]:
    builders = {
        'joint_fits': lambda: joint_fits(plates, spec, hinge_alignment),
        'fit_samples': lambda: fit_samples(plates, spec, stride),
        'normal_equations': lambda: normal_equations_table(plates, spec),
        'learning_curve': lambda: learning_curve(plates, spec),
        'landmarks': lambda: landmark_comparison(plates, spec, dataset, subject, trial),
    }
    return {table: build() for table, build in builders.items() if table in set(tables)}


def _load_spec_plates(subject: str, trial: str, dataset: str, spec: DatasetSpec,
                      allow_stale: bool = False) -> Dict[str, PlateTrial]:
    """The trial's plates, narrowed to the sensors this dataset's spec names.

    `allow_stale` reads the cached parquet WITHOUT the freshness check. The repo's rule is that
    the parquet is the interface and `load_trial` raises rather than serving a cache built from
    different code — that rule exists because a silent fallback makes a stale result invisible.
    This is not a silent fallback: it is opt-in per run, it prints a warning, and it stamps
    `built_from_stale_cache` into every manifest it writes, so an artifact produced this way
    says so for as long as it exists.

    It is here because the build layer can be under active development while an analysis is
    being iterated on, and rebuilding 280 trials to answer a question about a filter cutoff is
    the wrong trade. Results from it are for exploration and must be re-confirmed against a
    fresh build before they are quoted.

    THE TRIAL IS READ UNDER `spec.build_name`, NOT THE SPEC NAME, and the two differ. A spec is a
    way of LOOKING at a build tree, and `imove_biplane_vicon` is a second way of looking at
    `imove_biplane`'s — same recordings, marker-cluster poses instead of fluoroscopic ones. Passing
    the spec name here reached `sources.get_source`, which knows only the three build trees, and
    raised `Unknown dataset 'imove_biplane_vicon'` on every trial: the dataset was not merely
    unmeasured, it was unmeasurable. Outputs still go under the SPEC name, which is what keeps the
    two references from overwriting each other's tables.
    """
    build = build_name(dataset)
    if allow_stale:
        path = paths.cached_trial_path(build, subject, trial)
        if not path.exists():
            raise FileNotFoundError(f"{dataset}/{subject}/{trial}: no cached parquet at {path}")
        plates = trial_io.plates_from_frame(pd.read_parquet(path, engine='pyarrow'))
    else:
        plates = load_trial(subject, trial, dataset=build)
    selected = {sensor: plate for sensor, plate in plates.items()
                if sensor in set(spec.segment_sensor.values())}
    if not selected:
        raise ValueError(f"{dataset}/{subject}/{trial}: none of the spec's sensors are present.")
    return selected


def _trial_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                  dataset: str = 'alborno', tables: Sequence[str] = TRIAL_TABLES,
                  stride: int = SAMPLE_STRIDE, allow_stale: bool = False,
                  hinge_alignment: bool = False) -> None:
    subject, trial = row_key
    spec = get_dataset(dataset)
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()
    try:
        plates = _load_spec_plates(subject, trial, dataset, spec, allow_stale)
        for table, df in compute_trial(plates, spec, tables, stride, hinge_alignment,
                                       dataset, subject, trial).items():
            if df.empty:
                continue
            _save(df, trial_table_path(dataset, subject, trial, table), dataset,
                  subject=subject, trial=trial, table=table,
                  built_from_stale_cache=allow_stale)
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success"
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
    return None

def normal_equations_table(plates: Dict[str, PlateTrial], spec: DatasetSpec) -> pd.DataFrame:
    """One row per joint holding that joint's normal equations — 44 numbers that stand in for
    the whole trial.

    Written out so the per-SUBJECT pass can fit a subject's trials jointly, and cross-validate
    one trial's offsets on another, without reloading anything. See `normal_equations` for why
    44 numbers is enough.
    """
    rows = []
    for joint, (parent_sensor, child_sensor) in spec.joints.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        parent_plate, child_plate = plates[parent_sensor], plates[child_sensor]
        n = min(len(parent_plate), len(child_plate))
        valid = np.asarray(parent_plate.valid)[:n] & np.asarray(child_plate.valid)[:n]
        if valid.sum() < MIN_FIT_FRAMES:
            continue
        equations = normal_equations(parent_plate.world_trace, child_plate.world_trace, valid)
        row = {'joint': joint, 'sum_bb': equations['sum_bb'], 'n': equations['n']}
        row.update(dict(zip(NE_GRAM_COLUMNS, equations['gram'].reshape(-1))))
        row.update(dict(zip(NE_RHS_COLUMNS, equations['rhs'])))
        rows.append(row)
    return pd.DataFrame(rows)


def equations_from_row(row: pd.Series) -> Dict[str, np.ndarray]:
    """Rebuilds the normal-equation dict from a row of `normal_equations.parquet`."""
    return {'gram': row[list(NE_GRAM_COLUMNS)].to_numpy(dtype=float).reshape(6, 6),
            'rhs': row[list(NE_RHS_COLUMNS)].to_numpy(dtype=float),
            'sum_bb': float(row['sum_bb']), 'n': int(row['n'])}


def combined_fits(dataset: str, row_keys: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    """Per (subject, joint): the offsets fitted on ALL of that subject's trials at once, and the
    cross-validation that says whether combining is legitimate.

    The premise is that a subject's trials are one capture with the sensors left in place, so the
    offsets are anatomy plus mounting and cannot differ between them. If that holds, each trial's
    fit is a noisy estimate of one truth and the joint fit is strictly better — more samples, and
    more importantly more DIVERSE relative rotations, which is the thing the conditioning depends
    on (see `fit_conditioning`).

    If it does NOT hold, combining is averaging two different quantities and is worse than either.
    `cross_rms_mm` is what separates the two cases, and it is the column to read first. What it
    cannot say is WHY the offsets differ: re-taping is the intuitive culprit but `shift_decomposition`
    rules it out on Al Borno, leaving fit error and posture dependence, which want opposite
    responses.

        own_rms_mm     each trial scored with the offsets fitted on ITSELF. The floor.
        combined_rms   each trial scored with the offsets fitted on all trials together.
        cross_rms_mm   each trial scored with offsets fitted on the OTHER trials only, so the
                       scored data had no influence on the offsets at all.

    cross ~ own means the offsets transfer between trials and the between-trial spread was fit
    noise. cross >> own means they genuinely differ and the mounting moved. `combined` sits
    between by construction and is not evidence on its own.
    """
    rows = []
    for subject in subjects_of(row_keys):
        trials = [t for s, t in row_keys if s == subject]
        per_trial: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for trial in trials:
            path = trial_table_path(dataset, subject, trial, 'normal_equations')
            if not path.exists():
                continue
            table = pd.read_parquet(path, engine='pyarrow')
            for _, row in table.iterrows():
                per_trial.setdefault(str(row['joint']), {})[trial] = equations_from_row(row)
        for joint, by_trial in per_trial.items():
            if len(by_trial) < 2:
                continue
            everything = combine_normal_equations(list(by_trial.values()))
            combined = solve_offsets(everything)
            if combined is None:
                continue
            own, cross, under_combined, spread = [], [], [], []
            for trial, equations in by_trial.items():
                alone = solve_offsets(equations)
                others = [e for other, e in by_trial.items() if other != trial]
                held_out = solve_offsets(combine_normal_equations(others))
                own.append(residual_rms_from(equations, alone))
                cross.append(residual_rms_from(equations, held_out))
                under_combined.append(residual_rms_from(equations, combined))
                if alone is not None:
                    spread.append(np.linalg.norm((alone - combined).reshape(2, 3), axis=1))
            spread = np.array(spread) * 1000.0
            # Excitation of the combined fit, from the same closed form the per-trial one uses:
            # the Gram's smallest eigenvalue is N - sigma_max, so excitation = that over N.
            smallest = float(np.linalg.eigvalsh(everything['gram'])[0])
            rows.append({
                'subject': subject, 'joint': joint, 'n_trials': len(by_trial),
                'n_frames': everything['n'],
                'combined_excitation': smallest / everything['n'],
                'own_rms_mm': float(np.mean(own)),
                'combined_rms_mm': float(np.mean(under_combined)),
                'cross_rms_mm': float(np.mean(cross)),
                'parent_shift_mm': float(spread[:, 0].mean()) if len(spread) else np.nan,
                'child_shift_mm': float(spread[:, 1].mean()) if len(spread) else np.nan,
                **dict(zip((f"combined_{c}" for c in OFFSET_COLUMNS), combined)),
            })
    return pd.DataFrame(rows)


def report_combined(spec: DatasetSpec, fits: pd.DataFrame, combined: pd.DataFrame) -> None:
    """Is one set of offsets per SUBJECT better than one per trial?"""
    _header(7, "ONE FIT PER SUBJECT INSTEAD OF ONE PER TRIAL",
            "a subject's trials are one capture with the sensors left on, so the offsets cannot "
            "differ between them")
    if combined.empty:
        print("No subject has two or more trials with a fitted joint; nothing to combine.")
        return
    per_trial_excitation = (fits[fits['converged']].groupby('joint', observed=True)['excitation']
                            .mean() if not fits.empty else pd.Series(dtype=float))
    print(f"  {'joint':<12}{'own':>9}{'combined':>10}{'cross':>9}{'cross/own':>11}"
          f"{'shift':>9}{'excitation':>22}{'subj':>6}")
    print(f"  {'':<12}{'(mm)':>9}{'(mm)':>10}{'(mm)':>9}{'':>11}{'(mm)':>9}"
          f"{'per-trial -> combined':>22}{'':>6}")
    for joint in [j for j in spec.joints if j in set(combined['joint'])]:
        rows = combined[combined['joint'] == joint]
        own, cross = rows['own_rms_mm'].mean(), rows['cross_rms_mm'].mean()
        shift = max(rows['parent_shift_mm'].mean(), rows['child_shift_mm'].mean())
        before = per_trial_excitation.get(joint, np.nan)
        after = rows['combined_excitation'].mean()
        print(f"  {joint:<12}{own:>9.2f}{rows['combined_rms_mm'].mean():>10.2f}{cross:>9.2f}"
              f"{cross / own if own else np.nan:>11.2f}{shift:>9.1f}"
              f"{f'{before:.4f} -> {after:.4f}':>22}{len(rows):>6}")

    ratio = combined['cross_rms_mm'].mean() / combined['own_rms_mm'].mean()
    gain = (combined['combined_excitation'].mean()
            / max(per_trial_excitation.mean(), 1e-12) if len(per_trial_excitation) else np.nan)
    print(f"\nPooled cross/own ratio: {ratio:.2f}. Combined excitation is {gain:.1f}x the "
          f"per-trial mean.")
    print("\nRead cross/own first, because it is the one column that tests the premise rather "
          "than assuming\nit. Each trial is scored there with offsets fitted on the OTHER trials "
          "only, so the scored data\nhad no say in them. A ratio near 1 means the offsets "
          "transfer and the between-trial spread in\nsection 5 was fit noise that combining "
          "removes. A ratio well above 1 means the offsets genuinely\ndiffer between trials.")
    print("\nA HIGH RATIO DOES NOT SAY WHY, and on Al Borno the obvious explanation is ruled out. "
          "A re-taped\nsensor would move both of a segment's joint centres by the same vector; "
          "section 8 now measures\nthat common component directly and finds the two centres move "
          "INDEPENDENTLY, while the\nplate-to-landmark vector — the mounting itself, no fit "
          "involved — shifts only ~4.5 mm between\nactivities. What is left is fit error in the "
          "ill-determined transverse mode plus a genuinely\nposture-dependent centre (section "
          "11), and those pull opposite ways on whether to pool: pooling\naverages the first down "
          "and the second together.")
    print("\nThe excitation column is why combining can help even when each trial is already fine. "
          "The fit's\nconditioning depends on the DIVERSITY of relative rotations, not the sample "
          "count, so pooling a\nwalk with a trial containing sitting and stairs buys more than "
          "pooling two walks would.")
    print("\n`shift` is how far the combined offsets sit from each trial's own — the practical "
          "size of the\nchange, and the amount a downstream projection would move if it switched "
          "to the subject-level fit.")
    report_mounting_split(spec, combined)


def report_mounting_split(spec: DatasetSpec, combined: pd.DataFrame) -> None:
    """Splits the cross/own degradation by how the joint's two sensors are MOUNTED.

    Only IMoVE distinguishes the two, and the distinction matters more than it sounds. Its Mid
    sensors are bolted to the marker cluster and share one hardware offset constant
    (RIGID_SENSOR_OFFSET_MM), while every other sensor is taped on and has its cluster offset
    REFITTED FOR EVERY TRIAL (`imove_mocap._sensor_offsets`, falling back to
    NOMINAL_SENSOR_OFFSET_MM). A taped sensor's plate origin therefore genuinely moves between
    trials — by the fit's own 8.5-21.1 mm per-axis spread — before any physiology enters.

    That makes this the discriminator for what "the offsets differ between trials" means. If the
    joint centre itself were the trial-dependent thing, mounting would not matter and all three
    rows would degrade alike. If the per-trial sensor-offset refit is doing it, joints between
    two bolted sensors should degrade least and joints between two taped ones most.
    """
    try:
        from src.toolchest.building.imove_mocap import (NOMINAL_SENSOR_OFFSET_MM,
                                                        RIGID_SENSOR_OFFSET_MM)
    except Exception:
        return
    taped = set(NOMINAL_SENSOR_OFFSET_MM)
    sensors = set(spec.segment_sensor.values())
    if not (taped & sensors) or not (set(RIGID_SENSOR_OFFSET_MM) & sensors):
        return

    def mounting(joint: str) -> str:
        parent, child = spec.joints[joint]
        count = sum(sensor in taped for sensor in (parent, child))
        return ('both bolted', 'one taped', 'both taped')[count]

    frame = combined.copy()
    frame['ratio'] = frame['cross_rms_mm'] / frame['own_rms_mm'].replace(0, np.nan)
    frame['mounting'] = [mounting(j) for j in frame['joint']]
    grouped = frame.groupby('mounting', observed=True)['ratio'].agg(['mean', 'count'])

    print("\nBY SENSOR MOUNTING — the discriminator for WHY the offsets differ between trials:")
    print(f"  {'mounting':<16}{'cross/own':>11}{'joint-subjects':>17}")
    for label in ('both bolted', 'one taped', 'both taped'):
        if label not in grouped.index:
            continue
        row = grouped.loc[label]
        print(f"  {label:<16}{row['mean']:>11.2f}{int(row['count']):>17}")
    print("\nThe Mid sensors are BOLTED to the marker cluster and share one hardware offset "
          "constant. Every\nother sensor is TAPED, and its cluster offset is refitted for every "
          "trial, so its plate origin\nmoves between trials by the fit's own spread before any "
          "physiology enters. The ordering above\nis monotone in how many taped sensors a joint "
          "has, which says a large part of this dataset's\nbetween-trial offset variation is that "
          "REFIT rather than a moving joint centre.")
    print("\nIt does not explain all of it: joints between two bolted sensors still degrade, and "
          "Al Borno —\nwhere every sensor sits on a fixed marker plate and nothing is refitted — "
          "degrades too. So there\nis a real trial-dependent component underneath, and the taped "
          "refit is stacked on top of it.")


# ==============================================================================
# Summary
# ==============================================================================

SUMMARY_COLUMNS = (['dataset', 'subject', 'joint', 'metric', 'unit', 'n_trials', 'mean', 'std',
                    'min'] + [f"p{int(round(q * 100)):02d}" for q in QUANTILES] + ['max'])

SUMMARY_METRICS = {
    'residual_rms_mm': 'mm',
    'residual_holdout_mm': 'mm',
    'excitation': '-',
    'condition_number': '-',
    'unmasked_shift_mm': 'mm',
    'parent_norm_mm': 'mm',
    'child_norm_mm': 'mm',
    'imu_error_parent_mm': 'mm',
    'imu_error_child_mm': 'mm',
    'imu_error_relative': '-',
}


def summarize(dataset: str, fits: pd.DataFrame) -> pd.DataFrame:
    """Per-(subject, joint) and pooled quantiles of each scalar, over TRIALS.

    The unit of observation here is a trial's fit, not a sample: every number in `joint_fits` is
    already a reduction over a whole trial, so pooling samples would be meaningless and pooling
    trials is what the between-trial-stability question wants. That also sidesteps the
    length-weighting problem the sample-pooled experiments have to manage.
    """
    if fits.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    frames = []
    for metric, unit in SUMMARY_METRICS.items():
        if metric not in fits.columns:
            continue
        usable = fits[np.isfinite(fits[metric])]
        if usable.empty:
            continue
        for by_subject in (True, False):
            keys = (['subject'] if by_subject else []) + ['joint']
            grouped = usable.groupby(keys, observed=True)[metric]
            part = grouped.agg(n_trials='size', mean='mean', std='std', min='min', max='max')
            quantiles = grouped.quantile(QUANTILES).unstack()
            quantiles.columns = [f"p{int(round(q * 100)):02d}" for q in quantiles.columns]
            part = part.join(quantiles).reset_index()
            if not by_subject:
                part['subject'] = 'all'
            part['dataset'] = dataset
            part['metric'] = metric
            part['unit'] = unit
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.concat(frames, ignore_index=True)[SUMMARY_COLUMNS]


def offset_stability(fits: pd.DataFrame) -> pd.DataFrame:
    """Per (subject, joint): how far the fitted offset moves between that subject's trials.

    The practical error bar. The offset is anatomy plus mounting, both constant within a session,
    so anything this spread contains is fit error — and it is the number to quote when a
    downstream experiment uses one trial's offsets, because the in-trial residual describes how
    well the model fits that trial and says nothing about whether the six parameters are right.
    """
    usable = fits[fits['converged'] & np.isfinite(fits['parent_norm_mm'])]
    if usable.empty:
        return pd.DataFrame()

    rows = []
    for (subject, joint), group in usable.groupby(['subject', 'joint'], observed=True):
        if len(group) < 2:
            continue
        offsets = group[list(OFFSET_COLUMNS)].to_numpy() * 1000.0
        centre = offsets.mean(axis=0)
        deviation = np.linalg.norm((offsets - centre).reshape(len(group), 2, 3), axis=2)
        rows.append({
            'subject': subject, 'joint': joint, 'n_trials': len(group),
            'mocap_arm_mm': float(np.linalg.norm(offsets.reshape(len(group), 2, 3),
                                                 axis=2).mean()),
            # Per-axis sd, then the 3D spread, for the parent and child offsets separately.
            'parent_spread_mm': float(deviation[:, 0].mean()),
            'child_spread_mm': float(deviation[:, 1].mean()),
            'parent_max_dev_mm': float(deviation[:, 0].max()),
            'child_max_dev_mm': float(deviation[:, 1].max()),
            'mean_excitation': float(group['excitation'].mean()),
            'mean_residual_mm': float(group['residual_rms_mm'].mean()),
        })
    return pd.DataFrame(rows)

# ==============================================================================
# Segment lengths: a physical invariant the fit has to preserve
# ==============================================================================

def invariant_pairs(spec: DatasetSpec) -> Dict[str, List[Tuple[str, str, str, str]]]:
    """{segment: [(joint_a, role_a, joint_b, role_b), ...]} — pairs of DIFFERENT joints that a
    segment borders, whose joint centres therefore both live in that one segment's frame.

    The distance between them is a rigid anatomical dimension: femur length for the thigh, tibia
    length for the shank, and for the pelvis the spacings between the lumbar and the two hips.
    It cannot change between two trials of one subject, which is what makes it a check.

    Only ANATOMICALLY distinct joints are paired. On IMoVE the pelvis borders six joints, but
    R_Hip and R_Hip_H are the same hip seen from two thigh sensors, so the distance between their
    centres is a disagreement between estimates rather than a dimension of the subject —
    interesting, but not an invariant, and pairing them would report fit error as anatomy.
    """
    roles = segment_joint_roles(spec)
    pairs: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for segment, entries in roles.items():
        # Primary joints first, so when several placement variants measure the same anatomical
        # distance the one kept is the canonical pair rather than whichever came first.
        ordered = sorted(entries, key=lambda e: e[0] not in spec.primary_joints)
        found, seen = [], set()
        for index, (joint_a, role_a) in enumerate(ordered):
            for joint_b, role_b in ordered[index + 1:]:
                anatomical = frozenset((canonical_joint(joint_a, spec),
                                        canonical_joint(joint_b, spec)))
                # Two joints, not one seen twice: R_Hip and R_Hip_H are the same hip reached
                # through different thigh sensors, and their separation is fit disagreement
                # rather than a dimension of the subject.
                if len(anatomical) < 2 or anatomical in seen:
                    continue
                # ...and one row per anatomical pair per segment. IMoVE's pelvis borders six
                # joints that reduce to two hips, so without this the same pelvis width appears
                # nine times under different placement combinations.
                seen.add(anatomical)
                found.append((joint_a, role_a, joint_b, role_b))
        if found:
            pairs[segment] = found
    return pairs


def segment_lengths(fits: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Every invariant distance, per trial: one row per (subject, trial, segment, joint pair).

    A rigid dimension of the subject, so it cannot change between their trials, and what it
    catches cleanly is a SCALE difference: lengths changing by one common factor across
    independent segments is a mocap calibration difference between captures, not anatomy.

    IT CANNOT DIAGNOSE A RE-MOUNT, and this module used to claim it could. The argument was that a
    re-mounted sensor shifts both of a segment's joint offsets by the same vector, which cancels in
    the distance between them, so preserved lengths beside moving offsets meant the sensor moved.
    The first half is true and the inference is not, because length is a LOSSY projection of the
    shift: it depends only on the differential (the common part cancels exactly), and it is
    second-order insensitive to whatever part of that differential is perpendicular to the segment
    axis — a 15 mm transverse displacement on a 450 mm femur changes the length by 15^2/(2*450),
    a quarter of a millimetre. Preserved length is therefore equally consistent with two centres
    wandering independently and transversely, which is what Al Borno actually does; see
    `shift_decomposition`, which measures the common and differential parts directly instead.
    """
    # Read from the fits table rather than reloading trials: this runs at the summary stage,
    # where the offsets are already in hand and reopening 243 parquets to recompute them would
    # cost more than every other section combined.
    if fits.empty or not set(OFFSET_COLUMNS) <= set(fits.columns):
        return pd.DataFrame()
    rows = []
    pairs = invariant_pairs(spec)
    usable = fits[fits['converged']] if 'converged' in fits.columns else fits
    for (subject, trial), group in usable.groupby(['subject', 'trial'], observed=True):
        offsets = {}
        for _, row in group.iterrows():
            values = row[list(OFFSET_COLUMNS)].to_numpy(dtype=float)
            if np.isfinite(values).all():
                offsets[str(row['joint'])] = {'parent': values[:3], 'child': values[3:]}
        for segment, entries in pairs.items():
            for joint_a, role_a, joint_b, role_b in entries:
                if joint_a not in offsets or joint_b not in offsets:
                    continue
                separation = offsets[joint_a][role_a] - offsets[joint_b][role_b]
                rows.append({'subject': subject, 'trial': trial, 'segment': segment,
                             'pair': f"{joint_a}-{joint_b}",
                             'length_mm': float(np.linalg.norm(separation) * 1000.0)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # Each length as a ratio to that subject-and-pair's median across trials. A trial whose
    # ratios are all the same non-unit number has a SCALE problem; one whose ratios scatter has
    # something segment-specific.
    median = frame.groupby(['subject', 'pair'], observed=True)['length_mm'].transform('median')
    frame['ratio'] = frame['length_mm'] / median.replace(0, np.nan)
    return frame


def marker_closure(dataset: str, spec: DatasetSpec, subjects: Optional[Sequence[str]] = None
                   ) -> pd.DataFrame:
    """The anatomical centre scored as closure, for datasets whose landmarks live in a static file.

    NO FRAME TRANSPORT IS NEEDED, which is the one thing that makes this cheap. Closure is a
    distance between two constructed WORLD points, so re-basing a plate's body frame and its
    offset together leaves it exactly unchanged — verified at 7.7987 vs 7.7935 mm between a raw
    template reconstruction and the built plates, and the ambiguity radii come out bit-identical
    because Gram eigenvalues are invariant under an orthogonal re-basing too. So the marker
    centre can be scored in the raw standing-template frame and compared directly against rows
    computed in the built frame.

    What it still needs is the STANDING TEMPLATES carried into the trial, because the centre and
    the fit have to share one body frame even if that frame is arbitrary. Costs one .trc read and
    one reconstruction per trial.
    """
    from src.toolchest.building import landmarks as landmark_reader
    try:
        landmark_spec = landmark_reader.get_spec(dataset)
    except ValueError:
        return pd.DataFrame()
    if not landmark_spec.static_centres or landmark_spec.read_static is None:
        return pd.DataFrame()

    sensors = sorted({s for joint in landmark_spec.static_centres
                      for s in spec.joints.get(joint, ())})
    rows = []
    for subject, trial in enumerate_trials(dataset):
        if subjects and subject not in subjects:
            continue
        capture = landmark_spec.read_static(subject, sensors)
        if capture is None:
            continue
        # The centre in each bordering segment's own frame, medianed over the standing frames.
        centres: Dict[Tuple[str, str], np.ndarray] = {}
        for joint, names in landmark_spec.static_centres.items():
            points = [capture.markers.get(name) for name in names]
            if any(point is None for point in points):
                continue
            # Averaged only where EVERY marker of the pair is tracked, so a dropout on one drops
            # the frame rather than sliding the midpoint onto the survivor. Frames where none is
            # tracked stay NaN and are masked out below.
            stacked = np.stack(points, axis=0)
            complete = np.isfinite(stacked).all(axis=(0, 2))
            world = np.full(stacked.shape[1:], np.nan)
            world[complete] = stacked[:, complete].mean(axis=0)
            for sensor in spec.joints.get(joint, ()):
                if sensor not in capture.poses:
                    continue
                positions, rotations, valid = capture.poses[sensor]
                n = min(len(positions), len(world))
                usable = valid[:n] & np.isfinite(world[:n]).all(axis=1)
                if usable.sum() < landmark_reader.MIN_STATIC_FRAMES:
                    continue
                local = np.einsum('nji,nj->ni', rotations[:n][usable],
                                  world[:n][usable] - positions[:n][usable])
                centres[(joint, sensor)] = np.median(local, axis=0)

        try:
            poses, _ = landmark_reader.read_alborno_plates(
                landmark_reader._alborno_trc(subject, trial),
                {s: s for s in sensors}, templates=capture.templates)
        except Exception:
            continue
        for joint, names in landmark_spec.static_centres.items():
            pair = spec.joints.get(joint)
            if not pair or any((joint, s) not in centres for s in pair) \
                    or any(s not in poses for s in pair):
                continue
            traces = []
            for sensor in pair:
                positions, rotations, valid = poses[sensor]
                traces.append(WorldTrace(np.arange(len(positions), dtype=float), positions,
                                         rotations, valid=valid))
            n = min(len(traces[0]), len(traces[1]))
            mask = np.asarray(traces[0].valid)[:n] & np.asarray(traces[1].valid)[:n]
            if mask.sum() < MIN_FIT_FRAMES:
                continue
            equations = normal_equations(traces[0], traces[1], mask)
            own = solve_offsets(equations)
            if own is None:
                continue
            offsets = np.concatenate([centres[(joint, pair[0])], centres[(joint, pair[1])]])
            rows.append({'subject': subject, 'trial': trial, 'joint': joint, 'source': 'marker',
                         'in_sample': False,
                         'closure_mm': residual_rms_from(equations, offsets),
                         'offset_from_own_mm': float(np.linalg.norm(offsets - own) * 1000.0)})
    return pd.DataFrame(rows)


def stored_trial_keys(dataset: str, table: str = 'normal_equations') -> List[Tuple[str, str]]:
    """Every (subject, trial_key) with `table` written under this dataset's results tree.

    RECURSIVE, because a trial KEY MAY CONTAIN SLASHES. `trial_table_path` joins the key onto the
    subject directory verbatim, and the biplane source names its trials by the session and block
    they came from — 'Test1/A/RSDrop1' — so those tables sit three levels below the subject rather
    than one. The one-level `iterdir()` this replaces found none of them, and the failure was
    SILENT in the worst way: `closure_by_source` and `ambiguity_table` returned empty frames, main()
    skips writing an empty frame, and the report then simply had no headline for those datasets
    with nothing saying why. `built_trials` in global_assumptions already learned this lesson; this
    is the same rule applied to the results tree instead of the build tree.

    The key is the parquet's parent relative to the subject directory, forward-slashed on every
    platform, so it round-trips through `trial_table_path` and matches `enumerate_trials`.
    """
    root = dataset_dir(dataset)
    if not root.is_dir():
        return []
    return sorted((subject_dir.name,
                   artifact.parent.relative_to(subject_dir).as_posix())
                  for subject_dir in root.iterdir() if subject_dir.is_dir()
                  for artifact in subject_dir.rglob(f'{table}.parquet'))


def closure_by_source(dataset: str, spec: DatasetSpec, with_marker: bool = True
                      ) -> pd.DataFrame:
    """THE HEADLINE TABLE: every candidate offset pair scored on one metric, out of sample.

    The metric is closure — how nearly the two segments' offsets construct the SAME point,

        RMS_t | (p_p + R_p c_p) - (p_c + R_c c_c) |

    which is what a relative filter actually needs, since it compares two estimates of one
    physical point rather than locating that point anatomically. Any six numbers can be scored
    on any trial's normal equations, so mocap fits, marker midpoints and inertial fits land in
    one column of millimetres with no conversion.

    THE SOURCES, and which of them are out of sample:

        own            fitted on this trial. IN SAMPLE, so it is the floor and not a result:
                       what it measures is model mismatch, the part no offset can remove.
        own_holdout    fitted on the first half of this trial, scored on the second.
        cross_trial    fitted on the subject's OTHER trials only.
        inertial       the optimized Seel fit from gyro and accelerometer alone, scored here
                       against mocap poses. The one row that says whether this works without
                       a lab.
        inertial_ship  the shipped estimator, for the same comparison.
        marker         an anatomical centre, where the dataset has one.

    Everything but `marker` comes from stored tables, so this costs no trial loading; `marker`
    adds one .trc read per trial and is skipped with `with_marker=False`.

    Read `own` as the floor, then the gap from it. A source within a millimetre or two of `own`
    is as good as refitting; `ambiguity_radii` says how large an offset difference that
    tolerates, and it is usually tens of millimetres.
    """
    rows = []
    root = dataset_dir(dataset)
    inertial_root = paths.EXPERIMENTS_DIR / 'inertial_joint_center' / dataset
    by_subject: Dict[str, List[str]] = {}
    for subject, trial in stored_trial_keys(dataset):
        by_subject.setdefault(subject, []).append(trial)
    for subject, trials in by_subject.items():
        subject_dir = root / subject
        equations, own = {}, {}
        for trial in trials:
            path = subject_dir / trial / 'normal_equations.parquet'
            if not path.exists():
                continue
            table = pd.read_parquet(path).set_index('joint')
            for joint in table.index:
                equations[(trial, joint)] = equations_from_row(table.loc[joint])
                own[(trial, joint)] = solve_offsets(equations[(trial, joint)])

        for (trial, joint), current in equations.items():
            if own[(trial, joint)] is None:
                continue
            candidates = {'own': own[(trial, joint)]}

            # Fitted on the subject's OTHER trials only, so the scored data had no say.
            others = [equations[(t, j)] for (t, j) in equations
                      if j == joint and t != trial]
            if others:
                combined = solve_offsets(combine_normal_equations(others))
                if combined is not None:
                    candidates['cross_trial'] = combined

            fits_path = subject_dir / trial / 'joint_fits.parquet'
            if fits_path.exists():
                table = pd.read_parquet(fits_path).set_index('joint')
                if joint in table.index and 'residual_holdout_mm' in table.columns:
                    rows.append({'subject': subject, 'trial': trial, 'joint': joint,
                                 'source': 'own_holdout', 'in_sample': False,
                                 'closure_mm': float(table.loc[joint, 'residual_holdout_mm']),
                                 'offset_from_own_mm': np.nan})

            inertial_path = inertial_root / subject / trial / 'inertial_fits.parquet'
            if inertial_path.exists():
                table = pd.read_parquet(inertial_path).set_index('joint')
                if joint in table.index:
                    for label, prefix in (('inertial', 'opt'), ('inertial_ship', 'imu')):
                        columns = [f"{prefix}_{c}" for c in OFFSET_COLUMNS]
                        if set(columns) <= set(table.columns):
                            values = table.loc[joint, columns].to_numpy(dtype=float)
                            if np.isfinite(values).all():
                                candidates[label] = values

            for source, offsets in candidates.items():
                rows.append({
                    'subject': subject, 'trial': trial, 'joint': joint, 'source': source,
                    'in_sample': source == 'own',
                    'closure_mm': residual_rms_from(current, offsets),
                    'offset_from_own_mm': float(np.linalg.norm(
                        offsets - own[(trial, joint)]) * 1000.0),
                })
    frame = pd.DataFrame(rows)
    if with_marker:
        frame = pd.concat([frame, marker_closure(dataset, spec)], ignore_index=True)
    return frame


# The order sources are reported in: floor first, then increasingly independent of the trial.
SOURCE_ORDER = ('own', 'own_holdout', 'cross_trial', 'marker', 'inertial', 'inertial_ship')


def ambiguity_table(dataset: str) -> pd.DataFrame:
    """Per (subject, trial, joint) ambiguity radii, from the stored normal equations.

    Free: the Gram matrix is already on disk, and the radii are an eigendecomposition of it.
    """
    rows = []
    for subject, trial in stored_trial_keys(dataset):
        path = trial_table_path(dataset, subject, trial, 'normal_equations')
        for joint, row in pd.read_parquet(path).set_index('joint').iterrows():
            equations = equations_from_row(row)
            radii, _ = ambiguity_radii(equations)
            rows.append({'subject': subject, 'trial': trial, 'joint': joint,
                         'n': equations['n'], 'worst_mm': float(radii[0]),
                         'median_mm': float(np.median(radii)),
                         'best_mm': float(radii[-1]),
                         'anisotropy': float(radii[0] / max(radii[-1], 1e-12))})
    return pd.DataFrame(rows)


def shift_decomposition(fits: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Is a segment's between-trial offset wander COMMON to its two joint centres, or not?

    This is the re-mount test that section 8's segment length only gestures at. A re-mount moves
    the sensor, so both joint centres in that sensor's frame shift by the SAME vector: the common
    part is everything and the differential is zero. Independent fit error in two separately
    solved joints has no reason to be common at all.

    Per trial, against the subject's own median offset for that pair:

        common       = (dA + dB) / 2      what a re-mount produces, and length is blind to it
        differential = dA - dB            the only thing that can change the length

    and the differential is split along and across the segment axis, because only the along part
    moves the length at first order. `perp_costs_mm` is the second-order price of the across part,
    |perp|^2 / 2L, which is what makes a preserved length such weak evidence.

    Reported as RMS over a subject's trials, so it does not depend on which two are compared.
    """
    if fits.empty or not set(OFFSET_COLUMNS) <= set(fits.columns):
        return pd.DataFrame()
    usable = fits[fits['converged']] if 'converged' in fits.columns else fits
    pairs = invariant_pairs(spec)
    per_trial = {}
    for (subject, trial), group in usable.groupby(['subject', 'trial'], observed=True):
        offsets = {}
        for _, row in group.iterrows():
            values = row[list(OFFSET_COLUMNS)].to_numpy(dtype=float) * 1000.0
            if np.isfinite(values).all():
                offsets[str(row['joint'])] = {'parent': values[:3], 'child': values[3:]}
        per_trial[(subject, trial)] = offsets

    rows = []
    subjects = {s for s, _ in per_trial}
    for subject in sorted(subjects):
        trials = [t for s, t in per_trial if s == subject]
        if len(trials) < 2:
            continue
        for segment, entries in pairs.items():
            for joint_a, role_a, joint_b, role_b in entries:
                present = [t for t in trials
                           if joint_a in per_trial[(subject, t)]
                           and joint_b in per_trial[(subject, t)]]
                if len(present) < 2:
                    continue
                a = np.array([per_trial[(subject, t)][joint_a][role_a] for t in present])
                b = np.array([per_trial[(subject, t)][joint_b][role_b] for t in present])
                # Median rather than mean: with three or more trials one bad fit should not
                # define the reference every other trial is measured against.
                delta_a, delta_b = a - np.median(a, axis=0), b - np.median(b, axis=0)
                common = 0.5 * (delta_a + delta_b)
                differential = delta_a - delta_b
                axis = np.median(a, axis=0) - np.median(b, axis=0)
                length = float(np.linalg.norm(axis))
                if length < 1e-6:
                    continue
                unit = axis / length
                along = differential @ unit
                perpendicular = differential - along[:, None] * unit

                def rms(v):
                    return float(np.sqrt((np.linalg.norm(v, axis=1) ** 2).mean()))

                rows.append({
                    'subject': subject, 'segment': segment, 'pair': f"{joint_a}-{joint_b}",
                    'n_trials': len(present), 'length_mm': length,
                    'shift_mm': 0.5 * (rms(delta_a) + rms(delta_b)),
                    'common_mm': rms(common), 'differential_mm': rms(differential),
                    'diff_along_mm': float(np.sqrt((along ** 2).mean())),
                    'diff_perp_mm': rms(perpendicular),
                    'perp_costs_mm': float(rms(perpendicular) ** 2 / (2 * length)),
                })
    return pd.DataFrame(rows)


def length_diagnosis(lengths: pd.DataFrame, stability: pd.DataFrame,
                     preserved_mm: float = 8.0, scale_spread: float = 0.03
                     ) -> pd.DataFrame:
    """One row per subject: how much the invariants moved, and what that implies.

    The thresholds are deliberately loose and are declared here rather than tuned: `preserved_mm`
    is a little above the per-trial fit residual (6-13 mm), so "preserved" means "within what the
    fit could not have resolved anyway", and `scale_spread` calls a set of ratios uniform when
    they agree to 3%. They label a tendency, not a verdict — the numbers beside them are what a
    reader should quote.
    """
    if lengths.empty:
        return pd.DataFrame()
    rows = []
    spread_by_subject = (stability.groupby('subject', observed=True)[['parent_spread_mm',
                                                                     'child_spread_mm']]
                         .mean().max(axis=1) if not stability.empty else pd.Series(dtype=float))
    for subject, group in lengths.groupby('subject', observed=True):
        if group['trial'].nunique() < 2:
            continue
        by_pair = group.groupby('pair', observed=True)['length_mm']
        # Peak-to-peak across trials per invariant, then the MEDIAN over invariants. Median and
        # not mean: any pair involving a badly-fitted joint inherits that joint's instability —
        # on Al Borno the three pelvis pairs all involve the lumbar, whose offsets move 45-53 mm,
        # and a mean over seven pairs lets those three decide the subject's reading. The
        # per-invariant table below is where that detail belongs, not hidden in an average.
        per_pair = by_pair.apply(lambda s: s.max() - s.min())
        change = float(per_pair.median())
        by_trial = group.groupby('trial', observed=True)['ratio'].agg(['mean', 'std'])
        worst = by_trial['mean'].sub(1.0).abs().idxmax()
        offset_shift = float(spread_by_subject.get(subject, np.nan))

        # NO 'sensor moved' READING HERE ANY MORE. Preserved length beside a moving offset used to
        # be called a re-mount; it is not evidence of one, because length cannot see a transverse
        # differential (see `shift_decomposition`, which tests it properly). What survives is the
        # scale check, which is a genuine property of the ratios.
        if change <= preserved_mm:
            reading = 'lengths hold' if offset_shift > preserved_mm else 'stable'
        elif float(by_trial.loc[worst, 'std']) <= scale_spread:
            reading = f"scale {100 * (by_trial.loc[worst, 'mean'] - 1):+.1f}% in {worst}"
        else:
            reading = 'uneven — neither'
        rows.append({'subject': subject, 'n_trials': group['trial'].nunique(),
                     'length_change_mm': change,
                     'length_change_worst_mm': float(per_pair.max()),
                     'worst_pair': str(per_pair.idxmax()),
                     'offset_shift_mm': offset_shift,
                     'worst_trial': worst,
                     'worst_ratio': float(by_trial.loc[worst, 'mean']),
                     'ratio_spread': float(by_trial.loc[worst, 'std']),
                     'reading': reading})
    return pd.DataFrame(rows)


def report_segment_lengths(spec: DatasetSpec, lengths: pd.DataFrame, diagnosis: pd.DataFrame,
                           decomposition: Optional[pd.DataFrame] = None) -> None:
    _header(8, "SEGMENT LENGTHS, AND WHETHER THE SENSOR MOVED",
            "a rigid dimension of the subject — but a lossy view of the offsets behind it")
    if lengths.empty or diagnosis.empty:
        print("Not enough trials per subject to compare invariants.")
        return
    # Which invariants are stable, pooled over subjects. Read this before the per-subject
    # table: an invariant that moves for everyone is telling you about the joint fit behind it,
    # not about any one subject's capture.
    per_pair = (lengths.groupby(['segment', 'pair'], observed=True)
                .apply(lambda g: pd.Series({
                    'length_mm': g['length_mm'].median(),
                    'change_mm': (g.groupby('subject', observed=True)['length_mm']
                                  .apply(lambda s: s.max() - s.min()).median())}),
                       include_groups=False)
                .reset_index())
    print(f"  {'segment':<14}{'joint pair':<22}{'length':>10}{'between-trial change':>22}")
    print(f"  {'':<14}{'':<22}{'(mm)':>10}{'(mm, median over subjects)':>22}")
    for _, row in per_pair.sort_values('change_mm').iterrows():
        print(f"  {row['segment']:<14}{row['pair']:<22}{row['length_mm']:>10.1f}"
              f"{row['change_mm']:>22.1f}")
    print("\nA pair is only as good as the two joint fits behind it. Anything involving a poorly "
          "fitted\njoint inherits its instability and is not evidence about the capture — which "
          "is why the\nper-subject reading below uses the MEDIAN over pairs rather than the mean.")
    # The headline comparison, computed rather than asserted: the best-behaved half of the
    # invariants against how far the offsets themselves moved. Reported because it is what a
    # reader will notice anyway, immediately followed by why it does NOT mean what it looks like.
    steady = per_pair.nsmallest(max(len(per_pair) // 2, 1), 'change_mm')
    shift = float(diagnosis['offset_shift_mm'].median()) if not diagnosis.empty else np.nan
    steady_change = float(steady['change_mm'].median())
    if np.isfinite(shift) and steady_change > 0:
        print(f"\nThe steadiest invariants move {steady_change:.1f} mm between a subject's "
              f"trials, while the offsets\nthemselves move {shift:.1f} mm — a factor of "
              f"{shift / steady_change:.1f}.")
        print("\nTHAT FACTOR IS NOT EVIDENCE OF A RE-MOUNT, though this section used to read it "
              "as one. Length\ndepends only on the DIFFERENTIAL of the two centres' shifts (the "
              "common part cancels exactly),\nand it is second-order blind to whatever part of "
              "that differential is perpendicular to the\nsegment axis: 15 mm across a 450 mm "
              "femur costs 15^2/(2*450) = 0.25 mm of length. Two centres\nwandering "
              "independently and transversely preserve the length just as well as a rigid "
              "re-mount\ndoes. The table below measures the two parts instead of inferring them.")

    print(f"\n  {'subject':<9}{'trials':>7}{'length change':>15}{'offset shift':>14}"
          f"{'worst ratio':>13}{'spread':>9}  reading")
    print(f"  {'':<9}{'':>7}{'(mm, median)':>15}{'(mm)':>14}{'':>13}{'':>9}")
    for _, row in diagnosis.sort_values('subject').iterrows():
        print(f"  {row['subject']:<9}{int(row['n_trials']):>7}{row['length_change_mm']:>15.1f}"
              f"{row['offset_shift_mm']:>14.1f}{row['worst_ratio']:>13.3f}"
              f"{row['ratio_spread']:>9.3f}  {row['reading']}")

    counts = diagnosis['reading'].str.split(' ').str[0].value_counts().to_dict()
    print(f"\nAcross {len(diagnosis)} subjects: {counts}")
    print("\nWhat this table still settles is SCALE. Lengths changing by one common factor across "
          "independent\nsegments in one trial is a mocap calibration difference, not anatomy, and "
          "the `spread` column is\nwhat separates that from segment-specific trouble. 'Uneven' "
          "means neither explanation covers it\nand the trial deserves a look before its numbers "
          "are trusted.")

    if decomposition is not None and not decomposition.empty:
        print(f"\n  {'segment':<14}{'shift':>8}{'common':>8}{'diff':>8}{'diff':>8}{'diff':>8}"
              f"{'perp':>8}{'length':>8}")
        print(f"  {'':<14}{'(mm)':>8}{'(mm)':>8}{'(mm)':>8}{'along':>8}{'perp':>8}{'costs':>8}"
              f"{'(mm)':>8}")
        summary = (decomposition.groupby('segment', observed=True)
                   [['shift_mm', 'common_mm', 'differential_mm', 'diff_along_mm',
                     'diff_perp_mm', 'perp_costs_mm', 'length_mm']].median())
        for segment, row in summary.iterrows():
            print(f"  {segment:<14}{row['shift_mm']:>8.1f}{row['common_mm']:>8.1f}"
                  f"{row['differential_mm']:>8.1f}{row['diff_along_mm']:>8.1f}"
                  f"{row['diff_perp_mm']:>8.1f}{row['perp_costs_mm']:>8.2f}"
                  f"{row['length_mm']:>8.0f}")
        shift_mm = float(decomposition['shift_mm'].median())
        differential = float(decomposition['differential_mm'].median())
        common = float(decomposition['common_mm'].median())
        ratio = differential / shift_mm if shift_mm > 0 else np.nan
        print(f"\nPooled: shift {shift_mm:.1f} mm, common {common:.1f} mm, differential "
              f"{differential:.1f} mm — differential/shift {ratio:.2f}.")
        if ratio < 0.5:
            print("\nThe shifts ARE largely common, which is the re-mount signature: both of a "
                  "segment's centres\nmove together, so the sensor moved and the joints did not. "
                  "Each trial's offsets are then\ncorrect for their own trial and must not be "
                  "pooled.")
        else:
            print("\nTHE SHIFTS ARE NOT COMMON, so a re-mount is not what is happening. The two "
                  "centres of a segment\nmove as independently as two separately solved fits "
                  "would, and the length survives only because\nthe differential is mostly "
                  "TRANSVERSE — compare `diff perp` against `perp costs`, which is all\nthat "
                  "reaches the length. Read the wander as fit error (section 2's ill-determined "
                  "transverse\nmode) plus genuine posture dependence (section 11), not as the "
                  "sensor sitting somewhere else.")
        if spec.name == 'alborno':
            print("\nMEASURED DIRECTLY AND IT AGREES: the vector from each plate to an anatomical "
                  "marker on its own\nsegment — the mounting itself, no fit involved — moves a "
                  "median 4.5 mm between walking and\ncomplexTasks across all 8 subjects with "
                  "both, against offsets moving 15-20 mm.")


# ==============================================================================
# Console report
# ==============================================================================

def _has_columns(frame: pd.DataFrame, columns: Sequence[str], section: str) -> bool:
    """False, with a note, when a table on disk predates the columns a section needs.

    `--only-tables` makes mixed-vintage artifacts a normal state rather than an error: recomputing
    just one table leaves the others as they were, which is the point of the flag. A report that
    assumed every column existed raised KeyError halfway through instead of saying which section
    was stale.
    """
    missing = [c for c in columns if c not in frame.columns]
    if not missing:
        return True
    print(f"{section}: the table on disk predates {', '.join(missing[:3])}"
          f"{' and others' if len(missing) > 3 else ''} — rerun without --only-tables.")
    return False


def _header(number: Optional[int], title: str, subtitle: str) -> None:
    print("\n" + "=" * 96)
    # The headline section is deliberately unnumbered: it is the objective the numbered sections
    # are evidence about, not the first of them.
    print(f"{number}. {title}" if number is not None else title)
    print(f"   ({subtitle})")
    print("=" * 96)


def report_closure(spec: DatasetSpec, closure: pd.DataFrame, ambiguity: pd.DataFrame) -> None:
    _header(None, "CLOSURE BY OFFSET SOURCE — THE METRIC EVERYTHING ELSE SERVES",
            "how nearly two segments' offsets construct the SAME point, scored out of sample")
    if closure.empty:
        print("No closure table available.")
        return
    present = [s for s in SOURCE_ORDER if s in set(closure['source'])]
    print("  " + f"{'source':<15}{'closure':>10}{'vs floor':>10}{'offset':>10}{'free?':>8}"
                 f"{'n':>7}   what it was fitted on")
    print("  " + f"{'':<15}{'(mm)':>10}{'':>10}{'(mm)':>10}{'':>8}{'':>7}")
    floor = float(closure[closure['source'] == 'own']['closure_mm'].median())
    radius = float(ambiguity['worst_mm'].median()) if not ambiguity.empty else np.nan
    legend = {'own': 'this trial (IN SAMPLE — the floor)',
              'own_holdout': "this trial's first half, scored on its second",
              'cross_trial': "the subject's other trials only",
              'marker': 'nothing — anatomical landmarks',
              'inertial': 'gyro + accelerometer only, this trial',
              'inertial_ship': 'gyro + accelerometer, shipped solver'}
    for source in present:
        block = closure[closure['source'] == source]
        value = float(block['closure_mm'].median())
        offset = float(block['offset_from_own_mm'].median())
        # Is the offset difference inside the ambiguity ellipsoid, i.e. did it cost anything?
        free = '' if not np.isfinite(offset) else ('yes' if offset <= radius else 'no')
        print(f"  {source:<15}{value:>10.2f}{value / floor:>9.2f}x"
              f"{offset:>10.1f}{free:>8}{len(block):>7}   {legend.get(source, '')}")

    print(f"\nThe floor is {floor:.2f} mm and it is NOT achievable-by-trying: `own` is scored on "
          f"the data it was\nfitted on, so what it measures is the part of the pair's motion a "
          f"fixed-centre ball joint cannot\ndescribe at all — soft tissue, a knee that translates "
          f"as it flexes, marker error. Every other row\nis out of sample and the honest question "
          f"is how far above the floor it lands.")
    if np.isfinite(radius):
        print(f"\nWHY A LARGE OFFSET DIFFERENCE CAN BE CHEAP, which is the single most "
              f"counter-intuitive thing\nhere. The offsets live in a space where closure is "
              f"almost flat in some directions: the worst\nambiguity radius is {radius:.1f} mm, "
              f"meaning the pair can slide that far for one millimetre of\nclosure. An offset "
              f"difference smaller than that is nearly free, and the `free?` column says\nwhich "
              f"rows are in that regime. It is also why the offset and closure columns disagree "
              f"so\nwidely on which source is best.")
    print("\nAND THAT FLATNESS IS NOT A DEFECT HERE. Closure is blind to a simultaneous slide of "
          "both offsets\nalong the joint axis — and so is the downstream, because two segments "
          "that construct the same\npoint are both computing the acceleration of that one point "
          "whether or not it is anatomically\nthe joint. The metric's null space and the "
          "application's null space are the same, which is what\nmakes closure the objective "
          "rather than a proxy for one.")


def report_ambiguity(spec: DatasetSpec, ambiguity: pd.DataFrame, fits: pd.DataFrame) -> None:
    _header(2, "HOW AMBIGUOUS THE OFFSETS ARE",
            "the displacement that costs one millimetre of closure, per direction, in mm")
    if ambiguity.empty:
        print("No normal equations stored; run without --report-only first.")
        return
    excitation = (fits.groupby('joint', observed=True)['excitation'].mean()
                  if not fits.empty and 'excitation' in fits.columns else pd.Series(dtype=float))
    print(f"  {'joint':<11}{'worst':>9}{'median':>9}{'best':>8}{'anisotropy':>12}"
          f"{'excitation':>12}{'trials':>8}")
    print(f"  {'':<11}{'(mm)':>9}{'(mm)':>9}{'(mm)':>8}{'worst/best':>12}{'':>12}{'':>8}")
    summary = ambiguity.groupby('joint', observed=True).agg(
        worst=('worst_mm', 'median'), median=('median_mm', 'median'),
        best=('best_mm', 'median'), anisotropy=('anisotropy', 'median'), n=('worst_mm', 'size'))
    for joint, row in summary.sort_values('worst', ascending=False).iterrows():
        value = excitation.get(joint, np.nan)
        shown = f"{value:>12.4f}" if np.isfinite(value) else f"{'':>12}"
        print(f"  {joint:<11}{row['worst']:>9.1f}{row['median']:>9.2f}{row['best']:>8.2f}"
              f"{row['anisotropy']:>12.1f}{shown}{int(row['n']):>8}")

    print("\nStarting from the optimum the residual is EXACTLY quadratic, RSS(x*+d) = RSS(x*) + "
          "d^T G d, so\nthe displacement along a unit eigenvector that costs one millimetre of "
          "closure is sqrt(n/lambda)\nmillimetres. This table is that number — the condition "
          "number re-expressed in the units the\noffsets are actually quoted in, which is the "
          "form in which it can be compared against anything.")
    print("\nThe eigenvalues are exactly N +- sigma_i(sum_t R_parent^T R_child), so the geometry "
          "is fixed by the\nRELATIVE ROTATION and nothing else. A pair that never moves relative "
          "to each other has\nsigma_max = N, an infinite radius and no fit at all. For a PURE "
          "HINGE about axis n, R_rel(t) n = n\nat every sample, so n has sigma = N exactly and "
          "the offset component along it is unobservable\nhowever long the trial runs — an "
          "infinite radius in one specific, identifiable direction.")
    print("\nRead `worst` as the tolerance the metric grants you and `anisotropy` as how "
          "lopsided that grant\nis. A large worst radius is not bad news on its own: it says the "
          "data never pinned that direction,\nand for a relative filter it never needed to. It "
          "IS bad news for any downstream that needs the\ncentre anatomically placed.")


def report_fit_quality(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(1, "JOINT-CENTRE FIT QUALITY",
            "residual = the separation of the two segments' implied joint centres, per trial")
    if fits.empty:
        print("No fits found.")
        return
    usable = fits[fits['converged']]
    print(f"  {'joint':<12}{'|r_parent|':>12}{'|r_child|':>11}{'residual':>11}{'median':>10}"
          f"{'p95':>10}{'as % of r':>11}{'trials':>8}")
    print(f"  {'':<12}{'(mm)':>12}{'(mm)':>11}{'RMS (mm)':>11}{'(mm)':>10}{'(mm)':>10}{'':>11}{'':>8}")
    for joint in [j for j in spec.joints if j in set(usable['joint'])]:
        rows = usable[usable['joint'] == joint]
        means = rows[['parent_norm_mm', 'child_norm_mm', 'residual_rms_mm',
                      'residual_median_mm', 'residual_p95_mm']].mean()
        arm = 0.5 * (means['parent_norm_mm'] + means['child_norm_mm'])
        share = 100 * means['residual_rms_mm'] / arm if arm else np.nan
        print(f"  {joint:<12}{means['parent_norm_mm']:>12.1f}{means['child_norm_mm']:>11.1f}"
              f"{means['residual_rms_mm']:>11.2f}{means['residual_median_mm']:>10.2f}"
              f"{means['residual_p95_mm']:>10.2f}{share:>10.1f}%{len(rows):>8}")

    failed = fits[~fits['converged']]
    if not failed.empty:
        print(f"\n{len(failed)} joint-trial(s) had fewer than {MIN_FIT_FRAMES} valid frames and "
              f"were not fitted.")
    print("\nThe residual is NOT sensor noise. It is the amount by which a fixed-centre ball joint")
    print("fails to describe the pair — soft-tissue motion, a knee that translates as it flexes,")
    print("marker reconstruction error and residual sensor-to-segment misalignment all land in it.")
    print("Read it against the lever arm in the two columns to its left: that ratio is the fraction")
    print("of the projection distance the joint model cannot account for.")


def report_conditioning(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(2, "HOW WELL-DETERMINED THE OFFSETS ARE",
            "excitation = 1 - sigma_max(sum R_p^T R_c)/N; 0 means the pair never moved relative "
            "to each other")
    if fits.empty:
        print("No fits found.")
        return
    usable = fits[fits['converged']]
    print(f"  {'joint':<12}{'excitation':>12}{'condition':>12}{'worst dir (parent)':>26}"
          f"{'|dot| hinge':>13}")
    for joint in [j for j in spec.joints if j in set(usable['joint'])]:
        rows = usable[usable['joint'] == joint]
        direction = rows[['worst_parent_x', 'worst_parent_y', 'worst_parent_z']].mean().to_numpy()
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm else direction
        condition = rows['condition_number'].replace(np.inf, np.nan).median()
        hinge = rows['hinge_alignment'].mean()
        print(f"  {joint:<12}{rows['excitation'].mean():>12.4f}{condition:>12.0f}"
              f"{'[' + ', '.join(f'{v:+.2f}' for v in direction) + ']':>26}{hinge:>13.2f}")

    print("\nThe conditioning of this fit is set ENTIRELY by the relative rotation between the two")
    print("segments, and in closed form: stacking the constraint gives a Gram matrix whose")
    print("eigenvalues are exactly N +- sigma_i(sum_t R_parent^T R_child). Two things follow.")
    print("\nA pair that never moves relative to each other has sigma_max = N, excitation 0 and a")
    print("singular fit — correct, since a rigid pair carries no information about where the joint")
    print("between them is. For a PURE HINGE about axis n, R_rel(t) n = n at every sample, so n is")
    print("a singular vector with sigma = N exactly and the offset component along the hinge axis")
    print("is unobservable however long the trial runs. The last column tests that: it is the")
    print("alignment between the measured worst-determined direction and the joint axis from")
    print("get_primary_joint_axis, so a value near 1 says the degeneracy is the hinge and a middling")
    print("value says something else is limiting the fit.")
    print("\nThe worst-determined mode is a SIMULTANEOUS slide of both offsets, not a per-segment")
    print("error: the assumed joint centre moves and both frames absorb it, which is why the")
    print("residual can be small while the offsets themselves are poorly placed.")


def report_masking(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(3, "WHAT THE valid MASK IS WORTH",
            "how far the offsets move when padded, non-measured poses are excluded from the fit")
    usable = fits[fits['converged']] if not fits.empty else fits
    if usable.empty or 'unmasked_shift_mm' not in usable.columns:
        print("No fits found.")
        return
    coverage = usable['n_valid'] / usable['n_samples'].clip(lower=1)
    print(f"  {'joint':<12}{'valid %':>10}{'offset shift':>14}{'worst':>10}"
          f"{'residual masked':>17}{'unmasked':>11}")
    print(f"  {'':<12}{'':>10}{'(mm, mean)':>14}{'(mm)':>10}{'(mm RMS)':>17}{'(mm RMS)':>11}")
    for joint in [j for j in spec.joints if j in set(usable['joint'])]:
        rows = usable[usable['joint'] == joint]
        share = 100 * (rows['n_valid'] / rows['n_samples'].clip(lower=1)).mean()
        print(f"  {joint:<12}{share:>10.1f}{rows['unmasked_shift_mm'].mean():>14.2f}"
              f"{rows['unmasked_shift_mm'].max():>10.2f}"
              f"{rows['residual_rms_mm'].mean():>17.2f}"
              f"{rows['unmasked_residual_rms_mm'].mean():>11.2f}")

    print(f"\nWorst-case shift across every joint-trial: {usable['unmasked_shift_mm'].max():.2f} mm "
          f"(mean {usable['unmasked_shift_mm'].mean():.2f} mm).")
    print(f"Valid coverage runs {100 * coverage.min():.0f}-{100 * coverage.max():.0f}% of frames.")
    print("\nOutside `valid` the world trace holds a constant padded pose — a placeholder left "
          "where marker\nreconstruction failed, not a measurement. Such frames are not merely "
          "uninformative: one pose\nrepeated for thousands of samples is thousands of identical "
          "rows in the least squares, which the\nsolution is dragged toward satisfying exactly.")
    print("\nThis table is what that was worth. `WorldTrace.get_joint_center` used to fit over "
          "every frame and\nnow masks by `valid` by default, so the numbers here are the size of "
          "a defect that has been\nfixed rather than one still present — the unmasked column is "
          "reproduced deliberately, by fitting\nonce with an all-True mask, to keep the "
          "measurement available.")
    print("\nBoth residual columns are scored on the SAME valid frames, so the difference between "
          "them is the\ncost of the padded rows and not a change of scoring set.")


def report_generalization(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(4, "DOES THE FIXED-CENTRE MODEL GENERALIZE?",
            "fit on the first half of the valid frames, scored on the second")
    usable = (fits[fits['converged'] & np.isfinite(fits['residual_holdout_mm'])]
              if not fits.empty else fits)
    if usable.empty:
        print("No trial had enough valid frames for a holdout split.")
        return
    print(f"  {'joint':<12}{'in-sample':>12}{'held-out':>11}{'ratio':>9}{'trials':>8}")
    print(f"  {'':<12}{'(mm RMS)':>12}{'(mm RMS)':>11}{'':>9}{'':>8}")
    for joint in [j for j in spec.joints if j in set(usable['joint'])]:
        rows = usable[usable['joint'] == joint]
        in_sample = rows['residual_in_sample_mm'].mean()
        held = rows['residual_holdout_mm'].mean()
        print(f"  {joint:<12}{in_sample:>12.2f}{held:>11.2f}"
              f"{held / in_sample if in_sample else np.nan:>9.2f}{len(rows):>8}")
    overall = usable['residual_holdout_mm'].mean() / usable['residual_in_sample_mm'].mean()
    print(f"\nPooled held-out/in-sample ratio: {overall:.2f}.")
    # Does the ratio depend on how long the trial is? If it does, the split is picking up a
    # change of task between the halves rather than a failure of the offsets to generalize.
    ratios = usable['residual_holdout_mm'] / usable['residual_in_sample_mm'].replace(0, np.nan)
    with np.errstate(invalid='ignore'):
        duration_correlation = np.corrcoef(np.log(usable['n_valid'].clip(lower=1)),
                                           ratios.fillna(1.0))[0, 1]
    print(f"Correlation of that ratio with log(valid frames): {duration_correlation:+.2f}.")
    print("\nCAVEAT on this section, and it bites hardest on short single-task recordings. A time")
    print("split assumes the two halves sample the same joint behaviour. In a 40 s sit-to-stand they")
    print("plainly do not — the first half is one posture and the second is another — so a large")
    print("ratio there is the two halves differing, not the offsets failing to transfer. A negative")
    print("correlation above says exactly that: the shorter the trial, the worse the ratio. Compare")
    print("datasets on this number only when their trials are of comparable length and homogeneity.")
    print("\nSix free parameters will reduce any residual, so the in-sample column alone cannot say")
    print("whether the model describes the joint. A ratio near 1 says the offsets generalize to")
    print("poses the fit never saw and the residual is real joint behaviour; a large ratio says the")
    print("fit is absorbing the trial. The split is in TIME, not at random — a random split leaves")
    print("both halves covering the same poses and tests nothing.")


def report_stability(spec: DatasetSpec, stability: pd.DataFrame) -> None:
    _header(5, "BETWEEN-TRIAL STABILITY OF THE OFFSET",
            "the same subject and joint, fitted independently on each of their trials")
    if stability.empty:
        print("No subject has two or more trials for any joint.")
        return
    print(f"  {'joint':<12}{'parent spread':>15}{'child spread':>14}{'worst':>10}"
          f"{'excitation':>12}{'subjects':>10}")
    print(f"  {'':<12}{'(mm)':>15}{'(mm)':>14}{'(mm)':>10}{'':>12}{'':>10}")
    for joint in [j for j in spec.joints if j in set(stability['joint'])]:
        rows = stability[stability['joint'] == joint]
        worst = max(rows['parent_max_dev_mm'].max(), rows['child_max_dev_mm'].max())
        print(f"  {joint:<12}{rows['parent_spread_mm'].mean():>15.1f}"
              f"{rows['child_spread_mm'].mean():>14.1f}{worst:>10.1f}"
              f"{rows['mean_excitation'].mean():>12.4f}{len(rows):>10}")

    print("\nThis is the error bar to quote when a downstream experiment uses one trial's offsets.")
    print("The offset is anatomy plus mounting, both constant within a session, so everything in")
    print("this spread is fit error. Note how much larger it is than the residual in section 1:")
    print("the residual says how well six parameters fit one trial, this says whether they are the")
    print("SAME six parameters next time — and the worst-determined mode from section 2 is free to")
    print("wander between trials without the residual noticing.")

    # PRIMARY joints only. A placement variant spans the same segment pair as its canonical
    # joint, so it has the SAME relative rotation and therefore a bit-identical excitation —
    # including all 18 correlates three different spreads against one repeated x-value, which
    # drives the coefficient toward zero for a reason that has nothing to do with the physics.
    primary = stability[stability['joint'].isin(spec.primary_joints)]
    by_joint = primary.groupby('joint', observed=True)[
        ['mean_excitation', 'parent_spread_mm']].mean()
    correlation = (by_joint.corr().iloc[0, 1] if len(by_joint) > 2 else np.nan)
    span = (by_joint['mean_excitation'].max() / max(by_joint['mean_excitation'].min(), 1e-12)
            if len(by_joint) > 2 else np.nan)
    print(f"\nCorrelation between a joint's mean excitation and its offset spread, over the "
          f"{len(by_joint)} primary\njoints: {correlation:+.2f}, across a {span:.1f}x span in "
          f"excitation.")
    print("\nSection 2 predicts this should be NEGATIVE: a better-conditioned fit has less room to")
    print("wander between trials. THE TWO DATASETS DISAGREE ON IT. Al Borno gives -0.90 over a 3.6x")
    print("span in excitation, which supports the prediction; IMoVE gives +0.80 over a 1.5x span,")
    print("which does not. Do not quote either as settled.")
    print("\nNeither is strong evidence on its own — six or seven points, and IMoVE in particular")
    print("has almost no contrast in x for the correlation to work with, because its six joints sit")
    print("within a factor of 1.5 of each other in excitation. The prediction is about a mechanism")
    print("that is exactly true in the algebra; what is untested is whether excitation is the")
    print("DOMINANT term in between-trial spread, and on this evidence it is not established.")


def report_cost(spec: DatasetSpec, fits: pd.DataFrame, stability: pd.DataFrame) -> None:
    _header(6, "WHAT AN OFFSET ERROR COSTS THE ACCELERATION PROJECTION",
            "|da| <= (|alpha| + |omega|^2) |dr|, with the gain measured per trial")
    usable = fits[fits['converged'] & np.isfinite(fits['gain_median'])] if not fits.empty else fits
    if usable.empty:
        print("No fits found.")
        return
    spread = (stability.groupby('joint', observed=True)[['parent_spread_mm', 'child_spread_mm']]
              .mean().max(axis=1) if not stability.empty else pd.Series(dtype=float))
    print(f"  {'joint':<12}{'gain':>10}{'gain p95':>11}{'offset err':>12}{'-> acc err':>12}"
          f"{'p95':>10}{'vs acc_std':>12}")
    print(f"  {'':<12}{'(1/s^2)':>10}{'(1/s^2)':>11}{'(mm)':>12}{'(m/s^2)':>12}{'(m/s^2)':>10}"
          f"{'':>12}")
    acc_std = pipeline_constants()['acc_std']
    for joint in [j for j in spec.joints if j in set(usable['joint'])]:
        rows = usable[usable['joint'] == joint]
        gain, gain95 = rows['gain_median'].mean(), rows['gain_p95'].mean()
        error_mm = spread.get(joint, np.nan)
        acc_error = gain * error_mm / 1000.0
        print(f"  {joint:<12}{gain:>10.1f}{gain95:>11.1f}{error_mm:>12.1f}{acc_error:>12.3f}"
              f"{gain95 * error_mm / 1000.0:>10.3f}{acc_error / acc_std:>11.0f}x")

    print("\nThis is the translation into the currency experiments/acceleration_projection.py works")
    print("in. project_acc adds alpha x r + omega x (omega x r), both linear in r, so an offset")
    print("error dr propagates with a gain of at most |alpha| + |omega|^2 — computed here from the")
    print("MEASURED omega and its backward derivative, per trial, not from a nominal.")
    print("\nThe offset error used is the BETWEEN-TRIAL spread from section 5, not the in-trial")
    print("residual, because that is the quantity that behaves like an error in r: the residual is")
    print("a model mismatch that the fit has already absorbed into the offsets it returns.")
    print("\nIt is an upper bound — the two terms are perpendicular to different things and")
    print("partially cancel for any given dr — so read it as a budget, not an estimate. The last")
    print("column is that budget against the acc_std the filter is tuned with, which is what says")
    print("whether the joint-centre fit is a limiting error source or a rounding one.")


def report_precision(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(9, "HOW PRECISE ARE THE OFFSETS",
            "standard errors from sigma^2 (M^T M)^-1, corrected for the residual's autocorrelation")
    if fits.empty or not _has_columns(fits, ['se_worst_mm', 'effective_n'], "Section 9"):
        return
    usable = fits[fits['converged'] & np.isfinite(fits['se_worst_mm'])]
    if usable.empty:
        print("No precision estimates.")
        return
    axes = [f"se_{c}_mm" for c in OFFSET_COLUMNS]
    # 'parent_x' -> 'par.x', 'child_z' -> 'chi.z'
    print(f"  {'joint':<11}" + "".join(f"{c[:3] + '.' + c[-1]:>9}" for c in OFFSET_COLUMNS)
          + f"{'worst':>9}{'naive':>8}{'N_eff/N':>9}")
    print(f"  {'':<11}" + "".join(f"{'(mm)':>9}" for _ in OFFSET_COLUMNS)
          + f"{'(mm)':>9}{'(mm)':>8}{'':>9}")
    for joint in [j for j in spec.joints if j in set(usable['joint'])]:
        rows = usable[usable['joint'] == joint]
        cells = "".join(f"{rows[c].mean():>9.1f}" for c in axes)
        share = (rows['effective_n'] / rows['n_valid']).mean()
        print(f"  {joint:<11}{cells}{rows['se_worst_mm'].mean():>9.1f}"
              f"{rows['se_worst_naive_mm'].mean():>8.1f}{share:>9.3f}")

    print(f"\nThe naive column is what sigma^2 (M^T M)^-1 gives directly, and it is "
          f"OVER-CONFIDENT by the\nfactor in the last column: the residual is a smooth function "
          f"of pose, so consecutive frames are\nnearly identical and the effective sample size is "
          f"a small fraction of N. Measured "
          f"{usable['autocorr_time'].mean():.0f}-frame\nautocorrelation time, so N_eff/N is "
          f"~{(usable['effective_n'] / usable['n_valid']).mean():.3f}. The corrected `worst` "
          f"column is the number to quote.")
    print("\n`worst` is the largest eigenvalue direction of the covariance, not any single axis — "
          "the\nill-determined mode is a combination, as section 2 shows. Compare it against the "
          "between-trial\nspread in section 5: where the spread is much larger, something other "
          "than measurement noise is\nmoving the offsets and no amount of data will fix it. On "
          "Al Borno the spread is 4-6x this, and\nsection 8 narrows the cause to the "
          "ill-determined transverse mode plus posture dependence.")


def report_learning(spec: DatasetSpec, curve: pd.DataFrame) -> None:
    _header(10, "HOW MUCH DATA A FIT NEEDS",
            "distance from the same trial's full fit, by frame count — convergence, not accuracy")
    if curve.empty:
        print("No learning curves.")
        return
    pooled = curve.groupby('n_frames', observed=True)['error_mm'].agg(['median', 'mean', 'size'])
    print(f"  {'frames':>9}{'seconds @100Hz':>16}{'median err':>12}{'mean err':>11}{'n':>7}")
    for count, row in pooled.iterrows():
        print(f"  {int(count):>9}{count / 100.0:>16.1f}{row['median']:>12.1f}"
              f"{row['mean']:>11.1f}{int(row['size']):>7}")

    within = pooled[pooled['median'] <= 5.0]
    if not within.empty:
        need = int(within.index[0])
        print(f"\nMedian error falls under 5 mm at {need} frames ({need / 100.0:.0f} s at 100 Hz).")
    else:
        print(f"\nMedian error never falls under 5 mm within the ladder — the shortest tested "
              f"prefix is\nalready {pooled['median'].iloc[0]:.0f} mm from the full fit and the "
              f"longest {pooled['median'].iloc[-1]:.0f} mm.")
    print("\nThis is CONVERGENCE, not accuracy: the reference is the same trial's own full fit, so "
          "a flat\ncurve says more data would not move the answer — not that the answer is right. "
          "Read it with\nsection 5, which measures whether the converged answer is the same one "
          "next trial.")
    print("\nIt is also what separates duration from task content in IMoVE's poorer fits. If the "
          "curve is\nflat well below those trials' length, their problem is what the subject was "
          "doing, not how long\nfor.")


def report_migration(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(11, "DOES THE CENTRE MOVE WITH FLEXION?",
            "residual regressed on the relative rotation angle, in the parent's frame")
    if fits.empty or not _has_columns(fits, ['migration_mm_per_deg', 'migration_r2'],
                                      "Section 11"):
        return
    usable = fits[fits['converged'] & np.isfinite(fits['migration_mm_per_deg'])]
    if usable.empty:
        print("No migration estimates.")
        return
    print(f"  {'joint':<11}{'flexion range':>15}{'migration':>12}{'over range':>12}{'R^2':>8}"
          f"{'residual':>10}")
    print(f"  {'':<11}{'(deg)':>15}{'(mm/deg)':>12}{'(mm)':>12}{'':>8}{'(mm RMS)':>10}")
    for joint in [j for j in spec.joints if j in set(usable['joint'])]:
        rows = usable[usable['joint'] == joint]
        span = rows['flexion_range_deg'].mean()
        slope = rows['migration_mm_per_deg'].mean()
        print(f"  {joint:<11}{span:>15.1f}{slope:>12.3f}{slope * span:>12.1f}"
              f"{rows['migration_r2'].mean():>8.2f}{rows['residual_rms_mm'].mean():>10.2f}")

    print("\nThe direct test of the fixed-centre assumption. Sections 1 and 8 establish that the "
          "model\nfails; this says whether it fails AS A FUNCTION OF POSTURE. A joint that "
          "translates as it flexes\nputs a linear term in the residual; noise does not.")
    print("\nRead the slope and R^2 together. A large slope with low R^2 is a coincidence of "
          "range; both\ntogether is a translating joint. `over range` is the slope times the "
          "flexion actually observed —\nthe migration in millimetres a trial would have to "
          "absorb, which is the number comparable to the\nresidual beside it. Where the two are "
          "close, migration IS the residual and a fixed centre is the\nwrong model rather than a "
          "noisy one.")


def report_symmetry(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(12, "LEFT/RIGHT SYMMETRY",
            "a per-trial invariant: one subject's paired joints should agree")
    usable = fits[fits['converged']] if not fits.empty else fits
    if usable.empty:
        print("No fits.")
        return
    rows = []
    for (subject, trial), group in usable.groupby(['subject', 'trial'], observed=True):
        by_joint = group.set_index('joint')
        for joint in by_joint.index:
            other = contralateral(str(joint))
            # Once per pair, and only pairs where both sides fitted.
            if other is None or other not in by_joint.index or str(joint) > other:
                continue
            left, right = by_joint.loc[joint], by_joint.loc[other]
            rows.append({'pair': f"{joint}/{other}",
                         'parent_diff_mm': abs(left['parent_norm_mm'] - right['parent_norm_mm']),
                         'child_diff_mm': abs(left['child_norm_mm'] - right['child_norm_mm']),
                         'arm_mm': 0.5 * (left['parent_norm_mm'] + right['parent_norm_mm'])})
    table = pd.DataFrame(rows)
    if table.empty:
        print("No joint has both sides fitted.")
        return
    print(f"  {'pair':<22}{'|r_parent| diff':>17}{'|r_child| diff':>16}{'as % of arm':>13}{'n':>6}")
    print(f"  {'':<22}{'(mm)':>17}{'(mm)':>16}{'':>13}{'':>6}")
    for pair, group in table.groupby('pair', observed=True):
        share = 100 * group['parent_diff_mm'].mean() / group['arm_mm'].mean()
        print(f"  {pair:<22}{group['parent_diff_mm'].mean():>17.1f}"
              f"{group['child_diff_mm'].mean():>16.1f}{share:>12.1f}%{len(group):>6}")

    print(f"\nPooled |r| difference between sides: {table['parent_diff_mm'].mean():.1f} mm parent, "
          f"{table['child_diff_mm'].mean():.1f} mm child.")
    print("\nA free invariant, and unlike section 8's segment lengths it needs only ONE trial, so "
          "the\nbetween-trial effects that section confounds cannot explain it. But it does NOT "
          "isolate fit error:\nthree things contribute and this cannot separate them.")
    print("  * ANATOMY. Limbs are not perfectly symmetric.")
    print("  * PLACEMENT. The sensors are taped on, so the left one is not a mirror of the right — "
          "and this\n    is the offset from the SENSOR to the joint, so a placement difference "
          "enters it directly.")
    print("  * FIT ERROR.")
    print("\nWhere it sits is still informative: below the between-trial spread in section 5 and "
          "above the\nstandard errors in section 9, so the first two terms are real and the fit "
          "is looser than its own\ncovariance admits. Attributing the split needs anatomical "
          "landmarks, which would pin the joint\nindependently of the sensor.")


# ==============================================================================
# Comparison against anatomical landmarks
# ==============================================================================

def landmark_comparison(plates: Dict[str, PlateTrial], spec: DatasetSpec, dataset: str,
                        subject: str, trial: str) -> pd.DataFrame:
    """The fitted offsets against a landmark-derived centre, with BOTH error budgets.

    NOT A VALIDATION AGAINST TRUTH, and the section is written so it cannot be read as one. The
    landmark centre carries two errors of its own:

      * NOISE — soft tissue moving the marker relative to the bone. Visible as the scatter of the
        landmark in the segment frame, reported here as `landmark_sd_mm`. Measured at 2-7 mm, and
        it depends on which segment you view from: hjc_r scatters 2.4 mm in the pelvis frame and
        6.2 mm in the femur frame, because a thigh plate sits on softer tissue than a pelvis one.
      * PLACEMENT — the marker was taped somewhere slightly wrong. This is INVISIBLE to the
        scatter: a marker 10 mm out of position sits 10 mm out constantly, so `landmark_sd_mm` is
        a FLOOR on the landmark's error and not an estimate of it. Unknown and systematic.

    So `disagreement_mm` bounds neither estimate's error on its own. What it can do is flag where
    two independent methods disagree by more than their combined scatter, which is a real
    inconsistency, and confirm where they agree, which is weak mutual support rather than proof.

    Sync-free: the landmark and the plate come from the same mocap file at the same frames, so this
    is computed at native mocap rate with no lag, resampling or take merging (see
    building/landmarks.py). Nothing is persisted but these scalars.
    """
    from src.toolchest.building import landmarks as landmark_reader
    try:
        landmark_spec = landmark_reader.get_spec(dataset)
        timestamps, markers = landmark_spec.read(subject, trial)
    except Exception:
        return pd.DataFrame()
    if not markers:
        return pd.DataFrame()

    # A landmark is only expressed in the frames of the two segments bordering ITS joint.
    # Elsewhere it is separated by a moving joint and its scatter is meaningless — ankle_r in the
    # femur frame scatters 73-114 mm, which says nothing about the ankle marker.
    def bordering(centres: Dict[str, Sequence[str]]) -> Dict[str, list]:
        out: Dict[str, list] = {}
        for joint, names in centres.items():
            for sensor in spec.joints.get(joint, ()):
                out.setdefault(sensor, []).extend(names)
        return out

    wanted = bordering(landmark_spec.centres)
    static_wanted = bordering(landmark_spec.static_centres)
    if not wanted and not static_wanted:
        return pd.DataFrame()

    # THE STANDING CAPTURE FIRST, because its templates set the basis for everything after it.
    # Letting the trial estimate its own is usually a constant re-basing that cancels (0.08 deg,
    # 0.02 mm median), but on a poorly conditioned plate it is not constant and does not cancel —
    # Subject08's pelvis moves |r| by 13.3 mm. See building/landmarks.read_alborno_plates.
    static = None
    if landmark_spec.read_static is not None and static_wanted:
        try:
            static = landmark_spec.read_static(subject, sorted(static_wanted))
        except Exception:
            static = None

    if dataset == 'alborno':
        poses, _ = landmark_reader.read_alborno_plates(
            landmark_reader._alborno_trc(subject, trial),
            {sensor: sensor for sensor in set(wanted) | set(static_wanted)},
            templates=(static.templates if static is not None else None))
    else:
        from src.toolchest.building.imove_mocap import load_world_traces
        traces = load_world_traces(landmark_reader._imove_csv(subject, trial))
        # IMoVE mocap is keyed by SEGMENT ('THIGH_R'); the spec names SENSORS ('THIGH_R_M'), and
        # all placements on a segment share its pose.
        poses = {sensor: (traces[segment].positions, traces[segment].rotations,
                          np.asarray(traces[segment].valid, dtype=bool))
                 for sensor in wanted
                 if (segment := sensor.rsplit('_', 1)[0]) in traces}

    trial_table = (landmark_reader.in_segment_frames(timestamps, markers, poses, wanted)
                   if wanted else pd.DataFrame())
    static_table = (landmark_reader.in_segment_frames(
        static.timestamps, static.markers, static.poses, static_wanted)
        if static is not None else pd.DataFrame())
    if trial_table.empty and static_table.empty:
        return pd.DataFrame()

    # THE ORIENTATION FIT, RE-RUN ON THESE SAME RAW POSES. Not read from `plates`, and this is the
    # whole reason the vector comparison is possible: the built plates carry a sensor-to-segment
    # alignment (171-178 deg on Al Borno) that a raw template reconstruction does not, so an offset
    # from one and a landmark from the other are the same vector in different bases. Verified
    # equivalent — |r| from the raw and built fits agrees to 0.00 mm — so nothing is lost by
    # refitting here, and both quantities end up in one frame.
    raw_traces = {sensor: WorldTrace(timestamps[:len(pose[0])], pose[0], pose[1], valid=pose[2])
                  for sensor, pose in poses.items()}
    fitted = {}
    for joint in set(landmark_spec.centres) | set(landmark_spec.static_centres):
        parent_sensor, child_sensor = spec.joints.get(joint, (None, None))
        if parent_sensor not in raw_traces or child_sensor not in raw_traces:
            continue
        try:
            r_parent, r_child, _ = raw_traces[parent_sensor].get_joint_center(
                raw_traces[child_sensor], min_frames=MIN_FIT_FRAMES)
        except Exception:
            continue
        fitted[joint] = {'parent': r_parent, 'child': r_child}

    rows = []

    def emit(centres: Dict[str, Sequence[str]], table: pd.DataFrame, source: str,
             min_frames: int) -> None:
        """Append one row per (joint, bordering segment) for a set of centres.

        `source` distinguishes a centre measured during the trial from one measured standing.
        They are NOT interchangeable and the report must not average them: the standing scatter
        omits motion-driven soft tissue entirely, so it runs ~10x smaller for the same landmark.
        """
        if table.empty:
            return
        usable = table[table['tracked'] & table['plate_valid']]
        for joint, names in centres.items():
            if joint not in fitted or joint not in spec.joints:
                continue
            for sensor, role in zip(spec.joints[joint], ('parent', 'child')):
                block = usable[usable['segment'] == sensor]
                present = [n for n in names if n in set(block['landmark'])]
                if len(present) != len(names):
                    continue
                # The centre: one marker where the file holds a computed centre, the midpoint of a
                # medial/lateral pair otherwise. Averaged per frame so a dropout on either marker
                # drops the frame rather than biasing toward the survivor.
                per_marker = [block[block['landmark'] == n].set_index('timestamp')[['x', 'y', 'z']]
                              for n in present]
                joined = per_marker[0]
                for other in per_marker[1:]:
                    joined = joined.join(other, how='inner', rsuffix='_o')
                if len(joined) < min_frames:
                    continue
                stacked = joined.to_numpy(dtype=float).reshape(len(joined), len(present), 3)
                centre = stacked.mean(axis=1)
                spread = np.linalg.norm(stacked[:, 0] - stacked[:, -1], axis=1) \
                    if len(present) > 1 else np.zeros(len(centre))
                rows.append(_landmark_row(joint, sensor, role, source, present, joined,
                                          centre, spread, fitted[joint][role] * 1000.0))

    emit(landmark_spec.centres, trial_table, 'trial', MIN_FIT_FRAMES)
    emit(landmark_spec.static_centres, static_table, 'static',
         landmark_reader.MIN_STATIC_FRAMES)
    return pd.DataFrame(rows)


def _landmark_row(joint: str, sensor: str, role: str, source: str, present: Sequence[str],
                  joined: pd.DataFrame, centre: np.ndarray, spread: np.ndarray,
                  offset_mm: np.ndarray) -> Dict[str, object]:
    """One landmark-vs-fit comparison, both vectors already in the same segment basis."""
    median = np.median(centre, axis=0)
    return {
        'joint': joint, 'segment': sensor, 'role': role, 'source': source,
        'landmarks': '+'.join(present), 'n_frames': int(len(joined)),
        'landmark_median_mm': float(np.linalg.norm(median)),
        'fitted_norm_mm': float(np.linalg.norm(offset_mm)),
        # Both in the same basis, so the full vector difference is meaningful. The magnitude-only
        # version is kept beside it because the two answer different questions: |r| agreeing while
        # the vectors do not means the two methods place the centre at the same DISTANCE in a
        # different DIRECTION, which is a far more specific finding than a single scalar.
        'disagreement_mm': float(np.linalg.norm(median - offset_mm)),
        'angle_deg': float(np.degrees(np.arccos(np.clip(
            np.dot(median, offset_mm)
            / max(np.linalg.norm(median) * np.linalg.norm(offset_mm), 1e-9), -1.0, 1.0)))),
        'arm_disagreement_mm': float(abs(np.linalg.norm(median) - np.linalg.norm(offset_mm))),
        # The landmark's own noise — a FLOOR on its error, blind to placement bias, and for a
        # 'static' row a much lower floor still, since a standing subject has no motion-driven
        # soft-tissue movement to show.
        'landmark_sd_mm': float(np.linalg.norm(centre.std(axis=0))),
        # Inter-marker distance: near-constant for a correctly labelled pair, so its scatter is a
        # label-swap and dropout detector. Zero for a single-marker centre.
        'pair_distance_mm': float(np.median(spread)),
        'pair_distance_sd_mm': float(spread.std()),
    }


def report_landmarks(spec: DatasetSpec, comparison: pd.DataFrame) -> None:
    _header(13, "AGAINST ANATOMICAL LANDMARKS",
            "two independent estimates of the same point — NOT a validation against truth")
    if comparison.empty:
        print("No landmark comparison available for this dataset.")
        return
    if not _has_columns(comparison, ['disagreement_mm', 'angle_deg'], "Section 13"):
        return
    source = comparison['source'] if 'source' in comparison else pd.Series(
        'trial', index=comparison.index)
    print(f"  {'src':<7}{'joint':<10}{'segment':<14}{'fitted':>9}{'landmark':>10}{'|r| diff':>10}"
          f"{'vector':>9}{'angle':>8}{'lm noise':>10}{'pair sd':>9}")
    print(f"  {'':<7}{'':<10}{'':<14}{'|r| mm':>9}{'|c| mm':>10}{'(mm)':>10}{'(mm)':>9}{'(deg)':>8}"
          f"{'(mm)':>10}{'(mm)':>9}")
    # GROUPED BY SOURCE, never averaged across it. A standing centre and a moving one are
    # different measurements of the same point, and their disagreement is a finding in itself.
    for _, row in (comparison.assign(source=source)
                   .groupby(['source', 'joint', 'segment', 'landmarks'], observed=True)
                   .mean(numeric_only=True).reset_index()
                   .sort_values(['source', 'joint', 'segment']).iterrows()):
        print(f"  {row['source']:<7}{row['joint']:<10}{row['segment']:<14}"
              f"{row['fitted_norm_mm']:>9.1f}{row['landmark_median_mm']:>10.1f}"
              f"{row['arm_disagreement_mm']:>10.1f}{row['disagreement_mm']:>9.1f}"
              f"{row['angle_deg']:>8.1f}{row['landmark_sd_mm']:>10.1f}"
              f"{row['pair_distance_sd_mm']:>9.1f}")

    for label, block in comparison.assign(source=source).groupby('source', observed=True):
        print(f"\n{label}: |r| disagreement {block['arm_disagreement_mm'].mean():.1f} mm on arms "
              f"of {block['fitted_norm_mm'].mean():.0f} mm, against a\nlandmark noise floor of "
              f"{block['landmark_sd_mm'].mean():.1f} mm.")
    if (source == 'static').any():
        print("\nTHE TWO SOURCES ARE NOT INTERCHANGEABLE. A 'trial' centre is measured while the "
              "subject moves,\nso its noise column is a real soft-tissue estimate. A 'static' one "
              "comes from the standing\ncapture — the only Al Borno file with MEDIAL markers, and "
              "therefore the only route to a knee or\nankle centre rather than a lateral surface "
              "point — but a standing subject shows no motion-driven\nsoft-tissue movement at all, "
              "so its noise column runs ~10x smaller for the same landmark and is\na far weaker "
              "floor. Read it as reach, not as precision.")
        print("\nAL BORNO'S HIP LANDMARK IS THE WEAK ONE, and should not be read on a par with the "
              "knee and ankle.\nScored as a predictor of a trial's own frames, the standing knee "
              "and ankle sit at 1.5x that\ntrial's own best fit and only 1.25x behind a DIFFERENT "
              "trial's fitted centre; the hip is 2.6x. The\ncause is the reference rather than "
              "soft tissue — the same subject's HJC moves a median 11.8 mm\n(up to 30 mm) between "
              "the standing and walking captures while scattering 3-4 mm within either\none. "
              "Subject11's L_HJC is ~118 mm out in BOTH captures and should be excluded outright.")
    print("\nBOTH ESTIMATES ARE IN ONE BASIS, which is what makes the vector and angle columns "
          "meaningful. The\norientation fit is re-run here on the same raw native-rate "
          "reconstruction the landmark uses,\nrather than read from the built plates: those carry "
          "a sensor-to-segment alignment (171-178 deg on\nAl Borno) that a raw template "
          "reconstruction does not. Comparing across the two reads 316 mm where\nthe arms agree to "
          "8 mm — entirely the frame. |r| from the raw and built fits agrees to 0.00 mm, so\n"
          "refitting costs nothing.")
    print("\nRead |r| diff against vector. Similar lengths with a large vector difference means "
          "the two methods\nput the centre at the same DISTANCE in a different DIRECTION, which "
          "localises the disagreement in a\nway a single scalar cannot — and the angle column "
          "says how far apart.")
    print("\nHOW TO READ THIS, because it is the one section that could be over-claimed. Both "
          "estimates have\nerror and neither is truth:")
    print("  * the FIT's error bar is section 9's standard error plus section 5's between-trial "
          "spread;")
    print("  * the LANDMARK's is `lm noise` PLUS an unknown placement bias. Noise is soft tissue "
          "moving the\n    marker and is visible here; placement is the marker having been taped "
          "somewhere slightly\n    wrong, and is INVISIBLE — it sits constant, so it never enters "
          "a scatter. `lm noise` is a\n    FLOOR on the landmark's error, not an estimate.")
    print("\nSo a disagreement larger than the two scatters combined is a real inconsistency worth "
          "chasing;\none smaller is mutual support, and supports neither method being RIGHT. The "
          "honest summary is\nagreement or its absence, never accuracy.")
    print("\n`pair sd` is the quality gate: for a medial/lateral pair the inter-marker distance is "
          "joint\nwidth and should be near-constant, so a large value means a swapped label or a "
          "dropout, and that\nrow should not be trusted. It is zero where the centre comes from a "
          "single pre-computed marker.")
    print("\nCOVERAGE IS THIN AND ONE-SIDED PER DATASET. Al Borno's knee_* and ankle_* are LATERAL "
          "markers, not\ncentres, so only its hip can be checked; IMoVE has no hip centre "
          "(LGTR is a surface point) but\nhas both epicondyles and both malleoli. Each joint is "
          "therefore checked on one dataset only.")


def print_report(spec: DatasetSpec, fits: pd.DataFrame, stability: pd.DataFrame,
                 curve: pd.DataFrame, landmarks: pd.DataFrame,
                 closure: pd.DataFrame, ambiguity: pd.DataFrame) -> None:
    report_closure(spec, closure, ambiguity)
    report_fit_quality(spec, fits)
    report_ambiguity(spec, ambiguity, fits)
    report_masking(spec, fits)
    report_generalization(spec, fits)
    report_stability(spec, stability)
    report_precision(spec, fits)
    report_learning(spec, curve)
    report_migration(spec, fits)
    report_symmetry(spec, fits)
    report_landmarks(spec, landmarks)
    report_cost(spec, fits, stability)

# ==============================================================================
# CLI
# ==============================================================================

def select_trials(dataset: str, subjects: Optional[List[str]],
                  trials: Optional[List[str]]) -> List[Tuple[str, str]]:
    row_keys = enumerate_trials(dataset)
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
    parser.add_argument('--dataset', default='alborno', choices=sorted(DATASETS))
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--trials', nargs='+', default=None)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--stride', type=int, default=SAMPLE_STRIDE,
                        help="Keep every Nth sample in fit_samples. Every scalar in joint_fits is "
                             "computed on the full valid record regardless.")
    parser.add_argument('--only-tables', nargs='+', choices=TRIAL_TABLES,
                        default=list(TRIAL_TABLES), metavar='TABLE')
    parser.add_argument('--hinge-alignment', action='store_true',
                        help="Also measure whether the worst-determined direction IS the joint "
                             "axis. Costs ~920 ms per joint against the fit's 17 ms — 97%% of the "
                             "runtime — so it is off by default; the answer already recorded is "
                             "0.34-0.78, i.e. not cleanly the hinge.")
    parser.add_argument('--allow-stale', action='store_true',
                        help="Read the cached trial parquets WITHOUT the freshness check. For "
                             "iterating while the build layer is changing; every artifact "
                             "written is stamped built_from_stale_cache and the numbers must be "
                             "re-confirmed against a fresh build before being quoted.")
    parser.add_argument('--report-only', action='store_true',
                        help="Rebuild the summary and report from what is already on disk.")
    args = parser.parse_args()

    try:
        row_keys = select_trials(args.dataset, args.subjects, args.trials)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    spec = get_dataset(args.dataset)

    if not args.report_only:
        if args.allow_stale:
            print("WARNING: --allow-stale. Reading cached parquets without the freshness check; "
                  "every\n         artifact written will be stamped built_from_stale_cache. "
                  "Re-confirm against a\n         fresh build before quoting anything from this "
                  "run.")
        print(f"Fitting joint centres over {len(row_keys)} trials...")
        state, _ = run_tracked_grid(
            row_keys, ['Subject', 'Trial'], ['fit'],
            partial(_trial_worker, dataset=args.dataset, tables=args.only_tables,
                    stride=args.stride, allow_stale=args.allow_stale,
                    hinge_alignment=args.hinge_alignment),
            args.workers, title=f"JOINT CENTRE — {args.dataset}")
        # Say so when the compute pass did not actually compute anything. Without this the run
        # goes straight on to load whatever per-trial tables happen to be on disk and prints a
        # full report off them — which is exactly what happened when a concurrent change to the
        # build layer invalidated every trial cache mid-run: 243 trials "found", every worker
        # failed with StaleTrialCache, and the report described the PREVIOUS run's numbers with
        # nothing to indicate it.
        failures = {key: value for (key, stage), value in state.items()
                    if stage == 'fit' and isinstance(value, str) and value.startswith('Failed')}
        if failures:
            reasons = {}
            for message in failures.values():
                reasons[message.split(':')[0][:70]] = reasons.get(message.split(':')[0][:70], 0) + 1
            print(f"\n{len(failures)} of {len(row_keys)} trials FAILED to compute:")
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
                print(f"  {count:4d} x {reason}")
            if len(failures) == len(row_keys):
                print("\nEvery trial failed, so nothing was written. Anything below would "
                      "describe a PREVIOUS\nrun's tables, not this one — stopping instead.")
                return 1

    print("\nLoading per-trial tables...")
    fits = load_trial_table(args.dataset, 'joint_fits')
    if fits.empty:
        print(f"No results under {dataset_dir(args.dataset)}. Run without --report-only first.")
        return 1
    found = {(str(s), str(t)) for s, t in fits[['subject', 'trial']].drop_duplicates().to_numpy()}
    print(f"Found {len(found)} trial(s) across {len({s for s, _ in found})} subject(s).")

    stability = offset_stability(fits)
    lengths = segment_lengths(fits, spec)
    diagnosis = length_diagnosis(lengths, stability)
    decomposition = shift_decomposition(fits, spec)
    ambiguity = ambiguity_table(args.dataset)
    closure = closure_by_source(args.dataset, spec)
    for frame, name in ((ambiguity, 'ambiguity'), (closure, 'closure_by_source')):
        if not frame.empty:
            _save(frame, paths.ensure_parent(dataset_dir(args.dataset) / f'{name}.parquet'),
                  args.dataset)
    if not lengths.empty:
        path = paths.ensure_parent(dataset_dir(args.dataset) / 'segment_lengths.parquet')
        lengths.to_parquet(path, engine='pyarrow', index=False)
        paths.write_manifest(path, constants=analysis_constants(args.dataset),
                             experiment=EXPERIMENT_NAME, n_rows=len(lengths))
        print(f"Saved segment-length invariants to {path}")
    combined = combined_fits(args.dataset, sorted(found))
    if not combined.empty:
        path = paths.ensure_parent(dataset_dir(args.dataset) / 'subject_fits.parquet')
        combined.to_parquet(path, engine='pyarrow', index=False)
        paths.write_manifest(path, constants=analysis_constants(args.dataset),
                             experiment=EXPERIMENT_NAME, n_rows=len(combined))
        print(f"Saved per-subject combined fits to {path}")
    summary = summarize(args.dataset, fits)
    if not summary.empty:
        path = paths.ensure_parent(statistics_path(args.dataset))
        summary.to_parquet(path, engine='pyarrow', index=False)
        paths.write_manifest(path, constants=analysis_constants(args.dataset),
                             experiment=EXPERIMENT_NAME, n_rows=len(summary),
                             subjects=sorted({s for s, _ in found}))
        print(f"Saved summary to {path}")

    curve = load_trial_table(args.dataset, 'learning_curve')
    landmarks = load_trial_table(args.dataset, 'landmarks')
    print_report(spec, fits, stability, curve, landmarks, closure, ambiguity)
    report_combined(spec, fits, combined)
    report_segment_lengths(spec, lengths, diagnosis, decomposition)
    print(f"\nPer-trial tables under {dataset_dir(args.dataset)}")
    # The report above is fourteen sections to a terminal that nobody keeps, and it covers ONE
    # dataset per invocation. Point at the written summary, or the run's only lasting record of
    # "how good is the joint centre here" is a scrollback.
    print(f"Written summary, every dataset in one file: python -m plotting.joint_center_quality")
    print(f"Geometric sanity check the numbers cannot give you: "
          f"python -m plotting.joint_center --dataset {args.dataset}")
    print(f"Downstream: experiments.acceleration_projection and experiments.magnetic_projection "
          f"the offsets themselves are recomputed on demand via joint_offsets(), not read back.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
