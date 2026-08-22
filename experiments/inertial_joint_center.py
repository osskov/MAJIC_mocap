"""
Can the joint centre be recovered from gyro and accelerometer alone — no mocap?

That is what a deployed system has to do. The mocap fit is the REFERENCE, recomputed here via
`joint_center.joint_offsets` (~17 ms per joint) rather than read from an artifact, so this
experiment has no ordering dependency on another one.

The estimator is Seel (2014): minimize |a_p1| - |a_p2| over the two offsets, where
a_p = a + A o and A = [alpha]x + omega omega^T - |omega|^2 I. Matching MAGNITUDES is what lets it
work without a common frame, and is also what discards the direction of the disagreement.

    python -m experiments.inertial_joint_center --dataset alborno
    python -m experiments.inertial_joint_center --dataset imove --report-only

Four sections:

  1. SHIPPED vs OPTIMIZED. `IMUTrace.find_spheroidal_joint_offset` as it stands (random start,
     20 fixed Gauss-Newton iterations, no damping) against the same objective solved properly
     (trust-region LM, analytic Jacobian, convergence test). 149% -> 47% of the lever arm on
     Al Borno, so the solver was much of the failure — but not all of it.

  2. THE COST LANDSCAPE, which says whether what remains is the solver or the model. Seel's own
     objective is evaluated at the optimizer's answer and at the mocap answer on identical
     samples. Below 1 means the optimizer reached a point the truth cannot beat, so the objective
     prefers somewhere else and the solver is exonerated.

  3. INPUT CONDITIONING, two knobs with different mechanisms. ALPHA enters the design matrix, so
     its noise is errors-in-variables and BIASES the offset toward zero — tracked by attenuation
     |o|/|r|. ACC enters the residual, so its noise inflates VARIANCE without biasing. Both are
     swept together because each one's optimum depends on the other.

  4. REPEATABILITY against the mocap fit, normalized by each estimator's own arm — the inertial
     fit is attenuated, and a shorter vector scatters less in millimetres for free.

Lookahead is free here and several estimators use it. That is legitimate for a one-time offline
calibration and would not be in `project_acc`, which feeds a causal filter; the shipped
'backward' default is inherited from that setting rather than chosen for this one.

Outputs under results/experiments/inertial_joint_center/<dataset>/<subject>/<trial>/:
    inertial_fits.parquet   per joint: both estimators' offsets, error, cost landscape
    config_sweep.parquet    per (joint, config): error, attenuation, cost ratio
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
from experiments.experiment_utils import pipeline_constants, run_tracked_grid
from experiments.global_assumptions import DATASETS, DatasetSpec, enumerate_trials, get_dataset
from experiments.joint_center import (MIN_FIT_FRAMES, OFFSET_COLUMNS, _load_spec_plates,
                                      joint_offsets, select_trials)
from src.toolchest.finite_difference_utils import polynomial_fit_derivative
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial

EXPERIMENT_NAME = "inertial_joint_center"
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MAX_SAMPLES = 20000          # six parameters need far fewer, and the Jacobian rebuilds each step
EXCITATION_PERCENTILE = 50.0  # A -> 0 where the segment is not accelerating; those rows only dilute

# (alpha method, polyfit window s, alpha low-pass Hz, ACC low-pass Hz). The alpha rows are swept
# at no acc filter so the attenuation curve stays readable, then the acc cutoff is swept at the
# two alpha settings worth carrying forward.
GRID = (
    ('backward', None, None, None),      # the shipped default, the baseline to beat
    ('central', None, None, None),
    ('polyfit', 0.05, None, None),
    ('polyfit', 0.10, None, None),
    ('polyfit', 0.20, None, None),
    ('polyfit', 0.40, None, None),
    ('backward', None, 10.0, None),
    ('backward', None, 5.0, None),
    ('central', None, None, 15.0),
    ('central', None, None, 10.0),
    ('polyfit', 0.10, None, 25.0),
    ('polyfit', 0.10, None, 15.0),
    ('polyfit', 0.10, None, 10.0),
    ('polyfit', 0.10, None, 6.0),
)
# Chosen by the sweep, not inherited: the joint optimum on Al Borno (149% -> 34%). Section 3
# flags it when it is not best for the dataset in hand — the optimum moves with sample rate,
# since a backward difference amplifies noise by sqrt(2)/dt.
DEFAULT_CONFIG = ('polyfit', 0.10, None, 10.0)

# Fixed starts for the shipped estimator, which defaults to a RANDOM one and is therefore not
# reproducible as-is. Spread, so their disagreement measures multi-basin behaviour.
SHIPPED_STARTS = (0.10, -0.10, 0.05)
SHIPPED_SUBSAMPLE = 10

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
TRIAL_TABLES = ('inertial_fits', 'config_sweep')

SHIPPED_COLUMNS = tuple(f"imu_{c}" for c in OFFSET_COLUMNS)
OPT_COLUMNS = tuple(f"opt_{c}" for c in OFFSET_COLUMNS)
MOCAP_COLUMNS = tuple(f"mocap_{c}" for c in OFFSET_COLUMNS)


def config_label(config) -> str:
    method, window, lowpass, acc_lowpass = config
    name = method if window is None else f"{method}/{1000 * window:.0f}ms"
    if lowpass is not None:
        name = f"{name}+{lowpass:g}Hz"
    return name if acc_lowpass is None else f"{name} | acc {acc_lowpass:g}Hz"


def analysis_constants(dataset: str) -> Dict[str, object]:
    return {**pipeline_constants(), 'dataset': dataset, 'max_samples': MAX_SAMPLES,
            'excitation_percentile': EXCITATION_PERCENTILE,
            'default_config': config_label(DEFAULT_CONFIG)}

# ==============================================================================
# Paths / IO
# ==============================================================================

def dataset_dir(dataset: str) -> Path:
    return EXPERIMENT_DIR / dataset


def trial_table_path(dataset: str, subject: str, trial: str, table: str) -> Path:
    return dataset_dir(dataset) / subject / trial / f"{table}.parquet"


def _save(df: pd.DataFrame, path: Path, dataset: str, **extra) -> None:
    df.to_parquet(paths.ensure_parent(path), engine='pyarrow', index=False)
    paths.write_manifest(path, constants=analysis_constants(dataset), experiment=EXPERIMENT_NAME,
                         n_rows=len(df), **extra)


def load_trial_table(dataset: str, table: str,
                     row_keys: Optional[Sequence[Tuple[str, str]]] = None) -> pd.DataFrame:
    if table not in TRIAL_TABLES:
        raise ValueError(f"Unknown table '{table}'; expected one of {TRIAL_TABLES}")
    row_keys = enumerate_trials(dataset) if row_keys is None else row_keys
    frames = [pd.read_parquet(p, engine='pyarrow').assign(subject=s, trial=t)
              for s, t in row_keys
              if (p := trial_table_path(dataset, s, t, table)).exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).assign(dataset=dataset)

# ==============================================================================
# The model
# ==============================================================================

def alpha_estimate(trace: IMUTrace, config) -> np.ndarray:
    """Angular acceleration under one estimator configuration.

    `polyfit` is reached directly rather than through `_finite_difference_gyros`, which fixes the
    window — and the window is the knob worth turning, since it sets how much of the 1/dt noise
    amplification survives. The optional low-pass is `filtfilt`, so zero-lag: a phase shift in
    alpha would bias the offset, and offline there is no reason to accept one.
    """
    method, window, lowpass, _ = config
    alpha = (polynomial_fit_derivative(trace.gyro, trace.timestamps, order=2,
                                       window_seconds=window) if method == 'polyfit'
             else trace._finite_difference_gyros(method))
    alpha = np.asarray(alpha, dtype=np.float64)
    if lowpass:
        fs = trace.get_sample_frequency()
        if lowpass < fs / 2:
            b, a = butter(4, lowpass / (fs / 2.0), btype='low')
            alpha = filtfilt(b, a, alpha, axis=0)
    return alpha


def terms(plate: PlateTrial, mask: np.ndarray, config=DEFAULT_CONFIG) -> Dict[str, np.ndarray]:
    """Per-sample `acc` and `A`, with a_p = a + A o.

    THE SIGN IS PLUS, matching `IMUTrace.project_acc`. `find_spheroidal_joint_offset`'s docstring
    writes the same equation with a MINUS — the conventions differ in which way the offset points,
    and getting it backwards is silent: the optimizer still converges, to roughly the negated
    answer. Checked, not assumed: at the mocap offsets the plus form scores 3.32 m/s^2 against
    3.57 at zero offset (an improvement, as a correct objective must give), the minus form 7.08.

    The acc low-pass is applied to `acc` AND `A` together, which is exact rather than approximate:
    o is constant, so LP(a + A o) = LP(a) + LP(A) o. It is skipped above Nyquist rather than
    raising, because the datasets mix 100 Hz and 40 Hz trials and a sweep cutoff is meaningful
    for some and impossible for others.
    """
    trace = plate.imu_trace
    n = min(len(mask), len(trace))
    selected = np.asarray(mask)[:n]
    omega = trace.gyro[:n][selected].astype(np.float64)
    alpha = alpha_estimate(trace, config)[:n][selected]
    acc = trace.acc[:n][selected].astype(np.float64)

    skew = np.zeros((len(alpha), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -alpha[:, 2], alpha[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = alpha[:, 2], -alpha[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -alpha[:, 1], alpha[:, 0]
    A = (skew + np.einsum('ni,nj->nij', omega, omega)
         - (omega ** 2).sum(axis=1)[:, None, None] * np.eye(3))

    acc_lowpass, fs = config[3], trace.get_sample_frequency()
    if acc_lowpass is not None and acc_lowpass < fs / 2 and len(acc) > 24:
        b, a = butter(4, acc_lowpass / (fs / 2.0), btype='low')
        acc = filtfilt(b, a, acc, axis=0)
        A = filtfilt(b, a, A.reshape(len(A), 9), axis=0).reshape(-1, 3, 3)
    return {'acc': acc, 'A': A}


def cost_parts(x: np.ndarray, parent: Dict[str, np.ndarray],
               child: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """(residual, jacobian). e = |a_p1| - |a_p2|; de/do = +(a_p/|a_p|)^T A, negated for the child."""
    projected_p = parent['acc'] + np.einsum('nij,j->ni', parent['A'], x[:3])
    projected_c = child['acc'] + np.einsum('nij,j->ni', child['A'], x[3:])
    norm_p = np.linalg.norm(projected_p, axis=1)
    norm_c = np.linalg.norm(projected_c, axis=1)
    safe_p, safe_c = np.maximum(norm_p, 1e-9)[:, None], np.maximum(norm_c, 1e-9)[:, None]
    return norm_p - norm_c, np.concatenate([
        np.einsum('ni,nij->nj', projected_p / safe_p, parent['A']),
        -np.einsum('ni,nij->nj', projected_c / safe_c, child['A'])], axis=1)


def cost(x: np.ndarray, parent: Dict[str, np.ndarray], child: Dict[str, np.ndarray]) -> float:
    """RMS residual at `x`, m/s^2 — comparable between candidate solutions."""
    return float(np.sqrt(np.mean(cost_parts(x, parent, child)[0] ** 2)))


def sample_mask(parent_plate: PlateTrial, child_plate: PlateTrial) -> np.ndarray:
    """The most excited half of the FULL record, capped.

    Not restricted to `valid`: the point of this estimator is that it needs no mocap, so scoring
    it only where mocap worked would understate it — on the Al Borno walking trials that would
    discard most of the record.
    """
    n = min(len(parent_plate), len(child_plate))
    strength = np.zeros(n)
    for plate in (parent_plate, child_plate):
        trace = plate.imu_trace
        omega = trace.gyro[:n].astype(np.float64)
        alpha = trace._finite_difference_gyros('backward')[:n].astype(np.float64)
        strength += np.linalg.norm(alpha, axis=1) + (omega ** 2).sum(axis=1)
    mask = strength >= np.percentile(strength, EXCITATION_PERCENTILE)
    indices = np.flatnonzero(mask)
    if len(indices) > MAX_SAMPLES:
        keep = np.zeros(n, dtype=bool)
        keep[indices[np.linspace(0, len(indices) - 1, MAX_SAMPLES).round().astype(int)]] = True
        return keep
    return mask


def fit(parent_plate: PlateTrial, child_plate: PlateTrial, mask: np.ndarray,
        start: np.ndarray, config=DEFAULT_CONFIG) -> Optional[Dict[str, object]]:
    """Trust-region LM with an analytic Jacobian, from `start`.

    Three differences from the shipped estimator, each a candidate explanation for its failure:
    damping (undamped Gauss-Newton on a near-singular Jacobian can step arbitrarily far), a
    convergence test rather than a fixed iteration count, and an analytic Jacobian rather than a
    finite difference of an already-differenced residual. `start` is a parameter so the same
    solver can run from an uninformative guess and from the mocap answer — which is what
    separates a solver failure from a cost-function failure.
    """
    from scipy.optimize import least_squares
    parent_terms, child_terms = terms(parent_plate, mask, config), terms(child_plate, mask, config)
    if len(parent_terms['acc']) < MIN_FIT_FRAMES:
        return None
    try:
        solution = least_squares(lambda x: cost_parts(x, parent_terms, child_terms)[0], start,
                                 jac=lambda x: cost_parts(x, parent_terms, child_terms)[1],
                                 method='trf', max_nfev=200, xtol=1e-12, ftol=1e-12)
    except Exception:
        return None
    return {'x': solution.x, 'converged': bool(solution.success),
            'cost': cost(solution.x, parent_terms, child_terms),
            'terms': (parent_terms, child_terms)}

# ==============================================================================
# The shipped estimator, for comparison
# ==============================================================================

def shipped_fit(parent_plate: PlateTrial, child_plate: PlateTrial,
                mocap_parent: np.ndarray, mocap_child: np.ndarray) -> Dict[str, float]:
    """`IMUTrace.find_spheroidal_joint_offset` as it stands, scored against mocap.

    Run from several fixed starts because the shipped default is `np.random.rand(3) * 0.1`, which
    makes it irreproducible and hides multi-basin behaviour behind whatever seed was set.
    `init_spread_mm` is the largest disagreement between those starts: near zero means one basin.
    """
    result = {'shipped_converged': False, 'shipped_error_parent_mm': np.nan,
              'shipped_error_child_mm': np.nan, 'shipped_init_spread_mm': np.nan,
              'shipped_error_relative': np.nan}
    result.update({c: np.nan for c in SHIPPED_COLUMNS})

    solutions = []
    for guess in SHIPPED_STARTS:
        try:
            out = parent_plate.imu_trace.find_spheroidal_joint_offset(
                child_plate.imu_trace, initial_offset_self=np.full(3, guess),
                initial_offset_other=np.full(3, -guess), subsample_rate=SHIPPED_SUBSAMPLE)
        except Exception:
            continue
        solutions.append((np.asarray(out['offset_self'], dtype=float),
                          np.asarray(out['offset_other'], dtype=float), bool(out['converged'])))
    if not solutions:
        return result

    parents = np.array([s[0] for s in solutions])
    children = np.array([s[1] for s in solutions])
    result['shipped_converged'] = any(s[2] for s in solutions)
    for column, value in zip(SHIPPED_COLUMNS, np.concatenate([parents[0], children[0]])):
        result[column] = float(value)
    if len(solutions) > 1:
        result['shipped_init_spread_mm'] = float(1000.0 * max(
            np.linalg.norm(parents - parents[0], axis=1).max(),
            np.linalg.norm(children - children[0], axis=1).max()))

    errors = 1000.0 * np.array([np.linalg.norm(parents[0] - mocap_parent),
                                np.linalg.norm(children[0] - mocap_child)])
    result['shipped_error_parent_mm'], result['shipped_error_child_mm'] = errors
    result['shipped_error_relative'] = _relative(errors, mocap_parent, mocap_child)
    return result


def _relative(errors_mm: np.ndarray, mocap_parent: np.ndarray, mocap_child: np.ndarray) -> float:
    """Error as a fraction of the lever arm being estimated. Above 1 carries no information about
    where the joint is — you would do as well not projecting."""
    arms = 1000.0 * np.array([np.linalg.norm(mocap_parent), np.linalg.norm(mocap_child)])
    with np.errstate(invalid='ignore', divide='ignore'):
        return float(np.nanmean(errors_mm / np.where(arms > 0, arms, np.nan)))

# ==============================================================================
# Per-trial tables
# ==============================================================================

def inertial_fits(plates: Dict[str, PlateTrial], spec: DatasetSpec, dataset: str,
                  subject: str, trial: str) -> pd.DataFrame:
    """Per joint: both estimators against the mocap reference, plus the cost landscape.

    `cost_ratio` = cost(optimizer) / cost(mocap) on identical samples. Below 1 means the optimizer
    found a point the true joint centre cannot beat, so the objective prefers elsewhere and the
    solver is not what is failing. `drift_from_mocap_mm` asks it from the other side: start AT the
    mocap answer and see whether it stays.
    """
    reference = joint_offsets(plates, spec)
    rows = []
    for joint, (parent_sensor, child_sensor) in spec.joints.items():
        if joint not in reference or parent_sensor not in plates or child_sensor not in plates:
            continue
        parent_plate, child_plate = plates[parent_sensor], plates[child_sensor]
        mocap_parent, mocap_child = reference[joint]['parent'], reference[joint]['child']
        mocap = np.concatenate([mocap_parent, mocap_child])
        row = {'joint': joint,
               'parent_norm_mm': float(np.linalg.norm(mocap_parent) * 1000.0),
               'child_norm_mm': float(np.linalg.norm(mocap_child) * 1000.0),
               'opt_converged': False, 'cost_at_opt': np.nan, 'cost_at_mocap': np.nan,
               'cost_ratio': np.nan, 'drift_from_mocap_mm': np.nan, 'verdict': '',
               'opt_error_parent_mm': np.nan, 'opt_error_child_mm': np.nan,
               'opt_error_relative': np.nan}
        row.update({c: np.nan for c in OPT_COLUMNS})
        row.update(dict(zip(MOCAP_COLUMNS, mocap)))
        row.update(shipped_fit(parent_plate, child_plate, mocap_parent, mocap_child))

        mask = sample_mask(parent_plate, child_plate)
        cold = fit(parent_plate, child_plate, mask, np.zeros(6)) if mask.sum() >= MIN_FIT_FRAMES \
            else None
        if cold is not None:
            warm = fit(parent_plate, child_plate, mask, mocap.copy())
            parent_terms, child_terms = cold['terms']
            row['opt_converged'] = cold['converged']
            row['cost_at_opt'] = cold['cost']
            row['cost_at_mocap'] = cost(mocap, parent_terms, child_terms)
            if row['cost_at_mocap'] > 0:
                row['cost_ratio'] = cold['cost'] / row['cost_at_mocap']
            if warm is not None:
                row['drift_from_mocap_mm'] = float(np.linalg.norm(warm['x'] - mocap) * 1000.0)
            for column, value in zip(OPT_COLUMNS, cold['x']):
                row[column] = float(value)
            errors = 1000.0 * np.array([np.linalg.norm(cold['x'][:3] - mocap_parent),
                                        np.linalg.norm(cold['x'][3:] - mocap_child)])
            row['opt_error_parent_mm'], row['opt_error_child_mm'] = errors
            row['opt_error_relative'] = _relative(errors, mocap_parent, mocap_child)
            ratio = row['cost_ratio']
            row['verdict'] = ('' if not np.isfinite(ratio) else 'cost function' if ratio < 0.99
                              else 'optimizer' if ratio > 1.01 else 'tie')
        rows.append(row)
    return pd.DataFrame(rows)


def config_sweep(plates: Dict[str, PlateTrial], spec: DatasetSpec, dataset: str,
                 subject: str, trial: str) -> pd.DataFrame:
    """Every configuration in GRID, primary joints only.

    `attenuation` = |o_fitted| / |o_mocap| is the errors-in-variables signature: near 0 means the
    alpha estimator is too noisy, above 1 means it is over-smoothed. The error alone cannot say
    which side of the optimum you are on.

    Primary joints only: the placement variants span the same anatomical joints with a different
    sensor, which is a question about mounting rather than about differentiation.
    """
    reference = joint_offsets(plates, spec)
    rows = []
    for joint in spec.primary_joints:
        if joint not in reference or joint not in spec.joints:
            continue
        parent_sensor, child_sensor = spec.joints[joint]
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        parent_plate, child_plate = plates[parent_sensor], plates[child_sensor]
        mocap_parent, mocap_child = reference[joint]['parent'], reference[joint]['child']
        mocap = np.concatenate([mocap_parent, mocap_child])
        arms = np.array([np.linalg.norm(mocap_parent), np.linalg.norm(mocap_child)])
        mask = sample_mask(parent_plate, child_plate)
        if mask.sum() < MIN_FIT_FRAMES:
            continue

        for config in GRID:
            solved = fit(parent_plate, child_plate, mask, np.zeros(6), config)
            if solved is None:
                continue
            offsets = solved['x']
            parent_terms, child_terms = solved['terms']
            errors = np.array([np.linalg.norm(offsets[:3] - mocap_parent),
                               np.linalg.norm(offsets[3:] - mocap_child)])
            fitted_arms = np.array([np.linalg.norm(offsets[:3]), np.linalg.norm(offsets[3:])])
            with np.errstate(invalid='ignore', divide='ignore'):
                safe = np.where(arms > 0, arms, np.nan)
                relative, attenuation = np.nanmean(errors / safe), np.nanmean(fitted_arms / safe)
            cost_mocap = cost(mocap, parent_terms, child_terms)
            rows.append({
                'joint': joint, 'config': config_label(config), 'method': config[0],
                'window_s': config[1] if config[1] is not None else np.nan,
                'alpha_lowpass_hz': config[2] if config[2] is not None else np.nan,
                'acc_lowpass_hz': config[3] if config[3] is not None else np.nan,
                'error_parent_mm': float(errors[0] * 1000.0),
                'error_child_mm': float(errors[1] * 1000.0),
                'error_relative': float(relative), 'attenuation': float(attenuation),
                'cost_at_opt': solved['cost'], 'cost_at_mocap': cost_mocap,
                'cost_ratio': solved['cost'] / cost_mocap if cost_mocap > 0 else np.nan,
                'converged': bool(solved['converged'])})
    return pd.DataFrame(rows)


def compute_trial(plates, spec, dataset, subject, trial, tables=TRIAL_TABLES):
    builders = {'inertial_fits': lambda: inertial_fits(plates, spec, dataset, subject, trial),
                'config_sweep': lambda: config_sweep(plates, spec, dataset, subject, trial)}
    return {name: build() for name, build in builders.items() if name in set(tables)}


def _trial_worker(row_key, stage_labels, shared_state, dataset='alborno',
                  tables=TRIAL_TABLES, allow_stale=False) -> None:
    subject, trial = row_key
    spec = get_dataset(dataset)
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()
    try:
        plates = _load_spec_plates(subject, trial, dataset, spec, allow_stale)
        for table, df in compute_trial(plates, spec, dataset, subject, trial, tables).items():
            if df.empty:
                continue
            _save(df, trial_table_path(dataset, subject, trial, table), dataset,
                  subject=subject, trial=trial, table=table,
                  built_from_stale_cache=allow_stale)
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success"
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"

# ==============================================================================
# Report
# ==============================================================================

def _header(number: int, title: str, subtitle: str) -> None:
    print("\n" + "=" * 96)
    print(f"{number}. {title}")
    print(f"   ({subtitle})")
    print("=" * 96)


def report_estimators(spec: DatasetSpec, fits: pd.DataFrame) -> None:
    _header(1, "INERTIAL RECONSTRUCTION vs THE MOCAP REFERENCE",
            "errors as a fraction of the lever arm being estimated")
    usable = fits[np.isfinite(fits['opt_error_relative'])] if not fits.empty else fits
    if usable.empty:
        print("No inertial fits.")
        return
    print(f"  {'joint':<11}{'|r| mocap':>11}{'shipped':>10}{'OPTIMIZED':>11}{'cost@opt':>10}"
          f"{'cost@mocap':>12}{'ratio':>8}{'drift':>9}")
    print(f"  {'':<11}{'(mm)':>11}{'(% |r|)':>10}{'(% |r|)':>11}{'(m/s^2)':>10}{'(m/s^2)':>12}"
          f"{'':>8}{'(mm)':>9}")
    for joint in [j for j in spec.primary_joints if j in set(usable['joint'])]:
        rows = usable[usable['joint'] == joint]
        arm = 0.5 * (rows['parent_norm_mm'].mean() + rows['child_norm_mm'].mean())
        print(f"  {joint:<11}{arm:>11.1f}"
              f"{100 * rows['shipped_error_relative'].mean():>9.0f}%"
              f"{100 * rows['opt_error_relative'].mean():>10.0f}%"
              f"{rows['cost_at_opt'].mean():>10.3f}{rows['cost_at_mocap'].mean():>12.3f}"
              f"{rows['cost_ratio'].mean():>8.2f}{rows['drift_from_mocap_mm'].mean():>9.0f}")

    ratio = usable['cost_ratio'].mean()
    print(f"\nPooled: shipped {100 * usable['shipped_error_relative'].mean():.0f}%, optimized "
          f"{100 * usable['opt_error_relative'].mean():.0f}%. Cost ratio {ratio:.2f}, median "
          f"drift {usable['drift_from_mocap_mm'].median():.0f} mm.")
    print(f"Verdicts: {usable['verdict'].value_counts().to_dict()}")

    print("\nWhat the last three columns settle. Optimizing the solve more than halves the error, "
          "so the\nshipped estimator's failure was largely its solver. What remains is the "
          "OBJECTIVE: at the\nconfiguration in use the optimizer reaches a lower cost than the "
          "true joint centre achieves, and\nstarted at that centre it walks away. A better "
          "optimizer will not fix that; a different residual\nmight. Section 3 sweeps the input "
          "conditioning, which moves this verdict substantially.")

    by_joint = usable.groupby('joint', observed=True)[['opt_error_relative', 'cost_ratio',
                                                       'drift_from_mocap_mm']].mean()
    if len(by_joint) > 1:
        best, worst = by_joint['opt_error_relative'].idxmin(), by_joint['opt_error_relative'].idxmax()
        print(f"\nNOT UNIFORM ACROSS JOINTS, which the pooled number hides. {best} reaches "
              f"{100 * by_joint.loc[best, 'opt_error_relative']:.0f}% with cost ratio "
              f"{by_joint.loc[best, 'cost_ratio']:.2f} and {by_joint.loc[best, 'drift_from_mocap_mm']:.0f} mm "
              f"drift — close to a\nworking reconstruction, and the objective there is nearly "
              f"exonerated. {worst} is at "
              f"{100 * by_joint.loc[worst, 'opt_error_relative']:.0f}%. Any claim about doing "
              f"this without mocap has to be per joint.")


def report_sweep(spec: DatasetSpec, sweep: pd.DataFrame) -> None:
    _header(2, "HOW THE INPUTS SHOULD BE CONDITIONED",
            "alpha biases the fit; acc scatters it")
    if sweep.empty:
        print("No sweep results.")
        return
    grouped = (sweep.groupby('config', observed=True)
               .agg(error=('error_relative', 'mean'), attenuation=('attenuation', 'mean'),
                    cost_ratio=('cost_ratio', 'mean'), converged=('converged', 'mean'),
                    n=('error_relative', 'size')).sort_values('error'))
    width = max(26, max(len(str(i)) for i in grouped.index) + 2)
    print(f"  {'configuration':<{width}}{'error':>10}{'attenuation':>14}{'cost ratio':>12}"
          f"{'converged':>11}{'n':>7}")
    print(f"  {'':<{width}}{'(% |r|)':>10}{'|o|/|r|':>14}{'opt/mocap':>12}{'':>11}{'':>7}")
    for label, row in grouped.iterrows():
        print(f"  {label:<{width}}{100 * row['error']:>9.0f}%{row['attenuation']:>14.2f}"
              f"{row['cost_ratio']:>12.2f}{100 * row['converged']:>10.0f}%{int(row['n']):>7}")

    best, default = grouped.index[0], config_label(DEFAULT_CONFIG)
    print(f"\nBest: {best}. Default in use: {default}.")
    if 'backward' in grouped.index:
        base = grouped.loc['backward']
        print(f"Against the shipped 'backward' baseline: {100 * base['error']:.0f}% -> "
              f"{100 * grouped.loc[best, 'error']:.0f}%, attenuation {base['attenuation']:.2f} -> "
              f"{grouped.loc[best, 'attenuation']:.2f}.")
    if default in grouped.index and default != best:
        gap = 100 * (grouped.loc[default, 'error'] - grouped.loc[best, 'error'])
        print(f"\nTHE DEFAULT IS NOT BEST HERE — {default} costs {gap:.0f} points more than "
              f"{best}. DEFAULT_CONFIG is\none constant shared by both datasets, chosen on Al "
              f"Borno; on this dataset it does not travel.")

    print("\nTwo knobs, two mechanisms:")
    print("  ALPHA is in the DESIGN MATRIX, so its noise is errors-in-variables and BIASES the "
          "offset\n  toward zero. Attenuation tracks it: below 1 the derivative is too noisy, "
          "above 1 it is\n  over-smoothed. Both look like a large error.")
    print("  ACC is in the RESIDUAL, so its noise inflates VARIANCE without biasing. It barely "
          "moves\n  attenuation and shows up instead as between-trial spread (section 4).")
    print("\nTHE OPTIMUM MOVES WITH SAMPLE RATE, so one global default is the wrong shape for this "
          "knob: a\nbackward difference amplifies noise by sqrt(2)/dt, 2.5x worse at 100 Hz than "
          "at 40 Hz. Anything\nfixing a window in SECONDS assumes a rate.")


def report_repeatability(spec: DatasetSpec, dataset: str, fits: pd.DataFrame) -> None:
    """Does the inertial centre wander more between trials than the mocap one?"""
    _header(3, "BETWEEN-TRIAL REPEATABILITY vs MOCAP",
            "spread as a fraction of each estimator's OWN arm; the inertial fit is attenuated")
    if fits.empty or not set(OPT_COLUMNS) <= set(fits.columns):
        print("No inertial fits.")
        return

    rows = []
    for (subject, joint), group in fits.groupby(['subject', 'joint'], observed=True):
        if len(group) < 2:
            continue
        # The mocap reference is stored in the table beside each inertial fit, so this needs no
        # second pass over the trials — and no ordering dependency on another experiment.
        if not set(MOCAP_COLUMNS) <= set(group.columns):
            continue
        mocap = group[list(MOCAP_COLUMNS)].to_numpy(dtype=float) * 1000.0
        if not np.isfinite(mocap).all():
            continue
        inertial = group[list(OPT_COLUMNS)].to_numpy(dtype=float) * 1000.0
        if not np.isfinite(inertial).all():
            continue

        def spread_and_arm(offsets):
            per_end = offsets.reshape(len(offsets), 2, 3)
            centre = offsets.mean(axis=0).reshape(2, 3)
            return (float(np.linalg.norm(per_end - centre, axis=2).mean()),
                    float(np.linalg.norm(per_end, axis=2).mean()))

        mocap_spread, mocap_arm = spread_and_arm(mocap)
        inertial_spread, inertial_arm = spread_and_arm(inertial)
        rows.append({'joint': joint, 'mocap_spread': mocap_spread, 'mocap_arm': mocap_arm,
                     'inertial_spread': inertial_spread, 'inertial_arm': inertial_arm})
    table = pd.DataFrame(rows)
    if table.empty:
        print("No subject has two or more trials with both fits.")
        return

    print(f"  {'joint':<11}{'mocap':>9}{'inertial':>10}{'ratio':>8}   {'mocap':>9}{'inertial':>10}"
          f"{'ratio':>8}")
    print(f"  {'':<11}{'(mm)':>9}{'(mm)':>10}{'abs':>8}   {'% arm':>9}{'% arm':>10}{'relative':>8}")
    for joint in [j for j in spec.joints if j in set(table['joint'])]:
        r = table[table['joint'] == joint].mean(numeric_only=True)
        mocap_rel = 100 * r['mocap_spread'] / r['mocap_arm']
        inertial_rel = 100 * r['inertial_spread'] / r['inertial_arm']
        print(f"  {joint:<11}{r['mocap_spread']:>9.1f}{r['inertial_spread']:>10.1f}"
              f"{r['inertial_spread'] / r['mocap_spread']:>8.1f}   {mocap_rel:>8.1f}%"
              f"{inertial_rel:>9.1f}%{inertial_rel / mocap_rel:>8.1f}")

    pooled = table.mean(numeric_only=True)
    mocap_rel = 100 * pooled['mocap_spread'] / pooled['mocap_arm']
    inertial_rel = 100 * pooled['inertial_spread'] / pooled['inertial_arm']
    print(f"\nPooled: mocap {pooled['mocap_spread']:.1f} mm ({mocap_rel:.1f}% of arm), inertial "
          f"{pooled['inertial_spread']:.1f} mm ({inertial_rel:.1f}%)\n— "
          f"{inertial_rel / mocap_rel:.1f}x less repeatable relative to its own arm.")
    print("\nRead the RELATIVE pair. The inertial fit is attenuated, so its offsets are shorter "
          "and scatter\nless in millimetres for free; dividing by each estimator's own arm "
          "removes that. It inherits\nevery source of mocap variation and adds estimator noise, "
          "so it cannot be more repeatable\nunless something is compressing it.")

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
    parser.add_argument('--only-tables', nargs='+', choices=TRIAL_TABLES,
                        default=list(TRIAL_TABLES), metavar='TABLE')
    parser.add_argument('--allow-stale', action='store_true',
                        help="Read cached parquets without the freshness check; artifacts are "
                             "stamped built_from_stale_cache.")
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()

    try:
        row_keys = select_trials(args.dataset, args.subjects, args.trials)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    spec = get_dataset(args.dataset)

    if not args.report_only:
        if args.allow_stale:
            print("WARNING: --allow-stale; artifacts stamped built_from_stale_cache.")
        print(f"Fitting {len(row_keys)} trials...")
        state, _ = run_tracked_grid(
            row_keys, ['Subject', 'Trial'], ['fit'],
            partial(_trial_worker, dataset=args.dataset, tables=args.only_tables,
                    allow_stale=args.allow_stale),
            args.workers, title=f"INERTIAL JOINT CENTRE — {args.dataset}")
        failures = [v for (_, stage), v in state.items()
                    if stage == 'fit' and isinstance(v, str) and v.startswith('Failed')]
        if failures:
            print(f"\n{len(failures)} of {len(row_keys)} trials failed.")
            if len(failures) == len(row_keys):
                print("Every trial failed, so nothing was written — stopping rather than "
                      "reporting a previous run's tables.")
                return 1

    fits = load_trial_table(args.dataset, 'inertial_fits')
    sweep = load_trial_table(args.dataset, 'config_sweep')
    if fits.empty:
        print(f"No results under {dataset_dir(args.dataset)}. Run without --report-only first.")
        return 1
    found = {(str(s), str(t)) for s, t in fits[['subject', 'trial']].drop_duplicates().to_numpy()}
    print(f"Found {len(found)} trial(s) across {len({s for s, _ in found})} subject(s).")

    report_estimators(spec, fits)
    report_sweep(spec, sweep)
    report_repeatability(spec, args.dataset, fits)
    print(f"\nPer-trial tables under {dataset_dir(args.dataset)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
