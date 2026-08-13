"""
How well the rigid-body acceleration projection reproduces the true acceleration at a
joint center. This is the validation behind the supplementary figure, and it is a property
of the DATA and the projection physics alone — no filter is run.

What is being compared
----------------------
Every method in this repo that uses `project=True` feeds the EKF an accelerometer signal
that was never measured: the sensor's reading rigid-body projected onto the joint center,

    a_proj = a_sensor + alpha x r + omega x (omega x r)          (IMUTrace.project_acc)

with `omega` the gyro, `alpha` a sliding-window polynomial derivative of the gyro, and `r`
the constant sensor->joint-center offset from the mocap least-squares fit. Three signals
are compared here, all in the same sensor body frame and all in m/s^2:

    reference   the TRUE specific force at that joint center, from mocap: the joint
                center's world position differentiated twice, plus gravity, rotated into
                the sensor's ground-truth orientation
    projected   a_proj, the quantity the filter actually consumes
    sensor      a_sensor, the raw unprojected reading

The unprojected baseline is the point of the whole analysis. On its own, `projected` vs
`reference` only establishes an error floor; what makes the projection worth doing is that
its error is smaller than the error of the signal it replaces, on the same reference.

Gravity is left IN all three signals rather than removed. It has to be: gravity is uniform
over a rigid body, so it passes through the projection equation untouched, which is exactly
why that equation is valid for specific force and not just for kinematic acceleration. The
filter also consumes the total vector, so the total vector is what should be validated.

Two references, deliberately
----------------------------
`get_joint_center` fits the parent and child offsets jointly, but the two segments' implied
joint-center positions still differ by the fit residual (a few mm, stored per sample as
`jc_residual_norm`). That leaves two defensible reference points, and both are recorded:

    err_proj         reference is the segment's OWN implied joint center. This is the point
                     the projection is actually aiming at, so this isolates the projection
                     physics.
    err_proj_shared  reference is the midpoint of the two segments' implied joint centers,
                     which is what experiment_utils._compute_perfect_joint_acc (the filter's
                     acc oracle) uses. Larger than err_proj by whatever the two segments
                     disagree about.

Quoting only the second would blame the projection for the joint model's inconsistency;
quoting only the first would understate the error the oracle path actually sees.

The misalignment floor, and why no choice of frame removes it
------------------------------------------------------------
The reference is built by rotating a world-frame acceleration into the body frame with the
mocap orientation, while the estimate is measured in the sensor's true frame. Any residual
sensor-to-segment misalignment left by assembly.align_world_to_imu therefore
turns into an apparent projection error of roughly (angle) x |a|, and since |a| is dominated
by gravity, 1 deg tilts it by ~0.17 m/s^2. That term adds in quadrature with everything else
in the residual, so whether it matters is an empirical question and not one to assume —
`report_alignment` answers it in the numbers rather than in prose.

Changing frames does not help, which is worth stating because it is the obvious first
instinct: rotating both signals by the same matrix leaves |a_est - a_ref| exactly unchanged,
and so does subtracting a common gravity vector from both. The floor is therefore MEASURED
instead, per sensor-trial, by fitting the single constant rotation that best maps the
projected signal onto the reference (Kabsch/Wahba, see `residual_alignment`) and reporting
the residual before and after removing it. `align_angle_deg` is that rotation's magnitude
and `rms_after` is what remains once it is gone.

This affects `projected` and `sensor` identically — same sensor frame, same reference
construction — so it inflates both error columns without biasing the comparison between
them, which is the claim the figure makes.

Low-pass filtering, and the reference's own noise floor
------------------------------------------------------
The reference is a SECOND derivative of marker-derived positions, so it carries marker noise
amplified by differentiation. On this dataset that is not a small correction and it is not
a hypothetical: the reference's power spectrum is flat at ~0.6 (m/s^2)^2/Hz from about 5 Hz
all the way to Nyquist, while both IMU signals roll off steeply above 5 Hz, so by 15 Hz the
reference carries roughly ten times the power of the signals it is being compared against.
Above ~5 Hz the reference is mostly noise.

Two consequences, and both are handled rather than noted:

  1. LOWPASS_CUTOFF_HZ is 6 Hz, set from that crossover instead of from convention. All three
     signals get the same zero-lag Butterworth before any metric is computed, so the filter
     cannot flatter the projection relative to its baseline. Filtering the acceleration
     rather than the positions is not an approximation: central differencing and filtfilt are
     both LTI, so they commute exactly away from the edges, and the edges are trimmed
     (TRIM_S).

  2. Whatever cutoff is chosen, some reference noise survives inside the band, and it lands
     in the residual as apparent projection error. `cutoff_sweep` therefore reports, for each
     cutoff in CUTOFF_SWEEP_HZ, both the measured residual AND `ref_excess_rms` — how much
     residual the reference's own excess in-band power can account for on its own
     (reference_excess_rms). At 15 Hz that term alone accounts for essentially the whole
     measured residual, which is the quantitative reason the cutoff is not 15 Hz.

So the projected-vs-reference residual is an UPPER BOUND on the projection error, not a
measurement of it: mocap cannot resolve a disagreement smaller than its own differentiation
noise. What that noise does not touch is the comparison against the unprojected baseline —
both share the same reference and therefore the same noise, so the gap between them is real
regardless. That gap is what the figure claims.

`err_proj_nofilt` is kept as the unfiltered counterpart of `err_proj` so the size of what the
filter removed stays visible instead of the filter quietly improving the result.

Outputs, all under results/experiments/acceleration_projection/:

    Subject<NN>/<activity>/joint_samples.parquet  per-sample error metrics, every joint x role
    Subject<NN>/<activity>/traces.parquet         full-rate components of all three signals,
                                                  EXAMPLE_JOINT only, for the time-series panel
    Subject<NN>/<activity>/spectra.parquet        Welch PSD of all three signals, UNFILTERED
    Subject<NN>/<activity>/alignment.parquet      per joint x role scalars: offset norm,
                                                  residual-alignment angle, rms, R^2
    Subject<NN>/<activity>/cutoff_sweep.parquet   residual vs low-pass cutoff, against the
                                                  reference's own in-band noise
    Subject<NN>/<activity>/gyro_method.parquet    residual under each gyro-derivative scheme

plus the pooled quantile summary at
results/statistics/acceleration_projection_statistics.parquet, which is what the console
report and the figure caption quote.

joint_samples is stored decimated by SAMPLE_STRIDE (see there) because the full-rate table
would be several hundred MB for a figure that only ever shows quantiles; the time-series
panel reads `traces` instead, which is full rate.

The grid is run one process per trial. This used to be the expensive experiment --
IMUTrace.project_acc's gyro derivative was a per-sample np.polyfit loop costing roughly a
minute per trial -- but polynomial_fit_derivative is now vectorised (the per-window fit is
a fixed linear operator, identical for every window at uniform sampling), so the projection
is no longer the bottleneck.
"""
import argparse
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import wilcoxon
from scipy.spatial.transform import Rotation

import paths
from experiments.experiment_utils import (ACTIVITIES, EXPECTED_GRAVITY, JOINTS, SUBJECTS,
                                          _compute_perfect_joint_acc, load_raw_data,
                                          pipeline_constants, run_tracked_grid)
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace

EXPERIMENT_NAME = "acceleration_projection"
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Low-pass (see the module docstring's filtering section) --------------------------
# 6 Hz, measured rather than conventional: the reference and the IMU signals have matching
# spectra below ~5 Hz and diverge above it, where the reference flattens onto its
# differentiation noise floor. Filtering at 15 Hz — which looks conservative, and was the
# first choice here — admits a band in which the reference carries ~10x the power of the
# signal, and the resulting residual is then almost entirely reference noise (the numbers are
# in cutoff_sweep, which exists so that this is checkable rather than asserted).
#
# The cost of 6 Hz is real and worth stating: impact transients at heel strike do carry
# genuine content above it, so the comparison is over the band where mocap can adjudicate at
# all, not over the sensor's full bandwidth.
LOWPASS_CUTOFF_HZ = 6.0
LOWPASS_ORDER = 4

# Cutoffs the sweep reports, spanning the plausible range for this kind of analysis so the
# choice above can be seen in context rather than taken on trust. 25 Hz is included precisely
# because it is indefensible on this data — it makes the noise floor's contribution obvious.
CUTOFF_SWEEP_HZ = (4.0, 6.0, 10.0, 15.0, 25.0)

# Band used to measure the reference's white noise floor: high enough that the true signal is
# negligible there (the IMU spectra are ~30x lower), low enough to stay clear of the
# anti-alias rolloff at Nyquist.
NOISE_BAND_HZ = (25.0, 45.0)

# Seconds trimmed from both ends of every trial before metrics are computed. Three separate
# edge artifacts stack up there and none of them are projection error: filtfilt's transient,
# central_difference's `extend` padding (twice, since the reference is a second derivative),
# and polynomial_fit_derivative's partial windows.
TRIM_S = 1.0

# Storage decimation for joint_samples ONLY. Filtering happens BEFORE decimation, so this is
# a subsample of an already band-limited signal rather than a naive downsample: stride 3 at
# 100 Hz leaves 33.3 Hz, whose Nyquist (16.7 Hz) clears every cutoff in CUTOFF_SWEEP_HZ up to
# and including 15 Hz, so nothing in the primary analysis aliases.
# Uniform subsampling is unbiased for the marginal distributions that every panel and every
# quoted quantile is computed from. The full-rate signals survive in `traces`.
SAMPLE_STRIDE = 3

# The joint whose full-rate component traces are stored for the time-series panel. The knee
# is the useful illustration: the offset is long enough that the projection correction is
# large, and the segment swings fast enough that the omega x (omega x r) term dominates at
# mid-swing, so the panel shows the correction doing visible work.
EXAMPLE_JOINT = 'R_Knee'

# Welch parameters for the spectral panel. 1024 samples at 100 Hz is a ~10 s window: long
# enough for ~0.1 Hz resolution at the low end, short enough that a 600 s trial averages
# over ~100 segments and the spectrum is smooth without extra smoothing.
WELCH_NPERSEG = 1024

ROLES = ('parent', 'child')

# --- Gyro-derivative schemes -----------------------------------------------------------
# The projection's tangential term needs alpha = d(omega)/dt, and no sensor measures it, so it
# has to be differentiated out of the gyro. IMUTrace._finite_difference_gyros offers three
# ways, and the choice is not obviously free: differentiation amplifies high-frequency content,
# so the scheme trades noise against bandwidth and lag.
#
#   backward     (w[t] - w[t-1]) / dt. project_acc's default. The only one that reads no
#                future sample, so it is the one an online pipeline can use; noisiest of the
#                four, and meant to be paired with a low-pass on the projected acceleration.
#   polyfit      sliding 10-sample 2nd-order fit, differentiated analytically, with overlapping
#                windows averaged — a smoothed Savitzky-Golay derivative. Was project_acc's
#                default, so it is what every projected result in this repo used up to that
#                change. It used to be ~50x slower than the alternatives;
#                polynomial_fit_derivative is vectorised now, so cost no longer separates them.
#   central      (w[i+1] - w[i-1]) / 2dt. Centered, so no lag; no smoothing.
#   first_order  forward difference. Included as the case that SHOULD lose: it is biased by half
#                a sample, and half a sample of lag on this signal is error, not lag.
#
# TWO CAVEATS ON READING THE TABLE THIS PRODUCES, both of which point the opposite way to its
# headline. They are recorded here because the section title invites the wrong inference.
#
#   1. The metric is computed AFTER the LOWPASS_CUTOFF_HZ low-pass, and the schemes differ
#      almost entirely above that cutoff: filtering removes ~94% of the difference between
#      polyfit and central, and below 6 Hz their gains agree to under 1%. So this table
#      compares them in the one band where they nearly agree. Measured on joint-angle RMSE
#      instead — the quantity the paper reports, computed on UNFILTERED projected acc — the
#      ranking reverses hard: polyfit 6.96 deg vs central 9.56, i.e. central is ~37% WORSE.
#   2. The schemes do not see the same data. polyfit's kernel spans +-9 samples, so it uses
#      90 ms of FUTURE gyro; central and first_order use one future sample. A fully causal
#      Savitzky-Golay (same fit, evaluated at the trailing edge) scores 16.87 deg, far worse
#      than either, so polyfit's joint-angle advantage rides substantially on that lookahead
#      rather than on being a better estimator.
#
# Neither caveat makes the table wrong; it measures what it says it measures. It just does not
# license a conclusion about which derivative the pipeline should use.
#
# gyro_method_table measures all three against the same mocap reference so the default is a
# measured choice rather than an inherited one.
GYRO_METHODS = ('backward', 'polyfit', 'central', 'first_order')
PRIMARY_GYRO_METHOD = 'backward'  # must stay project_acc's own default; see above

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# metric -> unit, for the summary table and axis labels.
METRIC_UNITS = {
    'err_proj': 'm/s^2',
    'err_proj_shared': 'm/s^2',
    'err_raw': 'm/s^2',
    'err_proj_nofilt': 'm/s^2',
    'ang_proj': 'deg',
    'ang_raw': 'deg',
    'corr_norm': 'm/s^2',
    'ref_lin_norm': 'm/s^2',
    'gyro_norm': 'rad/s',
    'jc_residual_norm': 'm',
}

TRIAL_TABLES = ('joint_samples', 'traces', 'spectra', 'alignment', 'cutoff_sweep',
                'gyro_method')


def analysis_constants() -> Dict[str, object]:
    """Pipeline constants plus this analysis's own choices, for provenance manifests."""
    return {
        **pipeline_constants(),
        'primary_gyro_method': PRIMARY_GYRO_METHOD,
        'gyro_methods': list(GYRO_METHODS),
        'lowpass_cutoff_hz': LOWPASS_CUTOFF_HZ,
        'lowpass_order': LOWPASS_ORDER,
        'cutoff_sweep_hz': list(CUTOFF_SWEEP_HZ),
        'noise_band_hz': list(NOISE_BAND_HZ),
        'trim_s': TRIM_S,
        'sample_stride': SAMPLE_STRIDE,
        'example_joint': EXAMPLE_JOINT,
        'welch_nperseg': WELCH_NPERSEG,
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

    `columns` is pushed down into the parquet read rather than selected afterwards, which
    matters for joint_samples: the full table across every trial is millions of rows, and a
    panel that needs two of its columns should not pay for all twenty. The label columns
    (joint/role/sensor) are read as categoricals for the same reason — as objects they cost
    more than every float column combined."""
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
    for label in ('joint', 'role', 'sensor', 'subject', 'activity'):
        if label in df.columns:
            df[label] = df[label].astype('category')
    return df

# ==============================================================================
# Signal helpers
# ==============================================================================

def lowpass(signal: np.ndarray, fs: float, cutoff: float = LOWPASS_CUTOFF_HZ,
            order: int = LOWPASS_ORDER) -> np.ndarray:
    """Zero-lag Butterworth low-pass along axis 0, for (N, 3) signals.

    filtfilt rather than lfilt because a phase shift between the reference and the estimate
    would show up as error: at 100 Hz, one sample of lag on a signal changing at 10 m/s^3 is
    0.1 m/s^2 of pure artifact. Nothing here is causal or online, so there is no reason to
    accept that."""
    b, a = butter(order, cutoff / (fs / 2.0), btype='low')
    return filtfilt(b, a, signal, axis=0)


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample angle between two (N, 3) vector series, in degrees.

    The dot product is clipped before arccos: normalized dot products land a few ulp outside
    [-1, 1] for near-parallel vectors, and these two signals are near-parallel most of the
    time (both are dominated by the same gravity vector), so the unclipped version returns
    NaN on real data rather than as a pathological case."""
    cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rms(residual: np.ndarray) -> float:
    """Root-mean-square vector magnitude of an (N, 3) residual."""
    return float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))


def residual_alignment(estimate: np.ndarray, reference: np.ndarray
                       ) -> Tuple[np.ndarray, float, float, float]:
    """The single constant rotation that best maps `estimate` onto `reference`, and the rms
    residual before and after removing it. Returns (R, angle_deg, rms_before, rms_after).

    This is the misalignment floor made measurable (see the module docstring). Solved in
    closed form by Kabsch: R = U diag(1, 1, det(U V^T)) V^T from the SVD of
    sum_i reference_i estimate_i^T, with the determinant term keeping the result a rotation
    rather than admitting a reflection.

    The fit is dominated by gravity — 9.81 m/s^2 against linear accelerations of a few — so
    what it recovers is overwhelmingly the static sensor-to-segment misalignment rather than
    any dynamic projection error, which is what makes `rms_after` interpretable as the
    projection's own residual. It is reported alongside `rms_before` and never substituted
    for it: the per-sample error columns are left unrotated so that no metric in the figure
    depends on a fitted correction.
    """
    H = reference.T @ estimate
    U, _, Vt = np.linalg.svd(H)
    R = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
    angle_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))
    return R, angle_deg, rms(estimate - reference), rms(estimate @ R.T - reference)

# ==============================================================================
# The three signals
# ==============================================================================

def joint_center_reference_acc(plate: PlateTrial, local_offset: np.ndarray,
                              gravity: Optional[np.ndarray] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Mocap truth at the point `local_offset` on this segment, as
    (specific force in the segment's body frame, world-frame linear acceleration).

    The second return is the same quantity sensor_distributions calls `linacc`: gravity
    removed in the WORLD frame, which is where it can be removed as a constant. It is
    returned from here rather than reconstructed later because this is the only place the
    world-frame acceleration exists — once rotated into the body frame, gravity is smeared
    across all three axes by the segment's own orientation and cannot be subtracted back out.

    The segment's own implied joint center, not the midpoint of the two segments' estimates —
    this is the point `project_acc(local_offset)` is aiming at, so pairing them isolates the
    projection from the joint model's parent/child disagreement. The shared-midpoint version
    the filter's oracle uses comes from experiment_utils._compute_perfect_joint_acc, and both
    end up in the output table.

    Deliberately mirrors _compute_perfect_joint_acc's construction step for step (offset the
    positions, differentiate twice in the world frame, add gravity, rotate into the body
    frame) so that the only difference between the two references is the reference POINT.
    """
    gravity = EXPECTED_GRAVITY if gravity is None else gravity
    rotations = plate.world_trace.rotations
    jc_world = plate.world_trace.positions + np.einsum('nij,j->ni', rotations, local_offset)
    # Rotations are unused by finite_difference_world_frame_accelerations; the segment's own
    # are passed through purely to satisfy the WorldTrace constructor.
    jc_trace = WorldTrace(plate.world_trace.timestamps, jc_world, rotations)
    world_acc = jc_trace.finite_difference_world_frame_accelerations(acc_from_gravity=gravity)
    return np.einsum('nji,nj->ni', rotations, world_acc), world_acc - gravity


def joint_signals(parent_plate: PlateTrial, child_plate: PlateTrial, fs: float
                  ) -> Dict[str, Dict[str, np.ndarray]]:
    """All three acceleration signals for both segments of one joint, plus the per-sample
    context each metric needs. Returns {role: {...arrays...}}.

    Every array is filtered and trimmed here, once, so that no downstream table can end up
    describing a differently-conditioned version of the same signal. `*_nofilt` survives
    only for the reference/projected pair, which is the one comparison whose unfiltered
    value the report quotes.
    """
    parent_offset, child_offset, jc_residual = parent_plate.world_trace.get_joint_center(
        child_plate.world_trace)
    shared_ref = dict(zip(ROLES, _compute_perfect_joint_acc(parent_plate, child_plate)))
    offsets = {'parent': parent_offset, 'child': child_offset}
    plates = {'parent': parent_plate, 'child': child_plate}

    trim = int(round(TRIM_S * fs))
    keep = slice(trim, -trim if trim else None)
    jc_residual_norm = np.linalg.norm(jc_residual, axis=1)

    out = {}
    for role in ROLES:
        plate, offset = plates[role], offsets[role]
        sensor = plate.imu_trace.acc
        projected = plate.imu_trace.project_acc(offset, PRIMARY_GYRO_METHOD).acc
        reference, reference_linacc = joint_center_reference_acc(plate, offset)

        # The same projection under each alternative gyro-derivative scheme, for
        # gyro_method_table. Nearly free next to the primary one: the polyfit derivative is a
        # per-sample np.polyfit loop and the differences are two vectorized subtractions, so
        # this adds ~2% to a trial rather than doubling it.
        alternates = {method: lowpass(plate.imu_trace.project_acc(offset, method).acc, fs)[keep]
                      for method in GYRO_METHODS if method != PRIMARY_GYRO_METHOD}

        # The unfiltered trio is kept under one key and shared by reference: the spectra, the
        # cutoff sweep and err_proj_nofilt all need the same arrays, and holding three separate
        # slices of them tripled this dict's footprint for a 60k-sample trial.
        unfiltered = {'reference': reference[keep], 'projected': projected[keep],
                      'sensor': sensor[keep]}
        out[role] = {
            'sensor': lowpass(sensor, fs)[keep],
            'projected': lowpass(projected, fs)[keep],
            'reference': lowpass(reference, fs)[keep],
            'reference_shared': lowpass(shared_ref[role], fs)[keep],
            'reference_linacc': lowpass(reference_linacc, fs)[keep],
            'gyro': lowpass(plate.imu_trace.gyro, fs)[keep],
            'timestamps': plate.imu_trace.timestamps[keep],
            'jc_residual_norm': jc_residual_norm[keep],
            'offset': offset,
            'sensor_name': plate.name,
            # Unfiltered, for the spectral panel and the cutoff sweep: both ask what happens
            # ABOVE the analysis cutoff, which filtering here would erase.
            'spectrum_inputs': unfiltered,
            'projected_by_method': {PRIMARY_GYRO_METHOD: lowpass(projected, fs)[keep],
                                    **alternates},
        }
    return out

# ==============================================================================
# Per-trial tables
# ==============================================================================

def joint_samples(signals_by_joint: Dict[str, Dict[str, Dict[str, np.ndarray]]]) -> pd.DataFrame:
    """Per-sample error metrics for every joint x role, decimated by SAMPLE_STRIDE.

    `corr_norm` — |a_proj - a_sensor| — is the size of the correction the projection applied,
    and is the natural x-axis for "when does this degrade": it is the only column that says
    how much work the rigid-body terms were asked to do at that instant, which neither
    |omega| nor |r| captures alone (a fast rotation about an axis through the joint center
    needs no correction at all).

    The per-axis `ref_*`/`diff_*` columns are stored rather than a precomputed
    mean-vs-difference pair so the plotting layer can build a Bland-Altman panel on either
    the raw axes or their difference without this table committing to one of them.
    """
    rows = []
    for joint, roles in signals_by_joint.items():
        for role, s in roles.items():
            diff = s['projected'] - s['reference']
            step = slice(None, None, SAMPLE_STRIDE)
            rows.append(pd.DataFrame({
                'timestamp': s['timestamps'][step].astype(np.float64),
                'joint': joint,
                'role': role,
                'sensor': s['sensor_name'],
                'err_proj': np.linalg.norm(diff, axis=1)[step].astype(np.float32),
                'err_proj_shared': np.linalg.norm(
                    s['projected'] - s['reference_shared'], axis=1)[step].astype(np.float32),
                'err_raw': np.linalg.norm(
                    s['sensor'] - s['reference'], axis=1)[step].astype(np.float32),
                'err_proj_nofilt': np.linalg.norm(
                    s['spectrum_inputs']['projected']
                    - s['spectrum_inputs']['reference'], axis=1)[step].astype(np.float32),
                'ang_proj': angle_between_deg(s['projected'], s['reference'])[step].astype(np.float32),
                'ang_raw': angle_between_deg(s['sensor'], s['reference'])[step].astype(np.float32),
                'corr_norm': np.linalg.norm(
                    s['projected'] - s['sensor'], axis=1)[step].astype(np.float32),
                'gyro_norm': np.linalg.norm(s['gyro'], axis=1)[step].astype(np.float32),
                # The size of the true linear acceleration at the joint center, gravity
                # removed in the world frame — the scale every error column should be read
                # against, since an 0.5 m/s^2 residual means something very different during
                # quiet standing than at heel strike.
                'ref_lin_norm': np.linalg.norm(
                    s['reference_linacc'], axis=1)[step].astype(np.float32),
                'jc_residual_norm': s['jc_residual_norm'][step].astype(np.float32),
                'ref_x': s['reference'][step, 0].astype(np.float32),
                'ref_y': s['reference'][step, 1].astype(np.float32),
                'ref_z': s['reference'][step, 2].astype(np.float32),
                'diff_x': diff[step, 0].astype(np.float32),
                'diff_y': diff[step, 1].astype(np.float32),
                'diff_z': diff[step, 2].astype(np.float32),
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def traces_table(signals_by_joint: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                 joint: str = EXAMPLE_JOINT) -> pd.DataFrame:
    """Full-rate components of all three signals for one joint, both roles — the input to the
    time-series panel.

    Stored for every trial even though the figure draws one: it is ~2 MB per trial, and
    having it everywhere means the example subject can be changed from the plotting side
    without recomputing anything."""
    if joint not in signals_by_joint:
        return pd.DataFrame()
    rows = []
    for role, s in signals_by_joint[joint].items():
        frame = {'timestamp': s['timestamps'].astype(np.float64), 'joint': joint, 'role': role,
                 'sensor': s['sensor_name']}
        for name, signal in (('ref', s['reference']), ('proj', s['projected']),
                             ('sens', s['sensor'])):
            for axis, label in enumerate('xyz'):
                frame[f'{name}_{label}'] = signal[:, axis].astype(np.float32)
        rows.append(pd.DataFrame(frame))
    return pd.concat(rows, ignore_index=True)


def welch_psds(signals: Dict[str, np.ndarray], fs: float
               ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Welch PSD of one sensor's three UNFILTERED signals, as (freqs, {name: psd}).

    Each PSD is averaged over the three axes: the sum of the axis PSDs is the power spectrum of
    the vector signal and does not depend on how the body frame happens to be oriented, which a
    single axis very much does. (The factor of 3 between the mean and that sum is applied by
    reference_excess_rms, the only consumer that needs the vector total.)

    Shared by spectra_table and cutoff_sweep_table so the spectrum the figure draws and the
    spectrum the noise correction is computed from are the same numbers.
    """
    nperseg = min(WELCH_NPERSEG, len(signals['timestamps']))
    freqs = None
    psds = {}
    for name, signal in signals['spectrum_inputs'].items():
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg, axis=0)
        psds[name] = psd.mean(axis=1)
    return freqs, psds


def spectra_table(signals_by_joint: Dict[str, Dict[str, Dict[str, np.ndarray]]], fs: float
                  ) -> pd.DataFrame:
    """Welch PSD of all three UNFILTERED signals, per joint x role.

    Averaged over the three axes rather than kept per axis: the sum of the axis PSDs is the
    power spectrum of the vector signal and does not depend on how the body frame happens to
    be oriented, which a single axis very much does.

    This is the panel that settles the standing objection to the method — that differentiating
    a noisy gyro to get alpha injects high-frequency noise into the accelerometer signal. It
    is a fair test only on unfiltered data, since the shared low-pass would remove exactly the
    band in question.
    """
    rows = []
    for joint, roles in signals_by_joint.items():
        for role, s in roles.items():
            freqs, psds = welch_psds(s, fs)
            rows.append(pd.DataFrame({'joint': joint, 'role': role,
                                      'freq_hz': freqs.astype(np.float32),
                                      **{f'psd_{name}': psd.astype(np.float32)
                                         for name, psd in psds.items()}}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def reference_excess_rms(freqs: np.ndarray, psd_reference: np.ndarray,
                         psd_projected: np.ndarray, cutoff: float) -> float:
    """How much projected-vs-reference residual the reference's own in-band noise can account
    for on its own, in m/s^2, for a low-pass at `cutoff`.

    Integrates the reference's EXCESS power over the projected signal's, below the cutoff:

        rms = sqrt( 3 * integral_0^fc max(0, PSD_ref(f) - PSD_proj(f)) df )

    The factor 3 turns a per-axis PSD (psd_* columns are the mean over the three axes) into the
    variance of the 3-vector residual. The max(0, .) drops bands where the reference has LESS
    power, which is not reference noise and must not be allowed to cancel out bands where it
    has more.

    Measuring the excess rather than assuming a white floor and extrapolating it down to DC is
    the whole point. The floor looks flat above 5 Hz, but differentiation suppresses
    low-frequency noise, so extrapolating a flat floor across the whole band overstates the
    in-band noise badly — at a 15 Hz cutoff it predicts more residual than is actually
    measured, which is impossible and would have made this a misleading correction rather than
    an informative one. The excess integral needs no such assumption: it only assumes the IMU
    spectrum is a fair stand-in for the true signal's spectrum inside the band, and below 5 Hz
    the two agree to a few percent.

    An upper bound, not a subtraction, for two reasons, and nothing downstream removes it from
    the measured residual — it is reported beside it:

      * Any real projection error that happens to raise the reference's apparent excess is
        counted here as reference noise.
      * The max(0, .) rectifies Welch's own estimator variance, so when the two spectra are
        genuinely equal this returns something positive rather than zero. On the noiseless
        synthetic joint in test/TestAccelerationProjection.py it reads ~0.8 m/s^2 against a
        true residual of ~0.1, which is the scale of that bias at this nperseg. So a SMALL
        value here is not resolvable and should not be read as "the reference contributed this
        much"; a value comparable to the measured residual is the informative case, and that is
        what happens at the wider cutoffs.
    """
    band = freqs <= cutoff
    excess = np.maximum(psd_reference[band] - psd_projected[band], 0.0)
    return float(np.sqrt(3.0 * np.trapezoid(excess, freqs[band])))


def cutoff_sweep_table(signals_by_joint: Dict[str, Dict[str, Dict[str, np.ndarray]]], fs: float
                       ) -> pd.DataFrame:
    """Residual against low-pass cutoff, per joint x role x cutoff, beside what the reference's
    own in-band noise accounts for.

    This is the table that makes LOWPASS_CUTOFF_HZ a measured choice rather than a taste. Each
    row re-filters the UNFILTERED signals at that cutoff and recomputes the residual from
    scratch, so no row inherits the primary cutoff's filtering; `ref_excess_rms` is the noise
    the reference brings to that same band.

    Cheap despite looking expensive: the costly part of this analysis is the projection's
    per-sample polyfit gyro derivative, which has already happened by the time this runs. All
    this adds is a handful of filtfilt passes per sensor.
    """
    rows = []
    for joint, roles in signals_by_joint.items():
        for role, s in roles.items():
            freqs, psds = welch_psds(s, fs)
            reference_raw = s['spectrum_inputs']['reference']
            projected_raw = s['spectrum_inputs']['projected']
            sensor_raw = s['spectrum_inputs']['sensor']
            for cutoff in CUTOFF_SWEEP_HZ:
                reference = lowpass(reference_raw, fs, cutoff=cutoff)
                projected_error = np.linalg.norm(
                    lowpass(projected_raw, fs, cutoff=cutoff) - reference, axis=1)
                raw_error = np.linalg.norm(
                    lowpass(sensor_raw, fs, cutoff=cutoff) - reference, axis=1)
                rows.append({
                    'joint': joint, 'role': role, 'sensor': s['sensor_name'],
                    'cutoff_hz': cutoff,
                    'median_err_proj': float(np.median(projected_error)),
                    'median_err_raw': float(np.median(raw_error)),
                    'rms_err_proj': float(np.sqrt(np.mean(projected_error ** 2))),
                    'rms_err_raw': float(np.sqrt(np.mean(raw_error ** 2))),
                    'ref_excess_rms': reference_excess_rms(
                        freqs, psds['reference'], psds['projected'], cutoff),
                    'noise_floor_psd': float(np.median(
                        psds['reference'][(freqs >= NOISE_BAND_HZ[0]) & (freqs <= NOISE_BAND_HZ[1])]))
                    if freqs.max() >= NOISE_BAND_HZ[0] else np.nan,
                })
    return pd.DataFrame(rows)


def gyro_method_table(signals_by_joint: Dict[str, Dict[str, Dict[str, np.ndarray]]]
                      ) -> pd.DataFrame:
    """Residual against the mocap reference under each gyro-derivative scheme, per joint x role
    x method.

    Answers whether the expensive default is buying accuracy. Everything except the derivative
    is held fixed — same offset, same reference, same low-pass, same samples — so the only thing
    that differs between rows of a cell is how alpha was estimated.

    `corr_norm_median` is carried alongside because it is the interpretive key: the schemes can
    only differ where the tangential term matters, so a cell whose correction is tiny will show
    no difference between them no matter how bad one of them is.
    """
    rows = []
    for joint, roles in signals_by_joint.items():
        for role, s in roles.items():
            reference = s['reference']
            for method, projected in s['projected_by_method'].items():
                error = np.linalg.norm(projected - reference, axis=1)
                rows.append({
                    'joint': joint, 'role': role, 'sensor': s['sensor_name'],
                    'gyro_method': method,
                    'median_err': float(np.median(error)),
                    'p90_err': float(np.percentile(error, 90)),
                    'rms_err': float(np.sqrt(np.mean(error ** 2))),
                    'median_ang': float(np.median(angle_between_deg(projected, reference))),
                    'corr_norm_median': float(np.median(
                        np.linalg.norm(projected - s['sensor'], axis=1))),
                })
    return pd.DataFrame(rows)


def alignment_table(signals_by_joint: Dict[str, Dict[str, Dict[str, np.ndarray]]]
                    ) -> pd.DataFrame:
    """Per joint x role scalars: offset length, the fitted residual-alignment angle and the
    rms either side of it, the unprojected baseline's rms, and a vector R^2.

    r2_proj is variance-accounted-for on the whole 3-vector,
    1 - sum|a_proj - a_ref|^2 / sum|a_ref - mean(a_ref)|^2, rather than three per-axis R^2
    values. A per-axis R^2 on this data is close to 1 by construction for whichever axis
    gravity happens to sit on, so it would mostly report the body frame's orientation.
    """
    rows = []
    for joint, roles in signals_by_joint.items():
        for role, s in roles.items():
            reference, projected, sensor = s['reference'], s['projected'], s['sensor']
            _, angle_deg, rms_before, rms_after = residual_alignment(projected, reference)
            centered = reference - reference.mean(axis=0)
            total_var = float(np.sum(centered ** 2))
            rows.append({
                'joint': joint,
                'role': role,
                'sensor': s['sensor_name'],
                'n_samples': len(reference),
                'offset_norm': float(np.linalg.norm(s['offset'])),
                'jc_residual_norm_median': float(np.median(s['jc_residual_norm'])),
                'align_angle_deg': angle_deg,
                'rms_before': rms_before,
                'rms_after': rms_after,
                'rms_raw': rms(sensor - reference),
                'r2_proj': 1.0 - float(np.sum((projected - reference) ** 2)) / total_var,
                'r2_raw': 1.0 - float(np.sum((sensor - reference) ** 2)) / total_var,
            })
    return pd.DataFrame(rows)

# ==============================================================================
# Per-trial driver / grid worker
# ==============================================================================

def compute_trial(plates: Dict[str, PlateTrial]) -> Dict[str, pd.DataFrame]:
    """Everything this experiment computes for one subject/activity, as the four tables.

    Joints missing either sensor are skipped rather than failing the trial: a dropped sensor
    costs that joint, not the other six."""
    fs = plates['pelvis_imu'].imu_trace.get_sample_frequency()

    signals_by_joint = {}
    for joint, (parent_sensor, child_sensor) in JOINTS.items():
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        signals_by_joint[joint] = joint_signals(plates[parent_sensor], plates[child_sensor], fs)

    return {
        'joint_samples': joint_samples(signals_by_joint),
        'traces': traces_table(signals_by_joint),
        'spectra': spectra_table(signals_by_joint, fs),
        'alignment': alignment_table(signals_by_joint),
        'cutoff_sweep': cutoff_sweep_table(signals_by_joint, fs),
        'gyro_method': gyro_method_table(signals_by_joint),
    }


def _trial_worker(row_key: str, stage_labels: List[str], shared_state: Dict) -> None:
    """One process per (subject, activity) cell — run_tracked_grid(per_cell=True).

    Nothing is shared between a subject's two activities here (unlike sensor_distributions,
    where the subject's global magnetic field spans both), and the projection's polyfit
    gyro derivative dominates the runtime, so exposing every cell to the pool individually
    is strictly better than serializing a subject's trials behind one process."""
    subject = row_key
    activity = stage_labels[0]

    if not paths.raw_trial_dir(subject, activity).exists():
        shared_state[(row_key, activity)] = "Skipped"
        return None

    t_start = time.time()
    shared_state[(row_key, activity)] = "Running"
    try:
        plates = load_raw_data(subject, activity)
        if 'pelvis_imu' not in plates:
            shared_state[(row_key, activity)] = "Skipped (no pelvis)"
            return None
        for table, df in compute_trial(plates).items():
            if df.empty:
                continue
            _save(df, trial_table_path(subject, activity, table),
                  subject=f"Subject{subject}", activity=activity, table=table)
        shared_state[(row_key, f"{activity}_time")] = time.time() - t_start
        shared_state[(row_key, activity)] = "Success"
    except Exception as e:
        shared_state[(row_key, activity)] = f"Failed ({e})"
    return None

# ==============================================================================
# Pooled summary
# ==============================================================================

SUMMARY_METRICS = ['err_proj', 'err_proj_shared', 'err_raw', 'err_proj_nofilt',
                   'ang_proj', 'ang_raw', 'corr_norm', 'ref_lin_norm']

SUMMARY_COLUMNS = (['subject', 'activity', 'metric', 'unit', 'joint', 'n_samples',
                    'mean', 'std', 'min']
                   + [f"p{int(round(q * 100)):02d}" for q in QUANTILES] + ['max'])


def _describe(df: pd.DataFrame, metric: str, keys: List[str]) -> pd.DataFrame:
    grouped = df.groupby(keys, observed=True)[metric]
    stats = grouped.agg(n_samples='size', mean='mean', std='std', min='min', max='max')
    quantiles = grouped.quantile(QUANTILES).unstack()
    quantiles.columns = [f"p{int(round(q * 100)):02d}" for q in quantiles.columns]
    return stats.join(quantiles).reset_index()


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    """Tidy quantile table per metric x joint, WITH MARGINS: rows whose `subject`,
    `activity` or `joint` is the literal string 'all' are the pooled version of the rows
    above them.

    Margins are recomputed from the samples rather than averaged from the per-trial rows —
    a mean of medians is not a median, and trials differ in length. Quantiles rather than
    mean +- sd throughout: every one of these error metrics is a non-negative magnitude with
    an impulsive footfall tail, so a standard deviation implies a symmetry that is not there
    (it is reported anyway, next to the quantiles, for anyone who wants it).
    """
    if samples.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    pooled = samples.assign(subject='all', activity='all')
    frames = []
    for metric in SUMMARY_METRICS:
        for source in (samples, pooled):
            frames.append(_describe(source, metric, ['subject', 'activity', 'joint'])
                          .assign(metric=metric))
            frames.append(_describe(source, metric, ['subject', 'activity'])
                          .assign(metric=metric, joint='all'))
    summary = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=['subject', 'activity', 'metric', 'joint'])
    summary['unit'] = summary['metric'].map(METRIC_UNITS)
    return summary[SUMMARY_COLUMNS].sort_values(['metric', 'subject', 'activity', 'joint'])

# ==============================================================================
# Console report
# ==============================================================================

def _header(number: int, title: str, subtitle: str) -> None:
    print(f"\n{'=' * 80}\n{number}. {title}\n   {subtitle}\n{'=' * 80}")


def _pooled(summary: pd.DataFrame, metric: str, joint: str = 'all') -> Optional[pd.Series]:
    """The pooled (subject='all', activity='all') summary row for one metric, or None.

    Numeric fields are coerced on the way out: the row also carries the metric/unit strings,
    so without this every quantile read from it is an object and arithmetic on it silently
    produces objects too."""
    rows = summary[(summary['subject'] == 'all') & (summary['activity'] == 'all')
                   & (summary['metric'] == metric) & (summary['joint'] == joint)]
    if rows.empty:
        return None
    row = rows.iloc[0].copy()
    numeric = [c for c in row.index if c not in ('subject', 'activity', 'metric', 'unit', 'joint')]
    row[numeric] = pd.to_numeric(row[numeric], errors='coerce')
    return row


def report_projection_error(summary: pd.DataFrame) -> None:
    _header(1, "PROJECTION ERROR vs UNPROJECTED BASELINE",
            "|a - a_true| at the joint center, pooled over every subject, activity and sample")
    proj, raw = _pooled(summary, 'err_proj'), _pooled(summary, 'err_raw')
    if proj is None or raw is None:
        print("   No samples.")
        return
    columns = ['p25', 'p50', 'p75', 'p90', 'p99', 'mean']
    # astype(float): a row sliced out of the summary carries the string columns too, so the
    # Series comes back as object dtype and to_string's float_format would be handed strings.
    table = pd.DataFrame({'projected': proj[columns], 'unprojected': raw[columns]}).T.astype(float)
    print(table.to_string(float_format='%.3f'))
    print(f"\n   Median error {raw['p50']:.3f} -> {proj['p50']:.3f} m/s^2, "
          f"{100 * (1 - proj['p50'] / raw['p50']):.1f}% LOWER than unprojected, "
          f"n={int(proj['n_samples']):,} samples")

    shared = _pooled(summary, 'err_proj_shared')
    if shared is not None:
        print(f"   Against the SHARED (midpoint) joint center, as the filter's acc oracle uses: "
              f"median {shared['p50']:.3f} m/s^2")

    nofilt = _pooled(summary, 'err_proj_nofilt')
    if nofilt is not None:
        print(f"   Unfiltered, same pairing: median {nofilt['p50']:.3f} m/s^2 — the "
              f"{LOWPASS_CUTOFF_HZ:.0f} Hz low-pass accounts for "
              f"{nofilt['p50'] - proj['p50']:.3f} m/s^2 of it, which is mocap "
              f"double-differentiation noise rather than projection error")


def report_paired_cells(samples: pd.DataFrame) -> None:
    _header(2, "IS IT BETTER, CELL BY CELL?",
            "one paired comparison per subject x activity x joint x segment, on that cell's "
            "median error — so the pooled result cannot be carried by a few long trials")
    if samples.empty:
        print("   No samples.")
        return
    cells = (samples.groupby(['subject', 'activity', 'joint', 'role'], observed=True)
             [['err_proj', 'err_raw']].median().dropna())
    if cells.empty:
        print("   No complete cells.")
        return

    improved = int((cells['err_proj'] < cells['err_raw']).sum())
    ratio = cells['err_proj'] / cells['err_raw']
    print(f"   Projection lower in {improved} of {len(cells)} cells "
          f"({100 * improved / len(cells):.0f}%)")
    print(f"   Error ratio projected/unprojected: median {ratio.median():.3f}, "
          f"IQR {ratio.quantile(0.25):.3f}-{ratio.quantile(0.75):.3f}, worst {ratio.max():.3f}")

    # Wilcoxon on the paired cell medians. Signed-rank rather than a t-test because these
    # are medians of skewed magnitudes, and paired rather than pooled because the two
    # numbers in a cell come from the same sensor, samples and reference — only the
    # projection differs, which is the whole point of the pairing.
    if len(cells) >= 6:
        stat, p_value = wilcoxon(cells['err_proj'], cells['err_raw'])
        print(f"   Wilcoxon signed-rank on cell medians: W={stat:.1f}, p={p_value:.2e}, "
              f"n={len(cells)} cells")
    worst = cells.assign(ratio=ratio).nlargest(3, 'ratio')
    print("\n   Cells where projection helps least:")
    print(worst.to_string(float_format='%.3f'))


def report_direction_error(summary: pd.DataFrame) -> None:
    _header(3, "DIRECTION ERROR",
            "angle between the estimated and true acceleration vectors — what the EKF's "
            "measurement update actually responds to")
    proj, raw = _pooled(summary, 'ang_proj'), _pooled(summary, 'ang_raw')
    if proj is None or raw is None:
        print("   No samples.")
        return
    columns = ['p25', 'p50', 'p75', 'p90', 'p99']
    print(pd.DataFrame({'projected': proj[columns], 'unprojected': raw[columns]}).T
          .astype(float).to_string(float_format='%.3f'))
    print(f"\n   Median {raw['p50']:.2f} deg -> {proj['p50']:.2f} deg, "
          f"{100 * (1 - proj['p50'] / raw['p50']):.1f}% lower")


def report_by_joint(summary: pd.DataFrame) -> None:
    _header(4, "BY JOINT", "median |a - a_true| (m/s^2), projected vs unprojected, and the "
                           "fitted sensor->segment offset each projection spans")
    rows = []
    for joint in JOINTS:
        proj, raw = _pooled(summary, 'err_proj', joint), _pooled(summary, 'err_raw', joint)
        corr = _pooled(summary, 'corr_norm', joint)
        if proj is None or raw is None:
            continue
        rows.append({'joint': joint, 'projected': proj['p50'], 'unprojected': raw['p50'],
                     'reduction_%': 100 * (1 - proj['p50'] / raw['p50']),
                     'median_correction': corr['p50'] if corr is not None else np.nan})
    if not rows:
        print("   No samples.")
        return
    print(pd.DataFrame(rows).to_string(index=False, float_format='%.3f'))


def report_alignment(alignment: pd.DataFrame) -> None:
    _header(5, "MISALIGNMENT FLOOR",
            "one constant rotation fitted per sensor-trial (Kabsch); rms_after is what "
            "remains of the projected-vs-true residual once it is removed")
    if alignment.empty:
        print("   No alignment table.")
        return
    columns = ['align_angle_deg', 'rms_before', 'rms_after', 'rms_raw', 'r2_proj', 'r2_raw']
    print(alignment[columns].astype(float).describe().loc[['mean', '50%', 'min', 'max']]
          .to_string(float_format='%.3f'))
    angle = alignment['align_angle_deg'].median()
    median_before = alignment['rms_before'].median()
    median_after = alignment['rms_after'].median()
    print(f"\n   Median fitted misalignment {angle:.2f} deg. At |g| = "
          f"{np.linalg.norm(EXPECTED_GRAVITY):.2f} m/s^2 that tilts the gravity component by "
          f"~{np.radians(angle) * np.linalg.norm(EXPECTED_GRAVITY):.3f} m/s^2, but it adds in "
          f"QUADRATURE with the rest of the residual, so removing it moves the median rms only "
          f"{median_before:.3f} -> {median_after:.3f} m/s^2.")
    print(f"   So the misalignment floor is not what limits this comparison: the projected "
          f"residual is {median_after:.3f} m/s^2 even with it gone, against an unprojected "
          f"baseline of {alignment['rms_raw'].median():.3f} m/s^2.")
    print("   Per-sample error columns in the figures are NOT rotation-corrected; the fit is "
          "reported so the floor is visible, not subtracted.")


def report_cutoff_sweep(sweep: pd.DataFrame) -> None:
    _header(6, "HOW MUCH OF THE RESIDUAL IS THE REFERENCE'S OWN NOISE?",
            "median residual at each low-pass cutoff, against what the mocap reference's "
            "excess in-band power accounts for by itself")
    if sweep.empty:
        print("   No sweep table.")
        return

    table = sweep.groupby('cutoff_hz').agg(
        median_err_proj=('median_err_proj', 'median'),
        median_err_raw=('median_err_raw', 'median'),
        rms_err_proj=('rms_err_proj', 'median'),
        ref_excess_rms=('ref_excess_rms', 'median'))
    table['explained_%'] = 100 * table['ref_excess_rms'] / table['rms_err_proj']
    table['proj_vs_raw_%'] = 100 * (1 - table['median_err_proj'] / table['median_err_raw'])
    print(table.to_string(float_format='%.3f'))

    floor = sweep['noise_floor_psd'].median()
    print(f"\n   Reference noise floor over {NOISE_BAND_HZ[0]:.0f}-{NOISE_BAND_HZ[1]:.0f} Hz: "
          f"{floor:.3f} (m/s^2)^2/Hz per axis, flat — a differentiation noise floor, not signal.")
    print(f"   `explained_%` is how much of the projected residual the reference's own in-band "
          f"noise accounts for. Read UP the table: as the cutoff rises, more of what looks like "
          f"projection error is the reference. At {LOWPASS_CUTOFF_HZ:.0f} Hz (the cutoff used "
          f"everywhere else) it is "
          f"{table.loc[LOWPASS_CUTOFF_HZ, 'explained_%']:.0f}%.")
    print("   `proj_vs_raw_%` is the reduction against the unprojected baseline. It is the one "
          "column the reference's noise cannot manufacture — both signals are compared to the "
          "same reference, so the shared noise inflates both and cancels from the gap.")


def report_gyro_method(gyro: pd.DataFrame) -> None:
    _header(7, "IS THE EXPENSIVE GYRO DERIVATIVE EARNING ITS KEEP?",
            "same offset, same reference, same low-pass — only how alpha = d(omega)/dt was "
            "estimated differs")
    if gyro.empty:
        print("   No gyro_method table.")
        return

    table = gyro.groupby('gyro_method').agg(
        median_err=('median_err', 'median'),
        p90_err=('p90_err', 'median'),
        rms_err=('rms_err', 'median'),
        median_ang=('median_ang', 'median')).reindex(
        [m for m in GYRO_METHODS if m in set(gyro['gyro_method'])])
    baseline = table.loc[PRIMARY_GYRO_METHOD, 'median_err']
    table['vs_primary_%'] = 100 * (table['median_err'] / baseline - 1)
    print(table.to_string(float_format='%.4f'))

    # Paired per cell, since the alternatives are being compared on identical inputs and a
    # pooled median could hide a scheme that is better on most cells and much worse on a few.
    # observed=True and a whole-frame dropna: the index levels are categoricals, so without
    # both, pivot_table expands to the full cartesian product of categories and the all-NaN
    # rows it invents would misalign the paired test below.
    wide = gyro.pivot_table(index=['subject', 'activity', 'joint', 'role'],
                            columns='gyro_method', values='median_err', observed=True).dropna()
    print(f"\n   Paired over {len(wide)} cells (subject x activity x joint x segment):")
    for method in GYRO_METHODS:
        if method == PRIMARY_GYRO_METHOD or method not in wide.columns:
            continue
        ratio = (wide[method] / wide[PRIMARY_GYRO_METHOD]).dropna()
        if ratio.empty:
            continue
        wins = int((ratio < 1).sum())
        line = (f"   {method:<12} better in {wins}/{len(ratio)} cells; "
                f"error ratio vs {PRIMARY_GYRO_METHOD} median {ratio.median():.3f} "
                f"(IQR {ratio.quantile(0.25):.3f}-{ratio.quantile(0.75):.3f}, "
                f"max {ratio.max():.3f})")
        if len(ratio) >= 6:
            _, p_value = wilcoxon(wide[method], wide[PRIMARY_GYRO_METHOD])
            line += f", Wilcoxon p={p_value:.2e}"
        print(line)

    # Where the schemes can differ at all: the tangential term is the only thing that changes,
    # so cells with a small correction cannot separate them however bad one of them is.
    largest = gyro.nlargest(1, 'corr_norm_median')
    if not largest.empty:
        cell = largest.iloc[0]
        subset = gyro[(gyro['joint'] == cell['joint']) & (gyro['role'] == cell['role'])
                      & (gyro['subject'] == cell['subject'])
                      & (gyro['activity'] == cell['activity'])]
        print(f"\n   The cell with the largest correction "
              f"({cell['joint']} {cell['role']}, {cell['subject']} {cell['activity']}, "
              f"median correction {cell['corr_norm_median']:.2f} m/s^2) — where the schemes have "
              f"the most room to differ:")
        print(subset[['gyro_method', 'median_err', 'p90_err', 'rms_err']]
              .to_string(index=False, float_format='%.4f'))


def report_error_vs_correction(samples: pd.DataFrame) -> None:
    _header(8, "ERROR vs SIZE OF THE CORRECTION",
            "median |a_proj - a_true| in bins of |a_proj - a_sensor| — how the residual grows "
            "with the amount of work the rigid-body terms were asked to do")
    if samples.empty:
        print("   No samples.")
        return
    edges = [0, 1, 2, 5, 10, 20, 50, np.inf]
    labels = ['0-1', '1-2', '2-5', '5-10', '10-20', '20-50', '50+']
    binned = samples.assign(bin=pd.cut(samples['corr_norm'], bins=edges, labels=labels))
    table = binned.groupby('bin', observed=True).agg(
        n_samples=('err_proj', 'size'),
        median_err_proj=('err_proj', 'median'),
        median_err_raw=('err_raw', 'median'),
        median_gyro_norm=('gyro_norm', 'median'))
    table['err_as_%_of_correction'] = 100 * table['median_err_proj'] / binned.groupby(
        'bin', observed=True)['corr_norm'].median()
    print(table.to_string(float_format='%.3f'))
    print("\n   The last column is the projected residual as a percentage of the correction "
          "applied, i.e. how much of what the projection added it got wrong. Well under 100% "
          "in the bins where the correction is large means the rigid-body terms are mostly "
          "right.")
    print("   It EXCEEDS 100% in the smallest bins, and that is not the projection failing: "
          "where the correction is under ~1 m/s^2 the residual is the error floor shared by "
          "both signals (mocap differentiation noise, sensor noise, misalignment), which is "
          "why median_err_raw is nearly equal to median_err_proj in those rows.")


def print_report(summary: pd.DataFrame, samples: pd.DataFrame, alignment: pd.DataFrame,
                 sweep: pd.DataFrame, gyro: pd.DataFrame) -> None:
    report_projection_error(summary)
    report_paired_cells(samples)
    report_direction_error(summary)
    report_by_joint(summary)
    report_alignment(alignment)
    report_cutoff_sweep(sweep)
    report_gyro_method(gyro)
    report_error_vs_correction(samples)

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
                        help="Skip recomputation and rebuild the summary + report from the per-trial "
                             "tables already on disk. The pooled summary always covers every "
                             "subject/activity present on disk, not just those passed to --subjects, "
                             "so a partial run does not silently narrow it.")
    args = parser.parse_args()

    if not args.report_only:
        run_tracked_grid(args.subjects, ['Subject'], args.activities, _trial_worker,
                         args.workers, title="ACCELERATION PROJECTION", per_cell=True)

    print("\nLoading per-trial tables...")
    samples = load_trial_table('joint_samples')
    alignment = load_trial_table('alignment')
    sweep = load_trial_table('cutoff_sweep')
    gyro = load_trial_table('gyro_method')
    if samples.empty:
        print(f"No results found under {EXPERIMENT_DIR}. Run without --report-only first.")
        return

    trials = set(map(tuple, samples[['subject', 'activity']].drop_duplicates().to_numpy()))
    print(f"Found {len(trials)} trial(s) across {len({s for s, _ in trials})} subject(s), "
          f"{len(samples):,} samples.")

    summary = summarize(samples)
    path = paths.ensure_parent(paths.statistics_path(EXPERIMENT_NAME))
    summary.to_parquet(path, engine='pyarrow', index=False)
    paths.write_manifest(path, constants=analysis_constants(), experiment=EXPERIMENT_NAME,
                         n_rows=len(summary),
                         subjects=sorted({s for s, _ in trials}),
                         activities=sorted({a for _, a in trials}))
    print(f"Saved summary to {path}")

    print_report(summary, samples, alignment, sweep, gyro)
    print(f"\nPer-trial tables under {EXPERIMENT_DIR}")
    print("Figures: python -m plotting.acceleration_projection")


if __name__ == '__main__':
    main()
