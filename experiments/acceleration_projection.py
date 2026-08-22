"""
Does the rigid-body acceleration projection produce the acceleration it claims to? Measured
three ways, against three different references, none of which is a filter.

Every method in this repo that uses `project=True` feeds its filter an accelerometer signal that
was never measured: the sensor's reading rigid-body projected onto another point of the same
segment,

    a_p = a_sensor + alpha x r + omega x (omega x r)          (IMUTrace.project_acc)

with `omega` the gyro, `alpha` a finite difference of the gyro, and `r` a constant
sensor-to-target offset. This file asks whether that signal is real, and it asks it three times
because there are three references available and each one is blind to something the others see.

    python -m experiments.acceleration_projection --dataset alborno
    python -m experiments.acceleration_projection --dataset imove
    python -m experiments.acceleration_projection --dataset alborno --report-only

THE THREE QUESTIONS
===================

1. MARKER AGREEMENT — do the projected accelerometers agree with the markers?
   family='marker'. The estimate is a_p at the joint center; the truth is that joint center's
   world position differentiated twice plus gravity, rotated into the sensor's mocap
   orientation. This is the only family with an EXTERNAL reference, so it is the only one that
   can catch an error both segments make together.

   Its weakness is severe and quantified below: the reference is a second derivative of marker
   positions, so above ~5 Hz it is mostly differentiation noise, and the residual it reports is
   an UPPER BOUND on the projection error rather than a measurement of it.

2. CROSS-SEGMENT AGREEMENT — do the two projected accelerometers agree with each other?
   family='joint'. The parent and child sensors project to the SAME physical point, so their
   projected readings are one vector expressed in two frames:

       a_parent  =  R_parent^T R_child  a_child                                          (2)

   That is not a side observation, it is the measurement a relative filter actually consumes —
   the filter's accelerometer residual IS the violation of (2). So this family scores the
   projection in the currency the pipeline spends. It needs no differentiated markers, which is
   exactly the noise the marker family cannot escape: both sides are real accelerometers, so
   the floor here is sensor noise rather than mocap noise, roughly an order of magnitude lower.

   It is blind to a common-mode error. If both segments' offsets are wrong in a way that moves
   the shared point together, (2) still holds. That is what the marker family is for, and it is
   why quoting either alone would be a choice about which blindness to hide.

   The NORM channel of this family, | |a_parent| - |a_child| |, uses no mocap orientation at
   all — only the two offsets — so it survives on trials where the mocap window is too short to
   score anything else, and it is the part of the disagreement no orientation estimate can ever
   absorb. Seel's estimator (experiments/inertial_joint_center.py) fits the offsets by
   minimizing exactly this; here it is a diagnostic on offsets fitted from mocap instead.

3. SAME-SEGMENT AGREEMENT — do two accelerometers on ONE segment project to each other?
   family='segment', IMoVE only, which is the one dataset here carrying three sensors on each
   thigh and shank. Project sensor A onto sensor B's own location and compare with what B
   measured:

       a_A + alpha_A x r_AB + omega_A x (omega_A x r_AB)  =  C_AB a_B                    (3)

   This is the cleanest test of the projection physics available anywhere in the repository,
   because the truth side of (3) is a real accelerometer reading rather than a differentiated
   marker trajectory. No mocap enters the SIGNALS at all — only the geometry (r_AB, C_AB) — so
   the comparison is band-limited by the sensors and not by the reference, and it can be scored
   over the whole record instead of only inside the mocap window.

   Two targets are measured, and they answer different questions:
     target_kind='partner'  B's own site, ~50-150 mm away. Truth is B's reading.
     target_kind='joint'    both sensors projected to the same joint center, a 100-250 mm arm.
                            No truth; the metric is mutual agreement over a long lever, which
                            is the geometry the pipeline actually uses.

   What it inherits instead of mocap noise is mocap GEOMETRY, and on IMoVE that inheritance is
   total: all three placements on a segment are reconstructed from ONE marker cluster, each
   plate's origin shifted onto its own IMU by `imove_mocap._sensor_offsets`. So r_AB is exactly
   the difference of two fitted sensor offsets and C_AB is exactly the ratio of two fitted
   sensor-to-segment rotations — both constant by construction, neither an independent
   measurement. `segment_geometry` records that constancy rather than assuming it, and
   `fit_separation_inertial` closes the loop by re-solving r_AB from the IMUs alone: (3) is
   LINEAR in r_AB, so that is a 3-parameter least squares, and the gap between the inertial and
   mocap answers is the geometry error the projection is being blamed for.

   AND THE ANSWER IS NOT THE GEOMETRY, which is worth stating because it was this file's
   expectation and the numbers refused it. Re-solved from the IMUs, the separation moves by a
   median 26 mm and the residual barely moves with it (1.05 against 1.10 m/s^2). Nor is it the
   frame transform: fitting C_AB from the gyros instead of composing it from mocap changes it by
   ~1 deg and a few percent. What IS large is the thing a rigid segment cannot produce at all —
   two sensors 8-19 cm apart on one thigh or shank disagree about ANGULAR VELOCITY by 20-32% of
   its magnitude, and `gyro_sync_diagnostic` shows that shifting them in time does not fix it
   either. So this family's floor is the segment: skin-mounted sensors on a thigh are not on one
   rigid body at the precision the projection assumes. That is the most useful negative result
   here, because it bounds what ANY two-IMU-per-segment method can do, and only this family could
   have measured it.

ONE ABSTRACTION FOR ALL THREE
=============================
Every family reduces to the same object, `Comparison`: two acceleration signals that should be
equal, expressed in ONE COMMON FRAME, conditioned identically, plus the unprojected baseline of
the estimate. Every table in this file is generated from a list of Comparisons and does not know
which family it is looking at. That is deliberate — three parallel implementations of "RMS of a
difference" is three chances for the families to disagree for reasons that are about the code.

TRANSPORT HAPPENS BEFORE FILTERING. Where a family needs a frame change (families 2 and 3), it
is applied at full rate and the low-pass comes after. The other order does not commute: C(t)
varies, so C * lowpass(a) - lowpass(a') carries a spurious term in dC/dt that lowpass(C a - a')
does not. Human relative-segment motion is far below the 6 Hz cutoff so the term is small, but
it is avoidable for free and an avoidable artifact in a validation is not worth having.

THE UNPROJECTED BASELINE is in every family and is the point of the whole analysis. On its own,
"estimate agrees with truth to X" only establishes a floor; what makes the projection worth
doing is that X is smaller than the same number for the signal it REPLACES, against the same
truth on the same samples. Because the two share the truth, they share its noise, and the gap
between them survives whatever that noise is.

WHAT LIMITS EACH FAMILY, IN ONE PLACE
=====================================
    marker    mocap double-differentiation noise, flat above ~5 Hz. Dominates, and is why the
              cutoff is 6 Hz. Bounded, not removed: `reference_excess_rms` and `cutoff_sweep`.
    joint     the joint model — a fixed-centre ball joint. Soft tissue and a translating knee land
              here, as does any error in the two offsets that does NOT move the shared point. The
              cross-family check (`report_consistency`) shows this term is real and sizeable: the
              pair disagreement is ~1.4x what the two marker residuals predict, and re-predicting
              from the shared midpoint brings that to ~1.14, so most of the excess IS the joint
              model rather than the projection.
    segment   the SEGMENT, not the geometry and not the frame — measured, against this file's own
              expectation, and detailed under family 3 above.

MEASURED, NOT ASSUMED, AND TWO OF THEM CAME OUT THE OTHER WAY
============================================================
Two hypotheses in earlier versions of this file were wrong, and both are now reported with the
numbers that refused them rather than quietly dropped:

  * "The marker residual is inflated by shared reference noise, so it is an upper bound." The
    cross-family ratio says the opposite on this data — the two segments' marker residuals are
    close to independent, and the pair disagreement is LARGER than either. Section 7.
  * "The same-segment residual is mostly the mocap geometry it inherits." It is not; the inertial
    refit moves the separation 26 mm and the residual almost not at all. Section 8.

Anything in this docstring stated as a number was measured on the datasets named; anything stated
as a mechanism has a section that checks it.

VALID FRAMES ONLY, AND WHY THAT IS A CHANGE
===========================================
Outside `valid` the world trace holds a constant padded pose left where marker reconstruction
failed (WorldTrace.repair_reconstruction_glitches). A constant pose differentiates to zero, so
the mocap reference over those frames is exactly gravity while the IMU reads real motion — pure
fiction entering the residual as projection error. Al Borno trials are 42-58% valid, so this was
not a rounding correction. An earlier version of this file trimmed only the ends of the record
and scored everything in between; every marker-family number it produced was inflated.

Comparisons are therefore scored on contiguous RUNS of valid frames, each trimmed by TRIM_PERIODS
at both ends and low-passed run by run, never across a gap. The reference-free channels are
additionally scored over the whole record (`scope='full'`), which is where they are actually
available and where the mocap-free claim has to hold.

THE MISALIGNMENT FLOOR, AND WHY NO CHOICE OF FRAME REMOVES IT
=============================================================
Both signals in a comparison are brought into one frame with a rotation that was fitted
(assembly.align_world_to_imu, or the joint's mocap rotations). Residual misalignment turns into
an apparent error of roughly (angle) x |a|, and since |a| is dominated by gravity, 1 deg is
~0.17 m/s^2. Changing frames cannot help: rotating both signals by the same matrix leaves
|estimate - truth| exactly unchanged, and so does subtracting a common gravity vector. So the
floor is MEASURED instead, per comparison, by fitting the single constant rotation that best
maps estimate onto truth (Kabsch, `residual_alignment`) and reporting the residual either side
of it. Nothing downstream is rotation-corrected; the fit is reported so the floor is visible.

OUTPUTS, all under results/experiments/acceleration_projection/<dataset>/<subject>/<trial>/

    agreement_samples.parquet  per-sample metrics, EVERY family, strided by SAMPLE_STRIDE
    agreement_stats.parquet    per-comparison scalars: rms, Kabsch angle, R^2, arms, geometry
    traces.parquet             full-rate components of one example comparison per family
    spectra.parquet            Welch PSD of all three signals, UNFILTERED
    cutoff_sweep.parquet       residual vs low-pass cutoff, against the truth's own in-band noise
    gyro_method.parquet        residual under each gyro-derivative scheme, per family

plus the pooled quantile summary at
results/statistics/acceleration_projection_<dataset>_statistics.parquet, which is what the
console report and every figure caption quote.

Figures: python -m plotting.acceleration_projection --dataset <name>
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
from scipy.signal import butter, filtfilt, welch
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon

import paths
from experiments.experiment_utils import (EXPECTED_GRAVITY, _compute_perfect_joint_acc,
                                         pipeline_constants, run_tracked_grid)
from experiments.global_assumptions import (DATASETS, DatasetSpec, canonical_joint,
                                            enumerate_trials, get_dataset, subjects_of)
from experiments.joint_center import MIN_FIT_FRAMES, _load_spec_plates, joint_offsets, select_trials
from src.toolchest.finite_difference_utils import (backward_difference, central_difference,
                                                  forward_difference, polynomial_fit_derivative)
from src.toolchest.gyro_utils import calculate_best_fit_rotation
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import UnderdeterminedJointCenter, WorldTrace

EXPERIMENT_NAME = "acceleration_projection"
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Low-pass ---------------------------------------------------------------------------
# 6 Hz, measured rather than conventional: the mocap reference and the IMU signals have matching
# spectra below ~5 Hz and diverge above it, where the reference flattens onto its differentiation
# noise floor. Filtering at 15 Hz — which looks conservative, and was this file's first choice —
# admits a band in which the reference carries ~10x the power of the signal, and the resulting
# residual is then almost entirely reference noise. `cutoff_sweep` exists so that is checkable
# rather than asserted.
#
# The cost is real and worth stating: heel-strike transients do carry genuine content above 6 Hz,
# so the marker family compares over the band where mocap can adjudicate at all, not over the
# sensor's full bandwidth. The joint and segment families have no such excuse — their truth is an
# accelerometer with the same bandwidth as the estimate — and the same cutoff is applied to them
# anyway, so that one number is comparable across all three. What the cutoff costs THEM is
# reported separately by the sweep, whose 25 Hz row is a fair measurement for those two families
# and a meaningless one for the marker family.
LOWPASS_CUTOFF_HZ = 6.0
LOWPASS_ORDER = 4

# Cutoffs the sweep reports, spanning the plausible range so the choice above can be seen in
# context. Anything at or above a trial's Nyquist is skipped rather than clipped — IMoVE records
# 17 of its sensors at 40 Hz, where 25 Hz is not a cutoff.
CUTOFF_SWEEP_HZ = (4.0, 6.0, 10.0, 15.0, 25.0)

# Band used to measure the mocap reference's white noise floor: high enough that the true signal
# is negligible there (the IMU spectra are ~30x lower), low enough to stay clear of the
# anti-alias rolloff at Nyquist. NaN on a trial whose Nyquist is below it.
NOISE_BAND_HZ = (25.0, 45.0)

# Samples discarded at each end of every valid run, in PERIODS of LOWPASS_CUTOFF_HZ — 0.5 s at
# 6 Hz. Three edge artifacts stack up there and none of them are projection error: filtfilt's
# transient, central_difference's `extend` padding (twice, since the mocap reference is a second
# derivative), and polynomial_fit_derivative's partial windows.
#
# Expressed in periods rather than seconds because a fixed 1.0 s trim is a different fraction of
# the filter's memory at 40 Hz than at 100 Hz, and because the biplane trials are 0.5 s long
# end to end — a constant in seconds silently decides which datasets can be analysed at all.
TRIM_PERIODS = 3.0

# What a run must have LEFT after trimming to be worth scoring, in seconds. A run shorter than
# this contributes nothing but its own edges.
MIN_KEEP_S = 1.0

# Storage decimation for agreement_samples ONLY. Filtering happens BEFORE decimation, so this is
# a subsample of an already band-limited signal rather than a naive downsample: at 100 Hz, stride
# 10 leaves 10 Hz, whose Nyquist (5 Hz) is just under LOWPASS_CUTOFF_HZ, so the primary analysis
# band is preserved and only the sweep's wider cutoffs would alias — which is why the sweep
# recomputes from the unfiltered signals instead of reading this table. Uniform subsampling is
# unbiased for the marginal distributions every panel and every quoted quantile is computed from.
#
# 10 rather than the 3 this file used when it had one family: three families over 18 IMoVE joint
# pairs and 12 same-segment pairs is ~140 comparisons per trial against the old 14, so stride 3
# would put this table into the tens of gigabytes for quantiles that are identical at stride 10.
# Every scalar in agreement_stats is computed on the FULL conditioned record regardless, and the
# full-rate signals survive in `traces`.
SAMPLE_STRIDE = 10

# Welch parameters for the spectral panel. 1024 samples at 100 Hz is a ~10 s window: long enough
# for ~0.1 Hz resolution at the low end, short enough that a long trial averages over many
# segments and the spectrum is smooth without extra smoothing. Clamped to the run length.
WELCH_NPERSEG = 1024

# --- Example traces ---------------------------------------------------------------------
# `traces` stores FULL-RATE components, so it is the one table whose size is set by trial length
# rather than by row count, and it is stored for at most one comparison per family per trial.
# Storing every detailed comparison was 160 MB per Al Borno trial — 2 GB for one dataset — for a
# panel that draws four seconds of one signal.
#
# The window is the most dynamic TRACE_MAX_S inside one run, chosen by `_dynamic_slice` rather than
# by hand, so the panel shows the correction doing visible work instead of whatever happened at
# the start of the recording. 20 s is ~15 gait cycles: enough that the plotting layer can pick its
# own 4 s window out of it without this table committing to one.
TRACE_MAX_S = 20.0

# Which comparison each family contributes, when it has a choice. A knee is the useful
# illustration for the joint families — the offset is long enough that the correction is large, and
# the segment swings fast enough that the omega x (omega x r) term dominates at mid-swing — and
# High-Low is the widest same-segment pair, so its correction is the largest that family offers.
# Only a PREFERENCE: a dataset or trial without them falls back to whatever it has.
EXAMPLE_GROUP_HINT = 'Knee'
EXAMPLE_SEGMENT_VARIANT = 'High-Low'

# --- Gyro-derivative schemes -----------------------------------------------------------
# The projection's tangential term needs alpha = d(omega)/dt, and no sensor measures it, so it has
# to be differentiated out of the gyro. The choice is not free: differentiation amplifies
# high-frequency content, so a scheme trades noise against bandwidth and lag.
#
#   backward     (w[t] - w[t-1]) / dt. project_acc's default, and the only one that reads no
#                future sample, so the only one an online pipeline can use. Noisiest, and meant
#                to be paired with a low-pass on the projected acceleration.
#   polyfit      sliding 10-sample 2nd-order fit differentiated analytically, overlapping windows
#                averaged — a smoothed Savitzky-Golay derivative. Was project_acc's default, so
#                it is what every projected result in this repo used up to that change.
#   central      (w[i+1] - w[i-1]) / 2dt. Centered, so no lag; no smoothing.
#   first_order  forward difference. Included as the case that SHOULD lose: biased by half a
#                sample, and half a sample of lag on this signal is error, not lag.
#
# THE OLD CAVEAT ON THIS TABLE, AND WHAT FIXES IT. Scored against the mocap reference after a
# 6 Hz low-pass, the schemes differ almost entirely above the cutoff: filtering removes ~94% of
# the gap between polyfit and central, and below 6 Hz their gains agree to under 1%. So that table
# compared them in the one band where they nearly agree, while joint-angle RMSE on unfiltered
# projected acc ranked them the other way round (polyfit 6.96 deg vs central 9.56).
#
# The cross-segment family is the fix rather than another caveat: it scores the same schemes
# against a truth that is an accelerometer, so there is no reference noise forcing the cutoff
# down, and `gyro_method` reports every family. A scheme that wins on mocap and loses on
# sensor-to-sensor agreement was winning on the reference's noise.
#
# The schemes still do not see the same data, and that part no reference fixes: polyfit's kernel
# spans +-9 samples, so it uses 90 ms of FUTURE gyro against central's one sample. A fully causal
# Savitzky-Golay (same fit, evaluated at the trailing edge) scores 16.87 deg on joint angles, far
# worse than either, so polyfit's advantage rides substantially on lookahead rather than on being
# a better estimator.
GYRO_METHODS = ('backward', 'polyfit', 'central', 'first_order')
PRIMARY_GYRO_METHOD = 'backward'  # must stay project_acc's own default; see above

# Family names, in the order the report walks them. Also the order every figure sorts by.
FAMILIES = ('marker', 'joint', 'segment')
FAMILY_LABELS = {
    'marker': 'projected vs markers',
    'joint': 'parent vs child, both projected to the joint centre',
    'segment': 'two sensors on one segment, projected onto each other',
}

# Scopes a comparison can be scored on.
#   mocap  contiguous runs of frames where every plate involved is valid, trimmed. The only scope
#          the marker family and any transported channel can use.
#   full   the whole record minus its two ends. Available to a comparison whose frame transform is
#          a CONSTANT (the segment family) or whose channel needs no frame at all (the norm
#          channel of the joint family), which is where the mocap-free claim has to be tested.
SCOPES = ('mocap', 'full')

# What a comparison can actually compare, which is NOT the same question as which samples it is
# scored on.
#   vector  both signals live in one common frame, so the full 3-vector residual, its direction and
#           its magnitude are all defined.
#   norm    no common frame exists, so only the MAGNITUDES can be compared. `err_proj` is then
#           | |a_est| - |a_truth| |, `dnorm_proj` is the signed version of the same number, and
#           anything angular is NaN rather than a degenerate zero (see `_norm_only_comparison`).
# Carried as its own column instead of being inferred from `scope`, because the two are
# independent: a frame-free channel could be scored inside the mocap window, and conflating them
# is how a reader ends up comparing a scalar magnitude gap against a 3-vector residual in one
# column and reading the difference as a finding.
CHANNELS = ('vector', 'norm')

ROLES = ('parent', 'child')

# Display names ending in one of these are read as "same segment, different placement", which is
# how the same-segment family finds its pairs. Keyed on the DISPLAY name rather than the plate
# name so it reads off DatasetSpec.segment_sensor's public contract instead of IMoVE's private
# '<SEGMENT>_<H|M|L>' convention; a future dataset that names placements this way is picked up
# with no change here.
PLACEMENT_TOKENS = ('High', 'Mid', 'Low')

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# metric -> unit, for the summary table and axis labels.
METRIC_UNITS = {
    'err_proj': 'm/s^2',
    'err_raw': 'm/s^2',
    'err_proj_alt': 'm/s^2',
    'err_proj_nofilt': 'm/s^2',
    'ang_proj': 'deg',
    'ang_raw': 'deg',
    'dnorm_proj': 'm/s^2',
    'dnorm_raw': 'm/s^2',
    'absdnorm_proj': 'm/s^2',
    'absdnorm_raw': 'm/s^2',
    'corr_norm': 'm/s^2',
    'gyro_norm': 'rad/s',
    'gyro_mismatch': 'rad/s',
    'truth_lin_norm': 'm/s^2',
    'jc_residual_norm': 'm',
}

# The metrics the pooled summary tabulates, for every family. One list for all three is the point
# of the Comparison abstraction: the headline table is a pivot over `family`, not three tables.
SUMMARY_METRICS = ('err_proj', 'err_raw', 'ang_proj', 'ang_raw', 'absdnorm_proj', 'absdnorm_raw',
                   'dnorm_proj', 'corr_norm', 'err_proj_alt', 'err_proj_nofilt')

TRIAL_TABLES = ('agreement_samples', 'agreement_stats', 'traces', 'spectra', 'cutoff_sweep',
                'gyro_method')

# Tables whose cost scales with the number of comparisons rather than with the trial length, and
# which are therefore restricted to the spec's primary joints. IMoVE has 18 joint pairs against 6
# anatomical ones; a 5-cutoff sweep and a 4-method gyro table over all 18 triples the runtime for
# rows no report reads.
DETAILED_TABLES = ('traces', 'spectra', 'cutoff_sweep', 'gyro_method')


def analysis_constants(dataset: str) -> Dict[str, object]:
    """Pipeline constants plus this analysis's own choices, for provenance manifests."""
    return {
        **pipeline_constants(),
        'dataset': dataset,
        'primary_gyro_method': PRIMARY_GYRO_METHOD,
        'gyro_methods': list(GYRO_METHODS),
        'lowpass_cutoff_hz': LOWPASS_CUTOFF_HZ,
        'lowpass_order': LOWPASS_ORDER,
        'cutoff_sweep_hz': list(CUTOFF_SWEEP_HZ),
        'noise_band_hz': list(NOISE_BAND_HZ),
        'trim_periods': TRIM_PERIODS,
        'min_keep_s': MIN_KEEP_S,
        'sample_stride': SAMPLE_STRIDE,
        'welch_nperseg': WELCH_NPERSEG,
        'min_fit_frames': MIN_FIT_FRAMES,
        'families': list(FAMILIES),
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
    """Concatenates one per-trial table across trials, adding `subject`, `trial` and `dataset`.

    Missing trials are skipped silently — a partial run (`--subjects 06`) is a legitimate state,
    and the caller reports what it found.

    `columns` is pushed down into the parquet read rather than selected afterwards, which matters
    for agreement_samples: the full table across every trial is tens of millions of rows, and a
    panel that needs two of its columns should not pay for all thirty. The label columns are read
    back as categoricals for the same reason — as objects they cost more than every float column
    combined.
    """
    if table not in TRIAL_TABLES:
        raise ValueError(f"Unknown table '{table}'; expected one of {TRIAL_TABLES}")
    row_keys = enumerate_trials(dataset) if row_keys is None else row_keys
    if columns is not None:
        columns = list(dict.fromkeys(columns))  # de-duplicate, preserve the caller's order
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
                  'channel', 'gyro_method', 'subject', 'trial', 'dataset'):
        if label in df.columns:
            df[label] = df[label].astype('category')
    return df

# ==============================================================================
# Windows: which samples a comparison is scored on
# ==============================================================================

@dataclass(frozen=True)
class Window:
    """The samples one comparison is scored on, grouped into contiguous runs.

    `runs` are half-open [start, stop) intervals into the FULL record, before trimming; `index`
    is the samples that survive the trim, in order, and `run_id` says which run each came from.
    Both are kept because they answer different questions: the filter needs the untrimmed run so
    its transient lands in the discarded margin, and every table needs the trimmed index.

    Filtering is per run and never across a gap. Two sides of an invalid stretch can be minutes
    apart; a filter run over the join manufactures a transition that was never measured, and it
    is the same reason `joint_dof.joint_series` filters run by run.
    """
    scope: str
    runs: Tuple[Tuple[int, int], ...]
    trim: int
    index: np.ndarray
    run_id: np.ndarray
    fs: float
    n_record: int
    n_mask: int

    def __len__(self) -> int:
        return len(self.index)

    @property
    def nyquist(self) -> float:
        return 0.5 * self.fs

    @property
    def duration_s(self) -> float:
        return len(self.index) / self.fs


def valid_runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Contiguous [start, stop) runs of True at least `min_len` long."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def build_window(mask: np.ndarray, fs: float, scope: str,
                 cutoff: float = LOWPASS_CUTOFF_HZ) -> Optional[Window]:
    """The Window for `mask`, or None when nothing survives the run-length floor.

    None is the honest answer for a joint whose mocap never held long enough to be a motion, and
    every caller treats it as "this comparison is not available on this trial" rather than as an
    error. That distinction matters on the biplane trials, where the fluoroscopic window is ~0.5 s
    and the mocap scope legitimately yields nothing while the full scope yields the whole record.
    """
    trim = int(np.ceil(TRIM_PERIODS * fs / cutoff))
    # filtfilt's default padlen is 3 * max(len(a), len(b)) and it refuses a shorter signal, so a
    # run must clear that as well as leaving MIN_KEEP_S behind. Stated in terms of the actual
    # filter order rather than as a magic number so changing LOWPASS_ORDER cannot break it.
    min_filtfilt = 3 * (2 * LOWPASS_ORDER + 1) + 1
    min_run = max(2 * trim + int(round(MIN_KEEP_S * fs)), min_filtfilt)
    runs = valid_runs(mask, min_run)
    if not runs:
        return None
    index = np.concatenate([np.arange(start + trim, stop - trim) for start, stop in runs])
    run_id = np.concatenate([np.full((stop - start) - 2 * trim, i, dtype=np.int32)
                             for i, (start, stop) in enumerate(runs)])
    return Window(scope=scope, runs=tuple(runs), trim=trim, index=index, run_id=run_id, fs=fs,
                  n_record=len(mask), n_mask=int(np.asarray(mask, dtype=bool).sum()))


def lowpass(signal: np.ndarray, fs: float, cutoff: float = LOWPASS_CUTOFF_HZ,
            order: int = LOWPASS_ORDER) -> np.ndarray:
    """Zero-lag Butterworth low-pass along axis 0. Pass-through at or above Nyquist.

    filtfilt rather than lfilter because a phase shift between the two signals of a comparison
    would show up as error: at 100 Hz, one sample of lag on a signal changing at 10 m/s^3 is
    0.1 m/s^2 of pure artifact. Nothing here is causal or online, so there is no reason to accept
    that.

    The Nyquist pass-through is not a convenience. IMoVE records 17 of its 24 sensors at 40 Hz, so
    the sweep's 25 Hz row asks for a cutoff above Nyquist on most of that dataset; butter() would
    raise, and clipping to Nyquist would silently report a 20 Hz result in a row labelled 25.
    Returning the signal unfiltered is what "no filtering above this rate" means, and the sweep
    drops those rows rather than quoting them (see `cutoff_rows`).
    """
    if cutoff >= 0.5 * fs:
        return np.asarray(signal, dtype=float)
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, np.asarray(signal, dtype=float), axis=0)


def condition(signal: np.ndarray, window: Window,
              cutoff: Optional[float] = LOWPASS_CUTOFF_HZ) -> np.ndarray:
    """Low-pass `signal` run by run and return the trimmed interior, in `window.index` order.

    `cutoff=None` skips the filter and returns the same samples unfiltered, which is how the
    spectral panel and the cutoff sweep get an untouched copy on exactly the samples every
    filtered metric was computed on. Works for (N,) and (N, 3) alike.
    """
    if cutoff is None:
        return np.asarray(signal)[window.index]
    parts = []
    for start, stop in window.runs:
        block = lowpass(signal[start:stop], window.fs, cutoff)
        parts.append(block[window.trim:(stop - start) - window.trim])
    return np.concatenate(parts)

# ==============================================================================
# Signal helpers
# ==============================================================================

def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-sample angle between two (N, 3) vector series, in degrees.

    The dot product is clipped before arccos: normalized dot products land a few ulp outside
    [-1, 1] for near-parallel vectors, and these signals are near-parallel most of the time (both
    dominated by the same gravity vector), so the unclipped version returns NaN on real data
    rather than as a pathological case.
    """
    scale = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    # NaN where either vector is zero-length, rather than a divide warning and an arbitrary value.
    # That case is not hypothetical on this data: a norm-channel comparison carries a zero on two
    # of its three axes by construction, and a genuine acceleration vector passes through zero
    # whenever free-fall and the sensor's own rotation happen to cancel.
    cosine = np.divide(np.sum(a * b, axis=1), scale, out=np.full(len(scale), np.nan),
                       where=scale > 0)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rms(residual: np.ndarray) -> float:
    """Root-mean-square vector magnitude of an (N, 3) residual."""
    return float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))


def norm_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a| - |b| per sample, SIGNED.

    The one channel of any comparison that needs no common frame, so the only one available with
    no orientation reference at all. Signed rather than absolute because the sign is informative
    and cannot be recovered later: a projection that systematically overshoots reads positive
    here at every sample, which a magnitude cannot distinguish from symmetric noise. The absolute
    value is stored beside it for the quantile tables, where a signed median of a symmetric
    distribution is ~0 and says nothing about size.
    """
    return np.linalg.norm(a, axis=1) - np.linalg.norm(b, axis=1)


def residual_alignment(estimate: np.ndarray, reference: np.ndarray
                       ) -> Tuple[np.ndarray, float, float, float]:
    """The single constant rotation that best maps `estimate` onto `reference`, and the rms
    residual before and after removing it. Returns (R, angle_deg, rms_before, rms_after).

    This is the misalignment floor made measurable (see the module docstring). Solved in closed
    form by Kabsch: R = U diag(1, 1, det(U V^T)) V^T from the SVD of sum_i reference_i estimate_i^T,
    with the determinant term keeping the result a rotation rather than admitting a reflection.

    The fit is dominated by gravity — 9.81 m/s^2 against linear accelerations of a few — so what
    it recovers is overwhelmingly the static misalignment between the two frames rather than any
    dynamic projection error, which is what makes `rms_after` interpretable as the projection's
    own residual. It is reported alongside `rms_before` and never substituted for it: the
    per-sample error columns are left unrotated so that no metric in any figure depends on a
    fitted correction.
    """
    H = reference.T @ estimate
    U, _, Vt = np.linalg.svd(H)
    R = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
    angle_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))
    return R, angle_deg, rms(estimate - reference), rms(estimate @ R.T - reference)


def skew(vectors: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 3, 3) skew-symmetric matrices, so skew(v) @ w == cross(v, w)."""
    v = np.asarray(vectors, dtype=float)
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def lever_operator(gyro: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """(N, 3, 3) matrices K with K @ r == alpha x r + omega x (omega x r).

    The projection is LINEAR IN THE OFFSET, which is the whole reason `fit_separation_inertial`
    is a least squares rather than an optimization: a_p = a_sensor + K r exactly, with
    K = [alpha]x + [omega]x^2. Building it explicitly costs 9 floats per sample and buys the
    inertial geometry fit, the sensitivity |da| <= |K| |dr|, and a second implementation of
    project_acc's arithmetic that the tests diff against the first.
    """
    omega_skew = skew(gyro)
    return skew(alpha) + omega_skew @ omega_skew


def gyro_derivative(trace, method: str = PRIMARY_GYRO_METHOD) -> np.ndarray:
    """alpha = d(omega)/dt under one of GYRO_METHODS.

    Duplicates IMUTrace._finite_difference_gyros's dispatch rather than calling the private
    method. The alternative is reaching into IMUTrace for a name it does not export, and the
    mapping is four lines; if it drifts, `TestAccelerationProjection` compares this against
    project_acc's own output and fails.
    """
    if method == 'backward':
        return backward_difference(trace.gyro, trace.timestamps)
    if method == 'central':
        return central_difference(trace.gyro, trace.timestamps)
    if method == 'first_order':
        return forward_difference(trace.gyro, trace.timestamps)
    if method == 'polyfit':
        return polynomial_fit_derivative(trace.gyro, trace.timestamps, order=2)
    raise ValueError(f"Unknown gyro derivative method '{method}'; expected one of {GYRO_METHODS}")


def chordal_mean(rotations: np.ndarray) -> np.ndarray:
    """The chordal (Frobenius) mean rotation of an (N, 3, 3) stack, as a (3, 3) matrix.

    scipy's Rotation.mean, reached through the matrix form because that is what the world traces
    hold and round-tripping through quaternions per sample is the expensive part of this file's
    geometry step when it is not needed.
    """
    return Rotation.from_matrix(np.asarray(rotations, dtype=np.float64)).mean().as_matrix()


def rotation_spread_deg(rotations: np.ndarray, mean: np.ndarray) -> Tuple[float, float]:
    """(rms, max) angular deviation of a rotation stack from `mean`, in degrees.

    How constant a supposedly-constant frame transform actually is. On IMoVE the same-segment
    transform is constant to float precision by construction (one marker cluster, see the module
    docstring), so this reading near zero CONFIRMS that structure rather than measuring segment
    rigidity — and a reading that is not near zero would mean the two plates came from different
    clusters and the assumption behind `segment_geometry` had quietly stopped holding.
    """
    deviation = np.linalg.norm(
        Rotation.from_matrix(np.einsum('ij,njk->nik', mean.T, rotations)).as_rotvec(), axis=1)
    return float(np.degrees(np.sqrt(np.mean(deviation ** 2)))), float(np.degrees(deviation.max()))

# ==============================================================================
# The Comparison: two signals that should be equal
# ==============================================================================

@dataclass
class Comparison:
    """Two acceleration signals that SHOULD be equal, in one common frame, conditioned alike.

    This is the only object the table builders below know about. Every family produces these and
    nothing downstream asks which family it is looking at, which is what keeps the three questions
    scored by one implementation of "RMS of a difference" instead of three.

    `estimate` is the projected signal — the quantity a filter would consume. `estimate_raw` is
    the same sensor's UNPROJECTED reading, transported into the same frame by the same rotation,
    which is the baseline the whole analysis turns on. `truth` is whatever this family compares
    against: differentiated markers, the other segment's projection, or the partner sensor's own
    reading.

    All three are filtered and trimmed. `unfiltered` holds the same samples untouched, for the
    spectra and the cutoff sweep — the two consumers that ask what happens ABOVE the analysis
    cutoff, which filtering here would erase.

    `truth_alt` is a second, defensible reference where one exists, scored as `err_proj_alt`. Only
    the marker family has one (the midpoint of the two segments' implied joint centres, which is
    what the filter's acc oracle uses); it is carried rather than substituted because quoting only
    the midpoint blames the projection for the joint model's parent/child disagreement, and
    quoting only the segment's own centre understates the error the oracle path actually sees.

    `detailed` gates the tables whose cost scales with the number of comparisons rather than with
    trial length. `scalars` and `extras` carry the family-specific columns — one scalar per
    comparison and one array per sample respectively — so that adding a diagnostic to one family
    does not change any shared signature.
    """
    family: str
    group: str
    variant: str
    source: str
    target: str
    target_kind: str
    window: Window
    timestamps: np.ndarray
    estimate: np.ndarray
    estimate_raw: np.ndarray
    truth: np.ndarray
    by_method: Dict[str, np.ndarray]
    unfiltered: Dict[str, np.ndarray]
    # The TRUTH under each gyro-derivative scheme, where the truth is itself a projection. Absent
    # (None) when the truth does not depend on the derivative — differentiated markers, or a
    # partner sensor's raw reading — in which case `gyro_rows` holds it fixed.
    #
    # This matters and is easy to get wrong: in the cross-segment family both sides are projections,
    # so scoring the parent under 'polyfit' against a child fixed at 'backward' measures a
    # configuration nothing would ever run. Varying both together is what makes that row a
    # comparison of schemes rather than of mismatched pairs.
    truth_by_method: Optional[Dict[str, np.ndarray]] = None
    truth_alt: Optional[np.ndarray] = None
    extras: Dict[str, np.ndarray] = field(default_factory=dict)
    scalars: Dict[str, float] = field(default_factory=dict)
    detailed: bool = False
    channel: str = 'vector'

    @property
    def scope(self) -> str:
        return self.window.scope

    @property
    def key(self) -> Dict[str, object]:
        """The label columns identifying this comparison in every table."""
        return {'family': self.family, 'group': self.group, 'variant': self.variant,
                'source': self.source, 'target': self.target,
                'target_kind': self.target_kind, 'scope': self.scope, 'channel': self.channel}


def _finite(*arrays: np.ndarray) -> bool:
    """Whether every array is finite. A comparison with a NaN in it is dropped rather than
    written: one NaN sample poisons an rms and a Kabsch fit alike, and the causes seen here
    (a degenerate offset fit, a zero-length acceleration vector at a synthetic sample) are all
    conditions where the whole comparison is meaningless rather than one sample of it."""
    return all(np.all(np.isfinite(a)) for a in arrays)

# ==============================================================================
# The mocap reference (family 1's truth)
# ==============================================================================

def joint_center_reference_acc(plate: PlateTrial, local_offset: np.ndarray,
                              gravity: Optional[np.ndarray] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Mocap truth at the point `local_offset` on this segment, as
    (specific force in the segment's body frame, world-frame linear acceleration).

    Gravity is left IN the first return, and it has to be: gravity is uniform over a rigid body,
    so it passes through the projection equation untouched, which is exactly why that equation is
    valid for specific force and not just for kinematic acceleration. The filter also consumes the
    total vector, so the total vector is what should be validated.

    The second return is the same quantity `global_assumptions` calls `linacc`: gravity removed in
    the WORLD frame, which is where it can be removed as a constant. It comes from here rather
    than being reconstructed later because this is the only place the world-frame acceleration
    exists — once rotated into the body frame, gravity is smeared across all three axes by the
    segment's own orientation and cannot be subtracted back out.

    The segment's OWN implied joint center, not the midpoint of the two segments' estimates: this
    is the point `project_acc(local_offset)` is aiming at, so pairing them isolates the projection
    from the joint model's parent/child disagreement. The midpoint version the filter's oracle uses
    comes from `_compute_perfect_joint_acc` and is carried as `truth_alt`.

    Deliberately mirrors `_compute_perfect_joint_acc`'s construction step for step (offset the
    positions, differentiate twice in the world frame, add gravity, rotate into the body frame) so
    that the only difference between the two references is the reference POINT.
    """
    gravity = EXPECTED_GRAVITY if gravity is None else gravity
    rotations = plate.world_trace.rotations
    jc_world = plate.world_trace.positions + np.einsum('nij,j->ni', rotations, local_offset)
    # Rotations are unused by finite_difference_world_frame_accelerations; the segment's own are
    # passed through purely to satisfy the WorldTrace constructor.
    jc_trace = WorldTrace(plate.world_trace.timestamps, jc_world, rotations)
    world_acc = jc_trace.finite_difference_world_frame_accelerations(acc_from_gravity=gravity)
    return np.einsum('nji,nj->ni', rotations, world_acc), world_acc - gravity

# ==============================================================================
# Per-trial context: plates, windows, and a projection cache
# ==============================================================================

class TrialContext:
    """One trial's plates plus everything the three families share.

    The projection cache is the reason this is a class. A trial's marker family and joint family
    project the SAME sensor to the SAME offset — the marker family scores it against mocap, the
    joint family against the other segment — and each projection under each of four gyro schemes
    is the most expensive arithmetic in the file. Keyed on the sensor and the rounded offset, so
    the two families cost one projection between them rather than two.

    Rounding the offset to a micron for the key is safe and deliberate: offsets come from one
    deterministic least squares per joint, so two requests for "the same" offset are bit-identical
    and the rounding only guards against a caller that reconstructs the vector. A micron of offset
    is ~1e-5 m/s^2 of projected acceleration at these rotation rates, four orders below the
    smallest residual anything here reports.
    """

    def __init__(self, plates: Dict[str, PlateTrial], spec: DatasetSpec, dataset: str,
                 subject: str, trial: str):
        self.plates = plates
        self.spec = spec
        self.dataset = dataset
        self.subject = subject
        self.trial = trial
        self.n = min(len(plate) for plate in plates.values())
        self.fs = self._sample_rate()
        self.offsets = joint_offsets(plates, spec, min_frames=MIN_FIT_FRAMES)
        self._projection_cache: Dict[Tuple[str, str, Tuple[int, ...]], np.ndarray] = {}
        self._window_cache: Dict[Tuple[str, ...], Optional[Window]] = {}

    def _sample_rate(self) -> float:
        """The trial's rate, from the pelvis plate where there is one.

        Taken from the median timestamp difference over the whole record rather than from a run: a
        run may be short enough that its own diff is dominated by timestamp quantization, and the
        grid is uniform by construction — every reader puts a trial on one common grid before it is
        cached. IMoVE mixes 40 Hz and 100 Hz BETWEEN trials, never within one.
        """
        reference = self.plates.get(self.spec.pelvis_sensor) or next(iter(self.plates.values()))
        return float(1.0 / np.median(np.diff(reference.imu_trace.timestamps[:self.n])))

    # --- windows ---------------------------------------------------------------------
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
        """The whole record as one run. Available to any comparison whose mocap input is a
        constant, which is where the reference-free claim has to be tested."""
        if ('full',) not in self._window_cache:
            self._window_cache[('full',)] = build_window(np.ones(self.n, dtype=bool), self.fs,
                                                         'full')
        return self._window_cache[('full',)]

    # --- projections -----------------------------------------------------------------
    def projected(self, sensor: str, offset: np.ndarray,
                  method: str = PRIMARY_GYRO_METHOD) -> np.ndarray:
        """a_sensor projected to `offset`, at FULL RATE and unfiltered, memoized."""
        key = (sensor, method, tuple(np.round(np.asarray(offset, dtype=float), 6)))
        if key not in self._projection_cache:
            trace = self.plates[sensor].imu_trace
            self._projection_cache[key] = trace.project_acc(np.asarray(offset, dtype=float),
                                                            method).acc[:self.n]
        return self._projection_cache[key]

    def all_methods(self, sensor: str, offset: np.ndarray) -> Dict[str, np.ndarray]:
        """The same projection under every gyro-derivative scheme, primary first."""
        return {method: self.projected(sensor, offset, method) for method in
                (PRIMARY_GYRO_METHOD, *(m for m in GYRO_METHODS if m != PRIMARY_GYRO_METHOD))}

    def acc(self, sensor: str) -> np.ndarray:
        return self.plates[sensor].imu_trace.acc[:self.n]

    def gyro(self, sensor: str) -> np.ndarray:
        return self.plates[sensor].imu_trace.gyro[:self.n]

    def rotations(self, sensor: str) -> np.ndarray:
        """World-from-body mocap rotations, float64. Meaningless outside `valid`."""
        return np.asarray(self.plates[sensor].world_trace.rotations[:self.n], dtype=np.float64)

    def positions(self, sensor: str) -> np.ndarray:
        return np.asarray(self.plates[sensor].world_trace.positions[:self.n], dtype=np.float64)

    def timestamps(self, sensor: str) -> np.ndarray:
        return np.asarray(self.plates[sensor].imu_trace.timestamps[:self.n], dtype=float)

    def relative_rotation(self, parent: str, child: str) -> np.ndarray:
        """C(t) = R_parent^T R_child, which carries a vector from the CHILD body frame into the
        PARENT body frame. Getting this transposed is the one error in this file that produces a
        plausible-looking number instead of a crash, which is why it has a name and a test."""
        return np.einsum('nji,njk->nik', self.rotations(parent), self.rotations(child))

# ==============================================================================
# Family 1: marker agreement
# ==============================================================================

def marker_comparisons(context: TrialContext, joint: str, detailed: bool) -> List[Comparison]:
    """The projected acceleration at the joint centre against differentiated markers, per segment.

    Scored on the mocap scope only, necessarily: the truth IS the mocap, so there is no version of
    this comparison that outlives the valid window.
    """
    spec = context.spec
    parent_sensor, child_sensor = spec.joints[joint]
    offsets = context.offsets.get(joint)
    window = context.mocap_window(parent_sensor, child_sensor)
    if offsets is None or window is None:
        return []

    parent_plate, child_plate = context.plates[parent_sensor], context.plates[child_sensor]
    try:
        _, _, jc_residual = parent_plate.world_trace.get_joint_center(
            child_plate.world_trace, min_frames=MIN_FIT_FRAMES)
        shared = dict(zip(ROLES, _compute_perfect_joint_acc(parent_plate, child_plate)))
    except UnderdeterminedJointCenter:
        return []
    jc_residual_norm = np.linalg.norm(jc_residual[:context.n], axis=1)
    sensors = {'parent': parent_sensor, 'child': child_sensor}

    out = []
    for role in ROLES:
        sensor = sensors[role]
        offset = offsets[role]
        by_method = context.all_methods(sensor, offset)
        reference, reference_linacc = joint_center_reference_acc(context.plates[sensor], offset)
        raw = context.acc(sensor)

        estimate = condition(by_method[PRIMARY_GYRO_METHOD], window)
        truth = condition(reference[:context.n], window)
        comparison = Comparison(
            family='marker', group=joint, variant=role, source=sensor, target='markers',
            target_kind='markers', window=window,
            timestamps=condition(context.timestamps(sensor), window, cutoff=None),
            estimate=estimate, estimate_raw=condition(raw, window), truth=truth,
            truth_alt=condition(shared[role][:context.n], window),
            by_method={m: condition(a, window) for m, a in by_method.items()},
            unfiltered={'estimate': by_method[PRIMARY_GYRO_METHOD][window.index],
                        'estimate_raw': raw[window.index],
                        'truth': reference[:context.n][window.index]},
            extras={
                # The size of the TRUE linear acceleration at the joint centre, gravity removed in
                # the world frame — the scale every error column should be read against, since an
                # 0.5 m/s^2 residual means something very different during quiet standing than at
                # heel strike.
                'truth_lin_norm': np.linalg.norm(
                    condition(reference_linacc[:context.n], window), axis=1),
                'gyro_norm': np.linalg.norm(condition(context.gyro(sensor), window), axis=1),
                # How far apart the two segments think the joint is, per sample. This family's
                # truth uses the segment's own centre, so this is the part of the disagreement
                # `err_proj_alt` picks up and `err_proj` does not.
                'jc_residual_norm': jc_residual_norm[window.index],
            },
            scalars={'arm_source_m': float(np.linalg.norm(offset)),
                     'arm_target_m': np.nan,
                     'sep_m': float(np.linalg.norm(offset)),
                     'jc_residual_median_m': float(np.median(jc_residual_norm[window.index]))},
            detailed=detailed)
        if _finite(comparison.estimate, comparison.estimate_raw, comparison.truth):
            out.append(comparison)
    return out

# ==============================================================================
# Family 2: cross-segment agreement at the joint centre
# ==============================================================================

def joint_comparisons(context: TrialContext, joint: str, detailed: bool) -> List[Comparison]:
    """The two segments' projections of the SAME point, against each other.

    Both are transported into the PARENT body frame at full rate, before filtering (see the module
    docstring on why that order). The parent side is the `estimate` and the child side is the
    `truth` — an arbitrary but fixed choice, since neither is more true than the other; what makes
    it harmless is that the metrics are symmetric functions of the pair except for the sign of
    `dnorm`, whose convention is therefore parent-minus-child everywhere.

    Two scopes:
      mocap  every channel, using C(t) from the mocap rotations.
      full   the NORM channel only, over the whole record. |a_parent| - |a_child| needs no frame,
             so it is available wherever the two IMUs are, which on a biplane trial is a hundred
             times more data than the fluoroscopic window holds. The vector and angle channels are
             filled with the norm gap along a single axis so that no consumer reads a frame-
             dependent number out of a frame-free row — see `_norm_only_vectors`.
    """
    spec = context.spec
    parent_sensor, child_sensor = spec.joints[joint]
    offsets = context.offsets.get(joint)
    if offsets is None:
        return []

    parent_methods = context.all_methods(parent_sensor, offsets['parent'])
    child_methods = context.all_methods(child_sensor, offsets['child'])
    parent_raw, child_raw = context.acc(parent_sensor), context.acc(child_sensor)

    out = []
    window = context.mocap_window(parent_sensor, child_sensor)
    if window is not None:
        C = context.relative_rotation(parent_sensor, child_sensor)

        def to_parent(signal: np.ndarray) -> np.ndarray:
            return np.einsum('nij,nj->ni', C, signal)

        truth_full = to_parent(child_methods[PRIMARY_GYRO_METHOD])
        truth_raw_full = to_parent(child_raw)
        # The physical separation of the two sensors, in the parent frame: p_child - p_parent
        # equals R_parent r_parent - R_child r_child, i.e. r_parent - C r_child. Nearly constant
        # by construction — it is a rigid displacement between two segments only through the joint
        # — so the median is the number, and it is the natural lever-arm x-axis for this family.
        separation = np.linalg.norm(
            offsets['parent'] - np.einsum('nij,j->ni', C[window.index], offsets['child']), axis=1)

        comparison = Comparison(
            family='joint', group=joint, variant='parent-child', source=parent_sensor,
            target=child_sensor, target_kind='joint', window=window,
            timestamps=condition(context.timestamps(parent_sensor), window, cutoff=None),
            estimate=condition(parent_methods[PRIMARY_GYRO_METHOD], window),
            estimate_raw=condition(parent_raw, window),
            truth=condition(truth_full, window),
            by_method={m: condition(a, window) for m, a in parent_methods.items()},
            truth_by_method={m: condition(to_parent(a), window)
                             for m, a in child_methods.items()},
            unfiltered={'estimate': parent_methods[PRIMARY_GYRO_METHOD][window.index],
                        'estimate_raw': parent_raw[window.index],
                        'truth': truth_full[window.index]},
            extras={
                'gyro_norm': np.linalg.norm(condition(context.gyro(parent_sensor), window), axis=1),
                # The child side's own correction magnitude, so a reader can tell an agreement
                # that improved because the parent projection worked from one that improved
                # because both did.
                'partner_corr_norm': np.linalg.norm(
                    condition(child_methods[PRIMARY_GYRO_METHOD] - child_raw, window), axis=1),
            },
            scalars={'arm_source_m': float(np.linalg.norm(offsets['parent'])),
                     'arm_target_m': float(np.linalg.norm(offsets['child'])),
                     'sep_m': float(np.median(separation))},
            detailed=detailed)
        # The unprojected baseline needs the same transport, which is why it is built here rather
        # than by the generic layer: `estimate_raw` alone would be in the parent frame while
        # `truth` was the projected child, and the difference of those two is not a baseline.
        comparison.extras['truth_raw'] = condition(truth_raw_full, window)
        if _finite(comparison.estimate, comparison.truth, comparison.extras['truth_raw']):
            out.append(comparison)

    full = context.full_window()
    if full is not None:
        out.extend(_norm_only_comparison(
            context, family='joint', group=joint, variant='parent-child',
            source=parent_sensor, target=child_sensor, target_kind='joint', window=full,
            estimate=parent_methods[PRIMARY_GYRO_METHOD], estimate_raw=parent_raw,
            partner=child_methods[PRIMARY_GYRO_METHOD], partner_raw=child_raw,
            by_method=parent_methods, partner_by_method=child_methods,
            scalars={'arm_source_m': float(np.linalg.norm(offsets['parent'])),
                     'arm_target_m': float(np.linalg.norm(offsets['child'])),
                     'sep_m': np.nan},
            detailed=detailed))
    return out


def _as_magnitude(signal: np.ndarray) -> np.ndarray:
    """An (N, 3) signal reduced to its magnitude, carried on the first axis of an (N, 3) array.

    The frame-free channel can only compare LENGTHS, and this is how it says so in the same shape
    every other comparison uses. Putting each side's own magnitude on one axis — rather than
    putting their difference there — is what keeps the generic metrics correct: `err_proj` comes
    out as | |a_est| - |a_truth| |, and `dnorm_proj`, which is defined as |estimate| - |truth|,
    comes out as the SIGNED gap. Encoding the gap itself instead loses that sign, because
    `norm_gap` would then take the absolute value of it on the way through.

    The two zero axes make every angular metric degenerate (identically 0), which is why
    `channel='norm'` blanks those columns rather than letting them read as agreement.
    """
    out = np.zeros((len(signal), 3))
    out[:, 0] = np.linalg.norm(signal, axis=1)
    return out


def _norm_only_comparison(context: TrialContext, family: str, group: str, variant: str,
                          source: str, target: str, target_kind: str, window: Window,
                          estimate: np.ndarray, estimate_raw: np.ndarray,
                          partner: np.ndarray, partner_raw: np.ndarray,
                          by_method: Dict[str, np.ndarray],
                          partner_by_method: Dict[str, np.ndarray],
                          scalars: Dict[str, float], detailed: bool) -> List[Comparison]:
    """A Comparison carrying the magnitude channel alone, over a window that needs no frame.

    Built as an ordinary Comparison so that one row of `agreement_samples` is produced by one code
    path whatever it describes; what marks it out is `channel='norm'`, which every table reads to
    blank the columns a magnitude comparison cannot fill.

    FILTERED BEFORE THE MAGNITUDE IS TAKEN, matching every other family: the magnitude is a
    nonlinear function, so |lowpass(a)| and lowpass(|a|) differ, and the first is what a filter
    consuming a band-limited accelerometer would see.
    """
    filtered = {name: condition(signal, window) for name, signal in
                (('estimate', estimate), ('estimate_raw', estimate_raw),
                 ('partner', partner), ('partner_raw', partner_raw))}
    comparison = Comparison(
        family=family, group=group, variant=variant, source=source, target=target,
        target_kind=target_kind, window=window, channel='norm',
        timestamps=condition(context.timestamps(source), window, cutoff=None),
        estimate=_as_magnitude(filtered['estimate']),
        estimate_raw=_as_magnitude(filtered['estimate_raw']),
        truth=_as_magnitude(filtered['partner']),
        by_method={method: _as_magnitude(condition(signal, window))
                   for method, signal in by_method.items()},
        truth_by_method={method: _as_magnitude(condition(partner_by_method[method], window))
                         for method in by_method},
        unfiltered={'estimate': _as_magnitude(estimate[window.index]),
                    'estimate_raw': _as_magnitude(estimate_raw[window.index]),
                    'truth': _as_magnitude(partner[window.index])},
        extras={'gyro_norm': np.linalg.norm(condition(context.gyro(source), window), axis=1),
                'truth_raw': _as_magnitude(filtered['partner_raw'])},
        scalars=scalars, detailed=detailed)
    return [comparison] if _finite(comparison.estimate, comparison.estimate_raw,
                                   comparison.truth) else []

# ==============================================================================
# Family 3: same-segment agreement
# ==============================================================================

def same_segment_groups(spec: DatasetSpec) -> Dict[str, List[Tuple[str, str]]]:
    """{segment: [(placement, sensor), ...]} for segments carrying more than one sensor.

    Read off the spec's DISPLAY names — 'Thigh R High' / 'Thigh R Mid' / 'Thigh R Low' become one
    'Thigh R' with three placements — rather than off IMoVE's private plate-name convention, so a
    future dataset that labels placements the same way is picked up with no change. Empty for
    every dataset with one sensor per segment, which silences this whole family on Al Borno and
    on both biplane specs rather than having them report a degenerate self-comparison.

    Ordered proximal to distal within a segment, following PLACEMENT_TOKENS, because the pair
    labels ('High-Mid', 'Mid-Low', 'High-Low') are then the geometry: the first two span roughly
    half the segment and the third spans all of it.
    """
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for display, sensor in spec.segment_sensor.items():
        head, _, tail = display.rpartition(' ')
        if not head or tail not in PLACEMENT_TOKENS:
            continue
        groups.setdefault(head, []).append((tail, sensor))
    ordered = {}
    for segment, members in groups.items():
        if len(members) < 2:
            continue
        ordered[segment] = sorted(members, key=lambda m: PLACEMENT_TOKENS.index(m[0]))
    return ordered


def offsets_by_sensor(context: TrialContext) -> Dict[str, Dict[str, np.ndarray]]:
    """{sensor: {anatomical joint: offset}} across every fitted pair, placement variants folded in.

    'R_Knee_H' contributes the THIGH_R_H and SHANK_R_H offsets under 'R_Knee', so a sensor's
    lever arm to a joint is found by the joint's ANATOMICAL name however the pair that produced
    it was labelled. That is what lets the same-segment family project three thigh sensors to one
    knee: each placement's arm comes from its own placement-matched fit, which is the only fit
    that ever saw that sensor.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for joint, sides in context.offsets.items():
        anatomical = canonical_joint(joint, context.spec)
        for role, sensor in zip(ROLES, context.spec.joints[joint]):
            out.setdefault(sensor, {})[anatomical] = sides[role]
    return out


def gyro_sync_diagnostic(gyro_a: np.ndarray, gyro_b: np.ndarray, fs: float,
                         max_lag: int = 5) -> Dict[str, float]:
    """Why two sensors on ONE rigid segment appear to measure different angular velocities.

    omega_A = C omega_B holds exactly on a rigid body, so any residual is one of three things, and
    this separates the first from the other two by brute force: for every integer sample shift in
    +-`max_lag`, refit C at that shift and score the mismatch.

        RELATIVE TIMING. Two devices with independent clocks, shifted by dt, disagree by
        |domega/dt| dt whatever C is. IMoVE's build syncs the whole inertial record to mocap on the
        PELVIS (see `imove_mocap.load_trial`); nothing aligns two devices on the same segment to
        each other, so a per-device offset survives. At 40 Hz one sample is 25 ms, and at
        |domega/dt| ~ 10 rad/s^2 that is 0.25 rad/s of pure artifact — the right order to explain
        what is measured.
        A NON-RIGID SEGMENT. Soft tissue and mounting compliance. Real, and the thing the
        same-segment family would like to be measuring a floor for.
        GYRO SCALE OR BIAS. A per-device calibration difference, which no shift and no rotation
        removes.

    A large `improvement_%` at a nonzero `best_lag_samples` is the first; a mismatch that barely
    moves is the second or third. The distinction matters because only the first is fixable, and it
    is fixable in the BUILD layer rather than here — this function measures it and reports it, and
    deliberately does not shift the signals it was handed. Correcting a sync in an analysis module
    would put two different alignments in the repository.

    C is refitted per shift because lag and rotation are coupled: a lag rotates the apparent
    relationship between the two gyro vectors, so scoring shifts against one fixed C would
    understate what the best shift can do. Eleven 3x3 SVDs is nothing next to one projection.
    """
    n = min(len(gyro_a), len(gyro_b))
    if n <= 4 * max_lag:
        return {}
    core = slice(max_lag, n - max_lag)
    best = None
    at_zero = np.nan
    for shift in range(-max_lag, max_lag + 1):
        shifted = gyro_a[max_lag + shift:n - max_lag + shift]
        rotation = calculate_best_fit_rotation(shifted, gyro_b[core])
        residual = rms(shifted - gyro_b[core] @ rotation.T)
        if shift == 0:
            at_zero = residual
        if best is None or residual < best[1]:
            best = (shift, residual)
    lag, at_best = best
    return {
        'gyro_best_lag_samples': float(lag),
        'gyro_best_lag_ms': float(1000.0 * lag / fs),
        'gyro_mismatch_at_zero_lag': at_zero,
        'gyro_mismatch_at_best_lag': at_best,
        'gyro_lag_improvement_pct': float(100.0 * (1.0 - at_best / at_zero)) if at_zero else np.nan,
    }


def segment_geometry(context: TrialContext, sensor_a: str, sensor_b: str
                     ) -> Optional[Dict[str, object]]:
    """The constant geometry between two sensors on one segment, two ways.

    Returns {'r_ab', 'C_ab', 'C_mocap', ...diagnostics} where

        r_ab      the offset from A to B expressed in A's body frame, so `project_acc(r_ab)` on A
                  lands on B. From mocap; `fit_separation_inertial` re-solves it from the IMUs.
        C_mocap   R_A^T R_B averaged over valid frames — the frame transform mocap implies
        C_ab      THE ONE USED, fitted instead from the two GYROS by Wahba/Kabsch

    WHY THE GYRO FIT IS PRIMARY, which is a measured decision and not a preference. Two sensors on
    ONE rigid segment observe the same angular velocity, so omega_A = C_ab omega_B exactly, and the
    gyros therefore determine C_ab directly — over the whole record, with no mocap involved. The
    mocap route determines it only indirectly, as the ratio of two independently fitted
    sensor-to-segment rotations (`assembly.align_world_to_imu` solves each plate's from that
    plate's own gyro against the marker cluster), so it inherits both fits' errors.

    On IMoVE those two answers disagree by ~10 deg, and the cost is not subtle: under C_mocap the
    two sensors' gyros appear to disagree by 20-32% of their own magnitude, which for a rigid
    segment is impossible. Using it would have charged the projection for a frame error — the
    same-segment residual dropped ~40% when one constant rotation was fitted away, which is what
    first showed this up. `C_mocap_vs_gyro_deg` and the residual under each are both recorded so
    the choice is auditable rather than asserted.

    Fitting the transform on the GYROS and then scoring the ACCELEROMETERS is not circular: they
    are different measurements, and identifying a constant sensor-to-sensor rotation once is
    exactly the calibration a real two-IMU-per-segment mounting would do. Fitting it on the
    accelerometers would be circular, and is not done.

    The remaining diagnostics say out loud what mocap can and cannot see here:

        r_spread_mm       spread of the per-sample r_ab about its median, in millimetres
        C_spread_*_deg    angular deviation of C_mocap(t) from its own chordal mean

    Both are ~0 on IMoVE BY CONSTRUCTION, not by measurement: all placements on a segment are
    reconstructed from one marker cluster, so r_ab is exactly the difference of two fitted sensor
    offsets and C_mocap exactly the ratio of two fitted rotations. Their being zero is evidence for
    that structure, NOT evidence that the segment is rigid — mocap cannot see these two sensors
    independently, so it cannot report on their relative motion. A non-zero reading would mean the
    two plates no longer share a cluster and this function's constancy assumption had stopped
    holding, which is why they are computed rather than skipped.

    The median rather than the mean for r_ab: on a segment whose reconstruction glitches, a handful
    of frames inside `valid` still carry a bad pose, and a mean carries them into the offset every
    projection then uses.
    """
    valid = (np.asarray(context.plates[sensor_a].valid)[:context.n]
             & np.asarray(context.plates[sensor_b].valid)[:context.n])
    if int(valid.sum()) < MIN_FIT_FRAMES:
        return None

    rotations_a = context.rotations(sensor_a)[valid]
    rotations_b = context.rotations(sensor_b)[valid]
    displacement = (context.positions(sensor_b) - context.positions(sensor_a))[valid]
    r_samples = np.einsum('nji,nj->ni', rotations_a, displacement)
    r_ab = np.median(r_samples, axis=0)

    relative = np.einsum('nji,njk->nik', rotations_a, rotations_b)
    C_mocap = chordal_mean(relative)
    spread_rms_deg, spread_max_deg = rotation_spread_deg(relative, C_mocap)

    # Fitted over the WHOLE record, not over `valid`: the gyros do not care what the markers were
    # doing, and a constant rotation is better determined from more of the motion. Weighting is
    # implicit in Wahba and correct here — fast samples carry more information about the transform.
    # `calculate_best_fit_rotation(p, c)` returns the R with p = R c, so passing (A, B) gives the
    # B-to-A transform C_ab; TestAccelerationProjection pins that direction against the fixture.
    C_gyro = calculate_best_fit_rotation(context.gyro(sensor_a), context.gyro(sensor_b))
    disagreement = float(np.degrees(np.linalg.norm(
        Rotation.from_matrix(C_mocap.T @ C_gyro).as_rotvec())))

    return {
        'r_ab': r_ab,
        'C_ab': C_gyro,
        'C_mocap': C_mocap,
        **gyro_sync_diagnostic(context.gyro(sensor_a), context.gyro(sensor_b), context.fs),
        'sep_m': float(np.linalg.norm(r_ab)),
        'r_spread_mm': float(np.median(np.linalg.norm(r_samples - r_ab, axis=1)) * 1000.0),
        'C_spread_rms_deg': spread_rms_deg,
        'C_spread_max_deg': spread_max_deg,
        'C_mocap_vs_gyro_deg': disagreement,
        'n_geometry_frames': int(valid.sum()),
    }


def fit_separation_inertial(estimate_raw: np.ndarray, gyro: np.ndarray, alpha: np.ndarray,
                            partner_in_source_frame: np.ndarray) -> Dict[str, float]:
    """Re-solve the sensor separation from the two IMUs alone, and score it.

    Equation (3) is LINEAR in r: a_A + K(t) r = C a_B with K = [alpha]x + [omega]x^2, so

        r_hat = argmin_r  sum_t | K(t) r - (C a_B(t) - a_A(t)) |^2

    is an ordinary 3-parameter least squares over 3N rows — no optimizer, no initial guess, no
    local minimum. This is only possible because the two frames are related by a CONSTANT
    rotation; at a joint, C varies and both offsets are unknown, which is the harder problem
    `experiments/inertial_joint_center.py` solves with Seel's magnitude-matching objective.

    Why it is worth having: the same-segment residual is charged to the projection, but part of it
    is the mocap geometry the projection was handed. `r_hat` is the separation the accelerometers
    themselves prefer, so |r_hat - r_mocap| is the geometry disagreement in millimetres and
    `rms_at_inertial` is the floor the projection physics reaches when the geometry is not
    holding it back. If the two rms values are close, the mocap offsets were not the problem.

    K IS AN ERRORS-IN-VARIABLES REGRESSOR — it is built from a differentiated gyro, so its noise
    biases r_hat TOWARD ZERO rather than merely scattering it. `attenuation` (|r_hat| / |r_mocap|)
    is reported so that a short r_hat is read as the attenuation it is and not as a claim that the
    sensors are closer together than the markers say; `inertial_joint_center` measures the same
    effect on the joint version and finds it substantial.

    `condition` here is the least-squares condition number of the stacked design, which says
    whether the trial excited the segment enough to determine three numbers at all. A quiet trial
    leaves K near zero and r_hat unidentified; a large value is the warning that the millimetres
    below are noise.
    """
    K = lever_operator(gyro, alpha)
    design = K.reshape(-1, 3)
    target = (partner_in_source_frame - estimate_raw).reshape(-1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    singular = np.linalg.svd(design, compute_uv=False)
    residual_at_fit = design @ solution - target
    return {
        'r_inertial_x': float(solution[0]),
        'r_inertial_y': float(solution[1]),
        'r_inertial_z': float(solution[2]),
        'rms_at_inertial': float(np.sqrt(np.mean(residual_at_fit ** 2) * 3.0)),
        'design_condition': float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf,
    }


def segment_comparisons(context: TrialContext, segment: str, members: List[Tuple[str, str]],
                        arms: Dict[str, Dict[str, np.ndarray]], detailed: bool
                        ) -> List[Comparison]:
    """Every pair of sensors on one segment, projected onto each other and onto shared joints.

    Two target kinds, and they are different measurements rather than two views of one:

      partner  A projected onto B's own site. The truth is B's READING, so this comparison has no
               differentiated markers anywhere in it and is band-limited by the sensors. It is the
               sharpest test of the projection physics in the repository, and the only one whose
               truth has the same noise spectrum as its estimate.
      joint    both sensors projected to the same joint centre, over arms of 100-250 mm. No truth —
               neither side is more right — so the metric is mutual agreement, which is exactly
               what a relative filter would consume from a two-IMU-per-segment mounting. It tests
               the long lever the pipeline actually uses, where the partner target only spans the
               ~50-150 mm between two placements.

    Scored on the FULL scope: everything mocap supplies here is a constant (r_ab, C_ab), fitted on
    valid frames and then applied to the whole record, so there is no reason to throw away the
    frames where the markers happened to drop out. The mocap scope is emitted as well so this
    family lands on the same samples as the other two when they are compared side by side.
    """
    out: List[Comparison] = []
    for i, (placement_a, sensor_a) in enumerate(members):
        for placement_b, sensor_b in members[i + 1:]:
            if sensor_a not in context.plates or sensor_b not in context.plates:
                continue
            geometry = segment_geometry(context, sensor_a, sensor_b)
            if geometry is None:
                continue
            variant = f"{placement_a}-{placement_b}"
            targets: List[Tuple[str, str, np.ndarray, Optional[np.ndarray]]] = [
                ('partner', sensor_b, geometry['r_ab'], None)]
            for joint, arm_a in sorted(arms.get(sensor_a, {}).items()):
                arm_b = arms.get(sensor_b, {}).get(joint)
                if arm_b is not None:
                    targets.append((f'joint:{joint}', joint, arm_a, arm_b))

            for target_kind, target_name, offset_a, offset_b in targets:
                for window in _segment_windows(context, sensor_a, sensor_b):
                    comparison = _segment_comparison(
                        context, segment, variant, sensor_a, sensor_b, target_kind, target_name,
                        offset_a, offset_b, geometry, window, detailed)
                    if comparison is not None:
                        out.append(comparison)
    return out


def _segment_windows(context: TrialContext, sensor_a: str, sensor_b: str) -> List[Window]:
    """The full record first, then the mocap window when it adds anything.

    The mocap window is skipped when it covers the whole record anyway (nothing is invalid), since
    two identical rows differing only in a label would double this family's weight in every
    pooled quantile.
    """
    windows = [w for w in (context.full_window(),) if w is not None]
    mocap = context.mocap_window(sensor_a, sensor_b)
    if mocap is not None and len(mocap) < len(windows[0] if windows else mocap):
        windows.append(mocap)
    return windows


def _segment_comparison(context: TrialContext, segment: str, variant: str, sensor_a: str,
                        sensor_b: str, target_kind: str, target_name: str,
                        offset_a: np.ndarray, offset_b: Optional[np.ndarray],
                        geometry: Dict[str, object], window: Window, detailed: bool
                        ) -> Optional[Comparison]:
    """One same-segment comparison: A projected to the target, against B's version of it.

    `offset_b is None` means the target is B itself, so B's version is its raw reading; otherwise
    B projects to the same point over its own arm. Either way both sides are brought into A's body
    frame by the constant C_ab, and the unprojected baseline is A's raw reading against the same
    truth — which is what makes the projection's contribution readable off one row.
    """
    C_ab = np.asarray(geometry['C_ab'])
    C_mocap = np.asarray(geometry['C_mocap'])
    by_method_a = context.all_methods(sensor_a, offset_a)
    raw_a = context.acc(sensor_a)
    raw_b_in_a = context.acc(sensor_b) @ C_ab.T

    if offset_b is None:
        # The partner target's truth is B's own reading, which no derivative scheme touches — so
        # `truth_by_method` stays None and the gyro table varies one side only, correctly.
        truth_full = raw_b_in_a
        truth_raw_full = raw_b_in_a
        partner_methods = None
    else:
        partner_methods = {method: signal @ C_ab.T
                           for method, signal in context.all_methods(sensor_b, offset_b).items()}
        truth_full = partner_methods[PRIMARY_GYRO_METHOD]
        # The baseline for a joint target is BOTH sides unprojected: the question there is whether
        # projecting brought two sensors into agreement, and leaving the truth side projected
        # would compare a projected signal against a projected one and call it the baseline.
        truth_raw_full = raw_b_in_a

    gyro_a = condition(context.gyro(sensor_a), window)
    gyro_b_in_a = condition(context.gyro(sensor_b) @ C_ab.T, window)
    # The same comparison under the frame transform MOCAP implies, so the cost of preferring the
    # gyro fit is a number in the table rather than a claim in a docstring. Only the transform
    # changes: same estimate, same samples, same filter. See `segment_geometry`.
    gyro_b_mocap_frame = condition(context.gyro(sensor_b) @ C_mocap.T, window)
    estimate = condition(by_method_a[PRIMARY_GYRO_METHOD], window)
    truth_mocap_frame = condition(np.asarray(context.acc(sensor_b) if offset_b is None
                                             else context.projected(sensor_b, offset_b))
                                  @ C_mocap.T, window)

    comparison = Comparison(
        family='segment', group=segment, variant=variant, source=sensor_a, target=target_name,
        target_kind=target_kind, window=window,
        timestamps=condition(context.timestamps(sensor_a), window, cutoff=None),
        estimate=estimate,
        estimate_raw=condition(raw_a, window),
        truth=condition(truth_full, window),
        by_method={m: condition(a, window) for m, a in by_method_a.items()},
        truth_by_method=({m: condition(a, window) for m, a in partner_methods.items()}
                         if partner_methods is not None else None),
        unfiltered={'estimate': by_method_a[PRIMARY_GYRO_METHOD][window.index],
                    'estimate_raw': raw_a[window.index],
                    'truth': truth_full[window.index]},
        extras={
            'gyro_norm': np.linalg.norm(gyro_a, axis=1),
            # THE RIGIDITY FLOOR. Two sensors on one rigid segment measure the SAME angular
            # velocity, so this is zero for a rigid segment with a correct C_ab, and whatever it
            # reads is a floor the acceleration agreement cannot beat: an omega disagreement of
            # dw contributes ~|dw||omega||r| to the centripetal term alone. It costs one
            # subtraction and it is the only column here that separates "the projection is wrong"
            # from "the segment is not the rigid body the projection assumes".
            'gyro_mismatch': np.linalg.norm(gyro_a - gyro_b_in_a, axis=1),
            'gyro_mismatch_mocap_frame': np.linalg.norm(gyro_a - gyro_b_mocap_frame, axis=1),
            'truth_raw': condition(truth_raw_full, window),
        },
        scalars={'arm_source_m': float(np.linalg.norm(offset_a)),
                 'arm_target_m': float(np.linalg.norm(offset_b)) if offset_b is not None else 0.0,
                 'sep_m': float(geometry['sep_m']),
                 'r_spread_mm': float(geometry['r_spread_mm']),
                 'C_spread_rms_deg': float(geometry['C_spread_rms_deg']),
                 'C_spread_max_deg': float(geometry['C_spread_max_deg']),
                 'C_mocap_vs_gyro_deg': float(geometry['C_mocap_vs_gyro_deg']),
                 # What the mocap-implied frame transform would have cost, on the same estimate and
                 # the same samples. This is the evidence for `C_ab` being the gyro fit.
                 'rms_mocap_frame': rms(estimate - truth_mocap_frame),
                 'n_geometry_frames': int(geometry['n_geometry_frames']),
                 **{key: value for key, value in geometry.items()
                    if key.startswith('gyro_')}},
        detailed=detailed)

    # The inertial geometry fit, for the partner target only: it is the one case where the truth
    # side carries no assumed offset of its own, so a disagreement between the inertial and mocap
    # separations is unambiguously about r_ab.
    if offset_b is None:
        alpha = condition(gyro_derivative(context.plates[sensor_a].imu_trace,
                                          PRIMARY_GYRO_METHOD)[:context.n], window)
        fit = fit_separation_inertial(comparison.estimate_raw, gyro_a, alpha, comparison.truth)
        r_inertial = np.array([fit['r_inertial_x'], fit['r_inertial_y'], fit['r_inertial_z']])
        arm = np.linalg.norm(offset_a)
        comparison.scalars.update(fit)
        comparison.scalars['r_inertial_error_mm'] = float(
            np.linalg.norm(r_inertial - offset_a) * 1000.0)
        comparison.scalars['attenuation'] = float(np.linalg.norm(r_inertial) / arm) if arm else np.nan

    if not _finite(comparison.estimate, comparison.estimate_raw, comparison.truth):
        return None
    return comparison

# ==============================================================================
# Building every comparison for a trial
# ==============================================================================

def trial_comparisons(context: TrialContext) -> List[Comparison]:
    """Every comparison this trial supports, across all three families.

    A joint missing either sensor, or too poorly covered to fit an offset, is skipped rather than
    failing the trial: a dropped sensor costs that joint, not the other seventeen.
    """
    spec = context.spec
    comparisons: List[Comparison] = []
    for joint in spec.joints:
        parent_sensor, child_sensor = spec.joints[joint]
        if parent_sensor not in context.plates or child_sensor not in context.plates:
            continue
        detailed = canonical_joint(joint, spec) == joint
        comparisons.extend(marker_comparisons(context, joint, detailed))
        comparisons.extend(joint_comparisons(context, joint, detailed))

    arms = offsets_by_sensor(context)
    for segment, members in same_segment_groups(spec).items():
        present = [(placement, sensor) for placement, sensor in members
                   if sensor in context.plates]
        if len(present) < 2:
            continue
        comparisons.extend(segment_comparisons(context, segment, present, arms, detailed=True))
    return comparisons

# ==============================================================================
# Tables, all generated from a list of Comparisons
# ==============================================================================

def sample_rows(comparison: Comparison, stride: int = SAMPLE_STRIDE) -> pd.DataFrame:
    """Per-sample metrics for one comparison, decimated by `stride`.

    `corr_norm` — the size of the correction the projection applied, IN THE CURRENCY THIS
    COMPARISON IS SCORED IN — is the natural x-axis for "when does this degrade". For the marker
    family it is |a_proj - a_sensor|. For the two agreement families it is the change in the
    DISAGREEMENT, |(est - est_raw) - (truth - truth_raw)|: what matters there is not how much each
    projection moved but how much it moved the gap, and a pair of segments whose corrections move
    together has been asked to do nothing however large those corrections are.

    The per-axis `ref_*`/`diff_*` columns are stored rather than a precomputed mean-vs-difference
    pair so the plotting layer can build a Bland-Altman panel on either the raw axes or their
    difference without this table committing to one of them.
    """
    step = slice(None, None, stride)
    truth_raw = comparison.extras.get('truth_raw', comparison.truth)
    diff = comparison.estimate - comparison.truth
    correction = ((comparison.estimate - comparison.estimate_raw)
                  - (comparison.truth - truth_raw))
    dnorm_proj = norm_gap(comparison.estimate, comparison.truth)
    dnorm_raw = norm_gap(comparison.estimate_raw, truth_raw)

    frame = {
        **{key: value for key, value in comparison.key.items()},
        'timestamp': comparison.timestamps[step].astype(np.float64),
        'run_id': comparison.window.run_id[step].astype(np.int32),
        'err_proj': np.linalg.norm(diff, axis=1)[step].astype(np.float32),
        'err_raw': np.linalg.norm(comparison.estimate_raw - truth_raw, axis=1)[step].astype(np.float32),
        # NaN rather than the identical zero a magnitude-only comparison would produce. A zero here
        # would read as perfect directional agreement in every quantile, every boxplot and every
        # pooled median that did not know to exclude it — the single most misleading number this
        # table could contain.
        'ang_proj': (np.full(len(diff), np.nan) if comparison.channel == 'norm'
                     else angle_between_deg(comparison.estimate,
                                            comparison.truth))[step].astype(np.float32),
        'ang_raw': (np.full(len(diff), np.nan) if comparison.channel == 'norm'
                    else angle_between_deg(comparison.estimate_raw,
                                           truth_raw))[step].astype(np.float32),
        'dnorm_proj': dnorm_proj[step].astype(np.float32),
        'dnorm_raw': dnorm_raw[step].astype(np.float32),
        'absdnorm_proj': np.abs(dnorm_proj)[step].astype(np.float32),
        'absdnorm_raw': np.abs(dnorm_raw)[step].astype(np.float32),
        'corr_norm': np.linalg.norm(correction, axis=1)[step].astype(np.float32),
        'ref_x': comparison.truth[step, 0].astype(np.float32),
        'ref_y': comparison.truth[step, 1].astype(np.float32),
        'ref_z': comparison.truth[step, 2].astype(np.float32),
        'diff_x': diff[step, 0].astype(np.float32),
        'diff_y': diff[step, 1].astype(np.float32),
        'diff_z': diff[step, 2].astype(np.float32),
    }
    if comparison.truth_alt is not None:
        frame['err_proj_alt'] = np.linalg.norm(
            comparison.estimate - comparison.truth_alt, axis=1)[step].astype(np.float32)
    frame['err_proj_nofilt'] = np.linalg.norm(
        comparison.unfiltered['estimate'] - comparison.unfiltered['truth'],
        axis=1)[step].astype(np.float32)
    for name in ('gyro_norm', 'gyro_mismatch', 'gyro_mismatch_mocap_frame', 'truth_lin_norm',
                 'jc_residual_norm', 'partner_corr_norm'):
        if name in comparison.extras:
            frame[name] = comparison.extras[name][step].astype(np.float32)
    return pd.DataFrame(frame)


def stats_row(comparison: Comparison) -> Dict[str, object]:
    """One row of per-comparison scalars: rms either side of the fitted misalignment, R^2, the
    lever arms, and whatever geometry that family measured.

    `r2` is variance-accounted-for on the whole 3-vector,
    1 - sum|est - truth|^2 / sum|truth - mean(truth)|^2, rather than three per-axis values. A
    per-axis R^2 on this data is close to 1 by construction for whichever axis gravity happens to
    sit on, so it would mostly report the body frame's orientation.
    """
    truth_raw = comparison.extras.get('truth_raw', comparison.truth)
    if comparison.channel == 'norm':
        # A magnitude comparison has no orientation to misalign, and the Kabsch fit on two vectors
        # that both lie on one axis is degenerate: it returns something (the SVD of a rank-1
        # matrix), and what it returns is a rotation about an unconstrained axis with no meaning.
        # NaN says that; 0.000 would have read as "perfectly aligned".
        angle_deg, rms_after = np.nan, np.nan
        rms_before = rms(comparison.estimate - comparison.truth)
    else:
        _, angle_deg, rms_before, rms_after = residual_alignment(comparison.estimate,
                                                                comparison.truth)
    centered = comparison.truth - comparison.truth.mean(axis=0)
    total_var = float(np.sum(centered ** 2))
    row = {
        **comparison.key,
        'n_samples': len(comparison.estimate),
        'duration_s': comparison.window.duration_s,
        'n_runs': len(comparison.window.runs),
        'valid_fraction': comparison.window.n_mask / max(comparison.window.n_record, 1),
        'fs': comparison.window.fs,
        'align_angle_deg': angle_deg,
        'rms_before': rms_before,
        'rms_after': rms_after,
        'rms_raw': rms(comparison.estimate_raw - truth_raw),
        'rms_nofilt': rms(comparison.unfiltered['estimate'] - comparison.unfiltered['truth']),
        'r2_proj': 1.0 - float(np.sum((comparison.estimate - comparison.truth) ** 2)) / total_var
        if total_var > 0 else np.nan,
        'r2_raw': 1.0 - float(np.sum((comparison.estimate_raw - truth_raw) ** 2)) / total_var
        if total_var > 0 else np.nan,
        'median_corr_norm': float(np.median(np.linalg.norm(
            (comparison.estimate - comparison.estimate_raw)
            - (comparison.truth - truth_raw), axis=1))),
        'median_absdnorm_proj': float(np.median(np.abs(
            norm_gap(comparison.estimate, comparison.truth)))),
        'median_absdnorm_raw': float(np.median(np.abs(norm_gap(comparison.estimate_raw, truth_raw)))),
    }
    if comparison.truth_alt is not None:
        row['rms_alt'] = rms(comparison.estimate - comparison.truth_alt)
    if 'gyro_mismatch' in comparison.extras:
        row['median_gyro_mismatch'] = float(np.median(comparison.extras['gyro_mismatch']))
        row['median_gyro_norm'] = float(np.median(comparison.extras['gyro_norm']))
    row.update(comparison.scalars)
    return row


def _dynamic_slice(comparison: Comparison, max_s: float = TRACE_MAX_S) -> slice:
    """The most dynamic `max_s` of one comparison, as a slice, never straddling a run boundary.

    Scored by the total absolute change in the truth signal over the window — a sum of |diff|
    rather than a variance, because what a time-series panel needs to show is the signal MOVING,
    and a window sitting at a large constant offset has a big variance about the trial mean while
    showing nothing at all.

    Deterministic, so the stored trace does not depend on when it was computed, and confined to a
    single run so the panel never draws a straight line across a stretch of missing mocap as if it
    were data.
    """
    n = len(comparison.timestamps)
    width = min(n, int(round(max_s * comparison.window.fs)))
    if width >= n:
        return slice(0, n)
    activity = np.r_[0.0, np.abs(np.diff(comparison.truth, axis=0)).sum(axis=1)]
    # Zero out any step that crosses a run boundary, so a window is only ever scored on, and
    # placed inside, one contiguous run.
    activity[1:][np.diff(comparison.window.run_id) != 0] = 0.0
    totals = np.convolve(activity, np.ones(width), mode='valid')
    boundary_free = np.array([comparison.window.run_id[i] == comparison.window.run_id[i + width - 1]
                              for i in range(len(totals))])
    if boundary_free.any():
        totals = np.where(boundary_free, totals, -np.inf)
    start = int(np.argmax(totals))
    return slice(start, start + width)


def traces_row(comparison: Comparison) -> pd.DataFrame:
    """Full-rate components of all four signals over one window, for the time-series panel.

    Windowed rather than complete, and written for at most one comparison per family (see
    `trace_comparisons`). Storing every detailed comparison at full rate was 160 MB per Al Borno
    trial — more than every other table in this experiment combined, for a panel that draws four
    seconds of one signal.

    All four signals are stored, not three: `truth_raw` is what the truth side looked like before
    ITS projection, which is what makes a cross-segment panel readable at all — without it the
    reader sees two projected signals agreeing and cannot tell whether they started apart.
    """
    window = _dynamic_slice(comparison)
    truth_raw = comparison.extras.get('truth_raw', comparison.truth)
    frame = {**comparison.key, 'timestamp': comparison.timestamps[window].astype(np.float64),
             'run_id': comparison.window.run_id[window].astype(np.int32)}
    for name, signal in (('truth', comparison.truth), ('proj', comparison.estimate),
                         ('raw', comparison.estimate_raw), ('truth_raw', truth_raw)):
        for axis, label in enumerate('xyz'):
            frame[f'{name}_{label}'] = signal[window, axis].astype(np.float32)
    return pd.DataFrame(frame)


def trace_comparisons(comparisons: Sequence[Comparison]) -> List[Comparison]:
    """At most one comparison per family to store full-rate traces for.

    Preference order, applied per family: a vector channel over a magnitude one (a magnitude trace
    is a single line and shows nothing about direction), the mocap scope where both exist (so the
    truth on the panel is the family's real truth), the `EXAMPLE_GROUP_HINT` group, the partner
    target for the segment family, and the parent side. Every step is a preference rather than a
    filter, so a dataset with none of them still gets a trace.
    """
    chosen = []
    for family in FAMILIES:
        candidates = [c for c in comparisons if c.family == family and c.detailed]
        if not candidates:
            continue
        chosen.append(min(candidates, key=lambda c: (
            c.channel != 'vector',
            c.scope != 'mocap',
            EXAMPLE_GROUP_HINT.lower() not in c.group.lower(),
            c.target_kind != 'partner' if c.family == 'segment' else False,
            c.variant != EXAMPLE_SEGMENT_VARIANT if c.family == 'segment' else c.variant == 'child',
            c.group, c.variant)))
    return chosen


def welch_psds(comparison: Comparison) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Welch PSD of one comparison's three UNFILTERED signals, as (freqs, {name: psd}).

    Each PSD is averaged over the three axes: the sum of the axis PSDs is the power spectrum of
    the vector signal and does not depend on how the body frame happens to be oriented, which a
    single axis very much does. (The factor of 3 between the mean and that sum is applied by
    `reference_excess_rms`, the only consumer that needs the vector total.)

    Shared by `spectra_rows` and `cutoff_rows` so the spectrum a figure draws and the spectrum the
    noise correction is computed from are the same numbers.
    """
    nperseg = min(WELCH_NPERSEG, len(comparison.timestamps))
    freqs = None
    psds = {}
    for name, signal in comparison.unfiltered.items():
        freqs, psd = welch(signal, fs=comparison.window.fs, nperseg=nperseg, axis=0)
        psds[name] = psd.mean(axis=1)
    return freqs, psds


def spectra_rows(comparison: Comparison) -> pd.DataFrame:
    """Welch PSD of the three UNFILTERED signals.

    This is the panel that settles the standing objection to the method — that differentiating a
    noisy gyro to get alpha injects high-frequency noise into the accelerometer signal. It is a
    fair test only on unfiltered data, since the shared low-pass would remove exactly the band in
    question. For the marker family it also shows why the cutoff is where it is: the two IMU
    spectra roll off above ~5 Hz while the mocap truth flattens onto a differentiation noise floor
    and stays there to Nyquist. For the other two families the truth is an accelerometer, so its
    spectrum tracks the estimate's all the way up — which is the same fact stated as a picture.
    """
    freqs, psds = welch_psds(comparison)
    return pd.DataFrame({**comparison.key, 'freq_hz': freqs.astype(np.float32),
                         **{f'psd_{name}': psd.astype(np.float32)
                            for name, psd in psds.items()}})


def reference_excess_rms(freqs: np.ndarray, psd_truth: np.ndarray, psd_estimate: np.ndarray,
                         cutoff: float) -> float:
    """How much residual the TRUTH's own in-band noise can account for on its own, in m/s^2, for a
    low-pass at `cutoff`.

    Integrates the truth's EXCESS power over the estimate's, below the cutoff:

        rms = sqrt( 3 * integral_0^fc max(0, PSD_truth(f) - PSD_est(f)) df )

    The factor 3 turns a per-axis PSD (psd_* columns are the mean over the three axes) into the
    variance of the 3-vector residual. The max(0, .) drops bands where the truth has LESS power,
    which is not truth noise and must not be allowed to cancel out bands where it has more.

    Measuring the excess rather than assuming a white floor and extrapolating it down to DC is the
    whole point. The mocap floor looks flat above 5 Hz, but differentiation suppresses
    low-frequency noise, so extrapolating a flat floor across the whole band overstates the in-band
    noise badly — at a 15 Hz cutoff it predicts more residual than is actually measured, which is
    impossible and would have made this a misleading correction rather than an informative one.
    The excess integral needs no such assumption: it only assumes the estimate's spectrum is a fair
    stand-in for the true signal's inside the band, and below 5 Hz the two agree to a few percent.

    An upper bound, not a subtraction, for two reasons, and nothing downstream removes it from the
    measured residual — it is reported beside it:

      * Any real projection error that happens to raise the truth's apparent excess is counted
        here as truth noise.
      * The max(0, .) rectifies Welch's own estimator variance, so when the two spectra are
        genuinely equal this returns something positive rather than zero. On the noiseless
        synthetic joint in test/TestAccelerationProjection.py it reads ~0.8 m/s^2 against a true
        residual of ~0.1, which is the scale of that bias at this nperseg. So a SMALL value is not
        resolvable and should not be read as "the truth contributed this much"; a value comparable
        to the measured residual is the informative case.

    ONLY THE MARKER FAMILY GIVES THIS ITS INTENDED READING. There the truth is differentiated
    markers and its excess is genuinely reference noise. In the agreement families the truth is an
    accelerometer, so a nonzero excess means the two sensors differ in their own noise floors —
    still worth having, and the reason the column is computed for every family, but it is a
    statement about sensor matching rather than about mocap.
    """
    band = freqs <= cutoff
    excess = np.maximum(psd_truth[band] - psd_estimate[band], 0.0)
    return float(np.sqrt(3.0 * np.trapezoid(excess, freqs[band])))


def cutoff_rows(comparison: Comparison) -> pd.DataFrame:
    """Residual against low-pass cutoff, beside what the truth's own in-band noise accounts for.

    This is the table that makes LOWPASS_CUTOFF_HZ a measured choice rather than a taste. Each row
    re-filters the UNFILTERED signals at that cutoff and recomputes the residual from scratch, so
    no row inherits the primary cutoff's filtering.

    Cutoffs at or above the trial's Nyquist are DROPPED, not clipped: on 40 Hz IMoVE trials there
    is no such thing as a 25 Hz low-pass, and a row labelled 25 that silently held a 20 Hz result
    would be the kind of quiet error this whole file exists to catch.

    Cheap despite looking expensive: the costly part of this analysis is the projection, which has
    already happened by the time this runs. All this adds is a handful of filtfilt passes.
    """
    freqs, psds = welch_psds(comparison)
    estimate = comparison.unfiltered['estimate']
    estimate_raw = comparison.unfiltered['estimate_raw']
    truth = comparison.unfiltered['truth']
    fs = comparison.window.fs
    rows = []
    for cutoff in CUTOFF_SWEEP_HZ:
        if cutoff >= 0.5 * fs:
            continue
        truth_filtered = lowpass(truth, fs, cutoff)
        projected = lowpass(estimate, fs, cutoff)
        unprojected = lowpass(estimate_raw, fs, cutoff)
        error_proj = np.linalg.norm(projected - truth_filtered, axis=1)
        error_raw = np.linalg.norm(unprojected - truth_filtered, axis=1)
        in_band = (freqs >= NOISE_BAND_HZ[0]) & (freqs <= NOISE_BAND_HZ[1])
        rows.append({
            **comparison.key,
            'cutoff_hz': cutoff,
            'median_err_proj': float(np.median(error_proj)),
            'median_err_raw': float(np.median(error_raw)),
            'rms_err_proj': float(np.sqrt(np.mean(error_proj ** 2))),
            'rms_err_raw': float(np.sqrt(np.mean(error_raw ** 2))),
            'median_absdnorm_proj': float(np.median(np.abs(norm_gap(projected, truth_filtered)))),
            'ref_excess_rms': reference_excess_rms(freqs, psds['truth'], psds['estimate'], cutoff),
            'noise_floor_psd': float(np.median(psds['truth'][in_band])) if in_band.any() else np.nan,
        })
    return pd.DataFrame(rows)


def gyro_rows(comparison: Comparison) -> pd.DataFrame:
    """Residual under each gyro-derivative scheme, everything else held fixed.

    Same offset, same truth, same low-pass, same samples — the only thing that differs between
    rows of a cell is how alpha was estimated. `median_corr_norm` is carried alongside because it
    is the interpretive key: the schemes can only differ where the tangential term matters, so a
    cell whose correction is tiny will show no difference between them no matter how bad one is.

    Reported for EVERY family, which is the point. Scored against mocap the schemes are compared
    in the one band where they nearly agree; scored against another accelerometer there is no
    reference noise forcing the cutoff down, so the agreement families are where this table can
    actually separate them.
    """
    truth_raw = comparison.extras.get('truth_raw', comparison.truth)
    rows = []
    for method, projected in comparison.by_method.items():
        # BOTH sides move to this method where both are projections; see `truth_by_method`.
        truth = (comparison.truth if comparison.truth_by_method is None
                 else comparison.truth_by_method[method])
        error = np.linalg.norm(projected - truth, axis=1)
        rows.append({
            **comparison.key,
            'gyro_method': method,
            'median_err': float(np.median(error)),
            'p90_err': float(np.percentile(error, 90)),
            'rms_err': float(np.sqrt(np.mean(error ** 2))),
            'median_ang': (np.nan if comparison.channel == 'norm'
                           else float(np.median(angle_between_deg(projected, truth)))),
            'median_absdnorm': float(np.median(np.abs(norm_gap(projected, truth)))),
            'corr_norm_median': float(np.median(np.linalg.norm(
                (projected - comparison.estimate_raw) - (truth - truth_raw), axis=1))),
        })
    return pd.DataFrame(rows)

# ==============================================================================
# Per-trial driver / grid worker
# ==============================================================================

def compute_trial(plates: Dict[str, PlateTrial], spec: DatasetSpec, dataset: str, subject: str,
                  trial: str, tables: Sequence[str] = TRIAL_TABLES,
                  stride: int = SAMPLE_STRIDE) -> Dict[str, pd.DataFrame]:
    """Every table this experiment computes for one trial.

    Built from ONE list of comparisons rather than one pass per table, because a projection costs
    far more than any table does and the alternative recomputes it six times.
    """
    context = TrialContext(plates, spec, dataset, subject, trial)
    comparisons = trial_comparisons(context)
    if not comparisons:
        return {}
    wanted = set(tables)
    detailed = [c for c in comparisons if c.detailed]
    # A power spectrum and a cutoff sweep of a MAGNITUDE are not the quantities either table
    # documents — one axis of a rectified scalar has neither the estimate's spectrum nor its
    # bandwidth — so the frame-free channel is excluded from both. It is kept in `gyro_method`,
    # where it is the most valuable row in the table: scoring the derivative schemes against a
    # magnitude uses no mocap orientation at all.
    spectral = [c for c in detailed if c.channel == 'vector']

    def concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    builders = {
        'agreement_samples': lambda: concat([sample_rows(c, stride) for c in comparisons]),
        'agreement_stats': lambda: pd.DataFrame([stats_row(c) for c in comparisons]),
        'traces': lambda: concat([traces_row(c) for c in trace_comparisons(spectral)]),
        'spectra': lambda: concat([spectra_rows(c) for c in spectral]),
        'cutoff_sweep': lambda: concat([cutoff_rows(c) for c in spectral]),
        'gyro_method': lambda: concat([gyro_rows(c) for c in detailed]),
    }
    return {table: build() for table, build in builders.items() if table in wanted}


def _trial_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                  dataset: str = 'alborno', tables: Sequence[str] = TRIAL_TABLES,
                  stride: int = SAMPLE_STRIDE, allow_stale: bool = False) -> None:
    """One process per (subject, trial). Nothing is shared between two trials of a subject here —
    unlike global_assumptions, where the subject's global magnetic field spans them — so exposing
    every trial to the pool individually is strictly better than serializing."""
    subject, trial = row_key
    spec = get_dataset(dataset)
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()
    try:
        plates = _load_spec_plates(subject, trial, dataset, spec, allow_stale)
        results = compute_trial(plates, spec, dataset, subject, trial, tables, stride)
        if not results:
            shared_state[(row_key, stage)] = "Skipped (no comparisons)"
            return None
        for table, df in results.items():
            if df.empty:
                continue
            _save(df, trial_table_path(dataset, subject, trial, table), dataset,
                  subject=subject, trial=trial, table=table,
                  built_from_stale_cache=allow_stale)
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success"
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({type(e).__name__}: {e})"
    return None

# ==============================================================================
# Pooled summary
# ==============================================================================

SUMMARY_COLUMNS = (['dataset', 'family', 'channel', 'scope', 'subject', 'group', 'metric', 'unit',
                    'n_samples', 'n_trials', 'mean', 'std', 'min']
                   + [f"p{int(round(q * 100)):02d}" for q in QUANTILES] + ['max'])

# The keys every summary row is grouped by, before the margins. `channel` is in here rather than
# being folded into `scope` because pooling a 3-vector residual together with a magnitude gap under
# one label would produce a number that is neither.
SUMMARY_KEYS = ['family', 'channel', 'scope', 'subject', 'group']

# Column holding "<subject>/<trial>", added once before any margin is taken. `n_trials` counts
# DISTINCT (subject, trial) pairs, and it has to be materialised before pooling because the pooled
# rows overwrite `subject` with the literal 'all' — counting pairs from that frame collapses 22
# Al Borno trials to the 2 activity names they share, and the wrong number is then quoted in the
# report and in every figure caption.
TRIAL_CELL = '_subject_trial'


def _describe(df: pd.DataFrame, metric: str, keys: List[str]) -> pd.DataFrame:
    grouped = df.groupby(keys, observed=True)[metric]
    stats = grouped.agg(n_samples='size', mean='mean', std='std', min='min', max='max')
    quantiles = grouped.quantile(QUANTILES).unstack()
    quantiles.columns = [f"p{int(round(q * 100)):02d}" for q in quantiles.columns]
    trials = df.groupby(keys, observed=True)[TRIAL_CELL].nunique().rename('n_trials')
    return stats.join(quantiles).join(trials).reset_index()


def summarize(dataset: str, samples: pd.DataFrame) -> pd.DataFrame:
    """Tidy quantile table per family x scope x subject x group x metric, WITH MARGINS: rows whose
    `subject` or `group` is the literal string 'all' are the pooled version of the rows above them.

    Margins are recomputed from the samples rather than averaged from the per-subject rows — a mean
    of medians is not a median, and trials differ in length. Quantiles rather than mean +- sd
    throughout: every error metric here is a non-negative magnitude with an impulsive footfall
    tail, so a standard deviation implies a symmetry that is not there (it is reported anyway,
    next to the quantiles, for anyone who wants it).

    Keyed on the subject rather than on the trial. IMoVE has 243 trials against 26 subjects, and a
    per-trial summary over 10 metrics x 18 groups x 2 scopes is millions of rows describing
    individual walks nobody quotes. The per-trial numbers survive in agreement_stats, which is
    where the paired tests read them from.
    """
    if samples.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    metrics = [m for m in SUMMARY_METRICS if m in samples.columns]
    samples = samples.assign(**{TRIAL_CELL: samples['subject'].astype(str) + '/'
                                + samples['trial'].astype(str)})
    pooled = samples.assign(subject='all')
    frames = []
    for metric in metrics:
        present = samples[np.isfinite(samples[metric])] if metric in samples else samples
        if present.empty:
            continue
        pooled_present = pooled.loc[present.index]
        for source in (present, pooled_present):
            frames.append(_describe(source, metric, SUMMARY_KEYS).assign(metric=metric))
            frames.append(_describe(source, metric, SUMMARY_KEYS[:-1])
                          .assign(metric=metric, group='all'))
    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=SUMMARY_KEYS + ['metric'])
    summary['dataset'] = dataset
    summary['unit'] = summary['metric'].map(METRIC_UNITS)
    return summary[SUMMARY_COLUMNS].sort_values(
        ['family', 'channel', 'scope', 'metric', 'subject', 'group'])

# ==============================================================================
# Console report
# ==============================================================================

def _header(number: Optional[int], title: str, subtitle: str) -> None:
    label = f"{number}. {title}" if number is not None else title
    print(f"\n{'=' * 88}\n{label}\n   {subtitle}\n{'=' * 88}")


def _pooled(summary: pd.DataFrame, family: str, metric: str, group: str = 'all',
            scope: str = 'mocap', channel: str = 'vector') -> Optional[pd.Series]:
    """The pooled (subject='all') summary row for one family/metric/group/scope/channel, or None.

    Numeric fields are coerced on the way out: the row also carries the label strings, so without
    this every quantile read from it is an object and arithmetic on it silently produces objects
    too."""
    rows = summary[(summary['subject'] == 'all') & (summary['family'] == family)
                   & (summary['metric'] == metric) & (summary['group'] == group)
                   & (summary['scope'] == scope) & (summary['channel'] == channel)]
    if rows.empty:
        return None
    row = rows.iloc[0].copy()
    labels = ('dataset', 'family', 'channel', 'scope', 'subject', 'group', 'metric', 'unit')
    numeric = [c for c in row.index if c not in labels]
    row[numeric] = pd.to_numeric(row[numeric], errors='coerce')
    return row


def _pooled_any_scope(summary: pd.DataFrame, family: str, metric: str, group: str = 'all',
                      channel: str = 'vector') -> Optional[pd.Series]:
    """`_pooled` preferring the mocap scope and falling back to the full record.

    The preference is not arbitrary: the mocap scope is the one every family can be read on, so
    quoting it keeps a by-joint table comparable across families. But the segment family's honest
    scope is the whole record — nothing per-sample there comes from mocap — and on a dataset whose
    trials are fully valid the mocap scope is not emitted at all, so a strict `scope='mocap'`
    lookup silently drops the family from the table.
    """
    for scope in SCOPES:
        row = _pooled(summary, family, metric, group, scope, channel)
        if row is not None:
            return row
    return None


def _reduction(proj: Optional[pd.Series], raw: Optional[pd.Series]) -> float:
    if proj is None or raw is None or not raw['p50']:
        return np.nan
    return 100.0 * (1.0 - proj['p50'] / raw['p50'])


def report_headline(spec: DatasetSpec, summary: pd.DataFrame) -> None:
    _header(1, "THE THREE AGREEMENTS", "median error of the projected signal against each "
                                       "family's own truth, beside the UNPROJECTED baseline on "
                                       "the same truth and the same samples")
    rows = []
    for family in FAMILIES:
        for channel in CHANNELS:
            for scope in SCOPES:
                proj = _pooled(summary, family, 'err_proj', scope=scope, channel=channel)
                raw = _pooled(summary, family, 'err_raw', scope=scope, channel=channel)
                if proj is None:
                    continue
                signed = _pooled(summary, family, 'dnorm_proj', scope=scope, channel=channel)
                rows.append({
                    'family': family, 'channel': channel, 'scope': scope,
                    'projected': proj['p50'],
                    'unprojected': raw['p50'] if raw is not None else np.nan,
                    'reduction_%': _reduction(proj, raw),
                    'p90_proj': proj['p90'],
                    'signed_bias': signed['p50'] if signed is not None else np.nan,
                    'n_samples': int(proj['n_samples']), 'n_trials': int(proj['n_trials']),
                })
    if not rows:
        print("   No comparisons.")
        return
    print(pd.DataFrame(rows).to_string(index=False, float_format='%.3f'))
    print("\n   channel='vector' compares the full 3-vector in one common frame, so `projected` is "
          "|a_est - a_truth|\n   in m/s^2. channel='norm' compares MAGNITUDES only — "
          "| |a_est| - |a_truth| | — which is what is left\n   when there is no common frame, and "
          "is the part of a disagreement no orientation estimate can\n   ever absorb. The two are "
          "different quantities; do not read across them.")
    print("   `signed_bias` is the median of the signed magnitude gap. Near zero means the "
          "projection is not\n   systematically over- or under-shooting, which a magnitude cannot "
          "tell you.")
    print()
    for family in FAMILIES:
        if any(r['family'] == family for r in rows):
            print(f"   {family:<8} {FAMILY_LABELS[family]}")
    print("\n   scope='mocap' scores contiguous runs of valid mocap frames; scope='full' scores "
          "the whole\n   record and is available only where nothing per-sample comes from mocap. "
          "Different sample sets,\n   so read down a column within one scope, not across.")
    if not same_segment_groups(spec):
        print(f"\n   No 'segment' rows: {spec.name} carries one sensor per segment, so there is "
              f"no second\n   accelerometer on the same rigid body to project onto. IMoVE is the "
              f"one dataset here\n   that can answer that question.")


def report_marker(summary: pd.DataFrame) -> None:
    _header(2, "MARKER AGREEMENT",
            "|a - a_mocap| at the joint centre, pooled over every subject, trial and sample")
    proj, raw = _pooled(summary, 'marker', 'err_proj'), _pooled(summary, 'marker', 'err_raw')
    if proj is None or raw is None:
        print("   No marker comparisons.")
        return
    columns = ['p25', 'p50', 'p75', 'p90', 'p99', 'mean']
    # astype(float): a row sliced out of the summary carries the string columns too, so the Series
    # comes back as object dtype and to_string's float_format would be handed strings.
    table = pd.DataFrame({'projected': proj[columns], 'unprojected': raw[columns]}).T.astype(float)
    print(table.to_string(float_format='%.3f'))
    print(f"\n   Median error {raw['p50']:.3f} -> {proj['p50']:.3f} m/s^2, "
          f"{_reduction(proj, raw):.1f}% LOWER than unprojected, "
          f"n={int(proj['n_samples']):,} samples over {int(proj['n_trials'])} trials")

    alt = _pooled(summary, 'marker', 'err_proj_alt')
    if alt is not None:
        print(f"   Against the SHARED (midpoint) joint centre, as the filter's acc oracle uses: "
              f"median {alt['p50']:.3f} m/s^2 — larger by whatever the two segments disagree "
              f"about.")
    nofilt = _pooled(summary, 'marker', 'err_proj_nofilt')
    if nofilt is not None:
        print(f"   Unfiltered, same pairing: median {nofilt['p50']:.3f} m/s^2. The "
              f"{LOWPASS_CUTOFF_HZ:.0f} Hz low-pass accounts for "
              f"{nofilt['p50'] - proj['p50']:.3f} m/s^2 of it, which section 8 attributes to "
              f"mocap\n   double-differentiation noise rather than to projection error.")
    scale = _pooled(summary, 'marker', 'err_proj')
    lin = summary[(summary['subject'] == 'all') & (summary['family'] == 'marker')
                  & (summary['metric'] == 'corr_norm') & (summary['group'] == 'all')]
    if not lin.empty and scale is not None:
        correction = float(lin.iloc[0]['p50'])
        print(f"   The projection moved the signal by a median {correction:.3f} m/s^2 and the "
              f"residual is {proj['p50']:.3f}, i.e. it got "
              f"{100 * (1 - proj['p50'] / correction):.0f}% of what it added right.")


def report_paired_cells(stats: pd.DataFrame) -> None:
    _header(3, "IS IT BETTER, CELL BY CELL?",
            "one paired comparison per subject x trial x comparison, on that cell's own rms — so a "
            "pooled result cannot be carried by a few long trials")
    if stats.empty:
        print("   No stats table.")
        return
    for family in FAMILIES:
        for channel in CHANNELS:
            for scope in SCOPES:
                cells = stats[(stats['family'] == family) & (stats['scope'] == scope)
                              & (stats['channel'] == channel)][['rms_before', 'rms_raw']].dropna()
                if len(cells) < 2:
                    continue
                improved = int((cells['rms_before'] < cells['rms_raw']).sum())
                ratio = (cells['rms_before'] / cells['rms_raw']).replace(
                    [np.inf, -np.inf], np.nan).dropna()
                line = (f"   {family:<8} {channel:<6} scope={scope:<5} lower in "
                        f"{improved}/{len(cells)} cells ({100 * improved / len(cells):5.1f}%); "
                        f"ratio median {ratio.median():.3f} "
                        f"(IQR {ratio.quantile(0.25):.3f}-{ratio.quantile(0.75):.3f}, "
                        f"worst {ratio.max():.3f})")
                # Wilcoxon signed-rank rather than a t-test because these are rms values of skewed
                # magnitudes, and PAIRED rather than pooled because the two numbers in a cell come
                # from the same sensor, samples and truth — only the projection differs, which is
                # the whole point of the pairing.
                if len(cells) >= 6:
                    _, p_value = wilcoxon(cells['rms_before'], cells['rms_raw'])
                    line += f", Wilcoxon p={p_value:.1e}"
                print(line)


def report_direction(summary: pd.DataFrame) -> None:
    _header(4, "DIRECTION ERROR",
            "angle between the two acceleration vectors — what a filter's measurement update "
            "actually responds to, and blind to the magnitude error the norm channel measures")
    columns = ['p25', 'p50', 'p75', 'p90', 'p99']
    rows = []
    for family in FAMILIES:
        proj, raw = _pooled(summary, family, 'ang_proj'), _pooled(summary, family, 'ang_raw')
        if proj is None or raw is None:
            continue
        rows.append({'family': family, **{f'proj_{c}': proj[c] for c in columns},
                     'raw_p50': raw['p50'], 'reduction_%': _reduction(proj, raw)})
    if not rows:
        print("   No comparisons.")
        return
    print(pd.DataFrame(rows).to_string(index=False, float_format='%.3f'))
    print("\n   Angles are only defined where a common frame exists, so these are scope='mocap' "
          "rows;\n   the scope='full' rows carry no direction (see `_norm_only_vectors`).")


def report_by_group(spec: DatasetSpec, summary: pd.DataFrame) -> None:
    _header(5, "BY JOINT, AND BY SEGMENT",
            "median |a - truth| projected vs unprojected, per anatomical group, with the lever "
            "arm each projection spans")
    for family in FAMILIES:
        rows = []
        groups = sorted({g for g in summary[(summary['family'] == family)
                                            & (summary['channel'] == 'vector')]['group'].unique()
                         if g != 'all'})
        for group in groups:
            proj = _pooled_any_scope(summary, family, 'err_proj', group)
            raw = _pooled_any_scope(summary, family, 'err_raw', group)
            if proj is None or raw is None:
                continue
            corr = _pooled_any_scope(summary, family, 'corr_norm', group)
            rows.append({'group': group, 'projected': proj['p50'], 'unprojected': raw['p50'],
                         'reduction_%': _reduction(proj, raw),
                         'median_correction': corr['p50'] if corr is not None else np.nan,
                         'n_samples': int(proj['n_samples'])})
        if not rows:
            continue
        print(f"\n   family={family}  ({FAMILY_LABELS[family]})")
        print(pd.DataFrame(rows).to_string(index=False, float_format='%.3f'))
    print("\n   Read `reduction_%` against `median_correction`: a group whose correction is small "
          "cannot\n   show a large reduction however good the projection is, because there was "
          "nothing to fix.")


def report_cross_segment(stats: pd.DataFrame, summary: pd.DataFrame) -> None:
    _header(6, "DO THE TWO SEGMENTS AGREE WITH EACH OTHER?",
            "the parent and child sensors project to ONE point, so their projected readings are "
            "one vector in two frames — and that identity IS the relative filter's measurement")
    subset = stats[(stats['family'] == 'joint') & (stats['scope'] == 'mocap')
                   & (stats['channel'] == 'vector')]
    if subset.empty:
        print("   No cross-segment comparisons on the mocap scope.")
    else:
        columns = ['rms_before', 'rms_raw', 'rms_after', 'align_angle_deg', 'r2_proj',
                   'median_absdnorm_proj', 'median_absdnorm_raw', 'sep_m']
        print(subset[[c for c in columns if c in subset]].astype(float).describe()
              .loc[['mean', '50%', 'min', 'max']].to_string(float_format='%.3f'))
        print(f"\n   Median rms disagreement {subset['rms_raw'].median():.3f} -> "
              f"{subset['rms_before'].median():.3f} m/s^2 when both sides are projected to the "
              f"shared centre\n   ({100 * (1 - subset['rms_before'].median() / subset['rms_raw'].median()):.1f}% "
              f"lower), over a median sensor separation of "
              f"{100 * subset['sep_m'].median():.1f} cm.")
        print(f"   Removing a constant relative misalignment of "
              f"{subset['align_angle_deg'].median():.2f} deg takes it to "
              f"{subset['rms_after'].median():.3f} m/s^2, so the two frames' residual\n   "
              f"alignment is {'' if subset['rms_after'].median() < 0.8 * subset['rms_before'].median() else 'NOT '}"
              f"a substantial part of what is left.")

    full = _pooled(summary, 'joint', 'err_proj', scope='full', channel='norm')
    full_raw = _pooled(summary, 'joint', 'err_raw', scope='full', channel='norm')
    if full is not None and full_raw is not None:
        signed = _pooled(summary, 'joint', 'dnorm_proj', scope='full', channel='norm')
        print(f"\n   THE MOCAP-FREE CHANNEL, over the whole record rather than the valid window: "
              f"| |a_parent| - |a_child| |\n   median {full_raw['p50']:.3f} -> {full['p50']:.3f} "
              f"m/s^2 ({_reduction(full, full_raw):.1f}% lower), "
              f"n={int(full['n_samples']):,} samples"
              + (f", signed median {signed['p50']:+.3f}." if signed is not None else "."))
        print("   This channel uses the two offsets and nothing else — no mocap orientation, no "
              "differentiated\n   markers — so it is the one result here that a deployed system "
              "could reproduce. It is also\n   what Seel's estimator minimizes; "
              "experiments/inertial_joint_center.py fits the offsets to it.")
    print("\n   BLIND SPOT, stated because the number above is the most flattering in this file: "
          "this family\n   cannot see a common-mode offset error. If both segments' offsets are "
          "wrong in a way that moves\n   the shared point together, the two projections still "
          "agree exactly. Section 2 is what covers\n   that, and it is why neither section is "
          "quoted alone.")


def report_consistency(stats: pd.DataFrame) -> None:
    """Do the two mocap-referenced families predict the mocap-free one?

    The check that makes the three questions one experiment rather than three, and the only place
    the families constrain each other. If the parent's and child's marker residuals are e_p and
    e_c, then their disagreement with EACH OTHER is e_p - e_c, so

        rms(pair)  =  sqrt( rms(e_p)^2 + rms(e_c)^2 )        if e_p and e_c are independent
        rms(pair)  <<  that                                  if they share a term that CANCELS
        rms(pair)  >>  that                                  if each marker residual is itself
                                                             reduced by a term the pair keeps

    All three regimes occur in principle and the third is the one this repository actually lands
    in, so it is worth naming the mechanisms rather than only reporting the number:

      * A ratio BELOW 1 means the mocap reference's own noise is a common term. It enters both
        marker residuals and cancels from the pair, so section 2 is an upper bound by that factor.
      * A ratio ABOVE 1 means each marker residual is OPTIMISTIC — the estimate and its own
        reference share something that the pair comparison does not. Two candidates, and they are
        separable:
          (a) THE JOINT MODEL. The marker family compares each segment against its OWN implied
              joint centre; the pair comparison forces both onto one point. So the parent/child
              disagreement — the joint fit residual, soft tissue, a translating knee — is excluded
              from the marker residual by construction and included in full in the pair. The
              `rms_alt` column measures exactly this: it is the marker residual against the SHARED
              midpoint, so a prediction built from it should recover the measured pair
              disagreement if this is the whole story.
          (b) SENSOR-TO-SEGMENT ALIGNMENT. `assembly.align_world_to_imu` fits each plate's constant
              orientation offset from THAT PLATE'S OWN GYRO against its marker cluster. So a
              segment's mocap orientation is partly chosen to agree with the IMU it is then
              compared against, and the marker family is not fully independent of its own estimate.
              The pair comparison uses two independently-fitted plates, so it keeps what each
              marker residual absorbed.

    Both columns are printed so the reader can see which one carries it, instead of being told.
    """
    _header(7, "DO THE FAMILIES AGREE WITH EACH OTHER?",
            "the cross-segment disagreement predicted from the two marker residuals, against the "
            "measured one — the only place the three families constrain each other")
    marker = stats[(stats['family'] == 'marker') & (stats['channel'] == 'vector')]
    joint = stats[(stats['family'] == 'joint') & (stats['scope'] == 'mocap')
                  & (stats['channel'] == 'vector')]
    if marker.empty or joint.empty:
        print("   Needs both the marker and cross-segment families; one of them is absent.")
        return

    keys = ['subject', 'trial', 'group']
    columns = {}
    for label, source in (('own', 'rms_before'), ('shared', 'rms_alt')):
        if source not in marker.columns:
            continue
        sides = marker.pivot_table(index=keys, columns='variant', values=source, observed=True)
        if not {'parent', 'child'}.issubset(sides.columns):
            continue
        columns[f'predicted_{label}'] = np.sqrt(sides['parent'] ** 2 + sides['child'] ** 2)
    if 'predicted_own' not in columns:
        print("   Needs both segments of a joint in the marker family.")
        return
    columns['measured'] = joint.set_index(keys)['rms_before']
    paired = pd.DataFrame(columns).dropna(subset=['predicted_own', 'measured'])
    if paired.empty:
        print("   No joint has both families on the same trial.")
        return

    paired['ratio_own'] = paired['measured'] / paired['predicted_own']
    if 'predicted_shared' in paired:
        paired['ratio_shared'] = paired['measured'] / paired['predicted_shared']
    aggregations = {'n': ('measured', 'size')}
    aggregations.update({column: (column, 'median') for column in paired.columns})
    print(paired.groupby('group', observed=True).agg(**aggregations)
          .to_string(float_format='%.3f'))

    own = float(paired['ratio_own'].median())
    print(f"\n   Pooled ratio measured/predicted = {own:.3f} over {len(paired)} joint-trials, "
          f"using each segment's OWN\n   implied joint centre as its reference.")
    if own > 1.15 and 'ratio_shared' in paired:
        shared = float(paired['ratio_shared'].median())
        print(f"   ABOVE 1, so each marker residual is OPTIMISTIC relative to what the two "
              f"segments' projections\n   actually agree on. Re-predicted from the SHARED-midpoint "
              f"reference the ratio is {shared:.3f}, so\n   "
              + ("the joint model accounts for most of the gap: the marker family excludes the "
                 "parent/child\n   disagreement by comparing each segment against its own centre, "
                 "and the pair comparison cannot."
                 if abs(shared - 1.0) < abs(own - 1.0) * 0.6 else
                 "the joint model does NOT account for it. What is left is the alignment path: "
                 "each plate's\n   mocap orientation was fitted from that plate's own gyro "
                 "(assembly.align_world_to_imu), so the\n   marker reference is not independent of "
                 "the IMU it is compared against."))
        print("   Either way the CROSS-SEGMENT number is the more conservative of the two, and it "
              "is the one a\n   relative filter actually pays.")
    elif own < 0.85:
        print("   BELOW 1, so a substantial part of each marker residual CANCELS between the two "
              "segments — it is\n   common to both and therefore a property of the reference "
              "rather than of either projection.\n   Section 2's residual is an upper bound by "
              "roughly this factor.")
    else:
        print("   Close to 1, so the two segments' marker residuals are essentially INDEPENDENT. "
              "The marker\n   residual is not dominated by a reference noise common to both "
              "segments; each segment's\n   residual is its own offset, its own alignment and its "
              "own cluster's noise.")
    print("   Neither family can compute this alone. It is the reason both are run rather than "
          "whichever one\n   gives the more flattering number.")


def _report_frame_choice(partner: pd.DataFrame) -> None:
    """Which sensor-to-sensor transform this family uses, and WHAT ELSE the gyro mismatch is.

    Two questions that look like one and are not, which is why they are answered in one place:

      1. Is the frame transform the problem? Reported by `C_mocap_vs_gyro_deg` and
         `rms_mocap_frame` — the disagreement between the two available transforms and what the
         mocap-composed one costs. On IMoVE the answer is NO: they differ by ~1 deg and switching
         moves the residual a few percent. The gyro fit is still the one used, because it is better
         and because it needs no mocap, but it does not explain the mismatch.
      2. Then what does? `gyro_sync_diagnostic` refits the transform at every sample shift in
         +-5 and reports the best. A large improvement at a nonzero lag is RELATIVE TIMING between
         two independently-clocked devices; a mismatch that barely moves is a non-rigid segment or
         a per-device gyro calibration difference. Only the first is fixable, and only in the build
         layer.

    This section exists because the first hypothesis here was that the mocap-composed transform
    explained a 20-32% gyro mismatch, and the data said otherwise. Both numbers are printed so the
    next reader does not have to re-derive that.
    """
    columns = ['C_mocap_vs_gyro_deg', 'rms_mocap_frame', 'rms_before']
    if not set(columns).issubset(partner.columns) or partner[columns].dropna().empty:
        return
    rows = partner.dropna(subset=columns)
    disagreement = float(rows['C_mocap_vs_gyro_deg'].median())
    mocap_frame = float(rows['rms_mocap_frame'].median())
    gyro_frame = float(rows['rms_before'].median())
    print(f"\n   THE FRAME TRANSFORM IS FITTED FROM THE GYROS, NOT COMPOSED FROM MOCAP. "
          f"omega_A = C omega_B holds\n   exactly on a rigid segment, so the gyros DETERMINE C "
          f"directly, over the whole record and with no\n   mocap involved; mocap determines it "
          f"only as the ratio of two independently fitted "
          f"sensor-to-segment\n   rotations (assembly.align_world_to_imu solves each plate's from "
          f"that plate's own gyro against the\n   marker cluster), so it inherits both fits' "
          f"errors. Fitting the transform on the GYROS and scoring\n   the ACCELEROMETERS is not "
          f"circular — different measurements, and identifying one constant\n   sensor-to-sensor "
          f"rotation is exactly the calibration a two-IMU-per-segment mounting would do.")
    print(f"   The two transforms disagree by a median {disagreement:.2f} deg and the "
          f"mocap-composed one reads\n   {mocap_frame:.3f} m/s^2 against {gyro_frame:.3f} "
          f"({100 * (mocap_frame / gyro_frame - 1):+.0f}%). So THE FRAME IS NOT WHAT LIMITS THIS "
          f"FAMILY — which was\n   the first hypothesis here, and the numbers refused it.")

    if 'gyro_lag_improvement_pct' in rows and rows['gyro_lag_improvement_pct'].notna().any():
        lag_rows = rows.dropna(subset=['gyro_lag_improvement_pct'])
        improvement = float(lag_rows['gyro_lag_improvement_pct'].median())
        table = lag_rows.groupby('variant', observed=True).agg(
            n=('gyro_best_lag_ms', 'size'),
            best_lag_ms=('gyro_best_lag_ms', 'median'),
            nonzero_lag_frac=('gyro_best_lag_samples', lambda s: float((s != 0).mean())),
            mismatch_zero_lag=('gyro_mismatch_at_zero_lag', 'median'),
            mismatch_best_lag=('gyro_mismatch_at_best_lag', 'median'),
            improvement_pct=('gyro_lag_improvement_pct', 'median'))
        print("\n   SO WHAT IS THE GYRO MISMATCH? The transform is refitted at every sample shift "
              "in +-5 and the\n   best one reported. Two devices with independent clocks disagree "
              "by |domega/dt| dt whatever C is:")
        print(table.to_string(float_format='%.3f'))
        if improvement > 25.0:
            print(f"\n   A median {improvement:.0f}% of the mismatch is removed by a time shift, so "
                  f"much of it is RELATIVE\n   TIMING between two independently-clocked devices, "
                  f"not a non-rigid segment. IMoVE's build syncs\n   the whole inertial record to "
                  f"mocap on the PELVIS; nothing aligns two devices on one segment to\n   each "
                  f"other. That is a BUILD-layer correction, and this module measures it rather "
                  f"than applying\n   it — a second alignment living in an analysis module is "
                  f"worse than a measured floor.")
        else:
            print(f"\n   Only {improvement:.0f}% of the mismatch is removed by the best time shift, "
                  f"so it is NOT mostly relative\n   timing. What is left is a non-rigid segment "
                  f"(soft tissue, mounting compliance) or a per-device\n   gyro scale or bias "
                  f"difference — neither of which any shift or rotation removes, and both of "
                  f"which\n   are a genuine floor under this family rather than an artifact of it.")


def report_same_segment(spec: DatasetSpec, stats: pd.DataFrame, summary: pd.DataFrame) -> None:
    _header(8, "DO TWO SENSORS ON ONE SEGMENT PROJECT TO EACH OTHER?",
            "no differentiated markers anywhere: the truth is another accelerometer on the same "
            "rigid body, so this is band-limited by the sensors instead of by the reference")
    subset = stats[stats['family'] == 'segment']
    if subset.empty:
        print(f"   Not measurable on {spec.name}: one sensor per segment. "
              f"{'IMoVE' if spec.name != 'imove' else 'This dataset'} carries three on each "
              f"thigh and shank, which is what this family needs.")
        return

    partner = subset[subset['target_kind'] == 'partner']
    if not partner.empty:
        table = partner.groupby('variant', observed=True).agg(
            n=('rms_before', 'size'),
            sep_cm=('sep_m', lambda s: 100 * s.median()),
            rms_proj=('rms_before', 'median'),
            rms_raw=('rms_raw', 'median'),
            rms_aligned=('rms_after', 'median'),
            gyro_mismatch=('median_gyro_mismatch', 'median'),
            gyro_norm=('median_gyro_norm', 'median'))
        table['reduction_%'] = 100 * (1 - table['rms_proj'] / table['rms_raw'])
        table['gyro_mismatch_%'] = 100 * table['gyro_mismatch'] / table['gyro_norm']
        print("   TARGET = THE PARTNER SENSOR'S OWN SITE (truth is its reading):")
        print(table.to_string(float_format='%.3f'))
        print("\n   `gyro_mismatch` is |omega_A - omega_B| in the same frame, in rad/s. Two "
              "sensors on ONE rigid\n   segment measure the same angular velocity, so this is the "
              "rigidity floor: whatever it reads,\n   the acceleration agreement cannot beat it, "
              "because a dw disagreement puts ~|dw||omega||r| into\n   the centripetal term "
              "alone. `gyro_mismatch_%` expresses it as a fraction of the signal.")
        _report_frame_choice(partner)

    joint_targets = subset[subset['target_kind'].astype(str).str.startswith('joint')]
    if not joint_targets.empty:
        table = joint_targets.groupby('variant', observed=True).agg(
            n=('rms_before', 'size'),
            arm_source_cm=('arm_source_m', lambda s: 100 * s.median()),
            arm_target_cm=('arm_target_m', lambda s: 100 * s.median()),
            rms_proj=('rms_before', 'median'),
            rms_raw=('rms_raw', 'median'))
        table['reduction_%'] = 100 * (1 - table['rms_proj'] / table['rms_raw'])
        print("\n   TARGET = A SHARED JOINT CENTRE (no truth; the metric is mutual agreement over "
              "a long arm):")
        print(table.to_string(float_format='%.3f'))
        print("   Compare `reduction_%` here against the partner rows above: the arms are 2-4x "
              "longer, so the\n   projection is doing far more work, and whether it still agrees "
              "is the question the pipeline's\n   own geometry asks.")

    geometry = partner if not partner.empty else subset
    if 'r_inertial_error_mm' in geometry and geometry['r_inertial_error_mm'].notna().any():
        rows = geometry.dropna(subset=['r_inertial_error_mm'])
        print("\n   THE GEOMETRY, RE-SOLVED FROM THE IMUs ALONE. Equation (3) is linear in the "
              "separation, so it\n   is a 3-parameter least squares with no optimizer:")
        # Columns with nothing in them are dropped rather than described: `describe` on an all-NaN
        # column reduces an empty slice and warns, and a row of NaN in this table would read as a
        # measurement that came out zero.
        wanted = ['r_inertial_error_mm', 'attenuation', 'rms_at_inertial', 'rms_before',
                  'design_condition', 'r_spread_mm', 'C_spread_max_deg']
        available = [c for c in wanted if c in rows and rows[c].notna().any()]
        print(rows[available].astype(float).describe().loc[['mean', '50%', 'min', 'max']]
              .to_string(float_format='%.3f'))
        median_at_fit = rows['rms_at_inertial'].median()
        median_at_mocap = rows['rms_before'].median()
        print(f"\n   At the inertially-preferred separation the residual is "
              f"{median_at_fit:.3f} m/s^2 against "
              f"{median_at_mocap:.3f} at the mocap one,\n   so "
              f"{'the mocap geometry is most of what is left' if median_at_fit < 0.7 * median_at_mocap else 'the mocap geometry is NOT the limiting term'}: "
              f"the projection physics reaches\n   {median_at_fit:.3f} m/s^2 when the offset is "
              f"not holding it back. `attenuation` is |r_inertial| / |r_mocap|;\n   it is below 1 "
              f"by construction — the design matrix is built from a differentiated gyro, so its "
              f"noise\n   is errors-in-variables and biases the offset toward zero rather than "
              f"merely scattering it.")
        print(f"   `r_spread_mm` ({rows['r_spread_mm'].median():.2f} mm) and `C_spread_max_deg` "
              f"({rows['C_spread_max_deg'].median():.3f} deg) are ~0 BY CONSTRUCTION on this "
              f"dataset,\n   not by measurement: all placements on a segment come from one marker "
              f"cluster, so the separation\n   is exactly the difference of two fitted sensor "
              f"offsets. They are computed so that a dataset where\n   that stops holding says "
              f"so.")

    full = _pooled(summary, 'segment', 'err_proj', scope='full')
    mocap = _pooled(summary, 'segment', 'err_proj', scope='mocap')
    if full is not None and mocap is not None:
        print(f"\n   Scored over the WHOLE record (median {full['p50']:.3f} m/s^2, "
              f"n={int(full['n_samples']):,}) and over the mocap\n   window alone "
              f"(median {mocap['p50']:.3f}, n={int(mocap['n_samples']):,}). Everything mocap "
              f"supplies to this family is a\n   CONSTANT, so the full record is the honest "
              f"scope; the narrower one is here only so the three\n   families can be read side "
              f"by side on one sample set.")


def report_alignment(stats: pd.DataFrame) -> None:
    _header(9, "THE MISALIGNMENT FLOOR",
            "one constant rotation fitted per comparison (Kabsch); `rms_after` is what remains "
            "once it is removed")
    if stats.empty:
        print("   No stats table.")
        return
    vector = stats[stats['channel'] == 'vector']
    if vector.empty:
        print("   No vector-channel comparisons; a magnitude comparison has no frame to misalign.")
        return
    table = vector.groupby(['family', 'scope'], observed=True).agg(
        n=('rms_before', 'size'),
        align_angle_deg=('align_angle_deg', 'median'),
        rms_before=('rms_before', 'median'),
        rms_after=('rms_after', 'median'),
        rms_raw=('rms_raw', 'median'),
        r2_proj=('r2_proj', 'median'))
    table['explained_%'] = 100 * (1 - (table['rms_after'] / table['rms_before']) ** 2)
    print(table.to_string(float_format='%.3f'))
    gravity = float(np.linalg.norm(EXPECTED_GRAVITY))
    mocap_rows = vector[vector['scope'] == 'mocap']
    if not mocap_rows.empty:
        angle = float(mocap_rows['align_angle_deg'].median())
        print(f"\n   Median fitted misalignment {angle:.2f} deg. At |g| = {gravity:.2f} m/s^2 that "
              f"tilts the gravity\n   component by ~{np.radians(angle) * gravity:.3f} m/s^2, but "
              f"it adds in QUADRATURE with everything else in the\n   residual, which is why "
              f"`explained_%` is the column to read and not the angle.")
    print("   Per-sample error columns in every figure are NOT rotation-corrected; the fit is "
          "reported so the\n   floor is visible, not subtracted. No choice of frame removes it — "
          "rotating both signals by the\n   same matrix leaves |estimate - truth| exactly "
          "unchanged.")


def report_cutoff_sweep(sweep: pd.DataFrame) -> None:
    _header(10, "HOW MUCH OF THE RESIDUAL IS THE TRUTH'S OWN NOISE?",
            "median residual at each low-pass cutoff, against what the truth's excess in-band "
            "power accounts for by itself")
    if sweep.empty:
        print("   No sweep table.")
        return
    for family in FAMILIES:
        subset = sweep[sweep['family'] == family]
        if subset.empty:
            continue
        table = subset.groupby('cutoff_hz').agg(
            n=('median_err_proj', 'size'),
            median_err_proj=('median_err_proj', 'median'),
            median_err_raw=('median_err_raw', 'median'),
            rms_err_proj=('rms_err_proj', 'median'),
            ref_excess_rms=('ref_excess_rms', 'median'))
        table['explained_%'] = 100 * table['ref_excess_rms'] / table['rms_err_proj']
        table['proj_vs_raw_%'] = 100 * (1 - table['median_err_proj'] / table['median_err_raw'])
        print(f"\n   family={family}")
        print(table.to_string(float_format='%.3f'))

    marker = sweep[sweep['family'] == 'marker']
    if not marker.empty:
        # Checked for presence BEFORE taking a median: NOISE_BAND_HZ starts at 25 Hz, which is
        # above Nyquist on IMoVE's 40 Hz trials, so the whole column is legitimately NaN there and
        # a bare .median() reduces an empty slice and warns.
        measured = marker['noise_floor_psd'].dropna()
        if not measured.empty:
            print(f"\n   Mocap noise floor over {NOISE_BAND_HZ[0]:.0f}-{NOISE_BAND_HZ[1]:.0f} Hz: "
                  f"{measured.median():.3f} (m/s^2)^2/Hz per axis, flat — a\n   differentiation "
                  f"noise floor, not signal.")
        else:
            print(f"\n   No mocap noise floor: {NOISE_BAND_HZ[0]:.0f}-{NOISE_BAND_HZ[1]:.0f} Hz is "
                  f"above Nyquist on every trial here, so the band the\n   floor is measured in "
                  f"does not exist. The sweep's own rows are still the evidence for the cutoff.")
        print("   `explained_%` is how much of the projected residual the TRUTH's own in-band "
              "noise accounts for.\n   Read UP the marker table: as the cutoff rises, more of "
              "what looks like projection error is the\n   reference. That is the quantitative "
              "reason LOWPASS_CUTOFF_HZ is "
              f"{LOWPASS_CUTOFF_HZ:.0f} and not 15.")
        print("   In the agreement families the truth is an accelerometer, so `explained_%` there "
              "is a statement\n   about the two sensors' noise floors matching, NOT about mocap — "
              "and those families' residuals\n   do not blow up with the cutoff, which is the "
              "cleanest evidence that the marker family's does\n   for reasons that are about the "
              "reference.")
        print("   `proj_vs_raw_%` is the reduction against the unprojected baseline. It is the "
              "one column no\n   reference noise can manufacture: both signals are compared to "
              "the same truth, so shared noise\n   inflates both and cancels from the gap.")


def report_gyro_method(gyro: pd.DataFrame) -> None:
    _header(11, "IS THE GYRO DERIVATIVE THE LIMIT?",
            "same offset, same truth, same low-pass, same samples — only how alpha = d(omega)/dt "
            "was estimated differs")
    if gyro.empty:
        print("   No gyro_method table.")
        return
    present = [m for m in GYRO_METHODS if m in set(gyro['gyro_method'].astype(str))]
    for family, channel in ((f, c) for f in FAMILIES for c in CHANNELS):
        subset = gyro[(gyro['family'] == family) & (gyro['channel'] == channel)]
        if subset.empty:
            continue
        table = subset.groupby('gyro_method', observed=True).agg(
            median_err=('median_err', 'median'),
            p90_err=('p90_err', 'median'),
            rms_err=('rms_err', 'median'),
            median_ang=('median_ang', 'median'),
            median_absdnorm=('median_absdnorm', 'median')).reindex(present).dropna(how='all')
        if PRIMARY_GYRO_METHOD not in table.index:
            continue
        table['vs_primary_%'] = 100 * (table['median_err'] / table.loc[PRIMARY_GYRO_METHOD,
                                                                       'median_err'] - 1)
        print(f"\n   family={family}  channel={channel}")
        print(table.to_string(float_format='%.4f'))

        # Paired per cell, since the alternatives are compared on identical inputs and a pooled
        # median could hide a scheme that is better on most cells and much worse on a few.
        # observed=True and a whole-frame dropna: the index levels are categoricals, so without
        # both, pivot_table expands to the full cartesian product of categories and the all-NaN
        # rows it invents would misalign the paired test.
        wide = subset.pivot_table(
            index=['subject', 'trial', 'group', 'variant', 'target_kind', 'scope'],
            columns='gyro_method', values='median_err', observed=True).dropna()
        for method in present:
            if method == PRIMARY_GYRO_METHOD or method not in wide.columns or wide.empty:
                continue
            ratio = (wide[method] / wide[PRIMARY_GYRO_METHOD]).dropna()
            if ratio.empty:
                continue
            line = (f"      {method:<12} better in {int((ratio < 1).sum())}/{len(ratio)} cells; "
                    f"ratio vs {PRIMARY_GYRO_METHOD} median {ratio.median():.3f} "
                    f"(IQR {ratio.quantile(0.25):.3f}-{ratio.quantile(0.75):.3f})")
            if len(ratio) >= 6:
                _, p_value = wilcoxon(wide[method], wide[PRIMARY_GYRO_METHOD])
                line += f", Wilcoxon p={p_value:.1e}"
            print(line)
    print("\n   The MARKER rows compare the schemes in the one band where they nearly agree: they "
          "differ almost\n   entirely above the 6 Hz cutoff, which the mocap reference's noise "
          "floor forced down. The agreement\n   families have no such constraint — their truth is "
          "an accelerometer — so they are where this table\n   can separate the schemes, and "
          "where a scheme that wins on mocap and loses here was winning on\n   the reference's "
          "noise.")
    print("   None of these rows licenses a choice of derivative for the pipeline. polyfit's "
          "kernel reads 90 ms\n   of FUTURE gyro against central's one sample, so it is not "
          "solving the same problem; "
          f"'{PRIMARY_GYRO_METHOD}'\n   stays the default because it is the only causal one.")


def report_error_vs_correction(samples: pd.DataFrame) -> None:
    _header(12, "ERROR vs THE SIZE OF THE CORRECTION",
            "median |estimate - truth| in bins of how much the projection moved the comparison — "
            "how the residual grows with the work the rigid-body terms were asked to do")
    if samples.empty:
        print("   No samples.")
        return
    edges = [0, 0.5, 1, 2, 5, 10, 20, 50, np.inf]
    labels = ['0-0.5', '0.5-1', '1-2', '2-5', '5-10', '10-20', '20-50', '50+']
    for family in FAMILIES:
        subset = samples[(samples['family'] == family) & (samples['channel'] == 'vector')]
        if subset.empty:
            continue
        binned = subset.assign(bin=pd.cut(subset['corr_norm'], bins=edges, labels=labels))
        table = binned.groupby('bin', observed=True).agg(
            n_samples=('err_proj', 'size'),
            median_corr=('corr_norm', 'median'),
            median_err_proj=('err_proj', 'median'),
            median_err_raw=('err_raw', 'median'))
        table['err_as_%_of_corr'] = 100 * table['median_err_proj'] / table['median_corr']
        print(f"\n   family={family}")
        print(table.to_string(float_format='%.3f'))
    print("\n   The last column is the residual as a percentage of the correction applied, i.e. "
          "how much of what\n   the projection added it got wrong. Well under 100% where the "
          "correction is large means the\n   rigid-body terms are mostly right.")
    print("   It EXCEEDS 100% in the smallest bins, and that is not the projection failing: below "
          "~1 m/s^2 of\n   correction the residual is the error floor both signals share (truth "
          "noise, sensor noise,\n   misalignment), which is why median_err_raw is nearly equal to "
          "median_err_proj in those rows.")


def report_lever_arm(stats: pd.DataFrame) -> None:
    _header(13, "THE LEVER ARM",
            "every comparison in this file, binned by how far the projection had to reach — the "
            "one axis all three families share")
    if stats.empty or 'sep_m' not in stats:
        print("   No stats table.")
        return
    usable = stats[np.isfinite(stats['sep_m']) & (stats['sep_m'] > 0)
                   & (stats['channel'] == 'vector')]
    if usable.empty:
        print("   No comparisons with a measured separation.")
        return
    edges = [0, 0.05, 0.10, 0.15, 0.20, 0.30, np.inf]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-30', '30+']
    binned = usable.assign(bin=pd.cut(usable['sep_m'], bins=edges, labels=labels))
    table = binned.groupby(['family', 'bin'], observed=True).agg(
        n=('rms_before', 'size'),
        sep_cm=('sep_m', lambda s: 100 * s.median()),
        rms_proj=('rms_before', 'median'),
        rms_raw=('rms_raw', 'median'))
    table['reduction_%'] = 100 * (1 - table['rms_proj'] / table['rms_raw'])
    print(table.to_string(float_format='%.3f'))
    print("\n   Bins are centimetres of separation between the two things being compared: the "
          "offset for the\n   marker family, the two sensors' separation for the other two. "
          "`rms_raw` should GROW with the bin —\n   that is the rigid-body term the projection "
          "exists to remove — and `rms_proj` staying flat while it\n   does is the claim this "
          "whole file makes, in one table.")


def print_report(spec: DatasetSpec, summary: pd.DataFrame, samples: pd.DataFrame,
                 stats: pd.DataFrame, sweep: pd.DataFrame, gyro: pd.DataFrame) -> None:
    report_headline(spec, summary)
    report_marker(summary)
    report_paired_cells(stats)
    report_direction(summary)
    report_by_group(spec, summary)
    report_cross_segment(stats, summary)
    report_consistency(stats)
    report_same_segment(spec, stats, summary)
    report_alignment(stats)
    report_cutoff_sweep(sweep)
    report_gyro_method(gyro)
    report_error_vs_correction(samples)
    report_lever_arm(stats)

# ==============================================================================
# CLI
# ==============================================================================

# Columns the pooled report needs out of agreement_samples. Pushed into the parquet read: the
# per-axis Bland-Altman columns are half the table's width and only the figures want them.
REPORT_COLUMNS = ['family', 'group', 'variant', 'target_kind', 'scope', 'channel', 'err_proj',
                  'err_raw', 'ang_proj', 'ang_raw', 'dnorm_proj', 'dnorm_raw', 'absdnorm_proj',
                  'absdnorm_raw', 'corr_norm', 'err_proj_alt', 'err_proj_nofilt']


def _load_samples(dataset: str, row_keys: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    """agreement_samples across trials, with the report's columns only.

    Read column-by-column-tolerantly: `err_proj_alt` exists only where the marker family wrote it,
    and a dataset whose first trial has no marker comparisons would otherwise fail the whole read
    on a missing column.
    """
    try:
        return load_trial_table(dataset, 'agreement_samples', row_keys, REPORT_COLUMNS)
    except (KeyError, ValueError, OSError):
        return load_trial_table(dataset, 'agreement_samples', row_keys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='alborno', choices=sorted(DATASETS))
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--trials', nargs='+', default=None)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--stride', type=int, default=SAMPLE_STRIDE,
                        help="Keep every Nth sample in agreement_samples. Every scalar in "
                             "agreement_stats is computed on the full conditioned record "
                             "regardless.")
    parser.add_argument('--only-tables', nargs='+', choices=TRIAL_TABLES,
                        default=list(TRIAL_TABLES), metavar='TABLE')
    parser.add_argument('--allow-stale', action='store_true',
                        help="Read the cached trial parquets WITHOUT the freshness check. For "
                             "iterating while the build layer is changing; every artifact written "
                             "is stamped built_from_stale_cache and the numbers must be "
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
                  "every\n         artifact written will be stamped built_from_stale_cache.")
        print(f"Projecting accelerations over {len(row_keys)} trials, "
              f"{len(subjects_of(row_keys))} subject(s)...")
        state, _ = run_tracked_grid(
            row_keys, ['Subject', 'Trial'], ['project'],
            partial(_trial_worker, dataset=args.dataset, tables=args.only_tables,
                    stride=args.stride, allow_stale=args.allow_stale),
            args.workers, title=f"ACCELERATION PROJECTION — {args.dataset}")
        # Say so when the compute pass did not actually compute anything. Without this the run
        # goes straight on to load whatever tables happen to be on disk and prints a full report
        # off them, which reads exactly like a successful run of the new code.
        failures = {key: value for (key, stage), value in state.items()
                    if stage == 'project' and isinstance(value, str) and value.startswith('Failed')}
        if failures:
            reasons: Dict[str, int] = {}
            for message in failures.values():
                reason = message.split(':')[0][:70]
                reasons[reason] = reasons.get(reason, 0) + 1
            print(f"\n{len(failures)} of {len(row_keys)} trials FAILED:")
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
                print(f"  {count:4d} x {reason}")
            if len(failures) == len(row_keys):
                print("\nEvery trial failed, so nothing was written. Anything below would "
                      "describe a PREVIOUS\nrun's tables, not this one — stopping instead.")
                return 1

    print("\nLoading per-trial tables...")
    samples = _load_samples(args.dataset, row_keys)
    stats = load_trial_table(args.dataset, 'agreement_stats', row_keys)
    sweep = load_trial_table(args.dataset, 'cutoff_sweep', row_keys)
    gyro = load_trial_table(args.dataset, 'gyro_method', row_keys)
    if samples.empty:
        print(f"No results under {dataset_dir(args.dataset)}. Run without --report-only first.")
        return 1

    found = {(str(s), str(t)) for s, t in samples[['subject', 'trial']].drop_duplicates().to_numpy()}
    print(f"Found {len(found)} trial(s) across {len({s for s, _ in found})} subject(s), "
          f"{len(samples):,} samples, {len(stats):,} comparisons.")

    summary = summarize(args.dataset, samples)
    path = paths.ensure_parent(statistics_path(args.dataset))
    summary.to_parquet(path, engine='pyarrow', index=False)
    paths.write_manifest(path, constants=analysis_constants(args.dataset),
                         experiment=EXPERIMENT_NAME, n_rows=len(summary),
                         subjects=sorted({s for s, _ in found}), n_trials=len(found))
    print(f"Saved summary to {path}")

    print_report(spec, summary, samples, stats, sweep, gyro)
    print(f"\nPer-trial tables under {dataset_dir(args.dataset)}")
    print(f"Figures: python -m plotting.acceleration_projection --dataset {args.dataset}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
