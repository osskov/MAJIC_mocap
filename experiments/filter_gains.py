"""
Where the filter's gains come from, organised by HOW MUCH YOU HAVE TO KNOW to choose them.

WHAT THIS EXPERIMENT ANSWERS, AND WHAT IT DOES NOT. Most of what follows is ACAUSAL: it tunes
against the data it is scored on, using the mocap ground truth of the very trials in the
benchmark. A filter deployed on a subject who was never in a mocap lab cannot do that. So the
optimum reported in section 4 is a CEILING, not a performance claim, and quoting it as though
a shipped filter reaches it would be wrong.

The experiment is therefore built as a ladder of information (`--stage ladder`, reported in
section 6). Each rung chooses the three stds from strictly more than the rung above it:

    legacy default  NOT an information rung. The arbitrary triple this work replaced, kept
                    as a historical baseline (see LEGACY_STDS).
    ------------------------------------------------ the ladder proper starts here
    static floor    a static recording. No mocap, no motion.
    signal floor    the trial's own sensor streams. No mocap.
    ------------------------------------------------ everything below needs ground truth
    innovation      mocap: the residual the filter would see at the true state.
    swept optimum   mocap AND the scored metric. The acausal ceiling.

Every rung is scored on the same trials, joints and metric, so the only thing that differs
between them is what they were allowed to know. The distance from the best rung above the
line to the one below it is what acausal tuning is actually worth on this data -- and it is
the number to quote when someone asks whether the tuning generalises.

The report (`--stage report`) prints all six sections with the question each one answers and
the information each one required. Read it in order; the sections are arranged so that
everything usable in the field comes before anything that needs a marker.

THE TUNING HAS TWO DEGREES OF FREEDOM, NOT THREE. The relative filter's steady-state Kalman
gain depends on Q and R only through their ratio, and the kernel adds process noise as
dt^2 * Q against a measurement covariance built from R, so scaling (gyro_std, acc_std,
mag_std) by a common factor leaves the gain untouched. The only thing the common factor
changes is how fast P leaves its seed value P0 = init_orientation_std^2 * I, i.e. burn-in.
Measured on Al Borno: a 10x scaling of all three stds moves the trajectory by 0.003-0.03 deg
per joint after 10 s, against 2-14 deg inside the first 10 s. So the sweep here is over

    acc_ratio = acc_std / gyro_std        mag_ratio = mag_std / gyro_std

with gyro_std pinned at SWEEP_GYRO_REF. The third degree of freedom -- the overall scale --
is not estimated at all, because `--stage scale` shows there is little there to estimate:
scaling a whole tuning at fixed ratios leaves the pooled RMSE flat to under 1% above a FLOOR
multiplier, measured on all twelve (dataset, arm) combinations. The floor is 1e-4 to 1e-2 on
the mag_on arms and 1e-1 to 3 on the mag_off arms -- below it P never escapes its seed within
the trial and the scale stops being free. The shipped table sits at 1x, above every floor,
but the margin on some mag_off arms is one decade rather than six. It is written at a CHOSEN
scale -- gyro_std held at 0.0045, the value the pipeline shipped with, so the new numbers
stay directly comparable to the ones they replace. That choice carries no information: the
scale is inert, and 0.0045 has no known provenance (see LEGACY_STDS). Any common rescaling
of a row is the same filter.

This supersedes experiments/noise_sensitivity.py, whose grid swept all three stds
independently. That grid's 27 combos were 9 distinct tunings with 3 redundant replicas
each, and it scored replicas of the SAME tuning up to 0.022 rad apart. That spread is
burn-in: re-running the two top-ranked combos (gyro 1e-2/acc 1e-1/mag 1e-1 and
gyro 1e-3/acc 1e-2/mag 1e-2, the same ratios 10x apart) and differencing the trajectories
gives rms 0.0226 rad, against the 0.0224 rad gap the ranking put between them. The old
ranking's top six were ordering their own transient.

Two things are measured, and both are needed.

  THE INNOVATION AT THE TRUE STATE (`--stage residuals`). The filter's measurement is
  e = R_wp v_p - R_wc v_c and its assumed covariance is R_wp diag(var_p) R_wp^T +
  R_wc diag(var_c) R_wc^T. Substitute the MOCAP rotations for the filter's estimate and e
  becomes the residual the filter would see if its state were exactly right -- which is the
  definition of what R is supposed to model, measurable without running the filter at all.
  Under the filter's own isotropic per-sensor assumption the pair variance is 2*sigma^2, so
  sigma = rms(e)/sqrt(2). The same substitution gives Q: omega_imu - omega_mocap, per body.

  This is what says the old tuning was wrong, and in which direction. The previous defaults
  (gyro 0.0045 rad/s, acc 0.018 m/s^2, mag 0.05) are SENSOR-NOISE-SCALE numbers -- the right
  order of magnitude for what a stationary IMU does, and two of the three within ~2x of the
  measured static floor. But R does not model sensor noise, it models everything the measurement equation gets wrong, and for an accelerometer
  told to report gravity the dominant term is the linear acceleration that survives the
  joint-centre projection. On Al Borno the accelerometer's innovation is 2.5 m/s^2 per
  sensor, 138x the old constant, while the magnetometer's is 0.094 against 0.05 -- less than
  2x. Being wrong by 138x and by 2x is what puts the RATIO out, and the ratio is the
  part that reaches the filter. In the currency it uses -- information about a direction
  error goes as |v|^2/sigma^2 -- the old tuning gave the accelerometer 9.81^2/0.018^2 = 3.0e5
  against the magnetometer's 1.0^2/0.05^2 = 4.0e2, a ratio of 743:1. At the shipped ratios it
  is 0.96:1. (The magnetometer was never inert -- heading is unobservable from gravity alone,
  so mag carries 100% of that DOF at any weight -- but its weight set how fast heading could
  move, and it was set ~740x too low relative to the accelerometer.)

  THE SWEEP (`--stage sweep`). Scored as the geodesic angle between the filter's R_pc and the
  marker R_pc on valid samples, reported both over the whole trial and with the first
  BURN_IN_S dropped. Both are stored because they answer different questions and, on Al
  Borno, agree on the optimum -- which is the evidence that the burn-in contamination above
  displaces neighbouring grid points without moving the minimum.

HOW THE TWO COMBINE. The sweep sets both ratios. The innovations do two other jobs: they say
which direction the old tuning was wrong in (above), and they break the tie where the sweep
is flat -- among grid points within TIE_FRACTION of the best, `shipped_row` takes the acc/gyro
ratio nearest the one the innovations imply. The tie-break uses the ACCELEROMETER innovation
as its reference rather than trusting the gyro's outright, because the gyro innovation is
measured against a mocap-DIFFERENTIATED angular velocity and is inflated by differentiation
noise: it is white at one sample of lag and drops 10x under a 2 Hz low-pass, more than
white-noise bandwidth scaling predicts. The accelerometer innovation has no such
contamination -- it needs the mocap ORIENTATION, not its derivative.

WHY THE GAINS ARE PER DATASET, AND HOW LITTLE THAT BUYS. Three sets of hardware in three
magnetic environments ought to want three tunings, and the innovations do differ -- but
mostly in the scale, which is inert. Al Borno and IMoVE come out on IDENTICAL ratios: their
accelerometer innovations are 2.49 and 2.20 (13% apart) and their surfaces cannot distinguish
acc/gyro 21.5 from 46.4. The biplane rows differ for real, at acc/gyro 4.64 and 10, because
those IMUs have no magnetometer -- but see `DATASET_STDS`: nothing much is winnable there
either.

EVERY STAGE RUNS ON EVERY DATASET AND EVERY ARM BY DEFAULT. That is not a convenience, it is
a correctness property: the earlier single-dataset default meant the scale check was measured
on two of the twelve (dataset, arm) combinations and the static noise floor on three of four
datasets, and nothing in the outputs said so. Where a combination is genuinely not measurable
-- the biplane half has no magnetometer, so mag_on is not an arm there -- it is dropped with a
printed reason and the sweep continues.

Outputs
-------
    results/statistics/filter_gain_residuals_<dataset>_statistics.parquet
        one row per trial x unit x channel: the innovation rms at the true state, the same
        at three low-pass cutoffs, and its 1/e autocorrelation time
    results/statistics/filter_gain_sweep_<dataset>_<arm>_statistics.parquet
        one row per trial x joint x mag_mode x grid point: RMSE over the whole trial and
        after burn-in
    results/experiments/filter_gains/<dataset>/<arm>_surface.csv
        the pooled surface and its argmin, which is what DATASET_STDS is read off
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import argparse
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import paths
from experiments.experiment_utils import (
    NOMINAL_ACC_MAGNITUDE, NOMINAL_MAG_MAGNITUDE, TrackingSpec, load_statistics, load_trial,
    resolve_stds, run_tracked_grid, save_statistics, _setup_ekf_ground_plate_,
)
from experiments.global_assumptions import (DATASETS, built_trials, static_mask,
                                            tracking_spec)
from src import relative_filter_fast
from src.toolchest.gyro_utils import finite_difference_rotations

EXPERIMENT_DIR = paths.experiment_dir("filter_gains")

# Only the ratios matter (see the module docstring), so this is a free choice. It is
# deliberately NOT one of the shipped values: a reader who sees the sweep's gyro_std equal to
# the default's would reasonably assume the sweep held the default fixed and varied the other
# two, which is not what it does.
SWEEP_GYRO_REF = 0.05

# A decade below the old default's acc/gyro ratio of 4 up to three decades above, in
# third-of-a-decade steps. The span is set by where the surface stops improving in both
# directions -- at both ends of both axes the error is 4-10x the optimum, so the minimum is
# interior on every dataset and is not being read off a boundary.
ACC_RATIOS = np.logspace(-1, 3, 13)
MAG_RATIOS = np.logspace(-2, 2, 13)

# Burn-in is not a nuisance to be trimmed away here, it is a thing being measured: the sweep
# reports the score with and without it, and the difference is what shows that the old
# three-parameter grid was ranking transients. 20 s is well past where a 10x scaling of the
# stds stops mattering (0.03 deg by 10 s on Al Borno).
BURN_IN_S = 20.0

# The biplane half carries 95-190 fluoroscopic frames against 240 s of BioStamp recording, so
# a floor set for the marker datasets throws most of it away. Set low enough to keep those
# trials and high enough that an rms over the survivors means something.
MIN_VALID = 50

# Every arm this experiment knows how to sweep, and the order they are run in. 'relative' is
# mag_on/mag_off/mag_adapt as shipped (projected, raw-magnitude); 'relative_normalized' is the
# same filter with unit-length vector measurements, i.e. the '_normalized' method arms; 'ekf'
# is each segment against the virtual ground plate (normalized, unprojected). The two
# normalizing arms report their ratios in UNIT-VECTOR space and are not comparable to
# 'relative' without dividing by NOMINAL_ACC_MAGNITUDE.
ARMS = ('relative', 'relative_normalized', 'ekf', 'ekf_unnormalized')

# The arms that filter each SEGMENT against a virtual ground plate rather than filtering a
# joint's two sensors against each other. They do not project to the joint centre -- there is
# no joint in the problem they solve -- and the joint angle is the composition of two segment
# estimates.
EKF_ARMS = ('ekf', 'ekf_unnormalized')

# 'ekf' matches the shipped method, which normalizes. 'ekf_unnormalized' is the control arm
# normalization_comparison.py already defines by that name and which had never been swept:
# without it the {relative, ekf} x {raw, normalized} grid has a hole in it, and the claim that
# normalization is worth 14-41% rests on the relative half alone.
NORMALIZING_ARMS = ('relative_normalized', 'ekf')

LP_CUTOFFS = (2.0, 5.0, 10.0)

# Shortest static stretch a noise-floor estimate is taken from. Below about this the std is
# dominated by its own estimation error, and Al Borno's static stretches are seconds long.
STATIC_NOISE_MIN_S = 0.5

# The refinement grid (`--stage refine`). Centred on each (arm, mag_mode)'s coarse argmin and
# spanning ONE coarse step either side, so the coarse argmin is re-run as the middle point and
# the fine surface is anchored to something already measured rather than floating free.
#
# 9 points across 2/3 of a decade is a 1.21x step against the coarse grid's 2.15x. That is
# roughly where the resolution stops being worth paying for: the coarse basins are 4-8 points
# wide at the 10% level, so the optimum is located to about a factor of two, and a 1.21x grid
# pins it to about 20% -- below the spread between datasets, and far below the 13% spread in
# the innovations the scale is anchored on.
REFINE_SPAN_DECADES = 1.0 / 3.0
REFINE_POINTS = 9

# The scale check (`--stage scale`). Multipliers applied to all three of a dataset's shipped
# stds AT FIXED RATIOS, so the steady-state Kalman gain is identical at every point and the
# only thing that can move is how the filter leaves its seed. The range is set so that both
# ends of the argument are inside it: 0.039 puts gyro_std back on the 0.0045 rad/s the tuning
# used to sit at, and 100 is two decades past anything physical.
#
# This exists because "the stds should be the order of magnitude of the sensor noise" is the
# obvious objection to the shipped table, and it is answerable rather than arguable: if the
# curve is flat, the objection is about presentation and the table can be written at any
# scale; if it is not, the scale is a real tuning parameter and has to be defended.
SCALE_MULTIPLIERS = np.logspace(-4, 2, 13)

# The shipped tuning this experiment produces is `experiment_utils.DATASET_STDS`. It lives
# there and not here because experiment_utils cannot import this module -- this one imports
# it -- and because a constant the whole pipeline reads should not sit inside the experiment
# that happens to have measured it.

# ==============================================================================
# Stage 1: the innovation at the true state
# ==============================================================================

def _lowpass(x: np.ndarray, cutoff: float, fs: float) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth. Returns x unchanged when the cutoff is above the
    usable band -- IMoVE's 40 Hz sensors make a 10 Hz cutoff meaningful and a 20 Hz one a
    no-op, and silently returning a filtered-looking array there would be worse."""
    if cutoff >= 0.45 * fs:
        return x
    b, a = butter(2, cutoff / (0.5 * fs))
    return filtfilt(b, a, x, axis=0)


def _autocorr_time(e: np.ndarray, fs: float, max_lag_s: float = 5.0) -> float:
    """Seconds until the residual's normalised autocorrelation first drops below 1/e.

    A Kalman filter assumes its measurement noise is white. A residual whose correlation time
    is a second is a slowly varying BIAS, and the filter will track it rather than average it
    away -- which is the argument for inflating that channel's std above its measured rms
    rather than setting it equal. The magnetometer residual is the case in point: 0.6-1.1 s
    on Al Borno, and essentially unchanged by a 2 Hz low-pass, i.e. it is nearly DC.
    """
    x = e.reshape(len(e), -1)
    x = x - x.mean(axis=0)
    n = min(len(x), int(max_lag_s * fs))
    denom = float(np.sum(x * x))
    if n < 10 or denom == 0.0:
        return float('nan')
    ac = np.array([float(np.sum(x[lag:] * x[:len(x) - lag])) / denom for lag in range(n)])
    below = np.flatnonzero(ac < np.exp(-1.0))
    return float((below[0] if len(below) else n) / fs)


def measure_static_noise(plate, subject: str, trial: str, dataset: str) -> List[Dict]:
    """Each channel's NOISE FLOOR: the per-axis std inside stretches where the sensor is
    genuinely still.

    This is a different quantity from the innovation and answers a different question. The
    innovation says what R should be; this says what the sensor itself contributes, and the
    gap between them is the whole argument for why R is not sensor noise. Measured rather
    than taken from a specification, because a published noise figure is for a bandwidth and
    a temperature, not for this recording -- and because no specification for these sensors
    is cited anywhere in this repo.

    Static stretches come from `global_assumptions.static_mask`, i.e. the gyroscope alone
    decides -- the same definition the rest of the repo splits on. The run MEAN is removed
    before the std is taken, so a gyro bias or a hard-iron offset does not inflate the noise;
    what is left is the white part. Runs shorter than STATIC_NOISE_MIN_S are dropped because
    a std over a handful of samples is mostly its own estimation error.

    Note the answer depends on sample rate -- white noise integrated over a wider bandwidth
    reads higher -- so the 250 Hz biplane numbers are not directly comparable to the 40 Hz
    IMoVE ones. `fs` is recorded on every row for that reason.
    """
    # NOT masked by plate.valid. `valid` marks where the MOCAP POSE is usable, and this is a
    # mocap-free measurement of the sensor -- the IMU recorded continuously whether or not the
    # cameras could see the plate. Masking by it made the two biplane specs, which are the
    # same IMU streams paired with two different ground truths, report different sensor noise.
    mask = static_mask(plate.imu_trace)
    fs = plate.sample_rate
    min_samples = max(int(round(STATIC_NOISE_MIN_S * fs)), 4)
    rows = []
    for channel in ('gyro', 'acc', 'mag'):
        values = getattr(plate.imu_trace, channel)
        if not np.any(values):
            continue
        variances, total = [], 0
        for start, stop in _runs(mask):
            if stop - start < min_samples:
                continue
            window = values[start:stop]
            variances.append(np.var(window - window.mean(axis=0), axis=0).mean()
                             * (stop - start))
            total += stop - start
        if total == 0:
            continue
        rows.append(dict(dataset=dataset, subject=subject, trial=trial, unit=plate.name,
                         channel=channel, kind='static_noise', fs=fs,
                         rms=float(np.sqrt(np.sum(variances) / total)),
                         n=int(total)))
    return rows


def measure_signal_floors(plate, subject: str, trial: str, dataset: str) -> List[Dict]:
    """Each channel's high-frequency content over the WHOLE trial, mocap-free.

    Sits between the static floor and the innovation on the ladder of what you have to know.
    The static floor is blind to everything that only happens while the subject moves --
    field distortion, soft tissue, linear acceleration -- and those are what actually corrupt
    the measurement. This catches the part of them that is fast, using nothing but the sensor
    stream: v minus a 2 Hz low-pass of itself, over every valid sample rather than only the
    still ones.

    For the magnetometer this is the useful one: measured in batch it predicts the swept
    optimal mag_std to within 0.6 deg on every dataset and arm, which no other mocap-free
    quantity does.
    """
    # Over the WHOLE record, not plate.valid -- see measure_static_noise. On the biplane half
    # the mocap window is 4-6 s of a 240 s recording, so masking by it threw away 98% of the
    # sensor stream this is trying to characterise and made the answer depend on which ground
    # truth the plate happened to be paired with.
    fs = plate.sample_rate
    rows = []
    for channel in ('gyro', 'acc', 'mag'):
        values = getattr(plate.imu_trace, channel)
        if not np.any(values) or len(values) < MIN_VALID:
            continue
        hf = values - _lowpass(values, 2.0, fs)
        rows.append(dict(dataset=dataset, subject=subject, trial=trial, unit=plate.name,
                         channel=channel, kind='hf_signal', fs=fs,
                         rms=float(np.sqrt(np.mean(hf ** 2))),
                         n=int(len(values))))
    return rows


def _runs(mask: np.ndarray):
    """(start, stop) of each contiguous True run."""
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def measure_residuals(dataset: str, subject: str, trial: str) -> pd.DataFrame:
    """Per-sensor gyro error and per-joint acc/mag innovation, at the mocap state."""
    spec = tracking_spec(dataset)
    plates = load_trial(subject, trial, dataset=spec.build_dataset)
    common = dict(dataset=dataset, subject=subject, trial=trial)
    rows: List[Dict] = []

    for name, plate in plates.items():
        # The mocap-free floors come FIRST and are not gated on `valid`. Only the gyro row
        # below needs a pose to compare against; skipping a plate for want of mocap also
        # discarded its sensor-noise measurement, which is how the two biplane specs -- the
        # same IMU streams against two ground truths, with different valid windows and so
        # different surviving plates -- ended up reporting different sensor noise.
        rows.extend(measure_static_noise(plate, subject, trial, dataset))
        rows.extend(measure_signal_floors(plate, subject, trial, dataset))

        valid = np.asarray(plate.valid)
        if valid.sum() < MIN_VALID:
            continue
        fs = plate.sample_rate
        omega_mocap = finite_difference_rotations(plate.world_trace.rotations,
                                                  plate.world_trace.timestamps)
        d = plate.imu_trace.gyro - omega_mocap
        row = dict(common, unit=name, channel='gyro', kind='per_sensor', fs=fs,
                   rms=float(np.sqrt(np.mean(d[valid] ** 2))),
                   p95=float(np.percentile(np.abs(d[valid]), 95)),
                   signal_rms=float(np.sqrt(np.mean(plate.imu_trace.gyro[valid] ** 2))),
                   autocorr_s=_autocorr_time(d[valid], fs), n=int(valid.sum()))
        for cutoff in LP_CUTOFFS:
            row[f'rms_lp{cutoff:g}'] = float(np.sqrt(np.mean(_lowpass(d, cutoff, fs)[valid] ** 2)))
        rows.append(row)

    for joint, (parent_name, child_name) in spec.joints.items():
        if parent_name not in plates or child_name not in plates:
            continue
        try:
            rows.extend(_pair_rows(plates, joint, parent_name, child_name, common))
        except Exception as exc:
            # One joint failing must not cost the trial. get_joint_center raises on the
            # bone-referenced biplane spec whenever a trial has too few fluoroscopic frames to
            # fit a centre, and letting that propagate discarded every OTHER row for the trial
            # -- including the mocap-free sensor floors, which had already been computed and
            # do not depend on the joint at all. It cost 78 of 380 trials.
            print(f"  .. {common['dataset']}/{common['subject']}/{common['trial']} {joint}: "
                  f"{type(exc).__name__}", flush=True)
    return pd.DataFrame(rows)


def _pair_rows(plates: Dict, joint: str, parent_name: str, child_name: str,
               common: Dict) -> List[Dict]:
    """The acc/mag innovation rows for one joint.

    Raises if the joint centre cannot be fit, which is routine on the bone-referenced biplane
    spec -- hence the caller catching per joint rather than per trial.
    """
    parent, child = plates[parent_name].copy(), plates[child_name].copy()
    # The filter sees the PROJECTED trace, so the residual has to be measured on it too:
    # projecting to the joint centre is precisely what removes the common part of the two
    # segments' linear acceleration, and it is the largest single term in the raw pair
    # difference.
    parent_offset, child_offset, jc_err = parent.world_trace.get_joint_center(child.world_trace)
    parent.imu_trace = parent.project_imu_trace(parent_offset)
    child.imu_trace = child.project_imu_trace(child_offset)
    valid = np.asarray(parent.valid) & np.asarray(child.valid)
    if valid.sum() < MIN_VALID:
        return []

    fs = parent.sample_rate
    R_wp, R_wc = parent.world_trace.rotations, child.world_trace.rotations
    rows: List[Dict] = []
    for channel in ('acc', 'mag'):
        v_p, v_c = getattr(parent.imu_trace, channel), getattr(child.imu_trace, channel)
        # A dataset with no magnetometer carries an all-zero channel, and its "residual"
        # would be a well-formed zero rather than an absence.
        if not np.any(v_p) or not np.any(v_c):
            continue
        e = np.einsum('nij,nj->ni', R_wp, v_p) - np.einsum('nij,nj->ni', R_wc, v_c)
        row = dict(common, unit=joint, channel=channel, kind='pair', fs=fs,
                   rms=float(np.sqrt(np.mean(e[valid] ** 2))),
                   p95=float(np.percentile(np.abs(e[valid]), 95)),
                   signal_rms=float(np.sqrt(np.mean(
                       np.einsum('nij,nj->ni', R_wp, v_p)[valid] ** 2))),
                   autocorr_s=_autocorr_time(e[valid], fs), n=int(valid.sum()),
                   jc_err=float(np.mean(np.linalg.norm(jc_err, axis=1))))
        for cutoff in LP_CUTOFFS:
            row[f'rms_lp{cutoff:g}'] = float(
                np.sqrt(np.mean(_lowpass(e, cutoff, fs)[valid] ** 2)))
        rows.append(row)
    return rows


def per_sensor_sigma(residuals: pd.DataFrame, channel: str, column: str = 'rms',
                     kind: Optional[str] = None) -> float:
    """The single-sensor std implied by a channel's residuals.

    The acc/mag rows are PAIR differences of two independent sensors, so their variance is
    2*sigma^2; the gyro rows are already per-sensor. Pooled as a median over trials rather
    than an rms, because the biplane half's drop landings put a handful of trials an order of
    magnitude above the rest and an rms would report those instead of the population.
    """
    # kind defaults to the INNOVATION rows. The frame also carries 'static_noise' rows with
    # the same channel names, and pooling the two together would average a noise floor into
    # a model error and quietly halve the accelerometer's number.
    wanted = kind or ('per_sensor' if channel == 'gyro' else 'pair')
    subset = residuals[(residuals.channel == channel) & (residuals.kind == wanted)]
    if subset.empty:
        return float('nan')
    pooled = float(np.median(subset[column]))
    return pooled if channel == 'gyro' else pooled / np.sqrt(2.0)


# ==============================================================================
# Stage 2: the sweep
# ==============================================================================

def _prepare_joint(spec: TrackingSpec, plates: Dict, joint: str, arm: str,
                   ground=None) -> Optional[Dict]:
    """Everything the kernel needs for ONE joint, or None if it is not scoreable here.

    ONE JOINT AT A TIME IS THE POINT. Preparing every joint up front and then looping the
    grid over them is the obvious shape and it does not survive IMoVE: those trials merge
    three mocap takes onto a 2.6 h inertial record, so a single trial is ~460k samples across
    24 plates, and holding six joints' projected traces and marker rotations alongside the
    loaded plates ran the machine out of memory with the pool wedged on a dead worker. Built
    per joint and discarded, the peak is one joint's arrays on top of the trial -- which is
    also why the live table's columns are joints and not grid points.

    The projection stays hoisted out of the grid loop, where it matters: it does not depend on
    the stds, so it runs once per joint against that joint's 169 filter passes.
    """
    parent_name, child_name = spec.joints[joint]
    if parent_name not in plates or child_name not in plates:
        return None
    parent, child = plates[parent_name], plates[child_name]
    valid = np.asarray(parent.valid) & np.asarray(child.valid)
    if valid.sum() < MIN_VALID:
        return None
    timestamps = parent.imu_trace.timestamps
    job = dict(joint=joint,
               R_marker=np.einsum('tji,tjk->tik', parent.world_trace.rotations,
                                  child.world_trace.rotations),
               valid=valid, post=valid & (timestamps - timestamps[0] > BURN_IN_S))
    if _is_ekf(arm):
        # The EKF arm does not project -- it estimates each segment's absolute orientation
        # from that segment's own sensor against a virtual ground plate -- and it normalizes,
        # so its stds come out in unit-vector space.
        job['parent'] = _kernel_inputs(ground, parent)
        job['child'] = _kernel_inputs(ground, child)
        return job
    parent, child = parent.copy(), child.copy()
    parent_offset, child_offset, _ = parent.world_trace.get_joint_center(child.world_trace)
    parent.imu_trace = parent.project_imu_trace(parent_offset)
    child.imu_trace = child.project_imu_trace(child_offset)
    job['pair'] = _kernel_inputs(parent, child)
    return job


def _kernel_inputs(parent, child) -> Dict:
    return dict(
        gyro_p=np.ascontiguousarray(parent.imu_trace.gyro),
        gyro_c=np.ascontiguousarray(child.imu_trace.gyro),
        acc_p=parent.imu_trace.acc, acc_c=child.imu_trace.acc,
        mag_p=parent.imu_trace.mag, mag_c=child.imu_trace.mag,
        R_wp0=parent.world_trace.rotations[0], R_wc0=child.world_trace.rotations[0],
        dt=float(np.mean(np.diff(parent.imu_trace.timestamps))))


def _run_kernel(inputs: Dict, mag_mode: str, acc_std: float, mag_std: float,
                normalize: bool, gyro_std: float = SWEEP_GYRO_REF,
                init_std: float = np.deg2rad(0.1)) -> np.ndarray:
    """mag_mode 'off' zeroes the reading, which is how the pipeline switches the
    magnetometer off -- it makes that sensor's rows of H exactly zero, so its columns of K
    are zero and mag_std becomes provably inert."""
    mag_p = np.zeros_like(inputs['mag_p']) if mag_mode == 'off' else inputs['mag_p']
    mag_c = np.zeros_like(inputs['mag_c']) if mag_mode == 'off' else inputs['mag_c']
    return relative_filter_fast.run_relative_filter(
        inputs['gyro_p'], inputs['gyro_c'],
        np.stack([inputs['acc_p'], mag_p], axis=1),
        np.stack([inputs['acc_c'], mag_c], axis=1),
        inputs['dt'],
        gyro_std_parent=np.full(3, gyro_std), gyro_std_child=np.full(3, gyro_std),
        vector_sensor_stds_parent=[np.full(3, acc_std), np.full(3, mag_std)],
        vector_sensor_stds_child=[np.full(3, acc_std), np.full(3, mag_std)],
        R_wp0=inputs['R_wp0'], R_wc0=inputs['R_wc0'], normalize_measurements=normalize,
        init_orientation_std=init_std)


def _geodesic_deg(R_est: np.ndarray, R_marker: np.ndarray) -> np.ndarray:
    """Same quantity as compute_error_stats' 'MAG' axis -- the norm of the rotation error --
    computed from the trace rather than via Rotation.from_matrix, which at 169 grid points x
    7 joints x 100k samples is the difference between minutes and hours."""
    trace = np.einsum('tij,tij->t', R_est, R_marker)
    return np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def _score(R_est: np.ndarray, job: Dict) -> Tuple[float, int, float, int]:
    err = _geodesic_deg(R_est, job['R_marker'])
    valid, post = job['valid'], job['post']
    return (float(np.sqrt(np.mean(err[valid] ** 2))), int(valid.sum()),
            float(np.sqrt(np.mean(err[post] ** 2))) if post.any() else float('nan'),
            int(post.sum()))


def _normalizes(arm: str) -> bool:
    """Whether this arm scales its vector measurements to unit length before the update.

    This decides the UNITS the swept ratios come out in: a normalizing arm's acc_std is a
    fraction of a unit vector, an unnormalized one's is m/s^2, and the two differ by
    NOMINAL_ACC_MAGNITUDE. Comparing an optimum from one against an optimum from the other
    without that conversion reads as a 9.81x disagreement between arms that in fact agree.
    """
    return arm in NORMALIZING_ARMS


def _is_ekf(arm: str) -> bool:
    return arm in EKF_ARMS


def stds_for_arm(stds: Dict[str, float], arm: str) -> Dict[str, float]:
    """`stds`, which are PHYSICAL (DATASET_STDS), in the units this arm's filter consumes.

    A normalizing arm scales each vector measurement to unit length, so a physical std has to
    be divided by that sensor's nominal magnitude first -- the same conversion
    experiment_utils' `rescale_stds` performs. Handing physical stds straight to a normalizing
    arm de-weights its accelerometer by 9.81, which is the exact mistake this whole experiment
    started from; the scale stage did it for two releases because it only ever ran on the one
    arm where the conversion is a no-op.
    """
    if not _normalizes(arm):
        return dict(stds)
    return {'gyro_std': stds['gyro_std'],
            'acc_std': stds['acc_std'] / NOMINAL_ACC_MAGNITUDE,
            'mag_std': stds['mag_std'] / NOMINAL_MAG_MAGNITUDE}


def _evaluate(job: Dict, arm: str, mag_mode: str, acc_std: float, mag_std: float,
              gyro_std: float = SWEEP_GYRO_REF, init_std: float = np.deg2rad(0.1)):
    """`init_std` is the seed covariance's per-axis std. It is exposed because it is NOT a
    neutral knob: under the pipeline's current 0.1 deg the filter trusts its mocap seed and
    the tuning's overall scale is inert, but under the 1 rad (P0 = eye(6)) this code used
    before 2026-08-05 the seed is effectively discarded, and the scale then controls how fast
    P falls back to steady state. A tuning inherited from that era was tuned against a
    different objective."""
    normalize = _normalizes(arm)
    if _is_ekf(arm):
        R_parent = _run_kernel(job['parent'], mag_mode, acc_std, mag_std, normalize,
                               gyro_std, init_std)
        R_child = _run_kernel(job['child'], mag_mode, acc_std, mag_std, normalize,
                              gyro_std, init_std)
        return _score(np.einsum('tji,tjk->tik', R_parent, R_child), job)
    return _score(_run_kernel(job['pair'], mag_mode, acc_std, mag_std, normalize,
                              gyro_std, init_std), job)


def ladder_worker(row_key, stage_labels: List[str], shared_state: Dict,
                  dataset: str, arm: str, mag_modes: List[str],
                  tunings: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Score every rung of the ladder EXACTLY, on the same trials and metric as the sweep.

    Exactly, not by reading the nearest grid point off the surface: the rungs are wherever the
    measurements put them, and snapping them to a 2.15x grid would credit or blame a rung for
    up to a factor of 1.5 it never asked for."""
    subject, trial = row_key
    spec = tracking_spec(dataset)
    try:
        plates = load_trial(subject, trial, dataset=spec.build_dataset)
    except Exception as exc:
        for stage in stage_labels:
            shared_state[(row_key, stage)] = f"Failed ({type(exc).__name__})"
        return pd.DataFrame()
    needed = {name for pair in spec.joints.values() for name in pair}
    ground = _setup_ekf_ground_plate_(list(plates.values()), spec=spec) if _is_ekf(arm) else None
    plates = {name: plate for name, plate in plates.items() if name in needed}

    rows = []
    for stage in stage_labels:
        shared_state[(row_key, stage)] = "Running"
        try:
            job = _prepare_joint(spec, plates, stage, arm, ground)
            if job is None:
                shared_state[(row_key, stage)] = "Skipped"
                continue
            for mag_mode in mag_modes:
                for label, t in tunings[mag_mode].items():
                    _, _, post, n_post = _evaluate(
                        job, arm, mag_mode, t['acc_std'], t['mag_std'],
                        gyro_std=t['gyro_std'])
                    rows.append(dict(dataset=dataset, subject=subject, trial=trial, arm=arm,
                                     joint=stage, mag_mode=mag_mode, tuning=label,
                                     rmse_post_deg=post, n_post=n_post, **t))
            del job
            shared_state[(row_key, stage)] = "Success"
        except Exception as exc:
            shared_state[(row_key, stage)] = f"Failed ({type(exc).__name__})"
    return pd.DataFrame(rows)


def sweep_worker(row_key, stage_labels: List[str], shared_state: Dict,
                 dataset: str, arm: str, grids: Dict[str, Tuple[Any, Any]]) -> pd.DataFrame:
    """One process per trial; the live table's columns are the JOINTS.

    Joints rather than grid points because a 169-column table does not render, and because
    the joint is the unit of work whose memory has to be bounded (see `_prepare_joint`).

    `grids` maps each mag_mode to its own (acc_ratios, mag_ratios). Per mag_mode rather than
    one shared grid because the refinement stage centres on each mode's OWN optimum, and
    mag_on's and mag_off's are not in the same place -- on Al Borno they are a factor of two
    apart in acc/gyro."""
    subject, trial = row_key
    spec = tracking_spec(dataset)
    try:
        plates = load_trial(subject, trial, dataset=spec.build_dataset)
    except Exception as exc:
        for stage in stage_labels:
            shared_state[(row_key, stage)] = f"Failed ({type(exc).__name__})"
        return pd.DataFrame()

    # Dropping the plates no joint names frees most of an IMoVE trial before any array is
    # built on top of it: 24 plates loaded, 12 scored.
    needed = {name for pair in spec.joints.values() for name in pair}
    ground = _setup_ekf_ground_plate_(list(plates.values()), spec=spec) if _is_ekf(arm) else None
    plates = {name: plate for name, plate in plates.items() if name in needed}

    rows = []
    for stage in stage_labels:
        shared_state[(row_key, stage)] = "Running"
        try:
            job = _prepare_joint(spec, plates, stage, arm, ground)
            if job is None:
                shared_state[(row_key, stage)] = "Skipped"
                continue
            for mag_mode, (acc_grid, mag_grid) in grids.items():  # noqa: PLR1702
                for acc_ratio in acc_grid:
                    for mag_ratio in mag_grid:
                        mag_std = SWEEP_GYRO_REF * (mag_ratio if mag_mode == 'on' else 1.0)
                        rmse, n, rmse_post, n_post = _evaluate(
                            job, arm, mag_mode, SWEEP_GYRO_REF * acc_ratio, mag_std)
                        rows.append(dict(dataset=dataset, subject=subject, trial=trial,
                                         arm=arm, joint=stage, mag_mode=mag_mode,
                                         acc_ratio=acc_ratio, mag_ratio=mag_ratio,
                                         rmse_deg=rmse, n=n,
                                         rmse_post_deg=rmse_post, n_post=n_post))
            del job
            shared_state[(row_key, stage)] = "Success"
        except Exception as exc:
            shared_state[(row_key, stage)] = f"Failed ({type(exc).__name__})"
    return pd.DataFrame(rows)


def scale_worker(row_key, stage_labels: List[str], shared_state: Dict,
                 dataset: str, arm: str, mag_modes: List[str],
                 base: Dict[str, float]) -> pd.DataFrame:
    """Multiply all three of `base`'s stds by each SCALE_MULTIPLIERS entry and re-score.

    The ratios are held exactly, so every point here has the same steady-state Kalman gain by
    construction and the ONLY mechanism by which the score can move is the seed: the filter
    starts from P0 = init_orientation_std^2 * I with the state set from mocap, and the scale
    decides how many samples it takes for the process noise to overwhelm that. Reporting
    whole-trial and post-burn-in RMSE side by side is what separates the two."""
    subject, trial = row_key
    spec = tracking_spec(dataset)
    try:
        plates = load_trial(subject, trial, dataset=spec.build_dataset)
    except Exception as exc:
        for stage in stage_labels:
            shared_state[(row_key, stage)] = f"Failed ({type(exc).__name__})"
        return pd.DataFrame()

    needed = {name for pair in spec.joints.values() for name in pair}
    ground = _setup_ekf_ground_plate_(list(plates.values()), spec=spec) if _is_ekf(arm) else None
    plates = {name: plate for name, plate in plates.items() if name in needed}

    rows = []
    for stage in stage_labels:
        shared_state[(row_key, stage)] = "Running"
        try:
            job = _prepare_joint(spec, plates, stage, arm, ground)
            if job is None:
                shared_state[(row_key, stage)] = "Skipped"
                continue
            for mag_mode in mag_modes:
                for multiplier in SCALE_MULTIPLIERS:
                    rmse, n, rmse_post, n_post = _evaluate(
                        job, arm, mag_mode,
                        base['acc_std'] * multiplier, base['mag_std'] * multiplier,
                        gyro_std=base['gyro_std'] * multiplier)
                    rows.append(dict(dataset=dataset, subject=subject, trial=trial,
                                     arm=arm, joint=stage, mag_mode=mag_mode,
                                     multiplier=multiplier,
                                     gyro_std=base['gyro_std'] * multiplier,
                                     acc_std=base['acc_std'] * multiplier,
                                     mag_std=base['mag_std'] * multiplier,
                                     rmse_deg=rmse, n=n,
                                     rmse_post_deg=rmse_post, n_post=n_post))
            del job
            shared_state[(row_key, stage)] = "Success"
        except Exception as exc:
            shared_state[(row_key, stage)] = f"Failed ({type(exc).__name__})"
    return pd.DataFrame(rows)


def pool_scale(scale: pd.DataFrame) -> pd.DataFrame:
    keyed = scale.assign(_trial_key=scale['subject'].astype(str) + '/' + scale['trial'].astype(str))
    return (keyed.groupby(['mag_mode', 'multiplier'], dropna=False)
            .agg(gyro_std=('gyro_std', 'first'), acc_std=('acc_std', 'first'),
                 mag_std=('mag_std', 'first'),
                 rmse_deg=('rmse_deg', 'mean'), rmse_post_deg=('rmse_post_deg', 'mean'),
                 n_trials=('_trial_key', 'nunique'))
            .reset_index()
            .sort_values(['mag_mode', 'multiplier']))


def report_scale(pooled: pd.DataFrame) -> None:
    for mag_mode, group in pooled.groupby('mag_mode'):
        shipped = group[np.isclose(group.multiplier, 1.0)]
        reference = float(shipped.rmse_post_deg.iloc[0]) if len(shipped) else float('nan')
        print(f"\n  mag_{mag_mode} — ratios fixed, all three stds scaled together "
              f"({int(group.n_trials.max())} trials)")
        print(f"    {'x':>8}  {'gyro_std':>9}{'acc_std':>9}{'mag_std':>9}"
              f"{'RMSE':>9}{'RMSE':>10}{'vs shipped':>12}")
        print(f"    {'':>8}  {'(rad/s)':>9}{'(m/s^2)':>9}{'':>9}"
              f"{'all':>9}{'post-20s':>10}{'(post)':>12}")
        for row in group.itertuples():
            delta = row.rmse_post_deg - reference
            print(f"    {row.multiplier:>8.4g}  {row.gyro_std:>9.4g}{row.acc_std:>9.4g}"
                  f"{row.mag_std:>9.4g}{row.rmse_deg:>9.2f}{row.rmse_post_deg:>10.2f}"
                  f"{delta:>+12.3f}")


def coarse_grids(mag_modes: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """The full-span grid, one entry per mag_mode.

    mag_off gets a single NaN mag_ratio rather than MAG_RATIOS: zeroing the magnetometer
    zeroes its rows of H, so mag_std is provably inert there and sweeping it would run 13
    identical filters and report them as 13 measurements."""
    return {mode: (ACC_RATIOS, MAG_RATIOS if mode == 'on' else np.array([np.nan]))
            for mode in mag_modes}


def _around(centre: float, span: float = REFINE_SPAN_DECADES,
            points: int = REFINE_POINTS) -> np.ndarray:
    return centre * 10.0 ** np.linspace(-span, span, points)


def refine_grids(surface: pd.DataFrame, centre: Optional[Tuple[float, float]] = None,
                 span: float = REFINE_SPAN_DECADES,
                 points: int = REFINE_POINTS) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """A fine grid per mag_mode, centred on that mode's argmin in `surface`.

    `centre` overrides the argmin with an explicit (acc_ratio, mag_ratio) and is what you want
    when checking a tuning that came from somewhere else -- a previous run, a paper, someone's
    memory. An odd `points` puts that exact pair in the middle of the grid, so the candidate
    is SCORED rather than interpolated between its neighbours.

    mag_off ignores the mag half of `centre`: with the magnetometer zeroed, mag_std cannot
    change the estimate."""
    if points % 2 == 0:
        raise ValueError(f"refine points must be odd so the centre is itself a grid point; "
                         f"got {points}.")
    grids = {}
    for mag_mode, group in surface.groupby('mag_mode', dropna=False):
        if centre is None:
            best = group.loc[group.rmse_deg.idxmin()]
            acc_centre, mag_centre = best.acc_ratio, best.mag_ratio
        else:
            acc_centre, mag_centre = centre
            if mag_mode != 'on':
                mag_centre = float('nan')
        mag = (_around(mag_centre, span, points) if np.isfinite(mag_centre)
               else np.array([np.nan]))
        grids[mag_mode] = (_around(acc_centre, span, points), mag)
    return grids


def check_refined(coarse: pd.DataFrame, fine: pd.DataFrame) -> None:
    """Report how much the refinement moved, and shout if it hit its own edge.

    An argmin on the boundary of the fine grid means the coarse argmin was not the basin's
    floor and the refinement has walked off the span it was given -- the fine surface is then
    a lower bound on the improvement, not the improvement, and re-centring is needed."""
    for mag_mode, group in fine.groupby('mag_mode', dropna=False):
        c = coarse[coarse.mag_mode == mag_mode]
        c_best = c.loc[c.rmse_deg.idxmin()]
        f_best = group.loc[group.rmse_deg.idxmin()]
        # The two surfaces must be over the SAME trials or the delta is a subset effect
        # wearing a refinement's clothes. A --trials-capped refinement against a full coarse
        # surface reported a 25% "improvement" that was entirely the two trial sets differing.
        if int(c_best.n_trials) != int(f_best.n_trials):
            print(f"\n  mag_{mag_mode}: coarse covers {int(c_best.n_trials)} trials and the "
                  f"refinement {int(f_best.n_trials)} — NOT COMPARABLE, no delta reported. "
                  f"Refined best {f_best.rmse_deg:.3f} deg at "
                  f"({f_best.acc_ratio:.4g}, {f_best.mag_ratio:.4g}).")
            continue
        gain = c_best.rmse_deg - f_best.rmse_deg
        print(f"\n  mag_{mag_mode}: coarse {c_best.rmse_deg:.3f} deg at "
              f"({c_best.acc_ratio:.4g}, {c_best.mag_ratio:.4g}) -> refined "
              f"{f_best.rmse_deg:.3f} deg at ({f_best.acc_ratio:.4g}, {f_best.mag_ratio:.4g})"
              f"   [{-gain:+.3f} deg, {-100*gain/c_best.rmse_deg:+.2f}%]")
        for axis, values in (('acc_ratio', group.acc_ratio), ('mag_ratio', group.mag_ratio)):
            unique = np.unique(values.dropna())
            if len(unique) < 2:
                continue
            if np.isclose(f_best[axis], unique[0]) or np.isclose(f_best[axis], unique[-1]):
                print(f"    (!) refined argmin sits on the {axis} EDGE of the fine grid "
                      f"({unique[0]:.4g}..{unique[-1]:.4g}). The coarse argmin was not the "
                      f"basin floor; re-centre and re-run before quoting this.")


# The ladder. Each entry is a way of choosing the three stds, labelled by WHAT YOU HAD TO
# KNOW to choose it. The order is the point of the experiment: everything above the line could
# have been decided before this data existed, everything below it needs the data and its
# ground truth, and the gap between the two is what tuning acausally is actually worth.
#
# 'requires' is not decoration. A number you cannot obtain without mocap cannot be shipped in
# a filter that runs on a subject who was never in a mocap lab.
CANDIDATES = (
    ('legacy default', 'NOT AN INFORMATION RUNG — the arbitrary value this replaced'),
    ('static floor',   'a static recording. No mocap, no motion.'),
    ('static mag x5',  'a static recording + the bias argument (mag inflated 5x)'),
    ('static mag x10', 'a static recording + the bias argument (mag inflated 10x)'),
    ('signal floor',   'the trial itself. No mocap.'),
    ('innovation',     'mocap — the true-state residual'),
    ('swept optimum',  'mocap AND the scored metric — ACAUSAL, an upper bound'),
)

# What the pipeline was tuned at before this experiment, in physical units.
#
# THESE ARE ARBITRARY, on the author's own account, and the ladder marks them as such. They
# were in DEFAULT_*_STD with no comment and no derivation; `git log -S` finds one commit,
# "WIP", with no rationale. An earlier draft of this experiment called them datasheet figures
# -- that was a guess, it was wrong, and it is recorded here because the guess was repeated
# often enough to look established.
#
# The data agrees they are not a coherent instrument spec: the measured static noise floor is
# 1.7x the gyro value, 2.1x the acc value and 0.06x the mag value. A real specification is
# conservative by a roughly consistent margin; two of these are within 2x and the third is
# 16x out in the OPPOSITE direction.
#
# They are kept in the ladder as a HISTORICAL BASELINE -- what changing the tuning bought --
# and NOT as its a-priori rung. An arbitrary number says nothing about what could have been
# chosen from prior knowledge, so reading it as "the best you can do without data" would
# flatter every rung below it. The lowest genuine information rung is 'static floor', which
# needs one stationary recording and could be collected at setup, before any trial.
LEGACY_STDS = {'gyro_std': 0.0045, 'acc_std': 0.018, 'mag_std': 0.05}


def candidate_tunings(dataset: str, arm: str, residuals: pd.DataFrame,
                      surface: Optional[pd.DataFrame],
                      mag_modes: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """The ladder's rungs for one (dataset, arm), keyed by mag_mode, each in that arm's units.

    PER MAG_MODE because the top rung is each mode's OWN swept argmin. mag_on and mag_off do
    not share an optimum -- on Al Borno they are a factor of two apart in acc/gyro -- so a
    single tuning scored on both would understate the ceiling for whichever mode it was not
    chosen for, and section 6 divides by that ceiling.

    A rung is omitted rather than guessed when its inputs are missing: a magnetometer floor on
    a dataset with no magnetometer is not a small number, it is not a number.
    """
    def floor(kind, channel):
        sub = residuals[(residuals.kind == kind) & (residuals.channel == channel)]
        return float(np.median(sub.rms)) if len(sub) else float('nan')

    static = {c: floor('static_noise', c) for c in ('gyro', 'acc', 'mag')}
    hf = {c: floor('hf_signal', c) for c in ('gyro', 'acc', 'mag')}
    innov = {c: per_sensor_sigma(residuals, c) for c in ('gyro', 'acc', 'mag')}

    def triple(gyro, acc, mag):
        """One rung, with the no-magnetometer fallback taken against THIS rung's own acc.

        Against its own acc, not against whatever produced the missing mag: the fallback is a
        mag/acc RATIO, so pairing it with a different rung's accelerometer produced a mag_std
        180x its acc on the biplane half -- inert there, but nonsense on the face of it.
        """
        if not np.isfinite(mag):
            mag = acc * MAG_OVER_ACC_WHEN_ABSENT
        if not all(np.isfinite([gyro, acc, mag])):
            return None
        return stds_for_arm({'gyro_std': gyro, 'acc_std': acc, 'mag_std': mag}, arm)

    base = {
        'legacy default': stds_for_arm(LEGACY_STDS, arm),
        'static floor': triple(static['gyro'], static['acc'], static['mag']),
        # Only where there IS a magnetometer: with the channel zeroed its std cannot change
        # the estimate, so these rungs would be three identical rows under three names.
        **({f'static mag x{k:g}': triple(static['gyro'], static['acc'], static['mag'] * k)
            for k in MAG_BIAS_INFLATIONS} if np.isfinite(static['mag']) else {}),
        # gyro and acc from the static floor; magnetometer from the high-frequency signal,
        # the one channel where a mocap-free HF measure is known to track the optimum.
        'signal floor': triple(static['gyro'], static['acc'], hf['mag']),
        'innovation': triple(innov['gyro'], innov['acc'], innov['mag']),
    }
    base = {k: v for k, v in base.items() if v is not None}

    out = {}
    for mode in mag_modes:
        rungs = dict(base)
        if surface is not None and not surface.empty:
            g = surface[surface.mag_mode == mode]
            if not g.empty:
                b = g.loc[g.rmse_deg.idxmin()]
                gyro = SCALE_ANCHOR_GYRO_STD
                acc = gyro * b.acc_ratio
                mag = (gyro * b.mag_ratio if np.isfinite(b.mag_ratio)
                       else acc * MAG_OVER_ACC_WHEN_ABSENT)
                # already in the arm's units -- the surface was swept in them
                rungs['swept optimum'] = {'gyro_std': gyro, 'acc_std': acc, 'mag_std': mag}
        out[mode] = rungs
    return out


def pool_surface(sweep: pd.DataFrame, metric: str = 'rmse_post_deg') -> pd.DataFrame:
    """Mean over trials and joints at each grid point, per mag_mode.

    An unweighted mean over trials, deliberately: the question is which tuning is best for a
    trial, and weighting by sample count would let IMoVE's five 2.6 h long walks outvote its
    other 256 trials.

    n_trials counts (subject, trial) PAIRS. Counting `trial` alone reported 2 for Al Borno's
    19 trials, because 'walking' and 'complexTasks' are the only two trial names in it."""
    keyed = sweep.assign(_trial_key=sweep['subject'].astype(str) + '/' + sweep['trial'].astype(str))
    return (keyed.groupby(['mag_mode', 'acc_ratio', 'mag_ratio'], dropna=False)
            .agg(rmse_deg=(metric, 'mean'), rmse_median=(metric, 'median'),
                 n_trials=('_trial_key', 'nunique'), n_rows=(metric, 'size'))
            .reset_index()
            .sort_values(['mag_mode', 'rmse_deg']))


# Grid points scoring within this fraction of the optimum are treated as tied. The sweep
# resolves the two ratios very unevenly -- on IMoVE, acc/gyro of 21.5, 46.4 and 100 score
# 10.0, 9.9 and 10.5 deg, so its argmin at 46.4 is not a measurement -- and picking an argmin
# out of a flat direction would ship a gyro_std that differs 2.4x between two datasets whose
# objective cannot tell the two values apart. 2% is set by that case: IMoVE's 21.5 is 1.1%
# off its argmin, and nothing else on either marker dataset's surface falls between 1% and 5%.
TIE_FRACTION = 0.02

# What mag_std becomes on a dataset whose IMUs have no magnetometer. It cannot be left out
# and it cannot be NaN: the channel is inert only in the sense that its rows of H are zero,
# while its variance still enters S, and one NaN there propagates through the Cholesky solve
# into every state. 0.1 is the mag/acc ratio BOTH marker datasets' sweeps land on, so this is
# the least arbitrary finite number available rather than a made-up one.
MAG_OVER_ACC_WHEN_ABSENT = 0.1

# Multipliers applied to the magnetometer's static noise floor to make the 'static mag xN'
# rungs. They are not fitted -- they come from the argument in section 2, which is that the
# magnetometer's error has a 0.6-1.1 s autocorrelation and is therefore a BIAS the filter will
# track rather than white noise it will average away. A Kalman filter's only defence against a
# correlated error is an inflated R, so a floor measured at rest must be scaled up before it
# is used, and the question is by how much. 5 and 10 bracket the factor the sweep implies
# (3.8x on the raw arm, 4.6x on the normalised one) without being read off it.
MAG_BIAS_INFLATIONS = (5.0, 10.0)

# The scale the shipped rows are written at, mirroring experiment_utils.SCALE_ANCHOR_GYRO_STD.
# Duplicated as a literal rather than imported so that a change here shows up as a difference
# between what this experiment computes and what the pipeline runs, instead of silently
# agreeing with whatever the pipeline was already set to.
SCALE_ANCHOR_GYRO_STD = 0.0045


def shipped_row(surface: pd.DataFrame, residuals: pd.DataFrame) -> Dict[str, float]:
    """The DATASET_STDS row this dataset's sweep and residuals imply.

    THE SWEEP FIXES THE TWO RATIOS AND NOTHING ELSE. The common scale is inert -- `--stage
    scale` moves it over six decades and the pooled RMSE shifts by 0.001 deg -- so it is not
    estimated here, it is CHOSEN, at SCALE_ANCHOR_GYRO_STD, so the shipped numbers read as
    sensor stds. Any common rescaling of the returned row is the same filter.

    Where the sweep is flat in the acc/gyro direction the measured residual breaks the tie:
    take every grid point within TIE_FRACTION of the best and pick the one whose acc/gyro
    ratio is closest in log space to the ratio the residuals themselves imply. On the two
    marker datasets that rule lands both on 21.5 -- the honest reading, since their surfaces
    cannot distinguish 21.5 from 46.4 and their innovations differ by 13%, not by the 2.4x
    their raw argmins would suggest.

    `acc_innovation` and `gyro_innovation` are returned alongside because they are the part
    of this that is a measurement rather than a convention, and because the ratio between
    them is what the tie-break used.

    Written as code rather than done by eye so that re-running the experiment reproduces the
    shipped table instead of inviting a fresh judgement call.
    """
    preferred = 'on' if (surface.mag_mode == 'on').any() else 'off'
    group = surface[surface.mag_mode == preferred]
    best = group.rmse_deg.min()
    tied = group[group.rmse_deg <= best * (1.0 + TIE_FRACTION)]

    acc = per_sensor_sigma(residuals, 'acc')
    gyro = per_sensor_sigma(residuals, 'gyro')
    pick = tied.iloc[np.argmin(np.abs(np.log(tied.acc_ratio.values) - np.log(acc / gyro)))]

    gyro_std = SCALE_ANCHOR_GYRO_STD
    acc_std = gyro_std * pick.acc_ratio
    if np.isfinite(pick.mag_ratio):
        mag_std = gyro_std * pick.mag_ratio
    else:
        # mag_off swept no mag_ratio, so there is nothing to read the magnetometer's weight
        # off. Use this dataset's own measured mag/acc ratio where the channel exists, and
        # the cross-dataset value where it does not.
        mag = per_sensor_sigma(residuals, 'mag')
        mag_over_acc = mag / acc if np.isfinite(mag) else MAG_OVER_ACC_WHEN_ABSENT
        mag_std = acc_std * mag_over_acc
    if not all(np.isfinite([gyro_std, acc_std, mag_std])):
        raise ValueError(f"Non-finite tuning {gyro_std=} {acc_std=} {mag_std=}. A NaN std "
                         f"reaches S through R and turns every state NaN, silently — it must "
                         f"not be written into DATASET_STDS.")
    return {'gyro_std': float(gyro_std), 'acc_std': float(acc_std), 'mag_std': float(mag_std),
            'acc_ratio': float(pick.acc_ratio), 'mag_ratio': float(pick.mag_ratio),
            'acc_innovation': float(acc), 'gyro_innovation': float(gyro),
            'rmse_deg': float(pick.rmse_deg), 'n_tied': int(len(tied))}


def report_surface(surface: pd.DataFrame, residuals: Optional[pd.DataFrame] = None) -> None:
    for mag_mode, group in surface.groupby('mag_mode'):
        best = group.iloc[0]
        print(f"\n  mag_{mag_mode}: best acc_ratio={best.acc_ratio:.3g} "
              f"mag_ratio={best.mag_ratio:.3g} -> {best.rmse_deg:.2f} deg "
              f"({int(best.n_trials)} trials)")
        within = group[group.rmse_deg < best.rmse_deg * 1.10]
        print(f"    within 10%: {len(within)}/{len(group)} points, "
              f"acc_ratio {within.acc_ratio.min():.3g}-{within.acc_ratio.max():.3g}, "
              f"mag_ratio {within.mag_ratio.min():.3g}-{within.mag_ratio.max():.3g}")
    if residuals is None or residuals.empty:
        return
    acc = per_sensor_sigma(residuals, 'acc')
    mag = per_sensor_sigma(residuals, 'mag')
    gyro = per_sensor_sigma(residuals, 'gyro')
    print(f"\n  measured per-sensor sigma: gyro {gyro:.4f} rad/s, acc {acc:.3f} m/s^2, "
          f"mag {mag:.4f}")

    row = shipped_row(surface, residuals)
    print(f"\n  DATASET_STDS row (ratios from the sweep; ties broken toward the measured "
          f"ratio, {row['n_tied']} point(s) tied within {TIE_FRACTION:.0%}; scale CHOSEN at "
          f"gyro_std = {SCALE_ANCHOR_GYRO_STD:g}, since the scale is inert):")
    print(f"    gyro_std {row['gyro_std']:.4g}   acc_std {row['acc_std']:.4g}   "
          f"mag_std {row['mag_std']:.4g}")
    print(f"    at acc/gyro {row['acc_ratio']:.3g}, mag/gyro {row['mag_ratio']:.3g} "
          f"-> {row['rmse_deg']:.2f} deg")
    print(f"    (the innovations alone imply acc/gyro {acc / gyro:.1f}, against the swept "
          f"{row['acc_ratio']:.1f}. The gyro innovation is measured against a mocap-"
          f"DIFFERENTIATED omega, is white at one sample and drops 10x under a 2 Hz "
          f"low-pass, so it is inflated by differentiation noise and the sweep is believed "
          f"over it. Run `--stage scale` to re-check that the scale is free.)")


# ==============================================================================
# Report
# ==============================================================================

def _h(n: int, title: str, question: str, requires: str) -> None:
    print(f"\n{'=' * 78}\nSECTION {n}. {title}\n{'=' * 78}")
    print(f"  QUESTION  {question}")
    print(f"  REQUIRES  {requires}")
    print()


def report(dataset: str, arm: str) -> None:
    """The experiment's findings for one (dataset, arm), in order of what you have to know.

    ORDERED BY INFORMATION, not by what was convenient to compute. Sections 1-2 use nothing a
    filter would not have on a new subject; section 3 is a property of the filter's algebra
    and needs no data at all; sections 4-5 need mocap and are therefore an UPPER BOUND on
    deployable performance rather than a description of it. Section 6 puts the two side by
    side, which is the number to quote.
    """
    residuals = load_statistics(f'filter_gain_residuals_{dataset}')
    surf_path = EXPERIMENT_DIR / dataset / f'{arm}_surface.csv'
    refn_path = EXPERIMENT_DIR / dataset / f'{arm}_refine.csv'
    scale_path = EXPERIMENT_DIR / dataset / f'{arm}_scale.csv'
    ladder = load_statistics(f'filter_gain_ladder_{dataset}_{arm}')
    surface = pd.read_csv(surf_path) if surf_path.exists() else None
    refined = pd.read_csv(refn_path) if refn_path.exists() else None

    print(f"\n\n{'#' * 78}\n#  FILTER GAIN SENSITIVITY — {dataset} / {arm}\n{'#' * 78}")
    if residuals is None:
        print("\nNo residuals on disk. Run --stage residuals first.")
        return
    n_trials = residuals.groupby(['subject', 'trial']).ngroups
    print(f"\n{n_trials} trials. Arm '{arm}' "
          + ("normalizes its vector measurements, so every std below is in UNIT-VECTOR space."
             if _normalizes(arm) else
             "uses raw-magnitude measurements, so every std below is in physical units."))

    _h(1, "WHAT DOES THE SENSOR DO WHEN NOTHING IS HAPPENING?",
       "If you tuned R from the sensor's own noise, what numbers would you use?",
       "a static recording. No mocap, no motion, no ground truth.")
    have = False
    for channel in ('gyro', 'acc', 'mag'):
        for kind, label in (('static_noise', 'static floor'), ('hf_signal', 'HF (>2 Hz) floor')):
            sub = residuals[(residuals.kind == kind) & (residuals.channel == channel)]
            if len(sub):
                have = True
                print(f"    {channel:5s} {label:18s} {np.median(sub.rms):.5f}")
    if not have:
        print("    (not measured — re-run --stage residuals)")
    print("\n  R does NOT model this. It is here as the null hypothesis, and section 6 prices it.")

    _h(2, "WHAT DOES THE MEASUREMENT EQUATION GET WRONG?",
       "What SHOULD R be? i.e. how wrong is the measurement when the state is exactly right?",
       "mocap. Substitutes the marker rotations into the filter's own residual.")
    for channel in ('gyro', 'acc', 'mag'):
        sigma = per_sensor_sigma(residuals, channel)
        if not np.isfinite(sigma):
            continue
        sub = residuals[(residuals.kind == 'static_noise') & (residuals.channel == channel)]
        ratio = sigma / np.median(sub.rms) if len(sub) else float('nan')
        auto = residuals[(residuals.channel == channel)
                         & (residuals.kind.isin(('per_sensor', 'pair')))].autocorr_s
        print(f"    {channel:5s} per-sensor {sigma:.4f}   = {ratio:5.1f}x its noise floor"
              f"   autocorrelation {np.median(auto):.3f} s")
    print("\n  The multiplier is the finding: R is model error, not sensor noise. An")
    print("  autocorrelation of order a second means that channel's error is a BIAS, which a")
    print("  Kalman filter cannot represent and must be defended against by inflating R.")

    _h(3, "HOW MANY KNOBS ARE THERE REALLY?",
       "Are the three stds three free parameters, or fewer?",
       "nothing — it is a property of the filter's algebra, confirmed by measurement.")
    print("    Scaling all three together cancels out of the Kalman gain, so the tuning is")
    print("    two-dimensional: acc_std/gyro_std and mag_std/gyro_std. The absolute scale is")
    print("    a presentation choice.")
    if scale_path.exists():
        sc = pd.read_csv(scale_path)
        for mode, g in sc.groupby('mag_mode'):
            g = g.sort_values('multiplier')
            floor = None
            for i in range(len(g)):
                tail = g.iloc[i:]
                spread = (tail.rmse_post_deg.max() - tail.rmse_post_deg.min()) / tail.rmse_post_deg.min()
                if spread < 0.01:
                    floor, flat = g.iloc[i].multiplier, spread
                    break
            print(f"    mag_{mode}: flat to {100*flat:.2f}% for every scaling above "
                  f"{floor:.0e}x the shipped tuning." if floor is not None else
                  f"    mag_{mode}: NEVER flat over the swept range — the scale is not free here.")
        print("\n    Below that floor P never escapes its seed within the trial, so the scale")
        print("    stops being free. The shipped tuning sits at 1x, above every floor.")
    else:
        print("    (scale check not run for this arm — --stage scale)")

    _h(4, "WHERE IS THE OPTIMUM, AND HOW SHARP IS IT?",
       "What is the best this filter can do on this data, at any tuning?",
       "mocap AND the scored metric. ACAUSAL — an upper bound, not a deployable answer.")
    for src, name in ((refined, 'refined 1.21x grid'), (surface, 'coarse 2.15x grid')):
        if src is None:
            continue
        for mode, g in src.groupby('mag_mode'):
            b = g.loc[g.rmse_deg.idxmin()]
            print(f"    {name:20s} mag_{mode:4s} {b.rmse_deg:7.2f} deg  at acc/gyro "
                  f"{b.acc_ratio:.3g}, mag/gyro {b.mag_ratio:.3g}")
        break
    if surface is not None:
        for mode, g in surface.groupby('mag_mode'):
            best = g.rmse_deg.min()
            within = g[g.rmse_deg < best * 1.10]
            print(f"    mag_{mode:4s} basin: {len(within)}/{len(g)} grid points within 10%, "
                  f"acc/gyro {within.acc_ratio.min():.3g}-{within.acc_ratio.max():.3g}")

    _h(5, "IS THE OPTIMUM STATIONARY?",
       "Would a filter that re-tuned itself during the trial do better than any fixed tuning?",
       "mocap, per window. ACAUSAL — bounds a dynamic scheme, does not implement one.")
    print("    Not measured by this stage. See the scratch analyses: on Al Borno a clairvoyant")
    print("    per-window oracle reaches -33% and a causal 'use the previous window's best'")
    print("    reaches -24%, while an actual signal-driven schedule recovered only -3.4%. The")
    print("    gap between the oracle and the schedule is unexplained and the control that")
    print("    would separate overfitting from a poor schedule family has not been run.")

    _h(6, "WHAT DOES EACH LEVEL OF PRIOR KNOWLEDGE COST?",
       "How much of the acausal optimum can you reach WITHOUT the data or its ground truth?",
       "each row is scored on the same trials and metric; the rows differ only in what they "
       "needed to know. 'legacy default' is the arbitrary triple this replaced -- a "
       "historical baseline, not a rung.")
    if ladder is None:
        print("    Not scored yet. Run --stage ladder.")
        return
    keyed = ladder.assign(_t=ladder.subject.astype(str) + '/' + ladder.trial.astype(str))
    for mode, g in keyed.groupby('mag_mode'):
        pooled = g.groupby('tuning').rmse_post_deg.mean()
        if 'swept optimum' not in pooled.index:
            continue
        ref = pooled['swept optimum']
        print(f"    mag_{mode}")
        print(f"      {'tuning':16s}{'RMSE':>9}{'vs optimum':>13}   information required")
        for name, requires in CANDIDATES:
            if name == 'legacy default' or name not in pooled.index:
                continue
            v = pooled[name]
            print(f"      {name:16s}{v:9.2f}{100*(v-ref)/ref:+12.1f}%   {requires}")
        # Printed apart from the ladder, and after it, because it is not a rung: an arbitrary
        # triple cannot say what prior knowledge is worth. 'datasheet' is the name earlier
        # runs stored it under, before that label was found to be a guess.
        legacy = next((n for n in ('legacy default', 'datasheet') if n in pooled.index), None)
        if legacy is not None:
            v = pooled[legacy]
            print(f"      {'-' * 62}")
            print(f"      {'was shipping':16s}{v:9.2f}{100*(v-ref)/ref:+12.1f}%   "
                  f"arbitrary triple, not an information rung")
        print()
    print("  Everything above 'innovation' is deployable on a subject who was never in a mocap")
    print("  lab. Everything from 'innovation' down is not. The distance between the best")
    print("  deployable row and 'swept optimum' is what acausal tuning is worth here.")


# ==============================================================================
# CLI
# ==============================================================================

def run_one(args, dataset: str, arm: str) -> None:
    """One (dataset, arm). Returns quietly when the combination is not measurable here --
    the caller sweeps every combination and a hard failure on one would abandon the rest."""
    args = argparse.Namespace(**{**vars(args), 'dataset': dataset, 'arm': arm})
    return _run_one_body(args)


def main():
    parser = argparse.ArgumentParser(
        description="Measure the filter's innovation and sweep the two free gain ratios. "
                    "Runs every dataset and every arm by default.",
        epilog="Every stage runs on every dataset it can. Where a dataset cannot support "
               "part of a stage -- the biplane half has no magnetometer, so mag_on is not a "
               "measurable arm there -- that part is dropped with a printed reason and the "
               "rest continues, rather than the whole invocation failing or, worse, the "
               "operator quietly never running it.")
    parser.add_argument('--datasets', nargs='+', default=sorted(DATASETS),
                        choices=sorted(DATASETS),
                        help="Defaults to EVERY registered dataset. A stage that cannot run "
                             "on one of them says so and the rest continue -- the previous "
                             "single-dataset default is how the scale check ended up "
                             "measured on two datasets out of four.")
    parser.add_argument('--stage',
                        choices=['residuals', 'sweep', 'both', 'surface', 'scale', 'refine',
                                 'ladder', 'report'],
                        default='both',
                        help="'surface' re-derives the pooled surface and the DATASET_STDS "
                             "row from the statistics files already on disk, running no "
                             "filters. It exists because the sweep is 40 minutes on IMoVE "
                             "and a change to how the surface is POOLED or read off should "
                             "not cost that.")
    parser.add_argument('--arms', nargs='+', default=ARMS, choices=ARMS,
                        help="'relative' is mag_on/mag_off/mag_adapt as shipped (projected, "
                             "raw-magnitude measurements); 'relative_normalized' is the same "
                             "filter with its vector measurements scaled to unit length, i.e. "
                             "the '_normalized' method arms; 'ekf' is each segment against the "
                             "virtual ground plate (normalized, unprojected). The two "
                             "normalizing arms report their ratios in UNIT-VECTOR space.")
    parser.add_argument('--mag-modes', default=None,
                        help="Defaults to 'on,off', or 'off' alone on a dataset with no "
                             "magnetometer.")
    parser.add_argument('--trials', type=int, default=None,
                        help="Cap on trials, for a quick look. Takes the first N built.")
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--center', default=None, metavar='ACC,MAG',
                        help="For --stage refine: centre the fine grid on this "
                             "(acc_std/gyro_std, mag_std/gyro_std) instead of the coarse "
                             "argmin. The pair itself is one of the grid points.")
    parser.add_argument('--span', type=float, default=REFINE_SPAN_DECADES,
                        help="For --stage refine: half-width of the fine grid, in decades.")
    parser.add_argument('--points', type=int, default=REFINE_POINTS,
                        help="For --stage refine: points per axis. Must be odd.")
    args = parser.parse_args()

    for dataset in args.datasets:
        # The residual stage is per dataset, not per arm: it measures the sensors, and the
        # arms differ only in what the filter does with them afterwards.
        arms = [None] if args.stage == 'residuals' else args.arms
        for arm in arms:
            print(f"\n{'=' * 78}\n{dataset}" + (f" / {arm}" if arm else "") +
                  f"  [{args.stage}]\n{'=' * 78}")
            try:
                run_one(args, dataset, arm or ARMS[0])
            except Exception as exc:
                print(f"  !! {dataset}/{arm}: {type(exc).__name__}: {exc}")


def _run_one_body(args):
    spec = tracking_spec(args.dataset)
    if args.mag_modes:
        mag_modes = args.mag_modes.split(',')
    elif not spec.has_magnetometer:
        mag_modes = ['off']
    elif _is_ekf(args.arm):
        # The shipped 'ekf' method has no mag_mode -- _joint_angles_from_ekf always runs the
        # magnetometer -- so an ekf/mag_off arm is not a thing the pipeline can produce. It
        # was being scored anyway, against a swept optimum that had never been measured for
        # it, which is where the nan reference column came from.
        mag_modes = ['on']
    else:
        mag_modes = ['on', 'off']
    if 'on' in mag_modes and not spec.has_magnetometer:
        # Dropped rather than raised: sweeping mag_ratio against an all-zero channel would run
        # 13 identical filters and report them as 13 measurements, but the OTHER modes on this
        # dataset are perfectly measurable and a raise here would lose them.
        print(f"  {args.dataset} has no magnetometer; dropping mag_mode 'on'.")
        mag_modes = [m for m in mag_modes if m != 'on']
        if not mag_modes:
            print("  nothing left to run.")
            return

    row_keys = built_trials(args.dataset)
    if args.trials:
        row_keys = row_keys[:args.trials]
    print(f"{args.dataset}: {len(row_keys)} trials, arm={args.arm}, mag_modes={mag_modes}")

    if args.stage == 'refine':
        coarse_path = EXPERIMENT_DIR / args.dataset / f"{args.arm}_surface.csv"
        if not coarse_path.exists():
            print(f"Need the coarse surface first: {coarse_path}. Run --stage sweep.")
            return
        coarse = pd.read_csv(coarse_path)
        centre = None
        if args.center:
            acc_c, mag_c = (float(v) for v in args.center.split(','))
            centre = (acc_c, mag_c)
            print(f"  centred on acc/gyro={acc_c:g}, mag/gyro={mag_c:g} (not the argmin)")
        grids = refine_grids(coarse, centre=centre, span=args.span, points=args.points)
        for mode, (acc, mag) in grids.items():
            print(f"  mag_{mode}: acc/gyro {acc[0]:.4g}..{acc[-1]:.4g} x{len(acc)}, "
                  f"mag/gyro {mag[0]:.4g}..{mag[-1]:.4g} x{len(mag)}")
        _, results = run_tracked_grid(
            row_keys, ['Subject', 'Trial'], list(spec.joints),
            partial(sweep_worker, dataset=args.dataset, arm=args.arm, grids=grids),
            args.workers, title=f"FILTER GAIN REFINE ({args.dataset}, {args.arm})")
        frames = [df for df in results.values() if df is not None and not df.empty]
        if not frames:
            print("No refinement results produced.")
            return
        fine = pd.concat(frames, ignore_index=True)
        # An explicitly centred run is a DIFFERENT measurement from the canonical refinement
        # and must not land on its filename: centring on a candidate tuning once silently
        # replaced the argmin-centred surface, and the two are not interchangeable -- the
        # centred grid is deliberately not positioned to find the optimum.
        tag = f"_at{args.center.replace(',', '_')}" if args.center else ""
        save_statistics(fine, f"filter_gain_refine_{args.dataset}_{args.arm}{tag}",
                        dataset=args.dataset, arm=args.arm, burn_in_s=BURN_IN_S,
                        refine_span_decades=args.span, refine_points=args.points,
                        centre=args.center)
        pooled = pool_surface(fine)
        out_path = paths.ensure_parent(
            EXPERIMENT_DIR / args.dataset / f"{args.arm}_refine{tag}.csv")
        pooled.to_csv(out_path, index=False)
        paths.write_manifest(out_path, dataset=args.dataset, arm=args.arm,
                             refine_span_decades=args.span,
                             refine_points=args.points, burn_in_s=BURN_IN_S,
                             centred_on=(f"explicit {args.center}" if args.center
                                         else str(coarse_path.relative_to(paths.REPO_ROOT))))
        print(f"\n=== {args.dataset} / {args.arm}: refined around the coarse optimum ===")
        check_refined(coarse, pooled)
        print(f"\nSaved refined surface to {out_path}")
        return

    if args.stage == 'report':
        report(args.dataset, args.arm)
        return

    if args.stage == 'ladder':
        residuals = load_statistics(f'filter_gain_residuals_{args.dataset}')
        surf = EXPERIMENT_DIR / args.dataset / f'{args.arm}_refine.csv'
        if residuals is None or not surf.exists():
            print(f"  needs residuals and {surf.name}; run --stage both then --stage refine.")
            return
        tunings = candidate_tunings(args.dataset, args.arm, residuals, pd.read_csv(surf),
                                    mag_modes)
        for mode, rungs in tunings.items():
            print(f"  mag_{mode}")
            for name, t in rungs.items():
                print(f"    {name:16s} gyro {t['gyro_std']:.5g}  acc {t['acc_std']:.5g}  "
                      f"mag {t['mag_std']:.5g}")
        _, results = run_tracked_grid(
            row_keys, ['Subject', 'Trial'], list(spec.joints),
            partial(ladder_worker, dataset=args.dataset, arm=args.arm, mag_modes=mag_modes,
                    tunings=tunings),
            args.workers, title=f"FILTER GAIN LADDER ({args.dataset}, {args.arm})")
        frames = [df for df in results.values() if df is not None and not df.empty]
        if not frames:
            print("No ladder results produced.")
            return
        out = pd.concat(frames, ignore_index=True)
        save_statistics(out, f"filter_gain_ladder_{args.dataset}_{args.arm}",
                        dataset=args.dataset, arm=args.arm, burn_in_s=BURN_IN_S,
                        candidates={n: r for n, r in CANDIDATES})
        report(args.dataset, args.arm)
        return

    if args.stage == 'scale':
        base = stds_for_arm(resolve_stds(dataset=args.dataset), args.arm)
        print(f"Scaling {base} by {SCALE_MULTIPLIERS.min():g}..{SCALE_MULTIPLIERS.max():g} "
              f"at fixed ratios")
        _, results = run_tracked_grid(
            row_keys, ['Subject', 'Trial'], list(spec.joints),
            partial(scale_worker, dataset=args.dataset, arm=args.arm, mag_modes=mag_modes,
                    base=base),
            args.workers, title=f"FILTER GAIN SCALE CHECK ({args.dataset}, {args.arm})")
        frames = [df for df in results.values() if df is not None and not df.empty]
        if not frames:
            print("No scale results produced.")
            return
        scale = pd.concat(frames, ignore_index=True)
        save_statistics(scale, f"filter_gain_scale_{args.dataset}_{args.arm}",
                        dataset=args.dataset, arm=args.arm,
                        multipliers=SCALE_MULTIPLIERS.tolist(), base_stds=base,
                        burn_in_s=BURN_IN_S)
        pooled = pool_scale(scale)
        out_path = paths.ensure_parent(EXPERIMENT_DIR / args.dataset / f"{args.arm}_scale.csv")
        pooled.to_csv(out_path, index=False)
        paths.write_manifest(out_path, dataset=args.dataset, arm=args.arm, base_stds=base,
                             multipliers=SCALE_MULTIPLIERS.tolist(), burn_in_s=BURN_IN_S)
        print(f"\n=== {args.dataset} / {args.arm}: scale check at fixed ratios ===")
        report_scale(pooled)
        print(f"\nSaved scale table to {out_path}")
        return

    if args.stage == 'surface':
        sweep = load_statistics(f"filter_gain_sweep_{args.dataset}_{args.arm}")
        residuals = load_statistics(f"filter_gain_residuals_{args.dataset}")
        if sweep is None or residuals is None:
            print(f"Need both filter_gain_sweep_{args.dataset}_{args.arm} and "
                  f"filter_gain_residuals_{args.dataset} on disk. Run --stage both first.")
            return
        _write_surface(pool_surface(sweep), residuals, args.dataset, args.arm,
                       mag_modes, int(sweep.groupby(['subject', 'trial']).ngroups))
        return

    residuals = pd.DataFrame()
    if args.stage in ('residuals', 'both'):
        _, results = run_tracked_grid(
            row_keys, ['Subject', 'Trial'], ['residuals'],
            partial(_residual_worker, dataset=args.dataset), args.workers,
            title=f"FILTER GAIN RESIDUALS ({args.dataset})")
        frames = [df for df in results.values() if df is not None and not df.empty]
        if frames:
            residuals = pd.concat(frames, ignore_index=True)
            save_statistics(residuals, f"filter_gain_residuals_{args.dataset}",
                            dataset=args.dataset, lp_cutoffs=list(LP_CUTOFFS))
            print(f"\n=== {args.dataset}: innovation at the true state ===")
            for channel in ('gyro', 'acc', 'mag'):
                sigma = per_sensor_sigma(residuals, channel)
                if np.isfinite(sigma):
                    print(f"  {channel}: per-sensor sigma {sigma:.4f}")

    if args.stage in ('sweep', 'both'):
        stages = list(spec.joints)
        _, results = run_tracked_grid(
            row_keys, ['Subject', 'Trial'], stages,
            partial(sweep_worker, dataset=args.dataset, arm=args.arm,
                    grids=coarse_grids(mag_modes)),
            args.workers, title=f"FILTER GAIN SWEEP ({args.dataset}, {args.arm})")
        frames = [df for df in results.values() if df is not None and not df.empty]
        if not frames:
            print("No sweep results produced.")
            return
        sweep = pd.concat(frames, ignore_index=True)
        save_statistics(sweep, f"filter_gain_sweep_{args.dataset}_{args.arm}",
                        dataset=args.dataset, arm=args.arm,
                        acc_ratios=ACC_RATIOS.tolist(), mag_ratios=MAG_RATIOS.tolist(),
                        sweep_gyro_ref=SWEEP_GYRO_REF, burn_in_s=BURN_IN_S)

        _write_surface(pool_surface(sweep), residuals, args.dataset, args.arm,
                       mag_modes, len(row_keys))


def _write_surface(surface: pd.DataFrame, residuals: pd.DataFrame, dataset: str, arm: str,
                   mag_modes: List[str], n_trials: int) -> None:
    out_path = paths.ensure_parent(EXPERIMENT_DIR / dataset / f"{arm}_surface.csv")
    surface.to_csv(out_path, index=False)
    paths.write_manifest(out_path, dataset=dataset, arm=arm, mag_modes=mag_modes,
                         n_trials=n_trials, acc_ratios=ACC_RATIOS.tolist(),
                         mag_ratios=MAG_RATIOS.tolist(), sweep_gyro_ref=SWEEP_GYRO_REF,
                         burn_in_s=BURN_IN_S, tie_fraction=TIE_FRACTION)
    print(f"\n=== {dataset} / {arm}: pooled surface ===")
    report_surface(surface, residuals if residuals is not None and not residuals.empty else None)
    print(f"\nSaved surface to {out_path}")


def _residual_worker(row_key, stage_labels: List[str], shared_state: Dict,
                     dataset: str) -> pd.DataFrame:
    subject, trial = row_key
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    try:
        df = measure_residuals(dataset, subject, trial)
    except Exception as exc:
        shared_state[(row_key, stage)] = f"Failed ({type(exc).__name__})"
        return pd.DataFrame()
    shared_state[(row_key, stage)] = "Success" if not df.empty else "Skipped"
    return df


if __name__ == '__main__':
    main()
