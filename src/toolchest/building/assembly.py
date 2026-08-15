"""
Traces in, synchronized PlateTrials out.

The load-time half of PlateTrial: resampling onto a common grid, estimating the trial's lag,
placing both streams on one timeline, and solving the sensor-to-segment rotation. All of it
used to live on the class, which meant the data structure also owned the procedure for
building itself out of two unaligned recordings.

`assemble_plate_trials` is the entry point; the rest is its machinery.
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import scipy.signal as signal
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

from ..IMUTrace import IMUTrace
from ..PlateTrial import PlateTrial
from ..WorldTrace import WorldTrace
from ..gyro_utils import calculate_best_fit_rotation
from .report import BuildReport

# Robust spread (MAD) of the per-plate lag estimates above which the cross-correlation is
# finding different peaks rather than one noisy peak, making their median meaningless.
# Generous by three orders of magnitude: on real trials the plates agree to 0-1 samples,
# i.e. under 25 ms. It fires on 19 of IMoVE's 26 static poses -- a subject standing still
# gives two noise traces and no peak to correlate on -- while the other 7 have enough
# residual sway to sync, so this discriminates rather than blanket-rejecting them.
SYNC_SPREAD_LIMIT_S = 1.0


def _angle_to_nearest_axis(axis: np.ndarray, norm: float) -> float:
    """Angle from a rotation axis to the closest coordinate axis, in degrees.

    Separates a mounting convention from a fit that latched onto something else: a sensor
    clipped into a bracket the wrong way round is near an axis, an arbitrary direction is not.
    """
    if norm <= 1e-12:
        return 0.0
    unit = axis / norm
    return float(np.degrees(np.arccos(np.clip(np.abs(unit).max(), 0.0, 1.0))))


def align_world_to_imu(plate: 'PlateTrial',
                       report: 'BuildReport' = None) -> 'PlateTrial':
    """
    Aligns the WorldTrace's orientation to the IMUTrace's orientation.

    This method calculates the static rotational offset between the "ground truth"
    angular velocity (from WorldTrace) and the measured angular velocity
    (from IMUTrace). It then applies this offset to the WorldTrace's
    orientation data so that the coordinate frames are aligned.

    This is a crucial step for sensor-to-segment calibration.

    Returns:
        PlateTrial: A new PlateTrial object with the aligned WorldTrace.
    """
    # 1. Calculate a synthetic IMU gyro trace from the world (mocap) rotations.
    synthetic_imu_trace = plate.world_trace.calculate_imu_trace(skip_lin_acc=True)

    # 2. Find the rotation (R_wt_it) that maps the world trace to the imu trace
    #    by comparing their gyroscope data — over VALID frames only.
    #
    #    What this does and does not buy, measured rather than assumed:
    #
    #    HELD padding is already inert. A Procrustes fit accumulates H = sum a_i b_i^T, and a
    #    held pose has identically zero synthetic gyro, so those rows add nothing to H whether
    #    they are masked or not. Subject01's 430 s of pre-mocap padding was never competing
    #    with its 603 s of overlap, contrary to what this comment used to claim.
    #
    #    INTERPOLATED frames are the real hazard, and they are what `valid` marks. A failed
    #    reconstruction does not stop moving; it moves wrongly, so its rows are large and
    #    point the wrong way. Those genuinely drag the rotation, which is why the fit is
    #    restricted rather than left to average them out.
    valid = np.asarray(plate.valid)
    if valid.all():
        R_wt_it = synthetic_imu_trace.calculate_rotation_offset_from_gyros(plate.imu_trace)
    elif valid.any():
        R_wt_it = calculate_best_fit_rotation(synthetic_imu_trace.gyro[valid],
                                              plate.imu_trace.gyro[valid])
    else:
        raise ValueError(f"{plate.name}: no valid frames to align against.")
    
    if report is not None:
        # The rotation itself, which nothing has ever looked at. An offset near a plate axis
        # is a mounting convention; an arbitrary one is a fit that found something else. And
        # `n_frames_used` is the conditioning caveat: a static pose yields a small residual
        # and a meaningless rotation, which the residual alone cannot distinguish.
        angle = float(np.degrees(np.arccos(
            np.clip((np.trace(R_wt_it) - 1.0) / 2.0, -1.0, 1.0))))
        axis = Rotation.from_matrix(R_wt_it).as_rotvec()
        norm = float(np.linalg.norm(axis))
        report.add('S7_alignment', 'plate', plate.name,
                   offset_angle_deg=angle,
                   offset_axis=(axis / norm if norm > 1e-12 else np.zeros(3)),
                   angle_to_nearest_plate_axis_deg=_angle_to_nearest_axis(axis, norm),
                   n_frames_used=int(valid.sum()),
                   valid_fraction=float(valid.mean()))

    # 3. Apply this static rotation to all orientations in the world trace.
    #    new_R_world = old_R_world @ R_wt_it
    world_rots_np = plate.world_trace.rotations
    new_world_rotations = np.matmul(world_rots_np, R_wt_it)

    # 4. Create a new WorldTrace and PlateTrial with the aligned data.
    #    The validity mask carries through unchanged: applying a constant rotation to
    #    every frame cannot make a corrupt pose trustworthy or the reverse.
    new_world_trace = WorldTrace(plate.world_trace.timestamps, plate.world_trace.positions,
                                 new_world_rotations, valid=plate.world_trace.valid)
    return PlateTrial(plate.name, plate.imu_trace, new_world_trace)


def assemble_plate_trials(
    imu_traces: Dict[str, 'IMUTrace'],
    world_traces: Dict[str, 'WorldTrace'],
    align_plate_trials: bool,
    lag: float = None,
    report: 'BuildReport' = None,
) -> Dict[str, 'PlateTrial']:
    """
    Factory method to create a list of PlateTrial objects from raw data.

    This function performs the key "data wrangling" steps:
    1.  Finds matching pairs of IMU and World traces.
    2.  Resamples IMU data if its frequency doesn't match the World trace.
    3.  Synchronizes the traces in time using cross-correlation of gyro norms.
    4.  (Optionally) Aligns the coordinate frames.
    5.  Trims all resulting trials to the same minimum length for consistency.

    Args:
        imu_traces (Dict[str, 'IMUTrace']): A dictionary mapping IMU names
            to IMUTrace objects.
        world_traces (Dict[str, 'WorldTrace']): A dictionary mapping segment
            names to WorldTrace objects.
        align_plate_trials (bool): If True, performs sensor-to-segment
            alignment using `_align_world_trace_to_imu_trace`.

    Returns:
        List['PlateTrial']: A list of processed, synchronized, and aligned
            PlateTrial objects.
    """
    plate_trials = {}

    if not imu_traces:
        # {} not [], to match the annotation and the other empty return below. A caller
        # doing .items() on this got AttributeError instead of an empty result.
        if os.getenv("DISABLE_TQDM", "False") != "True":
            print("Warning: No IMU traces loaded.")
        return {}

    disable_tqdm = os.getenv("DISABLE_TQDM", "False") == "True"
    if not disable_tqdm:
        print("Processing and synchronizing traces...")

    paired = {name: (imu, world_traces[name])
              for name, imu in imu_traces.items() if name in world_traces}
    for name in imu_traces:
        if name not in world_traces and not disable_tqdm:
            print(f"IMU {name} not found in world traces. Skipping.")
    if report is not None:
        report.add('S3_pairing', 'trial', 'trial',
                   n_imu=len(imu_traces), n_world=len(world_traces), n_paired=len(paired),
                   unmatched_imu=sorted(set(imu_traces) - set(world_traces)),
                   unmatched_world=sorted(set(world_traces) - set(imu_traces)))
    if not paired:
        return {}

    # The trial runs at the SLOWEST rate any of its streams was recorded at. Everything
    # above that rate in the faster stream is content the slower one cannot be compared
    # against, and keeping it means differencing two signals that live in different bands.
    #
    # The old code did the opposite -- it upsampled the IMU onto the mocap grid -- which
    # invented 20-50 Hz IMU content that was never measured. Against the 40 Hz sessions
    # that biased the IMoVE cluster-to-IMU offset by 7-9 mm on the shank. Downsampling the
    # mocap instead also REDUCES noise, because a proper anti-alias filter averages over
    # its support: 100 -> 40 Hz measures 1.62x quieter.
    target_rate = min(min(imu.get_sample_frequency(), world.get_sample_frequency())
                      for imu, world in paired.values())

    # Estimate the lag BEFORE decimating, on the finer grid, and to sub-sample precision.
    # Decimating first would quantise the search to the target period -- 25 ms at 40 Hz --
    # and that resolution cannot be recovered afterwards.
    #
    # A caller may pass `lag` instead, meaning "these world traces are already on the IMU's
    # clock". IMoVE's long-walk sessions need that: one inertial record spans three separate
    # mocap takes at roughly 2 s, 1756 s and 3564 s into it, so there is no single lag to
    # find. Their reader syncs each take itself and merges them, then passes lag=0.
    if lag is None:
        lag = _shared_lag(paired, report=report, disable_tqdm=disable_tqdm)

    if report is not None:
        rates = sorted({round(float(rate), 6)
                        for imu, world in paired.values()
                        for rate in (imu.get_sample_frequency(),
                                     world.get_sample_frequency())})
        report.add('S5_resampling', 'trial', 'trial', target_rate_hz=target_rate,
                   n_distinct_source_rates=len(rates),
                   source_rates_hz=rates,
                   fastest_source_rate_hz=max(rates))

    resampled = {}
    for name, (imu, world) in paired.items():
        resampled[name] = _to_common_grid(imu, world, target_rate, lag,
                                          name=name, report=report)

    # One origin for the trial, applied to every plate. See _trial_origin: doing this per
    # plate silently puts joint-angle pairs on clocks that differ by a few samples.
    origin = _trial_origin(resampled, report=report)
    resampled = {name: _rezero(imu, world, origin)
                 for name, (imu, world) in resampled.items()}

    # Trimming to a common length only means anything once the heads agree, which is what
    # the shared origin above guarantees.
    n_frames = min(len(imu) for imu, _ in resampled.values())

    for name, (imu, world) in tqdm(resampled.items(), desc="Generating PlateTrials",
                                   disable=disable_tqdm):
        new_plate_trial = PlateTrial(name, imu[:n_frames], world[:n_frames])
        if align_plate_trials:
            new_plate_trial = align_world_to_imu(new_plate_trial, report=report)
        plate_trials[name] = new_plate_trial

    _assert_one_clock(plate_trials)

    if not disable_tqdm:
        print(f"Successfully generated {len(plate_trials)} PlateTrials.")
    return plate_trials


def _assert_one_clock(plates: Dict[str, PlateTrial]) -> None:
    """Every plate in a trial must be on the same timeline, to the sample.

    PlateTrial.__init__ already checks a plate's IMU against its own world trace. Nothing
    checked ACROSS plates, which is where the shared-clock invariant actually lives and where
    it was being broken. Cheap enough to assert on every build.
    """
    if len(plates) < 2:
        return
    reference_name, reference = next(iter(plates.items()))
    for name, plate in plates.items():
        if len(plate) != len(reference):
            raise ValueError(f"{name} has {len(plate)} frames but {reference_name} has "
                             f"{len(reference)}; plates in a trial share one timeline.")
        drift = float(np.max(np.abs(plate.imu_trace.timestamps
                                    - reference.imu_trace.timestamps)))
        if drift > 1e-8:
            raise ValueError(f"{name} and {reference_name} are on clocks that differ by up "
                             f"to {drift * 1000:.1f} ms. Joint angles difference plates by "
                             f"index, so this would silently compare offset samples.")


def shift_world_origin(plate: PlateTrial, offset_m: np.ndarray) -> PlateTrial:
    """Moves the world trace's origin onto the IMU, given the offset in the SENSOR frame.

    The mocap describes a marker cluster; the accelerometer describes itself. Those are not
    the same point, and on IMoVE they are 10-25 mm apart, so a plate's own mocap-implied
    acceleration disagrees with its accelerometer by a lever-arm term reaching 1-3 m/s^2 at
    running excitation. Translating the pose here makes the PlateTrial describe ONE point,
    after which calculate_imu_trace predicts what the sensor should actually read and no
    consumer has to know the offset exists.

    Only positions move. Rotations are untouched, so joint angles -- which are differences of
    rotations -- are bit-identical, and no orientation result changes.

    MUST RUN AFTER align_world_to_imu, because `offset_m` is in the sensor frame and the
    rotations only map sensor -> world once that alignment has been applied.

    The offset is a constant, so this is exactly reversible: re-measuring with
    PlateTrial.fit_sensor_offset on a shifted plate returns the offset that REMAINS, which is
    zero when the constant is right and the error when it is not. That closed loop is how the
    constant is checked, rather than by comparing it against a stored copy of itself.
    """
    offset_m = np.asarray(offset_m, dtype=np.float64)
    # Adding zero cannot change either the pose or the accumulated total, so return the plate
    # itself rather than a numerically-identical copy. The iterative fit relies on this: its
    # last step is small by construction and there is no reason to rebuild the arrays for it.
    if not offset_m.any():
        return plate

    rotations = np.asarray(plate.world_trace.rotations)
    positions = (np.asarray(plate.world_trace.positions)
                 + np.einsum('nij,j->ni', rotations, offset_m))
    shifted = PlateTrial(plate.name, plate.imu_trace,
                         WorldTrace(plate.world_trace.timestamps, positions, rotations,
                                    valid=plate.world_trace.valid))
    # ACCUMULATED, not replaced, so shifting twice records the total. That matters because
    # the fit is applied iteratively: it reads low, so one pass lands short of the truth.
    #
    # Recorded so the shift is auditable and, more importantly, so a re-derivation can add it
    # back. fit_sensor_offset on a shifted plate returns what REMAINS, and a refit that pasted
    # that in as the new constant would drive it to zero one run at a time.
    shifted.sensor_offset = np.asarray(plate.sensor_offset) + offset_m
    return shifted


def _to_common_grid(imu_trace: IMUTrace, world_trace: WorldTrace, target_rate: float,
                    lag: float, name: str = '', report: 'BuildReport' = None
                    ) -> Tuple[IMUTrace, WorldTrace]:
    """Puts one plate's two streams on a single grid at `target_rate`, t = 0 at first overlap.

    The IMU defines the timeline, decimated to `target_rate` if it was faster. It is never
    padded, so every inertial sample in the result is a real measurement and nothing
    downstream has to ask whether one is genuine. That matters more than it sounds: trimming
    to the overlap used to throw away 430 s of Subject01's walking trial and 80% of the
    stationary pauses the sensor noise floor is estimated from, and it cannot express IMoVE's
    long-walk trials at all, where one inertial recording spans three mocap takes.

    The world trace is then resampled DIRECTLY ONTO that grid, offset by `lag`. This is what
    replaces the old index arithmetic, and it is strictly better in two ways: the lag can be
    fractional, and the mocap is band-limited to the target rate on the way rather than
    subsampled raw. Where the grid runs off the end of the mocap record the interpolator
    extrapolates, so `valid` -- which resample_mask already sets False outside the source
    span -- is the only thing marking those frames, exactly as before.

    Still on the IMU's OWN clock when it returns. Re-zeroing is deliberately not done here:
    t = 0 has to be one instant for the whole trial, and this function only ever sees one
    plate. `_trial_origin` and `_rezero` do it afterwards, across all of them.
    """
    imu_rate = imu_trace.get_sample_frequency()
    world_rate = world_trace.get_sample_frequency()
    if report is not None:
        report.add('S5_resampling', 'plate', name,
                   imu_rate_hz=imu_rate, world_rate_hz=world_rate,
                   target_rate_hz=target_rate,
                   imu_decimated=not np.isclose(imu_rate, target_rate),
                   world_decimated=not np.isclose(world_rate, target_rate),
                   n_imu_samples_in=len(imu_trace),
                   n_world_samples_in=len(world_trace),
                   **_power_above(imu_trace, world_trace, target_rate))

    if not np.isclose(imu_rate, target_rate):
        imu_trace = imu_trace.resample(target_rate)

    # World clock -> IMU clock. resample_mask marks anything outside the mocap's own span
    # invalid, so extrapolated frames are flagged rather than silently trusted.
    world_times = imu_trace.timestamps - lag
    sampled = _world_on_timestamps(world_trace, world_times, target_rate)

    if not np.any(sampled.valid):
        raise ValueError("IMU and mocap records do not overlap at all.")

    # Relabelled onto the IMU's clock. `world_times` were only ever the instants to SAMPLE
    # the mocap at; the samples themselves belong to the IMU frames that asked for them, and
    # PlateTrial requires the two to agree to 1e-8.
    return imu_trace, WorldTrace(imu_trace.timestamps, sampled.positions,
                                 sampled.rotations, valid=sampled.valid)


def _power_above(imu_trace: IMUTrace, world_trace: WorldTrace,
                 target_rate: float) -> Dict[str, float]:
    """What fraction of each stream's power sits above the target Nyquist.

    Decimating to the slower stream's rate is the right call -- see the comment at the
    `target_rate` line -- but "right" is not "free", and the cost was never measured.

    BOTH STREAMS, because which one pays depends on the session. The 17-sensor IMoVE
    recordings put the IMU at 40 Hz and the mocap at 100, so the target is 40 and it is the
    MOCAP that gets decimated; the IMU is untouched and simply never captured anything above
    20 Hz in the first place, which is a property of the recording rather than of this
    pipeline. The 100 Hz long-walk sessions are the other way round. Measuring only the IMU
    would have reported nothing at all for the sessions that make up most of the dataset.

    THE MEAN IS REMOVED FIRST. Gravity is a ~9.81 DC term on the accelerometer and a marker's
    position is offset from the origin by metres; either would otherwise be most of the total
    and make every fraction look like a rounding error, regardless of how much genuine
    high-frequency content was thrown away.

    Reported per channel group rather than per axis: the question is how much of the signal
    is lost, and a per-axis breakdown of that is a plot, not a build metric.

    WHAT IT MEASURES, ON THE DATA IN HAND: the IMU is never the decimated stream. Every trial
    in both datasets is either all-100 Hz or a 40 Hz IMU against 100 Hz mocap, so the target
    is the IMU's own rate and only the mocap is ever resampled down. The mocap loses between
    5e-9 and 2e-5 of its power -- nothing, because Motive's output is already smoothed well
    below 20 Hz. So the decimation step itself is close to free, and the real band limit is
    the 40 Hz RECORDING rate, which is a property of the session rather than of this code.
    The acc/gyro fractions are kept anyway: they are what would catch a future dataset where
    that stops being true, and their absence is itself the finding.
    """
    fractions = {}
    for label, values, rate in (
            ('acc', imu_trace.acc, imu_trace.get_sample_frequency()),
            ('gyro', imu_trace.gyro, imu_trace.get_sample_frequency()),
            ('mocap_position', world_trace.positions,
             world_trace.get_sample_frequency())):
        fraction = _fraction_above(values, rate, target_rate)
        if fraction is not None:
            fractions[f'{label}_power_above_target_nyquist'] = fraction
    return fractions


def _fraction_above(values: np.ndarray, rate: float, target_rate: float):
    """Fraction of `values`' mean-removed power above `target_rate / 2`, or None."""
    # np.isclose, matching the guard that decides whether to decimate at all. A stream whose
    # measured rate differs from the target in the twelfth decimal is not decimated, and
    # reporting a meaningless ~0 fraction for it would put a row in the table for every plate
    # in every same-rate trial.
    finite = values[np.isfinite(values).all(axis=1)]
    if len(finite) < 4 or target_rate >= rate or np.isclose(rate, target_rate):
        return None
    power = np.abs(np.fft.rfft(finite - finite.mean(axis=0), axis=0)) ** 2
    above = np.fft.rfftfreq(len(finite), d=1.0 / rate) > target_rate / 2.0
    total = power.sum()
    return float(power[above].sum() / total) if total > 0 else 0.0


def _trial_origin(resampled: Dict[str, Tuple[IMUTrace, WorldTrace]],
                  report: 'BuildReport' = None) -> float:
    """The one instant that becomes t = 0, shared by every plate in the trial.

    THIS IS AN INVARIANT, not a convenience. Joint angles difference two plates against each
    other by index, so if their clocks disagree the difference is taken across an offset and
    labelled with whichever plate happened to iterate first. Computing the origin per plate --
    which is what this used to do, inside `_to_common_grid` -- breaks it whenever per-plate
    validity is ragged at the head of a trial. That is routine, not exotic: it fired on
    imove/s16/t0_static_pose_001, where four plates started 350 ms (14 samples) after the
    other eleven and nothing complained.

    The EARLIEST first-valid frame wins, so t = 0 is the first instant any plate has ground
    truth. Taking the latest instead would let one ragged plate drag the whole trial's clock
    forward and push every other plate's good data into negative time.
    """
    firsts, named = [], {}
    for name, (imu_trace, world_trace) in resampled.items():
        observed = np.flatnonzero(world_trace.valid)
        if len(observed):
            first = float(imu_trace.timestamps[observed[0]])
            firsts.append(first)
            named[name] = first
    if not firsts:
        raise ValueError("No plate has any overlap between its IMU and mocap records.")
    origin = min(firsts)

    if report is not None:
        # The spread is the s16 defect as a routine measurement: when plates disagree about
        # when ground truth starts, the ones that start late spend that long on a held pose.
        for name, first in named.items():
            report.add('S6_timeline', 'plate', name, first_valid_time_s=first,
                       deviation_from_origin_s=first - origin)
        report.add('S6_timeline', 'trial', 'trial', origin_s=origin,
                   origin_spread_s=max(firsts) - origin, n_plates_with_overlap=len(firsts),
                   n_plates_without_overlap=len(resampled) - len(firsts))
    return origin


def _rezero(imu_trace: IMUTrace, world_trace: WorldTrace, origin: float
            ) -> Tuple[IMUTrace, WorldTrace]:
    """Slides both streams so `origin` becomes t = 0. Inertial data before it goes negative."""
    timestamps = imu_trace.timestamps - origin
    return (IMUTrace(timestamps, imu_trace.gyro, imu_trace.acc, imu_trace.mag),
            WorldTrace(timestamps, world_trace.positions, world_trace.rotations,
                       valid=world_trace.valid))


def _world_on_timestamps(world_trace: WorldTrace, new_timestamps: np.ndarray,
                         target_rate: float) -> WorldTrace:
    """Band-limited resample of a WorldTrace onto an explicit (possibly offset) grid."""
    from ..resampling import resample_mask, resample_rotations, resample_values

    source_rate = world_trace.get_sample_frequency()
    return WorldTrace(
        new_timestamps,
        resample_values(world_trace.positions, world_trace.timestamps, new_timestamps,
                        source_rate=source_rate, target_rate=target_rate),
        resample_rotations(world_trace.rotations, world_trace.timestamps, new_timestamps,
                           source_rate=source_rate, target_rate=target_rate),
        valid=resample_mask(world_trace.valid, world_trace.timestamps, new_timestamps,
                            source_rate=source_rate, target_rate=target_rate),
    )


def _shared_lag(paired: Dict[str, Tuple[IMUTrace, WorldTrace]],
                report: 'BuildReport' = None,
                disable_tqdm: bool = True) -> float:
    """One lag in SECONDS for the whole trial: world clock + lag = IMU clock.

    Every plate in a trial comes from ONE inertial recording session and ONE mocap take, so
    physically there is a single lag. Estimating it per plate and taking the median is
    therefore a better estimator than any individual plate, not a compromise -- measured
    across this dataset the per-plate estimates agree to 0-1 samples.

    It is also required rather than merely tidy. Each plate's t = 0 has to be the same point
    on a shared clock, because joint angles difference two plates against each other.
    """
    lags = np.array([_lag_seconds(imu, world) for imu, world in paired.values()])
    lag = float(np.median(lags))

    if report is not None:
        # Per plate, because the trial-level MAD says a trial is fine without saying which
        # plate dragged it. A plate whose own estimate sits far from the median has a bad
        # gyro trace even when the median absorbs it and the trial builds cleanly.
        for name, value in zip(paired, lags):
            report.add('S4_sync', 'plate', name,
                       lag_s=float(value), deviation_from_median_s=float(value - lag))

    # Robust spread, not max-minus-min. One plate finding the wrong peak is exactly what the
    # median is there to absorb, and a range test throws away that protection: s24's
    # t4_lat_step has fourteen plates agreeing at 1.9 s and one at -5.7, which is a bad plate,
    # not a failed sync. The MAD ignores it; it only grows when the estimates are scattered
    # rather than clustered.
    deviation = np.abs(lags - lag)
    spread = float(1.4826 * np.median(deviation))
    if spread > SYNC_SPREAD_LIMIT_S:
        # Past this the plates are not disagreeing about one peak, they are each finding a
        # different one, and the median of that estimates nothing. The usual cause is a trial
        # with no motion to correlate on: IMoVE's t0_static_pose is a subject standing still,
        # so both gyro traces are noise and the peak lands wherever. s24's estimates span
        # 16 s. Raising here says THAT, instead of letting a bogus lag slide the mocap off
        # the end of the record and resurface as a confusing "records do not overlap".
        raise ValueError(
            f"Sync failed: per-plate lag estimates are scattered, not clustered "
            f"(robust spread {spread:.1f} s; min {lags.min():.1f}, median {lag:.1f}, "
            f"max {lags.max():.1f}). One recording session has one lag, so most likely the "
            f"trial has too little motion for gyro cross-correlation to find a peak.")

    if report is not None:
        report.add('S4_sync', 'trial', 'trial', lag_median_s=lag, lag_mad_s=spread,
                   spread_vs_limit=spread / SYNC_SPREAD_LIMIT_S, n_plates=len(lags),
                   lag_min_s=float(lags.min()), lag_max_s=float(lags.max()))

    outliers = int((deviation > SYNC_SPREAD_LIMIT_S).sum())
    if outliers and not disable_tqdm:
        print(f"Warning: {outliers} of {len(lags)} plates disagree with the trial lag by "
              f"over {SYNC_SPREAD_LIMIT_S:.0f} s. The median ({lag:.2f} s) is unaffected, "
              f"but those plates' own gyro traces are worth a look.")
    return lag


def _lag_seconds(imu_trace: IMUTrace, world_trace: WorldTrace) -> float:
    """Sub-sample time offset between one plate's two streams, by gyro-magnitude correlation.

    Run on the FINER of the two grids so the correlation peak is as well resolved as the data
    allows, then refined below the sample period by fitting a parabola to the peak and its
    two neighbours. Integer-sample alignment leaves up to half a sample of residual error,
    which against a 40 Hz IMU is 12.5 ms -- enough to matter during a jump landing, where the
    signals being compared change substantially over that interval.
    """
    fine_rate = max(imu_trace.get_sample_frequency(), world_trace.get_sample_frequency())
    if not np.isclose(imu_trace.get_sample_frequency(), fine_rate):
        imu_trace = imu_trace.resample(fine_rate)
    if not np.isclose(world_trace.get_sample_frequency(), fine_rate):
        world_trace = world_trace.resample(fine_rate)

    synthetic = world_trace.calculate_imu_trace(skip_lin_acc=True)
    imu_signal = np.linalg.norm(imu_trace.gyro, axis=1)
    world_signal = np.linalg.norm(synthetic.gyro, axis=1)

    shift = _correlation_peak(imu_signal - imu_signal.mean(),
                              world_signal - world_signal.mean())
    # `shift` is how far the world signal sits AFTER the IMU signal, in fine-grid samples.
    return imu_trace.timestamps[0] - world_trace.timestamps[0] + shift / fine_rate


def _correlation_peak(a: np.ndarray, b: np.ndarray) -> float:
    """Lag of a against b, in samples, refined below the sample period."""
    correlation = signal.correlate(a, b, mode='full', method='fft')
    peak = int(np.argmax(correlation))

    # Three points around a correlation maximum are locally quadratic, so the vertex of the
    # parabola through them estimates where the true peak falls between samples. Guarded
    # against a peak at either end, where there is no parabola to fit.
    offset = 0.0
    if 0 < peak < len(correlation) - 1:
        left, middle, right = correlation[peak - 1:peak + 2]
        denominator = left - 2 * middle + right
        if denominator != 0:
            offset = np.clip(0.5 * (left - right) / denominator, -0.5, 0.5)
    return peak + offset - (len(b) - 1)


def _sync_arrays(array1: np.ndarray, array2: np.ndarray) -> Tuple[slice, slice]:
    """
    Finds the optimal lag between two 1D arrays using cross-correlation.

    Args:
        array1 (np.ndarray): The first 1D array.
        array2 (np.ndarray): The second 1D array.

    Returns:
        Tuple[slice, slice]: A pair of slice objects (slice1, slice2)
            that trim the arrays to their overlapping, synchronized portions.
    """
    assert array1.ndim == array2.ndim == 1, "Input arrays must be 1D"
    
    # Pad the shorter array to match the longer one for correlation
    max_len = max(len(array1), len(array2))
    a1 = np.pad(array1, (0, max_len - len(array1)), mode='constant')
    a2 = np.pad(array2, (0, max_len - len(array2)), mode='constant')

    # Compute the full cross-correlation using FFT for speed
    correlation = signal.correlate(a1, a2, mode='full', method='fft')
    
    # Find the index of the peak correlation.
    # The lag is this index offset by (max_len - 1)
    lag = np.argmax(correlation) - (max_len - 1)

    # Calculate the start indices and new length for slicing
    # If lag is positive, array1 starts later (trim its start)
    # If lag is negative, array2 starts later (trim its start)
    i1 = max(0, lag)
    i2 = max(0, -lag)
    new_len = min(len(array1) - i1, len(array2) - i2)

    # Return the slice objects
    return slice(i1, i1 + new_len), slice(i2, i2 + new_len)
