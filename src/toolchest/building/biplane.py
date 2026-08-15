"""IMoVE's biplane half: MC10 IMUs against biplane-fluoroscopy bone poses.

This is the OTHER half of IMoveLab_Raw_Data, and it shares almost nothing with `mocap_ref`
beyond a subject roster. Where the mocap half gives marker clusters at 100 Hz for a minute,
this gives direct bone poses at 150 Hz for half a second -- the gold standard, and almost none
of it.

FOUR THINGS DIFFER FROM EVERY OTHER READER HERE, and each one is a trap.

  ABSOLUTE TIME, THREE CLOCKS. The IMUs are MC10 BioStamps timestamping in GMT epoch
  microseconds; Vicon's trigger log is in US EASTERN LOCAL time; the biplane camera has a
  clock of its own that is neither. Verified rather than assumed -- see `_utc_offset_hours`
  and `camera_clock_offset`.

  NO MAGNETOMETER. A BioStamp has an accelerometer and a gyroscope and nothing else, so every
  magnetometer-dependent path in this repo is inapplicable to these trials. `mag` is filled
  with zeros and `HAS_MAGNETOMETER` says so out loud, because silently-zero magnetometer data
  would sail through a filter and produce a heading.

  GROUND TRUTH IS ALREADY A POSE. The biplane pipeline outputs homogeneous transforms per
  bone, so there is no marker reconstruction: no template fit, no un-flip pass, no
  `S2_reconstruction`. The pose is what the fluoroscopy solved for.

  THE GROUND TRUTH IS 0.48 SECONDS LONG against a two-hour inertial record. That is not a
  defect to work around -- it is what `PlateTrial.valid` exists for. A trial here is one long
  IMUTrace whose mask is True for 73 frames.

UNITS AND CONVENTIONS, all checked against the files:
  accel        g          -> m/s^2
  gyro         deg/s      -> rad/s
  timestamps   epoch us   -> seconds since the trial's t=0
  transforms   ROW-VECTOR, so p_lab = p_bone @ M and the ordinary rotation is M[:3,:3].T.
               The translation sits in ROW 3, not column 3, and is in millimetres. Taking the
               top-left block as the rotation directly yields its transpose, which is a
               silent inverse; `Lab-to-Tibia` is on disk beside `Tibia-to-Lab` and the two
               compose to the identity, which is how the convention was established.
"""
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import paths

from ..IMUTrace import IMUTrace
from ..resampling import resample_values
from ..PlateTrial import PlateTrial
from ..WorldTrace import WorldTrace

BIPLANE_ROOT = paths.DATA_DIR / 'IMoveLab_Raw_Data' / 'biplane_ref' / 'data'
STUDY = 'HAKnee'

# A BioStamp measures acceleration and rotation only. Named rather than left implicit so a
# caller can refuse rather than quietly fuse a zero field.
HAS_MAGNETOMETER = False

GRAVITY_MS2 = 9.80665
MM_TO_M = 1e-3

# Bone -> the IMU site that sits on it. The biplane solves femur and tibia; the BioStamps are
# on the lateral thigh and shank, so the pairing is one-to-one per side.
BONE_TO_SITE = {'Femur': 'lateral_thigh', 'Tibia': 'lateral_shank'}

# Trials are named <side><task><n>: LSDrop2 is a left single-leg drop, RrunStance1 a right
# running stance. The leading letter picks which knee the biplane imaged, and therefore which
# two IMU sites the poses can be paired with.
_TRIAL_PATTERN = re.compile(r'^(?P<side>[LR])(?P<task>[A-Za-z]+?)(?P<index>\d+)$')

# Half-width of the search around the trigger time.
#
# WIDE, because the trigger is not merely quantised -- it is WRONG BY SECONDS. The log is
# whole-second (every value ends in .000), which alone would justify a bracket of about one
# second, and that reasoning produced a 6 s bracket that could not reach the answer. Measured
# on 12/LDDrop3 the true lag is -11.53 s, and the correlation finds it identically at bracket
# widths of 49 s and 289 s (|omega| correlation 0.86, against 0.21 for the best lag inside
# 6 s). So the trigger locates the trial within the two-hour record and nothing finer.
TRIGGER_BRACKET_S = 30.0

# Margin kept either side of the trigger when slicing the two-hour inertial record. Must
# comfortably exceed TRIGGER_BRACKET_S plus the Vicon capture length, or the correlation runs
# out of record before it runs out of bracket.
IMU_MARGIN_S = 60.0


def _utc_offset_hours(when: pd.Timestamp) -> int:
    """Hours to add to a Vicon computer-clock time to reach UTC.

    The log is US Eastern LOCAL time, so the offset is 4 during daylight time and 5 outside
    it. Determined here from the date rather than hardcoded, because the study spans September
    to November and crosses the changeover: subject 01 (September) needs +4 and subject 12
    (November) needs +5. Each was verified by checking that the shifted trigger span falls
    inside that subject's inertial record, which it does for exactly one of the two.
    """
    eastern = when.tz_localize('US/Eastern', ambiguous=True, nonexistent='shift_forward')
    return int(round(-eastern.utcoffset().total_seconds() / 3600.0))


@lru_cache(maxsize=1)
def trigger_times() -> pd.DataFrame:
    """The trigger workbook, restricted to this study and parsed onto real clocks.

    Returns one row per (subject, trial) with `vicon_utc` -- the trigger instant on the same
    clock the IMUs timestamp in -- and the raw camera time, which is on a third clock and is
    only useful as a difference.
    """
    frame = pd.read_excel(BIPLANE_ROOT / 'TriggerTimes.xlsx')
    # The workbook writes the subject once and leaves the rest of that block blank.
    frame['Proj/Sub/Date'] = frame['Proj/Sub/Date'].ffill()
    frame = frame[frame['Proj/Sub/Date'].astype(str).str.startswith(STUDY)].copy()

    frame['subject'] = frame['Proj/Sub/Date'].str.extract(r'Subject(\d+)')
    frame['trial'] = frame['Trial'].astype(str).str.strip()
    frame['camera'] = pd.to_datetime(frame['Biplane Trigger Time (Camera Clock)'],
                                     errors='coerce')
    frame['vicon_local'] = pd.to_datetime(frame['Vicon Trigger Time (Computer Clock)'],
                                          errors='coerce')
    frame = frame.dropna(subset=['subject', 'camera', 'vicon_local'])

    offsets = frame['vicon_local'].map(_utc_offset_hours)
    frame['vicon_utc'] = frame['vicon_local'] + pd.to_timedelta(offsets, unit='h')
    return frame[['subject', 'trial', 'camera', 'vicon_local', 'vicon_utc']]


def camera_clock_offset(subject: str) -> Optional[float]:
    """Seconds to subtract from a camera-clock time to reach the Vicon clock, per subject.

    PER SUBJECT and not global: the difference is stable to well under a second within a
    session (median absolute deviation 0.2-0.8 s for most subjects) but ranges from 4h00m to
    5h18m BETWEEN them, so the camera clock is evidently reset or drifts between sessions.
    A single global constant would be wrong by up to an hour.

    The median over that subject's ~29 trigger rows, so one mistyped row cannot move it.
    """
    rows = trigger_times()
    rows = rows[rows.subject == subject]
    if rows.empty:
        return None
    return float((rows.camera - rows.vicon_utc).dt.total_seconds().median())


# ------------------------------------------------------------------------------------ IMUs

def read_mc10_imu(sensor_dir: Path,
                  window: Optional[Tuple[float, float]] = None) -> Optional[IMUTrace]:
    """One BioStamp directory -> an IMUTrace on the UTC epoch clock, in SI units.

    `accel.csv` and `gyro.csv` are separate exports with their own timestamps, so the gyro is
    interpolated onto the accelerometer's grid rather than assumed to share it.

    `window` is (start, stop) in epoch SECONDS and is not an optimisation: these records run
    two hours at 250 Hz, and a trial needs half a second of that. Reading whole files for all
    four sensors would be 8 million samples to reach 120.
    """
    accel_path, gyro_path = sensor_dir / 'accel.csv', sensor_dir / 'gyro.csv'
    if not accel_path.exists() or not gyro_path.exists():
        return None

    accel = pd.read_csv(accel_path)
    gyro = pd.read_csv(gyro_path)
    accel_time = accel.iloc[:, 0].to_numpy(np.float64) / 1e6
    gyro_time = gyro.iloc[:, 0].to_numpy(np.float64) / 1e6

    if window is not None:
        keep = (accel_time >= window[0]) & (accel_time <= window[1])
        if keep.sum() < 2:
            return None
        accel, accel_time = accel[keep], accel_time[keep]
        # The gyro is windowed too, with a margin, so the interpolant has support either side
        # of the accelerometer's first and last sample instead of extrapolating to reach them.
        margin = 1.0
        gyro_keep = ((gyro_time >= accel_time[0] - margin)
                     & (gyro_time <= accel_time[-1] + margin))
        if gyro_keep.sum() < 8:
            return None
        gyro, gyro_time = gyro[gyro_keep], gyro_time[gyro_keep]

    acc = accel.iloc[:, 1:4].to_numpy(np.float64) * GRAVITY_MS2
    raw_gyro = np.radians(gyro.iloc[:, 1:4].to_numpy(np.float64))

    # BAND-LIMITED, not linear. The two exports are independent streams on their own
    # timestamps, so the gyro has to be put on the accelerometer's grid somehow -- and doing
    # that with np.interp is the same mistake this repo already paid for once. Linear
    # interpolation is not band-limited: its response is sinc^2, which cost -1.2 dB at 8 Hz
    # when IMUTrace.resample used it, and the resulting band mismatch biased the IMoVE
    # cluster-to-IMU offset by 7-9 mm on the shank. The upstream pipeline interpolates these
    # two channels linearly as well, so this is a divergence from it on purpose.
    resampled = resample_values(raw_gyro, gyro_time, accel_time,
                                source_rate=1.0 / float(np.median(np.diff(gyro_time))),
                                target_rate=1.0 / float(np.median(np.diff(accel_time))))

    return IMUTrace(accel_time, resampled, acc, np.zeros_like(acc))


def imu_sites(subject: str) -> List[str]:
    directory = BIPLANE_ROOT / 'IMUs' / STUDY / subject
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir()
                  if p.is_dir() and (p / 'accel.csv').exists())


# -------------------------------------------------------------------------------- biplane

def read_biplane_pose(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A HomoTransMatrices csv -> (times, positions in metres, rotations).

    ROW-VECTOR CONVENTION. The file stores M such that p_lab = p_bone @ M, so the rotation
    taking bone coordinates to lab coordinates is M[:3, :3].T and the translation is ROW 3.
    Reading the top-left block as the rotation gives its transpose -- a silent inverse that
    would leave every angle plausible and wrong. Established by checking that the shipped
    `Lab-to-Tibia` and `Tibia-to-Lab` compose to the identity as stored, which they do.
    """
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    columns = [f'[{row}][{column}]' for row in range(4) for column in range(4)]
    if not set(columns).issubset(frame.columns):
        return None

    matrices = frame[columns].to_numpy(np.float64).reshape(-1, 4, 4)
    finite = np.isfinite(matrices).all(axis=(1, 2))
    if finite.sum() < 2:
        return None

    times = frame['Time'].to_numpy(np.float64)[finite]
    matrices = matrices[finite]
    rotations = matrices[:, :3, :3].transpose(0, 2, 1)
    positions = matrices[:, 3, :3] * MM_TO_M
    return times, positions, rotations


def biplane_trials(subject: str) -> List[Tuple[str, str, str]]:
    """(session, block, trial) for every biplane capture this subject has."""
    root = BIPLANE_ROOT / 'Kinematics' / STUDY / subject
    if not root.is_dir():
        return []
    found = []
    for session in sorted(p for p in root.iterdir() if p.is_dir()):
        for block in sorted(p for p in session.iterdir() if p.is_dir()):
            for trial in sorted(p for p in block.iterdir() if p.is_dir()):
                if any(trial.glob('HomoTransMatrices_*-to-Lab.csv')):
                    found.append((session.name, block.name, trial.name))
    return found


def trial_side(trial: str) -> Optional[str]:
    """'left' or 'right' from the trial name, or None if it does not parse."""
    match = _TRIAL_PATTERN.match(trial)
    if match is None:
        return None
    return {'L': 'left', 'R': 'right'}[match.group('side')]


def trial_task(trial: str) -> Optional[str]:
    """The activity from the trial name -- 'SDrop', 'runStance', 'static' -- or None.

    Case is left alone. The names mix conventions (`LSDrop2` against `RrunStance1`), so
    normalizing here would only move the problem to whoever compares against a literal.
    """
    match = _TRIAL_PATTERN.match(trial)
    return None if match is None else match.group('task')


# ---------------------------------------------------------------------------------- Vicon

# The 4-marker clusters, one per segment per side. Names decode as
# <side><segment><superior|inferior><posterior|anterior>, so RTSP is the right thigh's
# superior-posterior marker. Ordered consistently across sides so a template estimated on one
# is comparable with the other.
VICON_CLUSTERS = {
    ('right', 'lateral_thigh'): ['RTSP', 'RTSA', 'RTIA', 'RTIP'],
    ('left', 'lateral_thigh'): ['LTSP', 'LTSA', 'LTIA', 'LTIP'],
    ('right', 'lateral_shank'): ['RSSP', 'RSSA', 'RSIA', 'RSIP'],
    ('left', 'lateral_shank'): ['LSSP', 'LSSA', 'LSIA', 'LSIP'],
}


def read_vicon_c3d(path: Path, labels: List[str]
                   ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """((N, M, 3) marker positions in METRES, (N,) seconds), or None if a label is absent.

    c3d stores points as (4, markers, frames) with the fourth row a residual: negative means
    the point was not reconstructed in that frame. Those become NaN here, which is what
    fit_plate_to_template treats as a gap -- leaving them at their last value would look like
    a stationary marker rather than a missing one.
    """
    import ezc3d

    handle = ezc3d.c3d(str(path))
    names = [name.strip() for name in handle['parameters']['POINT']['LABELS']['value']]
    index = {name: position for position, name in enumerate(names)}
    if not set(labels).issubset(index):
        return None

    points = handle['data']['points']                    # (4, markers, frames)
    rate = float(handle['header']['points']['frame_rate'])
    units = handle['parameters']['POINT']['UNITS']['value']
    scale = MM_TO_M if units and units[0].strip().lower() == 'mm' else 1.0

    columns = []
    for label in labels:
        marker = points[:3, index[label], :].T * scale
        residual = points[3, index[label], :]
        marker[residual < 0] = np.nan
        columns.append(marker)
    return np.stack(columns, axis=1), np.arange(points.shape[2]) / rate


def vicon_path(subject: str, session: str, trial: str) -> Path:
    """Vicon files key the subject as 'Subject12' where every other tree uses '12'."""
    return BIPLANE_ROOT / 'Vicon' / STUDY / f'Subject{subject}' / session / f'{trial}.c3d'


# ----------------------------------------------------------------------------------- sync

def _valid_span(world_trace: WorldTrace) -> WorldTrace:
    """The trace trimmed to its first..last valid frame, or unchanged if it has no mask.

    Returns the whole trace when nothing is valid, so a caller gets the same "no peak" answer
    it would have got anyway rather than an empty array to reason about.
    """
    valid = np.asarray(world_trace.valid)
    if valid.all() or not valid.any():
        return world_trace
    first, last = int(np.argmax(valid)), int(len(valid) - np.argmax(valid[::-1]))
    return WorldTrace(world_trace.timestamps[first:last],
                      world_trace.positions[first:last],
                      world_trace.rotations[first:last],
                      valid=valid[first:last])


def bracketed_lag(imu_trace: IMUTrace, world_trace: WorldTrace, expected_lag_s: float,
                  bracket_s: float = TRIGGER_BRACKET_S) -> Tuple[float, float]:
    """Refine a trigger-derived lag by gyro correlation, searching only near it.

    Returns (lag in seconds on the IMU clock, peak-to-sidelobe ratio).

    THE BRACKET IS NOT AN OPTIMISATION. Correlating a 0.48 s biplane window against a
    two-hour record at 250 Hz is 1.8 million lags, and a half-second pattern of a drop landing
    will match somewhere in two hours by chance. The trigger says roughly when; the
    correlation says exactly when; neither is sufficient alone -- the Vicon log is quantised
    to whole seconds, which is longer than the biplane window it is supposed to locate.

    The peak-to-sidelobe ratio comes back with the answer because a periodic signal -- gait,
    hopping -- produces near-equal peaks one cycle apart, and a lag chosen from those is
    confidently wrong. A caller that ignores it gets no warning.
    """
    from scipy import signal as scipy_signal

    # CORRELATE ON MEASURED FRAMES ONLY. The Vicon clusters lose markers heavily -- across 72
    # clusters only 65% of frames see all four, and 24% see fewer than the three a pose needs
    # -- and `fit_plate_to_template` fills those by interpolation. Differentiating an
    # interpolated pose gives an angular velocity that was never measured, and feeding it to a
    # correlation is what made the two sensors of a trial disagree about the lag by up to 40 s.
    #
    # Trimming to the valid span rather than masking inside it, because EVERY ONE of the 52
    # lost runs sits at a trial edge -- median 529 frames, 2.1 s, none in the interior. So the
    # valid part is contiguous and a correlation over it needs no holes punched in it. The
    # returned lag is unchanged in meaning: it is measured against the trimmed trace's own
    # first timestamp, and L = s - w0 is invariant to where w0 is taken.
    #
    # This is the same hazard `assembly.align_world_to_imu` already masks for when it fits the
    # sensor-to-segment rotation; the lag search simply never did.
    world_trace = _valid_span(world_trace)

    rate = float(world_trace.get_sample_frequency())
    resampled = imu_trace.resample(rate)
    reference = np.linalg.norm(
        world_trace.calculate_imu_trace(skip_lin_acc=True).gyro, axis=1)
    measured = np.linalg.norm(resampled.gyro, axis=1)
    if len(reference) < 4 or len(measured) < len(reference):
        return float('nan'), float('nan')

    correlation = scipy_signal.correlate(measured - measured.mean(),
                                         reference - reference.mean(),
                                         mode='valid', method='fft')
    starts = resampled.timestamps[:len(correlation)]
    lags = starts - world_trace.timestamps[0]

    inside = np.abs(lags - expected_lag_s) <= bracket_s
    if not inside.any():
        return float('nan'), float('nan')

    candidates = np.flatnonzero(inside)
    best = candidates[int(np.argmax(correlation[candidates]))]

    # Sidelobe = the best peak at least half a reference-length away, which is the nearest
    # place a periodic signal would put a competing match.
    exclusion = max(int(0.5 * len(reference)), 1)
    far = candidates[np.abs(candidates - best) > exclusion]
    peak = float(correlation[best])
    sidelobe = float(correlation[far].max()) if len(far) else 0.0
    ratio = peak / sidelobe if sidelobe > 0 else float('inf')
    return float(lags[best]), ratio


# The biplane Time column is NOT relative to the Vicon trigger: it starts a constant ~2.98 s
# earlier, which is presumably the fluoroscopy system's pre-trigger buffer. Measured at
# 2.98 +/- 0.06 s over 17 of 18 trials across three subjects; the eighteenth landed on a
# bracket edge, which is what a failed search looks like rather than a real value.
#
# This matters because assuming zero -- which is the obvious reading of a column called
# "Time" -- puts the biplane window 3 s from the truth, and the correlation that should catch
# that has a peak-to-sidelobe ratio of 1.31 against a two-hour record, so it does not.
BIPLANE_PRETRIGGER_S = 2.98
BIPLANE_PRETRIGGER_BRACKET_S = 0.5

# Below this, the correlation peak is not meaningfully better than its neighbours and the lag
# it names should not be trusted. Recorded per plate rather than enforced, because the honest
# evidence for these alignments is that they REPRODUCE across trials, not that any single
# peak is sharp: the ratios run 1.05-5.38 while the answers agree to 60 ms.
MIN_PEAK_TO_SIDELOBE = 1.5

# The window-landed check. See `_check_windows_landed`: the IMU and the reference, over the
# same frames, must at least agree about how much the limb was moving.
#
# Set from the measured distribution. On subject 12 alone the ratio looked cleanly bimodal --
# 1.00-2.75 for everything that synced and 5.88-23.70 for the one trial that did not -- and
# 4.0 sat in an empty gap.
#
# ACROSS ALL 15 SUBJECTS THE GAP IS NOT EMPTY, so that reading was too confident. Over 1462
# plates the ratio runs q50 1.27, q95 2.46, and the tail is
#
#     (2, 3]    88 plates        (4, 6]      7 plates
#     (3, 4]     8 plates        (6, 10]     3 plates
#                                (10, 124]  29 plates
#
# The mass above 10 is unambiguous and the bulk below 3 is clearly fine; 4.0 now separates 8
# unflagged plates from 7 flagged ones rather than nothing from nothing. It stays where it is
# because the alternative is worse in both directions -- tightening to 3 sweeps in the 88
# plates at 2-3, which are drop landings where soft tissue alone moves the ratio, and loosening
# to 10 gives up the 10 plates that are genuinely misplaced. The 15 in 3-10 are judgement, and
# `motion_ratio` is recorded per plate so that judgement can be revisited on the numbers.
#
# It fires on 39 of 1462 plates, 2.7%, concentrated in 14 trials across 8 subjects.
#
# DELIBERATELY NOT TIGHTER. Soft-tissue artifact, differentiation noise on a 1 s window and
# the sensor-to-segment rotation itself all move this ratio, and none of them is a sync
# failure. LDDrop2 reads 3.06 and is genuinely poor -- its alignment residual is the worst of
# any non-broken plate at 2.2x signal -- but poor is not misplaced, and
# `residual_fraction_of_signal` is the instrument for that. This check answers one question:
# did the window land where the IMU was doing the same thing.
MAX_SYNC_MOTION_RATIO = 4.0
# Below this the RMS is dominated by whatever few frames survived, and the ratio is noise.
MIN_SYNC_CHECK_FRAMES = 30


class SyncWindowWarning(UserWarning):
    """The reference window does not match the IMU over the frames it claims."""


def _vicon_world(subject: str, session: str, trial: str, side: str, site: str
                 ) -> Optional[WorldTrace]:
    """One Vicon marker cluster, reconstructed the same way every other dataset's is."""
    from .reconstruction import fit_plate_to_template

    labels = VICON_CLUSTERS.get((side, site))
    if labels is None:
        return None
    markers = read_vicon_c3d(vicon_path(subject, session, trial), labels)
    if markers is None:
        return None
    positions, timestamps = markers
    try:
        pose, rotations, valid, _ = fit_plate_to_template(
            positions, timestamps, name=f'{subject}/{trial}/{site}')
    except ValueError:
        return None
    return WorldTrace(timestamps, pose, rotations, valid=valid)


def _biplane_world(subject: str, session: str, block: str, trial: str,
                   bone: str) -> Optional[WorldTrace]:
    pose = read_biplane_pose(
        BIPLANE_ROOT / 'Kinematics' / STUDY / subject / session / block / trial /
        f'HomoTransMatrices_{bone}-to-Lab.csv')
    if pose is None:
        return None
    times, positions, rotations = pose
    return WorldTrace(times, positions, rotations)


def load_trial(subject: str, key: str, report=None) -> Dict[str, PlateTrial]:
    """One biplane capture -> PlateTrials on the IMU clock, keyed '<site>__<reference>'.

    `key` is '<session>/<block>/<trial>', e.g. 'Test1/B/LDDrop3'.

    THE SYNC IS CHAINED, and the order is not arbitrary -- it was established by making the
    two ground truths check each other and watching the obvious arrangement fail:

      1. Vicon -> IMU by gyro correlation, bracketed by the trigger time. Ten seconds of
         signal against a two-hour record; the best-conditioned link there is.
      2. Biplane -> Vicon by gyro correlation inside that ten seconds, with a tight bracket
         around BIPLANE_PRETRIGGER_S. Both streams are 150 Hz optical measurements of the same
         limb, so this is the easiest match in the chain.
      3. Compose.

    Aligning biplane straight to the IMU instead -- one step rather than two -- is the version
    that fails: a 0.48 s transient against 1.8 million lags returns a peak only 1.31x its
    nearest rival and lands 3 s out. Two ground truths agreeing is what caught it.

    Both references are emitted, suffixed, because a consumer has to be able to tell which one
    it is holding: biplane is the gold standard over 0.48 s, Vicon covers ten seconds, and
    silently mixing them would compare a bone pose against a skin-mounted cluster. The short
    coverage needs no special handling -- `valid` already exists for exactly this, so a
    biplane plate is a long IMUTrace whose mask is True for 73 frames.
    """
    session, block, trial = key.split('/')
    side = trial_side(trial)
    if side is None:
        raise ValueError(f"{trial}: cannot tell which knee this is from the name.")

    rows = trigger_times()
    row = rows[(rows.subject == subject) & (rows.trial == trial)]
    if row.empty:
        raise ValueError(f"{subject}/{trial}: no trigger time, so no clock to sync onto.")
    trigger = float(row.iloc[0].vicon_utc.timestamp())

    # ONE Vicon lag for the trial, not one per sensor. There is a single trigger and a single
    # Vicon capture, so the lag is a property of the trial; estimating it per sensor and using
    # each answer separately let the thigh and shank disagree by 3.5 s on the first attempt,
    # which is physically impossible and left the gyro residual larger than the signal.
    # Median over the sensors, for the same reason assembly._shared_lag takes one.
    sources, lags, ratios = {}, [], []
    for bone, site in BONE_TO_SITE.items():
        sensor = f'{site}_{side}'
        imu = read_mc10_imu(BIPLANE_ROOT / 'IMUs' / STUDY / subject / sensor,
                            window=(trigger - IMU_MARGIN_S, trigger + IMU_MARGIN_S))
        vicon = _vicon_world(subject, session, trial, side, site)
        if imu is None or vicon is None:
            continue
        on_imu = WorldTrace(vicon.timestamps + trigger, vicon.positions, vicon.rotations,
                            valid=vicon.valid)
        lag, ratio = bracketed_lag(imu, on_imu, 0.0)
        sources[sensor] = (bone, site, imu, vicon)
        if np.isfinite(lag):
            lags.append(lag)
            ratios.append(ratio)

    if not sources:
        return {}
    vicon_lag = float(np.median(lags)) if lags else 0.0
    vicon_start = trigger + vicon_lag

    # ONE origin AND ONE GRID for the trial. A shared origin is not sufficient on its own:
    # the four BioStamps free-run, so their sample instants differ by a couple of milliseconds
    # and subtracting the same origin still leaves each plate on its own timestamps. Joint
    # angles difference two plates BY INDEX, so the grid has to be shared too -- which is what
    # assembly._to_common_grid does for the mocap datasets.
    origin = vicon_start
    rate = float(np.median([imu.get_sample_frequency() for _, _, imu, _ in sources.values()]))
    start = max(float(imu.timestamps[0]) for _, _, imu, _ in sources.values())
    stop = min(float(imu.timestamps[-1]) for _, _, imu, _ in sources.values())
    grid = start + np.arange(int((stop - start) * rate)) / rate
    sources = {sensor: (bone, site, _on_grid(imu, grid, rate), vicon)
               for sensor, (bone, site, imu, vicon) in sources.items()}

    plates: Dict[str, PlateTrial] = {}
    for sensor, (bone, site, imu, vicon) in sources.items():
        plates[f'{sensor}__vicon'] = _as_plate(
            f'{sensor}__vicon', imu,
            WorldTrace(vicon.timestamps + vicon_start, vicon.positions, vicon.rotations,
                       valid=vicon.valid), origin)

        biplane = _biplane_world(subject, session, block, trial, bone)
        if biplane is None:
            continue
        # Matched in VICON'S OWN time base, both traces starting near zero. Correlating the
        # absolute-epoch Vicon trace against the relative biplane one put the true lag 1.6e9
        # outside the bracket, so the search silently returned nothing and the constant was
        # used unchecked -- which is exactly the sort of quiet fallback the peak-to-sidelobe
        # figure exists to expose.
        biplane_lag, biplane_ratio = bracketed_lag(
            _as_imu(vicon), biplane, BIPLANE_PRETRIGGER_S,
            bracket_s=BIPLANE_PRETRIGGER_BRACKET_S)
        if not np.isfinite(biplane_lag):
            biplane_lag, biplane_ratio = BIPLANE_PRETRIGGER_S, float('nan')

        plates[f'{sensor}__biplane'] = _as_plate(
            f'{sensor}__biplane', imu,
            WorldTrace(biplane.timestamps + vicon_start + biplane_lag, biplane.positions,
                       biplane.rotations, valid=biplane.valid), origin)

        if report is not None:
            report.add('S4_sync', 'plate', sensor,
                       trigger_utc=trigger, vicon_lag_s=vicon_lag,
                       vicon_lag_spread_s=float(np.ptp(lags)) if len(lags) > 1 else 0.0,
                       vicon_peak_to_sidelobe=float(np.median(ratios)) if ratios else np.nan,
                       biplane_lag_s=biplane_lag,
                       biplane_peak_to_sidelobe=biplane_ratio,
                       biplane_sync_weak=bool(not np.isfinite(biplane_ratio)
                                              or biplane_ratio < MIN_PEAK_TO_SIDELOBE))

    _check_windows_landed(plates, report)
    return plates


def _check_windows_landed(plates: Dict[str, PlateTrial], report=None) -> None:
    """Does the reference window sit where the IMU was actually doing the same thing?

    THE PEAK-TO-SIDELOBE RATIO DOES NOT CATCH THIS. It scores how sharp the correlation peak
    was, which says the estimator was confident, not that it was right -- and a confident
    wrong answer is what this is for. On 12/Test1/A/LSHop3 the biplane window landed on a
    stretch where the IMU reads 9.4 deg/s while the reference-derived angular velocity over
    the same frames reads 222.1 deg/s. A twenty-fold disagreement about whether the limb was
    moving at all is not soft-tissue artifact or a differentiation artifact; the window is
    simply in the wrong place, and nothing said so.

    Compared as a RATIO of RMS magnitudes, which is frame-independent -- no alignment has
    been applied yet, so the two are in different frames and only their magnitudes can be
    compared. That also makes the check blind to a rotation error, which is deliberate: this
    asks whether the window landed, and `residual_fraction_of_signal` asks whether the
    rotation is right.

    A warning rather than a raise. A trial whose sync failed still has an IMU record and a
    correctly-placed second reference, and dropping it outright would lose those too.
    """
    for name, plate in plates.items():
        valid = np.asarray(plate.valid)
        if valid.sum() < MIN_SYNC_CHECK_FRAMES:
            continue
        measured = float(np.degrees(
            np.linalg.norm(plate.imu_trace.gyro[valid], axis=1)).mean())
        synthetic = float(np.degrees(np.linalg.norm(
            plate.world_trace.calculate_imu_trace(skip_lin_acc=True).gyro[valid],
            axis=1)).mean())
        if min(measured, synthetic) <= 0:
            continue
        ratio = max(measured, synthetic) / min(measured, synthetic)
        landed = ratio <= MAX_SYNC_MOTION_RATIO
        if report is not None:
            report.add('S4_sync', 'plate', name,
                       measured_gyro_rms_deg_s=measured,
                       reference_gyro_rms_deg_s=synthetic,
                       motion_ratio=ratio, window_landed=landed)
        if not landed:
            warnings.warn(
                f"{name}: the reference window does not match the IMU over the same frames "
                f"-- measured {measured:.1f} deg/s against {synthetic:.1f} from the "
                f"reference, a factor of {ratio:.1f}. The lag estimate has almost certainly "
                f"put the window in the wrong place; treat this plate as unsynced.",
                SyncWindowWarning, stacklevel=2)


def _on_grid(imu: IMUTrace, grid: np.ndarray, rate: float) -> IMUTrace:
    """Resample one sensor onto the trial's shared grid, band-limited."""
    from ..resampling import resample_values

    return IMUTrace(grid, *(resample_values(getattr(imu, channel), imu.timestamps, grid,
                                            source_rate=imu.get_sample_frequency(),
                                            target_rate=rate)
                            for channel in ('gyro', 'acc', 'mag')))


def _as_imu(world: WorldTrace) -> IMUTrace:
    """A WorldTrace's synthetic IMU, so it can stand in as the 'measured' side of a match."""
    synthetic = world.calculate_imu_trace(skip_lin_acc=True)
    return IMUTrace(world.timestamps, synthetic.gyro, synthetic.acc, synthetic.mag)


def _as_plate(name: str, imu: IMUTrace, world: WorldTrace, origin: float,
              align: bool = True) -> PlateTrial:
    """Put a short ground truth on the IMU's timeline, t = 0 at its first valid frame.

    The world trace is resampled onto the IMU grid rather than the other way round, so every
    inertial sample stays a real measurement. Outside the ground truth's span `valid` is
    False, which is the whole mechanism that lets 0.48 s of biplane sit inside a 40 s record
    without anything downstream needing to know.

    `origin` IS PASSED IN, not chosen here. Letting each plate re-zero at its own first valid
    frame gives a trial four different clocks -- measured at 48.5 to 51.9 s apart on
    12/LDDrop3 -- and joint angles difference two plates by index, so they would silently be
    taken across that offset. This is the same invariant assembly._trial_origin exists to
    hold, and writing a second reader reintroduced the same bug.
    """
    from .assembly import _world_on_timestamps

    rate = float(imu.get_sample_frequency())
    placed = _world_on_timestamps(world, imu.timestamps, rate)
    timestamps = imu.timestamps - origin
    plate = PlateTrial(name, IMUTrace(timestamps, imu.gyro, imu.acc, imu.mag),
                       WorldTrace(timestamps, placed.positions, placed.rotations,
                                  valid=placed.valid))

    # The sensor-to-segment rotation, which every other dataset gets from
    # assemble_plate_trials. Skipping it leaves the mocap-derived and measured gyros in
    # DIFFERENT FRAMES, so their vector difference is meaningless however good the timing is
    # -- it read as a residual two to eight times the signal, which looks like a broken sync
    # and is not one. Magnitudes are frame-free and are what the timing should be judged on.
    if align and np.asarray(plate.valid).any():
        from .assembly import align_world_to_imu
        try:
            plate = align_world_to_imu(plate)
        except ValueError:
            pass
    return plate
