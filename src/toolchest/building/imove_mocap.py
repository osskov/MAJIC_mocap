"""IMoVE (CMU-MBL) OptiTrack + Xsens sessions in, PlateTrials out.

The second dataset. It differs from Al Borno in four ways that all had to be discovered from
the data rather than from documentation, and each one is a trap:

  THREE IMUS PER SEGMENT. The thighs and shanks carry High, Mid and Low sensors against a
  single 4-marker cluster. Each becomes its own PlateTrial sharing that segment's pose, so
  every sensor gets its own gyro-derived mounting rotation and nothing downstream has to know
  this dataset is unusual.

  TWO SAMPLE RATES. The 17-sensor sessions run at 40 Hz and the 7-sensor long-walk ones at
  100 Hz, because Xsens trades radio bandwidth against sensor count. Mocap is 100 Hz
  throughout. assembly puts each trial at the slower of its two streams, so IMoVE trials load
  at 40 Hz except the long walks.

  ONE INERTIAL RECORDING PER SESSION, THREE MOCAP TAKES. In the long-walk sessions a single
  4538 s Xsens record covers three separate ~950 s Motive takes, which sit at roughly 2 s,
  1756 s and 3564 s into it. Only the first shares a filename with the IMU. See
  `_merge_takes`.

  MARKER GROUPS ARE NOT SEGMENTS. s4, s5 and s6 put every marker in one flat
  'modified_rizzoli' group, so there is no per-segment prefix to match on and matching has to
  be on the label suffix alone.

The device-to-segment map is not in the data at all; it comes from the authors' code at
github.com/CMU-MBL/IMoveLab.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..PlateTrial import PlateTrial
from ..WorldTrace import WorldTrace
from .assembly import _lag_seconds, _world_on_timestamps, assemble_plate_trials
from .reconstruction import fit_plate_to_template
from .xsens import read_xsens_txt

# Device id -> "<SEGMENT>_<placement>", from CMU-MBL/IMoveLab. The eight sensors beyond the
# obvious ten are not upper body: they are the High/Low placements flanking each limb's Mid.
DEVICE_TO_SENSOR = {
    '00B4D7D2': 'THIGH_L_H', '00B4D7FD': 'THIGH_L_M', '00B4D7CD': 'THIGH_L_L',
    '00B4D7D0': 'THIGH_R_H', '00B4D6D1': 'THIGH_R_M', '00B4D7D8': 'THIGH_R_L',
    '00B4D7CF': 'SHANK_L_H', '00B4D7CE': 'SHANK_L_M', '00B4D7FA': 'SHANK_L_L',
    '00B4D7BA': 'SHANK_R_H', '00B4D7FB': 'SHANK_R_M', '00B4D7D5': 'SHANK_R_L',
    '00B4D7D3': 'PELVIS_M', '00B4D7FF': 'FOOT_L_M', '00B4D7FE': 'FOOT_R_M',
}

# The 4-marker cluster on each segment, in the order the plate template is built from.
#
# PELVIS is the same physical plate as the limbs but labelled by side rather than 1-4, and
# its correspondence is a CYCLIC SHIFT of the naive one: (RPS1, RPS2, LPS2, LPS1) matches the
# thigh cluster at 1.03 mm, while starting from LPS1 does not appear in the top three. Getting
# this wrong does not fail loudly -- it silently rotates the pelvis frame.
CLUSTER_MARKERS = {
    'THIGH_L': ['LTH1', 'LTH2', 'LTH3', 'LTH4'],
    'THIGH_R': ['RTH1', 'RTH2', 'RTH3', 'RTH4'],
    'SHANK_L': ['LSH1', 'LSH2', 'LSH3', 'LSH4'],
    'SHANK_R': ['RSH1', 'RSH2', 'RSH3', 'RSH4'],
    'PELVIS': ['RPS1', 'RPS2', 'LPS2', 'LPS1'],
}

# The feet have no plate -- anatomical markers only, and their names vary by session:
# L1MT/L2MT/L5MT/LCAL in 14, LCAL/LDP1/LMT1/LMT2/LMT5 in 6, LCAL/LMT1/LMT2/LMT5 in s2, and
# LCAL/LLCAL/LMCAL/LMT1/LMT2/LMT5 in the long walks. Rather than an alias table per session,
# take whatever foot markers that session has: a pose needs only three.
FOOT_TOKENS = ('CAL', 'MT', 'DP')

# Fit residual above which a frame is not trusted, per segment kind. The clusters are rigid
# plastic and fit at 0.11-0.64 mm, so 10 mm is enormously loose for them and still catches a
# displaced marker. The feet are skin-mounted anatomical markers that genuinely deform --
# measured 0.88-6.83 mm -- and holding them to the plate tolerance would mark most of a
# trial invalid for doing nothing wrong.
CLUSTER_TOLERANCE_M = 0.010
FOOT_TOLERANCE_M = 0.025

_HEADER_ROWS = 7          # rows before the first data row
_NAME_ROW = 3             # 'modified_rizzoli:LTH1' etc.


def _marker_names(path: Path) -> List[str]:
    """The label row, e.g. 'modified_rizzoli:LTH1' repeated once per X/Y/Z column."""
    with open(path) as handle:
        for index, line in enumerate(handle):
            if index == _NAME_ROW:
                return line.strip().split(',')
    raise ValueError(f"{path.name}: no marker-name row at line {_NAME_ROW}.")


def read_motive_header(path: Path) -> Dict[str, str]:
    """The key/value pairs on line 0 of a Motive export."""
    with open(path) as handle:
        fields = handle.readline().rstrip().split(',')
    return {fields[i]: fields[i + 1] for i in range(0, len(fields) - 1, 2)}


def read_motive_csv(path: Path, labels: List[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """((N, M, 3) marker positions in metres, (N,) seconds), or None if a label is absent.

    Matched on the LABEL SUFFIX after ':', never the group: s4/s5/s6 put every marker in one
    'modified_rizzoli' group, so the group name carries no segment information at all.

    The row count comes from the file, not from the header's `Total Exported Frames`, which
    overstates by exactly 30000 in all six long-walk exports. The `Capture Frame Rate` field
    IS reliable -- rows divided by the final timestamp comes to 100.00 Hz -- but the
    timestamps are read directly anyway, so nothing depends on it.
    """
    header = read_motive_header(path)
    units = header.get('Length Units', 'Meters')
    if units != 'Meters':
        raise ValueError(f"{path.name}: expected marker positions in Meters, got {units!r}.")

    names = _marker_names(path)
    frame = pd.read_csv(path, skiprows=_HEADER_ROWS, header=None, low_memory=False).apply(
        pd.to_numeric, errors='coerce')

    columns = []
    for label in labels:
        found = [i for i, value in enumerate(names) if value.endswith(':' + label)]
        if not found:
            return None
        columns.append(frame.iloc[:, found[0]:found[0] + 3].to_numpy(np.float64))

    timestamps = frame.iloc[:, 1].to_numpy(np.float64)
    return np.stack(columns, axis=1), timestamps


def foot_marker_labels(path: Path, side: str) -> List[str]:
    """Whatever foot markers this session happens to name, for 'L' or 'R'."""
    names = _marker_names(path)
    found = set()
    for value in names:
        if ':' not in value:
            continue
        suffix = value.split(':')[1]
        if suffix.startswith(side) and any(token in suffix[1:] for token in FOOT_TOKENS):
            found.add(suffix)
    return sorted(found)


def segment_labels(path: Path) -> Dict[str, List[str]]:
    """Marker labels per segment for one take, feet resolved against that session's names."""
    labels = dict(CLUSTER_MARKERS)
    for side in ('L', 'R'):
        foot = foot_marker_labels(path, side)
        if len(foot) >= 3:
            labels[f'FOOT_{side}'] = foot
    return labels


def load_world_traces(csv_path: Union[str, Path]) -> Dict[str, WorldTrace]:
    """One Motive take -> {segment: WorldTrace}, skipping segments it does not track.

    A missing segment is NOT an error. The treadmill trials drop whole marker groups -- s10's
    t2 has all four LTH absent for the entire take -- and a reader has to be able to say
    "this segment is untracked here" without failing the trial around it.
    """
    csv_path = Path(csv_path)
    traces = {}
    for segment, labels in segment_labels(csv_path).items():
        markers = read_motive_csv(csv_path, labels)
        if markers is None:
            continue
        positions, timestamps = markers
        tolerance = FOOT_TOLERANCE_M if segment.startswith('FOOT') else CLUSTER_TOLERANCE_M
        try:
            pose, rotations, valid, _ = fit_plate_to_template(
                positions, timestamps, residual_tolerance=tolerance,
                name=f'{csv_path.stem}/{segment}')
        except ValueError:
            # Raised when fewer than three markers are ever present, i.e. the segment was
            # not tracked in this take at all.
            continue
        traces[segment] = WorldTrace(timestamps, pose, rotations, valid=valid)
    return traces


def load_imu_traces(session_dir: Union[str, Path], trial: str) -> Dict[str, 'IMUTrace']:
    """{sensor name: IMUTrace} for one trial, keyed as DEVICE_TO_SENSOR names them."""
    session_dir = Path(session_dir)
    traces = {}
    for path in sorted((session_dir / 'imu_data').glob(f'{trial}-000_*.txt')):
        device = path.name.split('-000_')[1][:-4]
        sensor = DEVICE_TO_SENSOR.get(device)
        if sensor is not None:
            traces[sensor] = read_xsens_txt(path)
    return traces


def _sensors_for_segment(segment: str, available: Dict[str, 'IMUTrace']) -> List[str]:
    """Which of this segment's sensors are actually present, e.g. THIGH_L -> [_H, _M, _L]."""
    return [name for name in available if name.rsplit('_', 1)[0] == segment]


def _take_lag(world: Dict[str, WorldTrace], imu: Dict[str, 'IMUTrace']) -> float:
    """One lag in seconds for a single Motive take, median over the segments it tracks.

    Each take is its own Motive recording with its own clock start, so a session's three
    long-walk takes have three unrelated lags and `assembly._shared_lag` -- which assumes one
    per trial -- cannot express them.
    """
    lags = [_lag_seconds(imu[f'{segment}_M'], trace)
            for segment, trace in world.items() if f'{segment}_M' in imu]
    if not lags:
        raise ValueError("No segment has both a cluster and an M sensor to sync on.")
    return float(np.median(lags))


def _merge_takes(takes: List[Dict[str, WorldTrace]], lags: List[float],
                 timestamps: np.ndarray, target_rate: float) -> Dict[str, WorldTrace]:
    """Several mocap takes -> one WorldTrace per segment on the IMU's clock.

    The long-walk sessions record one continuous 4538 s inertial file across three separate
    ~950 s Motive takes. Splitting that into three trials would store the inertial record
    three times and pretend there were three recordings; instead each take is resampled onto
    the shared IMU timeline at its own lag and they are OR-ed together, giving one trial whose
    `valid` mask is true in three windows and false in the gaps between them.

    The takes do not overlap -- measured at [2.2, 634.5], [1756.0, 2392.3] and
    [3564.4, 4235.5] s -- so combining them is a straight per-frame choice with no blending.
    Outside every take the pose is whatever the last take extrapolated, which is exactly the
    situation `valid` exists to mark.
    """
    merged = {}
    for segment in {name for take in takes for name in take}:
        positions = np.zeros((len(timestamps), 3))
        rotations = np.tile(np.eye(3), (len(timestamps), 1, 1))
        valid = np.zeros(len(timestamps), dtype=bool)

        for take, lag in zip(takes, lags):
            if segment not in take:
                continue
            placed = _world_on_timestamps(take[segment], timestamps - lag, target_rate)
            # Only where this take actually observed the segment, so a later take cannot
            # overwrite an earlier one's real data with its own extrapolation.
            covered = np.asarray(placed.valid) & ~valid
            positions[covered] = placed.positions[covered]
            rotations[covered] = placed.rotations[covered]
            valid |= covered

        merged[segment] = WorldTrace(timestamps, positions, rotations, valid=valid)
    return merged


def load_trial(session_dir: Union[str, Path], trial: str,
               align_plate_trials: bool = True) -> Dict[str, PlateTrial]:
    """One session's trial -> its synchronized PlateTrials.

    `trial` names the INERTIAL record. Usually one Motive take shares that name and the two
    are synced against each other in the ordinary way. In the long-walk sessions several
    takes belong to the one inertial file, and they are merged onto its clock first -- see
    `_merge_takes` -- after which assembly is told the lag is already applied.
    """
    session_dir = Path(session_dir)
    imu_traces = load_imu_traces(session_dir, trial)
    if not imu_traces:
        raise ValueError(f"{session_dir.name}/{trial}: no IMU files for any known device.")

    take_paths = mocap_takes_for(session_dir, trial)
    if not take_paths:
        raise ValueError(f"{session_dir.name}/{trial}: no mocap take matches this record.")

    takes = [load_world_traces(path) for path in take_paths]
    takes = [take for take in takes if take]
    if not takes:
        raise ValueError(f"{session_dir.name}/{trial}: no segment reconstructed in any take.")

    if len(takes) == 1:
        world_traces = takes[0]
        lag = None                      # let assembly find it, the ordinary path
    else:
        reference = next(iter(imu_traces.values()))
        target_rate = min(reference.get_sample_frequency(),
                          min(t.get_sample_frequency() for take in takes
                              for t in take.values()))
        lags = [_take_lag(take, imu_traces) for take in takes]
        world_traces = _merge_takes(takes, lags, reference.timestamps, target_rate)
        lag = 0.0                       # already on the IMU's clock

    return assemble_plate_trials(imu_traces, _pair_world_traces(world_traces, imu_traces),
                                 align_plate_trials, lag=lag)


def mocap_takes_for(session_dir: Union[str, Path], trial: str) -> List[Path]:
    """The Motive takes belonging to one inertial record, in order.

    Normally that is the single take of the same name. The long walks are the exception: only
    `t12_longwalk_001` shares its name with the IMU file, while `_002` and `_003` are further
    takes covered by the same continuous recording, so all three belong to it.
    """
    mocap_dir = Path(session_dir) / 'mocap_data'
    exact = mocap_dir / f'{trial}.csv'
    if not exact.exists():
        return []
    if not trial.endswith('_001'):
        return [exact]

    stem = trial[:-len('_001')]
    siblings = sorted(p for p in mocap_dir.glob(f'{stem}_[0-9][0-9][0-9].csv')
                      if p.name != exact.name)
    # Extra takes only count when nothing else claims them: if `<stem>_002` has its own IMU
    # file it is a separate trial, not part of this one.
    unclaimed = [p for p in siblings
                 if not any((Path(session_dir) / 'imu_data').glob(f'{p.stem}-000_*.txt'))]
    return [exact] + unclaimed


def _pair_world_traces(world: Dict[str, WorldTrace],
                       imu: Dict[str, 'IMUTrace']) -> Dict[str, WorldTrace]:
    """Repeats each segment's pose under every sensor mounted on it.

    assemble_plate_trials pairs by exact name, so giving THIGH_L_H, THIGH_L_M and THIGH_L_L
    the same WorldTrace is all it takes to support three sensors per segment. They are not
    copied -- the same object is referenced three times, and nothing downstream mutates it.
    """
    paired = {}
    for segment, trace in world.items():
        for sensor in _sensors_for_segment(segment, imu):
            paired[sensor] = trace
    return paired
