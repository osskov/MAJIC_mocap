"""
Anatomical landmark markers, read straight from the source mocap and expressed in each segment's
own frame.

These are the only external reference in the repo for where a joint centre actually is: everything
in experiments/joint_center.py is self-consistency (two segments agreeing, or one subject's trials
agreeing), and a fit can be internally perfect while both offsets slide together — which is
exactly the least-determined mode.

NOT CACHED, and it needs no sync. The landmarks live in the SAME mocap file as the plate markers,
frame for frame, so the useful quantity

    c_segment(t) = R_plate(t)^T (p_landmark(t) - p_plate(t))

is formed entirely within one file at its native rate. No IMU clock, no cross-correlation lag, no
resampling, no take merging. Reading a trial costs ~3.5 s (0.5 s parse + 3.0 s plate
reconstruction), which is cheaper than the machinery a cached artifact would need.

Al Borno's STANDING CAPTURE is the one place two files are involved, and it still needs no sync,
because what crosses between them is a constant rather than a time series: the standing capture
contributes a fixed c_segment per joint, and the trial contributes the poses. What it DOES need is
a shared template — see `read_alborno_plates` — because the two captures otherwise estimate the
plate's body frame independently, and on a poorly conditioned plate those two frames are not
related by a constant rotation, so nothing cancels.

THE PLATE POSE MUST COME FROM HERE, NOT FROM A BUILT TRIAL, and this is the one trap in using this
module. `reconstruct_plate` returns the raw marker-template frame; the build additionally applies a
sensor-to-segment alignment, measured at 171-178 deg on Al Borno, so a landmark reduced against a
raw pose and an offset taken from a built plate are THE SAME VECTOR IN DIFFERENT BASES. Comparing
them reads 33 mm as 316 mm. Anything compared against `c_segment` has to be recomputed from these
poses — for the joint-centre offsets that means re-running `WorldTrace.get_joint_center` on the
traces built from `read_*_plates`, which returns |r| identical to 0.00 mm, so refitting is free.

Given that, the template frame being arbitrary costs nothing: it cancels out of every comparison.
Rotating it moves both vectors' components identically and leaves |r|, |c|, |r - c| and the angle
between them unchanged (verified against a 37 deg and an arbitrary 120 deg re-basing). No alignment
step is needed to compare a landmark against a fit — only to express either in the IMU's frame,
which no comparison here requires.

DELIBERATELY NOT IN THE TRIAL CACHE KEY. This module is absent from
experiment_utils._CORE_MODULES and _READER_MODULES, so editing it does not invalidate 280 built
trials. That is safe only because nothing in the build consumes it — if landmarks ever start
shaping a cached trial's CONTENTS, this must be added to that digest.

What is available differs by dataset:

    joint    Al Borno trial          Al Borno standing        IMoVE
    hip      hjc_r / hjc_l           R_HJC / L_HJC            LGTR only, a surface marker
    knee     LATERAL only            Knee + MKnee midpoint    LEP + MEP midpoint
    ankle    LATERAL only            Ankle + MAnkle midpoint  LML + MML midpoint

Al Borno's trial-file knee_r sits 169 mm from the ASIS midline against the lateral thigh plate's
180 mm and hjc_r's 89 mm, is 69 mm off the hip-ankle line, and knee_r<->knee_l is 363 mm against
hjc's 195 mm. All three say surface marker, not centre, and with no medial counterpart in that
file a centre cannot be constructed without inventing half a knee width. The standing capture is
what closes the gap: it carries the medial markers, so every joint gets a centre from both
datasets.

NOT ALL THREE JOINTS ARE EQUALLY GOOD REFERENCES, and the hip is the weak one. Scored as a
predictor of a trial's own frames — the closure error |(p_p + R_p c_p) - (p_c + R_c c_c)| against
what that trial's own best fit achieves — the standing knee and ankle come in at 1.5x the floor,
only 1.25x behind using a DIFFERENT trial's fitted centre, while the hip is 2.6x. The hip's
problem is the reference, not soft tissue: the same subject's R_HJC moves a median 11.8 mm (up to
30 mm) between the standing and walking captures while scattering only 3-4 mm within either one,
and it is not a fixed regression on pelvis geometry (12-24% CoV across subjects against 7% for
the width measurement itself). Subject11's L_HJC is 118 mm out in BOTH captures.

The landmarks carry soft-tissue error of the same order as the thing being checked, so this
supports METHOD AGREEMENT rather than validation against truth.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .reconstruction import estimate_plate_template, reconstruct_plate

# Millimetres in both source formats; the rest of the repo works in metres.
MM_TO_M = 1000.0

# Frames with all four corners visible needed before a template is estimated from a capture.
MIN_TEMPLATE_FRAMES = 10

# A static reference is a MEDIAN OF A CONSTANT, not a fit, so it needs enough frames to average
# marker noise down and nothing more -- there is no conditioning to satisfy. Deliberately far
# below the experiment's 500-frame fit floor, which five of the eleven static captures (200-394
# frames) would otherwise fail.
MIN_STATIC_FRAMES = 50


@dataclass(frozen=True)
class LandmarkSpec:
    """Which landmarks a dataset has, and what each one is for.

    `centres` maps a joint to the markers whose MIDPOINT is its centre, IN THE TRIAL'S OWN FILE —
    one marker where the file already holds a computed centre (Al Borno's hjc), two where the
    centre is the midpoint of a medial/lateral pair (IMoVE's epicondyles and malleoli). These are
    measured while the subject is MOVING, so their scatter is a real soft-tissue estimate.

    `static_centres` are the same idea sourced from a separate standing capture, which is the only
    place Al Borno's medial markers exist. Richer (it reaches the knee and ankle) and quieter, but
    the quiet is the point of caution: a standing capture cannot see motion-driven soft-tissue
    movement at all, so its scatter UNDERSTATES the landmark's error rather than measuring it.
    Both are kept because they fail differently, and where they overlap their disagreement is
    itself the measurement — Al Borno's hip centre moves 11.8 mm between the two captures while
    scattering 3-4 mm within either one.

    `surface` are landmarks that are NOT joint centres. They earn their place anyway: their scatter
    in a segment frame measures soft-tissue movement directly, which is the error budget the
    centres inherit.
    """
    name: str
    read: Callable[[str, str], Tuple[np.ndarray, Dict[str, np.ndarray]]]
    centres: Dict[str, Tuple[str, ...]]
    surface: Tuple[str, ...]
    static_centres: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # (subject, {sensor: ...}) -> StaticCapture, or None where the dataset has no such capture.
    read_static: Optional[Callable[[str, Sequence[str]], Optional['StaticCapture']]] = None

    def markers(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys([m for pair in self.centres.values() for m in pair]
                                   + list(self.surface)))

    def static_markers(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(m for pair in self.static_centres.values() for m in pair))


@dataclass(frozen=True)
class StaticCapture:
    """A standing capture, reduced to what a landmark comparison needs from it.

    `templates` is the load-bearing field: the caller must reconstruct the TRIAL's plates with
    these, or the centres below and the trial's fitted offsets are expressed in different
    arbitrary body frames and nothing about the comparison is valid.
    """
    timestamps: np.ndarray
    markers: Dict[str, np.ndarray]
    poses: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    templates: Dict[str, np.ndarray]


def _trc_columns(trc_path: Path) -> Tuple[list, pd.DataFrame]:
    with open(trc_path, 'r', encoding='utf-8') as handle:
        header = [handle.readline() for _ in range(6)]
    labels = header[3].strip().split('\t')
    return labels, pd.read_csv(trc_path, delimiter='\t', skiprows=6, header=None, engine='c')


def read_alborno_markers(trc_path: Path, names: Sequence[str]
                         ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """{name: (N, 3) in metres} plus the timestamps, from a .trc.

    Untracked samples are ZEROS in this format, not NaN — a real trap, since `isfinite` passes them
    and a marker sitting at the lab origin would drag any mean it entered. They are converted to
    NaN here so every consumer masks the same way.
    """
    labels, frame = _trc_columns(Path(trc_path))
    timestamps = frame.iloc[:, 1].to_numpy(dtype=np.float64)
    out = {}
    for name in names:
        if name not in labels:
            continue
        start = labels.index(name)
        xyz = frame.iloc[:, start:start + 3].to_numpy(dtype=np.float64) / MM_TO_M
        xyz[np.abs(xyz).sum(axis=1) < 1e-9] = np.nan
        out[name] = xyz
    return timestamps, out


def read_alborno_plates(trc_path: Path, stems: Mapping[str, str],
                        templates: Optional[Dict[str, np.ndarray]] = None,
                        suffixes: str = 'odxy'
                        ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                   Dict[str, np.ndarray]]:
    """({sensor: (positions, rotations, valid)}, {sensor: template}) at NATIVE mocap rate.

    Reconstructed here rather than read from the cached trial, because the cache is resampled onto
    the IMU clock and the landmarks are not. Doing both at native rate is what keeps the whole
    comparison sync-free.

    `stems` maps the sensor name this repo uses to what the plate is called IN THIS FILE, because
    the two Al Borno captures disagree: a trial spells the right thigh plate 'femur_r_imu_o' and
    the static capture spells the same physical plate 'R.Femur_IMU_O'.

    TEMPLATES ARE RETURNED SO THEY CAN BE CARRIED ACROSS CAPTURES, and that is the whole reason
    for the second value. Read the standing capture first, then feed its templates back in when
    reading the trial.

    The need is subtler than "each capture picks an arbitrary frame", which is what the estimator's
    own docstring might suggest. `estimate_plate_template` is classical MDS on the MEDIAN
    inter-marker distances, so it is a deterministic function of the plate's shape, and two
    captures of one plate mostly land in the SAME frame by themselves: over Al Borno's 88
    static/trial plate pairs, reconstructing the trial with the standing templates rather than its
    own turns out to be a constant re-basing to a median of 0.08 deg, which cancels out of every
    quantity reported here (median |r| shift 0.02 mm).

    THE TAIL IS WHY IT IS NOT LEFT TO CHANCE. The distances are medians of noisy markers, so a
    poorly conditioned plate estimates a measurably different SHAPE, and that changes the Kabsch
    fit rather than merely re-basing it — Subject08's pelvis plate differs by a rotation that is
    itself not constant, varying over 26.9 deg across the trial and moving |r| by up to 13.3 mm.
    Where that happens the two captures are not comparable at all, and no scalar check would show
    it. Passing the template makes the question moot instead of rare.
    """
    labels, frame = _trc_columns(Path(trc_path))
    timestamps = frame.iloc[:, 1].to_numpy(dtype=np.float64)
    poses, used = {}, {}
    for sensor, stem in stems.items():
        corners = [f"{stem}_{suffix}" for suffix in suffixes]
        if any(c not in labels for c in corners):
            continue
        markers = np.stack([frame.iloc[:, labels.index(c):labels.index(c) + 3]
                            .to_numpy(dtype=np.float64) / MM_TO_M for c in corners], axis=1)
        markers[np.abs(markers).sum(axis=2) < 1e-9] = np.nan
        template = (templates or {}).get(sensor)
        if template is None:
            complete = np.isfinite(markers).all(axis=(1, 2))
            if complete.sum() < MIN_TEMPLATE_FRAMES:
                continue
            template = estimate_plate_template(markers[complete])
        used[sensor] = template
        positions, rotations, valid, _ = reconstruct_plate(markers, timestamps, template=template,
                                                           name=sensor)
        poses[sensor] = (positions, rotations, np.asarray(valid, dtype=bool))
    return poses, used


# The static capture is a different export with different conventions: plates are named by side
# and bone rather than by OpenSim body, and the corner suffixes are upper case.
ALBORNO_STATIC_PLATES = {
    'pelvis_imu': 'Pelvis_IMU', 'torso_imu': 'Back_IMU',
    'femur_r_imu': 'R.Femur_IMU', 'tibia_r_imu': 'R.Tibia_IMU', 'calcn_r_imu': 'R.Foot_IMU',
    'femur_l_imu': 'L.Femur_IMU', 'tibia_l_imu': 'L.Tibia_IMU', 'calcn_l_imu': 'L.Foot_IMU',
}
ALBORNO_STATIC_SUFFIXES = 'ODXY'


def alborno_static_trc(subject: str) -> Path:
    """The standing capture for one subject.

    Filed under the walking trial rather than beside it, and deliberately: a second `.trc` in the
    trial folder itself would trip `alborno.load_trial`'s one-trial-per-folder guard, change the
    `*.trc` source inventory that keys the trial cache, and outrank `walking.trc` in the sorted
    glob `_alborno_trc` uses. A subdirectory is invisible to all three, none of which recurse.
    """
    import paths
    return paths.DATA_DIR / f"Subject{subject}" / 'walking' / 'static' / 'static_walking.trc'


def read_alborno_static(subject: str, sensors: Sequence[str],
                        names: Sequence[str]) -> Optional[StaticCapture]:
    """One subject's standing capture: its landmarks, its plate poses, and its templates.

    This is where Al Borno's MEDIAL markers live — R.MKnee, R.MAnkle and their left counterparts
    are absent from every trial file — so it is the only route to a knee or ankle centre in this
    dataset rather than a lateral surface point.

    Per-subject, not per-trial. The plates are taped once per session and measured stable across
    captures at 1.5-1.6 mm, which is what makes a standing reference transferable to a trial at
    all; whether the CENTRE transfers as well is the question the comparison exists to ask, and
    the answer differs sharply by joint.
    """
    trc = alborno_static_trc(subject)
    if not trc.exists():
        return None
    timestamps, markers = read_alborno_markers(trc, names)
    poses, templates = read_alborno_plates(
        trc, {s: ALBORNO_STATIC_PLATES[s] for s in sensors if s in ALBORNO_STATIC_PLATES},
        suffixes=ALBORNO_STATIC_SUFFIXES)
    if not poses:
        return None
    return StaticCapture(timestamps, markers, poses, templates)


def read_imove_markers(csv_path: Path, names: Sequence[str]
                       ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """{name: (N, 3) in metres} plus timestamps, from a Motive take.

    `read_motive_csv` already resolves a bare label, so no new parsing is needed — the landmarks
    were simply never asked for.
    """
    from .imove_mocap import read_motive_csv
    out, timestamps = {}, None
    for name in names:
        found = read_motive_csv(Path(csv_path), [name])
        if found is None:
            continue
        positions, timestamps = found
        out[name] = positions[:, 0, :]
    if timestamps is None:
        timestamps = np.zeros(0)
    return timestamps, out


def in_segment_frames(timestamps: np.ndarray, markers: Dict[str, np.ndarray],
                      poses: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                      wanted: Dict[str, Sequence[str]]) -> pd.DataFrame:
    """Long-form per-sample landmark coordinates in each segment's own frame, in millimetres.

    `wanted` is {segment: [landmark, ...]} — a landmark is only expressed in the frames of the
    segments that border its joint, since that is where it can be compared against a fitted offset.

    Raw markers are kept separate rather than pre-combined into centres. That is what makes the
    inter-marker distance available as a label-swap detector, lets each marker carry its own
    tracking, and leaves the centre convention changeable without re-reading anything.

    The coordinates are in whatever frame `poses` supplies. Callers comparing against a fitted
    offset must derive that fit from the SAME poses — see the module docstring.
    """
    blocks = []
    for segment, names in wanted.items():
        if segment not in poses:
            continue
        positions, rotations, valid = poses[segment]
        n = min(len(positions), len(timestamps))
        for name in names:
            if name not in markers:
                continue
            landmark = markers[name][:n]
            tracked = np.isfinite(landmark).all(axis=1)
            local = np.full((n, 3), np.nan)
            usable = tracked & valid[:n]
            if usable.any():
                local[usable] = np.einsum(
                    'nji,nj->ni', rotations[:n][usable],
                    landmark[usable] - positions[:n][usable]) * MM_TO_M
            blocks.append(pd.DataFrame({
                'timestamp': timestamps[:n], 'segment': segment, 'landmark': name,
                'x': local[:, 0].astype(np.float32), 'y': local[:, 1].astype(np.float32),
                'z': local[:, 2].astype(np.float32),
                'tracked': tracked, 'plate_valid': valid[:n]}))
    if not blocks:
        return pd.DataFrame()
    out = pd.concat(blocks, ignore_index=True)
    for column in ('segment', 'landmark'):
        out[column] = out[column].astype('category')
    return out


# ==============================================================================
# Per-dataset specs
# ==============================================================================

ALBORNO = LandmarkSpec(
    name='alborno',
    read=lambda subject, trial: read_alborno_markers(_alborno_trc(subject, trial),
                                                     ALBORNO.markers()),
    # hjc_* are computed hip joint centres: 89/105 mm from the ASIS midline, medial to every
    # other marker, and 195 mm apart, all consistent with centres rather than surface points.
    # The trial file holds no other centre — knee_* and ankle_* here are LATERAL markers.
    centres={'R_Hip': ('hjc_r',), 'L_Hip': ('hjc_l',)},
    # From the standing capture, which is the only Al Borno file carrying medial markers. The
    # knee and ankle are the medial/lateral midpoint; the file's own R_KJC/R_AJC columns are NOT
    # used because they are that same midpoint to 0.0 mm across all 40 subject-joints where they
    # exist, so they add nothing and are missing for Subject06 besides.
    static_centres={'R_Hip': ('R_HJC',), 'L_Hip': ('L_HJC',),
                    'R_Knee': ('R.Knee', 'R.MKnee'), 'L_Knee': ('L.Knee', 'L.MKnee'),
                    'R_Ankle': ('R.Ankle', 'R.MAnkle'), 'L_Ankle': ('L.Ankle', 'L.MAnkle')},
    read_static=lambda subject, sensors: read_alborno_static(subject, sensors,
                                                             ALBORNO.static_markers()),
    # Not centres, but still the best available probes of soft-tissue movement at those joints,
    # and unlike the static markers they are measured while the subject is moving.
    surface=('knee_r', 'knee_l', 'ankle_r', 'ankle_l', 'asis_r', 'asis_l', 'heel_r', 'heel_l'),
)

IMOVE = LandmarkSpec(
    name='imove',
    read=lambda subject, trial: read_imove_markers(_imove_csv(subject, trial), IMOVE.markers()),
    centres={'R_Knee': ('RLEP', 'RMEP'), 'L_Knee': ('LLEP', 'LMEP'),
             'R_Ankle': ('RLML', 'RMML'), 'L_Ankle': ('LLML', 'LMML')},
    # No hip centre here: LGTR is the greater trochanter, a surface point, and there is no medial
    # counterpart to pair it with.
    surface=('RGTR', 'LGTR', 'RASI', 'LASI'),
)

SPECS = {ALBORNO.name: ALBORNO, IMOVE.name: IMOVE}


def get_spec(dataset: str) -> LandmarkSpec:
    try:
        return SPECS[dataset]
    except KeyError:
        raise ValueError(f"No landmark spec for {dataset!r}. Have: {sorted(SPECS)}.") from None


def _alborno_trc(subject: str, trial: str) -> Path:
    import paths
    found = sorted((paths.DATA_DIR / f"Subject{subject}" / trial).glob("*.trc"))
    if not found:
        raise FileNotFoundError(f"No .trc for alborno/{subject}/{trial}")
    return found[0]


def _imove_csv(session: str, trial: str) -> Path:
    from .sources import IMOVE_ROOT
    from .imove_mocap import mocap_takes_for
    takes = mocap_takes_for(IMOVE_ROOT / session, trial)
    if not takes:
        raise FileNotFoundError(f"No mocap take for imove/{session}/{trial}")
    # The first take only. The long-walk trials hold three against one inertial record, and
    # merging them needs the per-take lag this module exists to avoid — each take is
    # self-consistent on its own clock, which is all the comparison requires.
    return takes[0]
