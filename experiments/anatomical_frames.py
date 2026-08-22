"""
The constant rotation from each sensor's frame to its segment's ANATOMICAL frame, measured from
the source mocap's anatomical landmarks.

WHAT THIS IS FOR. `compute_error_stats` splits every joint-angle error into components as well as
a magnitude, and until now those components were `X`, `Y`, `Z` — the axes of the PARENT SENSOR's
own frame. That is a physically well-defined split and an anatomically meaningless one: the
plates are re-strapped per subject, so `X` names a different direction for every subject, and
pooling eleven of them averages flexion error into rotation error. This module supplies the basis
that makes the split mean flexion / adduction / internal rotation instead, so the same error
vector can be reported per anatomical axis without re-running a single filter.

    python -m experiments.anatomical_frames --dataset alborno
    python -m experiments.anatomical_frames --dataset imove --subjects s13
    python -m experiments.anatomical_frames --dataset alborno --report-only
    python -m experiments.anatomical_frames --dataset alborno --validate

THE CHAIN, AND WHY IT NEEDS THREE LINKS
=======================================
The error vector lives in the BUILT plate's frame, which is the IMU's own frame — `assembly.
align_world_to_imu` rotates every world trace onto its sensor at build time. The landmarks live
in the mocap file, in the plate's MARKER-TEMPLATE frame. Those are two different bases and the
rotation between them is exactly what the build measured and threw into a report:

    A_imu  =  C^T  A_template                      (columns of A are the anatomical axes)

  A_template  the anatomical axes in the marker-template frame, from landmark positions reduced
              against the plate poses (`_segment_axes`). Measured ONCE PER SUBJECT — from Al
              Borno's standing capture, which is the only file in that dataset carrying medial
              markers, and per trial on IMoVE, where every landmark is in every take.

  C           `align_world_to_imu`'s rotation, R_{template <- IMU}, read back out of the trial's
              `*.build.parquet` sidecar (`align_rotation`). PER TRIAL, because it is fitted per
              trial: two trials of one subject are two independent estimates of one physical
              mounting, which is what makes their disagreement a check on the whole chain.

  A_imu       what this module writes. Columns are (FE, AA, IE) — see THE SIGN CONVENTION.

THE READER MUST BE THE BUILD'S READER, and this is the one trap. `landmarks.read_alborno_plates`
and `alborno.load_world_traces` are separate code paths over the same markers, and if they
disagreed by any constant the anatomical axes would be silently rotated by it. They do not:
measured on Subject01's five parent plates, the rotation between them is 0.0000 deg at every
frame, exactly. `--validate` re-measures that on whatever dataset it is pointed at rather than
trusting this paragraph, because the day one of the two readers changes its template estimator is
the day every number here goes quietly wrong.

WHAT IT COSTS TO BE WRONG, AND THE FOUR CHECKS THAT WOULD CATCH IT
==================================================================
A wrong basis does not raise. It rotates error between the three panels of a figure, which reads
as a finding. So every row carries its own evidence, and each check covers a different link:

  `align_check_deg` — the angle, in the SENSOR's frame, between the up direction the accelerometer
      reports and the up direction the mocap reports. Isolates `C`, the link read out of a
      sidecar: posture cancels because both directions are measured on the same samples in the
      same frame. Measured 1.3-6.0 deg median by segment on Al Borno, worst trial 11.8, which is
      walking dynamics rather than misalignment.

  `upright_check_deg` — the angle between the segment's SUPERIOR axis and the mocap's up
      direction. Isolates the landmark half instead: `C` cancels out of it algebraically. Posture
      does NOT cancel, so it is 6.5-10.5 deg over walking and 18-19 deg on the thighs over
      complexTasks, where the mean posture is not upright and is not supposed to be. It says
      nothing at all about the pelvis, whose superior axis is the normal to the ASIS/PSIS plane
      and is therefore tilted by the subject's own pelvic tilt: 5 deg on Al Borno, 17-26 on IMoVE.

  `hinge_angle_deg` (the `checks` table) — the landmark flexion axis against the axis the joint
      is actually observed to turn about, from the mocap alone. THE ONE CHECK ON THE AXIS NAMES,
      and at the knee it is sharp: 6-7 deg median, 18 deg worst over Al Borno's 37 knees, with
      87% of the joint's motion on that one axis. Blunt at the hip (11-13 deg, and a ball joint's
      principal axis is whatever the trial emphasised) and uninformative at the lumbar (80 deg at
      64% concentration — the trunk has no axis to agree with).

  `lateral_obliquity_deg` — how far the medial/lateral landmark pair sat from perpendicular to
      the long axis, i.e. the size of the Gram-Schmidt correction. 0.5-12 deg on Al Borno, which
      is skin over bone; 60 deg would be a swapped or mislabelled marker.

  cross-trial agreement — the console report's last table, and on Al Borno the check with the most
      POWER on `C`, since the landmark half is identical between a subject's trials by
      construction: its eight two-trial subjects agree to a median of 0.12-0.74 deg by sensor,
      worst case 1.9 deg. On IMoVE the landmarks are re-measured per trial (they are in every
      take), so the same table covers both halves at once and reads 6-13 deg — a real difference,
      driven by trials as short as a drop jump giving the alignment rotation little to fit.

WHAT THIS IS NOT
================
NOT a Cardan/Euler decomposition, and the difference matters when reading the figures. This
rotates the error VECTOR into an anatomical basis: the three components are the error's
projections onto the flexion, adduction and rotation axes of the PARENT segment, and because the
basis is orthonormal they satisfy

    RMSE_MAG^2 = RMSE_FE^2 + RMSE_AA^2 + RMSE_IE^2                       exactly, per group

so the three panels add up to the magnitude figure the paper already reports and cannot
contradict it. A clinical per-plane angle error — decompose the estimated and the reference pose
each into an ISB Cardan sequence, then difference the three angles — is a DIFFERENT quantity, is
sequence-dependent, needs a neutral pose, and does not decompose the total. Nothing here
computes it.

NOT a validation of anything against truth. The landmarks are skin markers over bone and carry
soft-tissue error of the same order as the offsets they define (see `landmarks.py`, which
measures Al Borno's hip centre moving 11.8 mm between two captures of the same subject). What
they buy is a basis that means the same thing across subjects, not a correct one.

NOT available on the biplane halves. `landmarks.SPECS` covers alborno and imove; the biplane
build has neither marker set registered, and its bone-pose spec would not need this module at all
if it did — those world traces are already in anatomical bone frames, so only `C` stands between
them and an anatomical basis. `--dataset imove_biplane` therefore refuses rather than guessing.

NOT usable for the child side of a joint. The error rotation is expressed in the parent frame, so
only parent segments get a basis: pelvis and both thighs and shanks. The feet and the torso are
children everywhere and are skipped, which is why the ankle's 'FE' is dorsiflexion measured about
the SHANK's mediolateral axis and not the foot's.

THE SIGN CONVENTION
===================
The intermediate frame is ISB-shaped: X anterior, Y superior, Z to the subject's RIGHT, right-
handed, built from one exact axis and one orthogonalized reference direction (`SEGMENT_FRAMES`).
The reported axes are then permuted and signed so that POSITIVE IS FLEXION, ADDUCTION AND
INTERNAL ROTATION on both sides of the body:

    FE = -Z   both sides          AA = -X  right, +X  left          IE = +Y  right, -Y  left

Both triads are right-handed (det = +1), which is asserted on every row. The left-side flips are
what make a pooled 'Knee' row legitimate: without them the same physical adduction error enters
the left and right legs with opposite sign, and any signed statistic pooled over sides — mean,
median, the quartiles — cancels toward zero while RMSE quietly does not. Verified against the
data rather than derived on paper: projected on this basis, Al Borno's marker knee angle carries
83 deg of range on FE against 25-28 on AA, and its one-sided excursion is POSITIVE on BOTH knees
at a skew of +1.0 — which is the sign of flexion and the mirroring at once.

For the pelvis there is no adduction or internal rotation, and the limb convention applied
verbatim makes +AA LEFT lateral bending and +IE LEFT axial rotation. The names are kept anyway so
that one `axis` vocabulary covers every joint; the magnitude-based metrics the figures use are
sign-blind, and anything reading `mean_rad` on a Lumbar row has to read this paragraph.

Outputs
-------
    results/experiments/anatomical_frames/<dataset>/bases.parquet
    results/experiments/anatomical_frames/<dataset>/checks.parquet

`bases` is one row per (subject, trial, sensor): the nine components of A_imu, the landmark
geometry it was built from, and the per-sensor checks. `checks` is one row per (subject, trial,
joint) and holds the flexion-axis agreement and the reference joint angle's range on each
anatomical axis — a joint rather than a sensor, because a sensor parents up to three of them.

`load_bases` reads either back. `basis_frame` re-keys the bases by joint, which is the shape
`compute_error_stats(df, basis=...)` consumes; `bases_by_joint` is the same thing as a dict.
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import paths
from experiments.experiment_utils import (ANATOMICAL_AXES, BASIS_COLUMNS, TrackingSpec,
                                          build_report_path, load_trial)
from experiments.global_assumptions import DATASETS, select_trials, tracking_spec
from src.toolchest.building import alborno, imove_mocap, landmarks
from src.toolchest.building.sources import IMOVE_ROOT

EXPERIMENT_NAME = 'anatomical_frames'
EXPERIMENT_DIR = paths.experiment_dir(EXPERIMENT_NAME)

# The anatomical axes, in the order the basis columns are written and the order the figures should
# read them. Owned by experiment_utils, which is where the statistics that carry them are built.
AXES = ANATOMICAL_AXES

# (X, Y, Z) = (anterior, superior, right)  ->  (FE, AA, IE), per side. See THE SIGN CONVENTION.
# A midline segment takes the right-side mapping; there is nothing for it to mirror against.
AXIS_SIGNS = {
    'R': np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]),
    'L': np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]),
}
AXIS_SIGNS[''] = AXIS_SIGNS['R']

# Frames whose landmark must be tracked before its position is averaged. A landmark offset in a
# segment frame is a MEDIAN OF A CONSTANT, so this only has to average marker noise down --
# deliberately the same floor `landmarks.MIN_STATIC_FRAMES` sets, for the same reason.
MIN_LANDMARK_FRAMES = landmarks.MIN_STATIC_FRAMES

# Above this, the medial/lateral pair is not plausibly perpendicular to the long axis and the
# frame is reported but flagged. Skin over an epicondyle is 5-20 deg; a swapped label is 60+.
MAX_LATERAL_OBLIQUITY_DEG = 45.0

# Above this, the accelerometer and the mocap disagree about which way is up IN THE SAME FRAME by
# more than walking dynamics can explain, which points at the build's alignment rotation or its
# sync rather than at anything this module computed. Posture cancels out of that comparison, so
# the threshold does not have to accommodate a trial that is not upright.
MAX_ALIGN_CHECK_DEG = 20.0

# How many flagged rows the console report lists before it just counts them. IMoVE runs 236 trials
# x 15 sensors, so an unbounded list of everything over threshold buries the tables above it.
FLAG_PRINT_LIMIT = 20


# ==============================================================================
# Anatomical frame definitions
# ==============================================================================

@dataclass(frozen=True)
class SegmentFrame:
    """How one segment's anatomical axes are built out of landmark POINTS.

    A point is a name in this dataset's `POINTS` table, which maps it to the markers whose
    MIDPOINT it is — one marker where the file already holds a centre, two where the centre is a
    medial/lateral pair. That indirection is deliberate: it keeps a joint centre defined by its
    raw markers rather than by a column some subjects are missing (`landmarks.py` documents Al
    Borno's own R_KJC agreeing with the midpoint to 0.0 mm and being absent for Subject06).

    `superior` is (proximal, distal) and `rightward` is (right, left) — each a pair of points
    whose DIFFERENCE gives that direction, so no side-dependent sign logic is needed here and a
    left limb simply names its pair the other way round.

    `primary` names the axis taken as exact; the other is orthogonalized against it and the third
    is their cross product. The long axis is primary on a limb because it spans 370-480 mm
    against the 80-135 mm of a medial/lateral pair, so the same marker error tilts it four times
    less. The pelvis has no long axis and takes `rightward` as primary instead, with `anterior`
    as the reference.
    """
    rightward: Tuple[str, str]
    primary: str
    side: str
    superior: Optional[Tuple[str, str]] = None
    anterior: Optional[Tuple[str, str]] = None

    def points(self) -> Tuple[str, ...]:
        pairs = [self.rightward, self.superior, self.anterior]
        return tuple(dict.fromkeys(p for pair in pairs if pair is not None for p in pair))

    def __post_init__(self):
        if self.primary not in ('superior', 'rightward'):
            raise ValueError(f"primary must be 'superior' or 'rightward', got {self.primary!r}")
        if self.primary == 'superior' and self.superior is None:
            raise ValueError("primary='superior' needs a superior pair")
        if self.primary == 'rightward' and self.anterior is None:
            raise ValueError("primary='rightward' needs an anterior pair to orthogonalize")


# Al Borno's landmark points, ALL FROM THE STANDING CAPTURE. The trial files carry hip centres
# and lateral knee/ankle markers but no medial counterparts, so a knee or ankle centre cannot be
# built from a trial without inventing half a joint width -- see landmarks.ALBORNO. The standing
# capture also holds its own midASIS/R_KJC/R_AJC columns; they are the same midpoints and are not
# used, so that a subject missing one still gets a frame.
ALBORNO_POINTS = {
    'midASIS': ('R.ASIS', 'L.ASIS'),
    'midPSIS': ('R.PSIS', 'L.PSIS'),
    'R.ASIS': ('R.ASIS',), 'L.ASIS': ('L.ASIS',),
    'R_HJC': ('R_HJC',), 'L_HJC': ('L_HJC',),
    'R_KJC': ('R.Knee', 'R.MKnee'), 'L_KJC': ('L.Knee', 'L.MKnee'),
    'R_AJC': ('R.Ankle', 'R.MAnkle'), 'L_AJC': ('L.Ankle', 'L.MAnkle'),
    'R.Knee': ('R.Knee',), 'R.MKnee': ('R.MKnee',),
    'L.Knee': ('L.Knee',), 'L.MKnee': ('L.MKnee',),
    'R.Ankle': ('R.Ankle',), 'R.MAnkle': ('R.MAnkle',),
    'L.Ankle': ('L.Ankle',), 'L.MAnkle': ('L.MAnkle',),
}

ALBORNO_FRAMES = {
    'pelvis_imu': SegmentFrame(rightward=('R.ASIS', 'L.ASIS'), anterior=('midASIS', 'midPSIS'),
                               primary='rightward', side=''),
    'femur_r_imu': SegmentFrame(superior=('R_HJC', 'R_KJC'), rightward=('R.Knee', 'R.MKnee'),
                                primary='superior', side='R'),
    'femur_l_imu': SegmentFrame(superior=('L_HJC', 'L_KJC'), rightward=('L.MKnee', 'L.Knee'),
                                primary='superior', side='L'),
    'tibia_r_imu': SegmentFrame(superior=('R_KJC', 'R_AJC'), rightward=('R.Ankle', 'R.MAnkle'),
                                primary='superior', side='R'),
    'tibia_l_imu': SegmentFrame(superior=('L_KJC', 'L_AJC'), rightward=('L.MAnkle', 'L.Ankle'),
                                primary='superior', side='L'),
}

# IMoVE needs no standing capture: every landmark below is in every take, tracked per frame, in
# the same file as the marker clusters. The one gap is the HIP CENTRE, which this marker set does
# not carry -- GTR is the greater trochanter, a surface point roughly 30 mm lateral and slightly
# distal to the centre. Over a 480 mm thigh that is a ~4-6 deg tilt of the long axis, which
# leaves the flexion axis (set by the epicondyles) intact and mixes ~10% between the adduction
# and rotation channels. The alternative is a published regression on pelvis geometry, which
# would trade a measured surface point for an unmeasured model; the surface point is declared
# instead.
IMOVE_POINTS = {
    'midASI': ('RASI', 'LASI'),
    'midPS': ('RPS1', 'RPS2', 'LPS1', 'LPS2'),
    'RASI': ('RASI',), 'LASI': ('LASI',),
    'RGTR': ('RGTR',), 'LGTR': ('LGTR',),
    'R_KJC': ('RLEP', 'RMEP'), 'L_KJC': ('LLEP', 'LMEP'),
    'R_AJC': ('RLML', 'RMML'), 'L_AJC': ('LLML', 'LMML'),
    'RLEP': ('RLEP',), 'RMEP': ('RMEP',), 'LLEP': ('LLEP',), 'LMEP': ('LMEP',),
    'RLML': ('RLML',), 'RMML': ('RMML',), 'LLML': ('LLML',), 'LMML': ('LMML',),
}

IMOVE_FRAMES = {
    'PELVIS': SegmentFrame(rightward=('RASI', 'LASI'), anterior=('midASI', 'midPS'),
                           primary='rightward', side=''),
    'THIGH_R': SegmentFrame(superior=('RGTR', 'R_KJC'), rightward=('RLEP', 'RMEP'),
                            primary='superior', side='R'),
    'THIGH_L': SegmentFrame(superior=('LGTR', 'L_KJC'), rightward=('LMEP', 'LLEP'),
                            primary='superior', side='L'),
    'SHANK_R': SegmentFrame(superior=('R_KJC', 'R_AJC'), rightward=('RLML', 'RMML'),
                            primary='superior', side='R'),
    'SHANK_L': SegmentFrame(superior=('L_KJC', 'L_AJC'), rightward=('LMML', 'LLML'),
                            primary='superior', side='L'),
}

DATASET_FRAMES = {'alborno': (ALBORNO_FRAMES, ALBORNO_POINTS),
                  'imove': (IMOVE_FRAMES, IMOVE_POINTS)}


def frames_for(dataset: str) -> Tuple[Dict[str, SegmentFrame], Dict[str, Tuple[str, ...]]]:
    """({segment: SegmentFrame}, {point: markers}) for a dataset, or a ValueError naming why not.

    REFUSES rather than falling back. The two biplane specs read a build with neither marker set
    registered, and a silently empty frame table would produce a statistics file whose anatomical
    axes are all NaN -- indistinguishable, three steps later, from a filter that failed.
    """
    try:
        return DATASET_FRAMES[dataset]
    except KeyError:
        raise ValueError(
            f"No anatomical frame definition for dataset {dataset!r}; have "
            f"{sorted(DATASET_FRAMES)}. The biplane halves carry no registered marker set (see "
            f"landmarks.SPECS), and the bone-pose spec would not need one: its world traces are "
            f"already in anatomical bone frames, so only the build's alignment rotation stands "
            f"between them and a basis.") from None


def segment_of(dataset: str, sensor: str) -> str:
    """The SEGMENT a sensor sits on, which is what has an anatomical frame.

    They differ only on IMoVE, where three sensors share each thigh and shank against one marker
    cluster: 'THIGH_R_H', 'THIGH_R_M' and 'THIGH_R_L' are all the right thigh. Each still gets
    its OWN basis, because each was fitted its own sensor-to-segment rotation at build time --
    same anatomy, different mounting.
    """
    if dataset == 'imove':
        return sensor.rsplit('_', 1)[0]
    return sensor


def parent_sensors(spec: TrackingSpec) -> Tuple[str, ...]:
    """The sensors that appear as a joint's PARENT, in joint-table order.

    The only ones that need a basis: `compute_error_stats` forms the error as
    rotvec(R_pc_imu R_pc_marker^-1), which is a rotation of the parent frame, so its components
    are parent-frame components and a child's basis would never be consulted.
    """
    return tuple(dict.fromkeys(parent for parent, _ in spec.joints.values()))


# ==============================================================================
# Geometry
# ==============================================================================

def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("cannot normalize a zero-length direction")
    return vector / norm


def anatomical_axes(frame: SegmentFrame, points: Dict[str, np.ndarray]
                    ) -> Tuple[np.ndarray, Dict[str, float]]:
    """(3x3 basis whose columns are (FE, AA, IE), geometry diagnostics).

    `points` holds each named point in the SEGMENT's frame, so the returned basis is in that same
    frame; composing it with the build's alignment rotation is what moves it to the sensor frame.

    The intermediate frame is (X, Y, Z) = (anterior, superior, right), right-handed, and the
    permutation to (FE, AA, IE) is `AXIS_SIGNS[frame.side]`. Both are orthonormal with det +1, so
    this is a change of basis and nothing else: the error vector's magnitude is untouched and the
    three components are its projections.
    """
    rightward = points[frame.rightward[0]] - points[frame.rightward[1]]
    superior = (points[frame.superior[0]] - points[frame.superior[1]]
                if frame.superior is not None else None)
    anterior = (points[frame.anterior[0]] - points[frame.anterior[1]]
                if frame.anterior is not None else None)

    if frame.primary == 'superior':
        y_axis = _unit(superior)
        z_axis = _unit(rightward - y_axis * (rightward @ y_axis))
        x_axis = np.cross(y_axis, z_axis)
        reference, primary_axis = rightward, y_axis
    else:
        z_axis = _unit(rightward)
        x_axis = _unit(anterior - z_axis * (anterior @ z_axis))
        y_axis = np.cross(z_axis, x_axis)
        reference, primary_axis = anterior, z_axis

    basis = np.column_stack([x_axis, y_axis, z_axis]) @ AXIS_SIGNS[frame.side]
    # How far the SECONDARY direction sat from perpendicular to the primary axis, i.e. the size of
    # the Gram-Schmidt correction, in [0, 90]. That is the mediolateral pair on a limb, where skin
    # over bone gives 5-20 deg and a swapped label gives 60+; on a `primary='rightward'` segment it
    # is the anterior reference instead, and the column name reads wrong for that one case.
    obliquity = 90.0 - np.degrees(np.arccos(np.clip(
        abs(_unit(reference) @ primary_axis), 0.0, 1.0)))
    diagnostics = {
        'lateral_obliquity_deg': float(obliquity),
        'rightward_mm': float(np.linalg.norm(rightward) * 1000.0),
        'superior_mm': float(np.linalg.norm(superior) * 1000.0) if superior is not None else np.nan,
        'anterior_mm': float(np.linalg.norm(anterior) * 1000.0) if anterior is not None else np.nan,
    }
    return basis, diagnostics


def align_rotation(dataset: str, subject: str, trial: str, sensor: str) -> np.ndarray:
    """C = R_{template <- IMU}: the constant rotation `align_world_to_imu` applied at build time.

    Read back out of the trial's `*.build.parquet` sidecar, which is the ONLY place it survives --
    the built world trace has it multiplied in, so nothing in the cached artifact can be used to
    recover it. `assembly.align_world_to_imu` records it as an angle plus a unit axis, which is a
    rotation vector split in two and reconstructs exactly.

    The sidecar is NOT part of the trial cache key (see `save_build_report`), so in principle it
    can be older than the parquet beside it. What would catch that is the cross-trial agreement
    in this module's report and the `gravity_check_deg` on every row; neither is optional reading.
    """
    path = build_report_path(dataset, subject, trial)
    if not path.exists():
        raise FileNotFoundError(
            f"No build report at {path}. The sensor-to-segment rotation only survives there; "
            f"rebuild the trial to regenerate it: python -m experiments.build_trials "
            f"--dataset {dataset} --subjects {subject} --trials {trial} --force")
    report = pd.read_parquet(path)
    rows = report[(report['step'] == 'S7_alignment') & (report['entity'] == sensor)]
    metrics = rows.set_index('metric')['value_num']
    if 'offset_angle_deg' not in metrics.index:
        raise KeyError(f"{dataset}/{subject}/{trial}: no S7_alignment for sensor {sensor!r}")
    axis = np.array([metrics[f'offset_axis_{i}'] for i in range(3)], dtype=float)
    return Rotation.from_rotvec(np.radians(metrics['offset_angle_deg']) * axis).as_matrix()


def to_sensor_frame(basis: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    """The anatomical basis moved from the marker-template frame into the SENSOR's frame.

    One line, and the composition order in it is the single least verifiable thing in this module.
    `align_world_to_imu` POST-multiplies:

        R_built(t) = R_template(t) C,     C = alignment = R_{template <- IMU}

    so a vector's template coordinates are C v_imu, its IMU coordinates are C^T v_template, and the
    axes — which are columns, i.e. vectors — transform the same way: A_imu = C^T A_template.

    WHY IT IS PINNED BY A TEST AND NOT ONLY BY THE DATA. `C A` is wrong and differs from `C^T A` by
    C^2. On Al Borno every plate is mounted essentially 180 deg flipped (`align_offset_deg` is
    177.7-180.0), so C^2 is a 0.48 deg rotation there and NO diagnostic in this module can tell the
    two apart: recomputing every basis the wrong way moves the knee's `hinge_angle_deg` by 0.03-0.7
    deg. On IMoVE, where the offsets are 173-176 deg, C^2 is 13 deg and the wrong order is visibly
    worse — `upright_check_deg` degrades on five of six joints (the ankle 7.8 -> 22.8, the hip 18.4
    -> 23.8, the knee 12.1 -> 19.3) and `hinge_angle_deg` at the hip goes 4.9-8.1 -> 13.4-14.2. So
    the data does prefer this order, on the one dataset able to express a preference; the test
    exists because a convention that only one of two datasets can check is not checked.
    """
    return alignment.T @ basis


def _reduce_landmarks(poses: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                      markers: Dict[str, np.ndarray], segment: str,
                      names: Sequence[str]) -> Tuple[Dict[str, np.ndarray], int]:
    """({marker: median position in the segment's frame}, frames the scarcest one had).

    c_segment(t) = R_plate(t)^T (p_marker(t) - p_plate(t)), medianed over the frames where both
    the plate and the marker are trustworthy. A median rather than a mean because a marker that
    drops out is NaN here but was a zero in both source formats, and one surviving zero would
    drag a mean across the room.
    """
    positions, rotations, valid = poses[segment]
    out, worst = {}, None
    for name in names:
        if name not in markers:
            continue
        n = min(len(positions), len(markers[name]))
        marker = markers[name][:n]
        usable = valid[:n] & np.isfinite(marker).all(axis=1)
        if usable.sum() < MIN_LANDMARK_FRAMES:
            continue
        out[name] = np.median(np.einsum('nji,nj->ni', rotations[:n][usable],
                                        marker[usable] - positions[:n][usable]), axis=0)
        worst = int(usable.sum()) if worst is None else min(worst, int(usable.sum()))
    return out, (worst or 0)


def _points_from_markers(reduced: Dict[str, np.ndarray], point_table: Dict[str, Tuple[str, ...]],
                         wanted: Sequence[str]) -> Optional[Dict[str, np.ndarray]]:
    """{point: position}, each the midpoint of its markers, or None if any is unavailable.

    All-or-nothing per segment. A frame built from a subset of the markers it asks for is not a
    degraded frame, it is a different one, and it would sit in the same column as the others.
    """
    points = {}
    for name in wanted:
        markers = point_table[name]
        if any(marker not in reduced for marker in markers):
            return None
        points[name] = np.mean([reduced[marker] for marker in markers], axis=0)
    return points


# ==============================================================================
# Per-dataset landmark sources
# ==============================================================================

def _alborno_sources(subject: str, trial: str, segments: Sequence[str]
                     ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, int], str]:
    """({segment: {marker: position in the segment frame}}, {segment: n frames}, source).

    TWO CAPTURES, ONE TEMPLATE. The medial markers exist only in the standing capture, so that is
    where the landmarks come from; the frame they have to land in is the one the TRIAL's plate
    reconstruction defines, because that is what the build's alignment rotation was measured
    against. Reading the trial first and feeding its templates to the standing capture puts the
    standing landmarks directly in the trial's frame -- no re-basing, nothing to cancel.

    `landmarks.read_alborno_plates` is used for the trial rather than `alborno.load_world_traces`
    only because it RETURNS the templates it estimated. The two agree exactly (see --validate).
    """
    trc = landmarks._alborno_trc(subject, trial)
    _, templates = landmarks.read_alborno_plates(trc, {s: s for s in segments})
    missing = [s for s in segments if s not in templates]
    if missing:
        print(f"  {subject}/{trial}: no marker plate for {missing} — skipped")

    names = tuple(dict.fromkeys(m for markers in ALBORNO_POINTS.values() for m in markers))
    static = landmarks.read_alborno_static(subject, list(templates), names)
    if static is None:
        raise FileNotFoundError(
            f"No standing capture at {landmarks.alborno_static_trc(subject)}. It is the only Al "
            f"Borno file carrying medial knee and ankle markers, so without it no segment frame "
            f"can be built for this subject.")
    poses, _ = landmarks.read_alborno_plates(
        landmarks.alborno_static_trc(subject),
        {s: landmarks.ALBORNO_STATIC_PLATES[s] for s in templates
         if s in landmarks.ALBORNO_STATIC_PLATES},
        templates=templates, suffixes=landmarks.ALBORNO_STATIC_SUFFIXES)

    reduced, counts = {}, {}
    for segment in poses:
        reduced[segment], counts[segment] = _reduce_landmarks(
            poses, static.markers, segment, names)
    return reduced, counts, 'standing'


def _imove_sources(session: str, trial: str, segments: Sequence[str]
                   ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, int], str]:
    """The same, from ONE Motive take -- landmarks and clusters are in the same file.

    Simpler than the Al Borno path in every way that matters: this marker set carries the medial
    epicondyles and malleoli in EVERY take, tracked per frame, beside the marker clusters, so
    there is no second capture to reconcile and no template to carry across files. The cost is
    that the landmark half is then measured per trial rather than once per session, which makes
    the cross-trial agreement in the report a check on BOTH halves here rather than on the
    alignment rotation alone.

    THE TAKE IS A CHOICE on the long walks, which merge three ~950 s Motive takes onto one
    inertial record. Each take's plate template is estimated independently by
    `fit_plate_to_template`, so the merged world trace's frame is whichever take covered a given
    frame, while the build fitted ONE alignment rotation across all of them. The take with the
    most tracked landmark frames is used and named in `source`.

    THE DISAGREEMENT BETWEEN TAKES IS NOT MEASURED HERE, and the omission is deliberate rather
    than an oversight: the build already mixes take frames inside one trial, so no single constant
    basis can be right for all of them and measuring the residual would not change what is
    written. Its scale is the template re-basing `landmarks.read_alborno_plates` documents at a
    median of 0.08 deg — small against the 6-13 deg the alignment rotation itself moves between
    IMoVE trials.
    """
    takes = imove_mocap.mocap_takes_for(IMOVE_ROOT / session, trial)
    if not takes:
        raise FileNotFoundError(f"No mocap take for imove/{session}/{trial}")
    names = tuple(dict.fromkeys(m for markers in IMOVE_POINTS.values() for m in markers))

    per_take = []
    for take in takes:
        poses = {segment: (trace.positions, trace.rotations, np.asarray(trace.valid))
                 for segment, trace in imove_mocap.load_world_traces(take).items()}
        _, markers = landmarks.read_imove_markers(take, names)
        reduced, counts = {}, {}
        for segment in [s for s in segments if s in poses]:
            reduced[segment], counts[segment] = _reduce_landmarks(poses, markers, segment, names)
        per_take.append((reduced, counts))

    best = max(range(len(per_take)), key=lambda i: sum(per_take[i][1].values()))
    return per_take[best][0], per_take[best][1], f'take{best + 1}of{len(takes)}'


LANDMARK_SOURCES = {'alborno': _alborno_sources, 'imove': _imove_sources}


# ==============================================================================
# The basis table
# ==============================================================================

def _median_direction(vectors: np.ndarray) -> Optional[np.ndarray]:
    """The componentwise median of a tight cluster of directions, renormalized."""
    if len(vectors) == 0:
        return None
    median = np.median(vectors, axis=0)
    norm = np.linalg.norm(median)
    return None if norm < 1e-6 else median / norm


def _up_checks(plate, basis: np.ndarray, side: str, gravity: np.ndarray) -> Dict[str, object]:
    """Two angles that between them test the two halves of the chain. Degrees.

    `align_check_deg` — the angle, IN THE SENSOR'S FRAME, between the up direction the
        accelerometer reports and the up direction the mocap reports. POSTURE CANCELS: both are
        measured on the same samples in the same frame, so whatever the subject was doing enters
        both identically and what is left is the rotation between the two frames, i.e. the
        build's alignment rotation (plus its sync). It does not involve the anatomical basis at
        all, which is exactly why it isolates the one link this module reads out of a sidecar
        rather than recomputing. A few degrees is the walking dynamics the accelerometer sees and
        the mocap's gravity direction does not.

    `upright_check_deg` — the angle between the segment's own SUPERIOR axis and the mocap's up
        direction. Tests the LANDMARK half: the alignment rotation cancels out of it algebraically
        (R_built A_imu = R_template A_template), so it is a statement about whether the axis the
        landmarks defined really points up the segment. POSTURE DOES NOT CANCEL here, so it is
        only small where the median posture is upright — true of the thigh and shank over a
        walking trial, not true over complexTasks, and the number is reported rather than enforced
        for that reason.

        NOT INTERPRETABLE FOR THE PELVIS, which has no long axis: its superior axis is the normal
        to the ASIS/PSIS plane, and that plane is tilted by the subject's own pelvic tilt. The
        10-26 deg it reads there is anatomy, not error.

    Both are measured over the VALID window only: outside it the world pose is interpolated or
    held (`_merge_takes` holds the last covered pose across the gaps between IMoVE takes), and
    comparing an accelerometer against a held pose measures the hold.
    """
    valid = np.asarray(plate.valid)
    if not valid.any():
        return {'align_check_deg': float('nan'), 'upright_check_deg': float('nan'),
                'n_valid_frames': 0}
    world_up = gravity / np.linalg.norm(gravity)
    # Up in the sensor frame, two ways: from the accelerometer, and from the mocap pose.
    from_acc = _median_direction(plate.imu_trace.acc[valid])
    from_mocap = _median_direction(np.einsum('nji,j->ni', plate.world_trace.rotations[valid],
                                            world_up))
    superior = basis[:, 2] * (-1.0 if side == 'L' else 1.0)

    def angle(a, b):
        return (float('nan') if a is None or b is None else
                float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))))

    return {'align_check_deg': angle(from_acc, from_mocap),
            'upright_check_deg': angle(from_mocap, superior),
            'n_valid_frames': int(valid.sum())}


def _mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """The chordal mean of a stack of rotations: the SO(3) projection of their arithmetic mean.

    Well defined however far the stack sits from the identity, which is why the reference pose is
    taken this way rather than by averaging rotation vectors.
    """
    u, _, vt = np.linalg.svd(rotations.mean(axis=0))
    mean = u @ vt
    if np.linalg.det(mean) < 0:                     # nearest ROTATION, not nearest orthogonal
        u[:, -1] *= -1.0
        mean = u @ vt
    return mean


def _hinge_checks(plates, sensor: str, basis: np.ndarray,
                  spec: TrackingSpec) -> List[Dict[str, object]]:
    """Per joint this sensor parents: is the landmark flexion axis the axis the joint MOVES about?

    THE ANATOMICAL CHECK, and the only one here that tests the axis NAMES rather than the frame's
    orthonormality. Two independent routes to one physical axis are compared:

        landmark    the mediolateral axis of the parent segment, from a medial/lateral marker
                    pair and a long axis — this module's `FE`.
        functional  the principal direction of the reference joint rotation itself, from the
                    MOCAP poses in the same frame, with no landmark anywhere in it.

    A knee is close to a hinge, so at the knee the two must nearly coincide and `hinge_angle_deg`
    is a sharp test: 0 deg means the markers named the axis the joint actually turns about.
    A hip is a ball joint and an ankle-to-foot-plate pair is not much better, so there the
    principal direction is whatever that trial's motion happened to emphasise and a large angle
    says nothing about the frame. `var_frac1` is how concentrated the motion was, and it is what
    decides whether `hinge_angle_deg` is evidence: read the two together or neither.

    `rom_fe/aa/ie_deg` are the reference joint angle's range on each anatomical axis, which is the
    other half of the same statement — at the knee, flexion should hold most of the range.
    """
    rows = []
    for joint, (parent, child) in spec.joints.items():
        if parent != sensor or child not in plates or parent not in plates:
            continue
        parent_plate, child_plate = plates[parent], plates[child]
        valid = np.asarray(parent_plate.valid) & np.asarray(child_plate.valid)
        if valid.sum() < MIN_LANDMARK_FRAMES:
            continue
        relative = np.einsum('tji,tjk->tik', parent_plate.world_trace.rotations[valid],
                             child_plate.world_trace.rotations[valid])
        # The excursion FROM this trial's mean posture, as a rotation of the parent frame:
        # rotvec(R_pc(t) R_0^T), which is the same algebraic form as the error this module's
        # basis is built to decompose. Subtracting the median of the rotvecs instead would be
        # wrong here rather than merely approximate -- these plate pairs carry a mounting offset
        # of order 170 deg, and near the antipode a rotation vector's components stop tracking
        # the physical angles at all: it reported 74 deg of hip 'adduction'.
        mean_pose = _mean_rotation(relative)
        centred = Rotation.from_matrix(
            np.einsum('tij,kj->tik', relative, mean_pose)).as_rotvec()
        eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred)
        principal = eigenvectors[:, np.argmax(eigenvalues)]
        components = np.degrees(centred @ basis)
        rows.append({
            'joint': joint, 'sensor': sensor,
            # An axis is a LINE, so the sign of the principal direction is arbitrary and the
            # angle is folded into [0, 90] with abs() before the arccos.
            'hinge_angle_deg': float(np.degrees(np.arccos(np.clip(
                abs(principal @ basis[:, 0]), 0.0, 1.0)))),
            'var_frac1': float(np.max(eigenvalues) / np.sum(eigenvalues)),
            **{f'rom_{axis.lower()}_deg': float(np.ptp(components[:, i]))
               for i, axis in enumerate(AXES)},
            # Positive skew on FE is the SIGN CHECK: a knee flexes and does not hyperextend, so
            # its one-sided excursion has to come out positive if the convention is right.
            'skew_fe': float(np.mean((components[:, 0] - components[:, 0].mean()) ** 3)
                             / max(np.std(components[:, 0]) ** 3, 1e-12)),
            'n_frames': int(valid.sum()),
        })
    return rows


def trial_bases(dataset: str, subject: str, trial: str,
                spec: Optional[TrackingSpec] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One trial's (bases, checks): a row per parent sensor, and a row per joint it parents.

    A sensor is skipped with a printed note rather than raising when its landmarks, its marker
    plate or its alignment rotation is missing — a biplane trial images one knee, an Al Borno
    subject can lose a plate to a reconstruction failure (Subject05's femur_r), and a partial
    trial should yield partial results the way every other per-joint loop here does.
    """
    spec = tracking_spec(dataset) if spec is None else spec
    frames, points = frames_for(dataset)
    sensors = [s for s in parent_sensors(spec) if segment_of(dataset, s) in frames]
    segments = tuple(dict.fromkeys(segment_of(dataset, s) for s in sensors))

    reduced, counts, source = LANDMARK_SOURCES[dataset](subject, trial, segments)
    plates = load_trial(subject, trial, dataset=spec.build_dataset)

    rows, checks = [], []
    for sensor in sensors:
        segment = segment_of(dataset, sensor)
        frame = frames[segment]
        if segment not in reduced:
            print(f"  {subject}/{trial}/{sensor}: segment {segment} not tracked — skipped")
            continue
        resolved = _points_from_markers(reduced[segment], points, frame.points())
        if resolved is None:
            have = sorted(reduced[segment])
            print(f"  {subject}/{trial}/{sensor}: missing landmarks for {frame.points()} "
                  f"(have {have}) — skipped")
            continue
        try:
            basis, geometry = anatomical_axes(frame, resolved)
            alignment = align_rotation(dataset, subject, trial, sensor)
        except (KeyError, ValueError, FileNotFoundError) as error:
            print(f"  {subject}/{trial}/{sensor}: {type(error).__name__}: {error} — skipped")
            continue

        basis_imu = to_sensor_frame(basis, alignment)
        determinant = float(np.linalg.det(basis_imu))
        if abs(determinant - 1.0) > 1e-6:
            raise AssertionError(
                f"{dataset}/{subject}/{trial}/{sensor}: basis determinant {determinant} is not "
                f"+1, so it is not a right-handed orthonormal frame and the sign convention "
                f"documented in this module does not hold for it.")

        labels = {'dataset': dataset, 'subject': subject, 'trial': trial}
        rows.append({
            **labels, 'sensor': sensor,
            'segment': segment, 'side': frame.side, 'source': source,
            **dict(zip(BASIS_COLUMNS, basis_imu.reshape(-1))),
            **geometry,
            'n_landmark_frames': counts.get(segment, 0),
            'align_offset_deg': float(np.degrees(np.arccos(np.clip(
                (np.trace(alignment) - 1.0) / 2.0, -1.0, 1.0)))),
            **(_up_checks(plates[sensor], basis_imu, frame.side, spec.gravity)
               if sensor in plates else
               {'align_check_deg': float('nan'), 'upright_check_deg': float('nan'),
                'n_valid_frames': 0}),
            'determinant': determinant,
        })
        checks.extend({**labels, **check}
                      for check in _hinge_checks(plates, sensor, basis_imu, spec))
    return pd.DataFrame(rows), pd.DataFrame(checks)


def table_path(dataset: str, table: str = 'bases') -> Path:
    return EXPERIMENT_DIR / dataset / f'{table}.parquet'


def bases_path(dataset: str) -> Path:
    return table_path(dataset, 'bases')


def build_bases(dataset: str, row_keys: Sequence[Tuple[str, str]]
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Every requested trial's bases and checks, concatenated, and written to `table_path`."""
    spec = tracking_spec(dataset)
    bases, checks = [], []
    for subject, trial in row_keys:
        print(f"{dataset}/{subject}/{trial}")
        try:
            trial_basis, trial_checks = trial_bases(dataset, subject, trial, spec=spec)
        except (FileNotFoundError, ValueError, KeyError) as error:
            print(f"  {type(error).__name__}: {error} — trial skipped")
            continue
        bases.append(trial_basis)
        checks.append(trial_checks)

    tables = {name: (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
              for name, frames in (('bases', bases), ('checks', checks))}
    for name, table in tables.items():
        if table.empty:
            continue
        path = paths.ensure_parent(table_path(dataset, name))
        table.to_parquet(path, engine='pyarrow', index=False)
        paths.write_manifest(path, experiment=EXPERIMENT_NAME, dataset=dataset, table=name,
                             axes=list(AXES), n_trials=len(row_keys))
        print(f"Wrote {len(table)} rows to {path}")
    return tables['bases'], tables['checks']


def load_bases(dataset: str, table: str = 'bases') -> Optional[pd.DataFrame]:
    """The dataset's basis (or checks) table, or None if it has not been built."""
    path = table_path(dataset, table)
    if not path.exists():
        return None
    return pd.read_parquet(path, engine='pyarrow')


def bases_by_joint(dataset: str, subject: str, trial: str,
                   spec: Optional[TrackingSpec] = None,
                   table: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
    """{joint_name: 3x3 basis of that joint's PARENT sensor}, for one trial.

    The shape `compute_error_stats` wants: it holds an error vector per joint in the parent's
    frame and needs the basis that goes with it, with the sensor-to-joint indirection already
    resolved. Joints whose parent has no basis are simply absent, and the caller reports what it
    did not get rather than filling in an identity — an identity here would not fail, it would
    silently relabel the sensor's own axes as anatomical ones.
    """
    spec = tracking_spec(dataset) if spec is None else spec
    table = load_bases(dataset) if table is None else table
    if table is None or table.empty:
        return {}
    rows = table[(table['subject'] == subject) & (table['trial'] == trial)]
    by_sensor = {row.sensor: np.array([getattr(row, col) for col in BASIS_COLUMNS]).reshape(3, 3)
                 for row in rows.itertuples()}
    return {joint: by_sensor[parent] for joint, (parent, _) in spec.joints.items()
            if parent in by_sensor}


def basis_frame(dataset: str, subject: Optional[str] = None, trial: Optional[str] = None,
                spec: Optional[TrackingSpec] = None,
                table: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The basis table re-keyed BY JOINT, which is what `compute_error_stats` merges against.

    Columns: subject, trial, joint_name, and `BASIS_COLUMNS`. One row per (trial, joint) rather
    than per (trial, sensor), because a sensor parents up to three joints and the error table is
    keyed by joint — the pelvis's basis is the basis of the lumbar AND both hips.

    `subject`/`trial` narrow it to one trial for the per-trial statistics worker; left out, the
    whole dataset comes back, which is what a pooled caller needs. THE KEY COLUMNS ARE ALWAYS
    PRESENT either way, so a pooled caller cannot accidentally apply one subject's mounting to
    another's error.

    THE SUBJECT COLUMN IS RELABELLED ON THE WAY OUT. `bases.parquet` stores the id the paths and
    the CLI use ('01'), and the joint-angle tables this is merged against store the label
    ('Subject01') — see `TrackingSpec.label_subject`. Emitting the raw id here matches nothing on
    the other side and silently produces no anatomical rows at all, which is how this was first
    written. The `subject` ARGUMENT is still the raw id, because that is what a caller has.
    """
    spec = tracking_spec(dataset) if spec is None else spec
    table = load_bases(dataset) if table is None else table
    if table is None or table.empty:
        return pd.DataFrame(columns=['subject', 'trial', 'joint_name', *BASIS_COLUMNS])
    if subject is not None:
        table = table[table['subject'] == subject]
    if trial is not None:
        table = table[table['trial'] == trial]

    parents = {}
    for joint, (parent, _) in spec.joints.items():
        parents.setdefault(parent, []).append(joint)
    rows = []
    for row in table.itertuples():
        for joint in parents.get(row.sensor, []):
            rows.append({'subject': spec.label_subject(row.subject), 'trial': row.trial,
                         'joint_name': joint,
                         **{column: getattr(row, column) for column in BASIS_COLUMNS}})
    return pd.DataFrame(rows, columns=['subject', 'trial', 'joint_name', *BASIS_COLUMNS])


# ==============================================================================
# Report and validation
# ==============================================================================

def cross_trial_agreement(table: pd.DataFrame) -> pd.DataFrame:
    """How far a subject's two trials disagree about the same physical mounting, in degrees.

    THE STRONGEST CHECK AVAILABLE HERE. The landmark half of the chain is measured once per
    subject and is identical between that subject's trials by construction, so everything this
    disagreement contains comes from the per-trial alignment rotation — the one link that is read
    out of a sidecar rather than recomputed. A subject with one trial contributes nothing and is
    absent rather than reported as agreeing perfectly.
    """
    rows = []
    for (subject, sensor), group in table.groupby(['subject', 'sensor'], observed=True):
        if len(group) < 2:
            continue
        matrices = [np.array([getattr(row, col) for col in BASIS_COLUMNS]).reshape(3, 3)
                    for row in group.itertuples()]
        angles = [np.degrees(np.linalg.norm(Rotation.from_matrix(a.T @ b).as_rotvec()))
                  for i, a in enumerate(matrices) for b in matrices[i + 1:]]
        rows.append({'subject': subject, 'sensor': sensor, 'n_trials': len(group),
                     'max_disagreement_deg': float(np.max(angles))})
    return pd.DataFrame(rows)


def report(table: pd.DataFrame, checks: Optional[pd.DataFrame] = None) -> None:
    """The console report: geometry, the checks, and whatever was flagged."""
    if table.empty:
        print("No bases built.")
        return
    pd.set_option('display.width', 220)
    print(f"\n{len(table)} bases over "
          f"{table[['subject', 'trial']].drop_duplicates().shape[0]} trial(s), "
          f"{table['subject'].nunique()} subject(s), "
          f"{table['sensor'].nunique()} sensor(s).")

    print("\n=== Landmark geometry, by segment ===")
    print(table.groupby(['segment', 'side'], observed=True).agg(
        n=('sensor', 'size'),
        superior_mm=('superior_mm', 'median'),
        rightward_mm=('rightward_mm', 'median'),
        obliquity_deg=('lateral_obliquity_deg', 'median'),
        align_offset_deg=('align_offset_deg', 'median'),
        landmark_frames=('n_landmark_frames', 'min')).round(2).to_string())

    print("\n=== Up-direction checks (deg): align_check tests the alignment rotation "
          "(posture cancels), upright_check tests the landmark axes (it does not) ===")
    print(table.groupby(['segment', 'trial'], observed=True).agg(
        n=('sensor', 'size'),
        align_med=('align_check_deg', 'median'), align_max=('align_check_deg', 'max'),
        upright_med=('upright_check_deg', 'median'),
        upright_max=('upright_check_deg', 'max')).round(2).to_string())

    if checks is not None and not checks.empty:
        print("\n=== Flexion axis vs the joint's own principal axis (deg). Read hinge_angle "
              "ONLY where var_frac1 is high — a ball joint has no axis to agree with ===")
        print(checks.groupby('joint', observed=True).agg(
            n=('sensor', 'size'),
            hinge_med=('hinge_angle_deg', 'median'), hinge_max=('hinge_angle_deg', 'max'),
            var_frac1=('var_frac1', 'median'),
            rom_fe=('rom_fe_deg', 'median'), rom_aa=('rom_aa_deg', 'median'),
            rom_ie=('rom_ie_deg', 'median'), skew_fe=('skew_fe', 'median')
        ).round(2).to_string())

    agreement = cross_trial_agreement(table)
    if not agreement.empty:
        print("\n=== Cross-trial agreement of the basis (deg) — the check with the most power "
              "on the alignment rotation ===")
        print(agreement.groupby('sensor', observed=True)['max_disagreement_deg']
              .agg(['count', 'median', 'max']).round(3).to_string())
        worst = agreement.nlargest(3, 'max_disagreement_deg')
        print("worst:", ", ".join(f"{r.subject}/{r.sensor} {r.max_disagreement_deg:.2f}"
                                  for r in worst.itertuples()))

    flags = (('lateral obliquity', 'lateral_obliquity_deg', MAX_LATERAL_OBLIQUITY_DEG),
             ('alignment-rotation check', 'align_check_deg', MAX_ALIGN_CHECK_DEG))
    clean = True
    for name, column, limit in flags:
        flagged = table[table[column] > limit]
        if flagged.empty:
            continue
        clean = False
        print(f"\n!!! {len(flagged)} row(s) over {limit:g} deg on the {name}"
              f"{f', worst {FLAG_PRINT_LIMIT}' if len(flagged) > FLAG_PRINT_LIMIT else ''}:")
        print(flagged.nlargest(FLAG_PRINT_LIMIT, column)[['subject', 'trial', 'sensor', column]]
              .round(2).to_string(index=False))
    if clean:
        print(f"\nNo row exceeds {MAX_LATERAL_OBLIQUITY_DEG:g} deg lateral obliquity or "
              f"{MAX_ALIGN_CHECK_DEG:g} deg on the alignment-rotation check.")


def validate(dataset: str, row_keys: Sequence[Tuple[str, str]]) -> bool:
    """Does the landmark reader reproduce the BUILD's plate frames, on this dataset's own data?

    The assumption the whole module rests on and the one that cannot be checked from inside it:
    the anatomical axes are expressed in the frame `landmarks.read_alborno_plates` produces, and
    the alignment rotation they are composed with was measured against the frame
    `alborno.load_world_traces` produces. If those two ever differ by a constant, every basis
    here is wrong by it and nothing else in this module would notice — the determinant is still
    +1, the cross-trial agreement is still perfect, and only the gravity check would drift.

    Both readers run over the same file at the same rate, so the comparison is per-frame and
    exact. Only meaningful on alborno, where two readers exist; imove has one.
    """
    if dataset != 'alborno':
        print(f"--validate only applies to alborno, which has two plate readers; "
              f"{dataset} has one ({imove_mocap.load_world_traces.__name__}).")
        return True
    ok = True
    for subject, trial in row_keys:
        trc = landmarks._alborno_trc(subject, trial)
        built = alborno.load_world_traces(trc)
        poses, _ = landmarks.read_alborno_plates(trc, {s: s for s in built})
        for sensor, (_, rotations, valid) in poses.items():
            reference = built[sensor]
            n = min(len(rotations), len(reference.rotations))
            usable = valid[:n] & np.asarray(reference.valid)[:n]
            relative = np.einsum('tji,tjk->tik', rotations[:n][usable],
                                 reference.rotations[:n][usable])
            worst = float(np.degrees(np.abs(
                Rotation.from_matrix(relative).as_rotvec()).max()))
            if worst > 1e-6:
                ok = False
                print(f"  MISMATCH {subject}/{trial}/{sensor}: {worst:.6f} deg between "
                      f"landmarks.read_alborno_plates and alborno.load_world_traces")
        print(f"  {subject}/{trial}: {len(poses)} plate(s) checked")
    print("\nReaders agree exactly." if ok else "\nREADERS DISAGREE — every basis is rotated "
                                                "by the difference. Do not use this artifact.")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='alborno', choices=sorted(DATASETS),
                        help="Which built dataset to measure frames for.")
    parser.add_argument('--subjects', nargs='+', default=None,
                        help="Restrict to these subject/session ids (default: everything built).")
    parser.add_argument('--trials', nargs='+', default=None,
                        help="Restrict to these trial names.")
    parser.add_argument('--report-only', action='store_true',
                        help="Report the existing artifact without rebuilding it.")
    parser.add_argument('--validate', action='store_true',
                        help="Check that the landmark reader reproduces the build's plate frames.")
    args = parser.parse_args()

    row_keys = select_trials(args.dataset, args.subjects, args.trials)
    if args.validate:
        raise SystemExit(0 if validate(args.dataset, row_keys) else 1)

    if args.report_only:
        table, checks = load_bases(args.dataset), load_bases(args.dataset, 'checks')
        if table is None:
            raise SystemExit(f"No artifact at {bases_path(args.dataset)}. Run without "
                             f"--report-only first.")
        if args.subjects:
            table = table[table['subject'].isin(args.subjects)]
            checks = None if checks is None else checks[checks['subject'].isin(args.subjects)]
    else:
        frames_for(args.dataset)     # refuse an unsupported dataset before reading 280 trials
        table, checks = build_bases(args.dataset, row_keys)
    report(table, checks)


if __name__ == "__main__":
    os.environ.setdefault("DISABLE_TQDM", "True")
    main()
