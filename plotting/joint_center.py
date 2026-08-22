"""
Where the sensors and the fitted joint centres actually are, in 3D, one figure per subject.

Every IMU is drawn as a rectangular prism at its plate's world position and oriented by that
plate's world-from-body rotation, so the picture shows the sensor's POSE and not just a point.
Every joint centre from experiments/joint_center.py is a dot, with a thin line back to each
sensor that projects to it — that line is the lever arm `IMUTrace.project_acc` moves the
accelerometer along, and its length is what an offset error gets multiplied by.

This is a sanity check the numbers cannot give you. A joint-centre fit can have a small residual
and still sit outside the limb, and the only cheap way to notice is to look: a knee dot floating
in front of the shank, a sensor prism rotated 90 degrees against its neighbours, or a hip centre
placed on the wrong side of the pelvis are all obvious here and all invisible in a table of
millimetres.

Reads only what is already on disk — the built trial parquet for the poses and
`joint_center.joint_offsets`, which recomputes them at ~17 ms per joint — the same deterministic
fit every other consumer gets, so what is drawn is what the projections use.

    python -m plotting.joint_center --dataset alborno
    python -m plotting.joint_center --dataset imove --subjects s13
    python -m plotting.joint_center --dataset imove --sensor-scale 1   # true sensor size

Two dataset-specific things worth knowing before reading a figure:

  * ON IMoVE, THE PRISM IS THE SENSOR. `imove_mocap.load_trial` shifts each plate's mocap origin
    onto its own IMU, so the position drawn is where the sensor is. Thigh and shank each carry
    three, and all three are drawn — they sit along the segment and their spacing is the real
    thing the placement analysis is about.
  * ON AL BORNO, THE PRISM IS THE MARKER PLATE. That build applies no sensor offset, so the IMU
    is assumed coincident with the plate it is mounted on and there is no separate sensor
    position to draw. The prism is therefore the plate's pose, which is the same pose the
    projection uses, but it is not independent evidence of where the sensor sits.

Joint centres are drawn as the MIDPOINT of the two segments' implied centres. The two disagree
by the fit residual — 6-13 mm, section 1 of the joint_center report — which is around a tenth of
a sensor's length and invisible at body scale, so drawing both would render as one dot anyway.
On IMoVE the placement variants each produce their own estimate and all are drawn, lighter: how
tightly those cluster is how much the joint centre depends on which sensor you fit it from.
"""
import argparse
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import paths
from plotting import utils as plot_utils  # noqa: F401  (applies the shared paper rcParams on import)
from experiments.experiment_utils import load_trial
from experiments.global_assumptions import DATASETS, enumerate_trials, get_dataset, subjects_of
from experiments.joint_center import joint_offsets
from src.toolchest.PlateTrial import PlateTrial

PLOTS_DIR = paths.plots_dir("joint_center")

# Xsens MTw2 outer dimensions, in metres. Drawn at true size by default: a 47 mm box against a
# 1.8 m subject is small, and that IS the point — the sensor is a tenth the length of the lever
# arm it is being projected along. --sensor-scale exaggerates it when the geometry matters more
# than the proportion.
SENSOR_SIZE_M = (0.047, 0.030, 0.013)
# Drawn larger than life by default. At true size a 47 mm box against a 1.8 m subject is four
# pixels and its orientation — the only reason it is a prism and not a dot — cannot be read. The
# exaggeration is stated in every figure's footer, and --sensor-scale 1 restores true scale for
# judging the sensor against the lever arm it is projected along.
DEFAULT_SENSOR_SCALE = 3.0

SENSOR_COLOR = '#e07b1f'      # orange
SENSOR_EDGE = '#8a4708'
JOINT_COLOR = '#c0243a'       # red
VARIANT_COLOR = '#e88a98'     # lighter red, for IMoVE's placement variants
ARM_COLOR = '#9aa4b0'

# (elevation, azimuth, title, depth axis) for the three panels. All three are 3D rather than
# flat projections so the prisms still read as prisms; a flat projection collapses the one thing
# the prism is there to show. The depth axis is the one pointing at the viewer in
# that view; its ticks are meaningless there and they collide with the vertical axis's, so they
# are suppressed. None keeps all three, which the oblique view needs.
# Named by the WORLD plane each looks onto, not by an anatomical one. A subject's heading in
# the capture volume is arbitrary and changes between trials, so calling a fixed azimuth
# "frontal" is a claim about where they happened to be facing — and on the two trials checked
# while building this it would have been wrong on one of them. Which of these two views
# separates the legs therefore varies by trial, and that is the honest rendering.
VIEWS = ((12, -88, 'World x–y', 'y'), (12, -2, 'World z–y', 'x'), (26, -55, 'Oblique', None))

# World up axis. Both datasets are Y-up, and matplotlib's 3D axes put their third coordinate
# vertically, so the drawing order is (x, z, y) throughout and the vertical axis is labelled Y.
UP_AXIS = 1


def sensor_prism(position: np.ndarray, rotation: np.ndarray,
                 size: Sequence[float] = SENSOR_SIZE_M, scale: float = 1.0) -> np.ndarray:
    """The six faces of a box centred at `position` and oriented by `rotation`, as (6, 4, 3).

    `rotation` is world-from-body (the same convention `get_imu_trace_in_global_frame` uses to
    rotate a reading into the world), so a corner at body coordinate c lands at
    position + rotation @ c. Getting that backwards draws every sensor with its orientation
    inverted, which looks plausible on a symmetric box — hence stating it here rather than
    trusting the reader to infer it.
    """
    half = 0.5 * scale * np.asarray(size, dtype=float)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=float)
    corners = position + (rotation @ (signs * half).T).T
    # Face vertex indices into `corners`, ordered so each quad traces its perimeter rather than
    # a bowtie. Index bits are (x, y, z) with 0 = negative.
    faces = [(0, 1, 3, 2), (4, 5, 7, 6),      # -x, +x
             (0, 1, 5, 4), (2, 3, 7, 6),      # -y, +y
             (0, 2, 6, 4), (1, 3, 7, 5)]      # -z, +z
    return np.array([[corners[i] for i in face] for face in faces])


def to_plot_frame(points: np.ndarray) -> np.ndarray:
    """World (x, y, z) -> plot (x, z, y), so the figure's vertical axis is the world up axis."""
    points = np.atleast_2d(points)
    order = [i for i in range(3) if i != UP_AXIS] + [UP_AXIS]
    return points[..., order]


def pick_frame(plates: Dict[str, PlateTrial]) -> Optional[int]:
    """The STILLEST frame at which every plate has trustworthy mocap.

    Two criteria, and the second matters more than it looks. Validity is obvious — a padded pose
    draws a segment where the subject never was. Stillness is what makes the figure readable: the
    middle of a valid run is usually mid-stride, which draws a scissored pose whose segments
    overlap in two of the three views, and nothing about the sensor geometry is easier to judge
    there. The quietest instant is a standing pose.

    Restricted to the longest valid RUN before choosing, because reconstruction is least reliable
    at the edges of a tracked stretch and a lone valid frame between two dropouts can be a glitch
    that happens to be slow.
    """
    if not plates:
        return None
    n = min(len(plate) for plate in plates.values())
    valid = np.ones(n, dtype=bool)
    for plate in plates.values():
        valid &= np.asarray(plate.valid)[:n]
    if not valid.any():
        return None
    padded = np.concatenate(([0], valid.astype(np.int8), [0]))
    edges = np.diff(padded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    start, end = [(s, e) for s, e in zip(starts, ends)][int(np.argmax(ends - starts))]
    if end - start < 3:
        return int((start + end) // 2)

    # Mean sensor speed, from the world positions. Gyro would do as well and is not needed:
    # translating and rotating stop together on a body.
    speed = np.zeros(end - start)
    for plate in plates.values():
        positions = plate.world_trace.positions[start:end]
        step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        speed += np.concatenate(([step[0]], step))
    return int(start + np.argmin(speed))


def joint_center_points(plates: Dict[str, PlateTrial], offsets: Dict[str, Dict[str, np.ndarray]],
                        frame: int, spec) -> List[dict]:
    """One entry per fitted joint: its world position at `frame`, and the two sensor positions
    it was fitted from.

    The position is the midpoint of the two segments' implied centres. Both are computed so the
    caller could draw the disagreement, but they differ by the fit residual — 6-13 mm — which is
    a fifth of a sensor's length and will not resolve at body scale.
    """
    points = []
    for joint, pair in offsets.items():
        if joint not in spec.joints:
            continue
        parent_sensor, child_sensor = spec.joints[joint]
        if parent_sensor not in plates or child_sensor not in plates:
            continue
        implied = []
        for sensor, key in ((parent_sensor, 'parent'), (child_sensor, 'child')):
            world = plates[sensor].world_trace
            implied.append(world.positions[frame] + world.rotations[frame] @ pair[key])
        points.append({
            'joint': joint,
            'position': 0.5 * (implied[0] + implied[1]),
            'separation_mm': float(np.linalg.norm(implied[0] - implied[1]) * 1000.0),
            'sensors': (parent_sensor, child_sensor),
            'primary': joint in spec.primary_joints,
        })
    return points


def draw_pose(ax: plt.Axes, plates: Dict[str, PlateTrial], joints: List[dict], frame: int,
              spec, scale: float, label_sensors: bool) -> None:
    """One panel: every sensor as an oriented prism, every joint centre as a dot, lever arms as
    thin lines between them."""
    sensor_segment = {sensor: segment for segment, sensor in spec.segment_sensor.items()}

    for entry in joints:
        target = entry['position']
        for sensor in entry['sensors']:
            if sensor not in plates:
                continue
            start = plates[sensor].world_trace.positions[frame]
            line = to_plot_frame(np.array([start, target]))
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=ARM_COLOR, linewidth=0.8,
                    alpha=0.7, zorder=1)

    for sensor, plate in plates.items():
        world = plate.world_trace
        faces = sensor_prism(world.positions[frame], world.rotations[frame], scale=scale)
        collection = Poly3DCollection([to_plot_frame(face) for face in faces],
                                      facecolor=SENSOR_COLOR, edgecolor=SENSOR_EDGE,
                                      linewidths=0.4, alpha=0.95)
        collection.set_zorder(3)
        ax.add_collection3d(collection)
        # One side only. Fifteen labels on a 2D projection of a standing subject collide no
        # matter how they are placed, and the left limb mirrors the right, so labelling the right
        # and the midline says everything the left labels would have.
        if label_sensors and ' L' not in f" {sensor_segment.get(sensor, sensor)}":
            spot = to_plot_frame(world.positions[frame])[0]
            ax.text(spot[0], spot[1], spot[2], f"   {sensor_segment.get(sensor, sensor)}",
                    fontsize=5.5, color='#333333', zorder=5, ha='left', va='center')

    for primary in (False, True):
        selected = [j for j in joints if j['primary'] == primary]
        if not selected:
            continue
        spots = to_plot_frame(np.array([j['position'] for j in selected]))
        ax.scatter(spots[:, 0], spots[:, 1], spots[:, 2],
                   color=JOINT_COLOR if primary else VARIANT_COLOR,
                   s=34 if primary else 16, depthshade=False,
                   edgecolor='white' if primary else 'none', linewidth=0.5, zorder=4)


def equalize(ax: plt.Axes, points: np.ndarray, pad: float = 0.22) -> None:
    """A cubic bounding box, so a metre along x is a metre along y.

    Not cosmetic here: the figure's whole job is to let a lever arm be judged by eye against a
    sensor, and matplotlib's default per-axis autoscale silently stretches one of them.
    """
    plotted = to_plot_frame(points)
    centre = 0.5 * (plotted.max(axis=0) + plotted.min(axis=0))
    span = float((plotted.max(axis=0) - plotted.min(axis=0)).max()) * (1.0 + pad)
    span = max(span, 1e-3)
    ax.set_xlim(centre[0] - span / 2, centre[0] + span / 2)
    ax.set_ylim(centre[1] - span / 2, centre[1] + span / 2)
    ax.set_zlim(centre[2] - span / 2, centre[2] + span / 2)
    ax.set_box_aspect((1, 1, 1))


def pick_trial(dataset: str, subject: str, trials: Sequence[str], spec) -> Optional[str]:
    """The subject's trial that shows the MOST SENSORS, tie-broken by fully-valid frames.

    Sensor count first, and that ordering is the whole point of this function. Ranking on
    fully-valid frames alone rewards a trial for having FEWER sensors: "every plate valid" is a
    conjunction, so dropping a marker group makes it easier to satisfy. On IMoVE that picked
    t3_treadmill_running — a take that drops whole marker groups and carries 8 of the 15 sensors
    — over t1_walking, which has all of them. The figure came out with no left leg and nothing
    said so.

    Chosen from the data rather than named because the two datasets share no trial naming.
    """
    known = set(spec.segment_sensor.values())
    best, best_score = None, (0, 0)
    for trial in trials:
        try:
            plates = {s: p for s, p in load_trial(subject, trial, dataset=dataset).items()
                      if s in known}
        except Exception:
            continue
        if not plates:
            continue
        n = min(len(p) for p in plates.values())
        valid = np.ones(n, dtype=bool)
        for plate in plates.values():
            valid &= np.asarray(plate.valid)[:n]
        score = (len(plates), int(valid.sum()))
        if score > best_score:
            best, best_score = trial, score
    return best


def plot_subject(dataset: str, subject: str, trial: str, scale: float = 1.0,
                 save: bool = True, show: bool = False) -> None:
    spec = get_dataset(dataset)
    plates = {sensor: plate for sensor, plate in load_trial(subject, trial, dataset=dataset).items()
              if sensor in set(spec.segment_sensor.values())}
    if not plates:
        print(f"{dataset}/{subject}/{trial}: no spec sensors present; skipping.")
        return
    frame = pick_frame(plates)
    if frame is None:
        print(f"{dataset}/{subject}/{trial}: no frame has every sensor valid; skipping.")
        return
    offsets = joint_offsets(plates, spec)
    joints = joint_center_points(plates, offsets, frame, spec)
    everything = np.array([plate.world_trace.positions[frame] for plate in plates.values()]
                          + [j['position'] for j in joints])

    fig = plt.figure(figsize=(5.2 * len(VIEWS), 6.0))
    for index, (elevation, azimuth, name, depth) in enumerate(VIEWS):
        ax = fig.add_subplot(1, len(VIEWS), index + 1, projection='3d')
        draw_pose(ax, plates, joints, frame, spec, scale, label_sensors=(index == len(VIEWS) - 1))
        equalize(ax, everything)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_title(name, fontsize=12, fontweight='bold', pad=0)
        ax.set_xlabel('' if depth == 'x' else 'x (m)', fontsize=8, labelpad=-6)
        ax.set_ylabel('' if depth == 'y' else 'z (m)', fontsize=8, labelpad=-6)
        ax.set_zlabel('y, up (m)', fontsize=8, labelpad=-6)
        if depth == 'x':
            ax.set_xticklabels([])
        elif depth == 'y':
            ax.set_yticklabels([])
        ax.tick_params(labelsize=6, pad=-2)
        ax.grid(False)

    separations = [j['separation_mm'] for j in joints]
    subtitle = (f"{len(plates)} sensors, {len(joints)} fitted joint centres; "
                f"parent/child disagreement {np.median(separations):.1f} mm median"
                if separations else f"{len(plates)} sensors, no fitted joint centres")
    scale_note = ("sensors at true size (47x30x13 mm)" if scale == 1.0
                  else f"sensors drawn {scale:g}x true size")
    plot_utils.finalize_and_save_plot(
        fig, f"Sensor poses and fitted joint centres — {dataset} {subject}/{trial}\n{subtitle}",
        f"{dataset}/pose_{subject}_{trial}.png", PLOTS_DIR,
        epilog=f"{scale_note}; prisms oriented by the plate's world-from-body rotation at the "
               f"stillest fully-valid frame ({frame}); grey lines are the lever arms the "
               f"projection moves along; right-side sensors labelled, left mirrors them",
        save=save, show=show)


# ==============================================================================
# Through a trial
# ==============================================================================

def pose_sequence_frames(plates: Dict[str, PlateTrial], count: int) -> List[int]:
    """`count` fully-valid frames spread evenly across the longest valid run."""
    n = min(len(plate) for plate in plates.values())
    valid = np.ones(n, dtype=bool)
    for plate in plates.values():
        valid &= np.asarray(plate.valid)[:n]
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return []
    picks = np.linspace(0, len(indices) - 1, count).round().astype(int)
    return [int(indices[i]) for i in picks]


def plot_pose_sequence(dataset: str, subject: str, trial: str, count: int = 5,
                       scale: float = DEFAULT_SENSOR_SCALE, save: bool = True,
                       show: bool = False) -> None:
    """The same subject at several instants across one trial, joint centres in red.

    The single-pose figure shows the geometry at one moment and cannot distinguish a joint
    centre that tracks the limb from one that happens to be in the right place once. This can:
    the offsets are constant in each segment's own frame, so if they are right the red dots stay
    at the joint through every pose, and if they are wrong they drift out of the limb as it
    rotates — which is exactly the failure a small residual does not rule out.

    Each panel is recentred on its own pose, so the subject does not shrink into a corner as they
    walk across the room; the axes are equal-scaled within a panel, so lengths remain comparable.
    """
    spec = get_dataset(dataset)
    plates = {sensor: plate for sensor, plate in load_trial(subject, trial, dataset=dataset).items()
              if sensor in set(spec.segment_sensor.values())}
    if not plates:
        print(f"{dataset}/{subject}/{trial}: no spec sensors; skipping.")
        return
    frames = pose_sequence_frames(plates, count)
    if not frames:
        print(f"{dataset}/{subject}/{trial}: no fully-valid frame; skipping.")
        return
    offsets = joint_offsets(plates, spec)
    timestamps = next(iter(plates.values())).imu_trace.timestamps
    fig = plt.figure(figsize=(3.6 * len(frames), 5.4))
    for index, frame in enumerate(frames):
        ax = fig.add_subplot(1, len(frames), index + 1, projection='3d')
        joints = joint_center_points(plates, offsets, frame, spec)
        draw_pose(ax, plates, joints, frame, spec, scale, label_sensors=False)
        here = np.array([plate.world_trace.positions[frame] for plate in plates.values()]
                        + [j['position'] for j in joints])
        equalize(ax, here)
        ax.view_init(elev=14, azim=-60)
        ax.set_title(f"t = {timestamps[frame] - timestamps[frames[0]]:.1f} s",
                     fontsize=11, fontweight='bold', pad=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(labelsize=6, pad=-2)
        ax.set_zlabel('y, up (m)', fontsize=7, labelpad=-6)
        ax.grid(False)

    plot_utils.finalize_and_save_plot(
        fig, f"Joint centres through a trial — {dataset} {subject}/{trial}",
        f"{dataset}/sequence_{subject}_{trial}.png", PLOTS_DIR,
        epilog=f"each panel recentred on its own pose; sensors drawn {scale:g}x true size; the "
               f"offsets are the SAME constants in every panel, so the dots staying at the "
               f"joints is the check",
        save=save, show=show)


def center_in_parent_frame(parent: PlateTrial, child: PlateTrial, offsets: Dict[str, np.ndarray],
                           mask: np.ndarray) -> np.ndarray:
    """The CHILD-implied joint centre expressed in the PARENT's own frame, per sample, in mm.

    The sharpest view of what the fit residual actually is. Writing the constraint out,

        q(t) = R_parent(t)^T [ p_child(t) + R_child(t) r_child - p_parent(t) ]

    is where the child segment says the joint is, seen from the parent. If a fixed-centre ball
    joint described the pair, q(t) would be the constant r_parent and the cloud would be a point.
    It is not, and the SHAPE of the cloud says why: an elongated cloud along the segment's long
    axis is the joint centre sliding as the limb flexes (a knee does this), a fat isotropic cloud
    is reconstruction noise, and a cloud that is offset from r_parent rather than centred on it
    means the least squares traded the two segments off against each other.

    Expressed in the parent's frame rather than the world's on purpose: in the world frame every
    joint centre moves several metres as the subject walks and the millimetres of interest are
    invisible.
    """
    n = min(len(parent), len(child))
    mask = np.asarray(mask)[:n]
    separation = (child.world_trace.positions[:n]
                  + np.einsum('nij,j->ni', child.world_trace.rotations[:n], offsets['child'])
                  - parent.world_trace.positions[:n])
    return 1000.0 * np.einsum('nji,nj->ni', parent.world_trace.rotations[:n], separation)[mask]


def plot_center_wander(dataset: str, subject: str, trials, stride: int = 20,
                       save: bool = True, show: bool = False) -> None:
    """Per joint, how far the joint centre wanders from being a fixed point — for one trial, or
    several overlaid on shared axes.

    One row per joint, three orthogonal projections of the cloud in the parent sensor's frame,
    with each trial's fitted offset marked by a cross. A perfect ball joint would put every
    sample on the cross.

    With two trials overlaid this becomes the diagnostic for the cross-validation result in the
    joint_center report — it shows WHY their offsets disagree rather than only that they do.
    Three readings the numbers cannot give:

      * TWO CLOUDS THAT OVERLAP with crosses in different places means the offsets differ by less
        than the spread within either trial, i.e. the disagreement is fit noise.
      * TWO CLOUDS IN DIFFERENT PLACES means the child segment genuinely puts the joint somewhere
        else in the two trials, and no amount of pooling will reconcile them.
      * ONE CLOUD SITTING INSIDE THE OTHER — typically a walking cloud landing on one lobe of a
        bimodal complexTasks cloud — means the trials sample different parts of one posture-
        dependent joint, which is the case the lumbar shows.

    Each trial's cloud uses ITS OWN fitted child offset, so what is plotted is each trial's own
    best answer to "where is the joint". That does mean the comparison includes the child
    offset's disagreement as well as the parent's, which is correct here: the question is whether
    the two fits agree, not whether one component of them does.
    """
    trials = [trials] if isinstance(trials, str) else list(trials)
    spec = get_dataset(dataset)
    known = set(spec.segment_sensor.values())
    plates, offsets = {}, {}
    for trial in trials:
        # Skip rather than abort. A session's trial list can contain one that will not load —
        # IMoVE's static poses can never be built, since the sync step needs motion to
        # cross-correlate — and losing eleven good activities to one bad one is the wrong
        # trade when the figure's whole point is to compare across them.
        try:
            plates[trial] = {s: p for s, p in load_trial(subject, trial, dataset=dataset).items()
                             if s in known}
            offsets[trial] = joint_offsets(plates[trial], spec)
        except Exception as e:
            print(f"  skipping {trial}: {type(e).__name__}")
            plates.pop(trial, None)
            offsets.pop(trial, None)
    trials = [t for t in trials if t in offsets and plates.get(t)]
    if len(trials) < 1:
        print(f"{dataset}/{subject}: no trial loaded; skipping.")
        return

    def has(trial: str, joint: str) -> bool:
        return (joint in offsets[trial]
                and all(s in plates[trial] for s in spec.joints[joint]))

    # Fitted in at least TWO of the requested trials, not in all of them. IMoVE trials drop whole
    # marker groups — a treadmill take can be missing a thigh for its whole duration — so
    # requiring every trial intersects a twelve-trial session down to almost nothing.
    joints = [j for j in spec.primary_joints if sum(has(t, j) for t in trials) >= 2]
    if not joints:
        print(f"{dataset}/{subject}: no joint is fitted in two or more of the requested trials.")
        return
    colours = trial_palette(trials)

    panels = ((0, 1, 'x', 'y'), (0, 2, 'x', 'z'), (1, 2, 'y', 'z'))
    fig, axes = plt.subplots(len(joints), len(panels),
                             figsize=(3.3 * len(panels), 2.6 * len(joints)), squeeze=False)
    for row, joint in enumerate(joints):
        parent_sensor, child_sensor = spec.joints[joint]
        label_parts, crosses = [], []
        for index, trial in enumerate(trials):
            if not has(trial, joint):
                continue
            parent, child = plates[trial][parent_sensor], plates[trial][child_sensor]
            n = min(len(parent), len(child))
            valid = np.asarray(parent.valid)[:n] & np.asarray(child.valid)[:n]
            indices = np.flatnonzero(valid)[::stride]
            if len(indices) > MAX_WANDER_POINTS:
                # Evenly across the trial, not the first N: a long walk subsampled from its head
                # would show one end of the room.
                indices = indices[np.linspace(0, len(indices) - 1,
                                              MAX_WANDER_POINTS).round().astype(int)]
            keep = np.zeros(n, dtype=bool)
            keep[indices] = True
            if not keep.any():
                continue
            cloud = center_in_parent_frame(parent, child, offsets[trial][joint], keep)
            fitted = 1000.0 * offsets[trial][joint]['parent']
            crosses.append(fitted)
            spread = float(np.sqrt(np.mean(np.sum((cloud - fitted) ** 2, axis=1))))
            if len(trials) <= 3:
                label_parts.append(f"{activity_label(trial)}: {spread:.1f} mm")
            colour = colours[index]
            for column, (i, j, _, _) in enumerate(panels):
                ax = axes[row][column]
                ax.scatter(cloud[:, i], cloud[:, j], s=3 if len(trials) > 3 else 2,
                           alpha=0.45 if len(trials) > 3 else 0.16, color=colour, linewidths=0)
                ax.plot(fitted[i], fitted[j], marker='+', color=colour, markersize=13,
                        markeredgewidth=2.4, zorder=6)
                ax.plot(fitted[i], fitted[j], marker='+', color='black', markersize=13,
                        markeredgewidth=0.8, zorder=7)

        # How far apart the two fits put the joint, in the same units as the clouds around them.
        # This is the number the cross-validation is really about, printed where the eye can
        # compare it against the spread of the clouds it sits in.
        crosses = np.array(crosses)
        if len(crosses) == 2:
            gap = float(np.linalg.norm(crosses[0] - crosses[1]))
        elif len(crosses) > 2:
            # With many trials the pairwise gap is a matrix; the useful scalar is how far the
            # per-trial fits scatter around their own centre.
            gap = float(np.sqrt(np.mean(np.sum((crosses - crosses.mean(0)) ** 2, axis=1))))
        else:
            gap = None
        for column, (i, j, name_i, name_j) in enumerate(panels):
            ax = axes[row][column]
            ax.set_xlabel(f"{name_i} (mm)", fontsize=8)
            ax.set_ylabel(f"{name_j} (mm)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_aspect('equal', adjustable='datalim')
        caption = f"{joint}\n" + "\n".join(label_parts)
        if gap is not None:
            caption += (f"\ncrosses {gap:.0f} mm apart" if len(crosses) == 2
                        else f"\n{len(crosses)} fits, {gap:.0f} mm RMS scatter")
        axes[row][0].text(-0.40, 0.5, caption, transform=axes[row][0].transAxes,
                          fontsize=8.5, fontweight='bold', ha='center', va='center', rotation=90)

    if len(trials) > 1:
        handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=7,
                          markerfacecolor=colours[i], markeredgecolor='none',
                          label=activity_label(trial))
                   for i, trial in enumerate(trials)]
        # Upper LEFT, below the title band. loc='upper right' put it straight through the
        # two-line suptitle, which finalize_and_save_plot places at y=1.02.
        fig.legend(handles=handles, loc='upper left', ncol=1 if len(trials) <= 6 else 2,
                   fontsize=9 if len(trials) > 6 else 11,
                   bbox_to_anchor=(0.015, 0.995), frameon=True, framealpha=0.9,
                   title='activity', title_fontsize=10)

    name = "_vs_".join(trials) if len(trials) <= 3 else f"{len(trials)}activities"
    subtitle = ("child-implied centre in the parent sensor's frame; crosses are each trial's "
                "fitted offset")
    plot_utils.finalize_and_save_plot(
        fig, f"Is the joint centre a fixed point? — {dataset} {subject}\n{subtitle}",
        f"{dataset}/wander_{subject}_{name}.png", PLOTS_DIR,
        epilog=f"every {stride}th valid sample, TRUE SCALE; axes equal-scaled so cloud SHAPE is "
               f"readable — elongation along a segment axis is the centre sliding, an isotropic "
               f"blob is reconstruction noise",
        save=save, show=show)


# ==============================================================================
# Comparing two trials' offsets
# ==============================================================================

COMPARE_COLORS = ('#c0243a', '#2f6fb5')   # trial A red, trial B blue
# Beyond two trials the red/blue pair runs out and the question changes: with a dozen tasks in
# one session the point is which ACTIVITIES cluster together, so the palette has to be
# categorical and distinguishable at low alpha. tab20 is the widest matplotlib ships that stays
# legible; past 20 trials the colours repeat and the legend says so.
MAX_WANDER_POINTS = 400   # per trial per joint; a long walk would otherwise bury ten short tasks


def trial_palette(trials: Sequence[str]) -> List[tuple]:
    if len(trials) <= len(COMPARE_COLORS):
        return [mcolors.to_rgba(c) for c in COMPARE_COLORS[:len(trials)]]
    cmap = plt.get_cmap('tab20')
    return [cmap(i % 20) for i in range(len(trials))]


def activity_label(trial: str) -> str:
    """'t2_treadmill_walking_001' -> 'treadmill_walking'. The leading index and the trailing
    take number carry no information once the trials are colour-coded, and spelling them out
    makes a twelve-entry legend unreadable."""
    return re.sub(r'^t\d+_', '', re.sub(r'_\d+$', '', trial)) or trial
# TRUE SCALE by default. The disagreement between two trials' offsets is 11-51 mm against a
# 1.8 m subject, so the two sets of dots very nearly coincide in a whole-body view — and that is
# the honest impression: relative to the body the offsets barely move, even though relative to
# the fit residual they move a lot. An earlier version drew this magnified, mode-shape style,
# which made the gap visible at the cost of a second thing for the reader to hold in mind.
# --exaggeration brings that back when the DIRECTION of the disagreement is what matters; the
# per-joint zoom panels show the same gap at true scale instead.
COMPARE_EXAGGERATION = 1.0


def centers_from(plates: Dict[str, PlateTrial], offsets: Dict[str, Dict[str, np.ndarray]],
                 frame: int, spec, joints: Sequence[str]) -> Dict[str, np.ndarray]:
    """{joint: world position} for `joints` at `frame`, using the supplied offsets.

    Split out from `joint_center_points` because the whole trick of the comparison figure is to
    evaluate SOMEONE ELSE'S offsets on this pose — the plates and the frame come from one trial
    and the offsets from another, which that function's signature does not invite.
    """
    out = {}
    for joint in joints:
        parent_sensor, child_sensor = spec.joints[joint]
        if parent_sensor not in plates or child_sensor not in plates or joint not in offsets:
            continue
        implied = []
        for sensor, key in ((parent_sensor, 'parent'), (child_sensor, 'child')):
            world = plates[sensor].world_trace
            implied.append(world.positions[frame] + world.rotations[frame] @ offsets[joint][key])
        out[joint] = 0.5 * (implied[0] + implied[1])
    return out


def plot_offset_comparison(dataset: str, subject: str, trial_a: str, trial_b: str,
                           scale: float = DEFAULT_SENSOR_SCALE,
                           exaggeration: float = COMPARE_EXAGGERATION,
                           save: bool = True, show: bool = False) -> None:
    """Two trials' offsets, drawn against BOTH trials' poses, plus the per-joint separation.

    The comparison that the cross-validation in the joint_center report quantifies. Each 3D panel
    shows one trial's pose with BOTH sets of joint centres on it — the trial's own in red and the
    other trial's in blue — so the gap between the dots is the disagreement and nothing else. It
    has to be done this way round: the two trials have different postures, so drawing each trial's
    centres on its own pose would confound the offset difference with the pose difference, which
    is the mistake this figure exists to avoid.

    Because the offsets are constants in each segment's frame, the MAGNITUDE of the gap is the
    same at every pose — a rotation preserves length — while its direction in the room rotates
    with the segment. That is why both panels show the same bar chart's worth of disagreement
    from different angles, and it is a useful check: a gap that changes size between the panels
    would mean something other than the offsets differs.
    """
    spec = get_dataset(dataset)
    known = set(spec.segment_sensor.values())
    plates, frames, offsets = {}, {}, {}
    for trial in (trial_a, trial_b):
        selected = {s: p for s, p in load_trial(subject, trial, dataset=dataset).items()
                    if s in known}
        frame = pick_frame(selected) if selected else None
        if frame is None:
            print(f"{dataset}/{subject}/{trial}: no usable frame; skipping comparison.")
            return
        offsets[trial] = joint_offsets(selected, spec)
        plates[trial], frames[trial] = selected, frame

    shared = [j for j in spec.primary_joints
              if j in offsets[trial_a] and j in offsets[trial_b]
              and all(s in plates[trial_a] and s in plates[trial_b] for s in spec.joints[j])]
    if not shared:
        print(f"{dataset}/{subject}: no joint is fitted in both trials; skipping comparison.")
        return

    # The offset difference itself, independent of any pose.
    separation = {j: (np.linalg.norm(offsets[trial_a][j]['parent'] - offsets[trial_b][j]['parent']),
                      np.linalg.norm(offsets[trial_a][j]['child'] - offsets[trial_b][j]['child']))
                  for j in shared}

    fig = plt.figure(figsize=(15.5, 6.4))
    for index, trial in enumerate((trial_a, trial_b)):
        ax = fig.add_subplot(1, 3, index + 1, projection='3d')
        here, frame = plates[trial], frames[trial]
        own = joint_center_points(here, offsets[trial], frame, spec)
        draw_pose(ax, here, [j for j in own if j['joint'] in shared], frame, spec, scale,
                  label_sensors=False)
        other = trial_b if trial == trial_a else trial_a
        alt = centers_from(here, offsets[other], frame, spec, shared)
        mine = centers_from(here, offsets[trial], frame, spec, shared)
        shown = {j: mine[j] + exaggeration * (alt[j] - mine[j]) for j in shared}
        for joint in shared:
            gap = to_plot_frame(np.array([mine[joint], shown[joint]]))
            ax.plot(gap[:, 0], gap[:, 1], gap[:, 2], color='#222222', linewidth=1.6, zorder=6)
        spots = to_plot_frame(np.array([shown[j] for j in shared]))
        ax.scatter(spots[:, 0], spots[:, 1], spots[:, 2], color=COMPARE_COLORS[1 - index],
                   s=34, depthshade=False, edgecolor='white', linewidth=0.5, zorder=7)

        everything = np.array([p.world_trace.positions[frame] for p in here.values()]
                              + [mine[j] for j in shared] + [shown[j] for j in shared])
        equalize(ax, everything)
        ax.view_init(elev=16, azim=-58)
        ax.set_title(f"{trial} pose", fontsize=12, fontweight='bold', pad=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zlabel('y, up (m)', fontsize=7, labelpad=-6)
        ax.tick_params(labelsize=6, pad=-2)
        ax.grid(False)

    ax = fig.add_subplot(1, 3, 3)
    positions = np.arange(len(shared))
    parent_mm = [1000 * separation[j][0] for j in shared]
    child_mm = [1000 * separation[j][1] for j in shared]
    ax.barh(positions + 0.18, parent_mm, height=0.34, color='#7b4fa8', label='parent offset')
    ax.barh(positions - 0.18, child_mm, height=0.34, color='#c9721f', label='child offset')
    ax.set_yticks(positions)
    ax.set_yticklabels(shared)
    ax.invert_yaxis()
    ax.set_xlabel('|offset difference| between the two trials (mm)')
    ax.grid(axis='y', visible=False)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_title('How far the offsets moved', fontsize=12, fontweight='bold')

    handles = [Line2D([0], [0], marker='o', linestyle='none', markersize=8,
                      markerfacecolor=COMPARE_COLORS[i], markeredgecolor='white',
                      label=f"from {trial}")
               for i, trial in enumerate((trial_a, trial_b))]
    fig.axes[0].legend(handles=handles, loc='upper left', fontsize=9,
                       bbox_to_anchor=(-0.02, 0.98), title='joint centres', title_fontsize=9)

    worst = max(max(parent_mm), max(child_mm))
    plot_utils.finalize_and_save_plot(
        fig, f"Joint centres from two trials' offsets — {dataset} {subject}\n"
             f"{trial_a} vs {trial_b}; largest disagreement {worst:.0f} mm",
        f"{dataset}/compare_{subject}_{trial_a}_vs_{trial_b}.png", PLOTS_DIR,
        epilog=("both sets of offsets evaluated on the SAME pose in each panel, so the "
                "connector is the offset difference and not a difference in posture"
                + ("; drawn at TRUE SCALE, which is why the two sets nearly coincide at body size"
                   if exaggeration == 1.0
                   else f"; displacement drawn {exaggeration:g}x exaggerated, bar chart is true")),
        save=save, show=show)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='alborno', choices=sorted(DATASETS))
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--trial', default=None,
                        help="Trial to draw for every subject (default: whichever of that "
                             "subject's trials has the most fully-valid frames).")
    parser.add_argument('--sensor-scale', type=float, default=DEFAULT_SENSOR_SCALE,
                        help="Multiply the drawn sensor size. 1.0 is true size; the default "
                             "exaggerates so the prism's ORIENTATION is legible, which is the "
                             "one thing a dot could not show.")
    parser.add_argument('--figures', nargs='+',
                        choices=['pose', 'sequence', 'wander', 'compare'],
                        default=['pose'],
                        help="pose: the three-view static figure. sequence: several instants "
                             "across the trial, which is what shows the centres TRACKING. "
                             "wander: per joint, how far the centre is from a fixed point. "
                             "compare: two trials' offsets on each other's poses.")
    parser.add_argument('--compare-trials', nargs=2, metavar=('A', 'B'), default=None,
                        help="Trial pair for --figures compare (default: the subject's first two).")
    parser.add_argument('--exaggeration', type=float, default=COMPARE_EXAGGERATION,
                        help="Magnify the offset difference in the 3D panels of --figures "
                             "compare. 1.0 (default) is true scale, where the two sets of dots "
                             "nearly coincide; raise it to see which DIRECTION the offsets "
                             "moved.")
    parser.add_argument('--sequence-frames', type=int, default=5)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    row_keys = enumerate_trials(args.dataset)
    subjects = args.subjects or subjects_of(row_keys)
    for subject in subjects:
        trials = [t for s, t in row_keys if s == subject]
        if args.trial:
            if args.trial not in trials:
                print(f"{subject}: no trial {args.trial}; skipping.")
                continue
            trial = args.trial
        else:
            trial = pick_trial(args.dataset, subject, trials, get_dataset(args.dataset))
        if trial is None:
            print(f"{subject}: no usable trial; skipping.")
            continue
        if 'pose' in args.figures:
            plot_subject(args.dataset, subject, trial, scale=args.sensor_scale, show=args.show)
        if 'sequence' in args.figures:
            plot_pose_sequence(args.dataset, subject, trial, count=args.sequence_frames,
                               scale=args.sensor_scale, show=args.show)
        if 'wander' in args.figures:
            # Overlay the compare pair when one was asked for, so `--figures wander compare`
            # gives two views of the same two trials rather than of different ones.
            wander_trials = (list(args.compare_trials)
                             if args.compare_trials and all(x in trials for x in args.compare_trials)
                             else [trial])
            plot_center_wander(args.dataset, subject, wander_trials, show=args.show)
        if 'compare' in args.figures:
            pair = tuple(args.compare_trials) if args.compare_trials else tuple(trials[:2])
            if len(pair) < 2 or any(x not in trials for x in pair):
                print(f"{subject}: need two of its own trials to compare; has {trials}.")
            else:
                plot_offset_comparison(args.dataset, subject, pair[0], pair[1],
                                       scale=args.sensor_scale,
                                       exaggeration=args.exaggeration, show=args.show)


if __name__ == '__main__':
    main()
