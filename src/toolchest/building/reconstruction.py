"""
Marker plates in, rigid-body poses out.

Everything here used to sit at module level in WorldTrace.py, which put ~700 lines of
reconstruction physics in the same file as the data class it produces. Moved so the trace
classes hold data and operations on that data, and nothing else; construction lives on the
build side, where the rest of the codebase never imports it.

Two detectors run, and neither subsumes the other — see `reconstruct_plate` for the full
argument:

    template residual    wrong SHAPE at one instant     a displaced or merged marker
    angular speed        wrong CHANGE between instants  a pose that jumps

`reconstruct_plate` is the entry point. `_reconstruct_from_markers` +
`repair_reconstruction_glitches` are the older rectangle-specific path, kept so the two can
be diffed on the same data.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from typing import Dict, List, Tuple

from ..WorldTrace import WorldTrace

def _compute_case(m_o, m_d, m_x, m_y, w, h, faulty_idx=None):
    """Computes coordinate frame orientation/location for a specific fault case."""
    if faulty_idx == 0:  # O is faulty
        x_v = (m_x - m_d) / np.linalg.norm(m_x - m_d, axis=1)[:, None]
        yt = (m_y - m_d) / np.linalg.norm(m_y - m_d, axis=1)[:, None]
        z_v = np.cross(x_v, yt) / np.linalg.norm(np.cross(x_v, yt), axis=1)[:, None]
        y_v = np.cross(z_v, x_v)
        o_est = m_d + w * x_v + h * y_v
        loc = (o_est + m_d + m_x + m_y) / 4.0
    elif faulty_idx == 1:  # D is faulty
        x_v = (m_o - m_y) / np.linalg.norm(m_o - m_y, axis=1)[:, None]
        yt = (m_o - m_x) / np.linalg.norm(m_o - m_x, axis=1)[:, None]
        z_v = np.cross(x_v, yt) / np.linalg.norm(np.cross(x_v, yt), axis=1)[:, None]
        y_v = np.cross(z_v, x_v)
        d_est = m_o - w * x_v - h * y_v
        loc = (m_o + d_est + m_x + m_y) / 4.0
    elif faulty_idx == 2:  # X is faulty
        x_v = (m_o - m_y) / np.linalg.norm(m_o - m_y, axis=1)[:, None]
        yt = (m_y - m_d) / np.linalg.norm(m_y - m_d, axis=1)[:, None]
        z_v = np.cross(x_v, yt) / np.linalg.norm(np.cross(x_v, yt), axis=1)[:, None]
        y_v = np.cross(z_v, x_v)
        x_est = m_y + w * x_v - h * y_v
        loc = (m_o + m_d + x_est + m_y) / 4.0
    elif faulty_idx == 3:  # Y is faulty
        x_v = (m_x - m_d) / np.linalg.norm(m_x - m_d, axis=1)[:, None]
        yt = (m_o - m_x) / np.linalg.norm(m_o - m_x, axis=1)[:, None]
        z_v = np.cross(x_v, yt) / np.linalg.norm(np.cross(x_v, yt), axis=1)[:, None]
        y_v = np.cross(z_v, x_v)
        y_est = m_x - w * x_v + h * y_v
        loc = (m_o + m_d + m_x + y_est) / 4.0
    else:  # Nominal
        x1 = (m_x - m_d) / np.linalg.norm(m_x - m_d, axis=1)[:, None]
        x2 = (m_o - m_y) / np.linalg.norm(m_o - m_y, axis=1)[:, None]
        x_v = (x1 + x2) / 2.0
        x_v /= np.linalg.norm(x_v, axis=1)[:, None]
        
        y1 = (m_o - m_x) / np.linalg.norm(m_o - m_x, axis=1)[:, None]
        y2 = (m_y - m_d) / np.linalg.norm(m_y - m_d, axis=1)[:, None]
        yt = (y1 + y2) / 2.0
        yt /= np.linalg.norm(yt, axis=1)[:, None]
        
        z_v = np.cross(x_v, yt) / np.linalg.norm(np.cross(x_v, yt), axis=1)[:, None]
        y_v = np.cross(z_v, x_v)
        loc = (m_o + m_d + m_x + m_y) / 4.0

    rot = np.stack((x_v, y_v, z_v), axis=2)
    return loc, rot


# Angular speed above which a reconstructed pose is judged corrupt rather than fast.
# 3000 deg/s is an order of magnitude past any human body segment; for scale, the 99.9th
# percentile of clean plates in this dataset is 500-800 deg/s.
#
# This exists because the distance-based fault isolation in _reconstruct_from_markers
# cannot see a marker SWAP. Swapping two labels leaves every inter-marker distance
# unchanged, so fault_scores stays clean, but the reconstructed frame flips: if x_v -> -x_v
# then z_v = x_v X yt -> -z_v while y_v = z_v X x_v is unchanged, which is exactly a 180 deg
# rotation about y. Subject06's femur_l plate does this in five short bursts.
MAX_RECONSTRUCTION_ANGULAR_SPEED_DEG_S = 3000.0

# Frames of dilation around each detected glitch. A single corrupt frame produces two large
# steps (into it and out of it), so both neighbours are already flagged; this covers the
# partially-corrupted frames on the shoulders of a longer burst.
GLITCH_DILATION_FRAMES = 2


# A marker label swap corrupts the reconstructed frame by a CONSTANT half turn in the
# plate's own frame, and only these four values are reachable. Swapping o<->d and x<->y
# sends x_v -> -x_v, hence z_v = x_v X yt -> -z_v while y_v = z_v X x_v is unchanged, so
# R_bad = R_true @ diag(-1, 1, -1) — a half turn about the plate's y axis, applied on the
# RIGHT because rot is built with the basis vectors as columns. The other swap pairings give
# the x and z half turns. Each is its own inverse, so applying it again undoes it.
_HALF_TURN_CANDIDATES = {
    'none': np.eye(3),
    'x': np.diag([1.0, -1.0, -1.0]),
    'y': np.diag([-1.0, 1.0, -1.0]),
    'z': np.diag([-1.0, -1.0, 1.0]),
}

# How closely a block's boundary discontinuity must match a half turn before it is treated
# as a marker swap. Generous because a few frames of real motion elapse across the gap; the
# discrimination is easy regardless, since the wrong choice leaves ~180 deg and the right one
# leaves a few.
MAX_FLIP_SNAP_RESIDUAL_DEG = 20.0

# Interpolation bridges a discontinuity only if that discontinuity was EXPLAINED — either it
# was small to begin with, or the blocks either side were reconciled by a marker swap. An
# unexplained jump is left exactly as it is.
#
# The reason is that interpolating across one manufactures a smooth ramp of motion that never
# happened and, worse, spreads the jump thin enough that it no longer trips the detector that
# found it. Subject08's calcn_l does this: a 176 deg jump about an axis that is NOT a
# marker-rectangle symmetry (the plates measure 86x104 mm, so only the three coordinate half
# turns are distance-preserving relabelings) became a 30-frame ramp whose every step sat under
# the speed limit.
#
# Note the gate cannot be the jump's SIZE: the leftover transition frames beside a
# successfully un-flipped block are themselves still flipped, so they show ~180 deg steps and
# are nonetheless safe to interpolate, being a handful of frames bracketed by known-good poses.


def _rotation_angle_deg(matrix: np.ndarray) -> float:
    """Geodesic magnitude of a single rotation matrix, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))))


def _step_angles_deg(rotations: np.ndarray) -> np.ndarray:
    """Rotation angle of each of the N-1 frame-to-frame transitions, in degrees.

    Uses the trace identity cos(theta) = (trace(R[t]^T R[t+1]) - 1) / 2 with
    trace(A^T B) = sum(A * B): exact to ~1e-9 deg but ~120x faster than building a Rotation,
    which matters because this runs on every plate of every trial. arccos is imprecise near
    zero, which is irrelevant against thresholds of tens of degrees.
    """
    cos_theta = (np.einsum('tij,tij->t', rotations[:-1], rotations[1:]) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def _offending_steps(rotations: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(mask of transitions exceeding the physical speed limit, their angles in degrees)."""
    step_deg = _step_angles_deg(rotations)
    dt = np.diff(timestamps)
    speed = step_deg / np.where(dt > 0, dt, np.inf)
    return speed > MAX_RECONSTRUCTION_ANGULAR_SPEED_DEG_S, step_deg


def _transition_mask(offending: np.ndarray, n_frames: int) -> np.ndarray:
    """Frames on either side of an offending step, dilated — the untrustworthy transitions."""
    bad = np.zeros(n_frames, dtype=bool)
    bad[:-1] |= offending    # the frame entering the jump
    bad[1:] |= offending     # the frame leaving it
    for _ in range(GLITCH_DILATION_FRAMES):
        # Sliced rather than np.roll: roll wraps, which would let a glitch in the last
        # frame flag the first one.
        dilated = bad.copy()
        dilated[1:] |= bad[:-1]
        dilated[:-1] |= bad[1:]
        bad = dilated
    return bad


def _unflip_swapped_blocks(rotations: np.ndarray, bad: np.ndarray,
                           name: str = None, candidates: dict = None) -> Tuple[np.ndarray, int, int]:
    """Undoes SUSTAINED marker swaps: whole blocks sitting a constant half turn off.

    A swap persists until the labels are corrected, which in this dataset means blocks
    thousands of frames long — Subject06's femur_l holds a 180 deg flip for 4214 frames
    (42 s). Interpolating the boundaries of such a block is worse than doing nothing: it
    silences the detector while leaving the interior corrupt.

    So the trace is split into clean runs at the detected discontinuities, and each run is
    compared with the previous, already-corrected one. The apparent body-frame change across
    the gap is snapped to the nearest of _HALF_TURN_CANDIDATES; if it matches one within
    MAX_FLIP_SNAP_RESIDUAL_DEG, that half turn is applied to the whole run.

    The first run anchors the chain and is never corrected, which costs nothing: a constant
    flip applied to an entire trial is absorbed by the sensor-to-segment alignment in
    assembly.align_world_to_imu. Only flips RELATIVE to the rest of the trial
    corrupt anything.

    Returns (rotations, frames un-flipped, unexplained gaps as (last_good, next_good) index
    pairs). Those gaps are what the caller must refuse to interpolate across.
    """
    # `candidates` are the frame rotations a relabeling can produce on THIS plate. Defaults
    # to the rectangle's three coordinate half turns, which is what the original callers
    # assume; `template_correction_candidates` derives the correct set for any shape and
    # reproduces exactly these four for an 86x104 rectangle.
    if candidates is None:
        candidates = _HALF_TURN_CANDIDATES

    clean_idx = np.flatnonzero(~bad)
    if len(clean_idx) == 0:
        return rotations, 0, []

    runs = np.split(clean_idx, np.flatnonzero(np.diff(clean_idx) > 1) + 1)
    out = np.array(rotations, dtype=np.float64, copy=True)
    n_flipped = 0
    unexplained = []
    previous_end = runs[0][-1]

    for run in runs[1:]:
        discontinuity = out[previous_end].T @ out[run[0]]
        label, correction, residual = min(
            ((label, candidate, _rotation_angle_deg(discontinuity.T @ candidate))
             for label, candidate in candidates.items()),
            key=lambda item: item[2])

        if residual > MAX_FLIP_SNAP_RESIDUAL_DEG:
            # Neither continuous nor a marker swap, so this is a genuine reconstruction
            # failure. Applying a half turn would invent data, and so would interpolating
            # across it, so the gap is handed back to the caller untouched.
            unexplained.append((int(previous_end), int(run[0])))
        elif label != 'none':
            out[run] = out[run] @ correction
            n_flipped += len(run)

        previous_end = run[-1]

    return out, n_flipped, unexplained


def repair_reconstruction_glitches(
    positions: np.ndarray,
    rotations: np.ndarray,
    timestamps: np.ndarray,
    name: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Repairs non-physical marker-plate reconstruction failures.

    Two distinct faults, which need different fixes and are easy to conflate:

      * SUSTAINED SWAPS. Two marker labels are exchanged for a stretch of the trial,
        rotating the reconstructed frame by a constant half turn until the labels recover.
        These blocks run to thousands of frames, so they are un-flipped, not interpolated.
        Invisible to the distance-based fault isolation in _reconstruct_from_markers,
        because a swap leaves every inter-marker distance unchanged.
      * TRANSITION FRAMES. The few frames either side of each discontinuity, where the
        reconstruction is genuinely mid-failure and neither pose is meaningful. These are
        rebuilt from the nearest clean frames: SLERP for rotation, linear for position.

    Order matters — un-flip first, then interpolate — because the blocks are separated by
    exactly the discontinuities that flag the transitions, and interpolating first would
    smear a 180 deg step into the surrounding frames.

    Interpolating rather than deleting keeps the uniform time grid that resampling,
    filtering and every finite-difference angular velocity in this repo assume. Only
    transition frames get their position replaced: a swap leaves the marker centroid
    unchanged, so a flipped block's positions are already correct.

    Returns (positions, rotations, valid, report).

    `report` counts 'flipped', 'interpolated' and 'unresolved' frames. Callers should
    surface a nonzero report — a silent repair would hide a marker-labelling problem that
    is a real property of the .trc.

    `valid` is a per-frame boolean: False wherever this function did not leave a MEASURED
    pose behind. Two cases, and note that they are the two the repair could not fully fix:

      * Interpolated transition frames. SLERP output is plausible, not observed. Using it
        as ground truth scores a filter against a guess.
      * Unresolved gaps. Left deliberately corrupt, because inventing a pose there would
        be worse than admitting the failure.

    Un-flipped frames stay VALID. A swap is a labelling error the repair genuinely undoes;
    the resulting pose is the measured one, correctly labelled.

    The mask is what separates the interval a filter RUNS over from the interval it is
    SCORED over. Interpolation exists to keep the uniform time grid every finite-difference
    in this repo assumes, so those frames must stay in the array — the mask is how they
    stay out of the error statistics.

    WHAT THIS MASK DOES NOT CATCH. It is built from angular SPEED (see
    MAX_RECONSTRUCTION_ANGULAR_SPEED_DEG_S), so it only sees faults that make the pose jump.
    A marker COLLISION — two labels resolving to one detected point — does not: the merged
    point lands near a centroid, roughly 23 mm from the marker's true location, which yields
    a pose that is wrong but perfectly smooth. Subject06/walking pelvis frames 58486-58497
    are such a window, and they peak at 1460 deg/s against this function's 3000 deg/s
    threshold, so nothing here fires and those frames come back marked valid.
    Catching them needs the other detector — marker-to-marker distance violation, as in
    _reconstruct_from_markers' fault isolation — fed in as a second input to this mask. Note
    that detector must be thresholded above the dataset's own noise floor: pairwise deviation
    reaches 14.87 mm at p99.99, so a 10 mm gate flags ~17,600 frames, of which the
    overwhelming majority are single-marker displacements the existing 3-marker path already
    reconstructs correctly (median pose error 1.00 deg).
    """
    empty = {'flipped': 0, 'interpolated': 0, 'unresolved': 0}
    all_valid = np.ones(len(rotations), dtype=bool)
    if len(rotations) < 3:
        return positions, rotations, all_valid, empty

    rotations = np.asarray(rotations, dtype=np.float64)
    offending, _ = _offending_steps(rotations, timestamps)
    if not np.any(offending):
        return positions, rotations, all_valid, empty

    bad = _transition_mask(offending, len(rotations))
    if bad.all():
        print(f"Warning: {name or 'segment'} looks corrupt in every frame; leaving it "
              f"untouched rather than interpolating from nothing.")
        # Every frame is suspect, but the caller gets the data untouched and a mask that
        # says so, rather than a silently empty repair report.
        return positions, rotations, ~bad, empty

    # 1. Un-flip sustained swaps, then re-detect: the surviving discontinuities are the
    #    genuine transitions, and the flips no longer masquerade as them.
    rotations, n_flipped, unexplained = _unflip_swapped_blocks(rotations, bad, name=name)
    offending, _ = _offending_steps(rotations, timestamps)
    bad = _transition_mask(offending, len(rotations))

    # 2. Withhold the unexplained gaps from interpolation, so a genuine reconstruction
    #    failure stays visible instead of being smoothed into plausible-looking invention.
    #    Those frames keep their corrupt poses, so they are the other half of the mask.
    unresolved_frames = np.zeros(len(rotations), dtype=bool)
    for last_good, next_good in unexplained:
        bad[last_good + 1:next_good] = False
        unresolved_frames[last_good + 1:next_good] = True

    # `bad` is now exactly the set that will be interpolated below, so the mask can be
    # built here and is correct on both the early return and the repaired path.
    valid = ~(bad | unresolved_frames)

    report = {'flipped': n_flipped, 'interpolated': int(bad.sum()),
              'unresolved': len(unexplained)}
    if unexplained:
        print(f"Warning: {name or 'segment'}: {len(unexplained)} discontinuity(ies) are "
              f"neither a marker swap nor safely interpolable, and have been LEFT IN PLACE. "
              f"This segment is corrupt around frames "
              f"{[gap[0] for gap in unexplained[:5]]}; treat any joint using it with suspicion.")
    if not np.any(bad):
        return positions, rotations, valid, report

    good_idx = np.flatnonzero(~bad)
    if len(good_idx) < 2:
        print(f"Warning: {name or 'segment'} has too few clean frames to interpolate from.")
        report['interpolated'] = 0
        # Nothing was interpolated, so those frames keep their original poses. They are
        # still transition frames, so they are still not trustworthy ground truth.
        return positions, rotations, valid, report

    # Only here, once a repair is known to be needed, is a Rotation object built — the
    # common clean case above stays pure numpy.
    rot = Rotation.from_matrix(rotations)
    # Clamp so a glitch at either end holds the nearest clean pose instead of failing.
    query = np.clip(timestamps[bad], timestamps[good_idx[0]], timestamps[good_idx[-1]])

    repaired_rotations = rotations.copy()
    repaired_rotations[bad] = Slerp(timestamps[good_idx], rot[good_idx])(query).as_matrix()

    repaired_positions = np.array(positions, dtype=np.float64, copy=True)
    for axis in range(3):
        repaired_positions[bad, axis] = np.interp(
            query, timestamps[good_idx], np.asarray(positions)[good_idx, axis])

    return repaired_positions, repaired_rotations, valid, report


def _reconstruct_from_markers(
    marker_o: np.ndarray,
    marker_d: np.ndarray,
    marker_x: np.ndarray,
    marker_y: np.ndarray,
    threshold: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstructs rigid body coordinate frames with fault isolation."""
    N = len(marker_o)
    pos = [marker_o, marker_d, marker_x, marker_y]
    
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    d = {pair: np.linalg.norm(pos[pair[0]] - pos[pair[1]], axis=1) for pair in pairs}
    bar_d = {pair: np.median(d[pair]) for pair in pairs}
    e = {pair: np.abs(d[pair] - bar_d[pair]) for pair in pairs}
    
    epsilon = 1e-3  # 1mm standard scale
    fault_scores = np.zeros((4, N))
    for i in range(4):
        num_pairs = [p for p in pairs if i in p]
        den_pairs = [p for p in pairs if i not in p]
        numerator = np.sum([e[p] for p in num_pairs], axis=0)
        denominator = np.sum([e[p] for p in den_pairs], axis=0)
        fault_scores[i] = numerator / (denominator + epsilon)
        
    w = (bar_d[(1, 2)] + bar_d[(0, 3)]) / 2.0
    h = (bar_d[(0, 2)] + bar_d[(1, 3)]) / 2.0

    # Evaluate nominal case
    loc_4, rot_4 = _compute_case(marker_o, marker_d, marker_x, marker_y, w, h, faulty_idx=None)
    
    max_idx = np.argmax(fault_scores, axis=0)
    is_faulty = np.max(fault_scores, axis=0) > threshold
    case_indices = np.where(is_faulty, max_idx, 4)

    positions = loc_4.copy()
    rotations = rot_4.copy()

    for c in range(4):
        mask = (case_indices == c)
        if np.any(mask):
            loc_c, rot_c = _compute_case(
                marker_o[mask], marker_d[mask], marker_x[mask], marker_y[mask], w, h, faulty_idx=c
            )
            positions[mask] = loc_c
            rotations[mask] = rot_c

    return positions, rotations


# ==============================================================================
# Template-based plate reconstruction
# ==============================================================================
# The general form of what _reconstruct_from_markers does for the special case of a
# RECTANGULAR four-marker plate. Fits a known rigid marker layout to each frame by
# Kabsch/SVD, which needs no assumption about the plate's shape.
#
# This exists because IMoVE's clusters are not rectangles. Their six inter-marker
# distances are 120.4 / 104.4 / 103.4 / 84.6 / 62.9 / 59.6 mm — measured across all five
# clusters and all 21 subjects with a per-subject sd of 0.2-1.3 mm, so it is one physical
# plate, but an irregular planar quadrilateral. _reconstruct_from_markers assumes
# rectangularity in three places (w and h average supposedly-equal opposite edges, x_v
# averages supposedly-parallel edges, and _compute_case rebuilds a missing marker as
# m_d + w*x_v + h*y_v), so none of it transfers.
#
# The fault handling generalizes too: instead of _compute_case's four hand-written
# branches, drop each marker in turn and keep the subset that fits best. That is the same
# idea, expressed once rather than four times, and it is correct for any plate shape.

# Per-marker fit residual above which a frame's pose is not trusted, in metres.
#
# Scale reference: the template itself is reproducible to 0.2-1.3 mm across subjects, and
# an independent survey of this repo's Al Borno plates put the p99.99 of inter-marker
# distance deviation at 14.87 mm — roughly 7 mm as a per-marker residual. 10 mm therefore
# sits above the noise floor of both datasets while being far below a real fault, which
# runs to tens of mm. It is a starting point, not a tuned value: the honest way to set it
# is from the residual distribution of the data in hand, which `report` returns for exactly
# that purpose.
DEFAULT_PLATE_RESIDUAL_TOLERANCE_M = 0.010

# Markers are reported as exactly zero by some pipelines when a gap is not filled, which is
# a valid coordinate in principle and never one in practice — the lab origin is not on the
# subject. Treated as missing alongside NaN.
_MISSING_MARKER_EPS = 1e-9


def estimate_plate_template(markers: np.ndarray) -> np.ndarray:
    """Recovers a plate's rigid marker layout from the markers' own median geometry.

    Takes (N, M, 3) world-frame marker positions and returns an (M, 3) layout in an
    arbitrary plate-fixed frame, centred on the marker centroid.

    Built from the MEDIAN inter-marker distances, so per-frame faults do not move it: a
    marker has to be displaced in more than half the trial before it shifts the template.
    Classical MDS turns that distance matrix back into coordinates.

    The frame's orientation is arbitrary and deliberately so. Any constant rotation of the
    template rotates every reconstructed pose identically, and that is absorbed downstream
    by assembly.align_world_to_imu, which solves the sensor-to-segment
    rotation from gyros anyway. So the template needs no anatomical justification.
    """
    markers = np.asarray(markers, dtype=np.float64)
    n_markers = markers.shape[1]

    present = _present_mask(markers)
    distances = np.zeros((n_markers, n_markers))
    for i in range(n_markers):
        for j in range(i + 1, n_markers):
            both = present[:, i] & present[:, j]
            if not np.any(both):
                raise ValueError(f"Markers {i} and {j} are never both present; "
                                 f"cannot estimate a template.")
            d = np.linalg.norm(markers[both, i] - markers[both, j], axis=1)
            distances[i, j] = distances[j, i] = np.median(d)

    # Classical MDS: centre the squared-distance matrix, then take the leading eigenvectors.
    centering = np.eye(n_markers) - np.ones((n_markers, n_markers)) / n_markers
    gram = -0.5 * centering @ (distances ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:3]
    # Clipped at zero: a planar plate's third eigenvalue is zero up to measurement noise and
    # can come out slightly negative, which would make the sqrt nan.
    return eigenvectors[:, order] * np.sqrt(np.clip(eigenvalues[order], 0.0, None))


def template_symmetries(template: np.ndarray) -> List[Tuple[Tuple[int, ...], float]]:
    """Marker relabelings a plate's geometry cannot distinguish, worst-first by margin.

    Returns (permutation, largest distance change in mm) for every relabeling, sorted so
    the most nearly invisible non-identity permutation comes first. A permutation with a
    small margin is one that leaves all six inter-marker distances essentially unchanged,
    which makes it invisible to `fit_plate_to_template` — and to any other residual-based
    method, since there is no residual to see.

    This is the question of whether the template fit is SUFFICIENT on its own:

      * Al Borno's plates are 86x104 mm rectangles, so both diagonals are 135 mm and four
        relabelings have a margin of exactly 0. Measured on real data, the template fit
        reproduces a swapped block with a median residual of 0.20 mm — it detects nothing
        — while disagreeing with the un-flipped truth by exactly 180 degrees. So those
        plates REQUIRE `_unflip_swapped_blocks`, and no amount of tolerance tuning helps.
      * IMoVE's cluster has six distinct distances; its best non-identity margin is about
        1 mm, against a measured fit residual of 0.1-0.6 mm. Swaps are visible there, so
        the template fit needs no un-flip pass.

    Returned as margins rather than a boolean because 'invisible' is relative to the
    residual the data actually achieves, which the caller knows and this function does not.
    """
    from itertools import permutations
    template = np.asarray(template, dtype=np.float64)
    n = len(template)
    distances = np.linalg.norm(template[:, None, :] - template[None, :, :], axis=2)

    out = []
    for perm in permutations(range(n)):
        permuted = distances[np.ix_(perm, perm)]
        out.append((perm, float(np.abs(permuted - distances).max() * 1000)))
    identity = tuple(range(n))
    return sorted(out, key=lambda item: (item[0] == identity, item[1]))


# Largest distance change, in mm, at which a relabeling still counts as invisible.
#
# Has to be set against a MEASURED template, not an ideal one. On the Al Borno plates the
# three genuine rectangle symmetries come out at 0.00-0.98 mm — the marker noise — while the
# next permutation is at 17.4-18.2 mm. IMoVE's irregular cluster has its best non-identity
# permutation at 17.0 mm. So the two populations are separated by a factor of ~20, and 5 mm
# sits in the middle of that gap: 5x above the real symmetries, 3.4x below everything else.
#
# Getting this wrong is not symmetric. Too tight and a real symmetry is missed, the un-flip
# has no candidate to snap to, and a swapped block is left corrupt. Too loose and a genuine
# discontinuity gets "corrected" into a plausible-looking pose that never happened.
DEFAULT_SYMMETRY_MARGIN_MM = 5.0


def template_correction_candidates(template: np.ndarray,
                                   max_margin_mm: float = DEFAULT_SYMMETRY_MARGIN_MM
                                   ) -> Dict[str, np.ndarray]:
    """The frame rotations a marker relabeling can produce on this plate, keyed by label.

    A relabeling that preserves every inter-marker distance maps the template onto a rotated
    copy of itself, so it shifts the fitted pose by a constant rotation and leaves the fit
    residual untouched. Those rotations are what `_unflip_swapped_blocks` has to choose from,
    and they are a property of the plate's shape — derivable, not something to hardcode.

    Generalizes `_HALF_TURN_CANDIDATES`, which it reproduces exactly (all four, including
    the identity) for the 86x104 mm Al Borno rectangle. For a plate with no symmetries it
    returns only the identity, which is the correct statement that no relabeling is
    invisible there and none should be attempted.
    """
    template = np.asarray(template, dtype=np.float64)
    template = template - template.mean(axis=0)
    out = {}
    for perm, margin in template_symmetries(template):
        if margin > max_margin_mm:
            continue
        target = template[list(perm)]
        u, _, vt = np.linalg.svd(template.T @ target)
        rotation = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
        out['identity' if perm == tuple(range(len(template))) else str(perm)] = rotation
    return out


def _interpolate_frames(positions: np.ndarray, rotations: np.ndarray,
                        timestamps: np.ndarray, fill: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Replaces `fill` frames by interpolating the surrounding ones — SLERP for rotation,
    linear for position. Clamped, so a gap at either end holds the nearest good pose."""
    good = np.flatnonzero(~fill)
    if len(good) < 2 or not np.any(fill):
        return positions, rotations
    positions, rotations = np.array(positions, copy=True), np.array(rotations, copy=True)
    query = np.clip(timestamps[fill], timestamps[good[0]], timestamps[good[-1]])
    rotations[fill] = Slerp(timestamps[good],
                            Rotation.from_matrix(rotations[good]))(query).as_matrix()
    for axis in range(3):
        positions[fill, axis] = np.interp(query, timestamps[good], positions[good, axis])
    return positions, rotations


def _present_mask(markers: np.ndarray) -> np.ndarray:
    """(N, M) bool: which markers are actually observed in each frame."""
    finite = np.isfinite(markers).all(axis=2)
    nonzero = (np.abs(markers) > _MISSING_MARKER_EPS).any(axis=2)
    return finite & nonzero


def _kabsch(template: np.ndarray, observed: np.ndarray,
            indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-fit rigid transform taking `template` onto `observed`, over a marker subset.

    Returns (rotations (n,3,3), translations (n,3), per-frame RMS residual (n,)) for the
    transform x = R p + t, where p is a template point and x its world position.

    `t` is the world position of the template's ORIGIN, not of the subset centroid, so it
    means the same physical point on the plate no matter which markers the fit used. That
    matters here: leave-one-out changes the subset frame to frame, and a position that
    jumped whenever a marker dropped out would be worse than useless.
    """
    P = template[indices]                                  # (k, 3)
    Q = observed[:, indices, :]                            # (n, k, 3)
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=1)
    Pc, Qc = P - p_mean, Q - q_mean[:, None, :]

    covariance = np.einsum('ki,nkj->nij', Pc, Qc)
    U, _, Vt = np.linalg.svd(covariance)
    # Reflection guard: without it a degenerate frame can produce a det = -1 "rotation",
    # which is a mirror image and silently flips the reconstructed segment.
    signs = np.sign(np.linalg.det(np.einsum('nji,nkj->nik', Vt, U)))
    correction = np.zeros((len(observed), 3, 3))
    correction[:, 0, 0] = correction[:, 1, 1] = 1.0
    correction[:, 2, 2] = signs
    rotations = np.einsum('nji,njk,nlk->nil', Vt, correction, U)

    translations = q_mean - np.einsum('nij,j->ni', rotations, p_mean)
    predicted = np.einsum('nij,kj->nki', rotations, P) + translations[:, None, :]
    residual = np.sqrt((np.linalg.norm(predicted - Q, axis=2) ** 2).mean(axis=1))
    return rotations, translations, residual


def fit_plate_to_template(
    markers: np.ndarray,
    timestamps: np.ndarray,
    template: np.ndarray = None,
    residual_tolerance: float = DEFAULT_PLATE_RESIDUAL_TOLERANCE_M,
    name: str = None,
    interpolate_invalid: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Reconstructs a rigid plate's pose per frame by fitting a known marker layout.

    Args:
        markers: (N, M, 3) world-frame marker positions in METRES. NaN or exactly-zero
            rows are treated as gaps.
        timestamps: (N,) seconds.
        template: (M, 3) plate-fixed layout. Estimated from the data if None.
        residual_tolerance: per-marker fit residual above which a frame is not trusted.
        name: for warnings.

    Returns (positions, rotations, valid, report), matching the shape
    repair_reconstruction_glitches returns so the two are interchangeable to a caller.

    Three things happen, in order:

      1. Fit every marker present in a frame. Frames that fit within tolerance are done.
      2. For frames that do not, and that have four markers, drop each in turn and keep the
         best three. One displaced marker is the overwhelmingly common fault, and this
         isolates it without assuming anything about the plate's shape.
      3. Frames still over tolerance, or with fewer than three markers, get no pose of their
         own: they are interpolated from their neighbours and marked INVALID.

    Step 3 keeps the uniform time grid that resampling, filtering and every finite-
    difference angular velocity in this repo assume, while `valid` keeps the invented poses
    out of any error statistic computed against them.
    """
    markers = np.asarray(markers, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    n_frames, n_markers = markers.shape[0], markers.shape[1]
    if n_markers < 3:
        raise ValueError(f"A plate needs at least 3 markers to define a frame; got {n_markers}.")

    present = _present_mask(markers)

    # A marker that is never seen carries no information and would sink the whole plate:
    # estimate_plate_template needs every pair co-present, and the fit would carry a column
    # of gaps. Drop those and continue on what is left — three markers still determine a
    # pose, just without the redundancy that makes fault isolation possible.
    #
    # IMoVE needs this: the treadmill trials lose whole marker groups (s2's t2 has all four
    # LTH and all four LSH at 0% present), and a reader has to be able to tell "this segment
    # is untracked in this trial" apart from "this segment failed to reconstruct".
    ever_present = present.any(axis=0)
    excluded = [int(i) for i in np.flatnonzero(~ever_present)]
    if excluded and ever_present.sum() < 3:
        raise ValueError(
            f"{name or 'plate'}: only {int(ever_present.sum())} of {n_markers} markers appear "
            f"anywhere in this trial (missing indices {excluded}); a pose needs at least 3.")

    if template is None:
        # Centred on the present markers by construction — nothing else is available. Note
        # the consequence: if a marker is absent for a whole trial, this trial's origin is
        # the centroid of the REMAINING markers, so its positions are not comparable with a
        # trial where all markers were seen. Pass an explicit template to avoid that.
        estimated = estimate_plate_template(markers[:, ever_present])
        template = np.full((n_markers, 3), np.nan)
        # Re-expanded so template rows stay aligned with marker columns. NaN rather than
        # zero for the absent rows: they are never indexed (their `present` is False in
        # every frame), and NaN turns a future indexing mistake into a loud failure instead
        # of a plausible-looking pose fitted against the origin.
        template[ever_present] = estimated
    else:
        template = np.asarray(template, dtype=np.float64)
        # Centred over ALL rows, not just the ones in use. The origin is then the caller's
        # plate origin, which is the same physical point in every trial whether or not a
        # marker went missing in this one — so positions stay comparable across trials.
        template = template - template.mean(axis=0)
    # Zeroed so a NaN in an unused marker cannot poison an einsum over the used ones.
    clean = np.where(present[:, :, None], markers, 0.0)

    positions = np.full((n_frames, 3), np.nan)
    rotations = np.full((n_frames, 3, 3), np.nan)
    residual = np.full(n_frames, np.inf)
    dropped = np.full(n_frames, -1, dtype=int)

    # --- 1. Fit each distinct availability pattern in one vectorized pass ---------------
    # At most 2^M patterns, and in practice one or two, so this is a small loop over
    # groups rather than a per-frame Python loop over tens of thousands of frames.
    patterns = np.unique(present, axis=0)
    for pattern in patterns:
        indices = np.flatnonzero(pattern)
        if len(indices) < 3:
            continue
        frames = np.flatnonzero((present == pattern).all(axis=1))
        R, t, r = _kabsch(template, clean[frames], indices)
        rotations[frames], positions[frames], residual[frames] = R, t, r

    # --- 2. Leave-one-out on the frames that did not fit -------------------------------
    retry = np.flatnonzero((residual > residual_tolerance) & (present.sum(axis=1) >= 4))
    if len(retry):
        best_r = residual[retry].copy()
        best_R, best_t = rotations[retry].copy(), positions[retry].copy()
        best_drop = np.full(len(retry), -1, dtype=int)
        for drop in range(n_markers):
            indices = np.array([i for i in range(n_markers) if i != drop])
            usable = present[retry][:, indices].all(axis=1)
            if len(indices) < 3 or not np.any(usable):
                continue
            sub = retry[usable]
            R, t, r = _kabsch(template, clean[sub], indices)
            improved = r < best_r[usable]
            where = np.flatnonzero(usable)[improved]
            best_r[where], best_drop[where] = r[improved], drop
            best_R[where], best_t[where] = R[improved], t[improved]
        rotations[retry], positions[retry] = best_R, best_t
        residual[retry], dropped[retry] = best_r, best_drop

    # --- 3. Everything still bad is interpolated and marked invalid --------------------
    valid = (residual <= residual_tolerance) & np.isfinite(residual)
    report = {
        'n_frames': int(n_frames),
        'n_invalid': int((~valid).sum()),
        'n_repaired_by_dropping_a_marker': int(((dropped >= 0) & valid).sum()),
        'n_frames_missing_a_marker': int((present.sum(axis=1) < n_markers).sum()),
        'fault_counts_per_marker': [int(((dropped == i) & valid).sum()) for i in range(n_markers)],
        # The standard mocap QA number, and the one the aggregate above cannot express. A
        # plate whose four markers are each present 75% of the time is a tracking problem;
        # one where a single marker is present 0% of the time is a protocol fact, and only
        # this distinguishes them. IMoVE's treadmill takes are the latter: they instrument
        # the right leg only, so every left-leg marker reads exactly 0.00% for the whole
        # take. Worth recording precisely because the artifact cannot tell that apart from a
        # tracking failure -- either way the plate is simply not there.
        'presence_fraction_per_marker': [float(present[:, i].mean()) for i in range(n_markers)],
        'min_marker_presence_fraction': float(present.mean(axis=0).min()),
        'residual_median_mm': float(np.median(residual[np.isfinite(residual)]) * 1000)
                              if np.any(np.isfinite(residual)) else float('nan'),
        'residual_p95_mm': float(np.percentile(residual[np.isfinite(residual)], 95) * 1000)
                           if np.any(np.isfinite(residual)) else float('nan'),
        'template_distances_mm': sorted(
            (float(np.linalg.norm(template[i] - template[j]) * 1000)
             for i in range(n_markers) for j in range(i + 1, n_markers)), reverse=True),
    }

    if not np.any(valid):
        print(f"Warning: {name or 'plate'} has no frame that fits its template within "
              f"{residual_tolerance * 1000:.0f} mm; leaving the reconstruction untouched.")
        return positions, rotations, valid, report

    if not np.all(valid):
        if interpolate_invalid:
            positions, rotations = _interpolate_frames(positions, rotations, timestamps, ~valid)
        if report['n_invalid'] > 0.05 * n_frames:
            print(f"Warning: {name or 'plate'}: {report['n_invalid']} of {n_frames} frames "
                  f"({100 * report['n_invalid'] / n_frames:.1f}%) do not fit the plate template "
                  f"and have been interpolated and marked invalid.")

    return positions, rotations, valid, report


def reconstruct_plate(
    markers: np.ndarray,
    timestamps: np.ndarray,
    template: np.ndarray = None,
    residual_tolerance: float = DEFAULT_PLATE_RESIDUAL_TOLERANCE_M,
    name: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """The full marker-plate reconstruction: template fit, then time-series repair.

    Supersedes `_reconstruct_from_markers` + `repair_reconstruction_glitches`, which it
    subsumes. The reason both existed is that they detect DIFFERENT things, and neither
    detector subsumes the other:

        template residual   wrong SHAPE at one instant        catches a displaced or
                                                              merged marker
        angular speed       wrong CHANGE between instants     catches a pose that jumps

    A marker collision has the right motion and the wrong shape, so only the residual sees
    it. A relabeling that preserves every distance has the right shape and the wrong motion,
    so only the speed detector sees it — measured on Subject06/complexTasks/femur_l, the
    template fit reproduces the swapped block at 0.20 mm residual, detecting nothing, while
    sitting exactly 180 degrees from the truth. Running both is the point.

    Order, and it matters:

      1. Fit the template. Kabsch, no assumption about plate shape, residual in millimetres.
      2. Leave-one-out on frames over tolerance: drop each marker in turn, keep the best
         three. Replaces `_compute_case`'s four rectangle-specific branches.
      3. Detect angular-speed discontinuities in the resulting pose sequence.
      4. Un-flip sustained relabelings, using corrections DERIVED from the template rather
         than the hardcoded rectangle half turns. A plate with no symmetries yields only the
         identity, so nothing is attempted and nothing can be wrongly "corrected".
      5. Interpolate transition frames and mark them invalid, along with anything the fit
         could not explain.

    Steps 3-4 must precede 5 for the reason `repair_reconstruction_glitches` already gives:
    interpolating first smears a 180 degree step across its neighbours, which both invents
    motion and hides the discontinuity from the detector that would have caught it.

    Returns (positions, rotations, valid, report), the same shape as both functions it
    replaces.
    """
    positions, rotations, fit_valid, report = fit_plate_to_template(
        markers, timestamps, template=template, residual_tolerance=residual_tolerance,
        name=name, interpolate_invalid=False)
    report = dict(report)
    report.update({'flipped': 0, 'interpolated': 0, 'unresolved': 0, 'symmetries': 0})

    if not np.any(fit_valid) or len(rotations) < 3:
        return positions, rotations, fit_valid, report

    # The un-flip needs a pose everywhere to compare across, so fill the frames the fit
    # rejected before looking for discontinuities. They stay invalid regardless.
    positions, rotations = _interpolate_frames(positions, rotations, timestamps, ~fit_valid)

    candidates = template_correction_candidates(
        template if template is not None else estimate_plate_template(
            markers[:, _present_mask(markers).any(axis=0)]))
    report['symmetries'] = len(candidates) - 1

    offending, _ = _offending_steps(rotations, timestamps)
    unexplained = []
    if np.any(offending):
        bad = _transition_mask(offending, len(rotations))
        # A frame the fit already rejected is not a trustworthy anchor for the comparison
        # across a gap, so it is excluded from the clean runs too.
        bad |= ~fit_valid
        if not bad.all():
            if report['symmetries']:
                rotations, n_flipped, unexplained = _unflip_swapped_blocks(
                    rotations, bad, name=name, candidates=candidates)
                report['flipped'] = n_flipped
            else:
                # No relabeling is invisible on this plate, so any surviving discontinuity
                # is a genuine failure rather than a swap. Reported, never "corrected".
                _, _, unexplained = _unflip_swapped_blocks(
                    rotations, bad, name=name, candidates={'identity': np.eye(3)})
            offending, _ = _offending_steps(rotations, timestamps)

    transition = _transition_mask(offending, len(rotations)) if np.any(offending) \
        else np.zeros(len(rotations), dtype=bool)
    unresolved_frames = np.zeros(len(rotations), dtype=bool)
    for last_good, next_good in unexplained:
        transition[last_good + 1:next_good] = False
        unresolved_frames[last_good + 1:next_good] = True

    valid = fit_valid & ~transition & ~unresolved_frames
    report['interpolated'] = int(transition.sum())
    report['unresolved'] = len(unexplained)
    report['n_invalid'] = int((~valid).sum())

    if unexplained:
        print(f"Warning: {name or 'plate'}: {len(unexplained)} discontinuity(ies) are "
              f"neither a marker relabeling this plate can hide nor safely interpolable, and "
              f"have been LEFT IN PLACE. Corrupt around frames "
              f"{[gap[0] for gap in unexplained[:5]]}; treat any joint using it with suspicion.")

    positions, rotations = _interpolate_frames(positions, rotations, timestamps, transition)
    return positions, rotations, valid, report


def world_trace_from_markers(markers: np.ndarray, timestamps: np.ndarray,
                             template: np.ndarray = None,
                             residual_tolerance: float = DEFAULT_PLATE_RESIDUAL_TOLERANCE_M,
                             name: str = None) -> WorldTrace:
    """One plate's markers -> one WorldTrace.

    Was WorldTrace.from_markers. It lives here rather than on the class because it BUILDS a
    trace out of something that is not one, which is the boundary this package draws: the
    trace classes hold data and answer questions about it, everything that constructs them
    from a measurement lives on the build side.

    Use `fit_plate_to_template` directly when the diagnostics matter — this returns only the
    trace, which is what the reader wants; the report is what its manifest wants.
    """
    positions, rotations, valid, _ = fit_plate_to_template(
        markers, timestamps, template=template,
        residual_tolerance=residual_tolerance, name=name)
    return WorldTrace(timestamps, positions, rotations, valid=valid)


def record_reconstruction(report, take, segment, fit, valid, timestamps, tolerance) -> None:
    """Everything fit_plate_to_template already computed and a reader would otherwise discard.

    Lives here rather than in one reader because every dataset reconstructs the same way and
    should report the same way. It was private to the IMoVE mocap reader, so the biplane
    reader -- which fits Vicon clusters with this very function -- recorded nothing at all.

    The residuals and per-marker fault counts are the only direct evidence of how good the
    ground truth is, and until now they reached a `print()` and nothing else -- so the answer
    to "which of these 281 trials should I not trust" required rebuilding and watching a
    terminal scroll past.
    """
    valid = np.asarray(valid, dtype=bool)
    runs = np.diff(np.flatnonzero(
        np.concatenate([[True], valid[1:] != valid[:-1], [True]])))
    invalid_runs = runs[0::2] if not valid[0] else runs[1::2]
    # fit's own keys first, so an explicit value here wins a name collision --
    # fit_plate_to_template already reports n_frames, and letting it override the
    # count taken from `timestamps` would silently mean two different things.
    metrics = {key: value for key, value in fit.items() if key != 'name'}
    metrics.update({
        'residual_tolerance_mm': tolerance * 1000.0,
        'valid_fraction': float(valid.mean()),
        'n_invalid_frames': int((~valid).sum()),
        'n_invalid_runs': int(len(invalid_runs)),
        'invalid_run_max': float(invalid_runs.max()) if len(invalid_runs) else 0.0,
        'n_frames': int(len(timestamps)),
    })
    report.add('S2_reconstruction', 'segment', f'{take}/{segment}', **metrics)
