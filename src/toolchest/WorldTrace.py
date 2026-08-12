import os
from pathlib import Path
import pandas as pd
from .IMUTrace import IMUTrace
from typing import List, Tuple, Union
import numpy as np
from .finite_difference_utils import central_difference
from .gyro_utils import finite_difference_rotations
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp
from typing import Dict

def _generate_smooth_motion_profile(
        num_samples: int,
        duration: float,
        num_waves: int = 4,
        max_amp: float = 1.0
    ) -> np.ndarray:
        """
        Generates a smooth, complex 1D motion profile using a sum of sine waves.

        Args:
            num_samples (int): The number of data points to generate.
            duration (float): The total time duration in seconds.
            num_waves (int): The number of sine waves to sum for complexity.
            max_amp (float): The maximum amplitude of the resulting motion.

        Returns:
            np.ndarray: A 1D array representing the motion profile.
        """
        t = np.linspace(0, duration, num_samples, endpoint=False)
        motion = np.zeros(num_samples)
        for i in range(1, num_waves + 1):
            amplitude = np.random.uniform(0.1, 1.0) * max_amp / num_waves
            frequency = np.random.uniform(0.1, 2.0) * i
            phase = np.random.uniform(0, 2 * np.pi)
            motion += amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return motion

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
                           name: str = None) -> Tuple[np.ndarray, int, int]:
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
    PlateTrial._align_world_trace_to_imu_trace. Only flips RELATIVE to the rest of the trial
    corrupt anything.

    Returns (rotations, frames un-flipped, unexplained gaps as (last_good, next_good) index
    pairs). Those gaps are what the caller must refuse to interpolate across.
    """
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
             for label, candidate in _HALF_TURN_CANDIDATES.items()),
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
    by PlateTrial._align_world_trace_to_imu_trace, which solves the sensor-to-segment
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
        template = estimate_plate_template(markers[:, ever_present])
        # Re-expand so template rows stay aligned with marker columns; the excluded rows are
        # never indexed, because `present` is False for them in every frame.
        full = np.zeros((n_markers, 3))
        full[ever_present] = template
        template = full
    template = np.asarray(template, dtype=np.float64)
    # Centred over the markers actually in use, so the origin — and therefore every reported
    # position — means the same physical point whether or not a marker was dropped.
    template = template - template[ever_present].mean(axis=0)
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
        good = np.flatnonzero(valid)
        query = np.clip(timestamps[~valid], timestamps[good[0]], timestamps[good[-1]])
        slerp = Slerp(timestamps[good], Rotation.from_matrix(rotations[good]))
        rotations[~valid] = slerp(query).as_matrix()
        for axis in range(3):
            positions[~valid, axis] = np.interp(query, timestamps[good], positions[good, axis])
        if report['n_invalid'] > 0.05 * n_frames:
            print(f"Warning: {name or 'plate'}: {report['n_invalid']} of {n_frames} frames "
                  f"({100 * report['n_invalid'] / n_frames:.1f}%) do not fit the plate template "
                  f"and have been interpolated and marked invalid.")

    return positions, rotations, valid, report


class WorldTrace:
    """
    This class contains a trace of a world frame over time. Optionally, this can attach an IMUTrace and manipulate it.
    Or, it can generate a synthetic trace by finite differencing the world frames over time.
    """

    def __init__(self, timestamps: np.ndarray, positions: Union[List[np.ndarray], np.ndarray],
                 rotations: Union[List[np.ndarray], np.ndarray],
                 valid: Union[List[bool], np.ndarray, None] = None):
        """`valid` marks, per frame, whether this pose is trustworthy GROUND TRUTH.

        Defaults to all-True, so every existing caller keeps its current meaning: a trace
        built by hand or from a synthetic generator is valid throughout. `from_trc` sets it
        from the marker-reconstruction repair (see repair_reconstruction_glitches), which is
        the only place in this repo that knows a pose was interpolated or left corrupt.

        The mask never affects the arrays. Poses stay on a uniform time grid because
        resampling, filtering and every finite-difference angular velocity here assume one;
        `valid` is how a frame is excluded from ERROR STATISTICS without being excluded from
        the signal a filter integrates through.
        """
        self.timestamps = timestamps
        self.positions = np.asarray(positions)
        self.rotations = np.asarray(rotations)
        self.valid = (np.ones(len(timestamps), dtype=bool) if valid is None
                      else np.asarray(valid, dtype=bool))
        if len(self.valid) != len(timestamps):
            raise ValueError(f"valid has length {len(self.valid)} but the trace has "
                             f"{len(timestamps)} frames.")

    def __len__(self):
        """
        Returns the number of samples in the WorldTrace. This allows us to call len(trace) on a WorldTrace instance.
        """
        return len(self.timestamps)

    def __sub__(self, other: 'WorldTrace') -> 'WorldTrace':
        """
        Allows us to subtract two WorldTrace instances. This will subtract the positions and rotations of the two traces.
        """
        if len(self) != len(other):
            raise ValueError(f"WorldTraces must have the same length to subtract them. Got self {len(self)} and other {len(other)}.")
        assert np.array_equal(self.timestamps[0],
                              other.timestamps[0]), "WorldTraces must have the same start time to subtract them."
        # A difference is only trustworthy where BOTH operands are.
        return WorldTrace(self.timestamps, self.positions - other.positions,
                          np.matmul(self.rotations, other.rotations.transpose(0, 2, 1)),
                          valid=self.valid & other.valid)

    def __getitem__(self, key) -> 'WorldTrace':
        """
        Allows us to use the square bracket notation to access the WorldTrace instance. This allows us to slice the
        WorldTrace instance and access ranges of items with `sub_trace = trace[1:4]`. If we pass an integer, we can
        return the corresponding item as a length 1 trace with `sub_trace = trace[2]` and `len(sub_trace) == 1`.
        """
        if isinstance(key, slice):
            # If key is a slice object, return a new WorldTrace instance with the sliced items
            return WorldTrace(self.timestamps[key], self.positions[key], self.rotations[key],
                              valid=self.valid[key])
        else:
            # If key is an integer, return the corresponding item as a length 1 trace
            return WorldTrace(np.array([self.timestamps[key]]), self.positions[key:key+1],
                              self.rotations[key:key+1], valid=self.valid[key:key+1])

    def __eq__(self, other):
        """
        Allows us to compare two WorldTrace instances for equality. This will return True if the timestamps, positions, and
        rotations are all _exactly_ equal.
        """
        if not isinstance(other, WorldTrace):
            return False
        if len(self) != len(other):
            return False
        if self.positions.shape[0] != other.positions.shape[0] or self.rotations.shape[0] != other.rotations.shape[0]:
            return False
        return (np.array_equal(self.timestamps, other.timestamps) and
                np.array_equal(self.positions, other.positions) and
                np.array_equal(self.rotations, other.rotations))

    @classmethod
    def from_markers(cls, markers: np.ndarray, timestamps: np.ndarray,
                     template: np.ndarray = None,
                     residual_tolerance: float = DEFAULT_PLATE_RESIDUAL_TOLERANCE_M,
                     name: str = None) -> 'WorldTrace':
        """One plate's markers -> one WorldTrace. The counterpart of IMUTrace.from_txt.

        Args:
            markers: (N, M, 3) world-frame marker positions in METRES, NaN or zero where
                a marker is missing. Unit conversion belongs to the reader, not here.
            timestamps: (N,) seconds.
            template: (M, 3) plate-fixed marker layout; estimated from the data if None.
            residual_tolerance: per-marker fit residual above which a frame is invalid.
            name: for warnings.

        Use `fit_plate_to_template` directly when the diagnostics matter — this returns only
        the trace, which is what a constructor should do; the report is what a reader wants
        for its manifest.
        """
        positions, rotations, valid, _ = fit_plate_to_template(
            markers, timestamps, template=template,
            residual_tolerance=residual_tolerance, name=name)
        return cls(timestamps, positions, rotations, valid=valid)

    @classmethod
    def from_trc(cls, trc_path: Union[str, Path]) -> Dict[str, 'WorldTrace']:
        """Parses a TRC file and extracts marker data into WorldTraces."""
        trc_path = Path(trc_path)
        if not trc_path.is_file():
            raise FileNotFoundError(f"No valid TRC file found at: {trc_path}")

        with open(trc_path, 'r', encoding='utf-8') as f:
            lines = [f.readline() for _ in range(6)]

        headers = lines[3].strip().split('\t')
        imu_headers = [h for h in headers if ('_o' in h.lower())]

        df = pd.read_csv(trc_path, delimiter='\t', skiprows=6, header=None, engine='c')
        timestamps = df.iloc[:, 1].to_numpy(dtype=np.float64)

        world_traces = {}
        for imu_o_name in imu_headers:
            o_idx = headers.index(imu_o_name)
            x_idx = headers.index(imu_o_name.replace('_o', '_x'))
            y_idx = headers.index(imu_o_name.replace('_o', '_y'))
            d_idx = headers.index(imu_o_name.replace('_o', '_d'))

            o_loc = df.iloc[:, o_idx: o_idx + 3].to_numpy(dtype=np.float64)
            x_loc = df.iloc[:, x_idx: x_idx + 3].to_numpy(dtype=np.float64)
            y_loc = df.iloc[:, y_idx: y_idx + 3].to_numpy(dtype=np.float64)
            d_loc = df.iloc[:, d_idx: d_idx + 3].to_numpy(dtype=np.float64)

            # Convert mm to meters if necessary
            if np.max(np.abs(o_loc)) > 1000.0:
                o_loc /= 1000.0
                x_loc /= 1000.0
                y_loc /= 1000.0
                d_loc /= 1000.0

            clean_name = imu_o_name.lower().replace('_o', '')
            positions, rotations = _reconstruct_from_markers(
                o_loc, d_loc, x_loc, y_loc, threshold=2.0
            )
            positions, rotations, valid, report = repair_reconstruction_glitches(
                positions, rotations, timestamps, name=clean_name)
            if any(report.values()):
                print(f"Warning: {trc_path.name}/{clean_name}: marker reconstruction repaired "
                      f"({report['flipped']} frame(s) un-flipped from swapped marker labels, "
                      f"{report['interpolated']} transition frame(s) interpolated, "
                      f"{report['unresolved']} discontinuity(ies) not attributable to a swap) "
                      f"— {int((~valid).sum())} frame(s) marked invalid; "
                      f"see repair_reconstruction_glitches.")
            world_traces[clean_name] = WorldTrace(timestamps, positions, rotations, valid=valid)
        return world_traces

    def transform(self, rotate: np.ndarray = np.eye(3), translate: np.ndarray = np.zeros(3)) -> 'WorldTrace':
        """
        This function transforms the WorldTrace by rotating and translating the positions and rotations.
        """
        return WorldTrace(self.timestamps,
                          np.einsum('ij,nj->ni', rotate, self.positions) + translate,
                          np.matmul(rotate, self.rotations),
                          valid=self.valid)

    def allclose(self, other, atol=1e-6):
        """
        Allows us to compare two WorldTrace instances for approximate equality. This will return True if the timestamps,
        positions, and rotations are all approximately equal within the specified tolerance.
        """
        if not isinstance(other, WorldTrace):
            return False
        if len(self) != len(other):
            return False
        if self.positions.shape[0] != other.positions.shape[0] or self.rotations.shape[0] != other.rotations.shape[0]:
            return False
        return (np.allclose(self.timestamps, other.timestamps, atol=atol) and
                np.allclose(self.positions, other.positions, atol=atol) and
                np.allclose(self.rotations, other.rotations, atol=atol))
    
    def copy(self) -> 'WorldTrace':
        """
        Returns a deep copy of the WorldTrace object.
        """
        return WorldTrace(
            self.timestamps.copy(),
            self.positions.copy(),
            self.rotations.copy(),
            valid=self.valid.copy()
        )
    
    def resample(self, new_frequency: float) -> 'WorldTrace':
        """
        This function resamples the WorldTrace to a new, specified frequency using linear interpolation for
        positions and spherical linear interpolation (slerp) for rotations.

        Args:
            new_frequency: The desired sampling frequency in Hz.

        Returns:
            A new WorldTrace instance resampled to the new frequency.
        """
        if new_frequency <= 0:
            raise ValueError("New frequency must be a positive number.")

        original_timestamps = self.timestamps
        if len(original_timestamps) < 2:
            return self.copy() # Cannot resample a trace with 0 or 1 samples

        # 1. Determine the new timestamps, prioritizing a perfect time step (dt)
        start_time = original_timestamps[0]
        last_original_time = original_timestamps[-1]
        new_dt = 1.0 / new_frequency
        
        # Define a small tolerance (epsilon) based on the new time step. 
        # This is used to make the 'stop' value in np.arange inclusive.
        # We use a fraction of the new_dt to ensure precision.
        epsilon = new_dt * 1e-6 
        
        # Calculate the theoretical end of the uniform grid. 
        # This is the last point generated by the perfect steps.
        time_duration = last_original_time - start_time
        num_intervals = np.round(time_duration / new_dt)
        theoretical_last_step = start_time + num_intervals * new_dt
        
        # np.arange creates a perfectly uniform time series
        new_timestamps = np.arange(start=start_time, 
                                   stop=theoretical_last_step, 
                                   step=new_dt)

        # 2. Interpolate Positions (Linear Interpolation)
        # Create interpolation functions for each dimension (x, y, z)
        interp_x = np.interp(new_timestamps, original_timestamps, self.positions[:, 0])
        interp_y = np.interp(new_timestamps, original_timestamps, self.positions[:, 1])
        interp_z = np.interp(new_timestamps, original_timestamps, self.positions[:, 2])

        # Combine interpolated axes back into a single array
        new_positions = np.column_stack((interp_x, interp_y, interp_z))

        # 3. Interpolate Rotations (SLERP via Quaternions)
        # Convert 3x3 rotation matrices to quaternions (x, y, z, w)
        original_quats = Rotation.from_matrix(self.rotations).as_quat(canonical=True)

        # Create a Rotation object for interpolation
        # from_quat creates a set of rotations
        original_rotations = Rotation.from_quat(original_quats)

        # Slerp the rotations to the new timestamps
        # The 'Slerp' method internally uses the original timestamps for interpolation
        slerp = Slerp(original_timestamps, original_rotations)
        new_rotations_obj = slerp(new_timestamps)

        # Convert interpolated Rotation objects back to an array of 3x3 matrices
        new_rotations = new_rotations_obj.as_matrix().copy()

        # 4. Carry the validity mask across, conservatively.
        # A resampled frame is interpolated from its two neighbours, so it inherits any
        # invalidity from either. np.interp over the mask as floats gives a nonzero value
        # wherever an invalid frame contributed at all; requiring 1.0 to stay valid is the
        # conservative reading, and it also keeps invalid RUNS from shrinking at the edges.
        valid_weight = np.interp(new_timestamps, original_timestamps, self.valid.astype(np.float64))
        new_valid = valid_weight >= 1.0

        return WorldTrace(new_timestamps, new_positions, new_rotations, valid=new_valid)

    def finite_difference_world_frame_accelerations(self, acc_from_gravity: np.ndarray = np.zeros(3)) -> np.ndarray:
        """
        This function computes the acceleration of the world frame by finite differencing the positions.
        """
        # central_difference treats the columns of an (N, 3) array independently, so
        # both differentiations run on all three axes at once.
        velocity = central_difference(self.positions, self.timestamps)
        return central_difference(velocity, self.timestamps) + acc_from_gravity

    def calculate_imu_trace(self,
                            acc_from_gravity: np.ndarray = np.zeros(3),
                            magnetic_field: np.ndarray = np.zeros(3),
                            skip_lin_acc=False) -> IMUTrace:
        """
        This function computes the IMU trace from the world trace by finite differencing the positions and rotations.
        """
        rotations_np = np.array(self.rotations)
        if not skip_lin_acc:
            world_acc = self.finite_difference_world_frame_accelerations(acc_from_gravity)
            world_acc_np = np.array(world_acc)
            local_acc = np.einsum('nji,nj->ni', rotations_np, world_acc_np)
        else:
            local_acc = np.einsum('nji,j->ni', rotations_np, acc_from_gravity)
        assert isinstance(magnetic_field, np.ndarray)
        local_mag = np.einsum('nji,j->ni', rotations_np, magnetic_field)
        local_gyros = finite_difference_rotations(self.rotations, self.timestamps)
        return IMUTrace(self.timestamps, local_gyros, local_acc, local_mag)

    def re_zero_timestamps(self) -> 'WorldTrace':
        """
        Start timestamps at 0
        """
        return WorldTrace(self.timestamps - self.timestamps[0], self.positions, self.rotations,
                          valid=self.valid)
        
    @staticmethod
    def generate_random_world_trace(duration: float = 10.0, fs: float = 100.0) -> 'WorldTrace':
        """
        Generates a WorldTrace with random but smooth position and orientation.

        Args:
            duration (float): The duration of the trial in seconds.
            fs (float): The sampling frequency in Hz.

        Returns:
            WorldTrace: The generated world trace.
        """
        num_samples = int(duration * fs)
        timestamps = np.linspace(0, duration, num_samples, endpoint=False)

        # --- Generate smooth random position ---
        pos_x = _generate_smooth_motion_profile(num_samples, duration, max_amp=0.5)
        pos_y = _generate_smooth_motion_profile(num_samples, duration, max_amp=0.3)
        pos_z = _generate_smooth_motion_profile(num_samples, duration, max_amp=0.5)
        positions = np.column_stack((pos_x, pos_y, pos_z))

        # --- Generate smooth random orientation ---
        # Create motion profiles for Euler angles
        rot_z = _generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi)
        rot_y = _generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi / 2)
        rot_x = _generate_smooth_motion_profile(num_samples, duration, max_amp=np.pi / 2)

        # Convert Euler angles to a stack of rotation matrices
        rotations_obj = Rotation.from_euler('zyx', np.vstack([rot_z, rot_y, rot_x]).T)
        rotations = rotations_obj.as_matrix()

        return WorldTrace(timestamps, positions, rotations)
    
    def get_sample_frequency(self):
        """
        This function returns the sample frequency of the WorldTrace.
        """
        return 1 / np.mean(np.diff(self.timestamps))

    def lowpass_filter(self, cutoff_freq: float, order: int):
        """
        This function applies a lowpass filter to the WorldTrace.
        """
        sample_freq = self.get_sample_frequency()
        nyquist_freq = 0.5 * sample_freq
        cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, cutoff, btype='low') # type: ignore
        positions = filtfilt(b, a, self.positions, axis=0)
        angle_axis = Rotation.from_matrix(self.rotations).as_rotvec()
        angle_axis = filtfilt(b, a, angle_axis, axis=0)
        rotations = Rotation.from_rotvec(angle_axis).as_matrix()
        # filtfilt is non-causal and spreads every sample over the filter's support, so an
        # invalid frame contaminates its neighbours. The mask is carried unchanged rather
        # than dilated: widening it would need the filter's effective support, which varies
        # with order and cutoff, and the caller who low-passes a trace with invalid frames
        # in it has a bigger problem than the mask's exact width.
        return WorldTrace(self.timestamps, positions, rotations, valid=self.valid)

    def get_rotation_errors_deg(self, other_trace: 'WorldTrace') -> np.ndarray:
        """
        This function returns a time series list of the rotation errors in degrees between two WorldTrace instances.
        """
        assert len(self) == len(other_trace), "WorldTraces must have the same length to compare them."
        
        errors = Rotation.from_matrix(np.matmul(self.rotations.transpose(0, 2, 1), other_trace.rotations))
        angle_axis = errors.as_rotvec()
        angle_deg = np.linalg.norm(angle_axis, axis=1) * 180.0 / np.pi
        return angle_deg

    def get_joint_center(self, other_world_trace: 'WorldTrace') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Given two world traces, solve for the best fit constant offset from a joint center.
        This is done by minimizing the sum of the squared differences between the two traces. """
        # Parent is other, child is self
        assert isinstance(other_world_trace, WorldTrace), "Must pass a WorldTrace instance to compare."
        assert len(self) == len(other_world_trace), "WorldTraces must have the same length to compare them."

        parent_loc = self.positions
        child_loc = other_world_trace.positions

        r_c_p = parent_loc - child_loc
        r_c_p = r_c_p.flatten()

        R_w_parent = self.rotations.reshape(-1, 3)
        R_w_child = other_world_trace.rotations.reshape(-1, 3)
        R_w = np.hstack((-R_w_parent, R_w_child))

        offsets, res, rank, S = np.linalg.lstsq(R_w, r_c_p, rcond=None)

        parent_offset = offsets[:3]
        child_offset = offsets[3:]
        error = R_w_parent @ parent_offset - R_w_child @ child_offset + r_c_p
        error = error.reshape(-1, 3)
        return parent_offset, child_offset, error
    
    def get_primary_joint_axis(self, other_world_trace: 'WorldTrace') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the primary axis of a hinge joint between two world traces using
        the Mean Axis method, expressed in both the parent's ('self') and
        child's ('other') coordinate frames.

        This method works by finding the axis of rotation for the relative
        orientation at each time step and then determining the mean direction of
        that axis over the entire trial. This is robust to joints where the
        instantaneous axis of rotation may wobble. The mean direction is found
        by computing the principal eigenvector of the covariance matrix of the
        instantaneous axes.

        Args:
            other_world_trace (WorldTrace): The 'child' segment's world trace.
                                          'self' is assumed to be the 'parent'.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - axis_in_self (np.ndarray): The (3,) unit vector for the primary
                                         axis in the `self` (parent) frame.
            - axis_in_other (np.ndarray): The (3,) unit vector for the primary
                                          axis in the `other` (child) frame.
        """
        assert isinstance(other_world_trace, WorldTrace), "Must pass a WorldTrace instance."
        assert len(self) == len(other_world_trace), "WorldTraces must have the same length."

        def _find_mean_axis(relative_rotations: Rotation) -> np.ndarray:
            """
            Finds the mean axis of rotation from a time series of rotations using
            Principal Component Analysis on the instantaneous axes.
            """
            # Get the rotation vectors (axis * angle) for each time step
            rot_vecs = relative_rotations.as_rotvec()  # shape (N, 3)
            
            # Normalize each rotation vector to get the instantaneous axis of rotation
            # Handle cases where the angle is zero to avoid division by zero
            norms = np.linalg.norm(rot_vecs, axis=1)
            non_zero_mask = norms > 1e-8
            
            # If there's no significant rotation anywhere, we can't determine an axis.
            if not np.any(non_zero_mask):
                # Return a default axis, as no motion was detected.
                return np.array([0., 0., 1.])

            axes = np.zeros_like(rot_vecs)
            # Normalize only the non-zero rotation vectors
            axes[non_zero_mask] = rot_vecs[non_zero_mask] / norms[non_zero_mask, np.newaxis]
            
            # To handle the axis ambiguity (v is the same as -v), we ensure all
            # axes point in the same general direction as the first axis.
            first_axis = axes[np.argmax(non_zero_mask)]
            for i in range(len(axes)):
                if np.dot(axes[i], first_axis) < 0:
                    axes[i] *= -1
            
            # The mean axis is the principal component of the distribution of axes,
            # which is the eigenvector of the covariance matrix corresponding to the
            # largest eigenvalue.
            covariance_matrix = np.cov(axes[non_zero_mask].T)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            
            # The mean axis is the eigenvector with the largest eigenvalue
            mean_axis = eigenvectors[:, np.argmax(eigenvalues)].real
            return mean_axis / np.linalg.norm(mean_axis)

        # --- Step 1: Get Relative Rotations from both perspectives ---
        R_wp_stack = Rotation.from_matrix(self.rotations)
        R_wc_stack = Rotation.from_matrix(other_world_trace.rotations)

        # --- Step 2: Calculate axis in the 'self' (parent) frame ---
        # Use R_pc = R_parent.inv() * R_child
        R_pc_stack = R_wp_stack.inv() * R_wc_stack
        axis_in_self = _find_mean_axis(R_pc_stack)

        # --- Step 3: Calculate axis in the 'other' (child) frame ---
        # Use R_cp = R_child.inv() * R_parent
        R_cp_stack = R_wc_stack.inv() * R_wp_stack
        axis_in_other = _find_mean_axis(R_cp_stack)

        return axis_in_self, axis_in_other
    
