"""
Covers experiments/acceleration_projection.py — the analysis that validates the rigid-body
acceleration projection three ways: against markers, against the other segment, and against a
second sensor on the same segment.

WHAT THESE TESTS ARE FOR. Every quantity in that module is a difference between two acceleration
signals, and the ways it can be wrong all produce a plausible number rather than a crash:

  * a transposed frame transform. C(t) = R_parent^T R_child carries a vector from the CHILD frame
    into the PARENT one; C^T carries it the other way. Either direction produces a residual of a
    few m/s^2 on real data, which reads as a finding.
  * a reference built with a flipped gravity sign or the other segment's joint centre. Still the
    right shape, still a plausible magnitude, just a larger error.
  * a magnitude channel that loses the sign of its gap, so a systematic overshoot reads as noise.
  * an alignment or angle column filled with a degenerate zero, which reads as perfect agreement.

So the tests below construct inputs whose answer is known in CLOSED FORM — a stationary plate, a
joint that is rigid by construction, two sensors on one rigid body — and check the vectors, not
the shapes. On those fixtures the projection is EXACT, so the tolerances are numerical rather than
statistical, and any convention error fails by orders of magnitude instead of by a few percent.

The projection itself is not retested here: IMUTrace.project_acc is covered by test/TestIMUTrace.py
(invertibility, the zero-offset no-op, the closed form for an offset orthogonal to the rotation
axis) and WorldTrace.get_joint_center by test/TestWorldTrace.py. What this module ADDS is the three
references and the windowing, and that is what is exercised.

Synthetic throughout — no dataset needed.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import experiments.acceleration_projection as ap
from experiments.acceleration_projection import (CHANNELS, LOWPASS_CUTOFF_HZ, PRIMARY_GYRO_METHOD,
                                                 TrialContext, _as_magnitude, angle_between_deg,
                                                 build_window, chordal_mean, condition,
                                                 fit_separation_inertial, gyro_derivative,
                                                 joint_center_reference_acc, lever_operator,
                                                 lowpass, norm_gap, residual_alignment,
                                                 rotation_spread_deg, same_segment_groups,
                                                 sample_rows, segment_geometry, skew, stats_row,
                                                 valid_runs)
from experiments.experiment_utils import EXPECTED_GRAVITY, _compute_perfect_joint_acc
from experiments.global_assumptions import DatasetSpec
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace

FS = 100.0
# Long enough that a run clears 2 x the trim (3 periods of LOWPASS_CUTOFF_HZ at each end, so 100
# samples total at 100 Hz) with several seconds left, AND clears joint_center's MIN_FIT_FRAMES.
DURATION = 12.0

PARENT_OFFSET = np.array([0.03, -0.19, 0.02])  # sensor -> joint center, parent side
CHILD_OFFSET = np.array([-0.01, 0.16, 0.04])   # and child side, pointing back up the limb

# Two sensors on ONE rigid segment, as positions in the segment's own frame. Separated by ~15 cm,
# matching the High-to-Low spacing on an IMoVE thigh, which is what the same-segment family
# measures.
SENSOR_A_IN_SEGMENT = np.array([0.02, 0.07, -0.01])
SENSOR_B_IN_SEGMENT = np.array([-0.01, -0.08, 0.03])
# Each sensor's own body-to-segment rotation. Distinct and non-trivial, because a same-segment
# comparison with two identical sensor frames would pass with the frame transform omitted
# entirely — the one error the constant C_ab exists to prevent.
MISALIGN_A = Rotation.from_euler('xyz', [4.0, -7.0, 11.0], degrees=True).as_matrix()
MISALIGN_B = Rotation.from_euler('xyz', [-9.0, 3.0, -5.0], degrees=True).as_matrix()


def _smooth_curve(timestamps: np.ndarray, amplitudes, frequencies, phases) -> np.ndarray:
    """Sum of a few sinusoids per axis — smooth, non-degenerate, and band-limited well below
    LOWPASS_CUTOFF_HZ so that the analysis filter is not what any tolerance here measures."""
    return np.column_stack([
        sum(a * np.sin(2 * np.pi * f * timestamps + p) for a, f, p in zip(amp, freq, phase))
        for amp, freq, phase in zip(amplitudes, frequencies, phases)
    ])


# Distinct frequencies per segment and per axis, so the joint-center least squares is well
# conditioned: a segment that only ever rotates about one axis leaves the offset along that axis
# unobservable and get_joint_center would return something arbitrary.
ROTATION_SPECS = (
    dict(amplitudes=[(0.45, 0.12), (0.30, 0.09), (0.25, 0.07)],
         frequencies=[(0.6, 1.6), (0.9, 2.1), (0.4, 1.3)],
         phases=[(0.3, 1.1), (1.4, 0.2), (2.2, 0.8)]),
    dict(amplitudes=[(0.60, 0.15), (0.20, 0.08), (0.35, 0.10)],
         frequencies=[(0.8, 1.4), (1.3, 2.5), (0.5, 1.8)],
         phases=[(1.0, 0.5), (0.2, 1.7), (0.9, 2.4)]),
)
PATH_SPEC = dict(amplitudes=[(0.25, 0.05), (0.10, 0.03), (0.18, 0.04)],
                 frequencies=[(0.7, 1.9), (1.1, 2.3), (0.5, 1.7)],
                 phases=[(0.0, 1.2), (0.6, 2.1), (1.9, 0.4)])


def _plate(name: str, timestamps: np.ndarray, positions: np.ndarray,
           rotations: np.ndarray) -> PlateTrial:
    """A PlateTrial whose IMU is the EXACT specific force at that pose's origin.

    `WorldTrace.calculate_imu_trace` is the inverse of what the analysis does, so a plate built
    this way is noiseless and its projection is exact — which is what makes every tolerance below
    numerical rather than statistical.
    """
    world_trace = WorldTrace(timestamps, positions, rotations)
    return PlateTrial(name=name, imu_trace=world_trace.calculate_imu_trace(EXPECTED_GRAVITY),
                      world_trace=world_trace)


def make_rigid_joint(fs: float = FS, duration: float = DURATION):
    """A synthetic joint that is exactly rigid, as (parent_plate, child_plate).

    Both segments are built BACKWARD from a shared joint-center path: given the joint center
    p_jc(t) and each segment's orientation R(t), the segment's sensor sits at p_jc(t) - R(t) @ offset,
    so the segment's own implied joint center R(t) @ offset + position is p_jc(t) exactly, for both
    segments and every sample. That is what makes this a fair test of the references: the two
    segments cannot disagree about where the joint is, so every reference point the module computes
    must coincide and every cross-segment residual must be zero.
    """
    n = int(duration * fs)
    timestamps = np.arange(n) / fs
    joint_center = _smooth_curve(timestamps, **PATH_SPEC)

    plates = []
    for name, offset, spec in zip(('femur_r_imu', 'tibia_r_imu'),
                                  (PARENT_OFFSET, CHILD_OFFSET), ROTATION_SPECS):
        rotations = Rotation.from_rotvec(_smooth_curve(timestamps, **spec)).as_matrix()
        plates.append(_plate(name, timestamps,
                             joint_center - np.einsum('nij,j->ni', rotations, offset), rotations))
    return plates[0], plates[1]


def make_shared_segment(fs: float = FS, duration: float = DURATION):
    """Two sensors on ONE rigid segment, as (plate_a, plate_b).

    Built the way IMoVE's placements really are: one segment pose, two sensors at fixed positions
    in that segment's frame, each carrying its own body-to-segment rotation. So

        r_ab  = MISALIGN_A^T (o_b - o_a)          constant, and what `segment_geometry` must recover
        C_ab  = MISALIGN_A^T MISALIGN_B           constant, and the transform every metric needs

    Both exactly constant, which is also true of the real dataset and for the same reason (see
    `segment_geometry`'s docstring), so this fixture reproduces the structure rather than an
    idealization of it.
    """
    n = int(duration * fs)
    timestamps = np.arange(n) / fs
    segment_position = _smooth_curve(timestamps, **PATH_SPEC)
    segment_rotation = Rotation.from_rotvec(_smooth_curve(timestamps, **ROTATION_SPECS[0])).as_matrix()

    plates = []
    for name, offset, misalign in (('femur_r_h', SENSOR_A_IN_SEGMENT, MISALIGN_A),
                                   ('femur_r_l', SENSOR_B_IN_SEGMENT, MISALIGN_B)):
        plates.append(_plate(
            name, timestamps,
            segment_position + np.einsum('nij,j->ni', segment_rotation, offset),
            np.einsum('nij,jk->nik', segment_rotation, misalign)))
    return plates[0], plates[1]


def expected_separation() -> np.ndarray:
    """r_ab as the fixture defines it, in sensor A's body frame."""
    return MISALIGN_A.T @ (SENSOR_B_IN_SEGMENT - SENSOR_A_IN_SEGMENT)


def expected_transform() -> np.ndarray:
    """C_ab = R_A^T R_B as the fixture defines it: B's body frame into A's."""
    return MISALIGN_A.T @ MISALIGN_B


# A spec with two sensors on the parent segment, so every family is exercised by one fixture:
# marker and joint on both knee pairs, and segment on the femur's two placements. The display
# names end in placement tokens because that is the contract `same_segment_groups` reads.
TEST_SPEC = DatasetSpec(
    name='synthetic',
    segment_sensor={'Femur R High': 'femur_r_h', 'Femur R Low': 'femur_r_l',
                    'Tibia R Mid': 'tibia_r_imu'},
    joints={'R_Knee': ('femur_r_h', 'tibia_r_imu'), 'R_Knee_L': ('femur_r_l', 'tibia_r_imu')},
    primary_joints=('R_Knee',),
    pelvis_sensor='femur_r_h',
    foot_sensors=(),
    subject_label='{}',
    field_reference={},
)


def make_trial_plates(fs: float = FS, duration: float = DURATION):
    """Three plates matching TEST_SPEC: two on a rigid femur, one tibia sharing a rigid knee.

    The femur's two placements come from `make_shared_segment` and the tibia is placed so that
    BOTH femur sensors imply the same knee centre, which makes every one of the three families
    exact on this trial at once.
    """
    n = int(duration * fs)
    timestamps = np.arange(n) / fs
    femur_h, femur_l = make_shared_segment(fs, duration)

    # The knee centre, as a fixed point of the femur segment: put it at PARENT_OFFSET from
    # sensor A, in A's frame, so `get_joint_center` on either femur placement recovers a
    # consistent point.
    knee = (femur_h.world_trace.positions
            + np.einsum('nij,j->ni', femur_h.world_trace.rotations, PARENT_OFFSET))
    tibia_rotations = Rotation.from_rotvec(_smooth_curve(timestamps, **ROTATION_SPECS[1])).as_matrix()
    tibia = _plate('tibia_r_imu', timestamps,
                   knee - np.einsum('nij,j->ni', tibia_rotations, CHILD_OFFSET), tibia_rotations)
    return {'femur_r_h': femur_h, 'femur_r_l': femur_l, 'tibia_r_imu': tibia}


def make_context(**kwargs) -> TrialContext:
    return TrialContext(make_trial_plates(**kwargs), TEST_SPEC, 'synthetic', 's1', 't1')


# ==============================================================================
# The mocap reference (family 1's truth)
# ==============================================================================

class TestReferenceConstruction(unittest.TestCase):
    """joint_center_reference_acc — the mocap truth the marker family is measured against."""

    def test_stationary_plate_reads_gravity_in_its_own_frame(self):
        """The sharpest available check of the gravity convention, and the one that needs no
        differentiation at all: a plate that never moves must produce EXPECTED_GRAVITY rotated
        into its body frame, whatever the offset is. A flipped sign or a transposed rotation both
        fail here; both survive a magnitude check, which is why this compares vectors."""
        n = 200
        rotation = Rotation.from_euler('xyz', [30.0, -50.0, 15.0], degrees=True)
        rotations = np.repeat(rotation.as_matrix()[None], n, axis=0)
        plate = _plate('femur_r_imu', np.arange(n) / FS, np.zeros((n, 3)), rotations)

        body_acc, linear_acc = joint_center_reference_acc(plate, PARENT_OFFSET)
        expected = rotation.as_matrix().T @ EXPECTED_GRAVITY
        np.testing.assert_allclose(body_acc, np.broadcast_to(expected, (n, 3)), atol=1e-9)
        np.testing.assert_allclose(linear_acc, 0.0, atol=1e-9)

    def test_linear_acceleration_return_is_gravity_removed_in_the_world_frame(self):
        """The second return has to be the first with gravity taken back out, in the WORLD frame.
        The truth_lin_norm column reads it as a linear acceleration and every figure scales
        against it, so a body-frame subtraction here would silently smear gravity across all
        three axes."""
        parent, _ = make_rigid_joint()
        body_acc, linear_acc = joint_center_reference_acc(parent, PARENT_OFFSET)
        world_acc = np.einsum('nij,nj->ni', parent.world_trace.rotations, body_acc)
        np.testing.assert_allclose(world_acc - EXPECTED_GRAVITY, linear_acc, atol=1e-9)

    def test_own_and_shared_joint_centers_agree_when_the_joint_is_rigid(self):
        """joint_center_reference_acc (the segment's OWN implied joint centre) and
        experiment_utils._compute_perfect_joint_acc (the MIDPOINT of both segments') are different
        reference points on real data, and the module reports both. On a joint that is rigid by
        construction they are the same point, so they must agree — which is what pins the local
        construction as MIRRORING the pipeline's rather than merely resembling it. Any difference
        in frame convention, gravity sign or differentiation order shows up here."""
        parent, child = make_rigid_joint()
        parent_offset, child_offset, _ = parent.world_trace.get_joint_center(child.world_trace)
        shared = dict(zip(('parent', 'child'), _compute_perfect_joint_acc(parent, child)))
        for role, plate, offset in (('parent', parent, parent_offset),
                                    ('child', child, child_offset)):
            own, _ = joint_center_reference_acc(plate, offset)
            np.testing.assert_allclose(own, shared[role], atol=1e-9)

    def test_reference_equals_the_projection_when_both_are_exact(self):
        """The claim the marker family exists to test, on a fixture where it holds up to the
        differentiators' own discretization: the mocap truth at the joint centre and the rigid-body
        projection of the sensor onto that same point are the same physical quantity.

        THE TOLERANCE IS THE FINITE-DIFFERENCE FLOOR, NOT A CHOICE. `alpha` comes from a 10-sample
        polynomial fit of the gyro, whose error scales as h^2 times the gyro's third derivative, and
        the mocap side is a second difference of positions. On this fixture that leaves ~0.05 m/s^2
        against signals of ~10 — so a tight absolute tolerance would be testing scipy's
        differentiators rather than this module's conventions.

        What makes the test discriminating is therefore the SECOND assertion, not the first: the
        same comparison against the other segment's reference point — the single most likely
        convention error, and one that produces a plausible-looking residual rather than a crash —
        must be an order of magnitude worse. A test that only bounded the residual would pass with
        the wrong reference at a looser tolerance; this one cannot.
        """
        parent, _ = make_rigid_joint()
        reference, _ = joint_center_reference_acc(parent, PARENT_OFFSET)
        wrong_point, _ = joint_center_reference_acc(parent, CHILD_OFFSET)
        projected = parent.imu_trace.project_acc(PARENT_OFFSET, 'polyfit').acc
        interior = slice(200, -200)  # clear of both differentiators' edge padding

        # ~0.11 m/s^2 as the fixture stands, against signals of ~10 and a wrong-reference residual
        # of ~1.5. The bound is set just above the measured floor rather than at it, so a real
        # regression trips it while a change to either differentiator's edge handling does not.
        residual = ap.rms(projected[interior] - reference[interior])
        self.assertLess(residual, 0.15)
        self.assertGreater(ap.rms(projected[interior] - wrong_point[interior]), 10 * residual)


# ==============================================================================
# Transport conventions
# ==============================================================================

class TestTransportConventions(unittest.TestCase):
    """The frame changes families 2 and 3 depend on. Every one of these can be transposed and
    still produce a plausible residual, so each is checked against the fixture's own definition."""

    def test_relative_rotation_carries_child_into_parent(self):
        """C(t) = R_parent^T R_child, applied as C @ v, must take a vector expressed in the CHILD
        body frame to the same physical vector expressed in the PARENT body frame. Checked on
        gravity, which is one known world vector both frames can express."""
        parent, child = make_rigid_joint()
        context = TrialContext({'femur_r_h': parent, 'tibia_r_imu': child}, TEST_SPEC,
                               'synthetic', 's1', 't1')
        C = context.relative_rotation('femur_r_h', 'tibia_r_imu')

        gravity_in_child = np.einsum('nji,j->ni', child.world_trace.rotations, EXPECTED_GRAVITY)
        gravity_in_parent = np.einsum('nji,j->ni', parent.world_trace.rotations, EXPECTED_GRAVITY)
        np.testing.assert_allclose(np.einsum('nij,nj->ni', C, gravity_in_child),
                                   gravity_in_parent, atol=1e-9)

    def test_lever_operator_reproduces_project_acc(self):
        """`lever_operator` is a second implementation of project_acc's arithmetic, used by the
        inertial geometry fit because it exposes the projection's LINEARITY in the offset. Two
        implementations that must agree only agree until they do not, so they are diffed here.

        Also the check on `gyro_derivative`: it duplicates IMUTrace's private dispatch, so if
        either mapping drifts this fails."""
        parent, _ = make_rigid_joint()
        trace = parent.imu_trace
        for method in ('backward', 'central', 'first_order', 'polyfit'):
            K = lever_operator(trace.gyro, gyro_derivative(trace, method))
            np.testing.assert_allclose(trace.acc + np.einsum('nij,j->ni', K, PARENT_OFFSET),
                                       trace.project_acc(PARENT_OFFSET, method).acc, atol=1e-9,
                                       err_msg=f"lever_operator disagrees with project_acc "
                                               f"under '{method}'")

    def test_skew_is_the_cross_product(self):
        vectors = np.array([[1.0, -2.0, 0.5], [0.0, 0.3, -0.7]])
        other = np.array([[-0.4, 1.1, 2.0], [0.9, 0.0, 0.2]])
        np.testing.assert_allclose(np.einsum('nij,nj->ni', skew(vectors), other),
                                   np.cross(vectors, other), atol=1e-12)


# ==============================================================================
# Family 2: cross-segment agreement
# ==============================================================================

class TestCrossSegmentAgreement(unittest.TestCase):
    """The two segments' projections of one point, against each other."""

    @classmethod
    def setUpClass(cls):
        cls.context = make_context()
        cls.comparisons = ap.joint_comparisons(cls.context, 'R_Knee', detailed=True)
        cls.by_channel = {c.channel: c for c in cls.comparisons}

    def test_both_channels_are_produced(self):
        self.assertEqual(set(self.by_channel), set(CHANNELS))
        self.assertEqual(self.by_channel['vector'].scope, 'mocap')
        self.assertEqual(self.by_channel['norm'].scope, 'full')

    def test_the_two_projections_agree_on_a_rigid_joint(self):
        """The whole claim of family 2. Both segments project to one physical point, so once
        transported into a common frame their projected accelerations are the same vector.

        The tolerance is the derivative scheme's discretization (see
        `test_reference_equals_the_projection_when_both_are_exact`); what makes it discriminating is
        that the TRANSPOSED transport — the one convention error that would otherwise pass unnoticed
        — is checked here too and must be an order of magnitude worse."""
        comparison = self.by_channel['vector']
        residual = ap.rms(comparison.estimate - comparison.truth)
        self.assertLess(residual, 0.15)

        C = self.context.relative_rotation('femur_r_h', 'tibia_r_imu')
        child = self.context.projected('tibia_r_imu', self.context.offsets['R_Knee']['child'])
        transposed = condition(np.einsum('nji,nj->ni', C, child), comparison.window)
        self.assertGreater(ap.rms(comparison.estimate - transposed), 10 * residual)

    def test_the_unprojected_signals_disagree_by_the_rigid_body_term(self):
        """The baseline has to be much worse, or the comparison above measures nothing. The two
        sensors are ~30 cm apart across the joint, so their raw readings differ by the rigid-body
        term the projection exists to remove."""
        comparison = self.by_channel['vector']
        truth_raw = comparison.extras['truth_raw']
        self.assertGreater(ap.rms(comparison.estimate_raw - truth_raw),
                           10 * ap.rms(comparison.estimate - comparison.truth))

    def test_the_baseline_is_transported_too(self):
        """`truth_raw` must be the CHILD's unprojected reading in the parent frame, not the
        child's projection. Comparing a raw parent against a projected child would make the
        'unprojected baseline' half-projected, and it would still look like a baseline."""
        comparison = self.by_channel['vector']
        C = self.context.relative_rotation('femur_r_h', 'tibia_r_imu')
        expected = condition(np.einsum('nij,nj->ni', C, self.context.acc('tibia_r_imu')),
                             comparison.window)
        np.testing.assert_allclose(comparison.extras['truth_raw'], expected, atol=1e-9)

    def test_norm_channel_carries_each_side_magnitude_and_keeps_the_sign(self):
        """The magnitude channel must encode each side's own norm, so that `err_proj` is the
        absolute gap and `dnorm_proj` is the SIGNED one. Encoding the gap directly loses the sign
        — norm_gap would take its absolute value on the way through — and a systematic overshoot
        would then be indistinguishable from symmetric noise."""
        comparison = self.by_channel['norm']
        gap = norm_gap(comparison.estimate, comparison.truth)
        rows = sample_rows(comparison, stride=1)
        np.testing.assert_allclose(rows['dnorm_proj'].to_numpy(), gap.astype(np.float32),
                                   rtol=1e-5)
        np.testing.assert_allclose(rows['err_proj'].to_numpy(),
                                   np.abs(gap).astype(np.float32), rtol=1e-5)
        # Both sides are magnitudes, so both must be positive and near |g| on this fixture.
        self.assertGreater(comparison.estimate[:, 0].min(), 0.0)
        self.assertGreater(comparison.truth[:, 0].min(), 0.0)

    def test_norm_channel_blanks_what_it_cannot_measure(self):
        """A magnitude comparison has no direction and no frame to misalign. Those columns must be
        NaN, not the identical zero the degenerate geometry would produce — a zero there reads as
        perfect agreement in every pooled median that did not know to exclude it."""
        comparison = self.by_channel['norm']
        rows = sample_rows(comparison, stride=1)
        self.assertTrue(rows['ang_proj'].isna().all())
        self.assertTrue(rows['ang_raw'].isna().all())
        stats = stats_row(comparison)
        self.assertTrue(np.isnan(stats['align_angle_deg']))
        self.assertTrue(np.isnan(stats['rms_after']))

    def test_both_sides_move_together_in_the_gyro_table(self):
        """When both signals of a comparison are projections, varying the derivative scheme on one
        side only scores a configuration nothing would ever run. `truth_by_method` is what fixes
        that, so it must be populated here and must match the child's own projections."""
        comparison = self.by_channel['vector']
        self.assertIsNotNone(comparison.truth_by_method)
        self.assertEqual(set(comparison.truth_by_method), set(comparison.by_method))
        np.testing.assert_allclose(comparison.truth_by_method[PRIMARY_GYRO_METHOD],
                                   comparison.truth, atol=1e-12)
        # With both sides on the same scheme, every scheme must agree to its own discretization
        # error — which differs between them by design, hence the loose bound. The tight statement
        # is the primary scheme's, which is the only one any pipeline here runs.
        gyro = ap.gyro_rows(comparison).set_index('gyro_method')
        self.assertLess(float(gyro.loc[PRIMARY_GYRO_METHOD, 'median_err']), 0.15)
        self.assertLess(float(gyro['median_err'].max()), 0.5)


# ==============================================================================
# Family 3: same-segment agreement
# ==============================================================================

class TestSameSegment(unittest.TestCase):
    """Two sensors on one rigid body, projected onto each other."""

    @classmethod
    def setUpClass(cls):
        cls.context = make_context()
        cls.geometry = segment_geometry(cls.context, 'femur_r_h', 'femur_r_l')

    def test_groups_are_read_off_the_display_names(self):
        groups = same_segment_groups(TEST_SPEC)
        self.assertEqual(groups, {'Femur R': [('High', 'femur_r_h'), ('Low', 'femur_r_l')]})
        # A segment with one sensor is not a group, and a spec with no placements has none at all.
        self.assertNotIn('Tibia R', groups)

    def test_geometry_recovers_the_fixture_separation_and_transform(self):
        """r_ab must be the offset from A to B in A's BODY frame — not in the segment frame and not
        the other direction. All four wrong answers are the same length, so this compares vectors."""
        np.testing.assert_allclose(self.geometry['r_ab'], expected_separation(), atol=1e-9)
        self.assertAlmostEqual(self.geometry['sep_m'],
                               float(np.linalg.norm(expected_separation())), places=9)

    def test_both_frame_transforms_recover_the_fixture_and_agree(self):
        """`C_ab` is fitted from the two GYROS and `C_mocap` composed from the two plates' mocap
        rotations. On this noiseless fixture they are the same rotation, and both must equal
        MISALIGN_A^T MISALIGN_B — which pins the DIRECTION of the Wahba fit, the one thing about
        `calculate_best_fit_rotation` that cannot be read off its signature.

        On real IMoVE data they disagree by ~10 deg, which is why the gyro fit is the one used; a
        fixture where they agree is what shows the disagreement is data and not code."""
        np.testing.assert_allclose(self.geometry['C_mocap'], expected_transform(), atol=1e-9)
        np.testing.assert_allclose(self.geometry['C_ab'], expected_transform(), atol=1e-6)
        self.assertLess(self.geometry['C_mocap_vs_gyro_deg'], 1e-3)

    def test_geometry_constancy_diagnostics_read_zero_on_a_rigid_segment(self):
        """These two are the evidence that the shared-cluster structure holds. They must be ~0
        here, and a non-zero reading on real data means the two plates no longer come from one
        cluster and `segment_geometry`'s constancy assumption has stopped holding."""
        self.assertLess(self.geometry['r_spread_mm'], 1e-6)
        self.assertLess(self.geometry['C_spread_max_deg'], 1e-6)

    def test_projecting_a_onto_b_reproduces_b_own_reading(self):
        """Equation (3): A projected onto B's site, brought into A's frame, IS what B measured. The
        sharpest comparison in the module — the truth side is a real accelerometer, so there is no
        differentiated marker trajectory anywhere in it.

        Three wrong conventions are checked alongside, because each produces a plausible residual
        rather than an error: transposing C_ab (transporting the wrong way), negating r_ab
        (projecting away from the partner instead of onto it), and omitting the transport
        altogether — which is the one a fixture with two identical sensor frames could never catch,
        and the reason MISALIGN_A and MISALIGN_B differ.
        """
        C = np.asarray(self.geometry['C_ab'])
        projected = self.context.plates['femur_r_h'].imu_trace.project_acc(
            self.geometry['r_ab'], 'polyfit').acc
        raw_b = self.context.acc('femur_r_l')
        interior = slice(200, -200)

        residual = ap.rms(projected[interior] - (raw_b @ C.T)[interior])
        self.assertLess(residual, 0.1)
        for label, wrong in (('C transposed', raw_b @ C),
                             ('no transport', raw_b),
                             ('r negated', self.context.plates['femur_r_h'].imu_trace.project_acc(
                                 -np.asarray(self.geometry['r_ab']), 'polyfit').acc @ C.T)):
            self.assertGreater(ap.rms(projected[interior] - wrong[interior]), 10 * residual,
                               f"'{label}' is not distinguishable from the correct convention")

    def test_gyro_mismatch_is_zero_on_a_rigid_segment(self):
        """Two sensors on one rigid body measure the SAME angular velocity. This is the rigidity
        floor the report quotes, so it must read zero when the segment really is rigid — otherwise
        the column measures the transform rather than the segment."""
        gyro_a = self.context.gyro('femur_r_h')
        gyro_b_in_a = self.context.gyro('femur_r_l') @ np.asarray(self.geometry['C_ab']).T
        self.assertLess(float(np.abs(gyro_a - gyro_b_in_a).max()), 1e-9)

    def test_inertial_fit_recovers_the_separation_without_mocap(self):
        """The projection is LINEAR in the offset, so the separation is recoverable from the two
        IMUs by ordinary least squares. On noiseless input it must come back to the millimetre,
        which is what makes |r_inertial - r_mocap| on real data a measurement of the mocap geometry
        rather than of this estimator."""
        window = self.context.full_window()
        raw_a = condition(self.context.acc('femur_r_h'), window)
        gyro_a = condition(self.context.gyro('femur_r_h'), window)
        alpha = condition(gyro_derivative(self.context.plates['femur_r_h'].imu_trace,
                                          'polyfit'), window)
        partner = condition(self.context.acc('femur_r_l')
                            @ np.asarray(self.geometry['C_ab']).T, window)

        fit = fit_separation_inertial(raw_a, gyro_a, alpha, partner)
        recovered = np.array([fit['r_inertial_x'], fit['r_inertial_y'], fit['r_inertial_z']])
        # 3 mm on a 158 mm separation, i.e. 2%. Not zero, and the reason is the mechanism the
        # report warns about: the design matrix is built from a DIFFERENTIATED gyro, so its
        # discretization error is errors-in-variables and biases the answer toward zero. Here it
        # recovers 98% of the length, which is the attenuation being measured, not an estimator bug.
        self.assertLess(float(np.linalg.norm(recovered - expected_separation()) * 1000.0), 3.0)
        self.assertLess(fit['attenuation'] if 'attenuation' in fit else 1.0, 1.001)
        self.assertLess(fit['rms_at_inertial'], 0.15)
        # Well conditioned on a fixture that rotates about all three axes; the column exists to
        # warn when a quiet trial leaves it unidentified.
        self.assertLess(fit['design_condition'], 10.0)

    def test_comparisons_cover_both_target_kinds(self):
        """The family must produce a partner target (truth is B's reading) AND a joint target
        (both sensors over a long arm), because they answer different questions and the report
        contrasts them."""
        arms = ap.offsets_by_sensor(self.context)
        comparisons = ap.segment_comparisons(
            self.context, 'Femur R', [('High', 'femur_r_h'), ('Low', 'femur_r_l')], arms,
            detailed=True)
        kinds = {c.target_kind for c in comparisons}
        self.assertIn('partner', kinds)
        self.assertTrue(any(k.startswith('joint:') for k in kinds),
                        f"no joint target among {sorted(kinds)}")
        # The partner target's truth is a raw reading, which no derivative scheme touches, so its
        # truth must NOT vary by method; the joint target's must.
        for comparison in comparisons:
            if comparison.target_kind == 'partner':
                self.assertIsNone(comparison.truth_by_method)
            else:
                self.assertIsNotNone(comparison.truth_by_method)

    def test_agreement_is_exact_and_the_baseline_is_not(self):
        arms = ap.offsets_by_sensor(self.context)
        comparisons = ap.segment_comparisons(
            self.context, 'Femur R', [('High', 'femur_r_h'), ('Low', 'femur_r_l')], arms,
            detailed=True)
        for comparison in comparisons:
            residual = ap.rms(comparison.estimate - comparison.truth)
            baseline = ap.rms(comparison.estimate_raw - comparison.extras['truth_raw'])
            self.assertLess(residual, 0.15, f"{comparison.target_kind} residual {residual}")
            self.assertGreater(baseline, 5 * residual,
                               f"{comparison.target_kind} baseline {baseline} is not a baseline")


# ==============================================================================
# Windows
# ==============================================================================

class TestWindows(unittest.TestCase):
    """Which samples a comparison is scored on. The change that mattered most in this module: the
    previous version trimmed only the ends of the record and scored everything between, including
    the padded constant poses where mocap reconstruction failed."""

    def test_valid_runs_finds_runs_and_drops_short_ones(self):
        mask = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
        self.assertEqual(valid_runs(mask, 1), [(1, 4), (6, 8), (9, 13)])
        self.assertEqual(valid_runs(mask, 3), [(1, 4), (9, 13)])
        self.assertEqual(valid_runs(np.zeros(10, dtype=bool), 1), [])

    def test_window_trims_each_run_and_labels_them(self):
        """The trim is per RUN, not per record: the filter transient and the differentiators'
        padding land at every run edge, not only at the two ends of the trial."""
        n = 3000
        mask = np.zeros(n, dtype=bool)
        mask[100:1200] = True
        mask[2000:2900] = True
        window = build_window(mask, FS, 'mocap')
        self.assertIsNotNone(window)
        trim = window.trim
        self.assertEqual(window.runs, ((100, 1200), (2000, 2900)))
        self.assertEqual(window.index[0], 100 + trim)
        self.assertEqual(window.index[-1], 2900 - trim - 1)
        self.assertEqual(len(window.index), (1100 - 2 * trim) + (900 - 2 * trim))
        self.assertEqual(set(np.unique(window.run_id)), {0, 1})
        # No sample from outside the mask ever survives.
        self.assertTrue(mask[window.index].all())

    def test_window_is_none_when_nothing_survives(self):
        """A run too short to trim is not a short window, it is no window — and the caller reads
        that as 'this comparison is not available on this trial' rather than as an error. The
        biplane trials are ~0.5 s of valid mocap, so this is the ordinary case there."""
        mask = np.zeros(1000, dtype=bool)
        mask[400:450] = True
        self.assertIsNone(build_window(mask, FS, 'mocap'))

    def test_condition_filters_run_by_run_and_never_across_a_gap(self):
        """A filter run across a gap manufactures a transition that was never measured. Two runs
        holding different constants must each come back as their own constant; filtering the
        concatenation instead smears one into the other."""
        n = 1200
        signal = np.zeros((n, 3))
        signal[:500] = 1.0
        signal[700:] = -3.0
        mask = np.zeros(n, dtype=bool)
        mask[:500] = True
        mask[700:] = True
        window = build_window(mask, FS, 'mocap')
        filtered = condition(signal, window)
        first = window.run_id == 0
        np.testing.assert_allclose(filtered[first], 1.0, atol=1e-6)
        np.testing.assert_allclose(filtered[~first], -3.0, atol=1e-6)

    def test_condition_without_a_cutoff_is_the_same_samples_untouched(self):
        """The spectra and the cutoff sweep need an unfiltered copy ON EXACTLY the samples every
        filtered metric used, or they are describing a different window than the one being
        corrected."""
        n = 1200
        signal = np.random.default_rng(0).normal(size=(n, 3))
        window = build_window(np.ones(n, dtype=bool), FS, 'full')
        np.testing.assert_array_equal(condition(signal, window, cutoff=None),
                                      signal[window.index])


# ==============================================================================
# Signal helpers
# ==============================================================================

class TestSignalHelpers(unittest.TestCase):

    def test_residual_alignment_recovers_a_known_rotation(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(size=(500, 3)) + np.array([0.0, 0.0, 9.81])
        rotation = Rotation.from_euler('xyz', [2.0, -1.5, 0.8], degrees=True)
        estimate = reference @ rotation.as_matrix()
        recovered, angle_deg, before, after = residual_alignment(estimate, reference)
        np.testing.assert_allclose(recovered, rotation.as_matrix(), atol=1e-9)
        self.assertAlmostEqual(angle_deg, float(np.degrees(rotation.magnitude())), places=6)
        self.assertGreater(before, 0.1)
        self.assertLess(after, 1e-9)

    def test_residual_alignment_is_a_rotation_not_a_reflection(self):
        """The determinant term in the Kabsch solution is what keeps this a rotation. Without it a
        reflection can win on data that is nearly planar, and a reflection is not a misalignment."""
        rng = np.random.default_rng(2)
        reference = rng.normal(size=(300, 3))
        estimate = reference * np.array([1.0, 1.0, -1.0])   # a reflection, not a rotation
        recovered, _, _, _ = residual_alignment(estimate, reference)
        self.assertAlmostEqual(float(np.linalg.det(recovered)), 1.0, places=9)

    def test_angle_between_deg_on_known_pairs(self):
        a = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        np.testing.assert_allclose(angle_between_deg(a, b), [0.0, 90.0, 180.0], atol=1e-9)

    def test_angle_between_deg_survives_identical_and_zero_vectors(self):
        """Two near-parallel vectors put the normalized dot product a few ulp outside [-1, 1], and
        these signals are near-parallel almost everywhere because both are dominated by gravity.
        A zero-length vector has no direction at all and must give NaN rather than a warning and
        an arbitrary answer."""
        a = np.array([[0.3, -0.7, 9.81], [0.0, 0.0, 0.0]])
        result = angle_between_deg(a, a)
        self.assertAlmostEqual(float(result[0]), 0.0, places=9)
        self.assertTrue(np.isnan(result[1]))

    def test_norm_gap_is_signed_and_as_magnitude_preserves_it(self):
        a = np.array([[3.0, 4.0, 0.0]])     # norm 5
        b = np.array([[1.0, 0.0, 0.0]])     # norm 1
        self.assertAlmostEqual(float(norm_gap(a, b)), 4.0, places=12)
        self.assertAlmostEqual(float(norm_gap(b, a)), -4.0, places=12)
        # _as_magnitude must keep the two sides separate so norm_gap can still see which is larger.
        self.assertAlmostEqual(float(norm_gap(_as_magnitude(a), _as_magnitude(b))), 4.0, places=12)
        self.assertAlmostEqual(float(norm_gap(_as_magnitude(b), _as_magnitude(a))), -4.0, places=12)

    def test_lowpass_keeps_the_signal_and_removes_the_noise_band(self):
        n, fs = 4000, FS
        t = np.arange(n) / fs
        slow = np.column_stack([np.sin(2 * np.pi * 1.0 * t)] * 3)
        fast = np.column_stack([0.5 * np.sin(2 * np.pi * 30.0 * t)] * 3)
        filtered = lowpass(slow + fast, fs)
        interior = slice(400, -400)
        np.testing.assert_allclose(filtered[interior], slow[interior], atol=0.02)

    def test_lowpass_is_zero_lag(self):
        """filtfilt, not lfilter: a phase shift between the two signals of a comparison would show
        up as error, and at 100 Hz one sample of lag on a signal changing at 10 m/s^3 is
        0.1 m/s^2 of pure artifact."""
        n, fs = 4000, FS
        t = np.arange(n) / fs
        signal = np.column_stack([np.sin(2 * np.pi * 2.0 * t)] * 3)
        filtered = lowpass(signal, fs)
        interior = slice(400, -400)
        lag = int(np.argmax([np.corrcoef(signal[interior, 0],
                                         np.roll(filtered[:, 0], shift)[interior])[0, 1]
                             for shift in (-2, -1, 0, 1, 2)]))
        self.assertEqual(lag, 2, "peak correlation is not at zero shift")

    def test_lowpass_passes_through_at_or_above_nyquist(self):
        """IMoVE records 17 of its 24 sensors at 40 Hz, so the sweep's 25 Hz row asks for a cutoff
        above Nyquist on most of that dataset. butter() would raise and clipping to Nyquist would
        report a 20 Hz result in a row labelled 25; returning the signal untouched is what 'no
        filtering above this rate' means, and `cutoff_rows` drops the row rather than quoting it."""
        signal = np.random.default_rng(3).normal(size=(500, 3))
        np.testing.assert_allclose(lowpass(signal, 40.0, cutoff=25.0), signal, atol=0)
        np.testing.assert_allclose(lowpass(signal, 40.0, cutoff=20.0), signal, atol=0)

    def test_lowpass_cutoff_is_below_nyquist_for_the_slowest_dataset(self):
        """A guard on the constant rather than on the code: LOWPASS_CUTOFF_HZ has to be usable on
        the slowest data in the repository, which is IMoVE's 40 Hz sensors."""
        self.assertLess(LOWPASS_CUTOFF_HZ, 0.5 * 40.0)

    def test_rotation_spread_is_zero_for_a_constant_and_grows_with_wobble(self):
        constant = np.repeat(np.eye(3)[None], 100, axis=0)
        rms_deg, max_deg = rotation_spread_deg(constant, chordal_mean(constant))
        self.assertLess(max_deg, 1e-9)

        wobble = Rotation.from_rotvec(
            np.column_stack([np.linspace(-0.02, 0.02, 100), np.zeros(100), np.zeros(100)])
        ).as_matrix()
        rms_deg, max_deg = rotation_spread_deg(wobble, chordal_mean(wobble))
        self.assertGreater(max_deg, 1.0)
        self.assertLess(rms_deg, max_deg)


# ==============================================================================
# The tables
# ==============================================================================

class TestTrialTables(unittest.TestCase):
    """The per-trial tables every figure and the whole report read, on the rigid synthetic trial."""

    @classmethod
    def setUpClass(cls):
        cls.plates = make_trial_plates()
        cls.tables = ap.compute_trial(cls.plates, TEST_SPEC, 'synthetic', 's1', 't1')

    def test_every_expected_table_is_produced_and_non_empty(self):
        for table in ap.TRIAL_TABLES:
            self.assertIn(table, self.tables, f"{table} missing")
            self.assertFalse(self.tables[table].empty, f"{table} is empty")

    def test_all_three_families_are_present(self):
        families = set(self.tables['agreement_samples']['family'].unique())
        self.assertEqual(families, set(ap.FAMILIES))

    def test_error_columns_are_measured_against_the_same_truth(self):
        """err_proj and err_raw must use the SAME truth, or the comparison the whole analysis rests
        on is between two different questions."""
        for comparison in ap.trial_comparisons(make_context()):
            rows = sample_rows(comparison, stride=1)
            truth_raw = comparison.extras.get('truth_raw', comparison.truth)
            np.testing.assert_allclose(
                rows['err_proj'].to_numpy(),
                np.linalg.norm(comparison.estimate - comparison.truth, axis=1).astype(np.float32),
                rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(
                rows['err_raw'].to_numpy(),
                np.linalg.norm(comparison.estimate_raw - truth_raw, axis=1).astype(np.float32),
                rtol=1e-4, atol=1e-4)

    def test_correction_is_the_change_in_the_disagreement(self):
        """`corr_norm` is the x-axis of the error-vs-correction section, and for an agreement family
        it must be how much the projection moved the GAP — not how much it moved either signal. Two
        segments whose corrections move together have been asked to do nothing, however large those
        corrections are, and a per-signal definition would claim otherwise."""
        for comparison in ap.trial_comparisons(make_context()):
            rows = sample_rows(comparison, stride=1)
            truth_raw = comparison.extras.get('truth_raw', comparison.truth)
            expected = np.linalg.norm((comparison.estimate - comparison.estimate_raw)
                                      - (comparison.truth - truth_raw), axis=1)
            np.testing.assert_allclose(rows['corr_norm'].to_numpy(),
                                       expected.astype(np.float32), rtol=1e-4, atol=1e-4)

    # Columns only some families write. Every family's rows are concatenated into one table, so a
    # column one family carries is necessarily NaN in another's rows — that is the cost of one table
    # per experiment rather than three, and it is checked per family below rather than waived.
    FAMILY_COLUMNS = {'err_proj_alt': {'marker'}, 'truth_lin_norm': {'marker'},
                      'jc_residual_norm': {'marker'}, 'partner_corr_norm': {'joint'},
                      'gyro_mismatch': {'segment'}, 'gyro_mismatch_mocap_frame': {'segment'}}

    def test_shared_columns_are_finite_and_family_columns_are_where_they_belong(self):
        """Two claims in one, because they are the same claim from either side: every column every
        family shares must be finite, and every family-specific column must be finite in exactly
        the families that write it and absent everywhere else.

        The second half is what stops a column from silently becoming all-NaN: `gyro_mismatch` is
        the segment family's rigidity floor, and if a refactor stopped populating it the report
        would print an empty median rather than fail."""
        samples = self.tables['agreement_samples']
        shared = [c for c in samples.select_dtypes(include=[np.floating]).columns
                  if c not in self.FAMILY_COLUMNS]

        vector = samples[samples['channel'] == 'vector']
        bad = [c for c in shared if not np.isfinite(vector[c]).all()]
        self.assertFalse(bad, f"non-finite values in shared columns {bad}")

        norm = samples[samples['channel'] == 'norm']
        for column in ('err_proj', 'err_raw', 'dnorm_proj', 'corr_norm'):
            self.assertTrue(np.isfinite(norm[column]).all(), f"{column} non-finite on norm rows")
        # The magnitude channel's angular columns are deliberately NaN; a zero there would read as
        # perfect directional agreement.
        self.assertTrue(norm['ang_proj'].isna().all())

        for column, families in self.FAMILY_COLUMNS.items():
            for family in ap.FAMILIES:
                rows = samples[samples['family'] == family]
                finite = np.isfinite(rows[column]).any()
                self.assertEqual(finite, family in families,
                                 f"{column} is {'present' if finite else 'absent'} in "
                                 f"family={family}, expected the opposite")

    def test_stats_reports_no_misalignment_floor_on_perfectly_aligned_data(self):
        """The marker family's synthetic sensor frames ARE the segment frames, so the fitted
        residual rotation must come out near zero. On real data that angle is the sensor-to-segment
        misalignment; here there is none, so a non-trivial angle means the Kabsch fit is absorbing
        something it should not."""
        stats = self.tables['agreement_stats']
        marker = stats[(stats['family'] == 'marker')]
        self.assertTrue((marker['align_angle_deg'] < 0.5).all())
        self.assertTrue((marker['r2_proj'] > 0.99).all())
        self.assertTrue((marker['r2_proj'] > marker['r2_raw']).all())

    def test_projection_beats_its_baseline_in_every_comparison(self):
        """The claim, stated as the weakest form that must hold on noiseless data: for every
        comparison in every family, the projected signal is closer to that family's truth than the
        unprojected one is."""
        stats = self.tables['agreement_stats']
        worse = stats[stats['rms_before'] >= stats['rms_raw']]
        self.assertTrue(worse.empty,
                        f"projection did not help in:\n{worse[['family', 'group', 'variant']]}")

    def test_traces_are_one_comparison_per_family_and_bounded(self):
        """`traces` is the only full-rate table, so it is the only one whose size is set by trial
        length. Storing every detailed comparison was 160 MB per Al Borno trial."""
        traces = self.tables['traces']
        per_family = traces.groupby('family', observed=True).size()
        self.assertLessEqual(len(per_family), len(ap.FAMILIES))
        self.assertEqual(len(traces.groupby(['family', 'group', 'variant', 'scope'],
                                            observed=True)), len(per_family))
        self.assertLessEqual(int(per_family.max()), int(round(ap.TRACE_MAX_S * FS)))

    def test_spectra_and_sweep_skip_the_magnitude_channel(self):
        """A power spectrum of a rectified scalar on one axis is neither the estimate's spectrum nor
        its bandwidth, so neither table should contain the magnitude channel. It IS kept in
        gyro_method, where scoring the derivative against a magnitude uses no mocap orientation."""
        for table in ('spectra', 'cutoff_sweep', 'traces'):
            self.assertEqual(set(self.tables[table]['channel'].unique()), {'vector'}, table)
        self.assertIn('norm', set(self.tables['gyro_method']['channel'].unique()))

    def test_sweep_drops_cutoffs_at_or_above_nyquist(self):
        sweep = self.tables['cutoff_sweep']
        self.assertTrue((sweep['cutoff_hz'] < 0.5 * FS).all())
        self.assertEqual(sorted(sweep['cutoff_hz'].unique()),
                         [c for c in ap.CUTOFF_SWEEP_HZ if c < 0.5 * FS])

    def test_summary_keeps_the_channels_apart(self):
        """Pooling a 3-vector residual together with a magnitude gap under one label produces a
        number that is neither, so `channel` has to be a grouping key and not a note."""
        # `summarize` reads the subject/trial columns `load_trial_table` adds on the way in, which
        # `compute_trial` does not write (they are the artifact's path, not its content).
        summary = ap.summarize('synthetic', self.tables['agreement_samples'].assign(
            subject='s1', trial='t1'))
        self.assertFalse(summary.empty)
        self.assertEqual(set(summary.columns), set(ap.SUMMARY_COLUMNS))
        joint = summary[(summary['family'] == 'joint') & (summary['metric'] == 'err_proj')
                        & (summary['group'] == 'all') & (summary['subject'] == 'all')]
        self.assertEqual(set(joint['channel']), set(CHANNELS))

    def test_report_runs_on_a_synthetic_trial(self):
        """A smoke test on the whole report, because a section that divides by a median of zero or
        indexes a column a family does not have would otherwise only fail on real data."""
        samples = self.tables['agreement_samples'].assign(subject='s1', trial='t1')
        stats = self.tables['agreement_stats'].assign(subject='s1', trial='t1')
        sweep = self.tables['cutoff_sweep'].assign(subject='s1', trial='t1')
        gyro = self.tables['gyro_method'].assign(subject='s1', trial='t1')
        summary = ap.summarize('synthetic', samples)
        ap.print_report(TEST_SPEC, summary, samples, stats, sweep, gyro)


if __name__ == '__main__':
    unittest.main()
