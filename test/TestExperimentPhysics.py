"""
Covers the physics helpers in experiment_utils that turn ground truth into filter
inputs: the acc/mag oracles, the virtual EKF ground plate, and the observability
metric that drives mag_adapt.

Every one of these is a frame conversion or a finite difference, which means the two
ways to get them wrong — a transposed rotation and a flipped sign — both produce
output of exactly the right shape and a plausible magnitude. Nothing downstream
raises; the filters converge and the joint angles look fine. So these tests build
synthetic plates whose true answer is known in closed form and check the values, not
just the shapes.

Synthetic throughout — no dataset needed. The dataset-facing counterpart to this is
test/TestGravityConvention.py, which checks EXPECTED_GRAVITY against real recordings.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
from scipy.spatial.transform import Rotation

from experiments import experiment_utils
from experiments.experiment_utils import (EXPECTED_GRAVITY, _calculate_observability_metric_,
                                          _compute_expected_mag_field, _compute_perfect_joint_acc,
                                          _compute_perfect_mag, _compute_perfect_segment_acc,
                                          _setup_ekf_ground_plate_, segment_observability)
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace

DT = 0.01
WORLD_MAG = np.array([20.0, -5.0, 40.0])

# The sensor location _compute_expected_mag_field takes the world magnetic field from,
# matched by substring on the plate name, and any other location for contrast.
REFERENCE_SENSOR = 'torso_imu'
OTHER_SENSOR = 'femur_r_imu'


def make_plate(name: str = 'pelvis_imu', n: int = 200, dt: float = DT,
               omega: np.ndarray = np.array([0.3, -0.7, 2.0]),
               positions: np.ndarray = None, world_acc: np.ndarray = None,
               world_mag: np.ndarray = WORLD_MAG) -> PlateTrial:
    """A plate rotating at a constant world-frame angular velocity, carrying the IMU
    signals that rotation implies: the world-frame acceleration and magnetic field seen
    in the body frame, and the matching body-frame gyro.

    `world_acc` is the linear part of the accelerometer's world-frame reading, on top of
    gravity; the default None means gravity alone. `positions` only moves the mocap
    trace — the two are set independently so a test can exercise one without the other,
    and make_shaking_plate below sets them consistently.
    """
    timestamps = np.arange(n) * dt
    rotations = Rotation.from_rotvec(np.outer(timestamps, omega)).as_matrix()
    positions = np.zeros((n, 3)) if positions is None else positions

    world_reading = EXPECTED_GRAVITY if world_acc is None else EXPECTED_GRAVITY + world_acc
    acc = np.einsum('nji,nj->ni', rotations, np.broadcast_to(world_reading, (n, 3)))
    mag = np.einsum('nji,j->ni', rotations, world_mag)
    gyro = np.einsum('nji,j->ni', rotations, omega)  # world-frame omega, in the body frame

    return PlateTrial(name, IMUTrace(timestamps, gyro, acc, mag),
                      WorldTrace(timestamps, positions, rotations))


def make_shaking_plate(name: str = 'pelvis_imu', n: int = 300, dt: float = DT,
                       omega: np.ndarray = np.array([0.0, 0.0, 2.0]),
                       amplitude: float = 1.0, rate: float = 3.0) -> PlateTrial:
    """A rotating plate that is also translating sinusoidally along world x, so its
    accelerometer really does change direction in the world frame — the observable case.
    Position and acceleration are the same motion differentiated twice, so the mocap
    trace and the IMU trace agree.
    """
    timestamps = np.arange(n) * dt
    zeros = np.zeros(n)
    positions = np.column_stack([amplitude * np.sin(rate * timestamps), zeros, zeros])
    world_acc = np.column_stack([-amplitude * rate ** 2 * np.sin(rate * timestamps),
                                 zeros, zeros])
    return make_plate(name, n=n, dt=dt, omega=omega, positions=positions,
                      world_acc=world_acc)


class TestPerfectSegmentAcc(unittest.TestCase):
    """The EKF acc oracle: gravity in the segment's own frame, no linear term."""

    def setUp(self):
        self.plate = make_plate()

    def test_rotating_the_oracle_back_to_the_world_recovers_gravity(self):
        """The one property that fixes both the transpose and the sign at once."""
        acc = _compute_perfect_segment_acc(self.plate)
        world = np.einsum('nij,nj->ni', self.plate.world_trace.rotations, acc)
        np.testing.assert_allclose(world, np.tile(EXPECTED_GRAVITY, (len(self.plate), 1)), atol=1e-9)

    def test_it_matches_a_hand_computed_asymmetric_rotation(self):
        """A rotation that is not its own inverse, so R and R.T give different answers:
        90 deg about +x sends world +y (up, where gravity reads +9.81) to body -z."""
        plate = make_plate(n=1, omega=np.zeros(3))
        plate.world_trace.rotations = Rotation.from_euler('x', 90, degrees=True).as_matrix()[None]
        np.testing.assert_allclose(_compute_perfect_segment_acc(plate)[0],
                                   np.array([0.0, 0.0, -9.81]), atol=1e-9)

    def test_it_carries_no_linear_acceleration(self):
        """Even for a segment that is genuinely accelerating through space, the EKF
        oracle is gravity only — its magnitude is |g| at every sample."""
        n = 200
        t = np.arange(n) * DT
        positions = np.column_stack([2.0 * t ** 2, np.zeros(n), np.sin(4.0 * t)])
        plate = make_plate(n=n, positions=positions)
        acc = _compute_perfect_segment_acc(plate)
        np.testing.assert_allclose(np.linalg.norm(acc, axis=1), np.linalg.norm(EXPECTED_GRAVITY),
                                   atol=1e-9)

    def test_the_gravity_argument_is_resolved_at_call_time(self):
        """The docstring promises the module constant is read per call, not frozen as a
        default argument. A frozen default would make EXPECTED_GRAVITY untestable and
        would silently ignore any override."""
        patched = np.array([0.0, 0.0, -1.0])
        original = experiment_utils.EXPECTED_GRAVITY
        experiment_utils.EXPECTED_GRAVITY = patched
        self.addCleanup(setattr, experiment_utils, 'EXPECTED_GRAVITY', original)

        acc = _compute_perfect_segment_acc(self.plate)
        world = np.einsum('nij,nj->ni', self.plate.world_trace.rotations, acc)
        np.testing.assert_allclose(world, np.tile(patched, (len(self.plate), 1)), atol=1e-9)

    def test_an_explicit_gravity_argument_wins(self):
        gravity = np.array([1.0, 2.0, 3.0])
        acc = _compute_perfect_segment_acc(self.plate, gravity=gravity)
        world = np.einsum('nij,nj->ni', self.plate.world_trace.rotations, acc)
        np.testing.assert_allclose(world, np.tile(gravity, (len(self.plate), 1)), atol=1e-9)

    def test_it_reproduces_the_synthetic_accelerometer(self):
        """make_plate builds its accelerometer from gravity alone, so for a stationary
        plate the oracle must reproduce the real reading exactly. This is the sanity
        check that the oracle and the forward model share a convention."""
        np.testing.assert_allclose(_compute_perfect_segment_acc(self.plate),
                                   self.plate.imu_trace.acc, atol=1e-9)


class TestPerfectMag(unittest.TestCase):
    def test_rotating_the_oracle_back_to_the_world_recovers_the_field(self):
        plate = make_plate()
        mag = _compute_perfect_mag(plate, WORLD_MAG)
        world = np.einsum('nij,nj->ni', plate.world_trace.rotations, mag)
        np.testing.assert_allclose(world, np.tile(WORLD_MAG, (len(plate), 1)), atol=1e-9)

    def test_it_reproduces_the_synthetic_magnetometer(self):
        plate = make_plate()
        np.testing.assert_allclose(_compute_perfect_mag(plate, WORLD_MAG),
                                   plate.imu_trace.mag, atol=1e-9)

    def test_it_uses_the_same_convention_as_the_acc_oracle(self):
        """Both oracles feed the same filter through the same override slots. If one
        were transposed relative to the other, the filter would see a self-inconsistent
        pair of reference vectors."""
        plate = make_plate()
        np.testing.assert_allclose(_compute_perfect_mag(plate, EXPECTED_GRAVITY),
                                   _compute_perfect_segment_acc(plate), atol=1e-9)


class TestExpectedMagField(unittest.TestCase):
    """The world magnetic field reference, taken from one nominated sensor location.
    Every test here selects plates by REFERENCE_SENSOR rather than by a literal, so
    moving the reference (it was the pelvis before it was the torso) is a one-line
    change here and not a rewrite."""

    def test_it_recovers_the_world_field_from_the_reference_sensor(self):
        np.testing.assert_allclose(_compute_expected_mag_field([make_plate(REFERENCE_SENSOR)]),
                                   WORLD_MAG, atol=1e-9)

    def test_only_the_reference_sensor_contributes(self):
        """The median is taken over the reference location only. A wildly disturbed
        sensor elsewhere on the body must not move the world reference."""
        reference = make_plate(REFERENCE_SENSOR)
        other = make_plate(OTHER_SENSOR, world_mag=np.array([500.0, 500.0, 500.0]))
        np.testing.assert_allclose(_compute_expected_mag_field([reference, other]),
                                   WORLD_MAG, atol=1e-9)

    def test_it_is_a_median_so_a_burst_of_disturbance_is_rejected(self):
        reference = make_plate(REFERENCE_SENSOR, n=201)
        reference.imu_trace.mag[:80] = np.array([300.0, 300.0, 300.0])  # < half the samples
        measured = _compute_expected_mag_field([reference])
        self.assertLess(float(np.linalg.norm(measured - WORLD_MAG)), 1.0)

    def test_no_reference_sensor_is_not_silently_zero(self):
        """Nothing to take a median over. This has to fail loudly rather than pass as a
        zero field, which the filter would happily accept as a magnetometer reference."""
        with np.errstate(all='ignore'):
            with self.assertRaises((ValueError, IndexError)):
                _compute_expected_mag_field([make_plate(OTHER_SENSOR)])


class TestEkfGroundPlate(unittest.TestCase):
    """The virtual parent the EKF path uses to turn the relative filter into an
    absolute one: a plate bolted to the world frame, reading gravity and the world
    field with no rotation of its own."""

    def setUp(self):
        self.plates = [make_plate(REFERENCE_SENSOR), make_plate(OTHER_SENSOR)]
        self.ground = _setup_ekf_ground_plate_(self.plates)

    def test_it_is_stationary_in_the_world_frame(self):
        np.testing.assert_allclose(self.ground.world_trace.rotations,
                                   np.tile(np.eye(3), (len(self.plates[0]), 1, 1)), atol=1e-12)
        np.testing.assert_allclose(self.ground.imu_trace.gyro, 0.0, atol=1e-12)

    def test_its_accelerometer_is_expected_gravity_in_the_world_frame(self):
        """With identity rotations the body frame *is* the world frame, so the ground
        plate's acc must be EXPECTED_GRAVITY verbatim. This is where a sign error in the
        constant enters the EKF's reference — see TestGravityConvention."""
        np.testing.assert_allclose(self.ground.imu_trace.acc,
                                   np.tile(EXPECTED_GRAVITY, (len(self.plates[0]), 1)), atol=1e-12)

    def test_its_magnetometer_is_the_expected_world_field(self):
        np.testing.assert_allclose(self.ground.imu_trace.mag,
                                   np.tile(WORLD_MAG, (len(self.plates[0]), 1)), atol=1e-9)

    def test_it_does_not_mutate_the_plate_it_was_built_from(self):
        """It is built by copying plates[0]. An in-place build would zero the gyro and
        overwrite the acc of a real segment, corrupting every joint computed after it."""
        base = self.plates[0]
        self.assertNotEqual(self.ground.name, base.name)
        self.assertGreater(float(np.abs(base.imu_trace.gyro).max()), 0.1)
        self.assertGreater(float(np.abs(base.world_trace.rotations - np.eye(3)).max()), 0.1)

    def test_it_is_named_ground(self):
        self.assertEqual(self.ground.name, 'ground')

    def test_it_is_the_same_length_as_the_trial(self):
        self.assertEqual(len(self.ground), len(self.plates[0]))


class TestPerfectJointAcc(unittest.TestCase):
    """The relative-filter acc oracle. Unlike the EKF one this keeps the linear term,
    since the shared joint center really does translate."""

    def _coincident_pair(self, n=200):
        """Parent and child origins both sit exactly on the joint center, so the fitted
        offsets are zero and the joint trajectory is just the shared position. Keeps the
        test independent of get_joint_center's offset sign convention."""
        t = np.arange(n) * DT
        positions = np.column_stack([0.5 * t ** 2, 0.3 * t ** 2, -0.2 * t ** 2])
        parent = make_plate('pelvis_imu', n=n, positions=positions, omega=np.array([0.0, 0.0, 1.5]))
        child = make_plate('femur_r_imu', n=n, positions=positions.copy(),
                           omega=np.array([0.5, 1.0, -0.5]))
        return parent, child, positions

    def test_it_equals_the_true_linear_acceleration_plus_gravity(self):
        """Positions are quadratic in t, so the true world acceleration is the constant
        [1.0, 0.6, -0.4]. Interior samples only — central differencing is one-sided at
        the two ends."""
        parent, child, _ = self._coincident_pair()
        expected_world = np.array([1.0, 0.6, -0.4]) + EXPECTED_GRAVITY

        acc_parent, acc_child = _compute_perfect_joint_acc(parent, child)
        for plate, acc in ((parent, acc_parent), (child, acc_child)):
            world = np.einsum('nij,nj->ni', plate.world_trace.rotations, acc)
            np.testing.assert_allclose(world[3:-3], np.tile(expected_world, (len(world) - 6, 1)),
                                       atol=1e-6)

    def test_both_segments_see_one_shared_world_vector(self):
        """The point of a joint oracle: one acceleration at the joint center, expressed
        in each segment's frame. Rotating both back must give the same world vector even
        though the two segments have different orientations."""
        parent, child, _ = self._coincident_pair()
        acc_parent, acc_child = _compute_perfect_joint_acc(parent, child)
        world_parent = np.einsum('nij,nj->ni', parent.world_trace.rotations, acc_parent)
        world_child = np.einsum('nij,nj->ni', child.world_trace.rotations, acc_child)
        np.testing.assert_allclose(world_parent, world_child, atol=1e-9)

    def test_a_stationary_joint_reduces_to_gravity_alone(self):
        parent, child, _ = self._coincident_pair()
        for plate in (parent, child):
            plate.world_trace.positions = np.zeros_like(plate.world_trace.positions)
        acc_parent, _ = _compute_perfect_joint_acc(parent, child)
        world = np.einsum('nij,nj->ni', parent.world_trace.rotations, acc_parent)
        np.testing.assert_allclose(world, np.tile(EXPECTED_GRAVITY, (len(world), 1)), atol=1e-9)

    def test_the_gravity_argument_is_resolved_at_call_time(self):
        parent, child, _ = self._coincident_pair()
        for plate in (parent, child):
            plate.world_trace.positions = np.zeros_like(plate.world_trace.positions)

        patched = np.array([0.0, 0.0, -1.0])
        original = experiment_utils.EXPECTED_GRAVITY
        experiment_utils.EXPECTED_GRAVITY = patched
        self.addCleanup(setattr, experiment_utils, 'EXPECTED_GRAVITY', original)

        acc_parent, _ = _compute_perfect_joint_acc(parent, child)
        world = np.einsum('nij,nj->ni', parent.world_trace.rotations, acc_parent)
        np.testing.assert_allclose(world, np.tile(patched, (len(world), 1)), atol=1e-9)


class TestObservabilityMetric(unittest.TestCase):
    """o^J, the quantity mag_adapt thresholds at 150.0 to decide whether to trust the
    magnetometer. It is |a x da| per segment, minimised over the two segments."""

    def setUp(self):
        self.parent = make_plate('pelvis_imu')
        self.child = make_plate('femur_r_imu', omega=np.array([0.1, 0.2, 0.4]))

    @staticmethod
    def _static_plate(name='pelvis_imu', n=50):
        timestamps = np.arange(n) * DT
        return PlateTrial(name,
                          IMUTrace(timestamps, np.zeros((n, 3)),
                                   np.tile(EXPECTED_GRAVITY, (n, 1)), np.tile(WORLD_MAG, (n, 1))),
                          WorldTrace(timestamps, np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1))))

    def test_it_returns_one_value_per_sample(self):
        """np.diff drops a sample and the result is padded back to length. If the pad
        were ever dropped, the boolean mask in the mag_adapt branch would misalign with
        the trace by one sample and silently zero the wrong magnetometer readings."""
        metric = _calculate_observability_metric_(self.parent, self.child)
        self.assertEqual(metric.shape, (len(self.parent),))

    def test_the_first_sample_is_zero(self):
        """There is no difference available at t=0, so it is padded with 0 — i.e. the
        first sample always counts as unobservable and keeps its magnetometer."""
        self.assertEqual(_calculate_observability_metric_(self.parent, self.child)[0], 0.0)

    def test_it_is_zero_for_a_completely_static_pair(self):
        static = self._static_plate()
        np.testing.assert_allclose(_calculate_observability_metric_(static, static), 0.0, atol=1e-12)

    def test_it_is_never_negative(self):
        metric = _calculate_observability_metric_(self.parent, self.child)
        self.assertGreaterEqual(float(metric.min()), 0.0)

    def test_it_takes_the_minimum_of_the_two_segments(self):
        """A joint is only as observable as its worse-conditioned side. One static
        segment must pin the whole joint to zero however much the other one moves."""
        shaking = make_shaking_plate(n=len(self.parent))
        static = self._static_plate(n=len(self.parent))
        np.testing.assert_allclose(_calculate_observability_metric_(shaking, static), 0.0,
                                   atol=1e-12)
        # For scale: this synthetic shake peaks in the low hundreds, the same order as
        # the median of a real walking trial.
        self.assertGreater(float(_calculate_observability_metric_(shaking, shaking).max()),
                           100.0)

    def test_it_is_symmetric_in_its_arguments(self):
        """Nothing about o^J distinguishes parent from child, so the joint's value must
        not depend on which sensor was named first."""
        np.testing.assert_allclose(_calculate_observability_metric_(self.parent, self.child),
                                   _calculate_observability_metric_(self.child, self.parent),
                                   atol=1e-12)

    def test_it_is_invariant_to_the_sensor_mounting_frame(self):
        """Rotating a sensor's whole body frame by a constant rotation (a different
        mounting orientation for the same physical motion) leaves the metric alone,
        since it is built from cross products of that sensor's own vectors."""
        rotation = Rotation.from_euler('xyz', [25.0, -40.0, 70.0], degrees=True).as_matrix()
        rotated = self.parent.copy()
        rotated.imu_trace.acc = self.parent.imu_trace.acc @ rotation.T
        rotated.imu_trace.gyro = self.parent.imu_trace.gyro @ rotation.T

        np.testing.assert_allclose(_calculate_observability_metric_(rotated, rotated),
                                   _calculate_observability_metric_(self.parent, self.parent),
                                   rtol=1e-9, atol=1e-9)

    def test_pure_rotation_under_gravity_is_unobservable(self):
        """THE test for this metric, and the one the pre-fix version failed.

        A plate spinning at a constant rate with gravity as its only acceleration is the
        maximally UNobservable case: the accelerometer vector does sweep around the body
        frame, but the sweep is entirely predicted by the gyro, so it carries no
        independent orientation information. The two terms of da must cancel and o^J must
        collapse to ~0, several orders of magnitude below the 1000 gating threshold.

        Before the missing dt was fixed, diff(a) came in ~100x too small to cancel the
        w x a term, this same case scored ~190, and it cleared the then-threshold of 150.
        """
        parent = make_plate(n=300, omega=np.array([0.0, 0.0, 2.0]))
        metric = _calculate_observability_metric_(parent, parent)

        self.assertLess(float(metric.max()), 1.0)
        self.assertLess(float(metric.max()), 0.001 * 1000.0)

    def test_it_is_not_merely_an_angular_rate_detector(self):
        """The specific degenerate form the pre-fix metric collapsed to was
        |a|^2 |w_perp| — a function of angular rate alone, blind to whether the
        acceleration was actually changing in the world frame. Two plates with the SAME
        angular rate and different linear acceleration must now score differently."""
        omega = np.array([0.0, 0.0, 2.0])
        spinning = make_plate(n=300, omega=omega)
        shaking = make_shaking_plate(n=300, omega=omega, rate=6.0)

        # Same angular rate, so the degenerate |a|^2 |w_perp| form would score them alike.
        np.testing.assert_allclose(spinning.imu_trace.gyro, shaking.imu_trace.gyro, atol=1e-12)
        self.assertGreater(float(np.abs(spinning.imu_trace.gyro).max()), 1.0)

        quiet = _calculate_observability_metric_(spinning, spinning)[1:]
        moving = _calculate_observability_metric_(shaking, shaking)[1:]
        self.assertGreater(float(moving.mean()), 100.0 * float(quiet.mean() + 1e-12))

    def test_it_is_invariant_to_the_sampling_rate(self):
        """Identical physics sampled at 50 Hz and 200 Hz must give the same metric. The
        pre-fix version did not: its difference term scaled with the sample interval, so
        the same motion scored differently at different rates."""
        means = []
        for dt in (0.02, 0.01, 0.005):
            plate = make_shaking_plate(n=int(2.0 / dt), dt=dt)
            means.append(float(_calculate_observability_metric_(plate, plate)[2:-2].mean()))

        np.testing.assert_allclose(means, means[0], rtol=0.05)

    def test_the_difference_term_is_divided_by_the_real_sample_interval(self):
        """Not by a hardcoded 0.01. The metric reads dt from the trace's own timestamps,
        so a trial recorded at another rate is scaled by its own interval."""
        n = 200
        t = np.arange(n) * 0.004  # 250 Hz
        acc = np.column_stack([np.linspace(0.0, 2.0, n), np.zeros(n), np.full(n, 9.81)])
        plate = PlateTrial('pelvis_imu',
                           IMUTrace(t, np.zeros((n, 3)), acc, np.tile(WORLD_MAG, (n, 1))),
                           WorldTrace(t, np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1))))
        # a_dot is 2.0/(n-1) per sample over a 0.004 s interval, along x; a is ~[.,0,9.81]
        expected_rate = (2.0 / (n - 1)) / 0.004
        metric = _calculate_observability_metric_(plate, plate)[1:]
        expected = np.linalg.norm(np.cross(acc[1:], np.column_stack(
            [np.full(n - 1, expected_rate), np.zeros(n - 1), np.zeros(n - 1)])), axis=1)
        np.testing.assert_allclose(metric, expected, rtol=1e-9)

    def test_it_is_the_min_of_two_independent_per_segment_values(self):
        """_calculate_observability_metric_ is exactly the elementwise min of
        segment_observability over the two sensors — the pair helper adds no coupling."""
        np.testing.assert_allclose(
            _calculate_observability_metric_(self.parent, self.child),
            np.minimum(segment_observability(self.parent.imu_trace),
                       segment_observability(self.child.imu_trace)),
            atol=1e-12)


if __name__ == '__main__':
    unittest.main()
