import os
import unittest

from test.fixtures import require_data

import numpy as np
from scipy.spatial.transform import Rotation

import paths
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from src.toolchest.building import alborno, assembly
from test import fixtures

# The loaders print progress unless this is set; keep the test output readable.
os.environ.setdefault("DISABLE_TQDM", "True")


def make_plate(name='plate', num_samples=50, rotations=None, positions=None):
    timestamps = np.arange(num_samples) / 100.0
    if rotations is None:
        rotations = np.array([np.eye(3)] * num_samples)
    if positions is None:
        positions = np.zeros((num_samples, 3))
    world_trace = WorldTrace(timestamps, positions, rotations)
    imu_trace = IMUTrace(timestamps, np.zeros((num_samples, 3)), np.zeros((num_samples, 3)),
                         np.zeros((num_samples, 3)))
    return PlateTrial(name, imu_trace, world_trace)


class TestPlateTrialBasics(unittest.TestCase):
    def test_rejects_unsynchronized_traces(self):
        num_samples = 10
        world_trace = WorldTrace(np.arange(num_samples) / 100.0, np.zeros((num_samples, 3)),
                                 np.array([np.eye(3)] * num_samples))
        shifted_imu = IMUTrace(np.arange(num_samples) / 100.0 + 1.0, np.zeros((num_samples, 3)),
                               np.zeros((num_samples, 3)), np.zeros((num_samples, 3)))
        with self.assertRaises(AssertionError):
            PlateTrial('bad', shifted_imu, world_trace)

    def test_rejects_mismatched_lengths(self):
        world_trace = WorldTrace(np.arange(10) / 100.0, np.zeros((10, 3)), np.array([np.eye(3)] * 10))
        short_imu = IMUTrace(np.arange(5) / 100.0, np.zeros((5, 3)), np.zeros((5, 3)), np.zeros((5, 3)))
        with self.assertRaises(AssertionError):
            PlateTrial('bad', short_imu, world_trace)

    def test_slicing(self):
        plate = make_plate(num_samples=50)
        sliced = plate[10:20]
        self.assertEqual(len(sliced), 10)
        np.testing.assert_allclose(sliced.imu_trace.timestamps, np.arange(10, 20) / 100.0)
        with self.assertRaises(TypeError):
            _ = plate[5]

    def test_copy_is_independent(self):
        plate = make_plate()
        copied = plate.copy()
        copied.imu_trace.acc[0] = np.array([1., 2., 3.])
        np.testing.assert_array_equal(plate.imu_trace.acc[0], np.zeros(3))

    def test_get_imu_trace_in_global_frame(self):
        num_samples = 40
        rotations = Rotation.from_rotvec(
            np.linspace(0, np.pi / 2, num_samples)[:, None] * np.array([0., 0., 1.])).as_matrix()
        plate = make_plate(num_samples=num_samples, rotations=rotations)

        # A constant world-frame gravity vector, as each sensor would measure it locally.
        world_gravity = np.array([0., 0., 9.81])
        plate.imu_trace.acc = np.einsum('nji,j->ni', rotations, world_gravity)
        plate.imu_trace.gyro = np.tile(np.array([1., 0., 0.]), (num_samples, 1))

        global_trace = plate.get_imu_trace_in_global_frame()

        np.testing.assert_allclose(global_trace.acc, np.tile(world_gravity, (num_samples, 1)), atol=1e-12)
        np.testing.assert_allclose(global_trace.gyro,
                                   np.einsum('nij,j->ni', rotations, np.array([1., 0., 0.])), atol=1e-12)

    def test_project_imu_trace_matches_project_acc(self):
        np.random.seed(6)
        num_samples = 40
        plate = make_plate(num_samples=num_samples)
        plate.imu_trace.gyro = np.random.randn(num_samples, 3)
        plate.imu_trace.acc = np.random.randn(num_samples, 3)
        offset = np.array([0.02, -0.03, 0.01])

        projected = plate.project_imu_trace(offset)
        self.assertTrue(projected.allclose(plate.imu_trace.project_acc(offset)))

    def test_align_world_trace_to_imu_trace_recovers_a_static_offset(self):
        # The sensor is mounted on the plate at a fixed unknown rotation. Alignment should
        # rotate the mocap frame onto the sensor frame using the gyros alone.
        np.random.seed(7)
        num_samples = 300
        timestamps = np.arange(num_samples) / 100.0
        angle = np.column_stack([
            0.9 * np.sin(2 * np.pi * 0.7 * timestamps),
            0.6 * np.sin(2 * np.pi * 1.1 * timestamps + 1.0),
            0.4 * np.sin(2 * np.pi * 1.9 * timestamps + 2.0),
        ])
        world_rotations = Rotation.from_rotvec(angle).as_matrix()
        world_trace = WorldTrace(timestamps, np.zeros((num_samples, 3)), world_rotations)

        R_plate_sensor = Rotation.from_euler('XYZ', [0.3, -0.7, 0.2]).as_matrix()
        synthetic = world_trace.calculate_imu_trace(np.array([0., 0., 9.81]), skip_lin_acc=True)
        sensor_gyro = np.einsum('ji,nj->ni', R_plate_sensor, synthetic.gyro)
        imu_trace = IMUTrace(timestamps, sensor_gyro, synthetic.acc, synthetic.mag)

        aligned = assembly.align_world_to_imu(PlateTrial('plate', imu_trace, world_trace))

        expected = np.matmul(world_rotations, R_plate_sensor)
        np.testing.assert_allclose(aligned.world_trace.rotations, expected, atol=1e-6)


class TestAlignmentOverRaggedValidity(unittest.TestCase):
    """align_world_to_imu fits over VALID frames only, and that is the whole argument.

    The existing test uses a clean plate where every frame is valid, so it only ever exercises
    the `valid.all()` fast path. The docstring's reasoning is entirely about the other one: on
    Subject01 the padded stretches hold a constant pose, so their synthetic gyro is zero while
    the sensor is still moving, and fitting over them pits 430 s of "the segment did not
    rotate" against 603 s of real overlap.
    """

    def setUp(self):
        rate, self.n = 100.0, 1200
        timestamps = np.arange(self.n) / rate
        angles = np.column_stack([
            0.9 * np.sin(2 * np.pi * 0.8 * timestamps),
            0.6 * np.sin(2 * np.pi * 1.3 * timestamps),
            0.4 * np.sin(2 * np.pi * 2.1 * timestamps)])
        rotations = Rotation.from_euler('zyx', angles).as_matrix()

        # A real sensor-to-segment rotation to recover.
        self.R_plate_sensor = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
        world = WorldTrace(timestamps, np.zeros((self.n, 3)), rotations)
        synthetic = world.calculate_imu_trace(skip_lin_acc=True)
        self.imu = IMUTrace(timestamps,
                            synthetic.gyro @ self.R_plate_sensor,
                            synthetic.acc @ self.R_plate_sensor,
                            synthetic.mag)

        # The first two thirds are CORRUPT: poses that still move, but wrongly. That is the
        # case masking actually protects against. Held padding turns out to be inert for this
        # estimator -- a Procrustes fit accumulates H = sum a_i b_i^T, and a held pose has
        # zero synthetic gyro, so those rows contribute nothing whether masked or not.
        # Interpolated frames from a failed reconstruction do rotate, and they bias it.
        corrupt = Rotation.from_euler('zyx', angles[:, [2, 0, 1]] * 1.7).as_matrix()
        ragged_rotations = rotations.copy()
        ragged_rotations[:800] = corrupt[:800]
        valid = np.zeros(self.n, dtype=bool)
        valid[800:] = True
        self.ragged = WorldTrace(timestamps, np.zeros((self.n, 3)), ragged_rotations,
                                 valid=valid)
        self.truth = WorldTrace(timestamps[800:], np.zeros((self.n - 800, 3)),
                                rotations[800:])

    def test_the_padding_does_not_drag_the_recovered_rotation(self):
        aligned = assembly.align_world_to_imu(PlateTrial('plate', self.imu, self.ragged))

        # Over the valid stretch the aligned world must reproduce the sensor frame.
        expected = np.matmul(self.ragged.rotations[800:], self.R_plate_sensor)
        np.testing.assert_allclose(aligned.world_trace.rotations[800:], expected, atol=1e-6)

    def test_it_matches_fitting_the_valid_stretch_alone(self):
        """The strongest form: masking must give the same answer as never having had the
        padding at all."""
        ragged = assembly.align_world_to_imu(PlateTrial('plate', self.imu, self.ragged))
        clean = assembly.align_world_to_imu(
            PlateTrial('plate', self.imu[800:], self.truth))

        np.testing.assert_allclose(ragged.world_trace.rotations[800:],
                                   clean.world_trace.rotations, atol=1e-6)

    def test_including_the_invalid_frames_would_have_given_a_different_answer(self):
        """Guards against the tests above passing for the wrong reason. Worth stating what
        this does NOT show: held padding is inert here, because zero synthetic gyro rows add
        nothing to the Procrustes cross-covariance. Masking earns its keep on frames that
        move WRONGLY -- interpolated poses from a failed reconstruction -- not on still ones.
        """
        all_valid = WorldTrace(self.ragged.timestamps, self.ragged.positions,
                               self.ragged.rotations)
        unmasked = assembly.align_world_to_imu(PlateTrial('plate', self.imu, all_valid))
        masked = assembly.align_world_to_imu(PlateTrial('plate', self.imu, self.ragged))

        self.assertFalse(np.allclose(unmasked.world_trace.rotations,
                                     masked.world_trace.rotations, atol=1e-3))

    def test_a_plate_with_no_valid_frames_raises(self):
        dead = WorldTrace(self.ragged.timestamps, self.ragged.positions,
                          self.ragged.rotations, valid=np.zeros(self.n, dtype=bool))
        with self.assertRaises(ValueError):
            assembly.align_world_to_imu(PlateTrial('plate', self.imu, dead))


class TestPlateTrialSyntheticGenerators(unittest.TestCase):
    def test_generate_random_plate_trial(self):
        plate = fixtures.generate_random_plate_trial(duration=2.0, fs=100.0, add_noise=False)

        self.assertEqual(len(plate), 200)
        self.assertEqual(plate.imu_trace.gyro.shape, (200, 3))
        self.assertEqual(plate.world_trace.rotations.shape, (200, 3, 3))
        np.testing.assert_allclose(plate.imu_trace.timestamps, plate.world_trace.timestamps)
        self.assertTrue(np.all(np.isfinite(plate.imu_trace.acc)))
        # Rotation matrices must stay orthonormal.
        products = np.einsum('nij,nkj->nik', plate.world_trace.rotations, plate.world_trace.rotations)
        np.testing.assert_allclose(products, np.array([np.eye(3)] * 200), atol=1e-10)

    def test_generate_random_plate_trial_noise_changes_the_imu_only(self):
        np.random.seed(8)
        quiet = fixtures.generate_random_plate_trial(duration=1.0, fs=100.0, add_noise=False)
        np.random.seed(8)
        noisy = fixtures.generate_random_plate_trial(duration=1.0, fs=100.0, add_noise=True,
                                                      gyro_noise_std=0.01, acc_noise_std=0.1)

        # Same seed, so the underlying trajectory is identical; only the IMU is perturbed.
        np.testing.assert_allclose(quiet.world_trace.positions, noisy.world_trace.positions)
        self.assertFalse(np.allclose(quiet.imu_trace.acc, noisy.imu_trace.acc))

    def test_generate_1dof_plate_holds_the_hinge_axis(self):
        np.random.seed(9)
        parent = fixtures.generate_random_plate_trial(duration=3.0, fs=100.0, add_noise=False)
        joint_center_parent = np.array([0.0, 0.0, 0.1])
        joint_center_child = np.array([0.0, 0.0, -0.15])
        R_p2j = Rotation.from_euler('XYZ', [0.2, -0.3, 0.5])
        R_c2j = Rotation.from_euler('XYZ', [-0.1, 0.4, 0.2])

        child = fixtures.generate_1dof_plate(parent, joint_center_parent, joint_center_child,
                                           parent_to_joint_rotation=R_p2j,
                                           child_to_joint_rotation=R_c2j,
                                           add_noise=False)

        self.assertEqual(len(child), len(parent))

        # The chain is R_wc = R_wp @ R_p2j @ Rz(angle) @ R_c2j^T, so undoing the two
        # mounting rotations must leave a pure rotation about the joint's z axis.
        R_pc = np.einsum('nji,njk->nik', parent.world_trace.rotations, child.world_trace.rotations)
        R_joint = R_p2j.as_matrix().T @ R_pc @ R_c2j.as_matrix()
        rotvecs = Rotation.from_matrix(R_joint).as_rotvec()
        np.testing.assert_allclose(rotvecs[:, :2], np.zeros((len(rotvecs), 2)), atol=1e-8)
        # And the hinge has to actually move, otherwise the check above is vacuous.
        self.assertGreater(float(np.ptp(rotvecs[:, 2])), 0.5)

        # Both bodies must agree on where the joint center is in the world.
        joint_from_parent = parent.world_trace.positions + np.einsum(
            'nij,j->ni', parent.world_trace.rotations, joint_center_parent)
        joint_from_child = child.world_trace.positions + np.einsum(
            'nij,j->ni', child.world_trace.rotations, joint_center_child)
        np.testing.assert_allclose(joint_from_parent, joint_from_child, atol=1e-10)


class TestPlateTrialFromFolder(unittest.TestCase):
    """Loads real source data; skips cleanly when data/ has not been populated."""

    @classmethod
    def setUpClass(cls):
        cls.trial_dir = paths.raw_trial_dir('02', 'complexTasks')
        if not cls.trial_dir.is_dir():
            require_data(False, f"no source data at {cls.trial_dir}")
        cls.plates = alborno.load_trial(cls.trial_dir)

    def test_loads_every_sensor_as_a_plate_trial(self):
        self.assertTrue(self.plates)
        for name, plate in self.plates.items():
            self.assertIsInstance(plate, PlateTrial)
            self.assertEqual(plate.name, name)

    def test_all_plates_share_one_timeline(self):
        lengths = {len(plate) for plate in self.plates.values()}
        self.assertEqual(len(lengths), 1, f"Plates have differing lengths: {lengths}")

        reference = next(iter(self.plates.values()))
        for plate in self.plates.values():
            np.testing.assert_allclose(plate.imu_trace.timestamps, reference.imu_trace.timestamps)
            np.testing.assert_allclose(plate.imu_trace.timestamps, plate.world_trace.timestamps, atol=1e-8)

        # t = 0 is the first OVERLAP sample, not the first sample of the trace. Inertial
        # data recorded before the mocap started is kept, at negative time — which is the
        # whole point of the non-destructive alignment, and is 430 s of Subject01's walking
        # trial. What has to hold is that t = 0 exists and is where the mocap begins.
        timestamps = reference.imu_trace.timestamps
        self.assertLessEqual(timestamps[0], 1e-9, "the trace should start at or before t=0")
        self.assertGreater(timestamps[-1], 0.0)
        zero_index = int(np.searchsorted(timestamps, 0.0))
        self.assertAlmostEqual(timestamps[zero_index], 0.0, places=7)
        self.assertTrue(reference.valid[zero_index],
                        "t=0 is the first overlap sample, so it must be scoreable")
        self.assertFalse(reference.valid[:zero_index].any(),
                         "nothing before t=0 has mocap coverage, so none of it is scoreable")

    def test_loaded_rotations_are_valid(self):
        plate = next(iter(self.plates.values()))
        rotations = plate.world_trace.rotations
        products = np.einsum('nij,nkj->nik', rotations, rotations)
        np.testing.assert_allclose(products, np.array([np.eye(3)] * len(rotations)), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(rotations), np.ones(len(rotations)), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
