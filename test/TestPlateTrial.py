import os
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import paths
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace

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


class TestPlateTrialSync(unittest.TestCase):
    def test_identical_arrays(self):
        array1 = [1, 2, 3, 4, 5]
        array2 = [1, 2, 3, 4, 5]
        expected_slice1 = slice(0, 5)
        expected_slice2 = slice(0, 5)
        slice1, slice2 = PlateTrial._sync_arrays(np.array(array1), np.array(array2))
        self.assertEqual(expected_slice1, slice1)
        self.assertEqual(expected_slice2, slice2)

    def test_offset_arrays(self):
        array1 = [1., 2, 3, 4, 5, 6, 7, -4]
        array2 = [0.1, 1, 2, 3, 4, 5, 6, 7]
        expected_slice1 = slice(0, 7)
        expected_slice2 = slice(1, 8)
        slice1, slice2 = PlateTrial._sync_arrays(np.array(array1), np.array(array2))
        self.assertEqual(expected_slice1, slice1)
        self.assertEqual(expected_slice2, slice2)

        # Try swapping order of arrays
        slice1, slice2 = PlateTrial._sync_arrays(np.array(array2), np.array(array1))
        self.assertEqual(expected_slice2, slice1)
        self.assertEqual(expected_slice1, slice2)

    def test_no_overlap(self):
        array1 = [1, 2, 3, 4, 5]
        array2 = [6, 7, 8, 9, 10]
        expected_slice1 = slice(0, 5)
        expected_slice2 = slice(0, 5)
        slice1, slice2 = PlateTrial._sync_arrays(np.array(array1), np.array(array2))
        self.assertEqual(expected_slice1, slice1)
        self.assertEqual(expected_slice2, slice2)

    def test_sync_arrays_long_function(self):
        length = 250
        start = 100
        t = np.linspace(0, 100, length + start)
        base_function = np.sin(t) + np.sin(17 * t) - np.cos(3 * t) - np.cos(60 * t)
        expected_slice1 = slice(start, start + length)
        expected_slice2 = slice(0, length)
        array1 = base_function
        array2 = base_function[expected_slice1]

        slice1, slice2 = PlateTrial._sync_arrays(np.array(array1), np.array(array2))
        self.assertEqual(expected_slice1, slice1)
        self.assertEqual(expected_slice2, slice2)

    def test_sync_traces_recovers_a_known_lag(self):
        # Build a world trace with distinctive rotation, derive the IMU trace it implies,
        # then delay the mocap by a known number of samples. _sync_traces should find it.
        lag = 30
        num_samples = 400
        timestamps = np.arange(num_samples) / 100.0
        angle = 0.8 * np.sin(2 * np.pi * 1.3 * timestamps) + 0.3 * np.sin(2 * np.pi * 4.1 * timestamps)
        rotations = Rotation.from_rotvec(angle[:, None] * np.array([0.3, 0.5, 0.8])).as_matrix()
        world_trace = WorldTrace(timestamps, np.zeros((num_samples, 3)), rotations)
        imu_trace = world_trace.calculate_imu_trace(np.array([0., 0., 9.81]), skip_lin_acc=True)

        delayed_world = WorldTrace(timestamps[:num_samples - lag], world_trace.positions[lag:],
                                   world_trace.rotations[lag:])

        imu_slice, world_slice = PlateTrial._sync_traces(imu_trace, delayed_world)

        self.assertEqual(imu_slice.start, lag)
        self.assertEqual(world_slice.start, 0)
        self.assertEqual(imu_slice.stop - imu_slice.start, world_slice.stop - world_slice.start)


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

        aligned = PlateTrial('plate', imu_trace, world_trace)._align_world_trace_to_imu_trace()

        expected = np.matmul(world_rotations, R_plate_sensor)
        np.testing.assert_allclose(aligned.world_trace.rotations, expected, atol=1e-6)


class TestPlateTrialSyntheticGenerators(unittest.TestCase):
    def test_generate_random_plate_trial(self):
        plate = PlateTrial.generate_random_plate_trial(duration=2.0, fs=100.0, add_noise=False)

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
        quiet = PlateTrial.generate_random_plate_trial(duration=1.0, fs=100.0, add_noise=False)
        np.random.seed(8)
        noisy = PlateTrial.generate_random_plate_trial(duration=1.0, fs=100.0, add_noise=True,
                                                      gyro_noise_std=0.01, acc_noise_std=0.1)

        # Same seed, so the underlying trajectory is identical; only the IMU is perturbed.
        np.testing.assert_allclose(quiet.world_trace.positions, noisy.world_trace.positions)
        self.assertFalse(np.allclose(quiet.imu_trace.acc, noisy.imu_trace.acc))

    def test_generate_1dof_plate_holds_the_hinge_axis(self):
        np.random.seed(9)
        parent = PlateTrial.generate_random_plate_trial(duration=3.0, fs=100.0, add_noise=False)
        joint_center_parent = np.array([0.0, 0.0, 0.1])
        joint_center_child = np.array([0.0, 0.0, -0.15])
        R_p2j = Rotation.from_euler('XYZ', [0.2, -0.3, 0.5])
        R_c2j = Rotation.from_euler('XYZ', [-0.1, 0.4, 0.2])

        child = parent.generate_1dof_plate(joint_center_parent, joint_center_child,
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
            raise unittest.SkipTest(f"No source data found at {cls.trial_dir}")
        cls.plates = PlateTrial.from_folder(cls.trial_dir)

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
        self.assertAlmostEqual(reference.imu_trace.timestamps[0], 0.0)

    def test_loaded_rotations_are_valid(self):
        plate = next(iter(self.plates.values()))
        rotations = plate.world_trace.rotations
        products = np.einsum('nij,nkj->nik', rotations, rotations)
        np.testing.assert_allclose(products, np.array([np.eye(3)] * len(rotations)), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(rotations), np.ones(len(rotations)), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
