import unittest
from src.toolchest.gyro_utils import (finite_difference_rotations, integrate_rotations)
import nimblephysics as nimble
import numpy as np
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.WorldTrace import WorldTrace
from typing import List


class TestIMUTrace(unittest.TestCase):
    def assertListOfNpArraysEqual(self, list1: List[np.ndarray], list2: List[np.ndarray]):
        for i in range(len(list1)):
            self.assertTrue((list1[i] == list2[i]).all())

    def setUp(self):
        # Sample data for testing
        self.timestamps = np.array([0, 1, 2, 3, 4])
        self.gyro = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3]),
                     np.array([4, 4, 4])]
        self.acc = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3]),
                    np.array([4, 4, 4])]
        self.mag = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3]),
                    np.array([4, 4, 4])]
        self.imu_trace = IMUTrace(self.timestamps, self.gyro, self.acc, self.mag)

        # Slightly different data for allclose test
        self.timestamps_close = np.array([0, 1, 2, 3, 4]) + 1e-7
        self.gyro_close = [np.array([0, 0, 0]) + 1e-7, np.array([1, 1, 1]) + 1e-7, np.array([2, 2, 2]) + 1e-7, np.array([3, 3, 3]) + 1e-7, np.array([4, 4, 4]) + 1e-7]
        self.acc_close = [np.array([0, 0, 0]) + 1e-7, np.array([1, 1, 1]) + 1e-7, np.array([2, 2, 2]) + 1e-7, np.array([3, 3, 3]) + 1e-7, np.array([4, 4, 4]) + 1e-7]
        self.mag_close = [np.array([0, 0, 0]) + 1e-7, np.array([1, 1, 1]) + 1e-7, np.array([2, 2, 2]) + 1e-7, np.array([3, 3, 3]) + 1e-7, np.array([4, 4, 4]) + 1e-7]
        self.imu_trace_close = IMUTrace(self.timestamps_close, self.gyro_close, self.acc_close, self.mag_close)

    def test_initialization(self):
        self.assertTrue((self.imu_trace.timestamps == self.timestamps).all())
        self.assertListOfNpArraysEqual(self.imu_trace.gyro, self.gyro)
        self.assertListOfNpArraysEqual(self.imu_trace.acc, self.acc)

    def test_length(self):
        self.assertEqual(len(self.imu_trace), 5)

    def test_getitem_slice(self):
        sliced_imu_trace = self.imu_trace[1:4]
        expected_timestamps = np.array([1, 2, 3])
        expected_gyro = [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        expected_acc = [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]

        self.assertTrue((sliced_imu_trace.timestamps == expected_timestamps).all())
        self.assertListOfNpArraysEqual(sliced_imu_trace.gyro, expected_gyro)
        self.assertListOfNpArraysEqual(sliced_imu_trace.acc, expected_acc)

    def test_getitem_index(self):
        single_item_imu_trace = self.imu_trace[2]
        expected_timestamps = np.array([2])
        expected_gyro = [np.array([2, 2, 2])]
        expected_acc = [np.array([2, 2, 2])]

        self.assertTrue((single_item_imu_trace.timestamps == expected_timestamps).all())
        self.assertListOfNpArraysEqual(single_item_imu_trace.gyro, expected_gyro)
        self.assertListOfNpArraysEqual(single_item_imu_trace.acc, expected_acc)

    def test_allclose(self):
        self.assertTrue(self.imu_trace.allclose(self.imu_trace_close))

    def test_allclose_false(self):
        # The lengths no longer match, so this should return False
        self.assertFalse(self.imu_trace.allclose(self.imu_trace_close[1:4]))

    def test_finite_difference_gyros_central(self):
        expected_gradient = [np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])]
        gradient = self.imu_trace._finite_difference_gyros(method='central')
        self.assertListOfNpArraysEqual(gradient, expected_gradient)

    def test_project_invertible(self):
        offset = np.array([1, 0, 0])
        projected = self.imu_trace.project_acc(offset, finite_difference_gyro_method='central')
        recovered = projected.project_acc(-offset, finite_difference_gyro_method='central')
        self.assertTrue(recovered.allclose(self.imu_trace))

    def test_projection_zero_effect(self):
        gyro = [np.array([1, 0, 0])] * 5
        acc = [np.array([0, 0, 0])] * 5
        mag = [np.array([0, 0, 0])] * 5
        imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)
        offset = np.array([1, 0, 0])
        projected = imu_trace.project_acc(offset, finite_difference_gyro_method='central')
        self.assertTrue(projected.allclose(imu_trace))

    def test_projection_orthogonal_effect(self):
        gyro = [np.array([1, 0, 0])] * 5
        acc = [np.array([0, 0, 0])] * 5
        mag = [np.array([0, 0, 0])] * 5
        imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)
        offset = np.array([0, 1, 0])
        projected = imu_trace.project_acc(offset, finite_difference_gyro_method='central')
        # We expect to accelerate back towards the center of rotation, which means in the negative direction of the
        # offset.
        expected_acc = [np.array([0, -1, 0])] * 5
        expected_imu_trace = IMUTrace(self.timestamps, gyro, expected_acc, mag)
        self.assertTrue(projected.allclose(expected_imu_trace))

    def test_rotation(self):
        atol = 1e-8
        # Create a new IMUTrace with a rotation of 90 degrees about the z-axis
        gyro = [np.array([0, 0, 1])] * 5
        acc = [np.array([0, 1, 0])] * 5
        mag = [np.array([0, -1, 0])] * 5
        imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)

        # Rotate the IMUTrace by 90 degrees about the x-axis
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotated = imu_trace.right_rotate(rotation_matrix.T)

        # The gyro should now be [0, -1, 0] and the acc should be [0, 0, 1]
        expected_gyro = [np.array([0, -1, 0])] * 5
        expected_acc = [np.array([0, 0, 1])] * 5
        expected_mag = [np.array([0, 0, -1])] * 5
        expected_imu_trace = IMUTrace(self.timestamps, expected_gyro, expected_acc, expected_mag)
        print(rotated.gyro)

        for i in range(len(rotated.timestamps)):
            print("Checking timestamp for the left rotate ", i)
            self.assertTrue(np.allclose(rotated.gyro[i], expected_imu_trace.gyro[i], atol=atol))
            self.assertTrue(np.allclose(rotated.acc[i], expected_imu_trace.acc[i], atol=atol))
            self.assertTrue(np.allclose(rotated.mag[i], expected_imu_trace.mag[i], atol=atol))

    def test_rotate_round_trip(self):
        atol = 1e-8
        # Create a new IMUTrace with a rotation of 90 degrees about the z-axis
        gyro = [np.array([0, 1, 0])] * 5
        R_world_imu = integrate_rotations(gyro, self.timestamps)
        acc = [R.T @ np.array([0, 1, 0]) for R in R_world_imu]
        mag = [R.T @ np.array([0, -1, 0]) for R in R_world_imu]

        imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)

        # Rotate the IMUTrace by 90 degrees about the x-axis
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        rotated = imu_trace.left_rotate(rotation_matrix)
        returned = rotated.left_rotate(rotation_matrix.T)

        self.assertTrue(np.allclose(imu_trace.timestamps, returned.timestamps, atol=atol))
        for i in range(len(imu_trace.timestamps)):
            print("Checking timestamp for the left rotate ", i)
            self.assertTrue(np.allclose(imu_trace.gyro[i], returned.gyro[i], atol=atol))
            self.assertTrue(np.allclose(imu_trace.acc[i], returned.acc[i], atol=atol))
            self.assertTrue(np.allclose(imu_trace.mag[i], returned.mag[i], atol=atol))

        rotated = imu_trace.right_rotate(rotation_matrix)
        returned = rotated.right_rotate(rotation_matrix.T)

        self.assertTrue(np.allclose(imu_trace.timestamps, returned.timestamps, atol=atol))
        for i in range(len(imu_trace.timestamps)):
            print("Checking timestamp for the left rotate ", i)
            self.assertTrue(np.allclose(imu_trace.gyro[i], returned.gyro[i], atol=atol))
            self.assertTrue(np.allclose(imu_trace.acc[i], returned.acc[i], atol=atol))
            self.assertTrue(np.allclose(imu_trace.mag[i], returned.mag[i], atol=atol))

    def test_rotate_multiple_rotations(self):
        timestamps = np.linspace(0, 1, 50)
        gyro = [np.array([0, 0, 0])] * len(timestamps)
        acc = [np.array([0, 0, 1])] * len(timestamps)
        mag = [np.array([0, 1, 0])] * len(timestamps)

        imu_trace = IMUTrace(timestamps, gyro, acc, mag)
        rotations = [nimble.math.eulerXYZToMatrix(np.array([0, 0, np.pi / 100 * t])) for t in timestamps]
        rotated_imu_trace = imu_trace.right_rotate(rotations)

        return_rotations = [rot.T for rot in rotations]
        returned_imu_trace = rotated_imu_trace.right_rotate(return_rotations)

        for i in range(len(timestamps)):
            np.testing.assert_allclose(rotations[i] @ return_rotations[i], np.eye(3), atol=1e-8)

            np.testing.assert_allclose(gyro[i], returned_imu_trace.gyro[i], atol=1e-8)
            np.testing.assert_allclose(returned_imu_trace.acc[i], acc[i], atol=1e-8)
            np.testing.assert_allclose(returned_imu_trace.mag[i], mag[i], atol=1e-8)

        rotated_imu_trace = imu_trace.left_rotate(rotations)
        returned_imu_trace = rotated_imu_trace.left_rotate(return_rotations)
        for i in range(len(timestamps)):
            np.testing.assert_allclose(rotations[i] @ return_rotations[i], np.eye(3), atol=1e-8)

            np.testing.assert_allclose(returned_imu_trace.gyro[i], gyro[i], atol=1e-8)
            np.testing.assert_allclose(returned_imu_trace.acc[i], acc[i], atol=1e-8)
            np.testing.assert_allclose(returned_imu_trace.mag[i], mag[i], atol=1e-8)


    def test_filter_gyros(self):
        gyro = [np.array([0, 0, 1])] * 50
        acc = [np.array([0, 1, 0])] * 50
        mag = [np.array([1, 0, 0])] * 50
        timesteps = np.linspace(0, 5, 50)
        imu_trace = IMUTrace(timesteps, gyro, acc, mag)
        filtered = imu_trace.lowpass_filter_gyro(0.01, 2)

        # The gyro should be unchanged
        expected_imu_trace = IMUTrace(timesteps, gyro, acc, mag)

        self.assertTrue(filtered.allclose(expected_imu_trace))

    def test_RMSD_zero(self):
        gyro = [np.array([0, 0, 0])] * 5
        acc = [np.array([0, 0, 0])] * 5
        mag = [np.array([0, 0, 0])] * 5
        imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)
        equal_imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)

        self.assertEqual(imu_trace.calculate_acc_RMSD(equal_imu_trace).all(), 0)
        self.assertEqual(imu_trace.calculate_gyro_RMSD(equal_imu_trace).all(), 0)
        self.assertEqual(imu_trace.calculate_mag_RMSD(equal_imu_trace).all(), 0)

    def test_RMSD_nonzero(self):
        gyro = [np.array([0, 0, 0])] * 5
        acc = [np.array([0, 0, 0])] * 5
        mag = [np.array([0, 0, 0])] * 5
        imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)
        non_equal_imu_trace = IMUTrace(self.timestamps, [np.array([1, 1, 1])] * 5, [np.array([1, 1, 1])] * 5, [np.array([1, 1, 1])] * 5)

        self.assertEqual(imu_trace.calculate_acc_RMSD(non_equal_imu_trace).all(), 1)
        self.assertEqual(imu_trace.calculate_gyro_RMSD(non_equal_imu_trace).all(), 1)
        self.assertEqual(imu_trace.calculate_mag_RMSD(non_equal_imu_trace).all(), 1)

    def test_pearson_linear_sensors(self):
        imu_trace = IMUTrace(self.timestamps, self.gyro, self.acc, self.mag)
        gyro_p_x, gyro_p_y, gyro_p_z, gyro_p_norm = imu_trace.calculate_gyro_pearson_correlation(self.imu_trace)
        self.assertAlmostEqual(gyro_p_x, 1)
        self.assertAlmostEqual(gyro_p_y, 1)
        self.assertAlmostEqual(gyro_p_z, 1)
        self.assertAlmostEqual(gyro_p_norm, 1)

        acc_p_x, acc_p_y, acc_p_z, acc_p_norm = self.imu_trace.calculate_acc_pearson_correlation(self.imu_trace)
        self.assertAlmostEqual(acc_p_x, 1)
        self.assertAlmostEqual(acc_p_y, 1)
        self.assertAlmostEqual(acc_p_z, 1)
        self.assertAlmostEqual(acc_p_norm, 1)

        mag_p_x, mag_p_y, mag_p_z, mag_p_norm = self.imu_trace.calculate_mag_pearson_correlation(self.imu_trace)
        self.assertAlmostEqual(mag_p_x, 1)
        self.assertAlmostEqual(mag_p_y, 1)
        self.assertAlmostEqual(mag_p_z, 1)
        self.assertAlmostEqual(mag_p_norm, 1)

    def test_pearson_non_linear_sensors(self):
        x = np.sin(self.timestamps)
        y = np.cos(self.timestamps)
        z = np.tan(self.timestamps)
        gyro = [np.array([x[i], y[i], z[i]]) for i in range(5)]
        acc = [np.array([2*x[i], 2*y[i], 2*z[i]]) for i in range(5)]
        mag = [np.array([3*x[i], 3*y[i], 3*z[i]]) for i in range(5)]
        imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)

        gyro_orig_p_x, gyro_orig_p_y, gyro_orig_p_z, gyro_orig_p_norm = imu_trace.calculate_gyro_pearson_correlation(self.imu_trace)
        self.assertNotAlmostEqual(gyro_orig_p_x, 1)
        self.assertNotAlmostEqual(gyro_orig_p_y, 1)
        self.assertNotAlmostEqual(gyro_orig_p_z, 1)
        self.assertNotAlmostEqual(gyro_orig_p_norm, 1)

        other_imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)

        gyro_p_x, gyro_p_y, gyro_p_z, gyro_p_norm = imu_trace.calculate_gyro_pearson_correlation(other_imu_trace)
        self.assertAlmostEqual(gyro_p_x, 1)
        self.assertAlmostEqual(gyro_p_y, 1)
        self.assertAlmostEqual(gyro_p_z, 1)
        self.assertAlmostEqual(gyro_p_norm, 1)


        acc_p_x, acc_p_y, acc_p_z, acc_p_norm = imu_trace.calculate_acc_pearson_correlation(other_imu_trace)
        self.assertAlmostEqual(acc_p_x, 1)
        self.assertAlmostEqual(acc_p_y, 1)
        self.assertAlmostEqual(acc_p_z, 1)
        self.assertAlmostEqual(acc_p_norm, 1)

        mag_p_x, mag_p_y, mag_p_z, mag_p_norm = imu_trace.calculate_mag_pearson_correlation(other_imu_trace)
        self.assertAlmostEqual(mag_p_x, 1)
        self.assertAlmostEqual(mag_p_y, 1)
        self.assertAlmostEqual(mag_p_z, 1)
        self.assertAlmostEqual(mag_p_norm, 1)

    def test_calculate_rotation_offset_identity(self):
        gyro = [np.array([1, 0, 0])] * 5
        acc = [np.array([0, 0, 0])] * 5
        mag = [np.array([0, 0, 1])] * 5
        imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)
        R_so = imu_trace.calculate_rotation_offset_from_gyros_and_mags(imu_trace)
        np.testing.assert_allclose(np.eye(3), R_so)

    def test_calculate_rotation_offset_arbitrary(self):
        gyro = [np.array([1, 0, 0])] * 5
        acc = [np.array([0, 0, 0])] * 5
        mag = [np.array([0, 0, 1])] * 5
        imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)
        R_so = nimble.math.eulerXYZToMatrix(np.array([np.pi / 3, np.pi / 7, np.pi / 5]))
        rotated_imu = imu_trace.right_rotate(R_so)
        R_so_recovered = imu_trace.calculate_rotation_offset_from_gyros_and_mags(rotated_imu)
        np.testing.assert_allclose(R_so, R_so_recovered)

    def test_sensor_offset(self):
        offset = np.array([1, 0, 0])
        imu_trace = self.imu_trace.add_offset_to_gyro(offset)
        imu_trace = imu_trace.add_offset_to_acc(offset)
        imu_trace = imu_trace.add_offset_to_mag(offset)
        estimated_offset_gyro = imu_trace.calculate_gyro_bias(self.imu_trace)
        estimated_offset_acc = imu_trace.calculate_acc_bias(self.imu_trace)
        estimated_offset_mag = imu_trace.calculate_mag_bias(self.imu_trace)
        np.testing.assert_allclose(offset, estimated_offset_gyro)
        np.testing.assert_allclose(offset, estimated_offset_acc)
        np.testing.assert_allclose(offset, estimated_offset_mag)

    def test_sensor_offset_arbitrary(self):
        offset = np.random.rand(3)
        imu_trace = self.imu_trace.add_offset_to_gyro(offset)
        imu_trace = imu_trace.add_offset_to_acc(offset)
        imu_trace = imu_trace.add_offset_to_mag(offset)
        estimated_offset_gyro = imu_trace.calculate_gyro_bias(self.imu_trace)
        estimated_offset_acc = imu_trace.calculate_acc_bias(self.imu_trace)
        estimated_offset_mag = imu_trace.calculate_mag_bias(self.imu_trace)
        np.testing.assert_allclose(offset, estimated_offset_gyro)
        np.testing.assert_allclose(offset, estimated_offset_acc)
        np.testing.assert_allclose(offset, estimated_offset_mag)


    def test_calculate_gyro_angle_error(self):
        gyro = [np.array([1, 0, 0])] * 5
        acc = [np.array([0, 0, 0])] * 5
        mag = [np.array([0, 0, 1])] * 5
        imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)
        angle_offset = imu_trace.calculate_gyro_angle_error(imu_trace)
        self.assertAlmostEqual(angle_offset.all(), 0)

        R = nimble.math.eulerXYZToMatrix(np.array([0, np.pi, 0]))
        rotated_imu = imu_trace.right_rotate(R.T)
        angle_offset = imu_trace.calculate_gyro_angle_error(rotated_imu)
        np.testing.assert_allclose(angle_offset, np.pi)

    def test_construct_imu_trace(self):
        z_val = -9.81 + 2 * 0.05
        timestamps = np.linspace(0, 1, 100)
        gyro = [np.array([1, 0, 0])] * 100
        acc = [np.array([0, 0, z_val])] * 100
        mag = [np.array([0, 0, 1])] * 100
        imu_trace = IMUTrace(timestamps, gyro, acc, mag)

        np.testing.assert_equal(imu_trace.acc[0], np.array([0, 0, z_val]))
        np.testing.assert_equal(imu_trace.acc[1], np.array([0, 0, z_val]))
        for i in range(2, 100):
            np.testing.assert_equal(imu_trace.acc[i], np.array([0, 0, z_val]))

    def test_re_zero_imu_trace(self):
        timestamps = np.linspace(5, 6, 100)
        gyro = [np.array([1, 0, 0])] * 100
        acc = [np.array([0, 0, 0])] * 100
        mag = [np.array([0, 0, 1])] * 100
        imu_trace = IMUTrace(timestamps, gyro, acc, mag)

        expected_timestamps = np.linspace(0, 1, 100)
        re_zerod_imu = imu_trace.re_zero_timestamps()

        np.testing.assert_allclose(re_zerod_imu.timestamps, expected_timestamps)

    def test_resample_same_frequency(self):
        resampled_trace = self.imu_trace.resample(1.0)
        np.testing.assert_array_equal(resampled_trace.timestamps, self.timestamps)
        np.testing.assert_array_equal(resampled_trace.gyro, np.array(self.gyro))
        np.testing.assert_array_equal(resampled_trace.acc, np.array(self.acc))
        np.testing.assert_array_equal(resampled_trace.mag, np.array(self.mag))


    def test_resample_higher_frequency(self):
        timestamps = np.linspace(0, 5, 6)  # 0, 1, 2, 3, 4, 5 seconds
        gyro = [np.array([t, t, t]) for t in timestamps]
        acc = [np.array([t, t, t]) for t in timestamps]
        mag = [np.array([t, t, t]) for t in timestamps]

        imu_trace = IMUTrace(timestamps, gyro, acc, mag)
        new_frequency = 2.0
        resampled_trace = imu_trace.resample(new_frequency)
        expected_timestamps = np.linspace(start=timestamps[0], stop=timestamps[-1], num=len(timestamps) * 2 - 1, endpoint=True)
        np.testing.assert_allclose(resampled_trace.timestamps, expected_timestamps, rtol=1e-5)

        # Calculate expected gyro values
        expected_gyro = [np.array([t, t, t]) for t in expected_timestamps]
        np.testing.assert_allclose(resampled_trace.gyro, expected_gyro, rtol=1e-5)

        # Calculate expected acc values
        expected_acc = [np.array([t, t, t]) for t in expected_timestamps]
        np.testing.assert_allclose(resampled_trace.acc, expected_acc, rtol=1e-5)

        # Calculate expected mag values
        expected_mag = [np.array([t, t, t]) for t in expected_timestamps]
        np.testing.assert_allclose(resampled_trace.mag, expected_mag, rtol=1e-5)



    def test_resample_lower_frequency(self):
        timestamps = np.linspace(0, 5, 6)  # 0, 1, 2, 3, 4, 5 seconds
        gyro = [np.array([t, t, t]) for t in timestamps]
        acc = [np.array([t, t, t]) for t in timestamps]
        mag = [np.array([t, t, t]) for t in timestamps]

        imu_trace = IMUTrace(timestamps, gyro, acc, mag)
        new_frequency = 0.5
        resampled_trace = imu_trace.resample(new_frequency)
        expected_timestamps = np.array([0.0, 2.0, 4.0])
        np.testing.assert_allclose(resampled_trace.timestamps, expected_timestamps, rtol=1e-5)

        # Calculate expected gyro values
        expected_gyro = [np.array([t, t, t]) for t in expected_timestamps]
        np.testing.assert_allclose(resampled_trace.gyro, expected_gyro, rtol=1e-5)

        # Calculate expected acc values
        expected_acc = [np.array([t, t, t]) for t in expected_timestamps]
        np.testing.assert_allclose(resampled_trace.acc, expected_acc, rtol=1e-5)

        # Calculate expected mag values
        expected_mag = [np.array([t, t, t]) for t in expected_timestamps]
        np.testing.assert_allclose(resampled_trace.mag, expected_mag, rtol=1e-5)

    def test_subtraction(self):
        timestamps = np.array([0, 1, 2, 3, 4])
        gyro1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        acc1 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
        mag1 = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]])

        gyro2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        acc2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        mag2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])

        imu1 = IMUTrace(timestamps, gyro1, acc1, mag1)
        imu2 = IMUTrace(timestamps, gyro2, acc2, mag2)

        result = imu1 - imu2

        expected_gyro = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]])
        expected_acc = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        expected_mag = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])

        np.testing.assert_array_equal(result.gyro, expected_gyro)
        np.testing.assert_array_equal(result.acc, expected_acc)
        np.testing.assert_array_equal(result.mag, expected_mag)

    def test_subtraction_different_lengths(self):
        timestamps1 = np.array([0, 1, 2, 3, 4])
        timestamps2 = np.array([0, 1, 2])
        gyro = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        acc = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
        mag = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]])

        imu1 = IMUTrace(timestamps1, gyro, acc, mag)
        imu2 = IMUTrace(timestamps2, gyro[:3], acc[:3], mag[:3])

        with self.assertRaises(AssertionError):
            _ = imu1 - imu2

    def test_subtraction_different_timestamps(self):
        timestamps1 = np.array([0, 1, 2, 3, 4])
        timestamps2 = np.array([0, 2, 4, 6, 8])
        gyro = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        acc = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
        mag = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]])

        imu1 = IMUTrace(timestamps1, gyro, acc, mag)
        imu2 = IMUTrace(timestamps2, gyro, acc, mag)

        with self.assertRaises(AssertionError):
            _ = imu1 - imu2

    def test_project_mag_zero_offset(self):
        timestamps = np.linspace(0, 5, 6)
        gyro = [np.array([1, 0, 0])] * 6
        acc = [np.array([0, 0, 0])] * 6
        mag = [np.array([0, 0, 1])] * 6

        second_mag = [np.array([0, 0, 2])] * 6

        first_imu = IMUTrace(timestamps, gyro, acc, mag)
        second_imu = IMUTrace(timestamps, gyro, acc, second_mag)

        projected_imu = first_imu.project_mag(second_imu, [0., 1., 0.,], [0., 0., 0.])
        np.testing.assert_allclose(projected_imu.mag, first_imu.mag, err_msg="Projecting to the same mag location should not change the mag")

        projected_imu = first_imu.project_mag(second_imu, [0., 1., 0.,], [0., 0., 1.])
        np.testing.assert_allclose(projected_imu.mag, first_imu.mag, err_msg="Projecting orthogonal to offset should not change the mag")

        projected_imu = first_imu.project_mag(second_imu, [0., 1., 0.,], [0., 1., 0.])
        np.testing.assert_allclose(projected_imu.mag, second_imu.mag, err_msg="Projecting to the same mag location should be equal")

        projected_imu = first_imu.project_mag(second_imu, [0., 1., 0.,], [0., 1., 1.])
        np.testing.assert_allclose(projected_imu.mag, second_imu.mag, err_msg="Projecting orthogonal to offset should not change the mag")

        projected_imu = first_imu.project_mag(second_imu, [0., 1., 0.,], [0., 0.5, 0.])
        for t in range(len(timestamps)):
            np.testing.assert_allclose(projected_imu.mag[t], [0, 0, 1.5], err_msg="Projecting to the middle should be the average of the two mags")

    def test_sphere_fit_mag(self):
        # Generate synthetic IMU data
        timestamps = np.linspace(0, 1, 100)
        gyro = [np.random.randn(3) for _ in range(100)]
        acc = [np.random.randn(3) for _ in range(100)]

        # Generate synthetic magnetic field data around a sphere of radius 50 centered at (10, 20, 30)
        radius = 50
        center = np.array([10, 20, 30])
        mag = []
        for _ in range(100):
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            mag.append(center + radius * direction)

        imu_trace = IMUTrace(timestamps, gyro, acc, mag)

        bias, radius = imu_trace.sphere_fit_mag()
        expected_center = np.array([10, 20, 30])
        expected_radius = 50

        # Assert that the calculated bias is close to the expected center
        np.testing.assert_allclose(bias, expected_center, atol=1)

        # Assert that the calculated radius is close to the expected radius
        self.assertAlmostEqual(radius, expected_radius, delta=1)

    def test_sphere_fit_mag_with_override_radius(self):
        # Generate synthetic IMU data
        timestamps = np.linspace(0, 1, 100)
        gyro = [np.random.randn(3) for _ in range(100)]
        acc = [np.random.randn(3) for _ in range(100)]

        # Generate synthetic magnetic field data around a sphere of radius 50 centered at (10, 20, 30)
        radius = 50
        center = np.array([10, 20, 30])
        mag = []
        for _ in range(100):
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            mag.append(center + radius * direction)

        imu_trace = IMUTrace(timestamps, gyro, acc, mag)

        override_radius = 60
        bias, radius = imu_trace.sphere_fit_mag(override_radius=override_radius)

        # Assert that the calculated radius is equal to the overridden radius
        self.assertEqual(radius, override_radius)



if __name__ == '__main__':
    unittest.main()
