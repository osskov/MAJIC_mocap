import os
import tempfile
import unittest
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from src.toolchest.IMUTrace import IMUTrace


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

    def test_initialization_rejects_mismatched_lengths(self):
        with self.assertRaises(AssertionError):
            IMUTrace(self.timestamps, self.gyro[:4], self.acc, self.mag)

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

    def test_eq(self):
        self.assertEqual(self.imu_trace, IMUTrace(self.timestamps, self.gyro, self.acc, self.mag))
        self.assertNotEqual(self.imu_trace, self.imu_trace_close)
        self.assertNotEqual(self.imu_trace, self.imu_trace[1:4])

    def test_allclose(self):
        self.assertTrue(self.imu_trace.allclose(self.imu_trace_close))

    def test_allclose_false(self):
        # The lengths no longer match, so this should return False
        self.assertFalse(self.imu_trace.allclose(self.imu_trace_close[1:4]))

    def test_copy_is_independent(self):
        copied = self.imu_trace.copy()
        copied.gyro[0] = np.array([99, 99, 99])
        np.testing.assert_array_equal(self.imu_trace.gyro[0], np.array([0, 0, 0]))

    def test_shallow_copy_shares_arrays(self):
        shallow = self.imu_trace.shallow_copy()
        self.assertIs(shallow.gyro, self.imu_trace.gyro)
        self.assertIs(shallow.acc, self.imu_trace.acc)
        self.assertIs(shallow.mag, self.imu_trace.mag)

    def test_get_sample_frequency(self):
        self.assertAlmostEqual(self.imu_trace.get_sample_frequency(), 1.0)
        hundred_hz = IMUTrace(np.arange(10) / 100.0, np.zeros((10, 3)), np.zeros((10, 3)), np.zeros((10, 3)))
        self.assertAlmostEqual(hundred_hz.get_sample_frequency(), 100.0)

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

    def test_projection_leaves_gyro_and_mag_alone(self):
        # Long enough for the default 'polyfit' gyro derivative, which needs a window
        # of 10 samples.
        np.random.seed(3)
        n = 30
        trace = IMUTrace(np.arange(n) / 100.0, np.random.randn(n, 3), np.random.randn(n, 3), np.random.randn(n, 3))
        projected = trace.project_acc(np.array([0.1, -0.2, 0.05]))
        np.testing.assert_array_equal(projected.gyro, trace.gyro)
        np.testing.assert_array_equal(projected.mag, trace.mag)
        np.testing.assert_array_equal(projected.timestamps, trace.timestamps)

    def test_calculate_rotation_offset_identity(self):
        np.random.seed(0)
        # The gyro data has to span all three axes, otherwise the best-fit rotation is
        # underdetermined and any rotation about the unexcited axis fits equally well.
        gyro = np.random.randn(20, 3)
        imu_trace = IMUTrace(np.arange(20) / 100.0, gyro, np.zeros((20, 3)), np.zeros((20, 3)))
        R_so = imu_trace.calculate_rotation_offset_from_gyros(imu_trace)
        np.testing.assert_allclose(np.eye(3), R_so, atol=1e-10)

    def test_calculate_rotation_offset_arbitrary(self):
        np.random.seed(1)
        timestamps = np.arange(20) / 100.0
        gyro = np.random.randn(20, 3)
        imu_trace = IMUTrace(timestamps, gyro, np.zeros((20, 3)), np.zeros((20, 3)))

        # The method solves self.gyro ~= R_so @ other.gyro, so build `other` in the
        # rotated frame by applying R_so backwards.
        R_so = Rotation.from_euler('XYZ', [np.pi / 3, np.pi / 7, np.pi / 5]).as_matrix()
        other_gyro = np.einsum('ji,nj->ni', R_so, gyro)
        other_trace = IMUTrace(timestamps, other_gyro, np.zeros((20, 3)), np.zeros((20, 3)))

        R_so_recovered = imu_trace.calculate_rotation_offset_from_gyros(other_trace)
        np.testing.assert_allclose(R_so, R_so_recovered, atol=1e-10)

    def test_calculate_gyro_angle_error(self):
        gyro = [np.array([1, 0, 0])] * 5
        acc = [np.array([0, 0, 0])] * 5
        mag = [np.array([0, 0, 1])] * 5
        imu_trace = IMUTrace(self.timestamps, gyro, acc, mag)

        angle_offset = imu_trace.calculate_gyro_angle_error(imu_trace)
        np.testing.assert_allclose(angle_offset, np.zeros(5), atol=1e-8)

        # Antiparallel gyros are pi radians apart.
        flipped = IMUTrace(self.timestamps, [-g for g in gyro], acc, mag)
        np.testing.assert_allclose(imu_trace.calculate_gyro_angle_error(flipped), np.full(5, np.pi))

        # Orthogonal gyros are pi/2 apart.
        orthogonal = IMUTrace(self.timestamps, [np.array([0, 1, 0])] * 5, acc, mag)
        np.testing.assert_allclose(imu_trace.calculate_gyro_angle_error(orthogonal), np.full(5, np.pi / 2))

    def test_calculate_gyro_angle_error_degenerate_cases(self):
        acc = [np.array([0, 0, 0])] * 5
        mag = [np.array([0, 0, 0])] * 5
        still = IMUTrace(self.timestamps, [np.zeros(3)] * 5, acc, mag)
        moving = IMUTrace(self.timestamps, [np.array([1., 0., 0.])] * 5, acc, mag)

        # Both stationary: the angle is defined to be zero.
        np.testing.assert_allclose(still.calculate_gyro_angle_error(still), np.zeros(5))
        # Only one stationary: the angle is undefined.
        self.assertTrue(np.all(np.isnan(still.calculate_gyro_angle_error(moving))))

    def test_add_noise_zero_std_is_a_no_op(self):
        self.assertTrue(self.imu_trace.add_noise().allclose(self.imu_trace))

    def test_add_noise_matches_requested_std(self):
        np.random.seed(2)
        n = 20000
        quiet = IMUTrace(np.arange(n) / 100.0, np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)))
        noisy = quiet.add_noise(gyro_noise_std=0.01, acc_noise_std=0.5, mag_noise_std=0.1)

        self.assertAlmostEqual(float(np.std(noisy.gyro)), 0.01, delta=0.001)
        self.assertAlmostEqual(float(np.std(noisy.acc)), 0.5, delta=0.05)
        self.assertAlmostEqual(float(np.std(noisy.mag)), 0.1, delta=0.01)
        # The original must be left untouched.
        np.testing.assert_array_equal(quiet.acc, np.zeros((n, 3)))

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

        # Every channel ramps linearly with time, so linear interpolation is exact.
        expected = [np.array([t, t, t]) for t in expected_timestamps]
        np.testing.assert_allclose(resampled_trace.gyro, expected, rtol=1e-5)
        np.testing.assert_allclose(resampled_trace.acc, expected, rtol=1e-5)
        np.testing.assert_allclose(resampled_trace.mag, expected, rtol=1e-5)

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

        expected = [np.array([t, t, t]) for t in expected_timestamps]
        np.testing.assert_allclose(resampled_trace.gyro, expected, rtol=1e-5)
        np.testing.assert_allclose(resampled_trace.acc, expected, rtol=1e-5)
        np.testing.assert_allclose(resampled_trace.mag, expected, rtol=1e-5)

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

    def test_from_txt(self):
        # Create a dummy IMU .txt file
        dummy_content = (
            "// Update Rate: 100.0Hz\n"
            "// Some metadata\n"
            "// Some metadata\n"
            "// Some metadata\n"
            "// Some metadata\n"
            "Acc_X\tAcc_Y\tAcc_Z\tGyr_X\tGyr_Y\tGyr_Z\tMag_X\tMag_Y\tMag_Z\n"
            "1.0\t2.0\t3.0\t4.0\t5.0\t6.0\t7.0\t8.0\t9.0\n"
            "1.1\t2.1\t3.1\t4.1\t5.1\t6.1\t7.1\t8.1\t9.1\n"
        )
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(dummy_content)
            temp_file_path = temp_file.name

        try:
            trace = IMUTrace.from_txt(temp_file_path)
            self.assertEqual(len(trace), 2)
            np.testing.assert_allclose(trace.timestamps, [0.0, 0.01])
            np.testing.assert_allclose(trace.acc, [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]])
            np.testing.assert_allclose(trace.gyro, [[4.0, 5.0, 6.0], [4.1, 5.1, 6.1]])
            np.testing.assert_allclose(trace.mag, [[7.0, 8.0, 9.0], [7.1, 8.1, 9.1]])
        finally:
            os.remove(temp_file_path)


if __name__ == '__main__':
    unittest.main()
