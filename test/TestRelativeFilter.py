import unittest
import numpy as np
import nimblephysics as nimble
from src.toolchest.RelativeFilter import RelativeFilter


class TestRelativeFilter(unittest.TestCase):
    def setUp(self):
        self.acc_std = 0.5
        self.gyro_std = 0.05
        self.mag_std = 0.3

        # Create instance of RelativeFilter
        self.filter = RelativeFilter(self.acc_std, self.gyro_std, self.mag_std)

    def test_initial_state(self):
        self.assertTrue(np.allclose(self.filter.q_wp.to_rotation_matrix(), np.eye(3)), "Initial q_wp is not identity.")
        self.assertTrue(np.allclose(self.filter.q_wc.to_rotation_matrix(), np.eye(3)), "Initial q_wc is not identity.")
        self.assertTrue(np.allclose(self.filter.P, np.eye(6)), "Initial P is not identity.")

    def test_measurement_jacobian_analytical_identity(self):
        R_wp = np.eye(3)
        R_wc = np.eye(3)
        acc_jc_p = np.array([0.0, 0.0, 1.0])
        acc_jc_c = np.array([0.0, 0.0, 1.0])
        mag_jc_p = np.array([1.0, 0.0, 0.0])
        mag_jc_c = np.array([1.0, 0.0, 0.0])

        H = self.filter.get_H_jacobian(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c)

        for col in range(6):
            epsilon = 1e-6
            perturbation = np.zeros(6)
            perturbation[col] = epsilon

            pos_measurements = self.filter.get_h(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, perturbation)
            neg_measurements = self.filter.get_h(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, -perturbation)
            delta_measurements = (pos_measurements - neg_measurements) / (2 * epsilon)
            for row in range(6):
                self.assertAlmostEqual(H[row, col], delta_measurements[row], places=3,
                                       msg=f"Jacobian element {row, col} is incorrect.")

    def test_measurement_jacobian_analytical_non_identity(self):
        R_wp = np.eye(3)
        R_wc = nimble.math.expMapRot(np.array([0.1, 0.2, 0.3]))
        R_cp = R_wc.T @ R_wp
        acc_jc_p = np.array([0.0, 0.0, 1.0])
        acc_jc_c = R_cp @ acc_jc_p
        mag_jc_p = np.array([1.0, 0.0, 0.0])
        mag_jc_c = R_cp @ mag_jc_p
        baseline_error = self.filter.get_h(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c)
        np.testing.assert_array_almost_equal(baseline_error, np.zeros(6), decimal=8)

        H = self.filter.get_H_jacobian(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c)

        for col in range(6):
            epsilon = 1e-6
            perturbation = np.zeros(6)
            perturbation[col] = epsilon

            pos_measurements = self.filter.get_h(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, perturbation)
            neg_measurements = self.filter.get_h(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, -perturbation)
            delta_measurements = (pos_measurements - neg_measurements) / (2 * epsilon)
            for row in range(6):
                self.assertAlmostEqual(H[row, col], delta_measurements[row], places=3,
                                       msg=f"Jacobian element {row, col} is incorrect.")

    def test_measurement_jacobian_analytical_non_identity(self):
        R_wp = nimble.math.expMapRot(np.array([0.6, -0.2, 0.03]))
        R_wc = nimble.math.expMapRot(np.array([0.1, 0.2, 0.3]))
        R_cp = R_wc.T @ R_wp
        acc_jc_p = np.array([0.0, 0.0, 1.0])
        acc_jc_c = R_cp @ acc_jc_p
        mag_jc_p = np.array([1.0, 0.0, 0.0])
        mag_jc_c = R_cp @ mag_jc_p
        baseline_error = self.filter.get_h(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c)
        np.testing.assert_array_almost_equal(baseline_error, np.zeros(6), decimal=8)

        H = self.filter.get_H_jacobian(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c)

        for col in range(6):
            epsilon = 1e-6
            perturbation = np.zeros(6)
            perturbation[col] = epsilon

            pos_measurements = self.filter.get_h(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, perturbation)
            neg_measurements = self.filter.get_h(R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, -perturbation)
            delta_measurements = (pos_measurements - neg_measurements) / (2 * epsilon)
            for row in range(6):
                self.assertAlmostEqual(H[row, col], delta_measurements[row], places=3,
                                       msg=f"Jacobian element {row, col} is incorrect.")

    def test_M_jacobian_identity(self):
        R_wp = np.eye(3)
        R_wc = np.eye(3)
        a_jc_p = np.array([0.0, 0.0, 1.0])
        a_jc_c = np.array([0.0, 0.0, 1.0])
        m_jc_p = np.array([1.0, 0.0, 0.0])
        m_jc_c = np.array([1.0, 0.0, 0.0])
        sensor_readings = np.hstack((a_jc_p, a_jc_c, m_jc_p, m_jc_c))

        M = self.filter.get_M_jacobian(R_wp, R_wc)

        for col in range(12):
            epsilon = 1e-6
            perturbation = np.zeros(12)
            perturbation[col] = epsilon

            pos_sensor = sensor_readings + perturbation
            neg_sensor = sensor_readings - perturbation

            pos_measurements = self.filter.get_h(R_wp, R_wc, pos_sensor[:3], pos_sensor[3:6], pos_sensor[6:9],
                                                 pos_sensor[9:12])
            neg_measurements = self.filter.get_h(R_wp, R_wc, neg_sensor[:3], neg_sensor[3:6], neg_sensor[6:9],
                                                 neg_sensor[9:12])
            delta_measurements = (pos_measurements - neg_measurements) / (2 * epsilon)
            for row in range(6):
                self.assertAlmostEqual(M[row, col], delta_measurements[row], places=3,
                                       msg=f"Jacobian element {row, col} is incorrect.")

    def test_time_update_same_rotation(self):
        initial_exp = np.zeros(3)
        gyro = np.array([0.01, 0.0, 0.0])
        dt = 1
        q_lin_wp = nimble.math.expToQuat(initial_exp)
        q_lin_wc = nimble.math.expToQuat(initial_exp)
        for i in range(10000):
            self.filter.set_qs(q_lin_wp, q_lin_wc)
            q_lin_wp, q_lin_wc = self.filter._get_time_update(gyro, gyro, dt)
            self.assertTrue(np.allclose(np.eye(3), self.filter.get_R_pc()), msg="R_pc is not identity after time update with same rotation.")

    def test_time_update_simple_rotation(self):
        # Start from identity and perform update with rotations around X-axis for child
        initial_exp = np.zeros(3)
        gyro = np.array([0.01, 0.0, 0.0])
        dt = 1
        q_lin_wp = nimble.math.expToQuat(initial_exp)
        q_lin_wc = nimble.math.expToQuat(initial_exp)
        for i in range(10000):
            self.filter.set_qs(q_lin_wp, q_lin_wc)
            q_lin_wp, q_lin_wc = self.filter._get_time_update(np.zeros(3), gyro, dt)
            self.assertTrue(np.allclose(q_lin_wp.wxyz(), nimble.math.expToQuat(initial_exp).wxyz()), msg="q_wp changed after time update with no rotation.")
            self.assertTrue(np.allclose(q_lin_wc.wxyz(), nimble.math.expToQuat(initial_exp + gyro * (i+1)).wxyz()), msg="q_wc is incorrect after time update with small rotation.")

    def test_time_update_random_rotation(self):
        # Start from identity and perform update with rotation about random axis
        initial_exp = np.zeros(3)
        final_exp = np.random.rand(3) * 0.01
        dt = 1
        gyro = final_exp / dt

        q_lin_wp = nimble.math.expToQuat(initial_exp)
        q_lin_wc = nimble.math.expToQuat(initial_exp)
        for i in range(10000):
            self.filter.set_qs(q_lin_wp, q_lin_wc)
            q_lin_wp, q_lin_wc = self.filter._get_time_update(np.zeros(3), gyro, dt)
            self.assertTrue(np.allclose(q_lin_wp.wxyz(), nimble.math.expToQuat(initial_exp).wxyz()), msg="q_wp changed after time update with no rotation.")
            self.assertTrue(np.allclose(q_lin_wc.wxyz(), nimble.math.expToQuat(final_exp * (i+1)).wxyz()), msg="q_wc is incorrect after time update with small rotation.")

    def test_measurement_update_perf_synth_data(self):
        q_lin_wp = nimble.math.expToQuat(np.array([np.pi / 2, 0., 0.]))  # 90 degrees around X-axis
        q_lin_wc = nimble.math.expToQuat(np.array([0., np.pi / 2, 0.]))  # 90 degrees around Y-axis
        acc_jc_p = q_lin_wp.to_rotation_matrix().T @ np.array([0.0, 0.0, 1.0])
        acc_jc_c = q_lin_wc.to_rotation_matrix().T @ np.array([0.0, 0.0, 1.0])
        mag_jc_p = q_lin_wp.to_rotation_matrix().T @ np.array([1.0, 0.0, 0.0])
        mag_jc_c = q_lin_wc.to_rotation_matrix().T @ np.array([1.0, 0.0, 0.0])

        updated_q_lin_wp, updated_q_lin_wc = self.filter._get_measurement_update(q_lin_wp, q_lin_wc, acc_jc_p, acc_jc_c,
                                                                                 mag_jc_p, mag_jc_c)
        updated_error = self.filter.get_h(updated_q_lin_wp.to_rotation_matrix(), updated_q_lin_wc.to_rotation_matrix(),
                                          acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c)

        np.testing.assert_array_almost_equal(updated_q_lin_wp.wxyz(), q_lin_wp.wxyz(), decimal=8)
        np.testing.assert_array_almost_equal(updated_q_lin_wc.wxyz(), q_lin_wc.wxyz(), decimal=8)
        np.testing.assert_array_almost_equal(updated_error, np.zeros(6), decimal=8)

    def test_measurement_update_offset_synth_data(self):
        q_lin_wp = nimble.math.expToQuat(np.array([np.pi / 2, 0., 0.]))  # 90 degrees around X-axis
        actual_q_lin_wc = nimble.math.expToQuat(np.array([0., np.pi / 2, 0.]))  # 90 degrees around Y-axis

        error_angle_axis = np.array([0.01, 0.01, 0.01])
        q_offset = nimble.math.expToQuat(error_angle_axis)
        noisy_q_lin_wc = actual_q_lin_wc.multiply(q_offset)

        acc_jc_p = q_lin_wp.to_rotation_matrix().T @ np.array([0.0, 0.0, 1.0])
        acc_jc_c = actual_q_lin_wc.to_rotation_matrix().T @ np.array([0.0, 0.0, 1.0])
        mag_jc_p = q_lin_wp.to_rotation_matrix().T @ np.array([1.0, 0.0, 0.0])
        mag_jc_c = actual_q_lin_wc.to_rotation_matrix().T @ np.array([1.0, 0.0, 0.0])

        initial_error = self.filter.get_h(q_lin_wp.to_rotation_matrix(), actual_q_lin_wc.to_rotation_matrix(),
                                          acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c)
        np.testing.assert_array_almost_equal(initial_error, np.zeros(6), decimal=8)
        offset_error = self.filter.get_h(q_lin_wp.to_rotation_matrix(), noisy_q_lin_wc.to_rotation_matrix(),
                                         acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c)

        updated_q_lin_wp, updated_q_lin_wc = self.filter._get_measurement_update(q_lin_wp, noisy_q_lin_wc, acc_jc_p,
                                                                                 acc_jc_c, mag_jc_p, mag_jc_c)

        updated_error = self.filter.get_h(updated_q_lin_wp.to_rotation_matrix(), updated_q_lin_wc.to_rotation_matrix(),
                                          acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c)

        np.testing.assert_array_less(abs(updated_error), abs(offset_error),
                                     "Error angle did not decrease after measurement update.")

    def test_update_no_motion(self):
        # Start from identity and perform update with no motion
        q_lin_wp = nimble.math.Quaternion(1., 0.0, 0.0, 0.0)
        q_lin_wc = nimble.math.Quaternion(1., 0.0, 0.0, 0.0)
        acc_jc_p = np.array([0.0, 0.0, 1.0])
        acc_jc_c = np.array([0.0, 0.0, 1.0])
        mag_jc_p = np.array([1.0, 0.0, 0.0])
        mag_jc_c = np.array([1.0, 0.0, 0.0])
        self.filter.set_qs(q_lin_wp, q_lin_wc)
        for i in range(100):
            self.filter.update(np.zeros(3), np.zeros(3), acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, 0.01)
        R_pc = self.filter.get_R_pc()
        self.assertTrue(np.allclose(R_pc, np.eye(3)), "R_pc is not identity after no motion update.")

        # Start from random orientation and perform update with no motion
        q_lin_wp = nimble.math.expToQuat(np.random.rand(3))
        q_lin_wc = nimble.math.expToQuat(np.random.rand(3))
        acc_jc_p = q_lin_wp.to_rotation_matrix().T @ np.array([0.0, 0.0, 1.0])
        acc_jc_c = q_lin_wc.to_rotation_matrix().T @ np.array([0.0, 0.0, 1.0])
        mag_jc_p = q_lin_wp.to_rotation_matrix().T @ np.array([1.0, 0.0, 0.0])
        mag_jc_c = q_lin_wc.to_rotation_matrix().T @ np.array([1.0, 0.0, 0.0])
        self.filter.set_qs(q_lin_wp, q_lin_wc)
        R_pc_original = self.filter.get_R_pc()
        for i in range(100):
            self.filter.update(np.zeros(3), np.zeros(3), acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c, 0.01)
        R_pc = self.filter.get_R_pc()
        self.assertTrue(np.allclose(R_pc, R_pc_original), "R_pc is not equal after no motion update.")

    def test_update_perfect_simple_rotation(self):
        # Start from identity and perform update with rotations around X-axis for child
        q_wp = nimble.math.Quaternion(1., 0.0, 0.0, 0.0)
        q_wc = nimble.math.Quaternion(1., 0.0, 0.0, 0.0)
        acc_jc_p = np.array([0.0, 0.0, 1.0])
        acc_jc_c = np.array([0.0, 0.0, 1.0])
        mag_jc_p = np.array([1.0, 0.0, 0.0])
        mag_jc_c = np.array([1.0, 0.0, 0.0])
        gyro_p = np.zeros(3)
        gyro_c = np.array([0.01, 0.0, 0.0])
        dt = 1
        self.filter.set_qs(q_wp, q_wc)
        for i in range(100):
            # Construct perfect accelerations and magnetometers for the test case
            expected_q_wc = nimble.math.expToQuat(gyro_c * (i+1))
            R_wc = expected_q_wc.to_rotation_matrix()
            self.filter.update(gyro_p, gyro_c, acc_jc_p, R_wc.T @ acc_jc_c, mag_jc_p, R_wc.T @ mag_jc_c, dt)
            # Make sure our update perfectly matches
            self.assertTrue(np.allclose(self.filter.q_wc.wxyz(), expected_q_wc.wxyz()), msg="q_wc is incorrect after update with small rotation.")

if __name__ == "__main__":
    unittest.main()
