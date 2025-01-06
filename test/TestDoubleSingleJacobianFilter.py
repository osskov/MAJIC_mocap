import unittest
import numpy as np
from src.toolchest.DoubleSingleJacobianFilter import DoubleSingleJacobianFilter
from src.toolchest.gyro_utils import angular_velocity_to_rotation_matrix


class TestDoubleSingleJacobianFilter(unittest.TestCase):
    def setUp(self):
        self.filter = DoubleSingleJacobianFilter()

    def test_initialization(self):
        self.assertTrue(np.array_equal(self.filter.mag, np.zeros(3)))
        self.assertTrue(np.array_equal(self.filter.jac, np.zeros((3, 3))))
        self.assertEqual(self.filter.jac_col, 1)
        self.assertEqual(self.filter.new_data_weight_mag, 0.05)
        self.assertEqual(self.filter.new_data_weight_jac, 0.05)

    def test_update(self):
        gyro = np.array([0.1, 0.2, 0.3])
        dt = 0.1
        mag_avg = np.array([1.0, 2.0, 3.0])
        mag_grad = np.array([0.1, 0.2, 0.3])

        initial_mag = self.filter.get_mag_estimate().copy()
        initial_jac = self.filter.get_grad_estimate().copy()

        self.filter.update(gyro, dt, mag_avg, mag_grad)

        updated_mag = self.filter.get_mag_estimate()
        updated_jac = self.filter.get_grad_estimate()

        self.assertFalse(np.array_equal(initial_mag, updated_mag), "Magnetometer estimate should be updated")
        self.assertFalse(np.array_equal(initial_jac, updated_jac), "Jacobian gradient should be updated")

    def test_get_mag_estimate(self):
        mag = self.filter.get_mag_estimate()
        self.assertTrue(np.array_equal(mag, self.filter.mag))

    def test_get_grad_estimate(self):
        grad = self.filter.get_grad_estimate()
        self.assertTrue(np.array_equal(grad, self.filter.jac[:, self.filter.jac_col]))

    def test_simulated_data(self):
        filter = DoubleSingleJacobianFilter()
        filter.new_data_weight_jac = 0.5

        mag_world = np.random.randn(3)
        jac_world = np.random.randn(3,3)
        # Make sure the jacobian is symmetric
        jac_world = jac_world @ jac_world.T
        mag_1_offset = np.array([0.0, 0.1, 0.0])
        mag_2_offset = np.array([0.0, -0.1, 0.0])

        running_rotation = np.eye(3)
        for t in range(1000):
            gyro = np.random.randn(3)
            # gyro = np.zeros(3)
            dt = 0.01
            dR = angular_velocity_to_rotation_matrix(gyro, dt)
            running_rotation = running_rotation @ dR

            local_mag_1 = running_rotation.T @ (mag_world + jac_world @ (running_rotation @ mag_1_offset))
            local_mag_2 = running_rotation.T @ (mag_world + jac_world @ (running_rotation @ mag_2_offset))

            local_mag_1 += np.random.randn(3) * 0.05
            local_mag_2 += np.random.randn(3) * 0.05

            avg = (local_mag_1 + local_mag_2) / 2.0
            grad = (local_mag_1 - local_mag_2) / (mag_1_offset[1] - mag_2_offset[1])

            filter.update(gyro, dt, avg, grad)

        global_mag_estimate = running_rotation @ filter.get_mag_estimate()
        local_jac = running_rotation.T @ jac_world @ running_rotation

        print('Global mag estimate')
        print(global_mag_estimate)
        print('Global mag')
        print(mag_world)

        print('Local jac estimate')
        print(filter.jac)
        print('True jac in local frame')
        print(local_jac)

        true_grad = jac_world @ (running_rotation @ np.array([0.0, 1.0, 0.0]))
        grad = running_rotation @ filter.get_grad_estimate()

        print('True gradient')
        print(true_grad)
        print('Estimated gradient')
        print(grad)


if __name__ == '__main__':
    unittest.main()
