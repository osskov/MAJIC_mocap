import unittest
import numpy as np
from src.toolchest.gyro_utils import (finite_difference_rotations,
                                      angular_velocity_to_rotation_matrix,
                                      angular_velocity_to_rotation_matrix_python,
                                      rotation_matrix_to_angular_velocity,
                                      rotation_matrix_to_angular_velocity_python,
                                      calculate_best_fit_rotation,
                                      integrate_rotations)
import nimblephysics as nimble


class TestGyroUtils(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.timestamps = np.array([0, 1, 2, 3, 4])

    def test_angular_velocity_to_rotation_matrix(self):
        omega = np.array([0.1, 0.2, 0.3])
        dt = 0.1
        expected_rotation_matrix = np.array([
            [0.99935008, - 0.02989301,  0.02014532],
             [0.03009299,  0.99950006, - 0.0096977],
            [-0.01984535,  0.01029763,  0.99975003]
        ])
        rotation_matrix = angular_velocity_to_rotation_matrix_python(omega, dt)
        np.testing.assert_array_almost_equal(rotation_matrix, expected_rotation_matrix, decimal=5)

    def test_nimble_equivalence(self):
        omega = np.array([0.1, 0.2, 0.3])
        dt = 0.1
        rotation_matrix = angular_velocity_to_rotation_matrix_python(omega, dt)
        nimble_rotation_matrix = angular_velocity_to_rotation_matrix(omega, dt)
        np.testing.assert_array_almost_equal(rotation_matrix, nimble_rotation_matrix, decimal=5)

    def test_angular_velocity_invertible(self):
        omega = np.array([0.1, 0.2, 0.3])
        dt = 0.1
        rotation_matrix = angular_velocity_to_rotation_matrix_python(omega, dt)
        rotation_matrix_inv = angular_velocity_to_rotation_matrix_python(-omega, dt)
        np.testing.assert_array_almost_equal(np.dot(rotation_matrix, rotation_matrix_inv), np.eye(3), decimal=5)

    def test_angular_velocity_nimble_equivalence(self):
        omega = np.array([0.1, 0.2, 0.3])
        dt = 0.1
        rotation_matrix = angular_velocity_to_rotation_matrix(omega, dt)
        rotation_matrix_inv = angular_velocity_to_rotation_matrix(-omega, dt)
        np.testing.assert_array_almost_equal(np.dot(rotation_matrix, rotation_matrix_inv), np.eye(3), decimal=5)

    def test_angular_velocity_round_trip(self):
        omega = np.array([0.1, 0.2, 0.3])
        dt = 0.1
        rotation_matrix = angular_velocity_to_rotation_matrix_python(omega, dt)
        omega_reconstructed = rotation_matrix_to_angular_velocity_python(rotation_matrix, dt)
        np.testing.assert_array_almost_equal(omega, omega_reconstructed, decimal=5)

    def test_nimble_angular_velocity_round_trip(self):
        omega = np.array([0.1, 0.2, 0.3])
        dt = 0.1
        rotation_matrix = angular_velocity_to_rotation_matrix(omega, dt)
        omega_reconstructed = rotation_matrix_to_angular_velocity(rotation_matrix, dt)
        np.testing.assert_array_almost_equal(omega, omega_reconstructed, decimal=5)

    def test_angular_velocity_zero(self):
        omega = np.array([0.0, 0.0, 0.0])
        dt = 0.1
        expected_rotation_matrix = np.eye(3)
        rotation_matrix = angular_velocity_to_rotation_matrix_python(omega, dt)
        np.testing.assert_array_equal(rotation_matrix, expected_rotation_matrix)

    def test_finite_difference_rotations_single_rotation(self):
        timestamps = np.array([0, 1])
        rotation_matrices = [np.eye(3), np.array([[0.9998477, -0.0174524, 0.0], [0.0174524, 0.9998477, 0.0], [0.0, 0.0, 1.0]])]
        expected_angular_velocities = [
            np.array([0.0, 0.0, 0.0174524]),
            np.array([0.0, 0.0, 0.0174524])
        ]
        angular_velocities = finite_difference_rotations(rotation_matrices, timestamps)
        for av, expected_av in zip(angular_velocities, expected_angular_velocities):
            np.testing.assert_array_almost_equal(av, expected_av, decimal=5)

    def test_angular_velocity_to_rotation_matrix_small_angle(self):
        omega = np.array([0.001, 0.002, 0.003])
        dt = 0.1
        expected_rotation_matrix = np.array([
            [0.999999, -0.000300, 0.000200],
            [0.000300, 0.999998, -0.000100],
            [-0.000200, 0.000100, 0.999999]
        ])
        rotation_matrix = angular_velocity_to_rotation_matrix_python(omega, dt)
        np.testing.assert_array_almost_equal(rotation_matrix, expected_rotation_matrix, decimal=5)

    def test_calculate_best_fit_rotation_same_vectors(self):
        parent_vectors = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]
        child_vectors = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]
        expected_rotation_identity = np.eye(3)
        assert np.allclose(calculate_best_fit_rotation(parent_vectors, child_vectors), expected_rotation_identity)

    def test_calculate_best_fit_rotation_rotated_vectors(self):
        # Test case where parent and child vectors are rotated versions of each other
        parent_vectors = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1])
        ]
        R_cp = nimble.math.eulerXYZToMatrix(np.array([np.pi / 3, np.pi / 3, np.pi / 3]))
        child_vectors = [R_cp @ v for v in parent_vectors]
        assert np.allclose(calculate_best_fit_rotation(parent_vectors, child_vectors), R_cp.T)

    def test_calculate_best_fit_rotation_non_unit_vectors(self):
        # Test case with arbitrary parent and child vectors
        parent_vectors = [
            np.array([1, 1, 1]),
            np.array([2, 3, 4]),
            np.array([0, 1, 2])
        ]
        R_cp = nimble.math.eulerXYZToMatrix(np.array([1, 0.5, 0.25]))
        child_vectors = [R_cp @ v for v in parent_vectors]
        assert np.allclose(calculate_best_fit_rotation(parent_vectors, child_vectors), R_cp.T)

        print("All test cases passed!")

    def test_finite_difference_integration_round_trip(self):
        R_initial_i = np.eye(3)
        timestamps = np.linspace(0, 1, 100)
        omega = [np.array([0.1, 0.2, 0.3]) * t for t in timestamps]
        dt = 0.1
        integrated_rotations = integrate_rotations(omega, timestamps, R_initial_i)
        finite_differenced_gyros = finite_difference_rotations(integrated_rotations, timestamps)
        for i in range(len(omega)-1):
            np.testing.assert_array_almost_equal(omega[i+1], finite_differenced_gyros[i], decimal=5)

if __name__ == '__main__':
    unittest.main()
