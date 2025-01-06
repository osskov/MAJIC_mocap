import unittest
import numpy as np
from src.toolchest.finite_difference_utils import central_difference, forward_difference, polynomial_fit_derivative


class TestDifferentiationMethods(unittest.TestCase):
    def setUp(self):
        self.signal = np.array([0, 1, 4, 9, 16])
        self.timesteps = np.array([0, 1, 2, 3, 4])

    def test_central_difference_pad(self):
        expected_gradient = np.array([2, 2, 4, 6, 6])
        gradient = central_difference(self.signal, self.timesteps, edges='extend')
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=5)

    def test_central_difference_zero(self):
        expected_gradient = np.array([0, 2, 4, 6, 0])
        gradient = central_difference(self.signal, self.timesteps, edges='zero')
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=5)

    def test_forward_difference_pad(self):
        expected_gradient = np.array([1, 3, 5, 7, 7])
        gradient = forward_difference(self.signal, self.timesteps, edges='extend')
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=5)

    def test_forward_difference_zero(self):
        expected_gradient = np.array([1, 3, 5, 7, 0])
        gradient = forward_difference(self.signal, self.timesteps, edges='zero')
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=5)

    def test_polynomial_fit_derivative(self):
        flat_signal = np.array([0, 1, 2, 3, 4])
        gradient = polynomial_fit_derivative(flat_signal, self.timesteps, order=3, window_size=4)
        expected_gradient = np.array([1, 1, 1, 1, 1])
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=5)

    def test_polynomial_fit_flat_2(self):
        flat_signal = np.array([0, 2, 4, 6, 8])
        gradient = polynomial_fit_derivative(flat_signal, self.timesteps, order=3, window_size=4)
        expected_gradient = np.array([2, 2, 2, 2, 2])
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=5)

    def test_polynomial_fit_polynomial(self):
        flat_signal = self.timesteps ** 2
        gradient = polynomial_fit_derivative(flat_signal, self.timesteps, order=3, window_size=4)
        expected_gradient = 2 * self.timesteps
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=5)


if __name__ == '__main__':
    unittest.main()
