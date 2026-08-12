import unittest
import numpy as np
from src.toolchest.finite_difference_utils import (backward_difference, central_difference,
                                                   forward_difference, polynomial_fit_derivative)


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

    def test_backward_difference_pad(self):
        expected_gradient = np.array([1, 1, 3, 5, 7])
        gradient = backward_difference(self.signal, self.timesteps, edges='extend')
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=5)

    def test_backward_difference_zero(self):
        expected_gradient = np.array([0, 1, 3, 5, 7])
        gradient = backward_difference(self.signal, self.timesteps, edges='zero')
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=5)


class TestMultiColumnSupport(unittest.TestCase):
    """Every differentiator takes (N,) or (N, k) and treats columns independently, so
    IMUTrace._finite_difference_gyros can pass all three gyro axes in one call."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.t = 0.01 * np.arange(60)
        self.sig = rng.standard_normal((60, 3))

    def test_columns_match_per_axis_calls(self):
        for func, kwargs in ((central_difference, {}), (forward_difference, {}),
                             (backward_difference, {}),
                             (polynomial_fit_derivative, {'order': 2})):
            with self.subTest(func=func.__name__):
                together = func(self.sig, self.t, **kwargs)
                separately = np.column_stack(
                    [func(self.sig[:, axis], self.t, **kwargs) for axis in range(3)])
                self.assertEqual(together.shape, (60, 3))
                np.testing.assert_allclose(together, separately, atol=1e-12)


class TestCausality(unittest.TestCase):
    """Which differentiators read future samples. This is what decides whether the
    projection can run online, and it is the reason 'backward' is project_acc's default.
    """

    def setUp(self):
        rng = np.random.default_rng(1)
        self.t = 0.01 * np.arange(80)
        self.sig = rng.standard_normal(80)
        self.cut = 40          # perturb everything at or after this index

    def _lookahead(self, func, **kwargs):
        """Largest k such that changing sample t+k alters the output at t."""
        base = func(self.sig, self.t, **kwargs)
        perturbed_signal = self.sig.copy()
        perturbed_signal[self.cut:] += 10.0
        perturbed = func(perturbed_signal, self.t, **kwargs)
        differing = np.nonzero(~np.isclose(base, perturbed, atol=1e-12))[0]
        # the earliest affected sample sits `lookahead` before the perturbation
        return self.cut - int(differing.min())

    def test_backward_difference_reads_no_future_sample(self):
        self.assertEqual(self._lookahead(backward_difference, edges='zero'), 0)

    def test_central_and_forward_read_one_future_sample(self):
        self.assertEqual(self._lookahead(central_difference, edges='zero'), 1)
        self.assertEqual(self._lookahead(forward_difference, edges='zero'), 1)

    def test_polyfit_reads_most_of_a_window_ahead(self):
        self.assertEqual(
            self._lookahead(polynomial_fit_derivative, order=2, window_size=10), 9)


if __name__ == '__main__':
    unittest.main()
