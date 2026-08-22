"""Contract between src/relative_filter_fast.py and src/RelativeFilterPlus.py.

RelativeFilter is the reference: the paper's equations are annotated against it and
TestRelativeFilter.py exercises its internals. relative_filter_fast is a compiled
optimisation of the one configuration the pipeline runs. These two are separate
implementations of the same arithmetic, so this file is what stops them drifting --
if the reference changes, these tests fail until the kernel is brought back into line.

The tolerance is on the resulting relative orientation rather than on intermediate
matrices, because the kernel does not expose intermediates: the two are not bit-for-bit
(the 6x6 products accumulate in a different order, and the kernel solves via Cholesky
where the reference inverts), so what is being asserted is that the difference stays at
roundoff over a realistic number of steps rather than growing.
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.RelativeFilterPlus import RelativeFilter
from src import relative_filter_fast as rff

N_STEPS = 400
DT = 0.01
# The two implementations agree to ~1e-6 deg over a full 60k-sample trial; over the
# shorter runs here the divergence is far smaller. This bound is loose enough not to
# be flaky and tight enough that any real disagreement in the maths trips it.
TOL_DEG = 1e-4


def _trajectory(seed, zero_mag=False, gate_mag=False):
    """Synthetic but realistically scaled gyro / accelerometer / magnetometer."""
    rng = np.random.default_rng(seed)
    gyro_p = 0.8 * rng.standard_normal((N_STEPS, 3))
    gyro_c = 0.8 * rng.standard_normal((N_STEPS, 3))
    acc_p = np.array([0.0, 9.81, 0.0]) + 0.4 * rng.standard_normal((N_STEPS, 3))
    acc_c = np.array([0.0, 9.81, 0.0]) + 0.4 * rng.standard_normal((N_STEPS, 3))
    mag_p = np.array([0.5, 0.0, 0.85]) + 0.03 * rng.standard_normal((N_STEPS, 3))
    mag_c = np.array([0.5, 0.0, 0.85]) + 0.03 * rng.standard_normal((N_STEPS, 3))

    if zero_mag:                       # mag_mode 'off'
        mag_p[:] = 0.0
        mag_c[:] = 0.0
    elif gate_mag:                     # mag_mode 'adapt': zeroed on some samples only
        gated = rng.random(N_STEPS) < 0.4
        mag_p[gated] = 0.0
        mag_c[gated] = 0.0

    vp = np.stack([acc_p, mag_p], axis=1)
    vc = np.stack([acc_c, mag_c], axis=1)
    return gyro_p, gyro_c, vp, vc


def _reference(gyro_p, gyro_c, vp, vc, stds, R_wp0, R_wc0, normalize, init_std,
               heading_only=None, heading_pure=False):
    gyro_std_p, gyro_std_c, sp, sc = stds
    f = RelativeFilter(
        gyro_std_parent=gyro_std_p, gyro_std_child=gyro_std_c,
        vector_sensor_stds_parent=sp, vector_sensor_stds_child=sc,
        normalize_measurements=normalize, heading_only_sensor=heading_only,
        heading_only_pure=heading_pure, init_orientation_std=init_std)
    f.set_qs(Rotation.from_matrix(R_wp0), Rotation.from_matrix(R_wc0))
    out = np.empty((len(gyro_p), 3, 3))
    out[0] = f.get_R_pc()
    for t in range(1, len(gyro_p)):
        f.update(gyro_p[t - 1], gyro_c[t - 1],
                 [vp[t, 0], vp[t, 1]], [vc[t, 0], vc[t, 1]], DT)
        out[t] = f.get_R_pc()
    return out


def _max_angle_deg(A, B):
    M = np.einsum('tki,tkj->tij', A, B)
    cos = np.clip((np.trace(M, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).max())


@unittest.skipUnless(rff.NUMBA_AVAILABLE, "numba is not installed")
class TestFastKernelMatchesReference(unittest.TestCase):
    """Sweep the configurations the pipeline actually produces."""

    def _compare(self, seed=0, normalize=False, zero_mag=False, gate_mag=False,
                 gyro_std=0.0045, acc_std=0.018, mag_std=0.05,
                 init_std=np.deg2rad(0.1), heading_only=None, heading_pure=False):
        gyro_p, gyro_c, vp, vc = _trajectory(seed, zero_mag, gate_mag)
        stds = (np.ones(3) * gyro_std, np.ones(3) * gyro_std,
                [np.ones(3) * acc_std, np.ones(3) * mag_std],
                [np.ones(3) * acc_std, np.ones(3) * mag_std])
        R_wp0 = Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix()
        R_wc0 = Rotation.from_rotvec([-0.2, 0.05, 0.4]).as_matrix()

        expected = _reference(gyro_p, gyro_c, vp, vc, stds, R_wp0, R_wc0,
                              normalize, init_std, heading_only, heading_pure)
        actual = rff.run_relative_filter(
            gyro_p, gyro_c, vp, vc, DT,
            gyro_std_parent=stds[0], gyro_std_child=stds[1],
            vector_sensor_stds_parent=stds[2], vector_sensor_stds_child=stds[3],
            R_wp0=R_wp0, R_wc0=R_wc0,
            normalize_measurements=normalize, heading_only_sensor=heading_only,
            heading_only_pure=heading_pure, init_orientation_std=init_std)

        self.assertEqual(actual.shape, expected.shape)
        deviation = _max_angle_deg(expected, actual)
        self.assertLess(deviation, TOL_DEG,
                        f"kernel diverged from RelativeFilter by {deviation:.3e} deg")
        return deviation

    def test_matches_with_magnetometer(self):
        self._compare()

    def test_matches_with_magnetometer_zeroed(self):
        """mag_mode 'off': a zeroed sensor must drop out of the update entirely."""
        self._compare(zero_mag=True)

    def test_matches_with_magnetometer_gated_per_sample(self):
        """mag_mode 'adapt': the sensor switches off on a subset of samples."""
        self._compare(gate_mag=True)

    def test_matches_with_normalized_measurements(self):
        self._compare(normalize=True)

    def test_matches_with_normalized_and_magnetometer_zeroed(self):
        self._compare(normalize=True, zero_mag=True)

    def test_matches_with_the_magnetometer_restricted_to_heading(self):
        self._compare(heading_only=1)

    def test_matches_with_heading_only_and_the_magnetometer_zeroed(self):
        """The projection axis is undefined when the sensor is off, and both implementations
        have to drop the row rather than normalize a zero vector. Without the guard the kernel
        divides by zero here and every subsequent estimate is NaN."""
        self._compare(heading_only=1, zero_mag=True)

    def test_matches_with_heading_only_and_the_magnetometer_gated(self):
        self._compare(heading_only=1, gate_mag=True)

    def test_matches_with_heading_only_and_normalized_measurements(self):
        self._compare(heading_only=1, normalize=True)

    def test_matches_with_the_pure_heading_attribution(self):
        self._compare(heading_only=1, heading_pure=True)

    def test_matches_with_pure_heading_and_the_magnetometer_zeroed(self):
        self._compare(heading_only=1, heading_pure=True, zero_mag=True)

    def test_matches_with_pure_heading_and_normalized_measurements(self):
        self._compare(heading_only=1, heading_pure=True, normalize=True)

    def test_matches_across_seeds(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                self._compare(seed=seed)

    def test_matches_across_tunings(self):
        for gyro_std, acc_std, mag_std in ((0.0045, 0.018, 0.05),
                                           (0.0116, 0.03, 0.05),
                                           (0.001, 0.1, 0.01),
                                           (0.05, 0.005, 0.2)):
            with self.subTest(gyro=gyro_std, acc=acc_std, mag=mag_std):
                self._compare(gyro_std=gyro_std, acc_std=acc_std, mag_std=mag_std)

    def test_matches_with_a_loose_initial_covariance(self):
        """init_orientation_std drives the startup transient, so it exercises a very
        different part of the update than the converged steady state does."""
        self._compare(init_std=1.0)

    def test_first_sample_is_the_seeded_state(self):
        """R_pc[0] must be the seed, untouched -- the pipeline's index convention
        depends on it (experiment_utils._run_relative_filter)."""
        gyro_p, gyro_c, vp, vc = _trajectory(0)
        R_wp0 = Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix()
        R_wc0 = Rotation.from_rotvec([-0.2, 0.05, 0.4]).as_matrix()
        actual = rff.run_relative_filter(
            gyro_p, gyro_c, vp, vc, DT,
            gyro_std_parent=np.ones(3) * 0.0045, gyro_std_child=np.ones(3) * 0.0045,
            vector_sensor_stds_parent=[np.ones(3) * 0.018, np.ones(3) * 0.05],
            vector_sensor_stds_child=[np.ones(3) * 0.018, np.ones(3) * 0.05],
            R_wp0=R_wp0, R_wc0=R_wc0)
        np.testing.assert_allclose(actual[0], R_wp0.T @ R_wc0, atol=1e-12)


@unittest.skipUnless(rff.NUMBA_AVAILABLE, "numba is not installed")
class TestFastKernelRefusesUnsupportedConfigurations(unittest.TestCase):
    """The kernel covers two vector sensors and no joint constraint. Anything else
    must raise rather than quietly return the wrong filter's answer."""

    def _call(self, n_sensors=2, sensors_shape=None, heading_only=None):
        gyro = np.zeros((10, 3))
        vp = np.zeros(sensors_shape or (10, 2, 3))
        return rff.run_relative_filter(
            gyro, gyro, vp, vp, DT,
            gyro_std_parent=np.ones(3), gyro_std_child=np.ones(3),
            vector_sensor_stds_parent=[np.ones(3)] * n_sensors,
            vector_sensor_stds_child=[np.ones(3)] * n_sensors,
            R_wp0=np.eye(3), R_wc0=np.eye(3), heading_only_sensor=heading_only)

    def test_rejects_heading_only_on_the_reference_sensor(self):
        """Sensor 0 defines the axis, so restricting it to that axis would delete it."""
        for bad in (0, 2, -1):
            with self.assertRaises(ValueError):
                self._call(heading_only=bad)

    def test_rejects_a_different_sensor_count(self):
        with self.assertRaises(NotImplementedError):
            self._call(n_sensors=3)
        with self.assertRaises(NotImplementedError):
            self._call(n_sensors=1)

    def test_rejects_mismatched_sensor_array_shape(self):
        with self.assertRaises(ValueError):
            self._call(sensors_shape=(10, 3, 3))


if __name__ == '__main__':
    unittest.main()
