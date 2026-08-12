import inspect
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.RelativeFilterPlus import RelativeFilter

GYRO_STD = np.ones(3) * 0.05
ACC_STD = np.ones(3) * 0.5
MAG_STD = np.ones(3) * 0.3

# World-frame references the synthetic measurements are built from.
WORLD_GRAVITY = np.array([0.0, 0.0, 1.0])
WORLD_MAG = np.array([1.0, 0.0, 0.0])


def make_filter(**kwargs) -> RelativeFilter:
    """Two vector sensors per body (accelerometer + magnetometer), as the pipeline uses."""
    return RelativeFilter(
        gyro_std_parent=GYRO_STD,
        gyro_std_child=GYRO_STD,
        vector_sensor_stds_parent=[ACC_STD, MAG_STD],
        vector_sensor_stds_child=[ACC_STD, MAG_STD],
        **kwargs
    )


def consistent_measurements(R_wp, R_wc):
    """Accelerometer and magnetometer readings that agree perfectly on both bodies."""
    data_p = [R_wp.T @ WORLD_GRAVITY, R_wp.T @ WORLD_MAG]
    data_c = [R_wc.T @ WORLD_GRAVITY, R_wc.T @ WORLD_MAG]
    return data_p, data_c


def numerical_H(filt, R_wp, R_wc, data_p, data_c, eps=1e-6):
    """dh/d_eta by central difference.

    eta is perturbed the way the filter applies its correction in
    _get_measurement_update: as a right-multiplied (body-frame) error rotation.
    """
    rows = len(filt.get_h(R_wp, R_wc, data_p, data_c))
    numerical = np.zeros((rows, 6))
    for col in range(6):
        delta = np.zeros(6)
        delta[col] = eps
        plus = filt.get_h(R_wp @ Rotation.from_rotvec(delta[:3]).as_matrix(),
                          R_wc @ Rotation.from_rotvec(delta[3:]).as_matrix(), data_p, data_c)
        minus = filt.get_h(R_wp @ Rotation.from_rotvec(-delta[:3]).as_matrix(),
                           R_wc @ Rotation.from_rotvec(-delta[3:]).as_matrix(), data_p, data_c)
        numerical[:, col] = (plus - minus) / (2 * eps)
    return numerical


def numerical_M(filt, R_wp, R_wc, data_p, data_c, eps=1e-6):
    """dh/d_nu by central difference over the sensor readings.

    The noise vector is ordered per sensor as [parent, child], matching the layout
    get_M_jacobian builds.
    """
    rows = len(filt.get_h(R_wp, R_wc, data_p, data_c))
    num_sensors = len(data_p)
    numerical = np.zeros((rows, 6 * num_sensors))
    for sensor in range(num_sensors):
        for axis in range(3):
            for side, offset in (('p', 0), ('c', 3)):
                bump = np.zeros(3)
                bump[axis] = eps
                plus_p = [v.copy() for v in data_p]
                plus_c = [v.copy() for v in data_c]
                minus_p = [v.copy() for v in data_p]
                minus_c = [v.copy() for v in data_c]
                if side == 'p':
                    plus_p[sensor] = plus_p[sensor] + bump
                    minus_p[sensor] = minus_p[sensor] - bump
                else:
                    plus_c[sensor] = plus_c[sensor] + bump
                    minus_c[sensor] = minus_c[sensor] - bump
                plus = filt.get_h(R_wp, R_wc, plus_p, plus_c)
                minus = filt.get_h(R_wp, R_wc, minus_p, minus_c)
                numerical[:, 6 * sensor + offset + axis] = (plus - minus) / (2 * eps)
    return numerical


class TestRelativeFilterSetup(unittest.TestCase):
    def test_initial_state(self):
        filt = make_filter()
        np.testing.assert_allclose(filt.q_wp.as_matrix(), np.eye(3))
        np.testing.assert_allclose(filt.q_wc.as_matrix(), np.eye(3))
        np.testing.assert_allclose(filt.get_R_pc(), np.eye(3))

        # Read the default rather than pin it: the value is a tuning choice that may be
        # re-swept, but P always has to be built from it. It must stay far below eye(6) —
        # see test_a_confident_seed_is_not_yanked_by_one_bad_sample for why.
        default_std = inspect.signature(RelativeFilter).parameters['init_orientation_std'].default
        np.testing.assert_allclose(filt.P, np.eye(6) * default_std ** 2)
        self.assertLess(default_std, np.deg2rad(5.0))

    def test_init_orientation_std_sets_p(self):
        filt = make_filter(init_orientation_std=np.deg2rad(30.0))
        np.testing.assert_allclose(filt.P, np.eye(6) * np.deg2rad(30.0) ** 2)

    def test_process_noise_is_gyro_variance(self):
        filt = make_filter()
        expected = np.diag(np.concatenate([GYRO_STD, GYRO_STD]) ** 2)
        np.testing.assert_allclose(filt.Q, expected)

    def test_measurement_noise_interleaves_parent_and_child_per_sensor(self):
        filt = make_filter()
        # Order is [acc_parent, acc_child, mag_parent, mag_child], three axes each.
        expected = np.diag(np.concatenate([ACC_STD, ACC_STD, MAG_STD, MAG_STD]) ** 2)
        self.assertEqual(filt.R.shape, (12, 12))
        np.testing.assert_allclose(filt.R, expected)

    def test_get_q_pc_is_the_relative_rotation(self):
        filt = make_filter()
        q_wp = Rotation.from_rotvec([0.6, -0.2, 0.03])
        q_wc = Rotation.from_rotvec([0.1, 0.2, 0.3])
        filt.set_qs(q_wp, q_wc)

        np.testing.assert_allclose(filt.get_R_pc(), q_wp.as_matrix().T @ q_wc.as_matrix(), atol=1e-12)
        np.testing.assert_allclose(filt.get_q_pc().as_matrix(), filt.get_R_pc(), atol=1e-12)

    def test_rejects_bad_gyro_std_shape(self):
        with self.assertRaises(ValueError):
            RelativeFilter(gyro_std_parent=np.ones(4), gyro_std_child=GYRO_STD,
                           vector_sensor_stds_parent=[ACC_STD], vector_sensor_stds_child=[ACC_STD])
        with self.assertRaises(ValueError):
            RelativeFilter(gyro_std_parent=GYRO_STD, gyro_std_child=np.ones(2),
                           vector_sensor_stds_parent=[ACC_STD], vector_sensor_stds_child=[ACC_STD])

    def test_rejects_mismatched_sensor_counts(self):
        with self.assertRaises(ValueError):
            RelativeFilter(gyro_std_parent=GYRO_STD, gyro_std_child=GYRO_STD,
                           vector_sensor_stds_parent=[ACC_STD, MAG_STD], vector_sensor_stds_child=[ACC_STD])

    def test_rejects_bad_sensor_std_shape(self):
        with self.assertRaises(ValueError):
            RelativeFilter(gyro_std_parent=GYRO_STD, gyro_std_child=GYRO_STD,
                           vector_sensor_stds_parent=[np.ones(2)], vector_sensor_stds_child=[ACC_STD])

    def test_rejects_unknown_joint_type(self):
        with self.assertRaises(ValueError):
            make_filter(joint_type='3dof')

    def test_rejects_incomplete_joint_parameters(self):
        with self.assertRaises(ValueError):
            make_filter(joint_type='1dof', dof1_axis_parent=np.array([0., 0., 1.]))
        with self.assertRaises(ValueError):
            make_filter(joint_type='2dof', dof2_axis_parent=np.array([0., 0., 1.]))

    def test_joint_axes_are_normalized(self):
        filt = make_filter(joint_type='1dof',
                           dof1_axis_parent=np.array([0., 0., 4.]),
                           dof1_axis_child=np.array([0., 3., 0.]),
                           dof1_std=np.ones(3) * 0.1)
        np.testing.assert_allclose(filt.joint_params['y_j'], np.array([0., 0., 1.]))
        np.testing.assert_allclose(filt.joint_params['y_k'], np.array([0., 1., 0.]))
        self.assertEqual(filt.joint_noise_dim, 3)
        self.assertEqual(filt.R.shape, (15, 15))


class TestRelativeFilterJacobians(unittest.TestCase):
    def setUp(self):
        self.filter = make_filter()

    def test_residual_is_zero_for_consistent_measurements(self):
        R_wp = Rotation.from_rotvec([0.6, -0.2, 0.03]).as_matrix()
        R_wc = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
        data_p, data_c = consistent_measurements(R_wp, R_wc)
        np.testing.assert_array_almost_equal(self.filter.get_h(R_wp, R_wc, data_p, data_c), np.zeros(6), decimal=12)

    def test_residual_is_the_world_frame_disagreement(self):
        R_wp = Rotation.from_rotvec([0.6, -0.2, 0.03]).as_matrix()
        R_wc = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
        data_p = [np.array([0.1, 0.2, 0.97]), np.array([0.9, 0.1, 0.0])]
        data_c = [np.array([0.3, -0.4, 0.86]), np.array([0.7, -0.3, 0.2])]

        h = self.filter.get_h(R_wp, R_wc, data_p, data_c)
        expected = np.concatenate([R_wp @ data_p[i] - R_wc @ data_c[i] for i in range(2)])
        np.testing.assert_allclose(h, expected)

    def test_measurement_jacobian_matches_numerical_at_identity(self):
        R_wp = np.eye(3)
        R_wc = np.eye(3)
        data_p, data_c = consistent_measurements(R_wp, R_wc)

        H = self.filter.get_H_jacobian(R_wp, R_wc, data_p, data_c)
        np.testing.assert_allclose(H, numerical_H(self.filter, R_wp, R_wc, data_p, data_c), atol=1e-6)

    def test_measurement_jacobian_matches_numerical_when_rotated(self):
        R_wp = Rotation.from_rotvec([0.6, -0.2, 0.03]).as_matrix()
        R_wc = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
        data_p, data_c = consistent_measurements(R_wp, R_wc)

        H = self.filter.get_H_jacobian(R_wp, R_wc, data_p, data_c)
        np.testing.assert_allclose(H, numerical_H(self.filter, R_wp, R_wc, data_p, data_c), atol=1e-6)

    def test_measurement_jacobian_matches_numerical_off_the_solution(self):
        # The Jacobian has to be right where the residual is non-zero too -- that is the
        # only place it does any work.
        R_wp = Rotation.from_rotvec([0.6, -0.2, 0.03]).as_matrix()
        R_wc = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
        data_p = [np.array([0.1, 0.2, 0.97]), np.array([0.9, 0.1, 0.0])]
        data_c = [np.array([0.3, -0.4, 0.86]), np.array([0.7, -0.3, 0.2])]

        H = self.filter.get_H_jacobian(R_wp, R_wc, data_p, data_c)
        np.testing.assert_allclose(H, numerical_H(self.filter, R_wp, R_wc, data_p, data_c), atol=1e-6)

    def test_noise_jacobian_matches_numerical(self):
        R_wp = Rotation.from_rotvec([0.6, -0.2, 0.03]).as_matrix()
        R_wc = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
        data_p, data_c = consistent_measurements(R_wp, R_wc)

        M = self.filter.get_M_jacobian(R_wp, R_wc)
        self.assertEqual(M.shape, (6, 12))
        np.testing.assert_allclose(M, numerical_M(self.filter, R_wp, R_wc, data_p, data_c), atol=1e-6)

    def test_1dof_joint_extends_the_residual_and_jacobian(self):
        y_parent = np.array([0., 0., 1.])
        y_child = np.array([0., 1., 0.])
        filt = make_filter(joint_type='1dof', dof1_axis_parent=y_parent, dof1_axis_child=y_child,
                           dof1_std=np.ones(3) * 0.1)
        R_wp = Rotation.from_rotvec([0.6, -0.2, 0.03]).as_matrix()
        R_wc = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
        data_p, data_c = consistent_measurements(R_wp, R_wc)

        h = filt.get_h(R_wp, R_wc, data_p, data_c)
        self.assertEqual(h.shape, (9,))
        # The joint block is the world-frame disagreement between the two hinge axes.
        np.testing.assert_allclose(h[6:], R_wp @ y_parent - R_wc @ y_child)

        H = filt.get_H_jacobian(R_wp, R_wc, data_p, data_c)
        self.assertEqual(H.shape, (9, 6))
        np.testing.assert_allclose(H, numerical_H(filt, R_wp, R_wc, data_p, data_c), atol=1e-6)

        M = filt.get_M_jacobian(R_wp, R_wc)
        self.assertEqual(M.shape, (9, 15))
        # The joint noise is additive, so its block is the identity and it does not mix
        # with the vector-sensor noise.
        np.testing.assert_allclose(M[6:, 12:], np.eye(3))
        np.testing.assert_allclose(M[6:, :12], np.zeros((3, 12)))
        np.testing.assert_allclose(M[:6, 12:], np.zeros((6, 3)))

    def test_1dof_joint_residual_vanishes_when_the_axes_coincide(self):
        axis = np.array([0., 0., 1.])
        filt = make_filter(joint_type='1dof', dof1_axis_parent=axis, dof1_axis_child=axis,
                           dof1_std=np.ones(3) * 0.1)
        # A relative rotation about the shared hinge axis satisfies the constraint.
        R_wp = Rotation.from_rotvec([0.6, -0.2, 0.03]).as_matrix()
        R_wc = R_wp @ Rotation.from_rotvec(0.4 * axis).as_matrix()
        data_p, data_c = consistent_measurements(R_wp, R_wc)

        h = filt.get_h(R_wp, R_wc, data_p, data_c)
        np.testing.assert_array_almost_equal(h[6:], np.zeros(3), decimal=12)

        # Rotating about anything else breaks it.
        R_wc_bad = R_wp @ Rotation.from_rotvec([0.4, 0., 0.]).as_matrix()
        data_p_bad, data_c_bad = consistent_measurements(R_wp, R_wc_bad)
        h_bad = filt.get_h(R_wp, R_wc_bad, data_p_bad, data_c_bad)
        self.assertGreater(np.linalg.norm(h_bad[6:]), 0.1)

    def test_2dof_joint_residual_and_shapes(self):
        v_parent = np.array([0., 0., 1.])
        u_child = np.array([0., 1., 0.])
        alpha = np.pi / 3
        filt = make_filter(joint_type='2dof', dof2_axis_parent=v_parent, dof2_axis_child=u_child,
                           dof2_angle_rad=alpha, dof2_std=0.1)
        self.assertEqual(filt.joint_noise_dim, 1)
        self.assertEqual(filt.R.shape, (13, 13))

        R_wp = Rotation.from_rotvec([0.6, -0.2, 0.03]).as_matrix()
        R_wc = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
        data_p, data_c = consistent_measurements(R_wp, R_wc)

        h = filt.get_h(R_wp, R_wc, data_p, data_c)
        self.assertEqual(h.shape, (7,))
        expected_scalar = (R_wp @ v_parent).dot(R_wc @ u_child) - np.sin(alpha)
        self.assertAlmostEqual(float(h[6]), float(expected_scalar), places=12)

        H = filt.get_H_jacobian(R_wp, R_wc, data_p, data_c)
        self.assertEqual(H.shape, (7, 6))
        M = filt.get_M_jacobian(R_wp, R_wc)
        self.assertEqual(M.shape, (7, 13))
        np.testing.assert_allclose(M[6:, 12:], np.eye(1))

    def test_rejects_a_filter_with_nothing_to_measure(self):
        # No vector sensors and no joint constraint leaves the relative orientation
        # unobservable, so construction must fail rather than run blind.
        with self.assertRaises(ValueError):
            RelativeFilter(gyro_std_parent=GYRO_STD, gyro_std_child=GYRO_STD,
                           vector_sensor_stds_parent=[], vector_sensor_stds_child=[])


class TestRelativeFilterTimeUpdate(unittest.TestCase):
    def setUp(self):
        self.filter = make_filter()

    def test_identical_gyros_keep_the_relative_rotation_fixed(self):
        gyro = np.array([0.01, 0.0, 0.0])
        dt = 1.0
        q_wp = Rotation.identity()
        q_wc = Rotation.identity()
        for _ in range(500):
            self.filter.set_qs(q_wp, q_wc)
            q_wp, q_wc = self.filter._get_time_update(gyro, gyro, dt)
            self.filter.set_qs(q_wp, q_wc)
            np.testing.assert_allclose(self.filter.get_R_pc(), np.eye(3), atol=1e-10)

    def test_child_only_rotation_integrates_on_the_gyro_axis(self):
        gyro = np.array([0.01, 0.0, 0.0])
        dt = 1.0
        q_wp = Rotation.identity()
        q_wc = Rotation.identity()
        for i in range(500):
            self.filter.set_qs(q_wp, q_wc)
            q_wp, q_wc = self.filter._get_time_update(np.zeros(3), gyro, dt)
            np.testing.assert_allclose(q_wp.as_matrix(), np.eye(3), atol=1e-12,
                                       err_msg="q_wp moved with no parent rotation.")
            expected = Rotation.from_rotvec(gyro * dt * (i + 1))
            np.testing.assert_allclose(q_wc.as_matrix(), expected.as_matrix(), atol=1e-10)

    def test_random_axis_rotation_integrates(self):
        np.random.seed(4)
        dt = 0.01
        gyro = np.random.rand(3)
        q_wp = Rotation.identity()
        q_wc = Rotation.identity()
        for i in range(200):
            self.filter.set_qs(q_wp, q_wc)
            q_wp, q_wc = self.filter._get_time_update(np.zeros(3), gyro, dt)
            expected = Rotation.from_rotvec(gyro * dt * (i + 1))
            np.testing.assert_allclose(q_wc.as_matrix(), expected.as_matrix(), atol=1e-10)

    def test_covariance_grows_with_process_noise(self):
        dt = 0.01
        before = self.filter.P.copy()
        self.filter._get_time_update(np.zeros(3), np.zeros(3), dt)
        # With no rotation the propagation is the identity, so P grows by exactly dt^2 * Q.
        np.testing.assert_allclose(self.filter.P, before + dt ** 2 * self.filter.Q, atol=1e-12)


class TestRelativeFilterMeasurementUpdate(unittest.TestCase):
    def setUp(self):
        self.filter = make_filter()

    def test_perfect_measurements_leave_the_state_alone(self):
        q_wp = Rotation.from_rotvec([np.pi / 2, 0., 0.])
        q_wc = Rotation.from_rotvec([0., np.pi / 2, 0.])
        data_p, data_c = consistent_measurements(q_wp.as_matrix(), q_wc.as_matrix())

        updated_wp, updated_wc = self.filter._get_measurement_update(q_wp, q_wc, data_p, data_c)

        np.testing.assert_array_almost_equal(updated_wp.as_matrix(), q_wp.as_matrix(), decimal=10)
        np.testing.assert_array_almost_equal(updated_wc.as_matrix(), q_wc.as_matrix(), decimal=10)
        residual = self.filter.get_h(updated_wp.as_matrix(), updated_wc.as_matrix(), data_p, data_c)
        np.testing.assert_array_almost_equal(residual, np.zeros(6), decimal=10)

    def test_measurement_update_shrinks_the_residual(self):
        q_wp = Rotation.from_rotvec([np.pi / 2, 0., 0.])
        true_q_wc = Rotation.from_rotvec([0., np.pi / 2, 0.])
        # Measurements describe the true orientation; the filter's estimate is off by a
        # small rotation, which the update should pull back.
        data_p, data_c = consistent_measurements(q_wp.as_matrix(), true_q_wc.as_matrix())
        drifted_q_wc = true_q_wc * Rotation.from_rotvec([0.01, 0.01, 0.01])

        before = self.filter.get_h(q_wp.as_matrix(), drifted_q_wc.as_matrix(), data_p, data_c)
        updated_wp, updated_wc = self.filter._get_measurement_update(q_wp, drifted_q_wc, data_p, data_c)
        after = self.filter.get_h(updated_wp.as_matrix(), updated_wc.as_matrix(), data_p, data_c)

        self.assertLess(np.linalg.norm(after), np.linalg.norm(before))
        # And it should move the child estimate towards the truth, not away from it.
        error_before = (true_q_wc.inv() * drifted_q_wc).magnitude()
        error_after = (true_q_wc.inv() * updated_wc).magnitude()
        self.assertLess(error_after, error_before)

    def test_measurement_update_shrinks_the_covariance(self):
        q_wp = Rotation.identity()
        q_wc = Rotation.identity()
        data_p, data_c = consistent_measurements(q_wp.as_matrix(), q_wc.as_matrix())

        trace_before = np.trace(self.filter.P)
        self.filter._get_measurement_update(q_wp, q_wc, data_p, data_c)
        self.assertLess(np.trace(self.filter.P), trace_before)


class TestNormalizedMeasurements(unittest.TestCase):
    """normalize_measurements scales each vector sensor reading to unit length before
    the update. Every failure here is silent: a wrong flag default, a NaN from the
    zeroed magnetometer, or a scaling that also moves R would all still produce
    finite rotations of the right shape."""

    def test_normalization_is_off_by_default(self):
        self.assertFalse(make_filter().normalize_measurements)

    def test_readings_reach_the_update_at_unit_length(self):
        scaled = [9.81 * WORLD_GRAVITY, 40.0 * WORLD_MAG]
        normalized = RelativeFilter._normalize_vector_measurements(scaled)
        for v in normalized:
            self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=12)
        np.testing.assert_allclose(normalized[0], WORLD_GRAVITY, atol=1e-12)
        np.testing.assert_allclose(normalized[1], WORLD_MAG, atol=1e-12)

    def test_a_zeroed_sensor_stays_zero_instead_of_going_nan(self):
        """mag_off zeroes the magnetometer on every sample and mag_adapt on gated ones,
        so the zero vector is the normal path here, not an edge case. A NaN would
        propagate through K and destroy the state for the rest of the trial."""
        normalized = RelativeFilter._normalize_vector_measurements(
            [9.81 * WORLD_GRAVITY, np.zeros(3)])
        np.testing.assert_array_equal(normalized[1], np.zeros(3))
        self.assertFalse(np.any(np.isnan(np.concatenate(normalized))))

    def test_a_zeroed_magnetometer_still_runs_a_finite_update(self):
        filt = make_filter(normalize_measurements=True)
        filt.set_qs(Rotation.identity(), Rotation.identity())
        for _ in range(10):
            filt.update(np.zeros(3), np.zeros(3),
                        [9.81 * WORLD_GRAVITY, np.zeros(3)],
                        [9.81 * WORLD_GRAVITY, np.zeros(3)], 0.01)
        self.assertTrue(np.all(np.isfinite(filt.get_R_pc())))
        np.testing.assert_allclose(filt.get_R_pc(), np.eye(3), atol=1e-8)

    def test_it_is_a_no_op_on_already_unit_measurements(self):
        """The synthetic references are unit vectors, so both arms must agree exactly
        here — this pins that the flag changes the measurement scale and nothing else."""
        q_wp = Rotation.from_rotvec([np.pi / 2, 0., 0.])
        q_wc = Rotation.from_rotvec([0., np.pi / 2, 0.])
        data_p, data_c = consistent_measurements(q_wp.as_matrix(), q_wc.as_matrix())

        plain = make_filter()._get_measurement_update(q_wp, q_wc, data_p, data_c)
        normalized = make_filter(normalize_measurements=True)._get_measurement_update(
            q_wp, q_wc, data_p, data_c)

        for a, b in zip(plain, normalized):
            np.testing.assert_allclose(a.as_matrix(), b.as_matrix(), atol=1e-12)

    def test_rescaling_the_std_by_the_nominal_restores_the_raw_update(self):
        """The premise of the '_rescaled' arm: normalizing and dividing the std by the
        same magnitude leaves the update untouched, because h, H and R then all scale
        together. Verified at |a| exactly nominal — where the two are algebraically
        identical — so any difference the arm shows on real data is the per-sample
        magnitude variation it is designed to isolate, not a weighting change."""
        nominal = 9.81
        q_wp = Rotation.identity()
        drifted_q_wc = Rotation.from_rotvec([0.05, 0.05, 0.05])
        raw_p = [nominal * WORLD_GRAVITY, WORLD_MAG]
        raw_c = [nominal * WORLD_GRAVITY, WORLD_MAG]

        plain = RelativeFilter(
            gyro_std_parent=GYRO_STD, gyro_std_child=GYRO_STD,
            vector_sensor_stds_parent=[ACC_STD, MAG_STD],
            vector_sensor_stds_child=[ACC_STD, MAG_STD],
        )._get_measurement_update(q_wp, drifted_q_wc, raw_p, raw_c)

        rescaled = RelativeFilter(
            gyro_std_parent=GYRO_STD, gyro_std_child=GYRO_STD,
            vector_sensor_stds_parent=[ACC_STD / nominal, MAG_STD],
            vector_sensor_stds_child=[ACC_STD / nominal, MAG_STD],
            normalize_measurements=True,
        )._get_measurement_update(q_wp, drifted_q_wc, raw_p, raw_c)

        for a, b in zip(plain, rescaled):
            np.testing.assert_allclose(a.as_matrix(), b.as_matrix(), atol=1e-10)

    def test_rescaling_still_differs_once_the_magnitude_leaves_nominal(self):
        """The other half of the premise. If the rescaled arm matched the control at ALL
        magnitudes it would be an exact no-op and the experiment would be measuring
        nothing — the arm only means something because off-nominal samples still differ."""
        nominal = 9.81
        q_wp = Rotation.identity()
        drifted_q_wc = Rotation.from_rotvec([0.05, 0.05, 0.05])
        # A hard foot-strike: acc well above gravity, which is exactly the sample the
        # normalized arms throw information away about.
        raw_p = [3.0 * nominal * WORLD_GRAVITY, WORLD_MAG]
        raw_c = [3.0 * nominal * WORLD_GRAVITY, WORLD_MAG]

        plain = RelativeFilter(
            gyro_std_parent=GYRO_STD, gyro_std_child=GYRO_STD,
            vector_sensor_stds_parent=[ACC_STD, MAG_STD],
            vector_sensor_stds_child=[ACC_STD, MAG_STD],
        )._get_measurement_update(q_wp, drifted_q_wc, raw_p, raw_c)

        rescaled = RelativeFilter(
            gyro_std_parent=GYRO_STD, gyro_std_child=GYRO_STD,
            vector_sensor_stds_parent=[ACC_STD / nominal, MAG_STD],
            vector_sensor_stds_child=[ACC_STD / nominal, MAG_STD],
            normalize_measurements=True,
        )._get_measurement_update(q_wp, drifted_q_wc, raw_p, raw_c)

        self.assertFalse(np.allclose(plain[1].as_matrix(), rescaled[1].as_matrix(), atol=1e-6))

    def test_normalizing_weakens_the_correction_at_a_fixed_std(self):
        """The documented consequence of holding the stds fixed: scaling a 9.81 m/s^2
        reading to unit length shrinks its residual and its Jacobian but not its entry
        in R, so the same geometric error pulls the estimate back less far. If this
        ever stopped being true, the two arms of the normalization comparison would no
        longer differ in tuning and the caveat around it would be wrong."""
        q_wp = Rotation.identity()
        true_q_wc = Rotation.identity()
        data_p = [9.81 * WORLD_GRAVITY, WORLD_MAG]
        data_c = [9.81 * WORLD_GRAVITY, WORLD_MAG]
        drifted_q_wc = true_q_wc * Rotation.from_rotvec([0.05, 0.05, 0.05])

        _, plain_wc = make_filter()._get_measurement_update(q_wp, drifted_q_wc, data_p, data_c)
        _, norm_wc = make_filter(normalize_measurements=True)._get_measurement_update(
            q_wp, drifted_q_wc, data_p, data_c)

        plain_error = (true_q_wc.inv() * plain_wc).magnitude()
        norm_error = (true_q_wc.inv() * norm_wc).magnitude()
        self.assertLess(plain_error, norm_error)


class TestRelativeFilterUpdate(unittest.TestCase):
    def setUp(self):
        self.filter = make_filter()

    def test_update_rejects_the_wrong_number_of_sensors(self):
        with self.assertRaises(ValueError):
            self.filter.update(np.zeros(3), np.zeros(3), [WORLD_GRAVITY], [WORLD_GRAVITY], 0.01)

    def test_no_motion_from_identity(self):
        self.filter.set_qs(Rotation.identity(), Rotation.identity())
        data_p, data_c = consistent_measurements(np.eye(3), np.eye(3))
        for _ in range(100):
            self.filter.update(np.zeros(3), np.zeros(3), data_p, data_c, 0.01)
        np.testing.assert_allclose(self.filter.get_R_pc(), np.eye(3), atol=1e-8)

    def test_no_motion_from_a_random_orientation(self):
        np.random.seed(5)
        q_wp = Rotation.from_rotvec(np.random.rand(3))
        q_wc = Rotation.from_rotvec(np.random.rand(3))
        self.filter.set_qs(q_wp, q_wc)
        R_pc_original = self.filter.get_R_pc()
        data_p, data_c = consistent_measurements(q_wp.as_matrix(), q_wc.as_matrix())

        for _ in range(100):
            self.filter.update(np.zeros(3), np.zeros(3), data_p, data_c, 0.01)

        np.testing.assert_allclose(self.filter.get_R_pc(), R_pc_original, atol=1e-8)

    def test_tracks_a_perfect_child_rotation(self):
        gyro_p = np.zeros(3)
        gyro_c = np.array([0.01, 0.0, 0.0])
        dt = 1.0
        self.filter.set_qs(Rotation.identity(), Rotation.identity())

        for i in range(100):
            expected_q_wc = Rotation.from_rotvec(gyro_c * dt * (i + 1))
            R_wc = expected_q_wc.as_matrix()
            data_p = [WORLD_GRAVITY, WORLD_MAG]
            data_c = [R_wc.T @ WORLD_GRAVITY, R_wc.T @ WORLD_MAG]
            self.filter.update(gyro_p, gyro_c, data_p, data_c, dt)
            np.testing.assert_allclose(self.filter.q_wc.as_matrix(), R_wc, atol=1e-8,
                                       err_msg=f"q_wc drifted at step {i}.")

    def test_update_converges_from_a_wrong_initial_guess(self):
        # Stationary bodies, measurements consistent with the truth, filter started at a
        # large error: the vector-sensor updates should pull it in. P has to be told the
        # seed is unreliable — the default assumes a state seeded from a known orientation
        # and would (correctly) refuse to move far from it.
        filt = make_filter(init_orientation_std=np.deg2rad(60.0))
        true_q_wp = Rotation.from_rotvec([0.2, -0.1, 0.05])
        true_q_wc = Rotation.from_rotvec([-0.3, 0.15, 0.4])
        data_p, data_c = consistent_measurements(true_q_wp.as_matrix(), true_q_wc.as_matrix())

        true_R_pc = true_q_wp.as_matrix().T @ true_q_wc.as_matrix()
        filt.set_qs(true_q_wp * Rotation.from_rotvec([0.3, 0.3, 0.3]), true_q_wc)
        error_before = Rotation.from_matrix(true_R_pc.T @ filt.get_R_pc()).magnitude()

        for _ in range(500):
            filt.update(np.zeros(3), np.zeros(3), data_p, data_c, 0.01)

        error_after = Rotation.from_matrix(true_R_pc.T @ filt.get_R_pc()).magnitude()
        self.assertLess(error_after, error_before)
        self.assertLess(error_after, np.deg2rad(1.0))

    def test_a_confident_seed_is_not_yanked_by_one_bad_sample(self):
        # The regression the default P guards against: with P = eye(6) a single
        # inconsistent measurement is applied almost in full as a state correction.
        true_q_wp = Rotation.from_rotvec([0.2, -0.1, 0.05])
        true_q_wc = Rotation.from_rotvec([-0.3, 0.15, 0.4])
        data_p, data_c = consistent_measurements(true_q_wp.as_matrix(), true_q_wc.as_matrix())
        data_c = [data_c[0] + np.array([4.0, -3.0, 2.0]), data_c[1]]   # dynamic acceleration

        true_R_pc = true_q_wp.as_matrix().T @ true_q_wc.as_matrix()

        def one_step_error(**kwargs):
            filt = make_filter(**kwargs)
            filt.set_qs(true_q_wp, true_q_wc)
            filt.update(np.zeros(3), np.zeros(3), data_p, data_c, 0.01)
            return Rotation.from_matrix(true_R_pc.T @ filt.get_R_pc()).magnitude()

        # 1.3 deg with the default; 92 deg if P is left at eye(6).
        self.assertLess(one_step_error(), np.deg2rad(2.0))
        self.assertGreater(one_step_error(init_orientation_std=1.0), np.deg2rad(45.0))


if __name__ == "__main__":
    unittest.main()
