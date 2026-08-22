"""
Pins the mathematical identities experiments/joint_dof.py rests on.

Every assertion here is an equality that is either exactly true or the module is wrong, and
every one of them would break SILENTLY. A transposed rotation, a curve derivative with the
wrong sign, a gauge that lets the optimizer hide curvature in the neutral pose — all of them still
return plausible-looking degree numbers, and the only way to notice is to check them against a
second, independent derivation. That is what this file is.

Synthetic throughout, with the DOF COUNT KNOWN BY CONSTRUCTION: joints are built to be exactly
a hinge, exactly a universal joint, or exactly a 1-DOF curve of known curvature, so every fit
has a right answer to be scored against rather than a recorded value to be compared with.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

import experiments.joint_dof as jd
from experiments.global_assumptions import ALBORNO, IMOVE
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace

FS = 100.0


def smooth_angles(n: int, seed: int, amplitude: float = 1.0) -> np.ndarray:
    """A smooth, non-repeating scalar trajectory — a joint angle, not white noise.

    Smooth on purpose: the conditioning of every fit here depends on how the joint SWEEPS, and
    a white sequence excites SO(3) far more evenly than a limb ever does, which would make
    these tests easier than the data.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    phases = rng.uniform(0, 2 * np.pi, 4)
    freqs = np.array([0.13, 0.29, 0.47, 0.71])
    return amplitude * sum(np.sin(2 * np.pi * f * t + p) / (k + 1)
                           for k, (f, p) in enumerate(zip(freqs, phases)))


def hinge_series(n=3000, seed=0, axis=None, range_rad=0.7) -> np.ndarray:
    """R_pc(t) for an exact hinge about `axis`, through roughly `range_rad`.

    The default range is deliberately human — `smooth_angles` sums four harmonics, so the peak
    is a few times this and a larger value swings the relative rotation past 150 deg from its
    own mean, where `_lowpass_run` refuses to filter. That refusal is correct (the log map is
    discontinuous at the pi wrap and no real joint goes there); it just makes an over-swung
    fixture fail for the wrong reason.
    """
    axis = np.array([0.2, -0.9, 0.35]) if axis is None else np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    theta = smooth_angles(n, seed, range_rad)
    R_0 = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
    return jd._rodrigues(axis, theta) @ R_0


def curved_series(n=3000, seed=0, curvature_rad=0.15, range_rad=0.6) -> np.ndarray:
    """A 1-DOF joint whose off-axis rotation is QUADRATIC in the joint angle.

    Quadratic rather than linear on purpose. A linear coupling is a hinge about a tilted axis
    and the hinge's axis is free, so a linear test case would pass whether or not the curve
    model works at all — it is exactly the case that cannot distinguish the two.
    """
    axis = np.array([0.0, 0.0, 1.0])
    theta = smooth_angles(n, seed, range_rad)
    perp = jd._tangent_basis(axis)[:, 0]
    scale = curvature_rad / max(float(np.max(theta ** 2)), 1e-9)
    nu = theta[:, None] * axis + (scale * theta ** 2)[:, None] * perp
    return Rotation.from_rotvec([0.1, 0.4, -0.25]).as_matrix() @ jd._exp_matrix(nu)


def universal_series(n=3000, seed=0, carrying_deg=70.0) -> np.ndarray:
    """R_pc(t) for an exact universal joint whose axes are `carrying_deg` apart.

    The carrying angle is the angle between u and R_0 v — both axes carried into the PARENT
    frame at the neutral pose — because u left-multiplies and v right-multiplies, so they live
    in different frames and the bare angle between their coordinate triples means nothing. v is
    therefore built by pulling the desired parent-frame direction back through R_0. Getting
    this wrong is not visible in the residual (the joint is still a perfectly good universal
    joint), only in the reported angle, which is exactly the trap this construction is meant to
    catch in the module.
    """
    u = np.array([0.0, 0.0, 1.0])
    angle = np.deg2rad(carrying_deg)
    R_0 = Rotation.from_rotvec([0.15, 0.05, -0.3]).as_matrix()
    v = R_0.T @ np.array([np.sin(angle), 0.0, np.cos(angle)])
    theta1 = smooth_angles(n, seed, 0.6)
    theta2 = smooth_angles(n, seed + 100, 0.35)
    return jd._rodrigues(u, theta1) @ R_0 @ jd._rodrigues(v, theta2)


def reuben_series(n=3000, seed=0, max_flexion_deg=85.0) -> np.ndarray:
    """A knee that obeys the PUBLISHED Reuben coupling exactly, through a knee-like range.

    Delegates the curve to the module's own `knee_coupling_curve` rather than re-transcribing
    the polynomials, so this fixture cannot drift away from the thing under test; what it adds
    is an independent CONSTRUCTION of the joint — rotation vector assembled here, by hand, and
    never through `fit_knee_coupling`'s parameterization.

    Flexion is non-negative, like a knee's. The polynomials are quartics with large negative
    coefficients and are zero at zero by construction, so at negative flexion they diverge
    rather than extrapolate, and a zero-mean profile would build a joint that is not a knee and
    that no correct estimator should fit.
    """
    axis = np.array([0.0, 0.0, 1.0])
    swing = smooth_angles(n, seed, 1.0)
    swing = swing - swing.min()
    flexion = np.deg2rad(max_flexion_deg) * swing / max(float(swing.max()), 1e-9)
    perp = jd._tangent_basis(axis)
    nu = flexion[:, None] * axis + jd.knee_coupling_curve(flexion) @ perp.T
    return Rotation.from_rotvec([0.2, -0.3, 0.45]).as_matrix() @ jd._exp_matrix(nu)


def plate_from_rotations(name: str, rotations: np.ndarray, valid=None) -> PlateTrial:
    """A PlateTrial carrying a given pose sequence and a matching zero IMU record."""
    n = len(rotations)
    timestamps = np.arange(n) / FS
    positions = np.zeros((n, 3))
    world = WorldTrace(timestamps, positions, rotations,
                       valid=np.ones(n, bool) if valid is None else valid)
    imu = IMUTrace(timestamps, np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)))
    return PlateTrial(name, imu, world)


class TestRotationPrimitives(unittest.TestCase):
    """The vectorized exp/log/Jacobian are rewrites of scipy for speed. They must agree."""

    def setUp(self):
        rng = np.random.default_rng(3)
        vectors = rng.normal(0, 0.8, size=(500, 3))
        # Kept strictly inside the pi ball. Beyond it the log map is not the inverse of the
        # exponential — it returns the SHORTER equivalent rotation — so a round-trip test out
        # there would be asserting that scipy and this module make the same arbitrary choice,
        # not that either is right.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = np.where(norms > 3.0, vectors * (3.0 / np.maximum(norms, 1e-12)), vectors)
        # Include the two regimes the series expansions guard: essentially zero, and near pi.
        self.vectors[0] = 0.0
        self.vectors[1] = np.array([1e-9, 0.0, 0.0])
        self.vectors[2] = np.array([0.0, 3.10, 0.0])

    def test_exp_matches_scipy(self):
        np.testing.assert_allclose(jd._exp_matrix(self.vectors),
                                   Rotation.from_rotvec(self.vectors).as_matrix(), atol=1e-12)

    def test_log_matches_scipy(self):
        R = Rotation.from_rotvec(self.vectors).as_matrix()
        np.testing.assert_allclose(jd._log_matrix(R), self.vectors, atol=1e-9)

    def test_log_inverts_exp_including_near_pi(self):
        np.testing.assert_allclose(jd._log_matrix(jd._exp_matrix(self.vectors)),
                                   self.vectors, atol=1e-9)

    def test_rodrigues_matches_exp_for_a_fixed_axis(self):
        axis = np.array([0.3, -0.5, 0.81])
        axis /= np.linalg.norm(axis)
        theta = np.linspace(-3.0, 3.0, 200)
        np.testing.assert_allclose(jd._rodrigues(axis, theta),
                                   jd._exp_matrix(theta[:, None] * axis), atol=1e-12)

    def test_right_jacobian_is_the_derivative_of_exp(self):
        """d/de exp(v + e d) = exp(v) [J_r(v) d] to first order — checked numerically.

        This identity is the entire reason the curve model can be differentiated: the curve
        lives in the rotation VECTOR and the residual lives in the group, and J_r is the only
        thing connecting them. A wrong J_r still converges, just to the wrong place and slowly,
        so nothing downstream would flag it.
        """
        rng = np.random.default_rng(7)
        v = rng.normal(0, 0.7, size=(40, 3))
        d = rng.normal(0, 1.0, size=(40, 3))
        eps = 1e-7
        exact = np.einsum('tij,tj->ti', jd._right_jacobian(v), d)
        numeric = jd._log_matrix(jd._transpose_multiply(
            jd._exp_matrix(v), jd._exp_matrix(v + eps * d))) / eps
        np.testing.assert_allclose(numeric, exact, atol=1e-5)

    def test_tangent_basis_is_orthonormal_and_perpendicular(self):
        rng = np.random.default_rng(11)
        for axis in rng.normal(size=(50, 3)):
            axis = axis / np.linalg.norm(axis)
            basis = jd._tangent_basis(axis)
            np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)
            np.testing.assert_allclose(basis.T @ axis, np.zeros(2), atol=1e-12)


class TestModelNesting(unittest.TestCase):
    """The ladder must nest. If it does not, no drop between rungs means anything."""

    def setUp(self):
        self.R = universal_series(n=1500, seed=2)
        self.init = jd.pca_initialization(self.R)
        self.models = jd.fit_all_models(self.R, self.init)
        self.rms = {name: float(np.sqrt(np.mean(jd.score_model(model, self.R)['error_deg'] ** 2)))
                    for name, model in self.models.items()}

    def test_errors_are_monotone_in_freedom(self):
        self.assertGreaterEqual(self.rms['weld'], self.rms['hinge'] - 1e-6)
        self.assertGreaterEqual(self.rms['weld'], self.rms['hinge'] - 1e-6)
        self.assertGreaterEqual(self.rms['hinge'], self.rms['universal'] - 1e-6)
        self.assertGreaterEqual(self.rms['universal'], self.rms['spherical'] - 1e-6)

    def test_spherical_is_exactly_zero(self):
        """Zero by construction, not by measurement: 3 DOF constrains no relative orientation."""
        np.testing.assert_array_equal(jd.score_model({'kind': 'spherical'}, self.R)['error_deg'],
                                      0.0)

    def test_a_zero_coefficient_curve_is_exactly_the_hinge(self):
        """The nesting has to hold in the CODE, not only in the algebra.

        Two different solvers reach these two models — the curve through the rotation vector and
        a right Jacobian, the hinge through a fixed-axis Rodrigues — so their agreeing at c == 0
        is a real cross-implementation check on both. Zeroing the channel map rather than the
        published polynomials is what makes c == 0 reachable at all now that the curve is fixed.
        """
        hinge = self.models['hinge']
        u_child = np.asarray(hinge['R_0']).T @ np.asarray(hinge['axis_parent'])
        flat = {'kind': 'coupling', 'axis_child': u_child,
                'axis_parent': hinge['axis_parent'], 'R_0': hinge['R_0'],
                'coupling_flexion_sign': 1.0, 'coeffs': np.zeros((2, 2))}
        np.testing.assert_allclose(jd.score_model(flat, self.R)['error_deg'],
                                   jd.score_model(hinge, self.R)['error_deg'], atol=1e-6)

class TestKnownJoints(unittest.TestCase):
    """Joints whose DOF count is exact by construction. A matched model must fit to ~0."""

    def test_hinge_is_recovered_exactly(self):
        axis = np.array([0.2, -0.9, 0.35]) / np.linalg.norm([0.2, -0.9, 0.35])
        R = hinge_series(seed=1, axis=axis)
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        error = jd.score_model(models['hinge'], R)['error_deg']
        self.assertLess(float(np.max(error)), 1e-3)

    def test_hinge_axis_is_recovered_as_an_undirected_line(self):
        axis = np.array([0.2, -0.9, 0.35]) / np.linalg.norm([0.2, -0.9, 0.35])
        R = hinge_series(seed=1, axis=axis)
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        fitted = np.asarray(models['hinge']['axis_parent'])
        self.assertLess(np.degrees(np.arccos(abs(float(fitted @ axis)))), 0.5)

    def test_universal_joint_costs_the_hinge_and_is_recovered_by_two_dof(self):
        R = universal_series(seed=4, carrying_deg=70.0)
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        hinge_rms = float(np.sqrt(np.mean(jd.score_model(models['hinge'], R)['error_deg'] ** 2)))
        universal = jd.score_model(models['universal'], R)['error_deg']
        self.assertGreater(hinge_rms, 2.0)
        self.assertLess(float(np.max(universal)), 0.5)

    def test_carrying_angle_is_recovered(self):
        R = universal_series(seed=4, carrying_deg=70.0)
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        self.assertAlmostEqual(models['universal']['carrying_angle_deg'], 70.0, delta=1.5)

class TestKneeCoupling(unittest.TestCase):
    """The published 1-DOF nonlinear knee — the model the whole question is about."""

    def test_a_knee_that_obeys_the_coupling_is_recovered(self):
        """If the estimator cannot find the coupling when it is true by construction, its
        residual on real data says nothing about the coupling."""
        R = reuben_series(seed=51)
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        models['knee_coupling'] = jd.fit_knee_coupling(models['hinge'], R)
        error = jd.score_model(models['knee_coupling'], R)['error_deg']
        self.assertLess(float(np.sqrt(np.mean(error ** 2))), 0.5)

    def test_the_coupling_beats_a_hinge_when_it_is_true(self):
        """The converse of the real-data finding, and the control that makes it meaningful."""
        R = reuben_series(seed=52)
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        models['knee_coupling'] = jd.fit_knee_coupling(models['hinge'], R)
        hinge = float(np.sqrt(np.mean(jd.score_model(models['hinge'], R)['error_deg'] ** 2)))
        coupling = float(np.sqrt(np.mean(
            jd.score_model(models['knee_coupling'], R)['error_deg'] ** 2)))
        self.assertLess(coupling, 0.5 * hinge)

    def test_a_free_hinge_axis_absorbs_almost_all_of_the_coupling(self):
        """THE central quantitative point, asserted rather than left as prose.

        On a knee that obeys Reuben EXACTLY through 85 deg of flexion — where the coupling
        reaches nearly 15 deg of internal rotation — a free-axis hinge is left with under a
        degree, because the coupling's rotation channel is 0.37 deg per deg of flexion and a
        LINEAR coupling is simply a hinge tilted arctan(0.37) ~ 20 deg off medio-lateral.

        Everything the report says about the coupling rests on this: the amplitude a coupling
        quotes is not what it can buy, because a model it is being compared against already has
        the linear part for free. Only the curvature is ever in play.
        """
        R = reuben_series(seed=59)
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        hinge = float(np.sqrt(np.mean(jd.score_model(models['hinge'], R)['error_deg'] ** 2)))
        amplitude = float(np.max(np.abs(np.degrees(
            jd.knee_coupling_curve(np.deg2rad(np.linspace(0, 85, 200)))))))
        self.assertGreater(amplitude, 12.0)
        self.assertLess(hinge, 0.1 * amplitude)

    def test_the_channel_orientation_is_searched_not_assumed(self):
        """The polynomials are in ANATOMICAL axes and the perpendicular basis is arbitrary, so
        an unaligned fit scores the right curve in the wrong frame. Measured on Subject01's
        right knee that was 23.2 deg against 15.1 for the same curve aligned, which is larger
        than every effect this module reports."""
        R = reuben_series(seed=55)
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        angle = jd.score_model(models['hinge'], R)['theta1']
        aligned = jd.fit_knee_coupling(models['hinge'], R)

        # The SAME published curve, turned 45 deg within the perpendicular plane — which is all
        # that separates a fit that searched the channel orientation from one that accepted
        # whatever `_tangent_basis` returned. Rotated by a fixed amount rather than reconstructed
        # unrotated, so the test does not quietly pass whenever the fitted angle lands near zero.
        turn = np.pi / 4
        rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        misaligned = {**aligned, 'coeffs': rotation @ np.asarray(aligned['coeffs'])}

        self.assertLess(
            float(np.sqrt(np.mean(jd.score_model(aligned, R)['error_deg'] ** 2))),
            float(np.sqrt(np.mean(jd.score_model(misaligned, R)['error_deg'] ** 2))))

    def test_it_reaches_every_table_a_fitted_model_reaches(self):
        """It is a rung, not an appendix: the per-sample errors, the error-against-angle curve
        and the coupling-shape table must all carry it, or the figures silently omit it."""
        R = reuben_series(seed=56)
        series = jd.JointSeries(R=R, R_raw=R, timestamps=np.arange(len(R)) / FS,
                                run_id=np.zeros(len(R), np.int32), fs=FS,
                                n_valid_total=len(R), n_runs=1)
        result = jd.analyze_joint(
            series, {'joint_key': 'R_Knee', 'joint': 'R_Knee', 'source': 'markers',
                     'placement': 'M'}, folds=2)
        self.assertIn('knee_coupling', {row['model'] for row in result.rows})
        self.assertIn('error_knee_coupling_deg', result.samples.columns)
        self.assertIn('knee_coupling_rms_deg', result.curve.columns)
        self.assertEqual(set(result.coupling['model']), {'knee_coupling'})

    def test_it_is_only_fitted_at_knees(self):
        R = universal_series(n=1200, seed=57)
        series = jd.JointSeries(R=R, R_raw=R, timestamps=np.arange(len(R)) / FS,
                                run_id=np.zeros(len(R), np.int32), fs=FS,
                                n_valid_total=len(R), n_runs=1)
        result = jd.analyze_joint(
            series, {'joint_key': 'R_Hip', 'joint': 'R_Hip', 'source': 'markers',
                     'placement': 'M'}, folds=0)
        self.assertNotIn('knee_coupling', {row['model'] for row in result.rows})

    def test_the_published_polynomials_are_transcribed_correctly(self):
        """Spot values against Reuben et al. (1986) as IMoveLab evaluates them, in degrees.
        A transcription slip here would be invisible — every number downstream stays
        plausible — and would silently redefine what is being tested."""
        adduction, rotation = np.degrees(jd.knee_coupling_curve(np.deg2rad([0.0, 60.0]))).T
        np.testing.assert_allclose(adduction, [0.0, 1.77], atol=0.05)
        np.testing.assert_allclose(rotation, [0.0, 13.16], atol=0.05)

    def test_the_coupling_is_mostly_a_tilted_hinge(self):
        """Its rotation channel is 0.37 deg per deg of flexion, and a LINEAR coupling is a hinge
        about an axis tilted arctan(0.37) ~ 20 deg. A free-axis hinge absorbs that for nothing,
        so the coupling's curvature — what it can actually buy — is far smaller than its
        headline amplitude, and the report must not confuse the two."""
        R = reuben_series(seed=58)
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        models['knee_coupling'] = jd.fit_knee_coupling(models['hinge'], R)
        angle = jd.score_model(models['knee_coupling'], R)['theta1']
        stats = jd.coupling_curve_stats(models['knee_coupling'], angle)
        amplitude = float(np.max(np.abs(np.degrees(jd.knee_coupling_curve(angle)))))
        self.assertGreater(amplitude, 5.0)
        self.assertLess(stats['curvature_max_deg'], 0.5 * amplitude)


class TestGaugeInvariance(unittest.TestCase):
    """The reported curvature must be a property of the JOINT, not of the fitted parameters."""

    def test_curvature_is_invariant_to_the_frames_the_joint_is_expressed_in(self):
        """Re-expressing the same joint in remounted sensor frames cannot change its curvature.

        R_pc -> Q_p^T R_pc Q_c is exactly what remounting either sensor does, and it changes
        every fitted axis and the whole neutral pose. Anything read off the curve's COEFFICIENTS
        moves with it; the distance from the best geodesic does not. Measured on a synthetic
        joint with a known 8 deg quadratic coupling, coefficient-based departures of 8.6 deg and
        44.8 deg came from two fits with identical residuals, which is why this test exists.
        """
        R = reuben_series(seed=12)
        Q_p = Rotation.from_rotvec([0.7, -0.4, 1.1]).as_matrix()
        Q_c = Rotation.from_rotvec([-0.3, 0.9, 0.2]).as_matrix()
        R_remounted = np.einsum('ji,tjk,kl->til', Q_p, R, Q_c)

        curvatures = []
        for series in (R, R_remounted):
            models = jd.fit_all_models(series, jd.pca_initialization(series))
            coupling = jd.fit_knee_coupling(models['hinge'], series)
            q = jd.score_model(coupling, series)['theta1']
            curvatures.append(jd.coupling_curve_stats(coupling, q)['curvature_max_deg'])
        self.assertAlmostEqual(curvatures[0], curvatures[1], delta=0.3)

    def test_residuals_are_invariant_to_the_frames(self):
        R = universal_series(seed=13)
        Q_p = Rotation.from_rotvec([0.2, 1.3, -0.5]).as_matrix()
        Q_c = Rotation.from_rotvec([0.8, -0.1, 0.6]).as_matrix()
        R_remounted = np.einsum('ji,tjk,kl->til', Q_p, R, Q_c)
        for name in ('weld', 'hinge', 'universal'):
            values = []
            for series in (R, R_remounted):
                models = jd.fit_all_models(series, jd.pca_initialization(series))
                values.append(float(np.sqrt(np.mean(
                    jd.score_model(models[name], series)['error_deg'] ** 2))))
            self.assertAlmostEqual(values[0], values[1], delta=0.05, msg=name)

    def test_carrying_angle_is_invariant_to_the_frames(self):
        """The one geometric quantity the cross-subject tables are allowed to pool.

        Remounting the parent by Q_p sends u -> Q_p' u and R_0 -> Q_p' R_0 Q_c, so R_0 v ->
        Q_p' R_0 v and the angle between them is unchanged. The report leans on this to justify
        pooling carrying angles across subjects while refusing to pool the axes themselves.
        """
        R = universal_series(seed=14, carrying_deg=62.0)
        Q_p = Rotation.from_rotvec([1.0, 0.2, -0.7]).as_matrix()
        Q_c = Rotation.from_rotvec([-0.5, 0.4, 0.9]).as_matrix()
        R_remounted = np.einsum('ji,tjk,kl->til', Q_p, R, Q_c)
        angles = []
        for series in (R, R_remounted):
            models = jd.fit_all_models(series, jd.pca_initialization(series))
            angles.append(models['universal']['carrying_angle_deg'])
        self.assertAlmostEqual(angles[0], angles[1], delta=1.0)


class TestExcitation(unittest.TestCase):
    """The closed-form conditioning measures, which involve no fitting at all."""

    def test_a_welded_pair_has_zero_excitation(self):
        R = np.broadcast_to(Rotation.from_rotvec([0.4, -0.2, 0.1]).as_matrix(), (500, 3, 3))
        measures = jd.excitation(np.ascontiguousarray(R))
        self.assertAlmostEqual(measures['excitation'], 0.0, places=9)
        self.assertAlmostEqual(measures['hinge_deficiency'], 0.0, places=9)

    def test_a_pure_hinge_has_zero_hinge_deficiency(self):
        """A hinge's axis is a singular vector of sum(R_pc) with sigma = N, exactly and however
        long the trial runs — so sigma_MAX is N and the deficiency is zero.

        Which end of the spectrum that is falls out of the algebra rather than intuition, and
        the first version of this module had it the other way round: it read 1 - sigma_max/N as
        "did the joint move", which is zero for a hinge sweeping 90 deg. The test is here
        because both quantities are plausible-looking numbers in [0, 1] either way.
        """
        measures = jd.excitation(hinge_series(n=2000, seed=8))
        self.assertLess(measures['hinge_deficiency'], 1e-9)
        self.assertGreater(measures['excitation'], 0.05)

    def test_a_universal_joint_has_no_invariant_axis(self):
        measures = jd.excitation(universal_series(n=2000, seed=8))
        self.assertGreater(measures['hinge_deficiency'], 0.01)


class TestValidMasking(unittest.TestCase):
    """Padded poses are the failure mode this module exists downstream of."""

    def test_padded_frames_are_excluded_from_the_series(self):
        """Outside `valid` a WorldTrace holds a CONSTANT pose, which every model fits
        perfectly. Including it does not add noise, it silently flatters the low-DOF models —
        which is the direction of error nobody looks for."""
        R = hinge_series(n=2000, seed=15)
        padded = np.concatenate([R, np.broadcast_to(R[-1], (3000, 3, 3))])
        valid = np.r_[np.ones(2000, bool), np.zeros(3000, bool)]
        parent = plate_from_rotations('p', np.broadcast_to(np.eye(3), (5000, 3, 3)).copy(), valid)
        child = plate_from_rotations('c', padded, valid)
        series = jd.joint_series(parent, child)
        self.assertEqual(len(series.R), 2000)
        self.assertEqual(series.n_runs, 1)

    def test_short_runs_are_dropped(self):
        R = hinge_series(n=2000, seed=16)
        valid = np.ones(2000, bool)
        valid[500:1500] = False
        valid[1500:1500 + jd.MIN_RUN_SAMPLES - 1] = True   # a run just under the floor
        valid[1500 + jd.MIN_RUN_SAMPLES - 1:] = False
        parent = plate_from_rotations('p', np.broadcast_to(np.eye(3), (2000, 3, 3)).copy(), valid)
        child = plate_from_rotations('c', R, valid)
        series = jd.joint_series(parent, child)
        self.assertEqual(series.n_runs, 1, "the sub-floor run must be dropped")
        self.assertEqual(len(series.R), 500)

    def test_no_valid_frames_returns_none(self):
        valid = np.zeros(1000, bool)
        parent = plate_from_rotations('p', np.broadcast_to(np.eye(3), (1000, 3, 3)).copy(), valid)
        child = plate_from_rotations('c', hinge_series(n=1000, seed=17), valid)
        self.assertIsNone(jd.joint_series(parent, child))

    def test_the_filter_does_not_smear_across_a_gap(self):
        """Two runs minutes apart must be filtered separately.

        A single filtfilt over the concatenation would ring across the join, and the samples
        either side of it are the ones a cross-validation block boundary also lands on.
        """
        R = hinge_series(n=1200, seed=18)
        valid = np.ones(1200, bool)
        valid[400:800] = False
        parent = plate_from_rotations('p', np.broadcast_to(np.eye(3), (1200, 3, 3)).copy(), valid)
        child = plate_from_rotations('c', R, valid)
        series = jd.joint_series(parent, child)
        self.assertEqual(series.n_runs, 2)
        # Filtering each run alone must reproduce the concatenation exactly.
        expected = np.concatenate([jd._lowpass_run(R[:400], FS, jd.LOWPASS_HZ),
                                   jd._lowpass_run(R[800:], FS, jd.LOWPASS_HZ)])
        np.testing.assert_allclose(series.R, expected, atol=1e-12)


class TestCrossValidation(unittest.TestCase):
    """Blocked folds. Without them the coupling's extra structure wins by construction."""

    def test_folds_partition_the_samples(self):
        run_id = np.zeros(1000, dtype=np.int32)
        masks = jd.cv_folds(run_id, folds=3, blocks_per_fold=4)
        self.assertEqual(len(masks), 3)
        np.testing.assert_array_equal(sum(m.astype(int) for m in masks), 1)

    def test_folds_never_straddle_a_run_boundary(self):
        """Blocks are cut inside runs, so a test sample is never adjacent to its own training
        neighbour across a gap that is minutes long in the source recording."""
        run_id = np.r_[np.zeros(300, np.int32), np.ones(300, np.int32)]
        masks = jd.cv_folds(run_id, folds=3, blocks_per_fold=2)
        for mask in masks:
            for run in (0, 1):
                self.assertTrue(mask[run_id == run].any(),
                                "every run must contribute to every fold")

    def test_folds_are_contiguous_blocks_not_alternating_samples(self):
        """An interleaved-by-SAMPLE split leaks: at 100 Hz a test sample's neighbours are 10 ms
        away and essentially identical, so the held-out error would measure nothing."""
        masks = jd.cv_folds(np.zeros(1200, dtype=np.int32), folds=3, blocks_per_fold=4)
        for mask in masks:
            transitions = int(np.sum(mask[1:] != mask[:-1]))
            self.assertLess(transitions, 20, "folds must be blocks, not alternating samples")

    def test_cross_validated_error_is_at_least_the_in_sample_error_on_noise(self):
        """On data with no structure to find, out-of-fold error cannot beat in-sample error.

        This is the property that makes the ladder comparable across models of different
        structure size, and it is the whole reason the report leads with the CV column.
        """
        rng = np.random.default_rng(21)
        R = Rotation.from_rotvec(np.cumsum(rng.normal(0, 0.05, (1200, 3)), axis=0)).as_matrix()
        init = jd.pca_initialization(R)
        models = jd.fit_all_models(R, init)
        cv = jd.cross_validate(R, jd.cv_folds(np.zeros(len(R), np.int32)))
        for name in ('hinge', 'universal'):
            in_sample = float(np.sqrt(np.mean(jd.score_model(models[name], R)['error_deg'] ** 2)))
            self.assertGreaterEqual(cv[name]['cv_rms_deg'], in_sample - 1e-6, msg=name)


class TestReferenceAgreement(unittest.TestCase):
    """Markers versus biplane — the noise floor the marker residuals are read against."""

    def test_two_references_related_by_a_constant_frame_change_agree_exactly(self):
        R = universal_series(n=800, seed=22)
        Q_p = Rotation.from_rotvec([0.4, -0.7, 0.2]).as_matrix()
        Q_c = Rotation.from_rotvec([0.1, 0.5, -0.8]).as_matrix()
        fitted = jd._fit_constant_frames(R, np.einsum('ji,tjk,kl->til', Q_p, R, Q_c))
        self.assertLess(float(np.max(fitted['error_deg'])), 1e-3)

    def test_a_wobbling_reference_does_not_agree(self):
        """Soft-tissue artifact is exactly a frame change that is NOT constant, so it must
        survive the fit. If it did not, the floor would read as zero and every marker residual
        would be certified against nothing."""
        rng = np.random.default_rng(23)
        R = universal_series(n=800, seed=24)
        wobble = Rotation.from_rotvec(
            np.cumsum(rng.normal(0, 0.01, (800, 3)), axis=0)).as_matrix()
        fitted = jd._fit_constant_frames(R, R @ wobble)
        self.assertGreater(float(np.sqrt(np.mean(fitted['error_deg'] ** 2))), 1.0)


class TestDatasetWiring(unittest.TestCase):
    """The three sources go through one code path, so the keys have to split back correctly."""

    def test_marker_datasets_report_one_source(self):
        for spec in (ALBORNO, IMOVE):
            self.assertFalse(jd.has_multiple_sources(spec))
            for joint_key in spec.joints:
                self.assertEqual(jd.split_source(joint_key)[1], jd.MARKER_SOURCE)

    def test_biplane_keys_split_into_a_joint_and_a_reference(self):
        self.assertTrue(jd.has_multiple_sources(jd.BIPLANE))
        self.assertEqual(jd.split_source('R_Knee__biplane'), ('R_Knee', 'biplane'))
        identity = jd.joint_identity('L_Knee__vicon', jd.BIPLANE)
        self.assertEqual((identity['joint'], identity['source']), ('L_Knee', 'vicon'))

    def test_imove_placement_variants_fold_onto_their_anatomical_joint(self):
        identity = jd.joint_identity('R_Knee_H', IMOVE)
        self.assertEqual(identity['joint'], 'R_Knee')
        self.assertEqual(identity['placement'], 'H')
        self.assertEqual(identity['source'], jd.MARKER_SOURCE)

    def test_every_biplane_joint_names_sensors_the_spec_declares(self):
        declared = set(jd.BIPLANE.segment_sensor.values())
        for parent, child in jd.BIPLANE.joints.values():
            self.assertIn(parent, declared)
            self.assertIn(child, declared)

    def test_both_references_cover_the_same_joints(self):
        by_source = {}
        for joint_key in jd.BIPLANE.joints:
            joint, source = jd.split_source(joint_key)
            by_source.setdefault(source, set()).add(joint)
        self.assertEqual(by_source['vicon'], by_source['biplane'])

    def test_pooled_group_follows_the_session_not_the_subject(self):
        """A pooled fit assumes the mounting did not move, so it can only span one capture."""
        self.assertEqual(jd.pooled_group('alborno', '01', ['walking']), '01')
        self.assertEqual(jd.pooled_group('imove_biplane', '12', ['Test1/B/LSHop3']), '12/Test1')


class TestReportAngle(unittest.TestCase):
    """The angle axis every sample-level table and figure is binned on."""

    def test_a_wrapped_angle_recovers_its_true_range(self):
        """theta and theta + 2 pi are the same rotation, so the solver returns either.

        On two of Al Borno's nineteen trials it returned both for one knee, and the percentile
        spread of a joint that moved 68 deg read as 339 — in `theta1_rom_deg` and on every angle
        axis in the figures.
        """
        true_range = np.deg2rad(np.linspace(146, 214, 2000))
        as_returned = np.where(true_range > np.pi, true_range - 2 * np.pi, true_range)
        self.assertGreater(jd._rom_deg(as_returned), 300.0)
        self.assertAlmostEqual(jd._rom_deg(jd.report_angle(as_returned)), 64.6, delta=1.0)

    def test_the_zero_is_the_trial_s_own_median(self):
        """So that pooling across trials means something. The fitted neutral pose is per trial:
        across nineteen trials one knee's median angle ran from -130 to +139 deg, which smeared
        a 68 deg range across 300 in the pooled figure."""
        for offset in np.deg2rad([-130.0, 0.0, 139.0]):
            angle = jd.report_angle(offset + np.deg2rad(np.linspace(-34, 34, 500)))
            self.assertAlmostEqual(float(np.median(angle)), 0.0, places=9)
            # 68 deg of travel, trimmed at the 2.5/97.5 percentiles the ROM uses.
            self.assertAlmostEqual(jd._rom_deg(angle), 68.0 * 0.95, delta=1.0)

    def test_it_is_a_rigid_shift_so_the_shape_survives(self):
        """Only an offset and a 2 pi choice may move; the spacing between samples may not, or
        the error-against-angle curve would be distorted rather than merely re-labelled."""
        angle = np.deg2rad(np.linspace(-40, 25, 300))
        np.testing.assert_allclose(np.diff(jd.report_angle(angle)), np.diff(angle), atol=1e-12)

    def test_an_empty_angle_is_returned_unchanged(self):
        np.testing.assert_array_equal(jd.report_angle(np.array([])), np.array([]))


class TestJointSelection(unittest.TestCase):
    """The two filters that decide what a run fits. IMoVE needs both and they differ."""

    def test_no_filter_selects_every_joint_the_spec_declares(self):
        for spec in (ALBORNO, IMOVE, jd.BIPLANE):
            self.assertEqual(set(jd.selected_joints(spec)), set(spec.joints))

    def test_placement_filter_reduces_imove_to_its_six_bolted_pairs(self):
        """Eighteen sensor pairs against six is four times the runtime, which is the whole
        reason the flag exists."""
        self.assertEqual(len(jd.selected_joints(IMOVE)), 18)
        mid = jd.selected_joints(IMOVE, placements=['M'])
        self.assertEqual(set(mid), set(IMOVE.primary_joints))

    def test_joint_filter_keeps_every_placement_of_the_joints_it_names(self):
        """The two filters are independent: naming a joint must not silently drop its High and
        Low sensors, which are the only evidence this repo has about placement."""
        selected = jd.selected_joints(IMOVE, joints=['R_Knee'])
        self.assertEqual(set(selected), {'R_Knee', 'R_Knee_H', 'R_Knee_L'})

    def test_the_filters_compose(self):
        selected = jd.selected_joints(IMOVE, joints=['R_Knee'], placements=['H'])
        self.assertEqual(set(selected), {'R_Knee_H'})

    def test_placement_filter_is_a_no_op_on_the_single_placement_datasets(self):
        for spec in (ALBORNO, jd.BIPLANE):
            self.assertEqual(set(jd.selected_joints(spec, placements=['M'])), set(spec.joints))


class TestTrialCounting(unittest.TestCase):
    def test_trials_are_counted_on_the_subject_trial_pair(self):
        """Al Borno names every subject's trials 'walking' and 'complexTasks', so counting the
        name alone reported 19 trials as 2 — in the report header and in every figure's
        provenance stamp."""
        fits = pd.DataFrame({'subject': ['01', '01', '02', '02'],
                             'trial': ['walking', 'complexTasks'] * 2})
        self.assertEqual(jd.n_trials(fits), 4)

    def test_missing_columns_do_not_raise(self):
        self.assertEqual(jd.n_trials(pd.DataFrame({'joint': ['R_Knee']})), 0)


class TestAxialStatistics(unittest.TestCase):
    """A joint axis is an undirected LINE. Every statistic over axes must respect that."""

    def test_axial_mean_ignores_sign(self):
        rng = np.random.default_rng(31)
        axis = np.array([0.3, 0.5, -0.81])
        axis /= np.linalg.norm(axis)
        noisy = axis + rng.normal(0, 0.02, size=(50, 3))
        noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
        flipped = noisy * rng.choice([-1.0, 1.0], size=(50, 1))
        mean_a, _ = jd.axial_mean(noisy)
        mean_b, _ = jd.axial_mean(flipped)
        self.assertLess(np.degrees(np.arccos(abs(float(mean_a @ mean_b)))), 0.5)
        self.assertLess(np.degrees(np.arccos(abs(float(mean_a @ axis)))), 2.0)

    def test_axial_angles_are_folded_into_a_right_angle(self):
        axes = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0]])
        _, angles = jd.axial_mean(axes)
        self.assertTrue(np.all((angles >= 0) & (angles <= 90 + 1e-9)))
        self.assertAlmostEqual(angles[0], angles[1], places=9)


class TestErrorStats(unittest.TestCase):
    def test_quantile_spine_matches_the_repo_convention(self):
        stats = jd.error_stats(np.arange(101, dtype=float))
        for quantile in jd.QUANTILES:
            self.assertIn(f'p{int(round(quantile * 100)):02d}_deg', stats)
        self.assertAlmostEqual(stats['p50_deg'], 50.0)
        self.assertAlmostEqual(stats['max_deg'], 100.0)

    def test_rms_is_not_the_mean(self):
        stats = jd.error_stats(np.array([0.0, 10.0]))
        self.assertAlmostEqual(stats['mean_deg'], 5.0)
        self.assertAlmostEqual(stats['rms_deg'], np.sqrt(50.0))


class TestAnisotropy(unittest.TestCase):
    def test_a_single_missing_axis_concentrates_the_residual(self):
        """A hinge fitted to a universal joint leaves its residual on ONE direction. That is
        the signature that says a second axis will fix it, as opposed to noise, which will
        not — and the two cases have the same RMS."""
        R = universal_series(n=1500, seed=33)
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        scored = jd.score_model(models['hinge'], R)
        residual = jd._residual_rotvec(models['hinge'], R, scored)
        stats = jd.residual_anisotropy(residual, models['hinge']['axis_parent'],
                                       models['hinge']['R_0'])
        self.assertGreater(stats['resid_frac1'], 0.8)

    def test_isotropic_noise_spreads_the_residual(self):
        rng = np.random.default_rng(34)
        R = Rotation.from_rotvec(np.cumsum(rng.normal(0, 0.04, (1500, 3)), axis=0)).as_matrix()
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        scored = jd.score_model(models['hinge'], R)
        residual = jd._residual_rotvec(models['hinge'], R, scored)
        stats = jd.residual_anisotropy(residual, models['hinge']['axis_parent'],
                                       models['hinge']['R_0'])
        self.assertLess(stats['resid_frac1'], 0.95)

    def test_nothing_is_left_along_the_fitted_axis(self):
        """The angle solve drives that component to zero, so a nonzero value is a convergence
        failure. It is reported for exactly that reason."""
        R = universal_series(n=1000, seed=35)
        models = jd.fit_all_models(R, jd.pca_initialization(R))
        scored = jd.score_model(models['hinge'], R)
        residual = jd._residual_rotvec(models['hinge'], R, scored)
        stats = jd.residual_anisotropy(residual, models['hinge']['axis_parent'],
                                       models['hinge']['R_0'])
        self.assertLess(stats['resid_along_axis_frac'], 1e-6)


class TestEndToEnd(unittest.TestCase):
    def test_analyze_joint_emits_one_row_per_model(self):
        R = universal_series(n=1200, seed=41)
        series = jd.JointSeries(R=R, R_raw=R, timestamps=np.arange(len(R)) / FS,
                                run_id=np.zeros(len(R), np.int32), fs=FS,
                                n_valid_total=len(R), n_runs=1)
        identity = {'joint_key': 'K', 'joint': 'K', 'source': 'synthetic', 'placement': 'M'}
        result = jd.analyze_joint(series, identity, folds=2)
        self.assertEqual({row['model'] for row in result.rows}, set(jd.MODELS))
        for row in result.rows:
            self.assertEqual(row['n_dof'], jd.MODEL_DOF[row['model']])
        self.assertFalse(result.samples.empty)
        self.assertFalse(result.curve.empty)
        # EMPTY, and that is the assertion. The coupling table is knee-only now that the free
        # curve is gone — it used to be populated at every joint by the spline. A knee's is
        # checked in TestKneeCoupling.test_it_reaches_every_table_a_fitted_model_reaches.
        self.assertTrue(result.coupling.empty)

    def test_error_is_reported_as_a_fraction_of_the_weld(self):
        """A joint that barely moved is fitted well by everything, and that must be visible."""
        R = universal_series(n=1200, seed=42)
        series = jd.JointSeries(R=R, R_raw=R, timestamps=np.arange(len(R)) / FS,
                                run_id=np.zeros(len(R), np.int32), fs=FS,
                                n_valid_total=len(R), n_runs=1)
        rows = {row['model']: row for row in jd.analyze_joint(
            series, {'joint_key': 'K', 'joint': 'K', 'source': 's', 'placement': 'M'},
            folds=0).rows}
        self.assertAlmostEqual(rows['weld']['rms_frac_of_weld'], 1.0, places=9)
        self.assertLess(rows['universal']['rms_frac_of_weld'], 0.2)

    def test_too_few_samples_returns_none_rather_than_a_meaningless_fit(self):
        R = hinge_series(n=jd.MIN_FIT_SAMPLES - 10, seed=43)
        series = jd.JointSeries(R=R, R_raw=R, timestamps=np.arange(len(R)) / FS,
                                run_id=np.zeros(len(R), np.int32), fs=FS,
                                n_valid_total=len(R), n_runs=1)
        self.assertIsNone(jd.analyze_joint(
            series, {'joint_key': 'K', 'joint': 'K', 'source': 's', 'placement': 'M'}))


if __name__ == '__main__':
    unittest.main()
