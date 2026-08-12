"""Unit tests for experiments/relative_vs_absolute.py.

The emphasis is on the failures that are otherwise silent. Everything this module computes is
an angle in a plausible range, so a transposed composition, a flipped quaternion sign, an
unwrapped spherical excess or a reference that is not a unit vector all produce output of the
right shape and a believable magnitude. Only a value check against an independent
implementation catches them, so every geometric quantity here is checked against scipy's
Rotation, and the four facts the paper's argument rests on are checked as exact identities
over a large random sample rather than on hand-picked configurations.
"""
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from experiments.experiment_utils import EXPECTED_GRAVITY, JOINTS
from experiments.relative_vs_absolute import (ANGLE_METRICS, FIELDS, INVARIANT_FIELDS,
                                              INVARIANT_METRICS, PRIMARY_REFERENCE, QUANTILES,
                                              REFERENCE_VARIANTS, SAMPLE_STRIDE,
                                              THEOREM_TOL_DEG, angle_between_deg,
                                              angle_samples_table, correction_angles,
                                              invariant_angles, invariant_samples_table,
                                              invariant_stats_table, joint_stats_table,
                                              normalize_rows, quat_angle_deg, quat_conjugate,
                                              quat_multiply, reference_directions,
                                              shortest_arc_quat, spherical_excess_deg,
                                              summarize)
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace

N_RANDOM = 20_000


def random_unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v, _ = normalize_rows(rng.normal(size=(n, 3)))
    return v


class TestQuaternionHelpers(unittest.TestCase):
    """The quaternion layer, against scipy. It exists only for speed, so it has to agree."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.a = random_unit(self.rng, N_RANDOM)
        self.b = random_unit(self.rng, N_RANDOM)

    def _as_scipy(self, quat: np.ndarray) -> Rotation:
        """[w, x, y, z] -> scipy's [x, y, z, w]."""
        return Rotation.from_quat(np.concatenate([quat[:, 1:], quat[:, :1]], axis=1))

    def test_shortest_arc_quat_carries_a_onto_b(self):
        rotated = self._as_scipy(shortest_arc_quat(self.a, self.b)).apply(self.a)
        np.testing.assert_allclose(rotated, self.b, atol=1e-12)

    def test_shortest_arc_quat_angle_is_the_angle_between(self):
        """The MINIMUM rotation, not just any rotation carrying a onto b: its angle must equal
        the great-circle distance. A rotation with an extra twist about b would still pass the
        test above."""
        np.testing.assert_allclose(quat_angle_deg(shortest_arc_quat(self.a, self.b)),
                                   angle_between_deg(self.a, self.b), atol=1e-9)

    def test_shortest_arc_quat_is_unit(self):
        np.testing.assert_allclose(
            np.linalg.norm(shortest_arc_quat(self.a, self.b), axis=1), 1.0, atol=1e-12)

    def test_quat_multiply_matches_matrix_product(self):
        """Pins the composition order. The opposite convention is a sign flip on the cross
        term and produces angles of the right magnitude for the wrong composition."""
        p = shortest_arc_quat(self.a, self.b)
        q = shortest_arc_quat(self.b, random_unit(self.rng, N_RANDOM))
        expected = np.einsum('nij,njk->nik', self._as_scipy(p).as_matrix(),
                             self._as_scipy(q).as_matrix())
        np.testing.assert_allclose(self._as_scipy(quat_multiply(p, q)).as_matrix(),
                                   expected, atol=1e-12)

    def test_quat_conjugate_is_the_inverse(self):
        q = shortest_arc_quat(self.a, self.b)
        product = quat_multiply(q, quat_conjugate(q))
        np.testing.assert_allclose(np.abs(product[:, 0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(product[:, 1:], 0.0, atol=1e-12)

    def test_quat_angle_is_sign_invariant(self):
        """q and -q are the same rotation, so the reported angle must not depend on which
        representative the arithmetic happened to produce."""
        q = shortest_arc_quat(self.a, self.b)
        np.testing.assert_allclose(quat_angle_deg(q), quat_angle_deg(-q), atol=1e-12)

    def test_angle_between_matches_scipy(self):
        expected = np.degrees(np.array([
            Rotation.align_vectors(b[None, :], a[None, :])[0].magnitude()
            for a, b in zip(self.a[:500], self.b[:500])]))
        np.testing.assert_allclose(angle_between_deg(self.a[:500], self.b[:500]),
                                   expected, atol=1e-9)

    def test_angle_between_is_clipped_for_parallel_vectors(self):
        """Normalized dot products for near-parallel vectors land a few ulp outside [-1, 1]
        routinely, and both fields here are near-parallel most of the time. Unclipped this
        returns NaN on ordinary data."""
        a = random_unit(self.rng, 1000)
        nudged, _ = normalize_rows(a + 1e-9 * random_unit(self.rng, 1000))
        angles = angle_between_deg(a, nudged)
        self.assertTrue(np.all(np.isfinite(angles)))
        self.assertTrue(np.all(angles >= 0.0))
        np.testing.assert_allclose(angle_between_deg(a, a), 0.0, atol=1e-5)
        np.testing.assert_allclose(angle_between_deg(a, -a), 180.0, atol=1e-5)

    def test_arccos_noise_floor_stays_under_the_theorem_tolerance(self):
        """Measures the floor THEOREM_TOL_DEG is set from, so the constant stops being a
        guess. arccos(1 - eps) ~ sqrt(2 eps), so an angle that should be exactly zero comes
        out at ~1e-6 deg, and report_theorem's residuals are differences of two such angles.
        If numpy's arccos ever got worse than this, the theorem section would start reporting
        violations that are not violations — this is what would catch it first."""
        a = random_unit(self.rng, 200_000)
        floor = float(np.max(angle_between_deg(a, a)))
        self.assertGreater(floor, 0.0, "expected a nonzero floor; the premise has changed")
        self.assertLess(floor, THEOREM_TOL_DEG / 10.0)


class TestSphericalExcess(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_octant_has_excess_of_ninety_degrees(self):
        """The positive octant is an eighth of the sphere: area 4pi/8 = pi/2, i.e. 90 deg. The
        one configuration whose answer is known without computing it."""
        e = np.eye(3)
        self.assertAlmostEqual(
            abs(float(spherical_excess_deg(e[0:1], e[1:2], e[2:3])[0])), 90.0, places=9)

    def test_degenerate_triangles_have_zero_area(self):
        a = random_unit(self.rng, 1000)
        b = random_unit(self.rng, 1000)
        np.testing.assert_allclose(spherical_excess_deg(a, b, a), 0.0, atol=1e-9)
        np.testing.assert_allclose(spherical_excess_deg(a, a, b), 0.0, atol=1e-9)

    def test_coplanar_triangles_have_zero_area(self):
        """Three points on a common great circle enclose nothing. This is the exact case in
        which the paper's inequality is tight, so it has to come out as exactly zero and not
        as a small number."""
        normal = random_unit(self.rng, 1)[0]
        e1, _ = normalize_rows(np.cross(normal, self.rng.normal(size=3))[None, :])
        e2 = np.cross(normal, e1[0])
        phi = self.rng.uniform(0, 2 * np.pi, (3, 1000))
        points = [np.cos(p)[:, None] * e1 + np.sin(p)[:, None] * e2[None, :] for p in phi]
        np.testing.assert_allclose(spherical_excess_deg(*points), 0.0, atol=1e-9)

    def test_sign_flips_with_orientation(self):
        a, b, c = (random_unit(self.rng, 1000) for _ in range(3))
        np.testing.assert_allclose(spherical_excess_deg(a, b, c),
                                   -spherical_excess_deg(a, c, b), atol=1e-9)

    def test_wrapped_into_half_open_interval(self):
        a, b, c = (random_unit(self.rng, N_RANDOM) for _ in range(3))
        excess = spherical_excess_deg(a, b, c)
        self.assertTrue(np.all(excess > -180.0) and np.all(excess <= 180.0))


class TestTheorem(unittest.TestCase):
    """The four facts the paper's argument rests on, as exact identities over random
    configurations on the whole sphere — not on the narrow, nearly-degenerate triangles this
    dataset actually produces, where a sign error could hide inside the noise floor."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2)
        cls.u_parent = random_unit(rng, N_RANDOM)
        cls.u_child = random_unit(rng, N_RANDOM)
        cls.reference = random_unit(rng, 1)[0]
        cls.angles = correction_angles(cls.u_parent, cls.u_child, cls.reference)

    def test_fact_1_geodesic_triangle_inequality(self):
        violation = self.angles['theta_rel'] - self.angles['theta_sum']
        self.assertLessEqual(float(violation.max()), THEOREM_TOL_DEG)

    def test_fact_2_minimum_rotation_property(self):
        violation = self.angles['theta_rel'] - self.angles['theta_comp']
        self.assertLessEqual(float(violation.max()), THEOREM_TOL_DEG)

    def test_fact_3_exact_closed_form(self):
        """cos(theta_comp/2) = cos(theta_rel/2) cos(psi/2). Checked as the residual
        correction_angles computes, which derives theta_comp from the composed quaternion and
        psi from the area independently — so this is a check and not a tautology."""
        self.assertLess(float(np.abs(self.angles['_identity_residual']).max()), 1e-9)

    def test_fact_4_excess_is_the_spherical_area(self):
        self.assertLess(float(np.abs(self.angles['_psi_residual']).max()), 1e-8)

    def test_composed_correction_carries_child_onto_parent(self):
        """The geometric fact fact 2 rests on. If this fails, theta_comp is measuring some
        other rotation and the inequality above is a coincidence."""
        q_parent = shortest_arc_quat(self.u_parent, np.broadcast_to(
            self.reference, self.u_parent.shape))
        q_child = shortest_arc_quat(self.u_child, np.broadcast_to(
            self.reference, self.u_child.shape))
        q_comp = quat_multiply(quat_conjugate(q_parent), q_child)
        rotated = Rotation.from_quat(
            np.concatenate([q_comp[:, 1:], q_comp[:, :1]], axis=1)).apply(self.u_child)
        np.testing.assert_allclose(rotated, self.u_parent, atol=1e-9)

    def test_equality_when_reference_is_coplanar(self):
        """The tight case: a reference on the great circle through the two measurements costs
        nothing, so theta_comp must equal theta_rel exactly. This is why fact 2 is <= and not
        <, and a version of the code that added a spurious positive term would still pass the
        inequality tests while failing this one."""
        rng = np.random.default_rng(3)
        normal = random_unit(rng, 1)[0]
        e1, _ = normalize_rows(np.cross(normal, rng.normal(size=3))[None, :])
        e2 = np.cross(normal, e1[0])
        phi = rng.uniform(0, 2 * np.pi, (3, 2000))
        u_parent, reference, u_child = [
            np.cos(p)[:, None] * e1 + np.sin(p)[:, None] * e2[None, :] for p in phi]

        # A per-sample reference is not what correction_angles takes, so this walks the same
        # arithmetic with a broadcast reference per row via a loop over a small sample.
        for i in range(0, 2000, 7):
            angles = correction_angles(u_parent[i:i + 1], u_child[i:i + 1], reference[i])
            self.assertAlmostEqual(float(angles['theta_comp'][0]),
                                   float(angles['theta_rel'][0]), places=8)
            self.assertAlmostEqual(float(angles['psi'][0]), 0.0, places=8)

    def test_relative_is_not_bounded_by_the_better_absolute(self):
        """The documented caveat, as a test. If one measurement sits exactly on the reference
        its absolute correction is zero while theta_rel is the other's full deviation, so
        theta_rel <= min(theta_parent, theta_child) is FALSE in general. A future refactor that
        starts reporting it as a guarantee should fail here."""
        reference = np.array([0.0, 0.0, 1.0])
        u_parent = reference[None, :]
        u_child = np.array([[np.sin(np.radians(10.0)), 0.0, np.cos(np.radians(10.0))]])
        angles = correction_angles(u_parent, u_child, reference)
        self.assertAlmostEqual(float(angles['theta_min_abs'][0]), 0.0, places=9)
        self.assertAlmostEqual(float(angles['theta_rel'][0]), 10.0, places=6)
        self.assertGreater(float(angles['theta_rel'][0]), float(angles['theta_min_abs'][0]))
        # ...while the two bounded claims still hold on the very same sample.
        self.assertLessEqual(float(angles['theta_rel'][0]),
                             float(angles['theta_comp'][0]) + THEOREM_TOL_DEG)
        self.assertLessEqual(float(angles['theta_rel'][0]),
                             float(angles['theta_sum'][0]) + THEOREM_TOL_DEG)

    def test_theta_rel_does_not_depend_on_the_reference(self):
        """theta_rel is a property of the pair alone. The report prints it once per reference
        variant and reads its repetition as a check that the variants differ only in g, so
        that has to actually be true."""
        rng = np.random.default_rng(4)
        first = correction_angles(self.u_parent, self.u_child, random_unit(rng, 1)[0])
        second = correction_angles(self.u_parent, self.u_child, random_unit(rng, 1)[0])
        np.testing.assert_allclose(first['theta_rel'], second['theta_rel'], atol=1e-12)


class TestExactValues(unittest.TestCase):
    """One configuration worked out by hand, so the tests above cannot all be satisfied by a
    self-consistently wrong implementation."""

    def test_orthogonal_octant_configuration(self):
        """u_parent = x, u_child = y, reference = z. Every pairwise angle is 90 deg, so
        theta_rel = 90, theta_sum = 180, and psi is the octant's 90 deg of area. The closed
        form then gives cos(theta_comp/2) = cos(45)cos(45) = 1/2, i.e. theta_comp = 120 deg."""
        angles = correction_angles(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]),
                                   np.array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(float(angles['theta_rel'][0]), 90.0, places=9)
        self.assertAlmostEqual(float(angles['theta_parent'][0]), 90.0, places=9)
        self.assertAlmostEqual(float(angles['theta_child'][0]), 90.0, places=9)
        self.assertAlmostEqual(float(angles['theta_sum'][0]), 180.0, places=9)
        self.assertAlmostEqual(abs(float(angles['psi'][0])), 90.0, places=9)
        self.assertAlmostEqual(float(angles['theta_comp'][0]), 120.0, places=6)
        self.assertAlmostEqual(float(angles['excess'][0]), 30.0, places=6)

    def test_small_angle_regime(self):
        """The regime this dataset actually lives in: a few degrees apart, reference off the
        plane. The excess is then second order in the angles, which is why it needs a large
        pooled sample to see and why the closed form rather than a linearization is used."""
        deg = np.radians(1.0)
        u_parent = np.array([[np.sin(deg), 0.0, np.cos(deg)]])
        u_child = np.array([[0.0, np.sin(deg), np.cos(deg)]])
        angles = correction_angles(u_parent, u_child, np.array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(float(angles['theta_rel'][0]), np.degrees(deg * np.sqrt(2)),
                               places=4)
        self.assertLess(float(angles['excess'][0]), 0.01)
        self.assertGreater(float(angles['excess'][0]), 0.0)


class TestQuadratureApproximation(unittest.TestCase):
    """The small-angle reading of fact 3, which is what the report uses to explain why fact 2's
    gap is negligible on this data. If theta_comp did NOT add in quadrature, the explanation in
    report_pooled_angles would be wrong even though the exact identity behind it is right."""

    def test_theta_comp_adds_in_quadrature_for_small_angles(self):
        rng = np.random.default_rng(20)
        axis = np.array([0.0, 0.0, 1.0])
        # A sliver triangle of the kind this dataset produces: everything within ~10 deg.
        u_parent, _ = normalize_rows(axis + 0.1 * rng.normal(size=(5000, 3)))
        u_child, _ = normalize_rows(axis + 0.1 * rng.normal(size=(5000, 3)))
        reference, _ = normalize_rows((axis + 0.1 * rng.normal(size=3))[None, :])

        angles = correction_angles(u_parent, u_child, reference[0])
        quadrature = np.hypot(angles['theta_rel'], angles['psi_abs'])
        # Third-order accurate, so a few degrees of angle leaves well under 0.01 deg of error.
        self.assertLess(float(np.max(np.abs(angles['theta_comp'] - quadrature))), 0.01)

    def test_quadrature_makes_the_excess_second_order(self):
        """The quantitative form of the negative result: a psi much smaller than theta_rel costs
        almost nothing, which is why fact 2's measured gap is ~0.06 deg."""
        angles = correction_angles(
            np.array([[np.sin(np.radians(5.0)), 0.0, np.cos(np.radians(5.0))]]),
            np.array([[np.sin(np.radians(-5.0)), 0.0, np.cos(np.radians(-5.0))]]),
            np.array([0.0, 0.0, 1.0]))
        # Reference on the great circle through both -> zero area -> zero cost.
        self.assertAlmostEqual(float(angles['psi'][0]), 0.0, places=8)
        self.assertAlmostEqual(float(angles['excess'][0]), 0.0, places=8)


class TestInvariant(unittest.TestCase):
    """Facts 5-7. The inter-field angle is the only quantity in this experiment that no
    orientation can change, so the tests here are about that invariance holding."""

    def _plate(self, rng, acc, mag):
        """A PlateTrial whose IMU carries the given body-frame readings and whose world_trace
        rotations are random — precisely so that a function that accidentally rotates anything
        produces a different answer on every run."""
        n = len(acc)
        timestamps = np.arange(n) / 100.0
        rotations = Rotation.random(n, random_state=int(rng.integers(1 << 30))).as_matrix()
        return PlateTrial('test', IMUTrace(timestamps, np.zeros((n, 3)), acc, mag),
                          WorldTrace(timestamps, np.zeros((n, 3)), rotations))

    def setUp(self):
        self.rng = np.random.default_rng(21)
        n = 500
        self.acc_parent = self.rng.normal(size=(n, 3)) + np.array([0.0, 9.81, 0.0])
        self.mag_parent = self.rng.normal(size=(n, 3)) * 0.05 + np.array([0.3, -0.9, 0.1])
        self.acc_child = self.rng.normal(size=(n, 3)) + np.array([0.0, 9.81, 0.0])
        self.mag_child = self.rng.normal(size=(n, 3)) * 0.05 + np.array([0.35, -0.88, 0.05])
        self.reference_mag = np.array([0.3, -0.9, 0.1])

    def test_beta_is_invariant_to_the_mocap_orientation(self):
        """The claim that makes facts 5-7 immune to the alignment floor. Two plates with the
        same readings and different ground-truth rotations must give identical betas."""
        parent_a = self._plate(self.rng, self.acc_parent, self.mag_parent)
        parent_b = self._plate(self.rng, self.acc_parent, self.mag_parent)
        child = self._plate(self.rng, self.acc_child, self.mag_child)
        first, _ = invariant_angles(parent_a, child, self.reference_mag, project=False)
        second, _ = invariant_angles(parent_b, child, self.reference_mag, project=False)
        for metric in INVARIANT_METRICS:
            np.testing.assert_allclose(first[metric], second[metric], atol=1e-12,
                                       err_msg=f"{metric} moved with the mocap rotation")

    def test_beta_is_invariant_to_rotating_both_readings(self):
        """The other half of invariance: rotating a sensor's own acc and mag together by any
        rotation leaves beta unchanged, since beta is the angle between them."""
        rotation = Rotation.random(1, random_state=7).as_matrix()[0]
        rotated_parent = self._plate(self.rng, self.acc_parent @ rotation.T,
                                     self.mag_parent @ rotation.T)
        parent = self._plate(self.rng, self.acc_parent, self.mag_parent)
        child = self._plate(self.rng, self.acc_child, self.mag_child)
        plain, _ = invariant_angles(parent, child, self.reference_mag, project=False)
        turned, _ = invariant_angles(rotated_parent, child, self.reference_mag, project=False)
        np.testing.assert_allclose(plain['beta_parent'], turned['beta_parent'], atol=1e-9)

    def test_fact_6_triangle_inequality(self):
        parent = self._plate(self.rng, self.acc_parent, self.mag_parent)
        child = self._plate(self.rng, self.acc_child, self.mag_child)
        metrics, _ = invariant_angles(parent, child, self.reference_mag, project=False)
        violation = metrics['dip_rel'] - metrics['dip_sum']
        self.assertLessEqual(float(violation.max()), THEOREM_TOL_DEG)

    def test_floors_are_half_the_mismatches(self):
        """Fact 5: the irreducible per-vector error is HALF the mismatch. A missing factor of
        two here would double one side of the paper's headline comparison."""
        parent = self._plate(self.rng, self.acc_parent, self.mag_parent)
        child = self._plate(self.rng, self.acc_child, self.mag_child)
        metrics, _ = invariant_angles(parent, child, self.reference_mag, project=False)
        np.testing.assert_allclose(metrics['floor_rel'], 0.5 * metrics['dip_rel'], atol=1e-12)
        np.testing.assert_allclose(
            metrics['floor_abs'],
            0.5 * np.maximum(metrics['dip_parent'], metrics['dip_child']), atol=1e-12)

    def test_wahba_splits_an_invariant_mismatch_in_half(self):
        """Fact 5 itself, against an independent Kabsch solve rather than against the formula
        the code uses. This is what licenses calling dip/2 an achievable error and not a bound."""
        rng = np.random.default_rng(22)
        for _ in range(300):
            body, _ = normalize_rows(rng.normal(size=(2, 3)))
            reference, _ = normalize_rows(rng.normal(size=(2, 3)))
            mismatch = abs(float(angle_between_deg(body[:1], body[1:])[0])
                           - float(angle_between_deg(reference[:1], reference[1:])[0]))
            matrix = sum(np.outer(r, b) for b, r in zip(body, reference))
            u, _, vt = np.linalg.svd(matrix)
            rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
            errors = angle_between_deg((rotation @ body.T).T, reference)
            np.testing.assert_allclose(errors, mismatch / 2.0, atol=1e-6)

    def test_global_beta_is_constant_across_samples(self):
        """beta_global is one number per trial by construction. If it ever varied per sample the
        absolute mismatch would be measuring the reference's noise instead of the sensor's."""
        parent = self._plate(self.rng, self.acc_parent, self.mag_parent)
        child = self._plate(self.rng, self.acc_child, self.mag_child)
        metrics, _ = invariant_angles(parent, child, self.reference_mag, project=False)
        self.assertEqual(len(np.unique(metrics['beta_global'])), 1)

    def test_reference_magnitude_does_not_matter(self):
        """beta_global is an angle, so scaling the global field must not move it."""
        parent = self._plate(self.rng, self.acc_parent, self.mag_parent)
        child = self._plate(self.rng, self.acc_child, self.mag_child)
        plain, _ = invariant_angles(parent, child, self.reference_mag, project=False)
        scaled, _ = invariant_angles(parent, child, self.reference_mag * 91.0, project=False)
        np.testing.assert_allclose(plain['beta_global'], scaled['beta_global'], atol=1e-12)

    def test_identical_sensors_have_zero_relative_mismatch(self):
        """The ideal case, and the direction the projection pushes toward: two sensors reading
        the same fields have no irreducible relative error at all, however wrong both are
        against the global reference."""
        parent = self._plate(self.rng, self.acc_parent, self.mag_parent)
        twin = self._plate(self.rng, self.acc_parent, self.mag_parent)
        metrics, _ = invariant_angles(parent, twin, self.reference_mag, project=False)
        np.testing.assert_allclose(metrics['dip_rel'], 0.0, atol=1e-9)
        np.testing.assert_allclose(metrics['floor_rel'], 0.0, atol=1e-9)
        self.assertGreater(float(np.median(metrics['dip_parent'])), 0.0)

    def test_invalid_when_either_reading_has_no_direction(self):
        n = len(self.acc_parent)
        acc = self.acc_parent.copy()
        acc[:10] = 0.0
        parent = self._plate(self.rng, acc, self.mag_parent)
        child = self._plate(self.rng, self.acc_child, self.mag_child)
        _, valid = invariant_angles(parent, child, self.reference_mag, project=False)
        self.assertFalse(valid[:10].any())
        self.assertTrue(valid[10:].all())
        self.assertEqual(len(valid), n)


class TestInvariantTables(unittest.TestCase):
    def _computed(self, n: int = 200):
        rng = np.random.default_rng(23)
        computed = {}
        for joint in list(JOINTS)[:2]:
            for field in INVARIANT_FIELDS:
                beta_parent = rng.uniform(140.0, 160.0, n)
                beta_child = beta_parent + rng.normal(0.0, 3.0, n)
                beta_global = np.full(n, 153.0)
                metrics = {
                    'beta_parent': beta_parent, 'beta_child': beta_child,
                    'beta_global': beta_global,
                    'dip_rel': np.abs(beta_parent - beta_child),
                    'dip_parent': np.abs(beta_parent - beta_global),
                    'dip_child': np.abs(beta_child - beta_global),
                }
                metrics['dip_sum'] = metrics['dip_parent'] + metrics['dip_child']
                metrics['floor_rel'] = 0.5 * metrics['dip_rel']
                metrics['floor_abs'] = 0.5 * np.maximum(metrics['dip_parent'],
                                                        metrics['dip_child'])
                metrics['valid'] = np.ones(n, dtype=bool)
                computed[(joint, field)] = metrics
        return computed

    def test_samples_table_shape_and_columns(self):
        table = invariant_samples_table(self._computed())
        self.assertEqual(len(table), 2 * len(INVARIANT_FIELDS) * len(range(0, 200, SAMPLE_STRIDE)))
        for metric in INVARIANT_METRICS:
            self.assertIn(metric, table.columns)

    def test_stats_table_reports_fact_6_holding(self):
        stats = invariant_stats_table(self._computed())
        self.assertTrue((stats['frac_rel_below_sum'] == 1.0).all())
        self.assertLessEqual(float(stats['max_violation_sum'].max()), THEOREM_TOL_DEG)

    def test_summarize_separates_the_two_families(self):
        """A reader comparing a 0.06 deg gap against a 3 deg one because the rows were pooled is
        the failure this column exists to prevent."""
        rng = np.random.default_rng(24)
        direction = pd.DataFrame({
            'subject': 'Subject01', 'activity': 'walking', 'field': 'mag', 'joint': 'R_Knee',
            **{m: rng.uniform(0, 5, 100) for m in ANGLE_METRICS}})
        invariant = pd.DataFrame({
            'subject': 'Subject01', 'activity': 'walking', 'field': INVARIANT_FIELDS[0],
            'joint': 'R_Knee', **{m: rng.uniform(0, 20, 100) for m in INVARIANT_METRICS}})
        summary = summarize(direction, invariant)
        self.assertEqual(set(summary['family']), {'direction', 'invariant'})
        self.assertEqual(set(summary.loc[summary['family'] == 'direction', 'metric']),
                         set(ANGLE_METRICS))
        self.assertEqual(set(summary.loc[summary['family'] == 'invariant', 'metric']),
                         set(INVARIANT_METRICS))
        self.assertEqual(set(summary['unit']), {'deg'})

    def test_summarize_handles_one_family_missing(self):
        summary = summarize(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(summary.empty)
        self.assertIn('family', summary.columns)


class TestReferenceDirections(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(5)
        self.u_parent = random_unit(self.rng, 500)
        self.u_child = random_unit(self.rng, 500)
        self.valid = np.ones(500, dtype=bool)
        self.subject_field = np.array([0.3, -0.1, 0.9])
        self.torso_field = np.array([0.35, -0.05, 0.88])

    def test_every_variant_is_present_and_unit(self):
        for field in FIELDS:
            references = reference_directions(field, self.u_parent, self.u_child, self.valid,
                                              self.subject_field, self.torso_field)
            self.assertEqual(set(references), set(REFERENCE_VARIANTS[field]))
            for variant, reference in references.items():
                self.assertAlmostEqual(float(np.linalg.norm(reference)), 1.0, places=12,
                                       msg=f"{field}/{variant} is not a unit vector")

    def test_primary_is_the_first_variant(self):
        for field in FIELDS:
            self.assertEqual(PRIMARY_REFERENCE[field], REFERENCE_VARIANTS[field][0])

    def test_gravity_variant_is_the_gravity_direction(self):
        """A sign flip here would silently turn every accelerometer angle into its
        supplement — 180 deg instead of 5 — which is exactly the kind of error the repo's
        gravity-convention test exists to catch elsewhere."""
        references = reference_directions('acc', self.u_parent, self.u_child, self.valid,
                                          self.subject_field, self.torso_field)
        expected = EXPECTED_GRAVITY / np.linalg.norm(EXPECTED_GRAVITY)
        np.testing.assert_allclose(references['expected_gravity'], expected, atol=1e-12)

    def test_spherical_mean_recovers_a_tight_cluster(self):
        """The steelman reference has to actually be near the measurements, or it is not a
        steelman and the sensitivity report understates how good a tuned reference can be."""
        axis = np.array([0.0, 1.0, 0.0])
        tight, _ = normalize_rows(axis + 0.02 * self.rng.normal(size=(500, 3)))
        references = reference_directions('acc', tight, tight, self.valid,
                                          self.subject_field, self.torso_field)
        self.assertLess(float(angle_between_deg(references['spherical_mean'][None, :],
                                                axis[None, :])[0]), 1.0)

    def test_spherical_mean_is_nan_when_nothing_is_valid(self):
        """A joint with no usable samples must produce a reference the caller can detect and
        skip, not a silently arbitrary direction."""
        references = reference_directions('acc', self.u_parent, self.u_child,
                                          np.zeros(500, dtype=bool),
                                          self.subject_field, self.torso_field)
        self.assertTrue(np.all(np.isnan(references['spherical_mean'])))

    def test_non_unit_input_fields_are_normalized(self):
        references = reference_directions('mag', self.u_parent, self.u_child, self.valid,
                                          self.subject_field * 137.0, self.torso_field * 0.01)
        for variant in ('subject_median', 'torso_median'):
            self.assertAlmostEqual(float(np.linalg.norm(references[variant])), 1.0, places=12)


class TestTables(unittest.TestCase):
    """The table builders, against a synthetic `computed` dict. These decide what the figure
    can draw and what the report can quote, so the checks are on structure and on the
    stride/validity handling rather than on the geometry, which is covered above."""

    def _computed(self, n: int = 200):
        rng = np.random.default_rng(6)
        computed = {}
        for joint in list(JOINTS)[:2]:
            for field in FIELDS:
                for variant in REFERENCE_VARIANTS[field]:
                    u_parent = random_unit(rng, n)
                    u_child, _ = normalize_rows(u_parent + 0.05 * rng.normal(size=(n, 3)))
                    angles = correction_angles(u_parent, u_child, random_unit(rng, 1)[0])
                    angles['valid'] = np.ones(n, dtype=bool)
                    computed[(joint, field, variant)] = angles
        return computed

    def test_angle_samples_keeps_only_primary_references(self):
        table = angle_samples_table(self._computed())
        self.assertFalse(table.empty)
        for field, group in table.groupby('field'):
            self.assertGreater(len(group), 0)
        # One row per kept sample per (joint, field), i.e. no reference-variant duplication.
        expected = 2 * len(FIELDS) * len(range(0, 200, SAMPLE_STRIDE))
        self.assertEqual(len(table), expected)

    def test_angle_samples_has_every_metric_the_summary_needs(self):
        table = angle_samples_table(self._computed())
        for metric in ANGLE_METRICS:
            self.assertIn(metric, table.columns)

    def test_angle_samples_drops_invalid_rows(self):
        computed = self._computed()
        for angles in computed.values():
            angles['valid'] = np.zeros(len(angles['valid']), dtype=bool)
        self.assertTrue(angle_samples_table(computed).empty)

    def test_joint_stats_counts_only_valid_samples(self):
        """degenerate_frac and n_samples are what the report divides by, so a validity mask
        that leaked into the counts would misstate every fraction in section 6."""
        computed = self._computed()
        for angles in computed.values():
            angles['valid'][:50] = False
        stats = joint_stats_table(computed)
        self.assertTrue((stats['n_samples'] == 150).all())
        self.assertTrue((stats['n_total'] == 200).all())
        np.testing.assert_allclose(stats['degenerate_frac'], 0.25, atol=1e-12)

    def test_joint_stats_computed_before_the_stride(self):
        """joint_stats is the table the report quotes, so it must see every sample. If it were
        computed from the strided table its n would be a fifth of this."""
        stats = joint_stats_table(self._computed(n=200))
        self.assertTrue((stats['n_samples'] == 200).all())

    def test_joint_stats_reports_every_reference_variant(self):
        stats = joint_stats_table(self._computed())
        for field in FIELDS:
            present = set(stats.loc[stats['field'] == field, 'reference'])
            self.assertEqual(present, set(REFERENCE_VARIANTS[field]))

    def test_joint_stats_violations_are_non_positive(self):
        stats = joint_stats_table(self._computed())
        self.assertLessEqual(float(stats['max_violation_sum'].max()), THEOREM_TOL_DEG)
        self.assertLessEqual(float(stats['max_violation_comp'].max()), THEOREM_TOL_DEG)
        self.assertTrue((stats['frac_rel_below_comp'] == 1.0).all())
        self.assertTrue((stats['frac_rel_below_sum'] == 1.0).all())

    def test_summarize_margins_are_recomputed_not_averaged(self):
        """The pooled row must be a quantile of the samples, not a mean of the per-trial
        quantiles. Two trials of very different length make the difference visible."""
        rng = np.random.default_rng(7)
        samples = pd.concat([
            pd.DataFrame({'subject': 'Subject01', 'activity': 'walking', 'field': 'mag',
                          'joint': 'R_Knee', **{m: rng.uniform(0, 1, 1000)
                                                for m in ANGLE_METRICS}}),
            pd.DataFrame({'subject': 'Subject02', 'activity': 'walking', 'field': 'mag',
                          'joint': 'R_Knee', **{m: rng.uniform(10, 11, 10)
                                                for m in ANGLE_METRICS}}),
        ], ignore_index=True)
        summary = summarize(samples, pd.DataFrame())
        pooled = summary[(summary['subject'] == 'all') & (summary['metric'] == 'theta_rel')
                         & (summary['joint'] == 'all')].iloc[0]
        self.assertEqual(int(pooled['n_samples']), 1010)
        expected = np.quantile(samples['theta_rel'], 0.5)
        self.assertAlmostEqual(float(pooled['p50']), float(expected), places=9)
        # A mean of the two per-trial medians would land near 5.5, not near 0.5.
        self.assertLess(float(pooled['p50']), 1.0)

    def test_summarize_covers_every_metric_and_field(self):
        rng = np.random.default_rng(8)
        samples = pd.concat([
            pd.DataFrame({'subject': 'Subject01', 'activity': 'walking', 'field': field,
                          'joint': joint,
                          **{m: rng.uniform(0, 5, 50) for m in ANGLE_METRICS}})
            for field in FIELDS for joint in list(JOINTS)[:3]], ignore_index=True)
        summary = summarize(samples, pd.DataFrame())
        self.assertEqual(set(summary['metric']), set(ANGLE_METRICS))
        self.assertEqual(set(summary['field']), set(FIELDS))
        self.assertEqual(set(summary['unit']), {'deg'})
        for q in QUANTILES:
            self.assertIn(f"p{int(round(q * 100)):02d}", summary.columns)

    def test_summarize_empty_input_returns_empty_frame(self):
        summary = summarize(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(summary.empty)
        self.assertIn('theta_rel', ANGLE_METRICS)


class TestNormalizeRows(unittest.TestCase):
    def test_zero_rows_return_zero_not_nan(self):
        """The degenerate path is the normal path here: the caller masks on the returned norms,
        so dividing first and masking afterwards would warn on every dropout sample."""
        v = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        unit, norms = normalize_rows(v)
        np.testing.assert_allclose(unit[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(norms, [0.0, 5.0])
        np.testing.assert_allclose(unit[1], [0.6, 0.8, 0.0])

    def test_norms_are_returned_unmodified(self):
        rng = np.random.default_rng(9)
        v = rng.normal(size=(100, 3)) * 137.0
        unit, norms = normalize_rows(v)
        np.testing.assert_allclose(norms, np.linalg.norm(v, axis=1), atol=1e-12)
        np.testing.assert_allclose(unit * norms[:, None], v, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
