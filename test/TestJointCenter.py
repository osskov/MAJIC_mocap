"""
Pins the mathematical identities experiments/joint_center.py rests on.

Every one of these is an equality that is either exactly true or the module is wrong, and every
one of them would break SILENTLY: a sign flip, a transpose or a float32 accumulation still
returns six plausible-looking millimetre numbers, and the only way to notice is to check them
against something independent. That is what this file is — the identities are asserted against a
second derivation, not against a recorded value.

Synthetic throughout, with the joint centre KNOWN BY CONSTRUCTION: two segments are given
arbitrary rotations and their positions are then placed so that both reach the same point, so
the fit has an exact right answer to be scored against.
"""
import os
import pathlib
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from experiments.global_assumptions import ALBORNO, IMOVE, canonical_joint
from src.toolchest.WorldTrace import (MIN_JOINT_CENTER_FRAMES,
                                      UnderdeterminedJointCenter)
from experiments.joint_center import (combine_normal_equations, fit_conditioning,
                                      invariant_pairs, normal_equations, residual_rms_from,
                                      solve_offsets, stored_trial_keys)
# aliased: the tests use a local `terms` for the returned dict, which would shadow it
from experiments.inertial_joint_center import cost_parts, terms as model_terms
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace

FS = 100.0
R_PARENT = np.array([-0.21, 0.03, 0.06])
R_CHILD = np.array([0.15, -0.02, 0.05])


def make_pair(n=2000, seed=0, parent_offset=R_PARENT, child_offset=R_CHILD, wobble=0.0):
    """Two WorldTraces that share one joint centre exactly, by construction.

    Rotations are a smooth random walk rather than white noise, because the conditioning of this
    fit depends entirely on the RELATIVE rotation between the pair and a white sequence would
    excite it far more evenly than a limb ever does.

    `wobble` displaces the child's implied centre by a random walk of that size in metres, which
    is how a joint that is not a fixed point is simulated.
    """
    rng = np.random.default_rng(seed)
    timestamps = np.arange(n) / FS

    def walk(scale):
        steps = np.cumsum(rng.normal(0.0, scale, size=(n, 3)), axis=0)
        return Rotation.from_rotvec(steps).as_matrix()

    rotations_parent, rotations_child = walk(0.02), walk(0.03)
    joint = np.column_stack([np.sin(timestamps), 0.9 + 0.05 * np.cos(timestamps),
                             0.3 * timestamps])
    if wobble:
        joint = joint + np.cumsum(rng.normal(0.0, wobble, size=(n, 3)), axis=0)

    positions_parent = joint - np.einsum('nij,j->ni', rotations_parent, parent_offset)
    positions_child = joint - np.einsum('nij,j->ni', rotations_child, child_offset)
    return (WorldTrace(timestamps, positions_parent, rotations_parent),
            WorldTrace(timestamps, positions_child, rotations_child))


def make_plate(world, seed=0, name='seg'):
    """A PlateTrial whose IMU trace is the one that world trace implies."""
    rng = np.random.default_rng(seed)
    n = len(world)
    gyro = rng.normal(0.0, 1.0, size=(n, 3))
    acc = rng.normal(0.0, 3.0, size=(n, 3)) + np.array([0.0, 9.81, 0.0])
    mag = np.tile([1.0, 0.0, 0.0], (n, 1))
    return PlateTrial(name, IMUTrace(world.timestamps, gyro, acc, mag), world)


class TestFitRecoversTheTruth(unittest.TestCase):
    def test_an_exact_ball_joint_is_recovered_exactly(self):
        """The whole module is downstream of this. If the fit cannot recover offsets it was
        handed by construction, nothing above it means anything."""
        parent, child = make_pair()
        r_parent, r_child, residual = parent.get_joint_center(child)
        np.testing.assert_allclose(r_parent, R_PARENT, atol=1e-9)
        np.testing.assert_allclose(r_child, R_CHILD, atol=1e-9)
        self.assertLess(np.abs(residual).max(), 1e-9)

    def test_the_residual_is_the_separation_of_the_two_implied_centres(self):
        """What the report calls the residual, in the terms it is described in. A joint that
        wobbles must produce exactly the wobble."""
        parent, child = make_pair(wobble=2e-4, seed=3)
        r_parent, r_child, residual = parent.get_joint_center(child)
        implied_parent = parent.positions + np.einsum('nij,j->ni', parent.rotations, r_parent)
        implied_child = child.positions + np.einsum('nij,j->ni', child.rotations, r_child)
        np.testing.assert_allclose(residual, implied_parent - implied_child, atol=1e-12)

    def test_the_mask_excludes_frames(self):
        """The valid mask is the module's stated improvement over `get_joint_center`. Corrupting
        the frames outside it must not move the answer at all."""
        parent, child = make_pair()
        mask = np.zeros(len(parent), dtype=bool)
        mask[:1000] = True
        clean = parent.get_joint_center(child, mask, min_frames=1)[0]

        rng = np.random.default_rng(1)
        corrupted = WorldTrace(parent.timestamps, parent.positions.copy(),
                               parent.rotations.copy())
        corrupted.positions[1000:] += rng.normal(0.0, 5.0, size=(len(parent) - 1000, 3))
        np.testing.assert_allclose(corrupted.get_joint_center(child, mask, min_frames=1)[0],
                                   clean, atol=1e-9)



class TestValidMaskAndGuard(unittest.TestCase):
    """`WorldTrace.get_joint_center` masks by `valid` and refuses an under-determined fit."""

    def test_padded_frames_are_excluded_from_the_fit(self):
        """The behaviour change: corrupting the invalid frames must not move the answer. Before
        the mask this shifted the offsets by up to 90 mm on real trials."""
        parent, child = make_pair(n=2000)
        valid = np.ones(len(parent), dtype=bool)
        valid[1000:] = False
        rng = np.random.default_rng(4)
        wrecked = WorldTrace(parent.timestamps,
                             parent.positions + np.where(valid[:, None], 0.0,
                                                         rng.normal(0, 5.0, (len(parent), 3))),
                             parent.rotations, valid=valid)
        clean = WorldTrace(parent.timestamps, parent.positions, parent.rotations, valid=valid)
        masked_child = WorldTrace(child.timestamps, child.positions, child.rotations, valid=valid)
        np.testing.assert_allclose(wrecked.get_joint_center(masked_child, min_frames=100)[0],
                                   clean.get_joint_center(masked_child, min_frames=100)[0],
                                   atol=1e-9)

    def test_an_explicit_all_true_mask_reproduces_the_unmasked_fit(self):
        """How a caller asks for the old fit-over-everything behaviour now that the default is
        `valid`. joint_fits uses exactly this to measure what masking is worth."""
        parent, child = make_pair(n=2000)
        valid = np.ones(len(parent), dtype=bool)
        valid[1000:] = False
        flagged = WorldTrace(parent.timestamps, parent.positions, parent.rotations, valid=valid)
        everything = np.ones(len(parent), dtype=bool)
        np.testing.assert_allclose(flagged.get_joint_center(child, everything)[0],
                                   parent.get_joint_center(child, everything)[0], atol=1e-9)

    def test_too_few_usable_frames_raises_rather_than_returning_a_number(self):
        """An under-determined offset is indistinguishable from a good one at the call site."""
        parent, child = make_pair(n=2000)
        # Well under MIN_JOINT_CENTER_FRAMES, and asserted against the constant rather than a
        # literal so the test tracks the policy instead of pinning today's number.
        valid = np.zeros(len(parent), dtype=bool)
        valid[:MIN_JOINT_CENTER_FRAMES // 5] = True
        thin = WorldTrace(parent.timestamps, parent.positions, parent.rotations, valid=valid)
        with self.assertRaises(UnderdeterminedJointCenter):
            thin.get_joint_center(child)

    def test_a_frame_count_at_the_floor_is_accepted(self):
        """The boundary, so the guard is off-by-one correct in the direction that matters."""
        parent, child = make_pair(n=2000)
        valid = np.zeros(len(parent), dtype=bool)
        valid[:MIN_JOINT_CENTER_FRAMES] = True
        at_floor = WorldTrace(parent.timestamps, parent.positions, parent.rotations, valid=valid)
        masked_child = WorldTrace(child.timestamps, child.positions, child.rotations, valid=valid)
        at_floor.get_joint_center(masked_child)   # must not raise

    def test_the_residual_covers_every_frame_not_only_the_fitted_ones(self):
        """So a caller can score the fit on frames it never saw — the holdout check relies on it."""
        parent, child = make_pair(n=2000)
        valid = np.ones(len(parent), dtype=bool)
        valid[1000:] = False
        flagged = WorldTrace(parent.timestamps, parent.positions, parent.rotations, valid=valid)
        self.assertEqual(len(flagged.get_joint_center(child, min_frames=100)[2]), len(parent))


class TestAccFilter(unittest.TestCase):
    def test_the_acc_lowpass_is_applied_to_acc_and_A_together(self):
        """LP(a + A o) = LP(a) + LP(A) o holds only because o is constant, so both have to be
        filtered. Filtering the accelerometer alone would leave the model inconsistent."""
        parent, _ = make_pair(n=600)
        plate = make_plate(parent)
        mask = np.ones(len(parent), dtype=bool)
        raw = model_terms(plate, mask, ('backward', None, None, None))
        filtered = model_terms(plate, mask, ('backward', None, None, 10.0))
        self.assertGreater(np.abs(raw['acc'] - filtered['acc']).max(), 1e-6)
        self.assertGreater(np.abs(raw['A'] - filtered['A']).max(), 1e-6)

    def test_a_cutoff_above_nyquist_is_skipped_not_raised(self):
        """The datasets mix 100 Hz and 40 Hz trials, so a sweep cutoff can exceed Nyquist for
        some of them; dropping the filter is the honest degradation."""
        parent, _ = make_pair(n=600)
        plate = make_plate(parent)
        mask = np.ones(len(parent), dtype=bool)
        raw = model_terms(plate, mask, ('backward', None, None, None))
        above = model_terms(plate, mask, ('backward', None, None, 5000.0))
        np.testing.assert_allclose(above['acc'], raw['acc'], atol=1e-12)


class TestNormalEquations(unittest.TestCase):
    """The sufficient-statistic representation: 44 numbers that must stand in for the trial."""

    def test_the_normal_equations_give_the_same_offsets_as_the_direct_fit(self):
        parent, child = make_pair(wobble=1e-4, seed=5)
        direct = np.concatenate(parent.get_joint_center(child)[:2])
        np.testing.assert_allclose(solve_offsets(normal_equations(parent, child)), direct,
                                   atol=1e-9)

    def test_the_gram_matrix_matches_its_closed_form(self):
        """The identity the whole conditioning section rests on: because both blocks of the
        design are orthonormal, M^T M = [[N I, -S], [-S^T, N I]] with S = sum R_p^T R_c. If this
        drifts, `fit_conditioning`'s excitation and condition number are describing a different
        matrix than the one actually solved."""
        parent, child = make_pair()
        n = len(parent)
        cross = np.einsum('nji,njk->ik', parent.rotations, child.rotations)
        expected = np.block([[n * np.eye(3), -cross], [-cross.T, n * np.eye(3)]])
        np.testing.assert_allclose(normal_equations(parent, child)['gram'], expected, atol=1e-6)

    def test_the_residual_can_be_read_back_out_without_the_samples(self):
        """RSS(x) = x^T G x - 2 x^T r + sum_bb. This is what makes cross-validation free."""
        parent, child = make_pair(wobble=2e-4, seed=7)
        equations = normal_equations(parent, child)
        offsets = solve_offsets(equations)
        _, _, residual = parent.get_joint_center(child)
        direct = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))) * 1000.0)
        self.assertAlmostEqual(residual_rms_from(equations, offsets), direct, places=6)

    def test_a_wrong_offset_scores_worse_than_the_fitted_one(self):
        parent, child = make_pair(wobble=2e-4, seed=8)
        equations = normal_equations(parent, child)
        offsets = solve_offsets(equations)
        self.assertLess(residual_rms_from(equations, offsets),
                        residual_rms_from(equations, offsets + 0.01))

    def test_combining_equations_equals_fitting_the_pooled_data(self):
        """The property the whole per-subject pass depends on: summing two trials' normal
        equations must give what fitting both trials together would."""
        parent_a, child_a = make_pair(n=1200, seed=11)
        parent_b, child_b = make_pair(n=900, seed=12)
        combined = combine_normal_equations([normal_equations(parent_a, child_a),
                                             normal_equations(parent_b, child_b)])
        pooled = WorldTrace(np.concatenate([parent_a.timestamps,
                                            parent_b.timestamps + 100.0]),
                            np.vstack([parent_a.positions, parent_b.positions]),
                            np.vstack([parent_a.rotations, parent_b.rotations]))
        pooled_child = WorldTrace(pooled.timestamps,
                                  np.vstack([child_a.positions, child_b.positions]),
                                  np.vstack([child_a.rotations, child_b.rotations]))
        np.testing.assert_allclose(solve_offsets(combined),
                                   solve_offsets(normal_equations(pooled, pooled_child)),
                                   atol=1e-9)
        self.assertEqual(combined['n'], len(parent_a) + len(parent_b))


class TestConditioning(unittest.TestCase):
    def test_a_rigid_pair_is_unobservable(self):
        """Two segments that never move relative to each other carry no information about where
        the joint between them is, and the excitation must say so rather than returning a
        confident number."""
        parent, _ = make_pair()
        rigid_child = WorldTrace(parent.timestamps, parent.positions - 0.3,
                                 parent.rotations.copy())
        conditioning = fit_conditioning(parent, rigid_child, np.ones(len(parent), dtype=bool))
        self.assertLess(conditioning['excitation'], 1e-6)

    def test_more_relative_motion_is_better_conditioned(self):
        quiet = fit_conditioning(*make_pair(seed=21), np.ones(2000, dtype=bool))
        lively_parent, lively_child = make_pair(seed=22)
        lively_child = WorldTrace(
            lively_child.timestamps, lively_child.positions,
            Rotation.from_rotvec(np.cumsum(
                np.random.default_rng(23).normal(0, 0.12, size=(2000, 3)), axis=0)).as_matrix())
        lively = fit_conditioning(lively_parent, lively_child, np.ones(2000, dtype=bool))
        self.assertGreater(lively['excitation'], quiet['excitation'])


class TestSeelProjection(unittest.TestCase):
    def test_A_times_o_is_the_rigid_body_projection_term(self):
        """A o == alpha x o + omega x (omega x o), by the vector triple product. The Jacobian in
        `seel_cost_parts` is written in terms of A, so if this identity fails the optimizer is
        descending the wrong surface."""
        parent, _ = make_pair(n=300)
        plate = make_plate(parent)
        # RAW config: no polyfit window, no accelerometer filter. This is testing the algebraic
        # identity behind A, and the default config filters both acc and A, which would compare a
        # filtered A against an unfiltered expectation.
        terms = model_terms(plate, np.ones(len(parent), dtype=bool), ('backward', None, None, None))
        offset = np.array([0.1, -0.2, 0.05])
        trace = plate.imu_trace
        omega = trace.gyro.astype(np.float64)
        alpha = trace._finite_difference_gyros('backward').astype(np.float64)
        expected = np.cross(alpha, offset) + np.cross(omega, np.cross(omega, offset))
        np.testing.assert_allclose(np.einsum('nij,j->ni', terms['A'], offset), expected,
                                   atol=1e-10)

    def test_the_sign_matches_project_acc(self):
        """a_p = a + A o, NOT a - A o. The two differ in which way the offset points and the
        mistake is silent — the optimizer still converges, to roughly the negated answer. Pinned
        against `IMUTrace.project_acc`, which is the convention the rest of the repo projects
        with."""
        parent, _ = make_pair(n=300)
        plate = make_plate(parent)
        offset = np.array([0.12, -0.05, 0.03])
        terms = model_terms(plate, np.ones(len(parent), dtype=bool), ('backward', None, None, None))
        np.testing.assert_allclose(terms['acc'] + np.einsum('nij,j->ni', terms['A'], offset),
                                   plate.imu_trace.project_acc(offset, 'backward').acc,
                                   atol=1e-9)

    def test_the_analytic_jacobian_matches_a_finite_difference(self):
        """The Jacobian is hand-derived, so it is checked against the thing it claims to be."""
        parent, child = make_pair(n=400)
        # Default config here on purpose: the Jacobian must be right for the terms actually
        # used, filtering included.
        terms_p = model_terms(make_plate(parent, seed=1), np.ones(400, dtype=bool))
        terms_c = model_terms(make_plate(child, seed=2), np.ones(400, dtype=bool))
        x = np.array([-0.2, 0.02, 0.05, 0.14, -0.01, 0.04])
        _, jacobian = cost_parts(x, terms_p, terms_c)
        for i in range(6):
            step = np.zeros(6)
            step[i] = 1e-7
            numeric = ((cost_parts(x + step, terms_p, terms_c)[0]
                        - cost_parts(x - step, terms_p, terms_c)[0]) / (2e-7))
            np.testing.assert_allclose(jacobian[:, i], numeric, atol=1e-4)


class TestInvariantPairs(unittest.TestCase):
    def test_only_anatomically_distinct_joints_are_paired(self):
        """R_Hip and R_Hip_H are one hip reached through two thigh sensors, so their separation
        is fit disagreement rather than a dimension of the subject."""
        for spec in (ALBORNO, IMOVE):
            for segment, entries in invariant_pairs(spec).items():
                for joint_a, _, joint_b, _ in entries:
                    self.assertNotEqual(canonical_joint(joint_a, spec),
                                        canonical_joint(joint_b, spec), f"{spec.name}/{segment}")

    def test_one_row_per_anatomical_pair_per_segment(self):
        """IMoVE's pelvis borders six joints that reduce to two hips; without deduping, the same
        pelvis width appears nine times under different placement combinations."""
        for spec in (ALBORNO, IMOVE):
            for segment, entries in invariant_pairs(spec).items():
                seen = [frozenset((canonical_joint(a, spec), canonical_joint(b, spec)))
                        for a, _, b, _ in entries]
                self.assertEqual(len(seen), len(set(seen)), f"{spec.name}/{segment}")

    def test_the_femur_and_tibia_are_found_on_both_datasets(self):
        for spec, segment in ((ALBORNO, 'Femur R'), (IMOVE, 'Thigh R Mid')):
            self.assertIn(segment, invariant_pairs(spec))

    def test_a_segment_bordering_one_joint_has_no_invariant(self):
        """The torso borders only the lumbar, so it carries one joint centre and no distance."""
        self.assertNotIn('Torso', invariant_pairs(ALBORNO))


class TestAmbiguity(unittest.TestCase):
    """The closure metric and the ambiguity radii that calibrate it.

    Closure is the objective the whole experiment now reports against, so the two identities it
    rests on are pinned here: that a displacement of one radius costs exactly the tolerance, and
    that closure does not care which body frame the plates were reconstructed in.
    """

    def _pair(self, seed=0, n=4000):
        """A parent/child pair rotating about a shared axis plus a little off-axis wobble."""
        rng = np.random.default_rng(seed)
        angle = np.linspace(0, 1.4, n)
        parent = Rotation.from_rotvec(np.column_stack(
            [angle * 0.05, angle * 0.02, np.zeros(n)]))
        child = Rotation.from_rotvec(np.column_stack(
            [angle, angle * 0.03, rng.normal(0, 0.01, n)]))
        r_parent, r_child = np.array([0.0, 0.02, 0.21]), np.array([0.01, 0.0, -0.19])
        centre = np.zeros((n, 3))
        positions_parent = centre - parent.apply(r_parent)
        positions_child = centre - child.apply(r_child)
        return (WorldTrace(np.arange(n) / 100.0, positions_parent, parent.as_matrix()),
                WorldTrace(np.arange(n) / 100.0, positions_child, child.as_matrix()))

    def test_one_radius_costs_exactly_the_tolerance(self):
        """The defining identity: r = tolerance*sqrt(n/lambda) inverts closure_cost."""
        from experiments.joint_center import ambiguity_radii, closure_cost, normal_equations
        parent, child = self._pair()
        equations = normal_equations(parent, child)
        radii, vectors = ambiguity_radii(equations, tolerance_mm=1.0)
        for index in (0, 3, 5):
            step = vectors[:, index] * radii[index] / 1000.0     # radii are mm, offsets metres
            self.assertAlmostEqual(closure_cost(equations, step), 1.0, places=6)

    def test_radii_are_returned_worst_first(self):
        """eigh is ascending in lambda and the radius goes as 1/sqrt(lambda), so no extra flip."""
        from experiments.joint_center import ambiguity_radii, normal_equations
        radii, _ = ambiguity_radii(normal_equations(*self._pair()))
        self.assertTrue(np.all(np.diff(radii) <= 1e-9), f"not descending: {radii}")

    def test_the_hinge_axis_is_the_ambiguous_direction(self):
        """A near-pure hinge should leave a common slide along its axis nearly free."""
        from experiments.joint_center import ambiguity_radii, normal_equations
        parent, child = self._pair()
        radii, vectors = ambiguity_radii(normal_equations(parent, child))
        # The worst mode should slide BOTH offsets together (a common shift, not a differential)
        # and do it along x, which is the axis both segments rotate about.
        worst = vectors[:, 0]
        self.assertGreater(radii[0] / radii[-1], 3.0)
        self.assertGreater(abs(worst[0]) + abs(worst[3]), 0.8)
        self.assertGreater(worst[0] * worst[3], 0.0)     # same sign = common, not differential

    def test_closure_does_not_care_about_the_body_frame(self):
        """Re-basing both plates and their offsets together leaves closure untouched.

        This is what lets a marker centre computed in a raw template frame be compared against
        rows computed in the built frame, with no transport step.
        """
        from experiments.joint_center import (normal_equations, residual_rms_from, solve_offsets,
                                              ambiguity_radii)
        parent, child = self._pair()
        plain = normal_equations(parent, child)
        basis = Rotation.from_rotvec([0.4, -0.7, 0.2]).as_matrix()
        rebased = normal_equations(
            WorldTrace(parent.timestamps, parent.positions, parent.rotations @ basis),
            WorldTrace(child.timestamps, child.positions, child.rotations @ basis))
        self.assertAlmostEqual(residual_rms_from(plain, solve_offsets(plain)),
                               residual_rms_from(rebased, solve_offsets(rebased)), places=9)
        np.testing.assert_allclose(ambiguity_radii(plain)[0], ambiguity_radii(rebased)[0],
                                   rtol=1e-9)



class TestStoredTrialKeys(unittest.TestCase):
    """Enumerating the RESULTS tree, where a trial key may contain slashes.

    The biplane source names its trials by session and block — 'Test1/A/RSDrop1' — so their tables
    sit three levels below the subject directory rather than one. The one-level `iterdir()` this
    replaced found none of them, and the failure was silent rather than loud: `closure_by_source`
    and `ambiguity_table` returned empty frames, `main()` declines to write an empty frame, and the
    report then carried no headline for those datasets with nothing saying why.

    Both cases are asserted, because a recursive walk that returns the WRONG KEY for a flat dataset
    would break every other dataset while fixing this one.
    """

    def _tree(self, root, keys):
        for subject, trial in keys:
            path = root / subject / trial / 'normal_equations.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'joint': ['L_Knee']}).to_parquet(path, engine='pyarrow', index=False)

    def test_nested_trial_keys_are_found_and_round_trip(self):
        import tempfile
        from unittest import mock
        import experiments.joint_center as jc
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            planted = [('01', 'Test1/A/RSDrop1'), ('01', 'Test1/C/RrunStance2'),
                       ('12', 'Test1/B/LDDrop2')]
            self._tree(root, planted)
            with mock.patch.object(jc, 'dataset_dir', return_value=root):
                found = jc.stored_trial_keys('whatever')
            self.assertEqual(found, sorted(planted))
            # The key has to be the one `trial_table_path` joins back onto the subject directory,
            # or every consumer reads a path that does not exist. Asserted INSIDE the temporary
            # directory's scope, since it is gone by the time the block exits.
            for subject, trial in found:
                self.assertTrue((root / subject / trial / 'normal_equations.parquet').exists())

    def test_flat_trial_keys_are_unchanged(self):
        import tempfile
        from unittest import mock
        import experiments.joint_center as jc
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            planted = [('01', 'walking'), ('01', 'complexTasks'), ('02', 'walking')]
            self._tree(root, planted)
            with mock.patch.object(jc, 'dataset_dir', return_value=root):
                found = jc.stored_trial_keys('whatever')
        self.assertEqual(found, sorted(planted))

    def test_missing_tree_is_empty_not_an_error(self):
        import tempfile
        from unittest import mock
        import experiments.joint_center as jc
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(jc, 'dataset_dir',
                                   return_value=pathlib.Path(tmp) / 'absent'):
                self.assertEqual(jc.stored_trial_keys('whatever'), [])

class TestShiftDecomposition(unittest.TestCase):
    """Separating a re-mount from independent fit wander.

    Section 8 used to infer a re-mount from "offsets moved, segment length held". That inference
    is invalid, and these tests pin why: length depends only on the DIFFERENTIAL of the two
    centres' shifts, and is second-order blind to the transverse part of even that. So both a
    genuine re-mount AND independent transverse wander preserve the length, and only the
    common/differential split tells them apart.
    """

    def _fits(self, shifts):
        """Two trials of one subject, with trial 2's offsets displaced by `shifts`.

        The thigh borders R_Hip (as child) and R_Knee (as parent), so those two offsets are the
        femur's two joint centres and the distance between them is femur length.
        """
        hip, knee = np.array([0.0, 0.0, 0.22]), np.array([0.0, 0.0, -0.22])
        rows = []
        for trial, (d_hip, d_knee) in shifts.items():
            for joint, parent, child in (('R_Hip', np.zeros(3), hip + d_hip),
                                         ('R_Knee', knee + d_knee, np.zeros(3))):
                rows.append({'subject': '01', 'trial': trial, 'joint': joint, 'converged': True,
                             **dict(zip(('parent_x', 'parent_y', 'parent_z'), parent)),
                             **dict(zip(('child_x', 'child_y', 'child_z'), child))})
        return pd.DataFrame(rows)

    def _femur(self, shifts):
        from experiments.joint_center import shift_decomposition
        out = shift_decomposition(self._fits(shifts), ALBORNO)
        return out[out['segment'] == 'Femur R'].iloc[0]

    # Everything below is a deviation from the subject's MEDIAN offset, matching
    # `offset_stability`'s convention, so with two trials each reads half the between-trial move.

    def test_a_re_mount_is_all_common_and_no_differential(self):
        """Both centres move by ONE vector, because the sensor moved and the joints did not."""
        move = np.array([0.008, -0.011, 0.004])       # 14.2 mm, so 7.1 mm either side of median
        row = self._femur({'a': (np.zeros(3), np.zeros(3)), 'b': (move, move)})
        self.assertAlmostEqual(row['common_mm'], 7.09, places=1)
        self.assertLess(row['differential_mm'], 1e-6)

    def test_independent_transverse_wander_holds_the_length_just_as_well(self):
        """The case the old inference misread. Two centres move independently and ACROSS the
        segment, so the length barely notices — but nothing common has happened."""
        row = self._femur({'a': (np.zeros(3), np.zeros(3)),
                           'b': (np.array([0.015, 0.0, 0.0]), np.array([-0.015, 0.0, 0.0]))})
        self.assertAlmostEqual(row['differential_mm'], 15.0, places=1)   # entirely transverse
        self.assertLess(row['common_mm'], 1e-6)                          # nothing shared
        # Almost nothing along the femur -- not exactly nothing, because the axis is taken from
        # the MEDIAN offsets and the wander itself tilts that axis by atan(15/440) = 1.95 deg.
        self.assertLess(row['diff_along_mm'], 0.05 * row['diff_perp_mm'])
        # ...and what that costs the length is second order: |perp|^2 / 2L on a 440 mm femur.
        self.assertLess(row['perp_costs_mm'], 0.3)

    def test_length_is_blind_to_the_common_part_exactly(self):
        """Adding a pure common shift changes no length, at any magnitude — which is why a
        preserved length cannot be read as evidence that the shift WAS common."""
        from experiments.joint_center import segment_lengths
        wander = (np.array([0.015, 0.0, 0.0]), np.array([-0.015, 0.0, 0.0]))
        huge = np.array([0.05, -0.07, 0.03])
        plain = self._fits({'a': (np.zeros(3), np.zeros(3)), 'b': wander})
        shifted = self._fits({'a': (huge, huge), 'b': (wander[0] + huge, wander[1] + huge)})
        femur = lambda f: (segment_lengths(f, ALBORNO)
                           .set_index(['trial', 'pair'])['length_mm'].loc[:, 'R_Hip-R_Knee'])
        np.testing.assert_allclose(femur(plain).to_numpy(), femur(shifted).to_numpy(), atol=1e-9)


def write_trc(path, labels, columns, rate=100.0):
    """A minimal .trc: tab-separated, three columns per marker, zero-filled gaps."""
    n = len(next(iter(columns.values())))
    header = [f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
              "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\t"
              "OrigDataStartFrame\tOrigNumFrames",
              f"{rate}\t{rate}\t{n}\t{len(labels)}\tmm\t{rate}\t1\t{n}",
              "Frame#\tTime\t" + "\t\t\t".join(labels) + "\t\t\t",
              "\t\t" + "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(labels))), ""]
    lines = list(header)
    for frame in range(n):
        values = [str(frame + 1), f"{frame / rate:.5f}"]
        for label in labels:
            values += [f"{v:.5f}" for v in columns[label][frame]]
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding='utf-8')


class TestSharedTemplate(unittest.TestCase):
    """`read_alborno_plates` must be able to put two captures in ONE body frame.

    A plate's template frame is arbitrary, so a landmark reduced against one and an offset fitted
    against another are the same vector in two bases. |r| survives that and the VECTOR does not,
    which is the trap: the scalar column keeps looking right while the two columns the comparison
    exists for go wrong silently.
    """

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Planar, like the real plates, which is the case the sign ambiguity bites.
        self.plate = np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0],
                               [0.0, 0.05, 0.0], [0.08, 0.05, 0.0]])

    def _capture(self, name, rotations, translation):
        """One .trc holding a single plate driven through `rotations`.

        `translation` must keep every marker off the lab origin: this format zero-fills gaps, so
        a marker sitting exactly at (0, 0, 0) is indistinguishable from an untracked one.
        """
        from pathlib import Path
        path = Path(self.tmp.name) / name
        corners = np.einsum('nij,mj->nmi', rotations, self.plate) + translation[:, None, :]
        labels = [f"seg_{s}" for s in 'odxy']
        write_trc(path, labels, {label: corners[:, i] * 1000.0
                                 for i, label in enumerate(labels)})
        return path

    def _poses(self, path, templates=None):
        from src.toolchest.building.landmarks import read_alborno_plates
        return read_alborno_plates(path, {'seg': 'seg'}, templates=templates)

    def test_a_passed_template_is_used_verbatim_rather_than_re_estimated(self):
        rng = np.random.default_rng(0)
        rotations = Rotation.from_rotvec(rng.normal(scale=0.4, size=(200, 3))).as_matrix()
        path = self._capture('a.trc', rotations, rng.normal(loc=1.0, scale=0.3, size=(200, 3)))
        _, estimated = self._poses(path)
        # A deliberately re-based template: the caller's basis must survive the round trip.
        turn = Rotation.from_rotvec([0.3, -0.2, 0.7]).as_matrix()
        _, used = self._poses(path, templates={'seg': estimated['seg'] @ turn})
        np.testing.assert_allclose(used['seg'], estimated['seg'] @ turn, atol=1e-12)

    def test_the_template_sets_the_basis_that_the_vector_columns_live_in(self):
        """The silent failure the shared template exists to prevent."""
        rng = np.random.default_rng(1)
        rotations = Rotation.from_rotvec(rng.normal(scale=0.4, size=(200, 3))).as_matrix()
        path = self._capture('b.trc', rotations, rng.normal(loc=1.0, scale=0.3, size=(200, 3)))
        _, estimated = self._poses(path)
        turn = Rotation.from_rotvec([0.0, 0.0, 0.9]).as_matrix()  # 51.6 deg about the plate normal
        here, _ = self._poses(path, templates=estimated)
        there, _ = self._poses(path, templates={'seg': estimated['seg'] @ turn})

        point = np.tile([0.2, 0.1, 0.3], (200, 1))

        def local(poses):
            positions, orientations, _ = poses['seg']
            return np.einsum('nji,nj->ni', orientations, point - positions)

        a, b = local(here), local(there)
        # |r| is blind to the basis -- which is why a basis mistake survives a scalar check...
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), atol=1e-6)
        # ...while the vector, the column the comparison reports, is a long way off.
        self.assertGreater(np.linalg.norm(a[0] - b[0]), 0.05)
        # And the two differ by exactly that CONSTANT rotation, which is what makes sharing one
        # template sufficient: fix the template and every downstream vector lines up.
        np.testing.assert_allclose(b, a @ turn, atol=1e-6)

    def test_the_sensor_name_is_decoupled_from_the_file_name(self):
        """The standing capture calls the right thigh 'R.Femur_IMU_O' and a trial calls it
        'femur_r_imu_o'; the poses must come back keyed the way the joint table names it."""
        from src.toolchest.building.landmarks import read_alborno_plates
        rng = np.random.default_rng(2)
        rotations = Rotation.from_rotvec(rng.normal(scale=0.3, size=(120, 3))).as_matrix()
        path = self._capture('c.trc', rotations, np.ones((120, 3)))
        poses, templates = read_alborno_plates(path, {'femur_r_imu': 'seg'})
        self.assertEqual(set(poses), {'femur_r_imu'})
        self.assertEqual(set(templates), {'femur_r_imu'})


class TestStaticLandmarkSource(unittest.TestCase):
    """What the standing capture is allowed to claim, and where it is filed."""

    def test_alborno_reaches_every_joint_only_through_the_standing_capture(self):
        """The trial files have no medial markers, so knee and ankle exist only in the static
        set. If this ever inverts, the lateral markers are being read as centres."""
        from src.toolchest.building.landmarks import ALBORNO as SPEC
        self.assertEqual(set(SPEC.centres), {'R_Hip', 'L_Hip'})
        self.assertEqual(set(SPEC.static_centres),
                         {'R_Hip', 'L_Hip', 'R_Knee', 'L_Knee', 'R_Ankle', 'L_Ankle'})
        for joint in ('R_Knee', 'L_Knee', 'R_Ankle', 'L_Ankle'):
            self.assertEqual(len(SPEC.static_centres[joint]), 2,
                             f"{joint} needs a medial AND a lateral marker to have a centre")

    def test_imove_has_no_standing_capture(self):
        from src.toolchest.building.landmarks import IMOVE as SPEC
        self.assertEqual(SPEC.static_centres, {})
        self.assertIsNone(SPEC.read_static)

    def test_the_standing_capture_is_filed_out_of_the_trial_glob(self):
        """A second .trc in the trial folder would trip alborno.load_trial's one-trial guard,
        change the *.trc cache inventory, and outrank walking.trc in the sorted glob."""
        import paths
        from src.toolchest.building.landmarks import alborno_static_trc
        static = alborno_static_trc('01')
        trial_dir = paths.raw_trial_dir('01', 'walking')
        self.assertEqual(static.name, 'static_walking.trc')
        self.assertNotIn(static, list(trial_dir.glob('*.trc')))
        self.assertEqual(static.parent.parent, trial_dir)

    def test_the_frame_floor_admits_the_shortest_standing_capture(self):
        """A static reference is a median of a constant, not a fit, so the experiment's
        500-frame conditioning floor does not apply — five captures run 200-394 frames."""
        from src.toolchest.building.landmarks import MIN_STATIC_FRAMES
        from experiments.joint_center import MIN_FIT_FRAMES
        self.assertLess(MIN_STATIC_FRAMES, 200)
        self.assertLess(MIN_STATIC_FRAMES, MIN_FIT_FRAMES)


if __name__ == '__main__':
    unittest.main()
