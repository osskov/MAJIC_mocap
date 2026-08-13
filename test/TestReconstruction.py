"""Marker positions in, plate pose out: src/toolchest/building/reconstruction.py.

Split out of TestWorldTrace, which had grown into four unrelated subjects. None of this is
about the WorldTrace container -- it is the physics of turning a cloud of labelled markers
into a rigid-body trajectory, and the two failure modes that has: a plate whose SHAPE is
wrong at an instant (a collision, a dropout, a mislabel) and a plate whose MOTION is wrong
between instants (a distance-preserving relabeling, which no shape check can see).
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.toolchest.WorldTrace import WorldTrace
from src.toolchest.building.reconstruction import (
    _HALF_TURN_CANDIDATES, _reconstruct_from_markers, estimate_plate_template,
    fit_plate_to_template, reconstruct_plate, template_correction_candidates,
    template_symmetries, world_trace_from_markers)
from test.fixtures import PLATE_HALF_HEIGHT, PLATE_HALF_WIDTH


class TestTemplatePlateFit(unittest.TestCase):
    """WorldTrace.from_markers / fit_plate_to_template — the shape-agnostic reconstruction.

    Written against IMoVE's real plate, which is an irregular planar quadrilateral, not a
    rectangle: its six inter-marker distances are 120.4 / 104.4 / 103.4 / 84.6 / 62.9 /
    59.6 mm. That is exactly the case _reconstruct_from_markers cannot handle.
    """

    # Recovered by MDS on the median inter-marker distances, in metres.
    TEMPLATE = np.array([[-60.1, 23.6, 0.0], [43.8, 34.6, 0.0],
                         [50.2, -24.6, 0.0], [-33.9, -33.6, 0.0]]) / 1000.0

    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = 400
        self.timestamps = np.arange(self.n) / 100.0
        self.template = self.TEMPLATE - self.TEMPLATE.mean(axis=0)
        self.rotations = Rotation.from_rotvec(
            np.cumsum(rng.normal(0, 0.02, (self.n, 3)), axis=0)).as_matrix()
        self.positions = np.cumsum(rng.normal(0, 0.002, (self.n, 3)), axis=0)
        self.markers = (np.einsum('nij,kj->nki', self.rotations, self.template)
                        + self.positions[:, None, :])

    def _errors(self, positions, rotations, valid):
        relative = np.einsum('nji,njk->nik', self.rotations, rotations)
        angle = np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=1))
        offset = np.linalg.norm(positions - self.positions, axis=1)
        return angle[valid].max(), offset[valid].max()

    def test_exact_on_clean_markers(self):
        pos, rot, valid, report = fit_plate_to_template(
            self.markers, self.timestamps, template=self.template)
        self.assertTrue(valid.all())
        angle, offset = self._errors(pos, rot, valid)
        self.assertLess(angle, 1e-6)
        self.assertLess(offset, 1e-9)
        self.assertEqual(report['fault_counts_per_marker'], [0, 0, 0, 0])

    def test_template_can_be_estimated_from_the_data(self):
        """No template supplied: the plate's own median geometry has to supply one."""
        estimated = estimate_plate_template(self.markers)
        actual = np.sort([np.linalg.norm(estimated[i] - estimated[j])
                          for i in range(4) for j in range(i + 1, 4)])
        expected = np.sort([np.linalg.norm(self.template[i] - self.template[j])
                            for i in range(4) for j in range(i + 1, 4)])
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_reconstruction_is_rigid(self):
        """Every reconstructed frame must be a rotation, never a reflection.

        A mirrored fit silently flips the segment and every joint angle taken from it.
        """
        _, rot, _, _ = fit_plate_to_template(self.markers, self.timestamps,
                                             template=self.template)
        np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.einsum('nij,nkj->nik', rot, rot),
                                   np.broadcast_to(np.eye(3), rot.shape), atol=1e-9)

    def test_leave_one_out_isolates_a_displaced_marker(self):
        """One marker off, the other three consistent — the common fault by far.

        This is the case _compute_case handles with four hand-written branches; here it
        falls out of dropping each marker in turn, with no assumption about plate shape.
        """
        markers = self.markers.copy()
        markers[100:130, 2] += np.array([0.04, 0.0, 0.0])

        pos, rot, valid, report = fit_plate_to_template(markers, self.timestamps,
                                                        template=self.template)
        self.assertTrue(valid.all(), "a single displaced marker must not lose the frame")
        self.assertEqual(report['fault_counts_per_marker'], [0, 0, 30, 0])
        angle, offset = self._errors(pos, rot, valid)
        self.assertLess(angle, 1e-6, "the surviving three markers give the pose exactly")

    def test_gaps_are_tolerated_while_three_markers_remain(self):
        markers = self.markers.copy()
        markers[200:260, 1] = np.nan      # NaN gap
        markers[300:320, 3] = 0.0         # exactly-zero gap, how some exporters mark one
        pos, rot, valid, _ = fit_plate_to_template(markers, self.timestamps,
                                                   template=self.template)
        self.assertTrue(valid.all())
        angle, _ = self._errors(pos, rot, valid)
        self.assertLess(angle, 1e-6)

    def test_two_missing_markers_is_invalid_not_guessed(self):
        """Two markers leave the pose underdetermined. It must be refused, not invented."""
        markers = self.markers.copy()
        markers[150:170, 0] = np.nan
        markers[150:170, 1] = np.nan
        _, _, valid, report = fit_plate_to_template(markers, self.timestamps,
                                                    template=self.template)
        self.assertFalse(valid[150:170].any())
        self.assertTrue(valid[:150].all())
        self.assertEqual(report['n_invalid'], 20)

    def test_invalid_frames_are_still_on_the_grid(self):
        """Invalid frames keep interpolated poses, because the uniform time grid is what
        every finite-difference angular velocity in this repo assumes."""
        markers = self.markers.copy()
        markers[150:170, :2] = np.nan
        pos, rot, valid, _ = fit_plate_to_template(markers, self.timestamps,
                                                   template=self.template)
        self.assertEqual(len(pos), self.n)
        self.assertTrue(np.isfinite(pos).all(), "no gaps left in the array")
        self.assertTrue(np.isfinite(rot).all())

    def test_a_permanently_absent_marker_degrades_to_three(self):
        markers = self.markers.copy()
        markers[:, 1] = np.nan
        pos, rot, valid, _ = fit_plate_to_template(markers, self.timestamps,
                                                   template=self.template)
        self.assertTrue(valid.all())
        angle, offset = self._errors(pos, rot, valid)
        self.assertLess(angle, 1e-6)
        self.assertLess(offset, 1e-9, "the origin must not move when a marker is dropped")

    def test_too_few_markers_fails_legibly(self):
        """s2's treadmill trials lose whole marker groups. A reader has to be able to tell
        'untracked in this trial' from 'failed to reconstruct'."""
        markers = self.markers.copy()
        markers[:, :2] = np.nan
        with self.assertRaises(ValueError) as ctx:
            fit_plate_to_template(markers, self.timestamps, name='THIGH_L')
        self.assertIn('THIGH_L', str(ctx.exception))

    def test_from_markers_returns_a_worldtrace_carrying_the_mask(self):
        markers = self.markers.copy()
        markers[150:170, :2] = np.nan
        trace = world_trace_from_markers(markers, self.timestamps, template=self.template)
        self.assertIsInstance(trace, WorldTrace)
        self.assertEqual(len(trace), self.n)
        self.assertFalse(trace.valid[150:170].any())
        self.assertTrue(trace.valid[:150].all())


class TestTemplateSymmetries(unittest.TestCase):
    """Whether a plate's shape makes a marker swap invisible — i.e. whether the template
    fit is sufficient on its own or needs the un-flip pass on top."""

    ALBORNO = np.array([[PLATE_HALF_WIDTH, PLATE_HALF_HEIGHT, 0.0],
                        [-PLATE_HALF_WIDTH, -PLATE_HALF_HEIGHT, 0.0],
                        [PLATE_HALF_WIDTH, -PLATE_HALF_HEIGHT, 0.0],
                        [-PLATE_HALF_WIDTH, PLATE_HALF_HEIGHT, 0.0]])
    IMOVE = np.array([[-60.1, 23.6, 0.0], [43.8, 34.6, 0.0],
                      [50.2, -24.6, 0.0], [-33.9, -33.6, 0.0]]) / 1000.0

    @staticmethod
    def _invisible(template, tol_mm=1e-6):
        identity = tuple(range(len(template)))
        return [p for p, margin in template_symmetries(template)
                if margin < tol_mm and p != identity]

    def test_a_rectangle_hides_three_relabelings(self):
        """Both diagonals equal, so the three coordinate half turns are distance-preserving.

        This is why _unflip_swapped_blocks exists and why no residual-based method can
        replace it on these plates: measured on Subject06/complexTasks/femur_l, the template
        fit reproduces the swapped block at 0.20 mm median residual — detecting nothing —
        while sitting exactly 180 degrees from the un-flipped truth.
        """
        self.assertEqual(len(self._invisible(self.ALBORNO)), 3)

    def test_the_imove_cluster_hides_nothing(self):
        """Six distinct distances, so every relabeling is visible in the geometry."""
        self.assertEqual(self._invisible(self.IMOVE), [])
        best = template_symmetries(self.IMOVE)[0][1]
        self.assertGreater(best, 5.0,
                           "the best non-identity margin must clear the fit residual "
                           "(measured at 0.1-0.6 mm on this plate) by a wide factor")

    def test_identity_is_always_last_and_free(self):
        for template in (self.ALBORNO, self.IMOVE):
            symmetries = template_symmetries(template)
            self.assertEqual(len(symmetries), 24)
            self.assertEqual(symmetries[-1][0], (0, 1, 2, 3))
            self.assertAlmostEqual(symmetries[-1][1], 0.0)


class TestMergedReconstruction(unittest.TestCase):
    """`reconstruct_plate` — template fit for shape faults, speed repair for swaps.

    The point of the merge is that the two detectors have disjoint blind spots. A marker
    collision has the right motion and the wrong shape; a distance-preserving relabeling
    has the right shape and the wrong motion. Each test below pins one of those.
    """

    RECTANGLE = np.array([[PLATE_HALF_WIDTH, PLATE_HALF_HEIGHT, 0.0],
                          [-PLATE_HALF_WIDTH, -PLATE_HALF_HEIGHT, 0.0],
                          [PLATE_HALF_WIDTH, -PLATE_HALF_HEIGHT, 0.0],
                          [-PLATE_HALF_WIDTH, PLATE_HALF_HEIGHT, 0.0]])
    IRREGULAR = np.array([[-60.1, 23.6, 0.0], [43.8, 34.6, 0.0],
                          [50.2, -24.6, 0.0], [-33.9, -33.6, 0.0]]) / 1000.0

    def _markers(self, template, n=600, seed=1):
        rng = np.random.default_rng(seed)
        timestamps = np.arange(n) / 100.0
        rotations = Rotation.from_rotvec(
            np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0)).as_matrix()
        positions = np.cumsum(rng.normal(0, 0.001, (n, 3)), axis=0)
        centred = template - template.mean(axis=0)
        markers = np.einsum('nij,kj->nki', rotations, centred) + positions[:, None, :]
        return markers, timestamps, rotations

    def test_candidates_are_derived_not_hardcoded(self):
        """The rectangle's three coordinate half turns fall out of its geometry."""
        candidates = template_correction_candidates(self.RECTANGLE)
        self.assertEqual(len(candidates), 4, "identity plus three half turns")
        for derived in candidates.values():
            self.assertTrue(
                any(np.allclose(derived, known, atol=1e-9)
                    for known in _HALF_TURN_CANDIDATES.values()),
                f"derived correction {derived} is not one of the known half turns")

    def test_an_irregular_plate_admits_no_correction(self):
        """Nothing to snap to, so nothing can be wrongly 'corrected'."""
        candidates = template_correction_candidates(self.IRREGULAR)
        self.assertEqual(list(candidates), ['identity'])
        np.testing.assert_allclose(candidates['identity'], np.eye(3), atol=1e-9)

    def test_a_sustained_swap_on_a_rectangle_is_un_flipped(self):
        """The case the template fit alone cannot see.

        Relabeling o<->d and x<->y on a rectangle preserves all six distances, so the fit
        reproduces the swapped pose at a clean residual. Only the pose discontinuity at the
        block boundary reveals it.
        """
        markers, timestamps, truth = self._markers(self.RECTANGLE)
        swapped = markers.copy()
        swapped[200:400] = markers[200:400][:, [1, 0, 3, 2], :]

        # Fit alone: blind to it, and says so with a clean residual.
        _, rot_fit, _, report_fit = fit_plate_to_template(swapped, timestamps,
                                                          template=self.RECTANGLE)
        self.assertLess(report_fit['residual_median_mm'], 0.01,
                        "a distance-preserving swap leaves no residual to detect")

        # Merged: catches it.
        _, rot_merged, valid, report = reconstruct_plate(swapped, timestamps,
                                                          template=self.RECTANGLE)
        self.assertEqual(report['symmetries'], 3)
        self.assertGreater(report['flipped'], 0)

        # A correct reconstruction holds a CONSTANT offset from truth in the plate frame,
        # whatever that offset happens to be. A swapped block sits half a turn away from
        # the rest, so counting frames far from the first one counts the un-recovered ones.
        def frames_half_a_turn_out(rot, keep=None):
            relative = np.einsum('nji,njk->nik', truth, rot)
            if keep is not None:
                relative = relative[keep]
            delta = np.einsum('ji,njk->nik', relative[0], relative)
            angles = np.degrees(np.linalg.norm(
                Rotation.from_matrix(delta).as_rotvec(), axis=1))
            return int((angles > 90).sum())

        self.assertGreater(frames_half_a_turn_out(rot_fit), 100,
                           "fixture check: the fit really is half a turn out on the block")
        self.assertEqual(frames_half_a_turn_out(rot_merged, keep=valid), 0,
                         "the merged path must leave no frame a half turn out")

    def test_a_shape_fault_is_caught_where_the_speed_detector_is_blind(self):
        """A marker collision held over a block: smooth motion, wrong shape.

        The angular-speed detector only ever sees the two edges of such a block. The
        residual sees the whole of it, which is the gap the merge closes.
        """
        markers, timestamps, _ = self._markers(self.RECTANGLE)
        collided = markers.copy()
        # A real collision, per the marker-geometry survey of this dataset: the merged
        # detection does NOT land on either marker's true position, it lands near their
        # midpoint, roughly 23 mm from either. Both labels then report that point.
        #
        # Assigning one marker's true position to both labels would be a different and much
        # easier fault: the other three markers stay correct, so leave-one-out repairs the
        # frame exactly and it is right to keep it.
        midpoint = 0.5 * (collided[200:300, 0] + collided[200:300, 1])
        collided[200:300, 0] = midpoint
        collided[200:300, 1] = midpoint

        _, _, valid, report = reconstruct_plate(collided, timestamps, template=self.RECTANGLE)

        n_flagged = int((~valid[200:300]).sum())
        self.assertGreater(n_flagged, 50,
                           "two bad markers leave no clean 3-subset, so the block cannot be "
                           "repaired and must be flagged — the residual sees all of it, "
                           "while the speed detector would see only its two edges")
        self.assertTrue(valid[:190].all(), "clean frames elsewhere stay valid")

    def test_report_carries_both_detectors(self):
        markers, timestamps, _ = self._markers(self.RECTANGLE)
        _, _, _, report = reconstruct_plate(markers, timestamps, template=self.RECTANGLE)
        for key in ('residual_median_mm', 'n_repaired_by_dropping_a_marker',
                    'flipped', 'interpolated', 'unresolved', 'symmetries'):
            self.assertIn(key, report)

    def test_clean_data_is_left_alone(self):
        markers, timestamps, truth = self._markers(self.IRREGULAR)
        pos, rot, valid, report = reconstruct_plate(markers, timestamps,
                                                    template=self.IRREGULAR)
        self.assertTrue(valid.all())
        self.assertEqual(report['flipped'], 0)
        self.assertEqual(report['interpolated'], 0)
        self.assertEqual(report['unresolved'], 0)
        relative = np.einsum('nij,njk->nik', truth.transpose(0, 2, 1), rot)
        delta = np.einsum('ij,nkj->nik', relative[0], relative.transpose(0, 2, 1))
        self.assertLess(np.degrees(np.linalg.norm(
            Rotation.from_matrix(delta).as_rotvec(), axis=1)).max(), 1e-6)


if __name__ == '__main__':
    unittest.main()
