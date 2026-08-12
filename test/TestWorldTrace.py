import os
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.WorldTrace import (WorldTrace, _reconstruct_from_markers,
                                      estimate_plate_template, fit_plate_to_template,
                                      repair_reconstruction_glitches)

# Marker layout of a plate, in the plate's own frame. This has to agree with the
# convention _reconstruct_from_markers reconstructs against: +x runs d->x and y->o,
# +y runs x->o and d->y, and the four markers are symmetric about the plate origin.
PLATE_HALF_WIDTH = 0.04
PLATE_HALF_HEIGHT = 0.03
MARKER_O_LOCAL = np.array([PLATE_HALF_WIDTH, PLATE_HALF_HEIGHT, 0.0])
MARKER_D_LOCAL = np.array([-PLATE_HALF_WIDTH, -PLATE_HALF_HEIGHT, 0.0])
MARKER_X_LOCAL = np.array([PLATE_HALF_WIDTH, -PLATE_HALF_HEIGHT, 0.0])
MARKER_Y_LOCAL = np.array([-PLATE_HALF_WIDTH, PLATE_HALF_HEIGHT, 0.0])


def _markers_from_poses(positions, rotations):
    """Places the four plate markers in the world, given the plate's pose over time."""
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    return tuple(
        np.einsum('nij,j->ni', rotations, local) + positions
        for local in (MARKER_O_LOCAL, MARKER_D_LOCAL, MARKER_X_LOCAL, MARKER_Y_LOCAL)
    )


class TestWorldTrace(unittest.TestCase):
    def setUp(self):
        self.timestamps = np.array([0, 1, 2, 3, 4])
        self.positions = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3]),
                          np.array([4, 4, 4])]
        self.rotations = [np.eye(3) for _ in range(5)]
        self.world_trace = WorldTrace(self.timestamps, self.positions, self.rotations)

    def test_initialization(self):
        np.testing.assert_array_equal(self.world_trace.timestamps, self.timestamps)
        for pos1, pos2 in zip(self.world_trace.positions, self.positions):
            np.testing.assert_array_equal(pos1, pos2)
        for rot1, rot2 in zip(self.world_trace.rotations, self.rotations):
            np.testing.assert_array_equal(rot1, rot2)

    def test_length(self):
        self.assertEqual(len(self.world_trace), 5)

    def test_getitem_slice(self):
        sliced_world_trace = self.world_trace[1:4]
        expected_timestamps = np.array([1, 2, 3])
        expected_positions = [np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])]
        expected_rotations = [np.eye(3) for _ in range(3)]

        np.testing.assert_array_equal(sliced_world_trace.timestamps, expected_timestamps)
        for pos1, pos2 in zip(sliced_world_trace.positions, expected_positions):
            np.testing.assert_array_equal(pos1, pos2)
        for rot1, rot2 in zip(sliced_world_trace.rotations, expected_rotations):
            np.testing.assert_array_equal(rot1, rot2)

    def test_getitem_index(self):
        single_item_world_trace = self.world_trace[2]
        expected_timestamps = np.array([2])
        expected_positions = [np.array([2, 2, 2])]
        expected_rotations = [np.eye(3)]

        np.testing.assert_array_equal(single_item_world_trace.timestamps, expected_timestamps)
        for pos1, pos2 in zip(single_item_world_trace.positions, expected_positions):
            np.testing.assert_array_equal(pos1, pos2)
        for rot1, rot2 in zip(single_item_world_trace.rotations, expected_rotations):
            np.testing.assert_array_equal(rot1, rot2)

    def test_eq(self):
        other_world_trace = WorldTrace(self.timestamps, self.positions, self.rotations)
        self.assertEqual(self.world_trace, other_world_trace)

    def test_re_zero(self):
        timestamps = self.timestamps + 5.5
        other_world_trace = WorldTrace(timestamps, self.positions, self.rotations)
        zeroed_world_trace = other_world_trace.re_zero_timestamps()
        np.testing.assert_array_equal(zeroed_world_trace.timestamps, self.timestamps)

    def test_not_eq(self):
        other_positions = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3]),
                           np.array([5, 5, 5])]
        other_world_trace = WorldTrace(self.timestamps, other_positions, self.rotations)
        self.assertNotEqual(self.world_trace, other_world_trace)

    def test_allclose(self):
        small_offset = 1e-7
        close_positions = [pos + small_offset for pos in self.positions]
        close_rotations = [rot + small_offset * np.eye(3) for rot in self.rotations]
        close_world_trace = WorldTrace(self.timestamps, close_positions, close_rotations)
        self.assertTrue(self.world_trace.allclose(close_world_trace))

    def test_not_allclose(self):
        large_offset = 1e-3
        far_positions = [pos + large_offset for pos in self.positions]
        far_rotations = [rot + large_offset * np.eye(3) for rot in self.rotations]
        far_world_trace = WorldTrace(self.timestamps, far_positions, far_rotations)
        self.assertFalse(self.world_trace.allclose(far_world_trace))

    def test_subtraction(self):
        rotations = Rotation.from_rotvec(np.linspace(0, 1, 5)[:, None] * np.array([0., 0., 1.])).as_matrix()
        trace_a = WorldTrace(self.timestamps, self.positions, rotations)
        trace_b = WorldTrace(self.timestamps, self.positions, self.rotations)

        difference = trace_a - trace_b
        np.testing.assert_array_almost_equal(difference.positions, np.zeros((5, 3)))
        # R_a @ R_b^T, and R_b is the identity here, so the relative rotation is R_a itself.
        np.testing.assert_array_almost_equal(difference.rotations, rotations)

    def test_transform(self):
        rotate = Rotation.from_rotvec([0., 0., np.pi / 2]).as_matrix()
        translate = np.array([1., 2., 3.])
        transformed = self.world_trace.transform(rotate=rotate, translate=translate)

        for i in range(len(self.world_trace)):
            np.testing.assert_array_almost_equal(transformed.positions[i],
                                                 rotate @ self.world_trace.positions[i] + translate)
            np.testing.assert_array_almost_equal(transformed.rotations[i],
                                                 rotate @ self.world_trace.rotations[i])

    def test_calculate_imu_trace(self):
        # Positive-up gravity, i.e. the specific-force convention: a stationary sensor
        # reads +9.81 along the up axis. calculate_imu_trace just adds this to the
        # finite-differenced world acceleration, so the sign here sets the sign we expect.
        gravity = np.array([0, 0, 9.81])
        magnetic_field = np.array([0, 0, 1])
        parabola_constant = 0.05
        timestamps = np.linspace(0, 1, 100)
        positions = [np.array([0, 0, parabola_constant * t ** 2]) for t in timestamps]
        rotations = [np.eye(3) for _ in range(100)]
        world_trace = WorldTrace(timestamps, positions, rotations)
        imu_trace = world_trace.calculate_imu_trace(gravity, magnetic_field)

        expected_acc = np.array([0, 0, 9.81 + (2 * parabola_constant)])
        expected_gyro = np.array([0, 0, 0])
        expected_mag = np.array([0, 0, 1])
        expected_imu_trace = IMUTrace(timestamps, [expected_gyro for _ in range(100)], [expected_acc for _ in range(100)], [expected_mag for _ in range(100)])
        # The first and last two samples are contaminated by the finite-difference edges.
        expected_imu_trace_trimmed = expected_imu_trace[2:-2]
        imu_trace_trimmed = imu_trace[2:-2]
        self.assertTrue(imu_trace_trimmed.allclose(expected_imu_trace_trimmed))

    def test_calculate_imu_trace_skip_lin_acc(self):
        # With skip_lin_acc the accelerometer sees gravity alone, rotated into the
        # sensor frame -- this is the path _sync_traces relies on.
        gravity = np.array([0., 9.81, 0.])
        timestamps = np.linspace(0, 1, 50)
        rotations = Rotation.from_rotvec(np.linspace(0, np.pi / 4, 50)[:, None] * np.array([0., 0., 1.])).as_matrix()
        positions = np.random.rand(50, 3)
        world_trace = WorldTrace(timestamps, positions, rotations)

        imu_trace = world_trace.calculate_imu_trace(gravity, skip_lin_acc=True)

        expected_acc = np.einsum('nji,j->ni', rotations, gravity)
        np.testing.assert_array_almost_equal(imu_trace.acc, expected_acc)

    def test_get_rotation_errors_deg_zero(self):
        parabola_constant = 0.05
        timestamps = np.linspace(0, 1, 100)
        positions = [np.array([0, 0, parabola_constant * t ** 2]) for t in timestamps]
        rotations = [np.eye(3) for _ in range(100)]

        world_trace_1 = WorldTrace(timestamps, positions, rotations)
        world_trace_2 = WorldTrace(timestamps, positions, rotations)
        errors = world_trace_1.get_rotation_errors_deg(world_trace_2)
        self.assertTrue(np.allclose(errors, np.zeros(100)))

    def test_get_rotation_errors_deg_rotating(self):
        parabola_constant = 0.05
        timestamps = np.linspace(0, 1, 100)
        positions = [np.array([0, 0, parabola_constant * t ** 2]) for t in timestamps]
        rotation_axis = np.array([1, 0, 0])
        rotation_amount = np.linspace(0, np.pi, 100)
        rotations = [Rotation.from_rotvec(rotation_axis * rotation_amount[i]).as_matrix() for i in range(100)]
        world_trace = WorldTrace(timestamps, positions, rotations)

        rotations_zero = [np.eye(3) for _ in range(100)]
        world_trace_2 = WorldTrace(timestamps, positions, rotations_zero)

        errors = world_trace.get_rotation_errors_deg(world_trace_2)

        np.testing.assert_allclose(errors, rotation_amount * 180 / np.pi, atol=0.01)

    def test_resample_preserves_pose_at_shared_timestamps(self):
        timestamps = np.linspace(0, 1, 101)  # 100 Hz
        positions = np.column_stack([timestamps, 2 * timestamps, np.zeros_like(timestamps)])
        rotations = Rotation.from_rotvec(timestamps[:, None] * np.array([0., 0., 1.])).as_matrix()
        world_trace = WorldTrace(timestamps, positions, rotations)

        resampled = world_trace.resample(50.0)

        self.assertAlmostEqual(resampled.get_sample_frequency(), 50.0, places=6)
        # Linear position and constant-rate rotation are both exactly representable,
        # so slerp/interp should reproduce them at the new timestamps.
        expected_positions = np.column_stack([resampled.timestamps, 2 * resampled.timestamps,
                                              np.zeros_like(resampled.timestamps)])
        expected_rotations = Rotation.from_rotvec(resampled.timestamps[:, None] * np.array([0., 0., 1.])).as_matrix()
        np.testing.assert_array_almost_equal(resampled.positions, expected_positions)
        np.testing.assert_array_almost_equal(resampled.rotations, expected_rotations)

    def test_reconstruct_from_markers_stationary(self):
        num_samples = 5
        expected_positions = np.zeros((num_samples, 3))
        expected_rotations = np.array([np.eye(3)] * num_samples)
        markers = _markers_from_poses(expected_positions, expected_rotations)

        positions, rotations = _reconstruct_from_markers(*markers)

        np.testing.assert_array_almost_equal(positions, expected_positions)
        np.testing.assert_array_almost_equal(rotations, expected_rotations)

    def test_reconstruct_from_markers_moving(self):
        num_samples = 5
        expected_positions = np.array([[i, i, 2 * i] for i in range(num_samples)], dtype=np.float64)
        expected_rotations = np.array([np.eye(3)] * num_samples)
        markers = _markers_from_poses(expected_positions, expected_rotations)

        positions, rotations = _reconstruct_from_markers(*markers)

        np.testing.assert_array_almost_equal(positions, expected_positions)
        np.testing.assert_array_almost_equal(rotations, expected_rotations)

    def test_reconstruct_from_markers_simple_rotation(self):
        num_samples = 5
        expected_rotations = np.array([Rotation.from_rotvec([0., 0., np.pi / 2]).as_matrix()] * num_samples)
        expected_positions = np.tile(np.array([0.1, -0.2, 0.3]), (num_samples, 1))
        markers = _markers_from_poses(expected_positions, expected_rotations)

        positions, rotations = _reconstruct_from_markers(*markers)

        np.testing.assert_array_almost_equal(positions, expected_positions)
        np.testing.assert_array_almost_equal(rotations, expected_rotations)

    def test_reconstruct_from_markers_rotating_and_moving(self):
        num_samples = 20
        expected_rotations = Rotation.from_euler('XYZ', np.random.rand(num_samples, 3)).as_matrix()
        expected_positions = np.random.rand(num_samples, 3)
        markers = _markers_from_poses(expected_positions, expected_rotations)

        positions, rotations = _reconstruct_from_markers(*markers)

        np.testing.assert_array_almost_equal(positions, expected_positions)
        np.testing.assert_array_almost_equal(rotations, expected_rotations)

    def test_reconstruct_from_markers_isolates_a_faulty_marker(self):
        # One marker jumps 5 cm off the plate for part of the trial. The fault-isolation
        # branch should notice, drop that marker, and rebuild the pose from the other three.
        num_samples = 40
        expected_rotations = Rotation.from_euler('XYZ', np.random.rand(num_samples, 3) * 0.5).as_matrix()
        expected_positions = np.random.rand(num_samples, 3)
        marker_o, marker_d, marker_x, marker_y = _markers_from_poses(expected_positions, expected_rotations)

        corrupted_o = marker_o.copy()
        bad_frames = slice(10, 20)
        corrupted_o[bad_frames] += np.array([0.05, 0.0, 0.0])

        positions, rotations = _reconstruct_from_markers(corrupted_o, marker_d, marker_x, marker_y)

        np.testing.assert_allclose(positions, expected_positions, atol=1e-6)
        np.testing.assert_allclose(rotations, expected_rotations, atol=1e-6)

    def test_repair_glitches_fixes_a_swapped_marker_flip(self):
        # Swapping two marker labels for a few frames is invisible to the distance-based
        # fault isolation — every inter-marker distance is preserved — but it flips the
        # reconstructed frame by 180 deg. This is the real failure seen on Subject06's
        # femur_l plate, reproduced here by swapping the o/d and x/y labels.
        num_samples = 60
        timestamps = np.arange(num_samples) / 100.0
        expected_rotations = Rotation.from_rotvec(
            np.linspace(0.0, 0.6, num_samples)[:, None] * np.array([0.0, 0.0, 1.0])).as_matrix()
        expected_positions = np.tile(np.array([0.1, 1.2, -0.3]), (num_samples, 1))
        marker_o, marker_d, marker_x, marker_y = _markers_from_poses(expected_positions,
                                                                    expected_rotations)
        bad = slice(30, 34)
        swapped_o, swapped_d = marker_o.copy(), marker_d.copy()
        swapped_x, swapped_y = marker_x.copy(), marker_y.copy()
        swapped_o[bad], swapped_d[bad] = marker_d[bad], marker_o[bad]
        swapped_x[bad], swapped_y[bad] = marker_y[bad], marker_x[bad]

        _, glitched = _reconstruct_from_markers(swapped_o, swapped_d, swapped_x, swapped_y)
        flip_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(
            np.matmul(expected_rotations.transpose(0, 2, 1), glitched)).as_rotvec(), axis=1))
        self.assertGreater(flip_deg[bad].min(), 179.0,
                           "fixture is not reproducing the 180 deg flip it is meant to")
        self.assertLess(flip_deg[:30].max(), 1e-6)

        positions, rotations, valid, report = repair_reconstruction_glitches(
            expected_positions.copy(), glitched, timestamps, name='femur_l')

        self.assertGreaterEqual(report['flipped'] + report['interpolated'], 4)
        # The repair interpolates across the burst, so it cannot be exact — but it must
        # land within a degree, versus the 180 deg it replaced.
        repaired_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(
            np.matmul(expected_rotations.transpose(0, 2, 1), rotations)).as_rotvec(), axis=1))
        self.assertLess(repaired_deg.max(), 1.0)
        np.testing.assert_allclose(positions, expected_positions, atol=1e-6)

    def test_repair_glitches_unflips_a_sustained_swap(self):
        # The real Subject06 femur_l failure: the swap PERSISTS, holding a 180 deg flip for
        # thousands of frames rather than glitching for a few. Interpolating the boundaries
        # of such a block leaves its interior corrupt, so it has to be un-flipped instead.
        num_samples = 400
        timestamps = np.arange(num_samples) / 100.0
        expected_rotations = Rotation.from_rotvec(
            np.linspace(0.0, 2.0, num_samples)[:, None] * np.array([0.2, 1.0, -0.4])).as_matrix()
        expected_positions = np.tile(np.array([0.0, 1.1, 0.2]), (num_samples, 1))

        # A half turn about the plate's y axis, applied on the right — what an o<->d, x<->y
        # label swap produces — held for 250 of the 400 frames.
        flipped_block = slice(100, 350)
        corrupted = expected_rotations.copy()
        corrupted[flipped_block] = corrupted[flipped_block] @ np.diag([-1.0, 1.0, -1.0])

        positions, rotations, valid, report = repair_reconstruction_glitches(
            expected_positions.copy(), corrupted, timestamps, name='femur_l')

        self.assertGreater(report['flipped'], 200,
                           "the sustained block should be un-flipped, not just its edges")
        self.assertEqual(report['unresolved'], 0)
        error_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(
            np.matmul(expected_rotations.transpose(0, 2, 1), rotations)).as_rotvec(), axis=1))
        # Every frame recovered, including deep inside the block where edge interpolation
        # would have left a full 180 deg of error.
        self.assertLess(error_deg.max(), 1.0)
        self.assertLess(error_deg[200], 1e-9)
        np.testing.assert_allclose(positions, expected_positions, atol=1e-6)

    def test_repair_glitches_leaves_an_unexplained_jump_alone(self):
        # A discontinuity that is not a half turn is a tracking dropout, not a swap. It must
        # be reported rather than "corrected" by inventing a rotation that fits.
        num_samples = 200
        timestamps = np.arange(num_samples) / 100.0
        rotations = Rotation.from_rotvec(
            np.linspace(0.0, 0.4, num_samples)[:, None] * np.array([0.0, 0.0, 1.0])).as_matrix()
        positions = np.tile(np.array([0.0, 1.0, 0.0]), (num_samples, 1))
        # 70 deg is far past the speed limit but nowhere near a half turn.
        rotations[100:] = rotations[100:] @ Rotation.from_euler('y', 70.0, degrees=True).as_matrix()

        _, _, valid, report = repair_reconstruction_glitches(positions, rotations, timestamps)

        self.assertEqual(report['flipped'], 0)
        self.assertEqual(report['unresolved'], 1)

    def test_repair_glitches_leaves_clean_data_untouched(self):
        # Fast but physical motion must not be "repaired". 600 deg/s is brisk for a
        # segment and still an order of magnitude under the threshold.
        num_samples = 50
        timestamps = np.arange(num_samples) / 100.0
        rotations = Rotation.from_rotvec(
            timestamps[:, None] * np.deg2rad(600.0) * np.array([0.0, 1.0, 0.0])).as_matrix()
        positions = np.random.rand(num_samples, 3)

        out_positions, out_rotations, valid, report = repair_reconstruction_glitches(
            positions, rotations, timestamps)

        self.assertEqual(report, {'flipped': 0, 'interpolated': 0, 'unresolved': 0})
        np.testing.assert_array_equal(out_rotations, rotations)
        np.testing.assert_array_equal(out_positions, positions)

    def test_from_trc(self):
        num_samples = 6
        # 0.05 rad of random attitude per axis at 10 ms spacing is a few hundred deg/s —
        # brisk but physical. The scale used to be 0.5, which is >3000 deg/s and trips
        # from_trc's glitch detector, so the whole fixture got flagged as corrupt.
        expected_rotations = Rotation.from_euler('XYZ', np.random.rand(num_samples, 3) * 0.05).as_matrix()
        expected_positions = np.random.rand(num_samples, 3)
        # Lift the plate to a realistic height: from_trc infers millimetres from the
        # magnitude of the coordinates, so the values have to exceed 1000 mm for the
        # mm->m branch to trigger.
        expected_positions[:, 1] += 1.5
        markers = _markers_from_poses(expected_positions, expected_rotations)
        timestamps = np.arange(num_samples) / 100.0

        # TRC layout from_trc expects: marker names on line 4 at the column of their X
        # coordinate, an X/Y/Z line, a blank line, then the samples. Positions are written
        # in millimetres to exercise the mm->m conversion.
        marker_names = ['torso_o', 'torso_d', 'torso_x', 'torso_y']
        header_cells = ['Frame#', 'Time']
        axis_cells = ['', '']
        for i, name in enumerate(marker_names):
            header_cells += [name, '', '']
            axis_cells += [f'X{i + 1}', f'Y{i + 1}', f'Z{i + 1}']

        lines = [
            'PathFileType\t4\t(X/Y/Z)\tsynthetic.trc',
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames',
            f'100.00\t100.00\t{num_samples}\t{len(marker_names)}\tmm\t100.00\t1\t{num_samples}',
            '\t'.join(header_cells),
            '\t'.join(axis_cells),
            '',
        ]
        for t in range(num_samples):
            row = [str(t + 1), f'{timestamps[t]:.6f}']
            for marker in markers:
                row += [f'{v * 1000.0:.6f}' for v in marker[t]]
            lines.append('\t'.join(row))

        with tempfile.NamedTemporaryFile(mode='w+', suffix='.trc', delete=False, encoding='utf-8') as temp_file:
            temp_file.write('\n'.join(lines) + '\n')
            temp_path = temp_file.name

        try:
            world_traces = WorldTrace.from_trc(temp_path)
        finally:
            os.remove(temp_path)

        self.assertEqual(list(world_traces.keys()), ['torso'])
        trace = world_traces['torso']
        np.testing.assert_allclose(trace.timestamps, timestamps)
        np.testing.assert_allclose(trace.positions, expected_positions, atol=1e-6)
        np.testing.assert_allclose(trace.rotations, expected_rotations, atol=1e-6)

    def test_get_joint_center(self):
        # A hinge-free ball joint: both segments rotate independently, but the same
        # world point is rigidly attached to each, at offset_parent in the parent's frame
        # and offset_child in the child's frame.
        num_samples = 60
        offset_parent = np.array([0.0, 0.0, 0.2])
        offset_child = np.array([0.0, -0.15, 0.0])
        timestamps = np.arange(num_samples) / 100.0

        parent_rotations = Rotation.from_euler('XYZ', np.random.rand(num_samples, 3)).as_matrix()
        child_rotations = Rotation.from_euler('XYZ', np.random.rand(num_samples, 3)).as_matrix()
        parent_positions = np.random.rand(num_samples, 3)

        joint_centers = parent_positions + np.einsum('nij,j->ni', parent_rotations, offset_parent)
        child_positions = joint_centers - np.einsum('nij,j->ni', child_rotations, offset_child)

        parent = WorldTrace(timestamps, parent_positions, parent_rotations)
        child = WorldTrace(timestamps, child_positions, child_rotations)

        estimated_parent, estimated_child, error = parent.get_joint_center(child)

        np.testing.assert_array_almost_equal(estimated_parent, offset_parent)
        np.testing.assert_array_almost_equal(estimated_child, offset_child)
        np.testing.assert_array_almost_equal(error, np.zeros((num_samples, 3)))


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
        trace = WorldTrace.from_markers(markers, self.timestamps, template=self.template)
        self.assertIsInstance(trace, WorldTrace)
        self.assertEqual(len(trace), self.n)
        self.assertFalse(trace.valid[150:170].any())
        self.assertTrue(trace.valid[:150].all())


class TestValidityMask(unittest.TestCase):
    """The mask says which frames are trustworthy GROUND TRUTH.

    Its whole job is to keep repaired frames on the uniform time grid (so filters and
    finite differences still work) while keeping them out of error statistics.
    """

    @staticmethod
    def _clean(num_samples=200, rate=100.0):
        timestamps = np.arange(num_samples) / rate
        rotations = Rotation.from_rotvec(
            np.linspace(0.0, 0.4, num_samples)[:, None] * np.array([0.0, 0.0, 1.0])).as_matrix()
        positions = np.tile(np.array([0.0, 1.0, 0.0]), (num_samples, 1))
        return positions, rotations, timestamps

    def test_defaults_to_all_valid(self):
        """Every trace built by hand or by a generator is valid throughout."""
        positions, rotations, timestamps = self._clean()
        self.assertTrue(WorldTrace(timestamps, positions, rotations).valid.all())

    def test_length_mismatch_raises(self):
        positions, rotations, timestamps = self._clean(num_samples=50)
        with self.assertRaises(ValueError):
            WorldTrace(timestamps, positions, rotations, valid=np.ones(49, dtype=bool))

    def test_clean_data_is_fully_valid(self):
        positions, rotations, timestamps = self._clean()
        _, _, valid, report = repair_reconstruction_glitches(positions, rotations, timestamps)
        self.assertEqual(report, {'flipped': 0, 'interpolated': 0, 'unresolved': 0})
        self.assertTrue(valid.all())

    def test_interpolated_frames_are_invalid(self):
        """A SLERP'd pose is plausible, not measured; it must not score a filter."""
        positions, rotations, timestamps = self._clean()
        # A 3-frame burst: large enough to trip the speed limit, small enough to be a
        # transition rather than a sustained swap.
        rotations[100:103] = rotations[100:103] @ Rotation.from_euler(
            'y', 40.0, degrees=True).as_matrix()

        _, _, valid, report = repair_reconstruction_glitches(positions, rotations, timestamps)

        self.assertGreater(report['interpolated'], 0)
        self.assertEqual(int((~valid).sum()), report['interpolated'])
        self.assertFalse(valid[100:103].any(), "the glitched frames must be marked invalid")
        self.assertTrue(valid[:95].all(), "clean frames elsewhere must stay valid")

    def test_unresolved_gap_is_invalid(self):
        """The frames deliberately LEFT corrupt are exactly what the mask is for."""
        positions, rotations, timestamps = self._clean()
        # 70 deg: past the speed limit, nowhere near a half turn, so unexplainable.
        rotations[100:] = rotations[100:] @ Rotation.from_euler('y', 70.0, degrees=True).as_matrix()

        _, _, valid, report = repair_reconstruction_glitches(positions, rotations, timestamps)

        self.assertEqual(report['unresolved'], 1)
        self.assertFalse(valid.all(), "an unresolved gap cannot leave every frame valid")
        self.assertFalse(valid[100], "the frame after the unexplained jump is corrupt")

    def test_unflipped_frames_stay_valid(self):
        """A swap is a labelling error the repair genuinely undoes — the pose is measured."""
        num_samples = 400
        timestamps = np.arange(num_samples) / 100.0
        rotations = Rotation.from_rotvec(
            np.linspace(0.0, 2.0, num_samples)[:, None] * np.array([0.2, 1.0, -0.4])).as_matrix()
        positions = np.tile(np.array([0.0, 1.1, 0.2]), (num_samples, 1))
        rotations[100:350] = rotations[100:350] @ np.diag([-1.0, 1.0, -1.0])

        _, _, valid, report = repair_reconstruction_glitches(positions, rotations, timestamps)

        self.assertGreater(report['flipped'], 200)
        # Only the block's two edges are transition frames; its 250-frame interior is
        # recovered exactly and must not be thrown away.
        self.assertTrue(valid[150:300].all(),
                        "un-flipped interior frames are correct and must stay valid")

    def test_mask_survives_slicing_and_rezeroing(self):
        positions, rotations, timestamps = self._clean()
        valid = np.ones(200, dtype=bool)
        valid[50:60] = False
        trace = WorldTrace(timestamps, positions, rotations, valid=valid)

        np.testing.assert_array_equal(trace[40:70].valid, valid[40:70])
        np.testing.assert_array_equal(trace.copy().valid, valid)
        np.testing.assert_array_equal(trace.re_zero_timestamps().valid, valid)
        np.testing.assert_array_equal(trace.transform(rotate=np.eye(3)).valid, valid)
        self.assertEqual(len(trace[100].valid), 1)

    def test_subtraction_intersects_masks(self):
        """A difference is trustworthy only where both operands are."""
        positions, rotations, timestamps = self._clean(num_samples=100)
        a_valid, b_valid = np.ones(100, dtype=bool), np.ones(100, dtype=bool)
        a_valid[10:20] = False
        b_valid[15:25] = False
        a = WorldTrace(timestamps, positions, rotations, valid=a_valid)
        b = WorldTrace(timestamps, positions, rotations, valid=b_valid)
        np.testing.assert_array_equal((a - b).valid, a_valid & b_valid)

    def test_resampling_is_conservative(self):
        """A resampled frame drawing on an invalid neighbour is itself invalid."""
        positions, rotations, timestamps = self._clean(num_samples=200)
        valid = np.ones(200, dtype=bool)
        valid[100:110] = False
        trace = WorldTrace(timestamps, positions, rotations, valid=valid)

        resampled = trace.resample(50.0)
        self.assertEqual(len(resampled.valid), len(resampled))
        self.assertFalse(resampled.valid.all(), "the invalid run must survive downsampling")
        # Conservative: the invalid window may widen at the edges but must never shrink
        # to nothing or drift away from where the original damage was.
        invalid_times = resampled.timestamps[~resampled.valid]
        self.assertGreaterEqual(invalid_times.min(), timestamps[99])
        self.assertLessEqual(invalid_times.max(), timestamps[110])


class TestValidityMaskOnRealTrials(unittest.TestCase):
    """Against the plate-trials this dataset is already known to have damaged.

    These are the eleven of 152 that repair_reconstruction_glitches reports on. They are
    the reason the mask exists, so they are what it is tested against.
    """

    #                                    subject     activity        plate         invalid
    KNOWN_DAMAGE = [
        ('08', 'walking',      'calcn_l_imu', 30),   # 17 interpolated + 2 unresolved gaps
        ('06', 'walking',      'pelvis_imu',  36),   # 6 unresolved gaps
        ('06', 'complexTasks', 'femur_l_imu', 57),   # 153 un-flipped (valid) + 57 interpolated
        ('02', 'complexTasks', 'pelvis_imu',   9),
        ('04', 'walking',      'torso_imu',   14),
    ]

    def _trace(self, subject, activity, plate):
        import contextlib, io
        import paths
        trc = paths.raw_trial_dir(subject, activity) / f'{activity}.trc'
        if not trc.is_file():
            self.skipTest(f"source data not present at {trc}")
        with contextlib.redirect_stdout(io.StringIO()):   # the repair warnings are expected
            return WorldTrace.from_trc(trc)[plate]

    def test_known_damaged_plates_are_masked(self):
        for subject, activity, plate, expected in self.KNOWN_DAMAGE:
            with self.subTest(subject=subject, activity=activity, plate=plate):
                trace = self._trace(subject, activity, plate)
                self.assertEqual(int((~trace.valid).sum()), expected)

    def test_damage_is_localized_not_diffuse(self):
        """Every known fault is a short burst. A mask spanning a large fraction of the
        trial would mean the detector had fired on real motion, not on a glitch."""
        for subject, activity, plate, _ in self.KNOWN_DAMAGE:
            with self.subTest(subject=subject, activity=activity, plate=plate):
                trace = self._trace(subject, activity, plate)
                self.assertLess((~trace.valid).mean(), 0.001)

    def test_subject08_calcn_l_brackets_the_known_boundary(self):
        """Subject08/walking/calcn_l is the 176 deg jump WorldTrace's own comments name.

        Independent cross-check: a separate marker-geometry analysis of this dataset put
        the corrupt window at roughly frames 59830-59836. The mask is derived from angular
        SPEED, not marker distances, so agreement here is two detectors meeting.
        """
        trace = self._trace('08', 'walking', 'calcn_l_imu')
        invalid = np.flatnonzero(~trace.valid)
        in_window = invalid[(invalid >= 59820) & (invalid <= 59870)]
        self.assertGreater(len(in_window), 0,
                           "the known 176 deg boundary must be masked")

    def test_undamaged_plates_are_fully_valid(self):
        """The other 141 plate-trials must not be masked at all — a detector that fires
        on clean data would quietly shrink every error statistic in the repo."""
        for subject, activity, plate in [('01', 'walking', 'femur_r_imu'),
                                         ('03', 'walking', 'tibia_l_imu'),
                                         ('01', 'complexTasks', 'torso_imu')]:
            with self.subTest(subject=subject, activity=activity, plate=plate):
                self.assertTrue(self._trace(subject, activity, plate).valid.all())


if __name__ == '__main__':
    unittest.main()
