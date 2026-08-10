import os
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.WorldTrace import WorldTrace, _reconstruct_from_markers

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

    def test_from_trc(self):
        num_samples = 6
        expected_rotations = Rotation.from_euler('XYZ', np.random.rand(num_samples, 3) * 0.5).as_matrix()
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


if __name__ == '__main__':
    unittest.main()
