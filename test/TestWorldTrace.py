import unittest
import numpy as np
import nimblephysics as nimble
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.WorldTrace import WorldTrace


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

    def test_calculate_imu_trace(self):
        gravity = np.array([0, 0, -9.81])
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
        expected_imu_trace_trimmed = expected_imu_trace[2:-2]
        imu_trace_trimmed = imu_trace[2:-2]
        self.assertTrue(imu_trace_trimmed.allclose(expected_imu_trace_trimmed))

    def test_get_rotation_errors_deg_zero(self):
        gravity = np.array([0, 0, -9.81])
        magnetic_field = np.array([0, 0, 1])
        parabola_constant = 0.05
        timestamps = np.linspace(0, 1, 100)
        positions = [np.array([0, 0, parabola_constant * t ** 2]) for t in timestamps]
        rotations = [np.eye(3) for _ in range(100)]

        world_trace_1 = WorldTrace(timestamps, positions, rotations)
        world_trace_2 = WorldTrace(timestamps, positions, rotations)
        errors = world_trace_1.get_rotation_errors_deg(world_trace_2)
        self.assertTrue(np.allclose(errors, np.zeros(100)))

    def test_get_rotation_errors_deg_rotating(self):
        gravity = np.array([0, 0, -9.81])
        magnetic_field = np.array([0, 0, 1])
        parabola_constant = 0.05
        timestamps = np.linspace(0, 1, 100)
        positions = [np.array([0, 0, parabola_constant * t ** 2]) for t in timestamps]
        rotation_axis = np.array([1, 0, 0])
        rotation_amount = np.linspace(0, np.pi, 100)
        rotations = [nimble.math.expMapRot(rotation_axis * rotation_amount[i]) for i in range(100)]
        world_trace = WorldTrace(timestamps, positions, rotations)

        rotations_zero = [np.eye(3) for _ in range(100)]
        world_trace_2 = WorldTrace(timestamps, positions, rotations_zero)

        errors = world_trace.get_rotation_errors_deg(world_trace_2)

        np.testing.assert_allclose(errors, rotation_amount * 180 / np.pi, atol=0.01)

    def test_construct_from_markers_stationary(self):
        marker_o = np.array([np.array([0, 1, 0])] * 5)
        marker_d = np.array([np.array([1, 0, 0])] * 5)
        marker_x = np.array([np.array([0, 0, 0])] * 5)
        marker_y = np.array([np.array([1, 1, 0])] * 5)
        timestamps = np.array([0, 1, 2, 3, 4])

        expected_positions = [np.array([0.5, 0.5, 0])] * 5
        expected_rotations = [np.eye(3)] * 5

        world_trace = WorldTrace.construct_from_markers(timestamps, marker_o, marker_d, marker_x, marker_y)

        np.testing.assert_array_equal(world_trace.positions, expected_positions)
        np.testing.assert_array_equal(world_trace.rotations, expected_rotations)

    def test_construct_from_markers_moving(self):
        marker_o = np.array([[0, 1, 0], [1, 2, 1], [2, 3, 2], [3, 4, 3], [4, 5, 4]])
        marker_d = np.array([[1, 0, 0], [2, 1, 1], [3, 2, 2], [4, 3, 3], [5, 4, 4]])
        marker_x = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        marker_y = np.array([[1, 1, 0], [2, 2, 1], [3, 3, 2], [4, 4, 3], [5, 5, 4]])
        timestamps = np.array([[0, 1, 2, 3, 4]])

        expected_positions = [np.array([0.5, 0.5, 0]), np.array([1.5, 1.5, 1]), np.array([2.5, 2.5, 2]),
                             np.array([3.5, 3.5, 3]), np.array([4.5, 4.5, 4])]
        expected_rotations = [np.eye(3)] * 5

        world_trace = WorldTrace.construct_from_markers(timestamps, marker_o, marker_d, marker_x, marker_y)

        np.testing.assert_array_equal(world_trace.positions, expected_positions)
        np.testing.assert_array_equal(world_trace.rotations, expected_rotations)

    def test_construct_from_markers_simple_rotation(self):
        marker_o_marker_frame = np.array([0, 1, 0])
        marker_d_marker_frame = np.array([1, 0, 0])
        marker_x_marker_frame = np.array([0, 0, 0])
        marker_y_marker_frame = np.array([1, 1, 0])

        R = nimble.math.eulerXYZToMatrix(np.array([0, 0, np.pi/2]))
        marker_o = np.array([R @ marker_o_marker_frame] * 5)
        marker_d = np.array([R @ marker_d_marker_frame] * 5)
        marker_x = np.array([R @ marker_x_marker_frame] * 5)
        marker_y = np.array([R @ marker_y_marker_frame] * 5)
        marker_loc = [R @ np.array([0.5, 0.5, 0])] * 5
        timestamps = np.array([0, 1, 2, 3, 4])

        world_trace = WorldTrace.construct_from_markers(timestamps, marker_o, marker_d, marker_x, marker_y)
        np.testing.assert_array_almost_equal(world_trace.positions, marker_loc)
        np.testing.assert_array_almost_equal(world_trace.rotations[0], R)
        np.testing.assert_array_almost_equal(world_trace.rotations[4], R)

    def test_construct_from_markers_rotating_and_moving(self):
        marker_o_marker_frame = np.array([0, 1, 0])
        marker_d_marker_frame = np.array([1, 0, 0])
        marker_x_marker_frame = np.array([0, 0, 0])
        marker_y_marker_frame = np.array([1, 1, 0])
        marker_loc = np.array([0.5, 0.5, 0])

        num_samples = 5
        angles = np.random.rand(num_samples, 3).tolist()
        R = [nimble.math.eulerXYZToMatrix(angle) for angle in angles]
        marker_o = np.array([R[i] @ marker_o_marker_frame for i in range(num_samples)])
        marker_d = np.array([R[i] @ marker_d_marker_frame for i in range(num_samples)])
        marker_x = np.array([R[i] @ marker_x_marker_frame for i in range(num_samples)])
        marker_y = np.array([R[i] @ marker_y_marker_frame for i in range(num_samples)])
        marker_loc = [R[i] @ marker_loc for i in range(num_samples)]
        timestamps = np.arange(num_samples)

        world_trace = WorldTrace.construct_from_markers(timestamps, marker_o, marker_d, marker_x, marker_y)
        np.testing.assert_array_almost_equal(world_trace.positions, marker_loc)
        for i in range(num_samples):
            np.testing.assert_array_almost_equal(world_trace.rotations[i], R[i])

    def test_get_joint_center(self):
        offset_1 = np.array([0, 0, 1])
        offset_2 = np.array([0, 1, 0])

        trace_1_positions = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3]),
                             np.array([4, 4, 4])]
        trace_1_rotations = [np.eye(3) for _ in range(5)]
        trace_1 = WorldTrace(self.timestamps, trace_1_positions, trace_1_rotations)

        trace_2_rotations = [np.eye(3) for _ in range(5)]
        trace_2_positions = [pos1 + rot1 @ offset_1 + rot2 @ offset_2 for pos1, rot1, rot2 in zip(trace_1_positions, trace_1_rotations, trace_2_rotations)]
        trace_2 = WorldTrace(self.timestamps, trace_2_positions, trace_2_rotations)

        estimated_offset1, estimated_offset2 = trace_1.get_joint_center(trace_2)
        np.testing.assert_array_almost_equal(estimated_offset1, offset_1)
        np.testing.assert_array_almost_equal(estimated_offset2, offset_2)

if __name__ == '__main__':
    unittest.main()