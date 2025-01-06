import unittest
import numpy as np
import torch
from src.toolchest.RawMarkerPlateTrace import RawMarkerPlateTrace


class TestRawMarkerPlateTrace(unittest.TestCase):
    def test_fit_point_corners(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define the observed points
        points = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 3.0]),
            np.array([1.0, 5.0, 3.0]),
            np.array([3.0, 5.0, 3.0])
        ]

        # Define an initial guess for the position and orientation
        initial_guess = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        initial_guess += np.random.randn(9) * 0.1

        # Fit the point corners
        position, width_axis, height_axis, loss = plate_trace.fit_point_corners(points, initial_guess, [True, False, True, True], tol=1e-12)

        # Check that the position is correct
        np.testing.assert_array_almost_equal(position, [1.0, 2.0, 3.0], decimal=1)

        # Check that the width_axis is correct
        np.testing.assert_array_almost_equal(width_axis, [1.0, 0.0, 0.0], decimal=1)

        # Check that the height_axis is correct
        np.testing.assert_array_almost_equal(height_axis, [0.0, 1.0, 0.0], decimal=1)

    def test_get_concatenated_point_corners(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define a position and orientation
        x = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)

        # Get the concatenated point corners
        corners = plate_trace.get_concatenated_point_corners(x)

        # Check that the corners are correct
        expected_corners = torch.tensor([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 3.0],
            [1.0, 5.0, 3.0],
            [3.0, 5.0, 3.0]
        ], dtype=torch.float32).view(-1)
        torch.testing.assert_allclose(corners, expected_corners)

    def test_initialize_to_points_3(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define the observed points
        points = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 3.0]),
            # np.array([1.0, 5.0, 3.0]),
            np.array([3.0, 5.0, 3.0])
        ]
        np.random.shuffle(points)

        # Initialize to points
        position, width_axis, height_axis = plate_trace.initialize_to_points(points)

        # Recovered points
        recovered_points = plate_trace.get_point_corners(position, width_axis, height_axis)

        for point in points:
            found = False
            for recovered_point in recovered_points:
                if np.linalg.norm(point - recovered_point) < 1e-6:
                    found = True
                    break
            self.assertTrue(found)


    def test_initialize_to_points_4(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define the observed points
        points = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 3.0]),
            np.array([1.0, 5.0, 3.0]),
            np.array([3.0, 5.0, 3.0])
        ]
        np.random.shuffle(points)

        # Initialize to points
        position, width_axis, height_axis = plate_trace.initialize_to_points(points)

        # Recovered points
        recovered_points = plate_trace.get_point_corners(position, width_axis, height_axis)

        for point in points:
            found = False
            for recovered_point in recovered_points:
                if np.linalg.norm(point - recovered_point) < 1e-6:
                    found = True
                    break
            self.assertTrue(found)

    def test_initialize_to_points_5(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define the observed points
        original_points = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 3.0]),
            np.array([1.0, 5.0, 3.0]),
            np.array([3.0, 5.0, 3.0])
        ]
        points = original_points.copy()
        points.append(np.array([2.0, 3.5, 3.0]))
        np.random.shuffle(points)

        # Initialize to points
        position, width_axis, height_axis = plate_trace.initialize_to_points(points)

        # Recovered points
        recovered_points = plate_trace.get_point_corners(position, width_axis, height_axis)

        for point in original_points:
            found = False
            for recovered_point in recovered_points:
                if np.linalg.norm(point - recovered_point) < 1e-6:
                    found = True
                    break
            self.assertTrue(found)

    def test_get_point_corners_and_get_concatenated_point_corners_order(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define the position and orientation
        position = np.array([1.0, 2.0, 3.0])
        width_axis = np.array([1.0, 0.0, 0.0])
        height_axis = np.array([0.0, 1.0, 0.0])

        # Get point corners
        point_corners = plate_trace.get_point_corners(position, width_axis, height_axis)

        # Convert to PyTorch tensor and get concatenated point corners
        x = torch.tensor(np.concatenate((position, width_axis, height_axis)), dtype=torch.float32)
        concatenated_corners = plate_trace.get_concatenated_point_corners(x).view(-1, 3).numpy()

        # Ensure both methods return the same points in the same order
        for pc, cc in zip(point_corners, concatenated_corners):
            np.testing.assert_array_almost_equal(pc, cc, decimal=6)

    def test_snap_to_points_4(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define the initial position and orientation
        initial_position = np.array([1.0, 2.0, 3.0])
        initial_width_axis = np.array([1.0, 0.0, 0.0])
        initial_height_axis = np.array([0.0, 1.0, 0.0])

        # Define the observed points
        points = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 3.0]),
            np.array([1.0, 5.0, 3.0]),
            np.array([3.0, 5.0, 3.0])
        ]
        np.random.shuffle(points)

        # Snap to points
        snapped_position, snapped_width_axis, snapped_height_axis, loss = plate_trace.snap_to_points(points,
                                                                                                     initial_position,
                                                                                                     initial_width_axis,
                                                                                                     initial_height_axis)

        # Check that the snapped position is correct
        np.testing.assert_array_almost_equal(snapped_position, [1.0, 2.0, 3.0], decimal=1)

        # Check that the snapped width_axis is correct
        np.testing.assert_array_almost_equal(snapped_width_axis, [1.0, 0.0, 0.0], decimal=1)

        # Check that the snapped height_axis is correct
        np.testing.assert_array_almost_equal(snapped_height_axis, [0.0, 1.0, 0.0], decimal=1)

        # Check that the loss is small (indicating a good fit)
        self.assertLess(loss, 1e-3)

    def test_snap_to_points_3(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define the initial position and orientation
        initial_position = np.array([1.0, 2.0, 3.0])
        initial_width_axis = np.array([1.0, 0.0, 0.0])
        initial_height_axis = np.array([0.0, 1.0, 0.0])

        # Define the observed points
        points = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 3.0]),
            # np.array([1.0, 5.0, 3.0]),
            np.array([3.0, 5.0, 3.0])
        ]
        np.random.shuffle(points)

        # Snap to points
        snapped_position, snapped_width_axis, snapped_height_axis, loss = plate_trace.snap_to_points(points,
                                                                                                     initial_position,
                                                                                                     initial_width_axis,
                                                                                                     initial_height_axis)

        # Check that the snapped position is correct
        np.testing.assert_array_almost_equal(snapped_position, [1.0, 2.0, 3.0], decimal=1)

        # Check that the snapped width_axis is correct
        np.testing.assert_array_almost_equal(snapped_width_axis, [1.0, 0.0, 0.0], decimal=1)

        # Check that the snapped height_axis is correct
        np.testing.assert_array_almost_equal(snapped_height_axis, [0.0, 1.0, 0.0], decimal=1)

        # Check that the loss is small (indicating a good fit)
        self.assertLess(loss, 1e-3)

    def test_snap_to_points_5(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define the initial position and orientation
        initial_position = np.array([1.0, 2.0, 3.0])
        initial_width_axis = np.array([1.0, 0.0, 0.0])
        initial_height_axis = np.array([0.0, 1.0, 0.0])

        # Define the observed points
        points = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 3.0]),
            np.array([1.0, 5.0, 3.0]),
            np.array([3.0, 5.0, 3.0]),
            np.array([2.0, 3.5, 3.0])
        ]
        np.random.shuffle(points)

        # Snap to points
        snapped_position, snapped_width_axis, snapped_height_axis, loss = plate_trace.snap_to_points(points,
                                                                                                     initial_position,
                                                                                                     initial_width_axis,
                                                                                                     initial_height_axis)

        # Check that the snapped position is correct
        np.testing.assert_array_almost_equal(snapped_position, [1.0, 2.0, 3.0], decimal=1)

        # Check that the snapped width_axis is correct
        np.testing.assert_array_almost_equal(snapped_width_axis, [1.0, 0.0, 0.0], decimal=1)

        # Check that the snapped height_axis is correct
        np.testing.assert_array_almost_equal(snapped_height_axis, [0.0, 1.0, 0.0], decimal=1)

        # Check that the loss is small (indicating a good fit)
        self.assertLess(loss, 1e-3)

    def test_snap_to_points_4_with_movement(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define the initial position and orientation
        initial_position = np.array([1.0, 2.0, 3.0])
        initial_width_axis = np.array([1.0, 0.0, 0.0])
        initial_height_axis = np.array([0.0, 1.0, 0.0])

        # Define the observed points
        offset = np.array([1.0, 2.0, 3.0])
        points = [
            np.array([1.0, 2.0, 3.0]) + offset,
            np.array([3.0, 2.0, 3.0]) + offset,
            np.array([1.0, 5.0, 3.0]) + offset,
            np.array([3.0, 5.0, 3.0]) + offset
        ]
        np.random.shuffle(points)

        # Snap to points
        snapped_position, snapped_width_axis, snapped_height_axis, loss = plate_trace.snap_to_points(points,
                                                                                                     initial_position,
                                                                                                     initial_width_axis,
                                                                                                     initial_height_axis)

        # Check that the snapped position is correct
        np.testing.assert_array_almost_equal(snapped_position, [1.0, 2.0, 3.0] + offset, decimal=1)

        # Check that the snapped width_axis is correct
        np.testing.assert_array_almost_equal(snapped_width_axis, [1.0, 0.0, 0.0], decimal=1)

        # Check that the snapped height_axis is correct
        np.testing.assert_array_almost_equal(snapped_height_axis, [0.0, 1.0, 0.0], decimal=1)

        # Check that the loss is small (indicating a good fit)
        self.assertLess(loss, 1e-3)

    def test_add_timestep(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define a few sets of observed points
        points_1 = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 3.0]),
            np.array([1.0, 5.0, 3.0]),
            np.array([3.0, 5.0, 3.0])
        ]
        points_2 = [
            np.array([2.0, 3.0, 4.0]),
            np.array([4.0, 3.0, 4.0]),
            # np.array([2.0, 6.0, 4.0]),
            np.array([4.0, 6.0, 4.0])
        ]
        points_3 = [
            np.array([3.0, 4.0, 5.0]),
            np.array([5.0, 4.0, 5.0]),
            np.array([3.0, 7.0, 5.0]),
            np.array([5.0, 7.0, 5.0]),
            np.array([6.0, 8.0, 6.0])
        ]

        # Add timesteps
        plate_trace.add_timestep(points_1, 0.0, tol=1e-12)
        plate_trace.add_timestep(points_2, 1.0, tol=1e-12)
        plate_trace.add_timestep(points_3, 2.0, tol=1e-12)

        # Check that the positions, width_axis_direction, height_axis_direction, and timesteps have been updated correctly
        self.assertEqual(len(plate_trace.positions), 3)
        self.assertEqual(len(plate_trace.width_axis_direction), 3)
        self.assertEqual(len(plate_trace.height_axis_direction), 3)
        self.assertEqual(len(plate_trace.timesteps), 3)

        # Check the content of the positions, width_axis_direction, height_axis_direction, and timesteps
        expected_positions = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0])
        ]
        expected_width_axes = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0])
        ]
        expected_height_axes = [
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0])
        ]
        expected_timesteps = [0.0, 1.0, 2.0]

        for i in range(3):
            plate_trace_points = plate_trace.get_point_corners(plate_trace.positions[i], plate_trace.width_axis_direction[i], plate_trace.height_axis_direction[i])
            expected_points = plate_trace.get_point_corners(expected_positions[i], expected_width_axes[i], expected_height_axes[i])
            for j in range(4):
                plate_point = plate_trace_points[j]
                found_in_expected = False
                for expected_point in expected_points:
                    if np.linalg.norm(plate_point - expected_point) < 1e-6:
                        found_in_expected = True
                        break
                if not found_in_expected:
                    print('Step:', i)
                    print('Error in point:', j)
                    print('Time:', plate_trace.timesteps[i])
                    print('Expected:', expected_points)
                    print('Actual:', plate_trace_points)
                self.assertTrue(found_in_expected)
            self.assertAlmostEqual(plate_trace.timesteps[i], expected_timesteps[i], places=1)

    def test_clean_up_points(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define the initial position and orientation
        position = np.array([1.0, 2.0, 3.0])
        width_axis = np.array([1.0, 0.0, 0.0])
        height_axis = np.array([0.0, 1.0, 0.0])

        # Define the raw points
        raw_points = [
            # np.array([1.0, 2.0, 3.0]),  # Close to the first corner
            np.array([3.01, 2.0, 3.0]),  # Close to the second corner
            np.array([1.0, 5.0, 3.0]),  # Close to the third corner
            np.array([3.0, 5.04, 3.0])  # Close to the fourth corner but slightly off
        ]
        np.random.shuffle(raw_points)

        # Expected points after cleaning up
        expected_points = [
            np.array([1.0, 2.0, 3.0]),  # Virtual corner because no point is available
            np.array([3.01, 2.0, 3.0]),  # Taken from raw_points
            np.array([1.0, 5.0, 3.0]),  # Taken from raw_points
            np.array([3.0, 5.0, 3.0])  # Virtual corner because raw_point is slightly off
        ]

        # Clean up points
        cleaned_points = plate_trace.clean_up_points(raw_points, position, width_axis, height_axis)

        # Check that the cleaned points match the expected points
        for cleaned_point, expected_point in zip(cleaned_points, expected_points):
            np.testing.assert_array_almost_equal(cleaned_point, expected_point, decimal=2)

    def test_resample(self):
        # Define the plate dimensions
        width = 2.0
        height = 3.0
        plate_trace = RawMarkerPlateTrace(width, height)

        # Define a few sets of observed points
        positions = [
            np.array([1.0, 2.0, 3.0]),
            # np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0])
        ]
        width_axis = np.array([1.0, 0.0, 0.0])
        height_axis = np.array([0.0, 1.0, 0.0])

        # Add timesteps
        plate_trace.timesteps = [0.0, 2.0]
        plate_trace.positions = positions
        plate_trace.width_axis_direction = [width_axis] * 2
        plate_trace.height_axis_direction = [height_axis] * 2

        # Resample the trace at a frequency of 1 Hz
        resampled_trace = plate_trace.resample(1.0)

        # Expected new timesteps
        expected_timesteps = [0.0, 1.0, 2.0]

        # Check the new timesteps
        np.testing.assert_array_almost_equal(resampled_trace.timesteps, expected_timesteps, decimal=6)

        # Check that the positions, width_axis_direction, and height_axis_direction are correctly interpolated
        expected_positions = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0]
        ])
        np.testing.assert_array_almost_equal(resampled_trace.positions, expected_positions, decimal=6)

        expected_width_axes = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        np.testing.assert_array_almost_equal(resampled_trace.width_axis_direction, expected_width_axes, decimal=6)

        expected_height_axes = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        np.testing.assert_array_almost_equal(resampled_trace.height_axis_direction, expected_height_axes, decimal=6)

if __name__ == '__main__':
    unittest.main()


