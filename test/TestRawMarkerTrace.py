import unittest
import numpy as np
from typing import List
from src.toolchest.RawMarkerTrace import RawMarkerTrace


class TestRawMarkerTrace(unittest.TestCase):
    def test_add_point(self):
        trace = RawMarkerTrace()
        point = np.array([1.0, 2.0, 3.0])
        timestep = 1.0
        trace.add_point(point, timestep)
        self.assertEqual(len(trace.points), 1)
        self.assertEqual(len(trace.timesteps), 1)
        np.testing.assert_array_equal(trace.points[0], point)
        self.assertEqual(trace.timesteps[0], timestep)

    def test_distance(self):
        trace = RawMarkerTrace()
        trace.add_point(np.array([1.0, 1.0, 1.0]), 1.0)
        trace.add_point(np.array([2.0, 2.0, 2.0]), 2.0)
        point = np.array([3.0, 3.0, 3.0])
        timestep = 3.0
        dist = trace.distance(point, timestep)
        expected_distance = np.linalg.norm(np.array([3.0, 3.0, 3.0]) - point)
        self.assertAlmostEqual(dist, expected_distance)

    def test_time_since_last_point(self):
        trace = RawMarkerTrace()
        trace.add_point(np.array([1.0, 1.0, 1.0]), 1.0)
        trace.add_point(np.array([2.0, 2.0, 2.0]), 2.0)
        self.assertEqual(trace.time_since_last_point(3.0), 1.0)

    def test_duration(self):
        trace = RawMarkerTrace()
        trace.add_point(np.array([1.0, 1.0, 1.0]), 1.0)
        trace.add_point(np.array([2.0, 2.0, 2.0]), 2.0)
        self.assertEqual(trace.duration(), 1.0)

    def test_overlap(self):
        trace1 = RawMarkerTrace()
        trace1.add_point(np.array([1.0, 1.0, 1.0]), 1.0)
        trace1.add_point(np.array([2.0, 2.0, 2.0]), 2.0)
        trace2 = RawMarkerTrace()
        trace2.add_point(np.array([1.5, 1.5, 1.5]), 1.5)
        trace2.add_point(np.array([2.5, 2.5, 2.5]), 2.5)
        self.assertEqual(trace1.overlap(trace2), 0.5)

    def test_mean_and_std_distance_at_overlap(self):
        trace1 = RawMarkerTrace()
        trace1.add_point(np.array([1.0, 1.0, 1.0]), 1.0)
        trace1.add_point(np.array([1.1, 1.1, 1.1]), 2.0)
        trace2 = RawMarkerTrace()
        trace2.add_point(np.array([2.0, 2.0, 2.0]), 1.0)
        trace2.add_point(np.array([2.1, 2.1, 2.1]), 2.0)
        mean_dist, std_dist = trace1.mean_and_std_distance_at_overlap(trace2)
        self.assertAlmostEqual(mean_dist, np.sqrt(3))
        self.assertAlmostEqual(std_dist, 0.0)

    def test_make_traces(self):
        points = [
            [np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])],
            [np.array([1.1, 1.1, 1.1]), np.array([2.1, 2.1, 2.1])]
        ]
        timestamps = [1.0, 2.0]
        traces = RawMarkerTrace.make_traces(points, timestamps, max_merge_distance=1.2, max_merge_time=1.2)
        self.assertEqual(len(traces), 2)
        self.assertEqual(len(traces[0].points), 2)
        self.assertEqual(len(traces[1].points), 2)

    def test_recompute_traces(self):
        points = [
            [np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])],
            [np.array([1.1, 1.1, 1.1]), np.array([2.1, 2.1, 2.1])]
        ]
        timestamps = [1.0, 2.0]
        traces = RawMarkerTrace.make_traces(points, timestamps, max_merge_distance=1.2, max_merge_time=1.2)
        recomputed_traces = RawMarkerTrace.recompute_traces(traces, max_merge_distance=1.2, max_merge_time=1.2)
        self.assertEqual(len(recomputed_traces), 2)

    def test_make_trace_clusters(self):
        points = [
            [np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])],
            [np.array([1.1, 1.1, 1.1]), np.array([2.1, 2.1, 2.1])]
        ]
        timestamps = [1.0, 2.0]
        traces = RawMarkerTrace.make_traces(points, timestamps, max_merge_distance=1.2, max_merge_time=1.2)
        clusters = RawMarkerTrace.make_trace_clusters(traces, max_distance=2.0)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 2)

    def test_recompute_traces_assuming_from_single_plate(self):
        # Create some sample data
        plate_width = 2.0
        plate_height = 3.0
        traces = []

        # Create some synthetic traces
        trace1 = RawMarkerTrace()
        trace1.add_point(np.array([1.0, 2.0, 3.0]), 0.0)
        trace1.add_point(np.array([2.0, 3.0, 4.0]), 1.0)
        trace1.add_point(np.array([3.0, 4.0, 5.0]), 2.0)
        traces.append(trace1)

        offset2 = np.array([plate_width, 0.0, 0.0])
        trace2 = RawMarkerTrace()
        trace2.add_point(np.array([1.0, 2.0, 3.0]) + offset2, 0.0)
        trace2.add_point(np.array([2.0, 3.0, 4.0]) + offset2, 1.0)
        trace2.add_point(np.array([3.0, 4.0, 5.0]) + offset2, 2.0)
        traces.append(trace2)

        offset3 = np.array([0.0, plate_height, 0.0])
        trace3 = RawMarkerTrace()
        trace3.add_point(np.array([1.0, 2.0, 3.0]) + offset3, 0.0)
        trace3.add_point(np.array([2.0, 3.0, 4.0]) + offset3, 1.0)
        trace3.add_point(np.array([3.0, 4.0, 5.0]) + offset3, 2.0)
        traces.append(trace3)

        # Recompute the traces assuming they are from a single plate
        corner_traces = RawMarkerTrace.recompute_traces_assuming_from_single_plate(traces, plate_width, plate_height)

        # Check the number of corner traces
        self.assertEqual(len(corner_traces), 4)

        # Check that each corner trace has the correct number of points
        expected_timesteps = [0.0, 1.0, 2.0]
        for corner_trace in corner_traces:
            self.assertEqual(len(corner_trace.points), len(expected_timesteps))
            self.assertEqual(len(corner_trace.timesteps), len(expected_timesteps))
            np.testing.assert_array_almost_equal(corner_trace.timesteps, expected_timesteps)

        # Check that the positions are consistent with the plate dimensions
        for t in range(len(expected_timesteps)):
            positions = [corner_trace.points[t] for corner_trace in corner_traces]
            assert len(positions) == 4
            width_vector = positions[1] - positions[0]
            height_vector = positions[2] - positions[0]
            np.testing.assert_array_almost_equal(np.linalg.norm(width_vector), plate_width, decimal=1)
            np.testing.assert_array_almost_equal(np.linalg.norm(height_vector), plate_height, decimal=1)

    def test_recompute_traces_assuming_from_single_plate_not_enough_points(self):
        # Create some sample data
        plate_width = 2.0
        plate_height = 3.0
        traces = []

        # Create some synthetic traces
        trace1 = RawMarkerTrace()
        trace1.add_point(np.array([1.0, 2.0, 3.0]), 0.0)
        trace1.add_point(np.array([2.0, 3.0, 4.0]), 1.0)
        trace1.add_point(np.array([3.0, 4.0, 5.0]), 2.0)
        traces.append(trace1)

        offset2 = np.array([plate_width, 0.0, 0.0])
        trace2 = RawMarkerTrace()
        trace2.add_point(np.array([1.0, 2.0, 3.0]) + offset2, 0.0)
        trace2.add_point(np.array([2.0, 3.0, 4.0]) + offset2, 1.0)
        trace2.add_point(np.array([3.0, 4.0, 5.0]) + offset2, 2.0)
        traces.append(trace2)

        # Recompute the traces assuming they are from a single plate
        corner_traces = RawMarkerTrace.recompute_traces_assuming_from_single_plate(traces, plate_width, plate_height)

        # Check the number of corner traces
        self.assertEqual(len(corner_traces), 0)

    def test_contain_timesteps(self):
        trace = RawMarkerTrace()
        trace.add_point(np.array([1.0, 2.0, 3.0]), 0.0)
        trace.add_point(np.array([2.0, 3.0, 4.0]), 1.0)
        trace.add_point(np.array([3.0, 4.0, 5.0]), 2.0)

        # Test when timesteps are fully contained
        timesteps_contained = [0.0, 2.0]
        self.assertTrue(trace.contain_timesteps(timesteps_contained))

        # Test when timesteps are not fully contained
        timesteps_not_contained = [-1.0, 2.0]
        self.assertFalse(trace.contain_timesteps(timesteps_not_contained))

        timesteps_not_contained = [0.0, 3.0]
        self.assertFalse(trace.contain_timesteps(timesteps_not_contained))

    def test_resample_timesteps(self):
        trace = RawMarkerTrace()
        trace.add_point(np.array([1.0, 2.0, 3.0]), 0.0)
        trace.add_point(np.array([2.0, 3.0, 4.0]), 1.0)
        trace.add_point(np.array([3.0, 4.0, 5.0]), 2.0)

        # Test resampling within the contained timesteps
        new_timesteps = [0.0, 0.5, 1.0, 1.5, 2.0]
        resampled_trace = trace.resample_timesteps(new_timesteps)
        self.assertIsNotNone(resampled_trace)
        self.assertEqual(len(resampled_trace.points), len(new_timesteps))
        self.assertEqual(len(resampled_trace.timesteps), len(new_timesteps))

        expected_points = np.array([
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [2.0, 3.0, 4.0],
            [2.5, 3.5, 4.5],
            [3.0, 4.0, 5.0]
        ])

        for i in range(len(new_timesteps)):
            np.testing.assert_array_almost_equal(resampled_trace.points[i], expected_points[i], decimal=6)

        # Test resampling outside the contained timesteps
        out_of_bound_timesteps = [-0.5, 0.5, 1.0, 1.5, 2.5]
        resampled_trace = trace.resample_timesteps(out_of_bound_timesteps)
        self.assertIsNone(resampled_trace)

if __name__ == '__main__':
    unittest.main()
