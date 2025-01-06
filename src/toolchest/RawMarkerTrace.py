import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np
from .RawMarkerPlateTrace import RawMarkerPlateTrace
from scipy.interpolate import interp1d


class RawMarkerTrace:
    """
    This class exists to help with automatically labelling possibly slightly messy marker data, especially for
    automatically analyzing marker plate trials.
    """
    points: List[np.ndarray]
    timesteps: List[float]
    label_counts: Dict[str, int]

    def __init__(self):
        self.points = []
        self.label_counts = {}
        self.timesteps = []

    def get_dominant_label(self) -> Optional[str]:
        return max(self.label_counts, key=self.label_counts.get)

    def add_point(self, point: np.ndarray, label: str, timestep: float):
        self.points.append(point)
        self.timesteps.append(timestep)
        if label not in self.label_counts:
            self.label_counts[label] = 1
        else:
            self.label_counts[label] += 1

    def distance(self, point: np.ndarray, timestep: float):
        vel = np.zeros(3)
        if len(self.points) > 5:
            vel = (self.points[-1] - self.points[-4]) / (self.timesteps[-1] - self.timesteps[-4])
        projected_point = self.points[-1] + vel * (timestep - self.timesteps[-1])
        return np.linalg.norm(projected_point - point)

    def time_since_last_point(self, timestep: float):
        return timestep - self.timesteps[-1]

    def duration(self):
        return self.timesteps[-1] - self.timesteps[0]

    def overlap(self, other_trace: 'RawMarkerTrace') -> float:
        start_time = max(self.timesteps[0], other_trace.timesteps[0])
        end_time = min(self.timesteps[-1], other_trace.timesteps[-1])
        return end_time - start_time

    def mean_and_std_distance_at_overlap(self, other_trace: 'RawMarkerTrace'):
        # Find the overlap in time between the two traces
        start_time = max(self.timesteps[0], other_trace.timesteps[0])
        end_time = min(self.timesteps[-1], other_trace.timesteps[-1])
        assert(start_time < end_time)
        self_points = []
        other_points = []
        # It is possible for one trace or another to skip timesteps, so we need to check to ensure that each timestep
        # that we compare is actually present in both traces
        self_cursor = 0
        other_cursor = 0
        while self_cursor < len(self.timesteps) and other_cursor < len(other_trace.timesteps):
            if self.timesteps[self_cursor] < start_time:
                self_cursor += 1
                continue
            if other_trace.timesteps[other_cursor] < start_time:
                other_cursor += 1
                continue
            if self.timesteps[self_cursor] > end_time or other_trace.timesteps[other_cursor] > end_time:
                break
            if self.timesteps[self_cursor] < other_trace.timesteps[other_cursor]:
                self_cursor += 1
                continue
            if other_trace.timesteps[other_cursor] < self.timesteps[self_cursor]:
                other_cursor += 1
                continue
            self_points.append(self.points[self_cursor])
            other_points.append(other_trace.points[other_cursor])
            self_cursor += 1
            other_cursor += 1
        # print(len(self_points), len(other_points))
        assert(len(self_points) == len(other_points))
        distances = []
        for i in range(len(self_points)):
            distances.append(np.linalg.norm(self_points[i] - other_points[i]))
        if len(distances) == 0:
            return 0, 0
        return np.mean(distances), np.std(distances)

    def __len__(self):
        return len(self.timesteps)

    def contain_timesteps(self, timesteps: List[float]) -> bool:
        return timesteps[0] >= self.timesteps[0] and timesteps[-1] <= self.timesteps[-1]

    def resample_timesteps(self, timesteps: List[float]) -> Optional['RawMarkerTrace']:
        if not self.contain_timesteps(timesteps):
            return None
        # Interpolators for positions, width_axis_direction, and height_axis_direction
        points_interpolator = interp1d(self.timesteps, np.array(self.points), axis=0, kind='linear', fill_value='extrapolate')
        new_points = points_interpolator(timesteps)
        new_trace = RawMarkerTrace()
        for i in range(len(timesteps)):
            new_trace.add_point(new_points[i], timesteps[i])
        return new_trace

    @staticmethod
    def make_traces(marker_timesteps: List[Dict[str, np.ndarray]], timestamps: List[float],
                    max_merge_distance: float = 0.02, max_merge_time: float = 0.05) -> List['RawMarkerTrace']:
        traces: List['RawMarkerTrace'] = []
        active_traces: List['RawMarkerTrace'] = []

        for i in range(len(marker_timesteps)):
            if i % 1000 == 0:
                print('Processing timestep: ', i, ' of ', len(marker_timesteps))

            time = timestamps[i]
            marker_dict = marker_timesteps[i]
            point_labels: List[str] = [label for label in marker_dict.keys()]
            marker_points: List[np.ndarray] = [marker_dict[label] for label in point_labels]

            points_found_homes: List[bool] = [False for _ in range(len(marker_points))]

            # Match each trace to the closest point, and then turn remaining points into new traces
            for trace in active_traces:
                closest_index = -1
                closest_dist = 1000
                for j, point in enumerate(marker_points):
                    if points_found_homes[j]:
                        continue
                    dist = trace.distance(point, time)
                    if dist < max_merge_distance and dist < closest_dist:
                        closest_index = j
                        closest_dist = dist
                if closest_index > -1:
                    trace.add_point(marker_points[closest_index], point_labels[closest_index], timestamps[i])
                    points_found_homes[closest_index] = True

            for j, point in enumerate(marker_points):
                if points_found_homes[j]:
                    continue
                new_trace = RawMarkerTrace()
                new_trace.add_point(point, point_labels[j], timestamps[i])
                traces.append(new_trace)
                active_traces.append(new_trace)

            # Remove traces that haven't seen a point in max_merge_time
            to_remove = []
            for trace in active_traces:
                if trace.time_since_last_point(time) > max_merge_time:
                    to_remove.append(trace)
            if len(to_remove) > 0:
                active_traces = [trace for trace in active_traces if trace not in to_remove]

        return traces

    @staticmethod
    def recompute_traces(traces: List['RawMarkerTrace'], max_merge_distance: float = 0.02, max_merge_time: float = 0.05) -> List['RawMarkerTrace']:
        # First we need to regenerate the points list from the traces
        points_by_timestep: Dict[float, List[np.ndarray]] = {}
        for trace in traces:
            for i in range(len(trace.points)):
                if trace.timesteps[i] not in points_by_timestep:
                    points_by_timestep[trace.timesteps[i]] = []
                points_by_timestep[trace.timesteps[i]].append(trace.points[i])

        points: List[List[np.ndarray]] = []
        timestamps: List[float] = []
        for key in sorted(points_by_timestep.keys()):
            points.append(points_by_timestep[key])
            timestamps.append(key)

        return RawMarkerTrace.make_traces(points, timestamps, max_merge_distance, max_merge_time)

    @staticmethod
    def traces_to_plates(traces: List['RawMarkerTrace'], plate_width: float, plate_height: float, max_marker_jump_dist: float = 0.01) -> List[RawMarkerPlateTrace]:
        # First we need to regenerate the points list from the traces
        points_by_timestep: Dict[float, List[np.ndarray]] = {}
        for trace in traces:
            for i in range(len(trace.points)):
                if trace.timesteps[i] not in points_by_timestep:
                    points_by_timestep[trace.timesteps[i]] = []
                points_by_timestep[trace.timesteps[i]].append(trace.points[i])

        points: List[List[np.ndarray]] = []
        timestamps: List[float] = []
        for key in sorted(points_by_timestep.keys()):
            # We only want to include timesteps where we have at least 3 points
            if len(points_by_timestep[key]) >= 3:
                points.append(points_by_timestep[key])
                timestamps.append(key)

        # This means that we don't have enough timestamps with enough points to make a plate trace, so return an empty
        # list to indicate this.
        if len(points) < 3:
            return []

        marker_plate_trace = RawMarkerPlateTrace(plate_width, plate_height)
        plates = [marker_plate_trace]
        for i in range(len(points)):
            if i % 1000 == 0:
                print('  Plate fitting timestep: ', i, ' of ', len(points))
            if i > 0 and len(points[i]) < len(points[i - 1]):
                print("  Going down to ", len(points[i]), " points from ", len(points[i - 1]), " points at ", timestamps[i])
            if i > 0 and len(points[i]) > len(points[i - 1]):
                print("  Going up to ", len(points[i]), " points from ", len(points[i - 1]), " points at ", timestamps[i])
            if i > 0 and timestamps[i] - timestamps[i - 1] > 0.1:
                marker_plate_trace = RawMarkerPlateTrace(plate_width, plate_height)
                plates.append(marker_plate_trace)

            try:
                marker_plate_trace.add_timestep(points[i], timestamps[i], max_marker_jump_dist=max_marker_jump_dist)
            except Exception as e:
                marker_plate_trace = RawMarkerPlateTrace(plate_width, plate_height)
                plates.append(marker_plate_trace)
                marker_plate_trace.add_timestep(points[i], timestamps[i], max_marker_jump_dist=max_marker_jump_dist)
        return plates

    @staticmethod
    def clean_up_traces_using_plate(traces: List['RawMarkerTrace'], plate_trace: RawMarkerPlateTrace) -> List['RawMarkerTrace']:
        # First we need to regenerate the points list from the traces. This is slightly different than in trace_to_plate
        # because we keep all the timesteps of points, even if they have only 1 point observed
        points_by_timestep: Dict[float, List[np.ndarray]] = {}
        for trace in traces:
            for i in range(len(trace.points)):
                if trace.timesteps[i] not in points_by_timestep:
                    points_by_timestep[trace.timesteps[i]] = []
                points_by_timestep[trace.timesteps[i]].append(trace.points[i])

        points: List[List[np.ndarray]] = []
        timestamps: List[float] = []
        for key in sorted(points_by_timestep.keys()):
            points.append(points_by_timestep[key])
            timestamps.append(key)

        # Now we will generate 4 clean traces for the edges of the plate, using real markers where possible and
        # synthesizing markers where that is not possible.
        corner_traces = [RawMarkerTrace() for _ in range(4)]
        original_points_cursor = 0
        time_threshold = 0.001
        for i in range(len(plate_trace.timesteps)):
            t = plate_trace.timesteps[i]
            while timestamps[original_points_cursor] < t - time_threshold:
                original_points_cursor += 1

            if np.abs(timestamps[original_points_cursor] - t) < time_threshold:
                raw_points = points[original_points_cursor]
                clean_points = plate_trace.clean_up_points(raw_points, plate_trace.positions[i], plate_trace.width_axis_direction[i], plate_trace.height_axis_direction[i])
            else:
                clean_points = plate_trace.get_point_corners(plate_trace.positions[i], plate_trace.width_axis_direction[i], plate_trace.height_axis_direction[i])
            assert(len(clean_points) == 4)
            for j in range(4):
                corner_traces[j].add_point(clean_points[j], t)
        return corner_traces

    @staticmethod
    def recompute_traces_assuming_from_single_plate(traces: List['RawMarkerTrace'], plate_width: float, plate_height: float, max_marker_jump_dist=0.01) -> List[List['RawMarkerTrace']]:
        marker_plate_traces = RawMarkerTrace.traces_to_plates(traces, plate_width, plate_height, max_marker_jump_dist=max_marker_jump_dist)
        if len(marker_plate_traces) == 0:
            return []
        median_dt = np.median(np.diff(marker_plate_traces[0].timesteps))
        # Fill in the gaps in plate space, rather than marker space, to get the best interpolation
        resampled_traces = [marker_plate_trace.resample(median_dt) for marker_plate_trace in marker_plate_traces]
        return [RawMarkerTrace.clean_up_traces_using_plate(traces, resampled_trace) for resampled_trace in resampled_traces]


    @staticmethod
    def make_trace_clusters(traces: List['RawMarkerTrace'],
                            std_dev_limit=0.01,
                            min_distance=0.04,
                            max_distance=0.12,
                            min_overlap_time=0.5) -> List[List['RawMarkerTrace']]:
        # We want to begin clustering traces based on their average distance from each other
        trace_clusters: List[List[RawMarkerTrace]] = []
        for trace in traces:
            closest_cluster = None
            for cluster in trace_clusters:
                for other_trace in cluster:
                    if trace.overlap(other_trace) < min_overlap_time:
                        continue
                    dist, dev = trace.mean_and_std_distance_at_overlap(other_trace)
                    print(dist, dev)
                    if dev < std_dev_limit and max_distance > dist > min_distance:
                        closest_cluster = cluster
            if closest_cluster is None:
                trace_clusters.append([trace])
            else:
                closest_cluster.append(trace)
        return trace_clusters

    @staticmethod
    def convert_traces_to_list_of_dicts(traces: List['RawMarkerTrace']) -> Tuple[List[Dict[str, np.ndarray]], List[float]]:
        print(f'Converting {len(traces)} traces to list of dicts')
        print(f'Trace average lengths: {np.mean([len(trace) for trace in traces])}')

        # Kill the traces that are shorter than a certain length
        traces = [trace for trace in traces if len(trace) > 100]

        print(f'Have {len(traces)} traces remaining after minimum length filter')
        print(f'Trace average lengths: {np.mean([len(trace) for trace in traces])}')

        # Check trace jump velocities
        for trace in traces:
            jump_velocity_values = []
            for t in range(1, len(trace)):
                jump_velocity = np.linalg.norm(trace.points[t] - trace.points[t - 1]) / (trace.timesteps[t] - trace.timesteps[t - 1])
                jump_velocity_values.append(jump_velocity)
            max_jump_velocity = max(jump_velocity_values) if jump_velocity_values else 0
            max_jump_index = jump_velocity_values.index(max_jump_velocity)

            if max_jump_velocity > 10.0:
                print(f'Trace marker: {trace.get_dominant_label()}, max jump velocity: {max_jump_velocity} at frame {max_jump_index}')

        # Bucket the traces by their dominant label
        trace_groups: Dict[str, List[RawMarkerTrace]] = {}
        for trace in traces:
            label = trace.get_dominant_label()
            if label not in trace_groups:
                trace_groups[label] = []
            trace_groups[label].append(trace)

        # Print the number of traces in each group
        for label, group in trace_groups.items():
            print(f'{label}: {len(group)} traces')

        marker_timesteps_dict: Dict[float, Dict[str, np.ndarray]] = {}

        for trace in traces:
            label = trace.get_dominant_label()
            for i in range(len(trace.timesteps)):
                t = trace.timesteps[i]
                if t not in marker_timesteps_dict:
                    marker_timesteps_dict[t] = {}
                marker_timesteps_dict[t][label] = trace.points[i]

        marker_timesteps: List[Dict[str, np.ndarray]] = []
        timestamps: List[float] = []

        for key in sorted(marker_timesteps_dict.keys()):
            marker_timesteps.append(marker_timesteps_dict[key])
            timestamps.append(key)

        return marker_timesteps, timestamps

    @staticmethod
    def export_traces_to_c3d_format(traces: List['RawMarkerTrace']) -> List[Dict[str, np.ndarray]]:
        # First we find the number of time steps by finding the last time step and the framerate
        max_time = max(trace.timesteps[-1] for trace in traces if trace.timesteps)
        framerate = np.round(1 / np.median(np.diff(traces[0].timesteps)))
        num_steps = int(np.ceil(max_time * framerate) + 1)
        timestamps = np.linspace(0, max_time, num_steps)

        # First we create a List of Dicts, where each dict is a timestep and the keys are the marker names
        marker_timesteps: List[Dict[str, np.ndarray]] = [{} for _ in range(num_steps)]

        # loop through all the traces, loop through their time steps and add the points to the marker_timesteps
        for trace in traces:
            # Calculate the index offset according to the timestep
            offset = np.searchsorted(timestamps, trace.timesteps[0])
            for i in range(len(trace.timesteps)):
                marker_timesteps[i+offset][trace.get_dominant_label()] = trace.points[i]

        return marker_timesteps, timestamps


    @staticmethod
    def save_traces_to_file(traces: List['RawMarkerTrace'], filename: str):
        with open(filename, 'wb') as f:
            pickle.dump([{
                'points': trace.points,
                'timesteps': trace.timesteps,
                'label_counts': trace.label_counts
            } for trace in traces], f)

    @staticmethod
    def load_traces_from_file(filename: str) -> List['RawMarkerTrace']:
        with open(filename, 'rb') as f:
            data_list = pickle.load(f)
            traces = []
            for data in data_list:
                trace = RawMarkerTrace()
                trace.points = data['points']
                trace.timesteps = data['timesteps']
                trace.label_counts = data['label_counts']
                traces.append(trace)
            return traces
