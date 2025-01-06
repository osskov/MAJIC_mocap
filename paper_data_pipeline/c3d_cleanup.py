import os
import re
from typing import List, Dict

import nimblephysics as nimble
import numpy as np

from toolchest.RawMarkerTrace import RawMarkerTrace


def evaluate_marker_quality(marker_timesteps: List[Dict[str, np.ndarray]], timestamps: List[float]):
    unknown_marker_regex = re.compile(r'U[0-9]+')

    marker_names = set()
    for marker_frame in marker_timesteps:
        marker_names.update(marker_frame.keys())

    unknown_markers = set()
    known_markers = set()
    for marker_name in marker_names:
        if unknown_marker_regex.match(marker_name):
            unknown_markers.add(marker_name)
        else:
            known_markers.add(marker_name)

    print(f'Unknown markers: {unknown_markers}')
    print(f'Known markers: {known_markers}')

    for marker in known_markers:
        num_frames_observed = 0
        num_frames_missing = 0
        for frame in marker_timesteps:
            if marker not in frame:
                num_frames_missing += 1
            else:
                num_frames_observed += 1
        percent_frames_observed = num_frames_observed / len(marker_timesteps)

        jump_velocity_values = []
        for t in range(1, len(marker_timesteps)):
            if marker in marker_timesteps[t] and marker in marker_timesteps[t - 1]:
                jump_velocity = np.linalg.norm(marker_timesteps[t][marker] - marker_timesteps[t - 1][marker]) / (
                        timestamps[t] - timestamps[t - 1])
                jump_velocity_values.append(jump_velocity)
        max_jump_velocity = max(jump_velocity_values) if jump_velocity_values else 0
        max_jump_index = jump_velocity_values.index(max_jump_velocity)

        if max_jump_velocity > 10.0:
            print(
                f'Marker: {marker}, percent frames observed: {percent_frames_observed}, max jump velocity: {max_jump_velocity} at frame {max_jump_index}')


def check_data_quality(input_path: str):
    # Use traces to relink unlabeled markers to the labeled markers
    traces_path = os.path.splitext(input_path)[0] + '_traces.pkl'

    if os.path.isfile(traces_path):
        print("Loading traces from file...")
        traces = RawMarkerTrace.load_traces_from_file(traces_path)
    else:
        print('Loading C3D file: ', input_path)

        # Load the raw C3D
        c3d: nimble.biomechanics.C3D = nimble.biomechanics.C3DLoader.loadC3D(os.path.abspath(input_path))
        timestamps: List[float] = c3d.timestamps
        marker_timesteps: List[Dict[str, np.ndarray]] = c3d.markerTimesteps

        evaluate_marker_quality(marker_timesteps, timestamps)

        print("Generating traces...")
        traces = RawMarkerTrace.make_traces(marker_timesteps, timestamps, max_merge_time=0.05, max_merge_distance=0.04)
        print("Saving traces to file...")
        RawMarkerTrace.save_traces_to_file(traces, traces_path)

    marker_timesteps, timestamps = RawMarkerTrace.convert_traces_to_list_of_dicts(traces)

    evaluate_marker_quality(marker_timesteps, timestamps)


if __name__ == '__main__':
    trial_path = '../../six_imu_data/raw_data/Subj1'
    # isolate the c3d file in this directory
    c3d_paths = [os.path.join(trial_path, f) for f in os.listdir(trial_path) if f.endswith('.c3d')]

    for c3d_path in c3d_paths:
        check_data_quality(c3d_path)
