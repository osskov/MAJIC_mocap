import json
import os
import re
from typing import List, Dict

import nimblephysics as nimble
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from toolchest.RawMarkerTrace import RawMarkerTrace


def generate_trc_from_c3d(input_path: str, mapping_path: str, output_path: str, plot: bool = False):
    # Load the mapping
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    # Get the names of the IMU plates
    imu_names = []
    for values in mapping.values():
        if values[-2:] == '_1' or values[-2:] == '_2':
            if values[:-2] not in imu_names:
                imu_names.append(values[:-2])

    # Use traces to relink unlabeled markers to the labeled markers
    traces_path = os.path.join(os.path.dirname(input_path), 'traces.pkl')
    if os.path.isfile(traces_path):
        print("Loading traces from file...")
        traces = RawMarkerTrace.load_traces_from_file(traces_path)
    else:
        # Load the raw C3D
        c3d: nimble.biomechanics.C3D = nimble.biomechanics.C3DLoader.loadC3D(os.path.abspath(input_path))
        timestamps: List[float] = c3d.timestamps
        marker_timesteps: List[Dict[str, np.ndarray]] = c3d.markerTimesteps
        rescaled_marker_timesteps = []
        # We need to immediately rescale the C3D into mm instead of meters
        for t in range(len(marker_timesteps)):
            rescaled_marker_timesteps.append(
                {marker: marker_timesteps[t][marker] * 1000 for marker in marker_timesteps[t]})

        print("Generating traces...")
        traces = RawMarkerTrace.make_traces(rescaled_marker_timesteps, timestamps, max_merge_time=0.1, max_merge_distance=10)
        # Discard traces that are only U markers
        traces = [trace for trace in traces if not re.match(r'U\d+', trace.get_dominant_label())]
        RawMarkerTrace.save_traces_to_file(traces, traces_path)

    trace_timesteps, timestamps = RawMarkerTrace.export_traces_to_c3d_format(traces)
    traces_trc_path = os.path.join(os.path.dirname(output_path), 'traces.trc')
    nimble.biomechanics.OpenSimParser.saveTRC(os.path.abspath(traces_trc_path), timestamps, trace_timesteps)
    output_marker_timesteps: List[Dict[str, np.ndarray]] = [{} for _ in range(len(timestamps))]

    # Now we need to generate our TRC file for just the IMU markers
    for imu_name in imu_names:
        marker_names = [imu_name + '_1', imu_name + '_2', imu_name + '_3', imu_name + '_4']
        output_marker_names = [imu_name + '_Y', imu_name + '_O', imu_name + '_X', imu_name + '_D']

        observed_corner: List[np.ndarray] = []
        observed_angle_axis: List[np.ndarray] = []
        observed_timestamps: List[float] = []
        for t in range(len(trace_timesteps)):
            missing_markers = [marker_name for marker_name in marker_names if marker_name not in trace_timesteps[t]]
            if len(missing_markers) > 1:
                print(f"Less than 3 {imu_name} markers present at time: {t}")
                continue
            elif len(missing_markers) == 1:
                if missing_markers[0] == imu_name + '_1':
                    x_axis = trace_timesteps[t][imu_name + '_4'] - trace_timesteps[t][imu_name + '_3']
                    x_axis /= np.linalg.norm(x_axis)
                    trace_timesteps[t][imu_name + '_1'] = trace_timesteps[t][imu_name + '_2'] + x_axis * 91.3
                elif missing_markers[0] == imu_name + '_2':
                    x_axis = trace_timesteps[t][imu_name + '_4'] - trace_timesteps[t][imu_name + '_3']
                    x_axis /= np.linalg.norm(x_axis)
                    trace_timesteps[t][imu_name + '_2'] = trace_timesteps[t][imu_name + '_1'] - x_axis * 91.3
                elif missing_markers[0] == imu_name + '_3':
                    x_axis = trace_timesteps[t][imu_name + '_1'] - trace_timesteps[t][imu_name + '_2']
                    x_axis /= np.linalg.norm(x_axis)
                    trace_timesteps[t][imu_name + '_3'] = trace_timesteps[t][imu_name + '_4'] - x_axis * 42
                elif missing_markers[0] == imu_name + '_4':
                    x_axis = trace_timesteps[t][imu_name + '_1'] - trace_timesteps[t][imu_name + '_2']
                    x_axis /= np.linalg.norm(x_axis)
                    trace_timesteps[t][imu_name + '_4'] = trace_timesteps[t][imu_name + '_3'] + x_axis * 42
            marker_1 = trace_timesteps[t][marker_names[0]]
            marker_2 = trace_timesteps[t][marker_names[1]]
            marker_3 = trace_timesteps[t][marker_names[2]]
            marker_4 = trace_timesteps[t][marker_names[3]]

            # Avg the two versions of the x-axis
            x_axis = (marker_1 - marker_2) + (marker_4 - marker_3)
            x_axis /= np.linalg.norm(x_axis)
            # Avg the two versions of the second in plane vector
            y_axis_temp = (marker_2 - marker_3) + (marker_1 - marker_4)
            y_axis_temp /= np.linalg.norm(y_axis_temp)
            # Form the out of plane vector
            z_axis = _cross_(x_axis, y_axis_temp)
            z_axis /= np.linalg.norm(z_axis)
            # Recalculate the second in plane vector or y-axis
            y_axis = _cross_(z_axis, x_axis)

            original_y_axis = (marker_2 - marker_3)
            original_y_axis /= np.linalg.norm(original_y_axis)
            disagreement = np.arccos(np.dot(y_axis, original_y_axis)) * 180 / np.pi

            R: np.ndarray = np.array([x_axis, y_axis, z_axis]).T
            axis = nimble.math.logMap(R)
            observed_corner.append(marker_2)
            observed_angle_axis.append(axis)
            observed_timestamps.append(timestamps[t])

        # Resample the angle-axis data using scipy.interpolate.interp1d at `timestamps`
        # Convert observed_angle_axis to a numpy array for easier manipulation
        observed_angle_axis_array = np.array(observed_angle_axis)  # Shape: (n_observed, 3)

        # Initialize an array to hold the resampled angle-axis data
        resampled_angle_axis_array = np.zeros((len(timestamps), 3))
        resampled_corner_array = np.zeros((len(timestamps), 3))

        # Interpolate each component of the angle-axis vector separately
        for i in range(3):
            # Create an interpolation function for the i-th component
            interp_func = interp1d(
                observed_timestamps,
                observed_angle_axis_array[:, i],
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            # Evaluate the interpolation function at all timestamps
            resampled_angle_axis_array[:, i] = interp_func(timestamps)

        # Interpolate each marker separately
        for i in range(3):
            # Create an interpolation function for the i-th component
            interp_func = interp1d(
                observed_timestamps,
                [corner[i] for corner in observed_corner],
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            # Evaluate the interpolation function at all timestamps
            resampled_corner_array[:, i] = interp_func(timestamps)

        filtered_angle_axis_array = resampled_angle_axis_array #np.zeros((len(timestamps), 3))
        filtered_corner_array = resampled_corner_array #np.zeros((len(timestamps), 3))

        # # Filter the rotations and the corner location to reduce noise
        # for i in range(3):
        #     filtered_angle_axis_array[:, i] = _lowpass_filter_(np.unwrap(resampled_angle_axis_array[:, i]), 20, timestamps[1] - timestamps[0])
        #     filtered_corner_array[:, i] = _lowpass_filter_(resampled_corner_array[:, i], 20, timestamps[1] - timestamps[0])

        # Convert the filtered array back into a list of numpy arrays
        filtered_angle_axis = [filtered_angle_axis_array[t] for t in range(len(timestamps))]
        filtered_corner = [filtered_corner_array[t] for t in range(len(timestamps))]

        # Now we need to convert the angle-axis data back to marker locations
        for t in range(len(filtered_angle_axis)):
            axis = filtered_angle_axis[t]
            R = nimble.math.expMapRot(axis)
            x_axis = R[:, 0]
            y_axis = R[:, 1]
            z_axis = R[:, 2]
            scaled_x_distance = x_axis * 42
            scaled_y_distance = y_axis * 113

            # This plate geometry is the width and height of the IMU plate if the far corner was slid in.
            corner = filtered_corner[t]
            plus_x = corner + scaled_x_distance
            minus_y = corner - scaled_y_distance
            far_corner = corner + scaled_x_distance - scaled_y_distance

            output_marker_timesteps[t][output_marker_names[0]] = plus_x
            output_marker_timesteps[t][output_marker_names[1]] = corner
            output_marker_timesteps[t][output_marker_names[2]] = minus_y
            output_marker_timesteps[t][output_marker_names[3]] = far_corner
    if plot:
        plot_timesteps(output_marker_timesteps)
    nimble.biomechanics.OpenSimParser.saveTRC(os.path.abspath(output_path), timestamps, output_marker_timesteps)


def _cross_(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # For some reason, numpy's cross product causes the IDE to freak out and say that any code after it is
    # unreachable.
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


# Filtering each marker to reduce noise
def _lowpass_filter_(trace: np.array, cutoff_freq: float, dt: float):
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * 1 / dt
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(3, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, trace)
    return y


def _list_dict_to_dict_list_(marker_timesteps: List[Dict[str, np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    output_dict = {marker: np.zeros((len(marker_timesteps), 3)) for marker in marker_timesteps[0]}
    for t in range(len(marker_timesteps)):
        for marker in marker_timesteps[t]:
            if marker not in output_dict:
                output_dict[marker] = np.zeros((len(marker_timesteps), 3))
            output_dict[marker][t] = marker_timesteps[t][marker]
    return output_dict


def _dict_list_to_list_dict_(marker_dict: Dict[str, List[np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    output_list = []
    for t in range(len(marker_dict[next(iter(marker_dict))])):
        output_list.append({marker: marker_dict[marker][t] for marker in marker_dict if
                            not np.array_equal(marker_dict[marker][t], [0., 0., 0.])})
    return output_list


def plot_timesteps(marker_timesteps: List[Dict[str, np.ndarray]]):
    marker_dict = _list_dict_to_dict_list_(marker_timesteps)
    marker_groups = list(set([item.rsplit('_', 1)[0] for item in marker_dict.keys()]))
    for marker_group in marker_groups:
        fig, ax = plt.subplots(3, 1)
        for marker in marker_dict.keys():
            if marker_group in marker:
                for i in range(3):
                    ax[i].plot(marker_dict[marker][:, i], label=marker + str(i))
            plt.legend()
        plt.title('Marker group ' + marker_group)
        plt.show()


if __name__ == '__main__':
    trial_path = '../../six_imu_data/raw_data/Subj1'
    # isolate the c3d file in this directory
    c3d_path = next((os.path.join(trial_path, f) for f in os.listdir(trial_path) if f.endswith('.c3d')), None)
    # All trials have the same mapping
    mapping_path = '../../test_data/S1_IMU_Data/mapping.json'
    output_path = os.path.join(trial_path, 'corrected.trc')

    generate_trc_from_c3d(c3d_path, mapping_path, output_path, False)
