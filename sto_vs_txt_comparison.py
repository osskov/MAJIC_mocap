import nimblephysics as nimble
from src.toolchest.PlateTrial import PlateTrial
import scipy.signal as signal
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
JOINT_SEGMENT_DICT = {'R_Hip': ('pelvis_imu', 'femur_r_imu'),
                      'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
                      'R_Ankle': ('tibia_r_imu', 'calcn_r_imu'),
                      'L_Hip': ('pelvis_imu', 'femur_l_imu'),
                      'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
                      'L_Ankle': ('tibia_l_imu', 'calcn_l_imu'),
                      'Lumbar': ('pelvis_imu', 'torso_imu'),
                      }
def load_sensor_rotations_from_folder(imu_folder_path: str) -> Dict[str, 'IMUTrace']:
    imu_traces = {}

    # Find the mapping xml file
    mapping_file = next((f for f in os.listdir(imu_folder_path) if f.endswith('.xml')), None)
    if mapping_file is None:
        raise FileNotFoundError("No mapping file found in IMU folder")

    # find the subdirectories in the IMU folder
    subdirs = [d for d in os.listdir(imu_folder_path) if os.path.isdir(os.path.join(imu_folder_path, d))]

    # Parse the XML file
    tree = ET.parse(os.path.join(imu_folder_path, mapping_file))
    root = tree.getroot()
    trial_prefix = root.find('.//trial_prefix').text
    rotation_matrices = {}
    # Iterate over each ExperimentalSensor element and load its IMUTrace
    for sensor in root.findall('.//ExperimentalSensor'):
        sensor_name = sensor.get('name').strip()
        name_in_model = sensor.find('name_in_model').text.strip()
        # check if this sensor is already in the rotation matrices dict
        rotation_matrices[name_in_model] = {}
        for subdir in subdirs:
            file_name = f"{trial_prefix}{sensor_name}.txt"
            file_path = os.path.join(imu_folder_path, subdir, 'LowerExtremity', file_name)

            # Extract update rate
            with open(file_path, "r") as f:
                for line in f:
                    if line.startswith("// Update Rate"):
                        freq = float(line.split(":")[1].split("Hz")[0])
                        break

            # Read the file into a DataFrame
            df = pd.read_csv(file_path, delimiter='\t', skiprows=5)
            df = df.apply(pd.to_numeric)

            if any(df['UTC_Valid'].isna()) is False:
                # Shift data over by one column
                df = df.shift(1, axis=1)

            # Extract data
            timestamps = 1 / freq * np.arange(len(df))
            acc = [np.array(row) for row in df[['Acc_X', 'Acc_Y', 'Acc_Z']].values]
            gyro = [np.array(row) for row in df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values]
            mag = [np.array(row) for row in df[['Mag_X', 'Mag_Y', 'Mag_Z']].values]
            rotation_matrices[name_in_model][subdir] = [np.array(row).reshape(3, 3) for row in
                                                       df[['Mat[1][1]', 'Mat[1][2]', 'Mat[1][3]',
                                                           'Mat[2][1]', 'Mat[2][2]', 'Mat[2][3]',
                                                           'Mat[3][1]', 'Mat[3][2]', 'Mat[3][3]']].values]

            # Create IMUTrace objects
            # imu_traces[name_in_model] = IMUTrace(timestamps=timestamps, acc=acc, gyro=gyro, mag=mag)
    return rotation_matrices

def load_segment_orientations_from_folder(imu_folder_path: str):
    # find the subdirectories in the IMU folder
    subdirs = [d for d in os.listdir(imu_folder_path) if os.path.isdir(os.path.join(imu_folder_path, d))]
    subdirs += ['markers']
    rotation_matrices = {}
    for subdir in subdirs:
        file_path = os.path.join(imu_folder_path, subdir, 'walking_orientations.sto')
        if 'markers' in subdir:
            file_path = "/Users/six/projects/work/MAJIC_mocap/data/ODay_Data/Subject03/walking/Mocap/walking_orientations.sto"
        # Read the file to find the endHeader index
        end_header_index = 0
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                end_header_index = i + 1
                if line.startswith('endheader'):
                    break
        # Read the file into a DataFrame
        df = pd.read_csv(file_path, delimiter='\t', skiprows=end_header_index)

        # Apply the function to convert columns from strings, to arrays of 4 numbers
        df = df.map(lambda x: np.array([float(i) for i in x.split(',')]) if isinstance(x, str) else x)
        df = df.map(lambda x: nimble.math.expMapRot(nimble.math.quatToExp(nimble.math.Quaternion(x[0], x[1], x[2], x[3]))) if isinstance(x, np.ndarray) and len(x) == 4 else x)

        for col in df.columns:
            if col != 'time':
                if col not in rotation_matrices:
                    rotation_matrices[col] = {}
                # Convert the column to a list
                rotation_matrices[col][subdir] = df[col].tolist()
    return rotation_matrices

def _sync_arrays(array1: np.array, array2: np.array) -> Tuple[slice, slice]:
    """
    This function takes two arrays and time syncs them. It returns the time synced arrays.
    """
    assert isinstance(array1, np.ndarray) and isinstance(array2, np.ndarray), "Arrays must be numpy arrays"
    assert len(array1) > 0 and len(array2) > 0, "Arrays must have at least one element"
    assert array1.ndim == 1 and array2.ndim == 1, "Arrays must be 1D"
    assert array1.dtype == array2.dtype, "Arrays must have the same dtype"

    array1_len = len(array1)
    array2_len = len(array2)
    max_len = max(array1_len, array2_len)

    array1_padded = np.pad(array1, (0, max_len - array1_len), mode='constant')
    array2_padded = np.pad(array2, (0, max_len - array2_len), mode='constant')
    correlation = signal.correlate(array1_padded, array2_padded, mode='full')
    lag = np.argmax(correlation) - (max_len - 1)
    index1 = max(0, lag)
    index2 = max(0, -lag)
    new_length = min(array1_len - index1, array2_len - index2)
    return slice(index1, index1 + new_length), slice(index2, index2 + new_length)

def load_kinematics_mot(mot_file_path: str) -> pd.DataFrame:
    """
    Load kinematics data from a .mot file.
    :param mot_file_path: Path to the .mot file.
    :return: A dictionary with segment names as keys and lists of rotation matrices as values.
    """
    # read the file to figure out where the 'endheader' is
    end_header_index = 0
    with open(mot_file_path, 'r') as f:
        for i, line in enumerate(f):
            end_header_index = i + 1
            if line.startswith('endheader'):
                break
    data = pd.read_csv(mot_file_path, delimiter='\t', skiprows=end_header_index)
    return data

def get_joint_orientations_from_segment_orientations(
        segment_orientations: Dict[str, Dict[str, List[np.ndarray]]],
        ) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Extract joint orientations from segment orientations for a given condition.
    :param segment_orientations: Dictionary of segment orientations.
    :param condition: Condition to filter the segment orientations.
    :return: Dictionary of joint orientations.
    """
    joint_orientations = {}
    for joint, (parent_segment, child_segment) in JOINT_SEGMENT_DICT.items():
        joint_orientations[joint] = {}
        if parent_segment not in segment_orientations or child_segment not in segment_orientations:
            continue
        for method, orientations in segment_orientations[parent_segment].items():
            joint_orientations[joint][method] = [R_wp.T @ R_wc for R_wp, R_wc in zip(segment_orientations[parent_segment][method],
                                                  segment_orientations[child_segment][method])]
    return joint_orientations

# plates = PlateTrial.load_trial_from_folder('data/Subject03/walking')
imu_folder_path = os.path.join('data', 'ODay_Data', 'Subject03', 'walking', 'IMU')
sensor_rotation_matrices = load_sensor_rotations_from_folder(imu_folder_path)
segment_rotation_matrices = load_segment_orientations_from_folder(imu_folder_path)

# Also add the orientations for the majic filter

# # Print the rotation matrices for each sensor and method
# segment = "tibia_l_imu"
# filter = "xsens"
# # plt.figure()
# # flm0_txt = [rot[0,0] for rot in sensor_rotation_matrices[segment][filter]]
# # flm0_sto = [rot[0,0] for rot in segment_rotation_matrices[segment][filter]]
# # plt.plot(flm0_txt, label='.txt index')
# # plt.plot(flm0_sto, label='.sto index')
# # plt.show()
#
slice1_list = []
slice2_list = []
for sensor_name in sensor_rotation_matrices:
    if sensor_name not in segment_rotation_matrices:
        continue
    for method in sensor_rotation_matrices[sensor_name]:
        for i in range(3):
            for j in range(3):
                array_1 = np.array([rot[i, j] for rot in sensor_rotation_matrices[sensor_name][method]])
                array_2 = np.array([rot[i, j] for rot in segment_rotation_matrices[sensor_name][method]])
                slice1, slice2 = _sync_arrays(array_1, array_2)
                slice1_list.append(slice1)
                slice2_list.append(slice2)

# Treat None as 0 for sorting purposes
sorted_slice1 = sorted(slice1_list, key=lambda s: s.start if s.start is not None else 0)
slice1 = sorted_slice1[len(sorted_slice1) // 2]  # Take the middle slice
sorted_slice2 = sorted(slice2_list, key=lambda s: s.start if s.start is not None else 0)
slice2 = sorted_slice2[len(sorted_slice2) // 2]  # Take the middle slice

rotation_difference = {}
for sensor_name in sensor_rotation_matrices:
    if sensor_name not in segment_rotation_matrices:
        continue
    rotation_difference[sensor_name] = {}
    for method in sensor_rotation_matrices[sensor_name]:
        rotation_difference[sensor_name][method] = [rot1 @ rot2.T for rot1, rot2 in
                                                    zip(sensor_rotation_matrices[sensor_name][method][slice1],
                                                        segment_rotation_matrices[sensor_name][method][slice2])]
        # Calculate the mean and standard deviation for each element in the rotation matrices
        # Convert the rotation matrices to euler angles
        rotation_difference[sensor_name][method] = [nimble.math.matrixToEulerXYZ(rot) for rot in
                                                    rotation_difference[sensor_name][method]]
        for i in range(3):
            # for j in range(3):
                mean_value = np.mean([rot[i] for rot in rotation_difference[sensor_name][method]])
                std_value = np.std([rot[i] for rot in rotation_difference[sensor_name][method]])
                print(f"Sensor: {sensor_name}, Method: {method}, Element ({i}): Mean = {mean_value}, Std = {std_value}")

# for sensor_name in sensor_rotation_matrices:
#     rotation_difference[sensor_name] = {}
#     if sensor_name not in segment_rotation_matrices:
#         continue
#     xsens_start = []
#     mahony_start = []
#     madgwick_start = []
#     for method in sensor_rotation_matrices[sensor_name]:
#         if method == 'xsens':
#             xsens_start = sensor_rotation_matrices[sensor_name][method][slice1][0]
#         elif method == 'mahony':
#             mahony_start = sensor_rotation_matrices[sensor_name][method][slice1][0]
#         elif method == 'madgwick':
#             madgwick_start = sensor_rotation_matrices[sensor_name][method][slice1][0]
#     rotation_difference[sensor_name]['xsens to madgwick'] = nimble.math.matrixToEulerXYZ(xsens_start.T @ madgwick_start)
#     rotation_difference[sensor_name]['xsens to mahony'] = nimble.math.matrixToEulerXYZ(xsens_start.T @ mahony_start)
#     rotation_difference[sensor_name]['mahony to madgwick'] = nimble.math.matrixToEulerXYZ(mahony_start.T @ madgwick_start)
#
# # Print the rotation differences for each sensor and method
# for sensor_name in rotation_difference:
#     if sensor_name not in segment_rotation_matrices:
#         continue
#     print(f"Sensor: {sensor_name}")
#     for method, diff in rotation_difference[sensor_name].items():
#         print(f"  Method: {method}, Difference: {diff}")
#
#
#
# joint_orientations = get_joint_orientations_from_segment_orientations(segment_rotation_matrices)

#
# for sensor_name in sensor_rotation_matrices:
#     for method in sensor_rotation_matrices[sensor_name]:
#         sensor_rotation_matrices[sensor_name][method] = sensor_rotation_matrices[sensor_name][method][slice1]
#         segment_rotation_matrices[sensor_name][method] = segment_rotation_matrices[sensor_name][method][slice2]
#     # Print the rotation matrices for each sensor and method
# plt.figure()
# flm0_txt = [rot[0, 0] for rot in sensor_rotation_matrices[segment][filter]]
# flm0_sto = [rot[0, 0] for rot in segment_rotation_matrices[segment][filter]]
# plt.plot(flm0_txt, label='.txt index')
# plt.plot(flm0_sto, label='.sto index')
# plt.show()

print("done")