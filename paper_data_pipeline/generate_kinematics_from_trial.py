import os
from typing import List, Tuple, Dict, Any
import numpy as np
import nimblephysics as nimble
from toolchest.PlateTrial import PlateTrial
from RelativeFilter import RelativeFilter

JOINT_SEGMENT_DICT = {'R_Hip': ('pelvis_imu', 'femur_r_imu'),
                      'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
                      'R_Ankle': ('tibia_r_imu', 'calcn_r_imu'),
                      'L_Hip': ('pelvis_imu', 'femur_l_imu'),
                      'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
                      'L_Ankle': ('tibia_l_imu', 'calcn_l_imu'),
                      'Lumbar': ('pelvis_imu', 'torso_imu'),
                      }

def _generate_orientation_sto_file_(output_directory: str,
                                    plate_trials: List[PlateTrial],
                                    num_frames: int,
                                    condition: str = 'Cascade') -> Tuple[float, List[str]]:
    """
    Generates a .sto file containing segment orientations from IMU data.

    Args:
        output_directory (str): Directory to save the .sto file.
        trc_name (str): Name of the TRC file.
        num_frames (int): Number of frames to include.
        joints_to_include (List[str]): Joints to include in the analysis.
        condition (str): Filter condition (Cascade, EKF, etc.).

    Returns:
        Tuple[float, List[str]]: Returns the final time and the list of segment names.
    """
    # Load the IMU data
    # plate_trials = PlateTrial.load_cheeseburger_trial_from_folder(output_directory, trc_name)
    num_frames = num_frames if num_frames > 0 else len(plate_trials[0])
    plate_trials = [plate[:num_frames] for plate in plate_trials]
    timestamps = plate_trials[0].imu_trace.timestamps
    segment_orientations = {}

    if condition == 'Marker':
        for joint, (parent, child) in JOINT_SEGMENT_DICT.items():
            parent_plate = next((p for p in plate_trials if p.name.__contains__(parent)), None)
            child_plate = next((p for p in plate_trials if p.name.__contains__(child)), None)
            if not parent_plate or not child_plate:
                continue

            segment_orientations[parent_plate.name] = parent_plate.world_trace.rotations[:num_frames]
            segment_orientations[child_plate.name] = child_plate.world_trace.rotations[:num_frames]

    else:
        # Estimate joint angles using RelativeFilter
        for joint_name, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
            # Find the two relevant plate trials
            parent_trial = next((p for p in plate_trials if p.name.__contains__(parent_name)), None)
            child_trial = next((p for p in plate_trials if p.name.__contains__(child_name)), None)
            if not parent_trial or not child_trial:
                continue
            print(f'Processing {joint_name} between {parent_name} and {child_name}...')

            joint_orientations = _get_joint_orientations_from_plate_trials_(parent_trial, child_trial, condition)

            # Form the segment orientations
            if parent_trial.name not in segment_orientations:
                segment_orientations[parent_trial.name] = parent_trial.world_trace.rotations[:num_frames]

            segment_orientations[child_trial.name] = [R_wp @ R_pc for R_wp, R_pc in
                                                      zip(segment_orientations[parent_trial.name], joint_orientations)]

    output_path = os.path.join(output_directory,
                               f'walking_orientations.sto' if 'walking' in output_directory else 'complexTasks_orientations.sto')
    _export_to_sto_(output_path, timestamps, segment_orientations)
    return timestamps[-1], list(segment_orientations.keys())


def _get_joint_orientations_from_plate_trials_(parent_trial: PlateTrial,
                                               child_trial: PlateTrial,
                                               condition: str = 'Never Project') -> List[np.ndarray]:
    """
    Estimates joint orientations between parent and child trials using specified filter conditions.

    Args:
        parent_trial (PlateTrial): Parent trial data.
        child_trial (PlateTrial): Child trial data.
        condition (str): Filter condition for the joint orientations.

    Returns:
        List[np.ndarray]: A list of joint orientation matrices.
    """
    # Create the filter structure
    joint_filter = RelativeFilter()
    joint_filter.set_qs(nimble.math.expToQuat(nimble.math.logMap(parent_trial.world_trace.rotations[0])),
                        nimble.math.expToQuat(nimble.math.logMap(child_trial.world_trace.rotations[0])))
    dt = parent_trial.imu_trace.timestamps[1] - parent_trial.imu_trace.timestamps[0]
    R_pc = []

    # If we're going to need some projected information, we should generate it now.
    if condition != 'Unprojected':
        parent_joint_center_offset, child_joint_center_offset, error = parent_trial.world_trace.get_joint_center(
            child_trial.world_trace)

        parent_jc_imu_trace = parent_trial.project_imu_trace(parent_joint_center_offset)
        child_jc_imu_trace = child_trial.project_imu_trace(child_joint_center_offset)

    # Also generate the observability metric in advance if we need it
    if condition == 'Cascade':
        da_parent = np.diff(parent_jc_imu_trace.acc, axis=0)
        da_child = np.diff(child_jc_imu_trace.acc, axis=0)
        o_parent = np.cross(parent_jc_imu_trace.acc[:-1], da_parent) / np.linalg.norm(parent_jc_imu_trace.acc[:-1],
                                                                                      axis=1)[:, None]
        o_child = np.cross(child_jc_imu_trace.acc[:-1], da_child) / np.linalg.norm(child_jc_imu_trace.acc[:-1], axis=1)[
                                                                    :, None]

        observability_metric = np.minimum(np.linalg.norm(o_parent, axis=1),
                                          np.linalg.norm(o_child, axis=1))
        observability_metric = np.concatenate(([observability_metric[0]], observability_metric))

    for t in range(len(parent_trial)):
        if condition == 'Unprojected':
            joint_filter.update(parent_trial.imu_trace.gyro[t], child_trial.imu_trace.gyro[t],
                                parent_trial.imu_trace.acc[t], child_trial.imu_trace.acc[t],
                                parent_trial.imu_trace.mag[t], child_trial.imu_trace.mag[t], dt)

        elif condition == 'Mag Free':
            joint_filter.update(parent_jc_imu_trace.gyro[t], child_jc_imu_trace.gyro[t],
                                parent_jc_imu_trace.acc[t], child_jc_imu_trace.acc[t],
                                np.zeros(3), np.zeros(3), dt)

        elif condition == 'Always Project':
            joint_filter.update(parent_jc_imu_trace.gyro[t], child_jc_imu_trace.gyro[t],
                                parent_jc_imu_trace.acc[t], child_jc_imu_trace.acc[t],
                                parent_jc_imu_trace.mag[t], child_jc_imu_trace.mag[t], dt)

        else:
            unproj_parent_mag = (parent_trial.imu_trace.mag[t] + parent_trial.second_imu_trace.mag[t]) / 2 if \
                parent_trial.second_imu_trace is not None else parent_trial.imu_trace.mag[t]
            unproj_child_mag = (child_trial.imu_trace.mag[t] + child_trial.second_imu_trace.mag[t]) / 2 if \
                child_trial.second_imu_trace is not None else child_trial.imu_trace.mag[t]

            mag_pairs = [
                (unproj_parent_mag, unproj_child_mag),
                # (unproj_parent_mag, child_jc_imu_trace.mag[t]),
                # (parent_jc_imu_trace.mag[t], unproj_child_mag),
                (parent_jc_imu_trace.mag[t], child_jc_imu_trace.mag[t])
            ]

            # Find the pair with the smallest difference in norm
            closest_parent_mag, closest_child_mag = min(
                mag_pairs,
                key=lambda pair: abs(np.linalg.norm(pair[0]) - np.linalg.norm(pair[1]))
            )

            if condition == 'Never Project':
                joint_filter.update(parent_jc_imu_trace.gyro[t], child_jc_imu_trace.gyro[t],
                                    parent_jc_imu_trace.acc[t], child_jc_imu_trace.acc[t],
                                    unproj_parent_mag, unproj_child_mag, dt)

            elif condition == 'Closest Mag Pair':
                joint_filter.update(parent_jc_imu_trace.gyro[t], child_jc_imu_trace.gyro[t],
                                    parent_jc_imu_trace.acc[t], child_jc_imu_trace.acc[t],
                                    closest_parent_mag, closest_child_mag, dt)

            elif condition == 'Cascade':
                if observability_metric[t] > OBSERVABILITY_THRESHOLD:
                    joint_filter.update(parent_jc_imu_trace.gyro[t], child_jc_imu_trace.gyro[t],
                                        parent_jc_imu_trace.acc[t], child_jc_imu_trace.acc[t],
                                        np.zeros(3), np.zeros(3), dt)
                else:
                    joint_filter.update(parent_jc_imu_trace.gyro[t], child_jc_imu_trace.gyro[t],
                                        parent_jc_imu_trace.acc[t], child_jc_imu_trace.acc[t],
                                        closest_parent_mag, closest_child_mag, dt)

        # Store the joint orientation
        R_pc.append(joint_filter.get_R_pc())

    return R_pc


def _export_to_sto_(filename,
                    timestamps,
                    segment_orientations,
                    datatype="Quaternion",
                    version=3,
                    opensim_version="4.2"):
    """
        Exports segment orientations into a .sto file format.

        Args:
            filename (str): File path to save the .sto file.
            timestamps (List[float]): List of timestamps for the orientation data.
            segment_orientations (Dict[str, List[np.ndarray]]): Segment orientations for each body segment.
            datatype (str): Data type for the .sto file.
            version (int): STO file version.
            opensim_version (str): OpenSim version to be added in the file header.
    """
    datarate = 1 / np.mean(np.diff(timestamps))

    # Format the Dict of Lists into list of rows for the STO file
    headers = list(segment_orientations.keys())
    # Add time to the front of the headers
    headers.insert(0, 'time')

    data = []
    for i, timestamp in enumerate(timestamps):
        row = [str(timestamp)]
        for segment_name in segment_orientations:
            quaternion_str = str(
                nimble.math.expToQuat(nimble.math.logMap(segment_orientations[segment_name][i])).wxyz())
            quaternion_str = ",".join(quaternion_str.strip('[]').split())
            row.append(quaternion_str)
        data.append(row)

    with open(filename, 'w') as file:
        # Write the header
        file.write(f"DataRate={datarate}\n")
        file.write(f"DataType={datatype}\n")
        file.write(f"version={version}\n")
        file.write(f"OpenSimVersion={opensim_version}\n")
        file.write("endheader\n")

        # Write the column headers
        file.write("\t".join(headers) + "\n")

        # Write the data
        for row in data:
            file.write("\t".join(row) + "\n")

def _read_mot_file_(file_path) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    Reads a .mot file and parses its header and data.

    Args:
        file_path (str): Path to the .mot file.

    Returns:
        Tuple[Dict[str, Any], Dict[str, List[float]]]: Returns the header information and data as a dictionary.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

        # Parse header
        header_info = {}
        i = 0
        while lines[i].strip() != "endheader":
            if "=" in lines[i]:
                key, value = lines[i].strip().split("=")
                header_info[key] = value
            i += 1

        # Skip the "endheader" line
        i += 1

        # Parse column headers
        headers = lines[i].strip().split()
        i += 1

        # Parse data rows
        data = []
        for line in lines[i:]:
            values = list(map(float, line.strip().split()))
            data.append(values)

        # Create a dictionary where each header maps to its corresponding data column
        data_dict = {headers[j]: [row[j] for row in data] for j in range(len(headers))}

    return header_info, data_dict

# GENERATING STO FILES
activity = 'walking'
subject_num = '06'

plate_trials = PlateTrial.load_trial_from_folder(
    f"/Users/six/projects/work/MAJIC_mocap/data/ODay_Data/Subject{subject_num}/{activity}",
    align_plate_trials=True
)

for condition in ['Marker', 'Unprojected', 'Mag Free', 'Never Project']:
    output_dir = f"/Users/six/projects/work/MAJIC_mocap/data/ODay_Data/Subject{subject_num}/{activity}/IMU/" + condition
    if condition == 'Marker':
        output_dir = f"/Users/six/projects/work/MAJIC_mocap/data/ODay_Data/Subject{subject_num}/{activity}/Mocap/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    _generate_orientation_sto_file_(output_dir,
                                    plate_trials,
                                    100, condition)
