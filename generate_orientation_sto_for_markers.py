from typing import List, Tuple
import os
import numpy as np
from src.toolchest.PlateTrial import PlateTrial  # Assuming PlateTrial is defined in plate_trial.py
import nimblephysics as nimble

JOINT_SEGMENT_DICT = {'R_Hip': ('pelvis_imu', 'femur_r_imu'),
                      'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
                      'R_Ankle': ('tibia_r_imu', 'calcn_r_imu'),
                      'L_Hip': ('pelvis_imu', 'femur_l_imu'),
                      'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
                      'L_Ankle': ('tibia_l_imu', 'calcn_l_imu'),
                      'Lumbar': ('pelvis_imu', 'torso_imu'),
                      }

def _generate_orientation_sto_file_(output_directory: str,
                                    plate_trials: List[PlateTrial]) -> Tuple[float, List[str]]:
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
    num_frames = len(plate_trials[0])
    plate_trials = [plate[:num_frames] for plate in plate_trials if plate is not None]
    timestamps = plate_trials[0].imu_trace.timestamps
    segment_orientations = {}

    for joint, (parent, child) in JOINT_SEGMENT_DICT.items():
        parent_plate = next((p for p in plate_trials if p.name.__contains__(parent)), None)
        child_plate = next((p for p in plate_trials if p.name.__contains__(child)), None)
        if not parent_plate or not child_plate:
            continue

        segment_orientations[parent_plate.name] = parent_plate.world_trace.rotations[:num_frames]
        segment_orientations[child_plate.name] = child_plate.world_trace.rotations[:num_frames]

    # else:
    #     # Estimate joint angles using RelativeFilter
    #     for joint_name, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
    #         # Find the two relevant plate trials
    #         parent_trial = next((p for p in plate_trials if p.name.__contains__(parent_name)), None)
    #         child_trial = next((p for p in plate_trials if p.name.__contains__(child_name)), None)
    #         if not parent_trial or not child_trial:
    #             continue
    #         print(f'Processing {joint_name} between {parent_name} and {child_name}...')
    #
    #         joint_orientations = _get_joint_orientations_from_plate_trials_(parent_trial, child_trial, condition)
    #
    #         # Form the segment orientations
    #         if parent_trial.name not in segment_orientations:
    #             segment_orientations[parent_trial.name] = [np.eye(3) for _ in range(len(timestamps))]
    #
    #         segment_orientations[child_trial.name] = [R_wp @ R_pc for R_wp, R_pc in
    #                                                   zip(segment_orientations[parent_trial.name], joint_orientations)]

    output_path = os.path.join(output_directory, f'walking_orientations.sto')
    _export_to_sto_(output_path, timestamps, segment_orientations)
    return timestamps[-1], list(segment_orientations.keys())

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
            # nimble.math.expToQuat returns a list like [w, x, y, z]. We need to convert it to a string "w,x,y,z"
            quaternion_str = ",".join(map(str, nimble.math.expToQuat(nimble.math.logMap(segment_orientations[segment_name][i])).wxyz()))
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

plate_trials = PlateTrial.load_trial_from_folder("data/ODay_Data/Subject03/walking", align_plate_trials=True)

output_directory = "data/ODay_Data/Subject03/walking/Mocap"

final_time, segment_names = _generate_orientation_sto_file_(output_directory, plate_trials)