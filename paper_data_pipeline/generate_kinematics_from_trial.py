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
                                    condition: str = 'Never Project') -> Tuple[float, List[str]]:
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
    num_frames = num_frames if num_frames > 0 else len(plate_trials[0])
    plate_trials = [plate[:num_frames] for plate in plate_trials]
    timestamps = plate_trials[0].imu_trace.timestamps
    segment_orientations = {}

    if condition == 'marker':
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
                                               condition: str = 'never project') -> List[np.ndarray]:
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
    if condition != 'unprojected':
        parent_joint_center_offset, child_joint_center_offset, error = parent_trial.world_trace.get_joint_center(
            child_trial.world_trace)

        parent_trial.imu_trace = parent_trial.project_imu_trace(parent_joint_center_offset)
        child_trial.imu_trace = child_trial.project_imu_trace(child_joint_center_offset)

        if condition == 'mag free':
            parent_trial.imu_trace.mag = [np.zeros(3)] * len(parent_trial)
            child_trial.imu_trace.mag = [np.zeros(3)] * len(child_trial)

    for t in range(len(parent_trial)):
        joint_filter.update(parent_trial.imu_trace.gyro[t], child_trial.imu_trace.gyro[t],
                            parent_trial.imu_trace.acc[t], child_trial.imu_trace.acc[t],
                            parent_trial.imu_trace.mag[t], child_trial.imu_trace.mag[t], dt)

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

if __name__ == "__main__":
    # GENERATING STO FILES
    num_frames = -1  # Use -1 to indicate all frames

    for subject_num in ['06', '07', '08', '09', '10', '11']:
        for activity in ['walking', 'complexTasks']:
            print(f"-------Processing Subject {subject_num}, Activity {activity}...--------")
            # Load the plate trials for the current subject and activity
            try:
                plate_trials = PlateTrial.load_trial_from_folder(
                    f"/Users/six/projects/work/MAJIC_mocap/data/ODay_Data/Subject{subject_num}/{activity}",
                    align_plate_trials=True
                )

                for condition in ['marker', 'unprojected', 'mag free', 'never project']:
                    output_dir = f"/Users/six/projects/work/MAJIC_mocap/data/ODay_Data/Subject{subject_num}/{activity}/IMU/" + condition
                    if condition == 'marker':
                        output_dir = f"/Users/six/projects/work/MAJIC_mocap/data/ODay_Data/Subject{subject_num}/{activity}/Mocap/"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # _generate_orientation_sto_file_(output_dir,
                    #                                 plate_trials,
                    #                                 num_frames, condition)
            except Exception as e:
                print(f"Failed to process Subject {subject_num}, Activity {activity}: {e}")
                continue
