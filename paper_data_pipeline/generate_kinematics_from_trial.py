import os
import subprocess
from typing import List, Tuple, Dict, Any
import numpy as np
import nimblephysics as nimble
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from paper_data_pipeline.marker_gap_filler import generate_trc_from_c3d

from toolchest.AHRSFilter import AHRSFilter
from toolchest.PlateTrial import PlateTrial
from RelativeFilter import RelativeFilter

JOINT_SEGMENT_DICT = {
    'Hip': ('Pelvis', 'Femur'),
    'Knee': ('Femur', 'Shank'),
    'Ankle': ('Shank', 'Foot'),
    'Lumbar_1': ('Pelvis', 'Torso'),
    'Lumbar_2': ('Pelvis', 'Sternum'),
    'Shoulder_1': ('Torso', 'Upper_Arm'),
    'Shoulder_2': ('Sternum', 'Upper_Arm'),
    'Elbow': ('Upper_Arm', 'Lower_Arm')
}

OBSERVABILITY_THRESHOLD = 0.2


def load_kinematics_from_trial_folder(trial_directory: str,
                                      filter_type: str,
                                      regenerate: bool = False,
                                      override_weights: dict = {},
                                      num_frames: int = -1,
                                      joints_to_include: List[str] = None) -> \
        Tuple[nimble.dynamics.Skeleton, nimble.biomechanics.OpenSimMot, Any]:
    """
    Loads the kinematics from the trial folder, generates or uses pre-existing inverse kinematics (IK) results.

    Args:
        trial_directory (str): Path to the trial folder.
        filter_type (str): Type of filter to use for IMU data.
        regenerate (bool): Whether to regenerate IK results. Defaults to False.
        override_weights (dict): Weights to override for specific segments.
        num_frames (int): Number of frames to use. Defaults to -1 (use all frames).
        joints_to_include (List[str]): List of joint names to include in the processing.

    Returns:
        Tuple[nimble.dynamics.Skeleton, Dict[str, List[float]]]: Returns the skeleton and kinematics.
    """
    if joints_to_include is None:
        joints_to_include = JOINT_SEGMENT_DICT.keys()

    trial_directory = os.path.abspath(trial_directory)

    # Check if IK results are already in the directory, unless the user wants to override
    ik_results = os.path.join(trial_directory, filter_type.lower() + '_ik.mot')
    if os.path.exists(ik_results) and (not regenerate or 'OMC' in filter_type):
        print(f"Found existing IK results at {ik_results}. Skipping IK generation.")

    else:
        # Check if the segment orientations are present, unless the user wants to override
        segment_orientations_path = os.path.join(trial_directory, filter_type.lower() + '_segment_orientations.sto')

        if os.path.exists(segment_orientations_path) and not regenerate:
            print(f"Found existing segment orientations file at {segment_orientations_path}.")

            # load the orientation file to get the final time
            with open(segment_orientations_path, 'r') as f:
                lines = f.readlines()
                imu_names = lines[5].split()[1:]
                final_time = float(lines[-1].split()[0])
        else:
            print(f"Generating {segment_orientations_path} now...")

            trc_file = next((f for f in os.listdir(trial_directory) if f.endswith('.trc') and 'traces' not in f), None)
            if not trc_file:
                print(f"Could not find a TRC file in {trial_directory}. Generating now...")
                c3d_file = next((f for f in os.listdir(trial_directory) if f.endswith('.c3d')), None)
                subject_dir = os.path.abspath(os.path.join(trial_directory, os.pardir))
                mapping_path = os.path.join(subject_dir, 'marker_mapping.json')
                generate_trc_from_c3d(c3d_file, mapping_path, trc_file)
            final_time, imu_names = _generate_orientation_sto_file_(trial_directory, trc_file, num_frames,
                                                                    joints_to_include, filter_type)

        # Get all the paths and generate the set-up file
        weights = {imu_name: 1.0 for imu_name in imu_names}
        for imu_name, weight in override_weights.items():
            try:
                weights[imu_name] = weight
            except KeyError:
                print(f"Could not find {imu_name} in the orientations file. Skipping...")

        xml_file_path = os.path.join(trial_directory, filter_type + '_ik_setup.xml')
        ik_output_path = os.path.join(trial_directory, filter_type + '_ik')
        model_file_path = os.path.join(os.path.abspath(os.path.dirname(trial_directory)), 'scaled_with_imus.osim')

        _generate_ik_setup_file_(xml_file_path, trial_directory, final_time, ik_output_path,
                                 segment_orientations_path, model_file_path, weights)

        # Command to run
        command = ['opensense', '-IK', xml_file_path]

        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Output the result
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    model_file_path = os.path.join(os.path.abspath(os.path.dirname(trial_directory)), 'scaled_with_imus.osim')
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Could not find the model file at {model_file_path}")

    skeleton = nimble.biomechanics.OpenSimParser.parseOsim(model_file_path).skeleton
    kinematics = nimble.biomechanics.OpenSimParser.loadMot(skeleton, ik_results)
    try:
        errors = _read_mot_file_(os.path.join(trial_directory, filter_type.lower() + '_ik_orientationErrors.sto'))[1]
    except FileNotFoundError:
        print("Could not find the orientation errors file. Skipping...")
        errors = None
    return skeleton, kinematics, errors


def _generate_orientation_sto_file_(output_directory: str,
                                    trc_name: str,
                                    num_frames: int,
                                    joints_to_include: List[str],
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
    plate_trials = PlateTrial.load_cheeseburger_trial_from_folder(output_directory, trc_name)
    num_frames = num_frames if num_frames > 0 else len(plate_trials[0])
    plate_trials = [plate[:num_frames] for plate in plate_trials]
    timestamps = plate_trials[0].imu_trace.timestamps
    segment_orientations = {}

    if condition == 'Marker':
        for joint, (parent, child) in JOINT_SEGMENT_DICT.items():
            if joint not in joints_to_include:
                continue
            parent_plate = next((p for p in plate_trials if p.name.__contains__(parent)), None)
            child_plate = next((p for p in plate_trials if p.name.__contains__(child)), None)
            if not parent_plate or not child_plate:
                continue

            segment_orientations[parent_plate.name] = parent_plate.world_trace.rotations[:num_frames]
            segment_orientations[child_plate.name] = child_plate.world_trace.rotations[:num_frames]

    elif condition == 'EKF':
        gravity = np.array([0., -9.81, 0.])
        magnetic_field = np.mean([plate.estimate_world_magnetic_field() for plate in plate_trials], axis=0)
        for plate in plate_trials:
            imu_trace = plate.imu_trace
            ahrs_filter = AHRSFilter('EKF', world_reference_acc=-gravity, world_reference_mag=magnetic_field)
            dt = imu_trace.timestamps[1] - imu_trace.timestamps[0]
            segment_orientations[plate.name] = []
            for t in range(len(timestamps)):
                ahrs_filter.update(dt=dt, acc=imu_trace.acc[t], gyro=imu_trace.gyro[t], mag=imu_trace.mag[t])
                segment_orientations[plate.name].append(ahrs_filter.get_last_R())

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
                segment_orientations[parent_trial.name] = [np.eye(3) for _ in range(len(timestamps))]

            segment_orientations[child_trial.name] = [R_wp @ R_pc for R_wp, R_pc in
                                                      zip(segment_orientations[parent_trial.name], joint_orientations)]

    output_path = os.path.join(output_directory, f'{condition.lower()}_segment_orientations.sto')
    _export_to_sto_(output_path, timestamps, segment_orientations)
    return timestamps[-1], list(segment_orientations.keys())


def _get_joint_orientations_from_plate_trials_(parent_trial: PlateTrial,
                                               child_trial: PlateTrial,
                                               condition: str = 'Cascade') -> List[np.ndarray]:
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
            unproj_parent_mag = (parent_trial.imu_trace.mag[t] + parent_trial.second_imu_trace.mag[t]) / 2
            unproj_child_mag = (child_trial.imu_trace.mag[t] + child_trial.second_imu_trace.mag[t]) / 2

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


def _generate_ik_setup_file_(xml_file_path,
                             results_directory,
                             final_time,
                             ik_output_path,
                             orientation_input_file_path,
                             model_file_path,
                             weights: Dict[str, float]):
    """
    Generates an XML setup file for the OpenSim Inverse Kinematics (IK) tool.

    Args:
        xml_file_path (str): Path to save the XML file.
        results_directory (str): Directory where results should be saved.
        final_time (float): Final time of the trial.
        ik_output_path (str): Path to save the IK output file.
        orientation_input_file_path (str): Path to the segment orientations input file.
        model_file_path (str): Path to the model file (.osim).
        weights (Dict[str, float]): Weights for each body segment.
    """
    # Root element
    root = ET.Element("OpenSimDocument", Version="40000")

    # IMUInverseKinematicsTool element
    ik_tool = ET.SubElement(root, "IMUInverseKinematicsTool")

    # Add results_directory
    results_dir = ET.SubElement(ik_tool, "results_directory")
    results_dir.text = os.path.abspath(results_directory)

    # Add model_file
    model_file = ET.SubElement(ik_tool, "model_file")
    model_file.text = os.path.abspath(model_file_path)

    # Add time_range
    time_range = ET.SubElement(ik_tool, "time_range")
    time_range.text = "0 " + str(final_time)

    # Add output_motion_file
    output_motion = ET.SubElement(ik_tool, "output_motion_file")
    output_motion.text = os.path.abspath(ik_output_path)

    # Add report_errors
    report_errors = ET.SubElement(ik_tool, "report_errors")
    report_errors.text = "true"

    # Add sensor_to_opensim_rotations
    sensor_rotations = ET.SubElement(ik_tool, "sensor_to_opensim_rotations")
    sensor_rotations.text = "0 0 0"

    # Add orientations_file
    orientations_file = ET.SubElement(ik_tool, "orientations_file")
    orientations_file.text = orientation_input_file_path

    # OrientationWeightSet element
    orientation_weights = ET.SubElement(ik_tool, "OrientationWeightSet", name="orientation_weights")
    objects = ET.SubElement(orientation_weights, "objects")

    for segment_name, segment_weight in weights.items():
        # Add OrientationWeight for segment
        segment = ET.SubElement(objects, "OrientationWeight", name=segment_name)
        weight = ET.SubElement(segment, "weight")
        weight.text = str(segment_weight)

    # Convert to string and format using minidom for pretty-printing
    xml_string = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(xml_string)
    pretty_xml_as_string = parsed.toprettyxml(indent="  ")

    # Write the formatted XML to file
    with open(xml_file_path, "w") as file:
        file.write(pretty_xml_as_string)
