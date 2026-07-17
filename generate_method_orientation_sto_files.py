from src.toolchest.dataset_loaders import AlBornoLoader, SkovLoader
import os
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation
from src.toolchest.PlateTrial import PlateTrial
from src.RelativeFilterPlus import RelativeFilter
from concurrent.futures import ProcessPoolExecutor

JOINT_SEGMENT_DICT = {'Lumbar': ('pelvis_imu', 'torso_imu'),
                      'R_Hip': ('pelvis_imu', 'femur_r_imu'),
                      'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
                      'R_Ankle': ('tibia_r_imu', 'calcn_r_imu'),
                      'L_Hip': ('pelvis_imu', 'femur_l_imu'),
                      'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
                      'L_Ankle': ('tibia_l_imu', 'calcn_l_imu'),
                      }

SUBJECTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
METHODS = ['Mag On', 'Mag Adapt', 'Mag Off', 'Unprojected', 'EKF']
TRIALS = ['walking', 'complexTasks']

def _generate_orientation_sto_file_(output_directory: str,
                                    plate_trials: List[PlateTrial],
                                    num_frames: int,
                                    condition: str = 'Never Project') -> Tuple[float, List[str]]:
    """
    Generates a .sto file containing segment orientations from IMU data.

    Args:
        output_directory (str): Directory to save the .sto file.
        plate_trials (List[PlateTrial]): List of loaded plate trials.
        num_frames (int): Number of frames to include.
        condition (str): Filter condition (marker, mag free, unprojected, etc.).

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
    elif condition == 'ekf':
        segment_orientations = {}
        
        # Set up a single "ground" plate to represent the global frame for all EKF runs
        if plate_trials:
            base_plate = plate_trials[0]
            expected_gravity = np.array([0.0, 1.0, 0.0]) # Expected gravity in Z-axis
            
            # Precompute expected magnetic field as the median of all global magnetic field readings
            all_global_mags = []
            for plate in plate_trials:
                # Transform mag from sensor frame to global frame: v_global = R_sw @ v_sensor
                mag_global = np.einsum('tij,tj->ti', plate.world_trace.rotations, plate.imu_trace.mag)
                all_global_mags.append(mag_global)
            
            all_global_mags_concat = np.concatenate(all_global_mags, axis=0)
            expected_mag = np.median(all_global_mags_concat, axis=0)
            
            ground_plate = base_plate.copy()
            ground_plate.name = "ground_virtual_parent"
            ground_plate.world_trace.rotations = np.tile(np.eye(3), (len(base_plate), 1, 1))
            ground_plate.imu_trace.gyro = np.zeros_like(base_plate.imu_trace.gyro)
            ground_plate.imu_trace.acc = np.tile(expected_gravity, (len(base_plate), 1))
            ground_plate.imu_trace.mag = np.tile(expected_mag, (len(base_plate), 1))
            
        for plate in plate_trials:
            R_ws = _get_joint_orientations_from_plate_trials_(
                ground_plate, plate, condition='ekf',
                gyro_std_parent=1e-4, acc_std_parent=1e-4, mag_std_parent=1e-4,
                project_imu=False, warmup_steps=2000
            )
            segment_orientations[plate.name] = R_ws
    else:
        # Estimate joint angles using RelativeFilter
        for joint_name, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
            # Find the two relevant plate trials
            parent_trial = next((p for p in plate_trials if p.name.__contains__(parent_name)), None)
            child_trial = next((p for p in plate_trials if p.name.__contains__(child_name)), None)
            if not parent_trial or not child_trial:
                continue
            # print(f'Processing {joint_name} between {parent_name} and {child_name}...')

            joint_orientations = _get_joint_orientations_from_plate_trials_(parent_trial, child_trial, condition)

            # Form the segment orientations
            if parent_trial.name not in segment_orientations:
                segment_orientations[parent_trial.name] = parent_trial.world_trace.rotations[:num_frames]

            segment_orientations[child_trial.name] = [R_wp @ R_pc for R_wp, R_pc in
                                                      zip(segment_orientations[parent_trial.name], joint_orientations)]

    output_path = os.path.join(output_directory,
                               f'walking_orientations_{condition.lower().replace(" ", "_")}.sto' if 'walking' in output_directory else f'complexTasks_orientations_{condition.lower().replace(" ", "_")}.sto')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    _export_to_sto_(output_path, timestamps, segment_orientations)
    return timestamps[-1], list(segment_orientations.keys())

def _get_joint_orientations_from_plate_trials_(parent_trial: PlateTrial,
                                               child_trial: PlateTrial,
                                               condition: str = 'mag on',
                                               gyro_std_parent: float = np.sqrt(0.01),
                                               acc_std_parent: float = np.sqrt(0.05),
                                               mag_std_parent: float = np.sqrt(0.05),
                                               gyro_std_child: float = np.sqrt(0.01),
                                               acc_std_child: float = np.sqrt(0.05),
                                               mag_std_child: float = np.sqrt(0.05),
                                               mag_adapt_threshold: float = 150.0,
                                               project_imu: bool = True,
                                               precomputed_observability_metric: np.ndarray = None,
                                               warmup_steps: int = 0) -> List[np.ndarray]:
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
    parent_trial = parent_trial.copy()
    child_trial = child_trial.copy()
    joint_filter = RelativeFilter(
        gyro_std_parent=np.ones(3) * gyro_std_parent,
        gyro_std_child=np.ones(3) * gyro_std_child,
        vector_sensor_stds_parent=[np.ones(3) * acc_std_parent, np.ones(3) * mag_std_parent],
        vector_sensor_stds_child=[np.ones(3) * acc_std_child, np.ones(3) * mag_std_child]
    )
    joint_filter.set_qs(Rotation.from_matrix(parent_trial.world_trace.rotations[0]), Rotation.from_matrix(child_trial.world_trace.rotations[0]))
    dt = np.mean(parent_trial.imu_trace.timestamps[1:] - parent_trial.imu_trace.timestamps[:-1])
    R_pc = []

    # If we're going to need some projected information, we should generate it now.
    if condition not in ['unprojected', 'ekf']:
        if project_imu:
            parent_joint_center_offset, child_joint_center_offset, error = parent_trial.world_trace.get_joint_center(
                child_trial.world_trace)
            
            if np.mean(np.linalg.norm(error, axis=1)) > 0.05:
                print(f"Warning: High joint center error ({np.mean(np.linalg.norm(error, axis=1))} m) between "
                      f"{parent_trial.name} and {child_trial.name}. Check marker placement.")

            parent_trial.imu_trace = parent_trial.project_imu_trace(parent_joint_center_offset)
            child_trial.imu_trace = child_trial.project_imu_trace(child_joint_center_offset)

        if condition == 'mag adapt':
            if precomputed_observability_metric is None:
                da_parent = np.diff(parent_trial.imu_trace.acc, axis=0) + np.cross(parent_trial.imu_trace.gyro[1:], parent_trial.imu_trace.acc[1:])
                da_child = np.diff(child_trial.imu_trace.acc, axis=0) + np.cross(child_trial.imu_trace.gyro[1:], child_trial.imu_trace.acc[1:])
                o_parent = np.cross(parent_trial.imu_trace.acc[1:], da_parent)
                o_child = np.cross(child_trial.imu_trace.acc[1:], da_child)
                observability_metric = np.minimum(np.linalg.norm(o_parent, axis=1),
                                                np.linalg.norm(o_child, axis=1))
                observability_metric = np.concatenate(([0.0], observability_metric))
            else:
                observability_metric = precomputed_observability_metric

            # Set mag to 0 where observability is high
            threshold = mag_adapt_threshold
            high_indexes = observability_metric > threshold
            parent_trial.imu_trace.mag[high_indexes] = 0.0
            child_trial.imu_trace.mag[high_indexes] = 0.0
        elif condition == 'mag off':
            parent_trial.imu_trace.mag = np.zeros_like(parent_trial.imu_trace.mag)
            child_trial.imu_trace.mag = np.zeros_like(child_trial.imu_trace.mag)
        elif condition == 'mag on':
            pass  # Use mag as is
        else:
            raise ValueError(f"Unknown condition '{condition}' specified for joint orientation estimation.")
    elif condition in ['unprojected', 'ekf']:
        pass  # Use raw IMU data without projection
    else:
        raise ValueError(f"Unknown condition '{condition}' specified for joint orientation estimation.")
        
    if warmup_steps > 0:
        for _ in range(warmup_steps):
            joint_filter.update(parent_trial.imu_trace.gyro[0], child_trial.imu_trace.gyro[0],
                                [parent_trial.imu_trace.acc[0], parent_trial.imu_trace.mag[0]],
                                [child_trial.imu_trace.acc[0], child_trial.imu_trace.mag[0]], dt)

    for t in range(len(parent_trial)):
        joint_filter.update(parent_trial.imu_trace.gyro[t], child_trial.imu_trace.gyro[t],
                            [parent_trial.imu_trace.acc[t], parent_trial.imu_trace.mag[t]],
                            [child_trial.imu_trace.acc[t], child_trial.imu_trace.mag[t]], dt)

        # Store the joint orientation
        R_pc.append(joint_filter.get_R_pc())

    return R_pc

def _export_to_sto_(filename,
                    timestamps,
                    segment_orientations: Dict[str, List[np.ndarray]],
                    datatype="Quaternion",
                    version=3,
                    opensim_version="4.2"):
    """
        Exports segment orientations into a .sto file format.

        Args:
            filename (str): File path to save the .sto file.
            timestamps (List[float]): List of timestamps for the orientation data.
            segment_orientations (Dict[str, List[np.ndarray]]): Segment orientations for each body segment (3x3 matrices).
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
    # Pre-calculate all quaternions for efficiency
    all_segment_quaternions: Dict[str, np.ndarray] = {}
    for segment_name, rotations in segment_orientations.items():
        # Convert List[np.ndarray] (3x3 matrices) to a single Rotation object, then to quat array (N, 4)
        # SciPy returns [x, y, z, w]. OpenSim/STO typically expects [w, x, y, z].
        # The original nimble.math.Quaternion.wxyz() returned [w, x, y, z].
        quats_xyzw = Rotation.from_matrix(rotations).as_quat(canonical=True)
        
        # Reorder to [w, x, y, z] to match the original nimble output format for OpenSim
        quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]
        all_segment_quaternions[segment_name] = quats_wxyz


    for i, timestamp in enumerate(timestamps):
        row = [str(timestamp)]
        for segment_name in segment_orientations:
            quat = all_segment_quaternions[segment_name][i]
            # Format the quaternion string (w,x,y,z) separated by commas
            quaternion_str = ",".join([f"{val:.16f}" for val in quat])
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

def process_subject_activity(subject_num: str, activity: str, num_frames: int):
    print(f"-------Processing Subject {subject_num}, Activity {activity}...--------")
    # Load the plate trials for the current subject and activity
    try:
        subject_activity_folder = os.path.abspath(os.path.join("data", f"Subject{subject_num}", activity))
        plate_trials = SkovLoader(subject_activity_folder).load_plate_trials(align_plate_trials=True)

        print(f"Loaded {len(plate_trials)} plate trials for Subject {subject_num}, {activity}.")
        print(f"Identified segments: {[plate.name for plate in plate_trials]}")

        for condition in METHODS:
            print(f"Generating STO file for Subject {subject_num}, {activity}, condition: {condition}...")
            condition_lower = condition.lower()
            output_dir = subject_activity_folder
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            _generate_orientation_sto_file_(output_dir,
                                            plate_trials,
                                            num_frames, condition_lower)
    except Exception as e:
        print(f"Failed to process Subject {subject_num}, Activity {activity}: {e}")

if __name__ == "__main__":
    # GENERATING STO FILES
    num_frames = -1  # Use -1 to indicate all frames

    tasks = []
    for subject_num in SUBJECTS:
        for activity in TRIALS:
            tasks.append((subject_num, activity, num_frames))

    print(f"Starting parallel generation of STO files for {len(tasks)} tasks...")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_subject_activity, *task) for task in tasks]
        # Wait for completion and raise exception if any task failed
        for future in futures:
            future.result()

