import os
from typing import Dict, List
import pandas as pd
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from src.toolchest.IMUTrace import IMUTrace
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from generate_method_orientation_sto_files import JOINT_SEGMENT_DICT

subject_id = "Subject"
TRIAL_TYPES = ['walking']
METHODS = ['Mag Adapt', 'Mag Off', 'Mag On']#'Marker', 'EKF', 'Madgwick (Al Borno)', 'Never Project', 'Cascade', 'Unprojected', 'Mag Free']

plate_trials_by_method: Dict[str, List['PlateTrial']] = {}
world_traces_by_method: Dict[str, Dict[str, 'WorldTrace']] = {}

def estimate_joint_axis(rot_parent_list, rot_child_list):
    """
    Estimates the mean joint axis of rotation from two lists of rotations.

    Args:
        rot_parent_list (scipy.Rotation): A Rotation object with n rotations
                                          for the parent segment (e.g., thigh).
        rot_child_list (scipy.Rotation): A Rotation object with n rotations
                                         for the child segment (e.g., shank).

    Returns:
        np.ndarray: A (3,) unit vector representing the estimated joint axis
                    in the parent segment's coordinate system.
    """
    
    # --- Step 2: Calculate all relative rotations ---
    # R_relative(t) = R_child(t) * R_parent(t)^-1
    # This calculates the orientation of the child relative to the parent.
    rot_relative = rot_parent_list.inv() * rot_child_list

    # --- Step 3: Formulate the optimization problem ---
    # We want to find the vector v that minimizes sum(||R_rel(t) * v - v||^2)
    # This is equivalent to minimizing sum(|| (R_rel(t) - I) * v ||^2)
    
    # Get all relative rotation matrices as a single (n, 3, 3) array
    all_rel_matrices = rot_relative.as_matrix()
    
    # Create the identity matrix
    I = np.identity(3)
    
    # Create the (n, 3, 3) stack of (R_rel(t) - I) matrices
    # NumPy broadcasting handles the subtraction
    A_stack = all_rel_matrices - I
    
    # Reshape into a single (3*n, 3) matrix A
    n_samples = len(rot_relative)
    A = A_stack.reshape(n_samples * 3, 3)
    
    # --- Step 4: Solve using SVD ---
    # We are solving A*v = 0. The best solution for v is the
    # right-singular vector of A corresponding to the *smallest* singular value.
    #
    # Alternatively, we can find the eigenvector of A.T @ A
    # corresponding to the *smallest* eigenvalue.
    
    # Compute the (3, 3) matrix ATA
    ATA = A.T @ A
    
    # Find its eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(ATA)

    print(f"Eigenvalues: {eigenvalues}")
    
    # The joint axis v is the eigenvector with the *smallest* eigenvalue
    smallest_eigenvalue_index = np.argmin(eigenvalues)
    joint_axis = eigenvectors[:, smallest_eigenvalue_index]
    
    # Ensure the axis is a unit vector (it should be, but good practice)
    return joint_axis / np.linalg.norm(joint_axis)


for subject_num in ['02']:#, '02', '03', '04', '05', '06', '07', '08', '09', '10']:
    for trial_type in TRIAL_TYPES:
        try:
            subject_id = f"Subject{subject_num}"
            data_folder_path = os.path.join("data", subject_id, trial_type)
            data_folder_path = os.path.abspath(data_folder_path)
            plate_trials_by_method['Marker'] = PlateTrial.load_trial_from_folder(data_folder_path)
            imu_traces = {trial.name: trial.imu_trace.copy() for trial in plate_trials_by_method['Marker']}
        except:
            print(f"Couldnt find {trial_type} {subject_id}")
            continue

        min_length = min(len(trial) for trial in plate_trials_by_method['Marker'])
        min_length = min(min_length, 60000)  # Cap at 60000 frames to avoid excessive lengths
        try:
            for method in METHODS:
                if method == 'Marker':
                    continue
                else:
                    print(f"Loading {method} for {subject_id} {trial_type}")
                    if method == 'Madgwick (Al Borno)':
                        world_traces = WorldTrace.load_WorldTraces_from_folder(
                            os.path.abspath(os.path.join("data", subject_id, trial_type, "madgwick (al borno)"))
                        )
                    else:
                        world_traces = WorldTrace.load_from_sto_file(
                            os.path.abspath(os.path.join("data", subject_id, trial_type, f"{trial_type}_orientations_{method.replace(' ', '_').lower()}.sto"))
                        )
                    plate_trials_by_method[method] = PlateTrial.generate_plate_from_traces(
                        imu_traces,
                        world_traces,
                        align_plate_trials=False
                    )
                    min_length = min(min_length, min(len(trial) for trial in plate_trials_by_method[method]))
                    
        except Exception as e:
            print(f"Error loading data for {subject_id}: {e}. Skipping subject.")
            continue

        # --- Trim Trials to Common Length ---
        for method, trials in plate_trials_by_method.items():
            for i, trial in enumerate(trials):
                plate_trials_by_method[method][i] = trial[-min_length:]

        for joint, (parent, child) in JOINT_SEGMENT_DICT.items():
            parent_plate = next((p for p in plate_trials_by_method['Marker'] if p.name.__contains__(parent)), None)
            child_plate = next((p for p in plate_trials_by_method['Marker'] if p.name.__contains__(child)), None)
            if not parent_plate or not child_plate:
                continue
            fig, ax = plt.subplots(3, 1, figsize=(10, 8))
            for method in METHODS:
                parent_plate_method = next((p for p in plate_trials_by_method[method] if p.name.__contains__(parent)), None)
                child_plate_method = next((p for p in plate_trials_by_method[method] if p.name.__contains__(child)), None)
                if not parent_plate_method or not child_plate_method:
                    continue

                parent_plate_other = next((p for p in plate_trials_by_method[method] if p.name.__contains__(parent)), None)
                child_plate_other = next((p for p in plate_trials_by_method[method] if p.name.__contains__(child)), None)
                if not parent_plate_other or not child_plate_other:
                    continue

                joint_rots = [r_wp.T @ r_wc for r_wp, r_wc in zip(parent_plate.world_trace.rotations, child_plate.world_trace.rotations)]
                joint_rots_other = [r_wp.T @ r_wc for r_wp, r_wc in zip(parent_plate_other.world_trace.rotations, child_plate_other.world_trace.rotations)]

                joint_rots = Rotation.from_matrix(joint_rots)
                joint_rots_other = Rotation.from_matrix(joint_rots_other)

                joint_angle_diff = joint_rots.inv() * joint_rots_other

                for i in range(3):
                    ax[i].plot(joint_angle_diff.as_euler('XYZ')[:, i] * (180/np.pi))
                    # ax[i].set_title(f"{joint} Axis {i} Angle Difference (degrees)")
                    # ax[i].set_xlabel("Frame")
                    # ax[i].set_ylabel("Angle Difference (degrees)")
            plt.tight_layout()
            plt.show()
