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
METHODS = ['Mag Adapt']#'Marker', 'EKF', 'Madgwick (Al Borno)', 'Never Project', 'Cascade', 'Unprojected', 'Mag Free']

plate_trials_by_method: Dict[str, List['PlateTrial']] = {}
world_traces_by_method: Dict[str, Dict[str, 'WorldTrace']] = {}

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
            plt.figure(figsize=(10, 6))
            parent_plate = next((p for p in plate_trials_by_method['Marker'] if p.name.__contains__(parent)), None)
            child_plate = next((p for p in plate_trials_by_method['Marker'] if p.name.__contains__(child)), None)
            if not parent_plate or not child_plate:
                continue
            
            for method in METHODS:
                parent_plate_method = next((p for p in plate_trials_by_method[method] if p.name.__contains__(parent)), None)
                child_plate_method = next((p for p in plate_trials_by_method[method] if p.name.__contains__(child)), None)
                if not parent_plate_method or not child_plate_method:
                    continue
                parent_joint_center_offset, child_joint_center_offset, error = parent_plate.world_trace.get_joint_center(child_plate.world_trace)
                
                if np.mean(np.linalg.norm(error, axis=1)) > 0.05:
                    print(f"Warning: High joint center error ({np.mean(np.linalg.norm(error, axis=1))} m) between "
                        f"{parent_plate_method.name} and {child_plate_method.name}. Check marker placement.")

                parent_plate_method.imu_trace = parent_plate_method.project_imu_trace(parent_joint_center_offset)
                child_plate_method.imu_trace = child_plate_method.project_imu_trace(child_joint_center_offset)

                da_parent = np.diff(parent_plate_method.imu_trace.acc, axis=0) + np.cross(parent_plate_method.imu_trace.gyro[1:], parent_plate_method.imu_trace.acc[1:])
                da_child = np.diff(child_plate_method.imu_trace.acc, axis=0) + np.cross(child_plate_method.imu_trace.gyro[1:], child_plate_method.imu_trace.acc[1:])
                o_parent = np.cross(parent_plate_method.imu_trace.acc[1:], da_parent)
                o_child = np.cross(child_plate_method.imu_trace.acc[1:], da_child)
                observability_metric = np.minimum(np.linalg.norm(o_parent, axis=1),
                                                np.linalg.norm(o_child, axis=1))
                observability_metric = np.concatenate(([0.0], observability_metric))

                parent_plate_other = next((p for p in plate_trials_by_method[method] if p.name.__contains__(parent)), None)
                child_plate_other = next((p for p in plate_trials_by_method[method] if p.name.__contains__(child)), None)
                if not parent_plate_other or not child_plate_other:
                    continue

                joint_rots = [r_wp.T @ r_wc for r_wp, r_wc in zip(parent_plate.world_trace.rotations, child_plate.world_trace.rotations)]
                joint_rots_other = [r_wp.T @ r_wc for r_wp, r_wc in zip(parent_plate_other.world_trace.rotations, child_plate_other.world_trace.rotations)]

                joint_rots = Rotation.from_matrix(joint_rots)
                joint_rots_other = Rotation.from_matrix(joint_rots_other)

                joint_angle_diff = joint_rots.inv() * joint_rots_other

                plt.plot(joint_angle_diff.magnitude() * 4000, label=f'{method} - {joint}')
                plt.plot(observability_metric, label=f'Observability - {joint}', linestyle='--')

            plt.tight_layout()
            plt.legend()
            plt.show()
