import os
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt  # <-- Import matplotlib
from src.toolchest.PlateTrial import PlateTrial
from src.RelativeFilterPlus import RelativeFilter
from generate_method_orientation_sto_files import JOINT_SEGMENT_DICT

# Load plate trials
# Estimate jerk by simple finite difference method on the imu trace
# Estimate jerk by finite difference on the world trace
# Estimate jerk by finite difference with cross product on synthetic imu trace without gravity
# Compare jerks

if __name__ == "__main__":
    
    # -----------------------------------------------------------------
    # 1. Load the plate trials
    # -----------------------------------------------------------------
    print("Step 1: Loading plate trials...")
    subject_num = '01'
    activity = 'complexTasks'
    subject_activity_folder = os.path.abspath(os.path.join("data", f"Subject{subject_num}", activity))
    
    try:
        plate_trials = PlateTrial.load_trial_from_folder(
            subject_activity_folder,
            align_plate_trials=True
        )
    except Exception as e:
        print(f"Failed to load data. Make sure your 'data/data/Subject01/walking' folder exists.")
        print(f"Error: {e}")
        exit()
    plt.figure(figsize=(10, 6))
    for joint, (parent, child) in JOINT_SEGMENT_DICT.items():
        parent_trial = next((p for p in plate_trials if p.name.__contains__(parent)), None)
        child_trial = next((p for p in plate_trials if p.name.__contains__(child)), None)
        if not parent_trial or not child_trial:
            print(f"Could not find trials for {parent} or {child}. Skipping joint {joint}.")
            continue
        parent_joint_center_offset, child_joint_center_offset, error = parent_trial.world_trace.get_joint_center(
            child_trial.world_trace)
        
        if np.mean(np.linalg.norm(error, axis=1)) > 0.05:
            print(f"Warning: High joint center error ({np.mean(np.linalg.norm(error, axis=1))} m) between "
                  f"{parent_trial.name} and {child_trial.name}. Check marker placement.")

        parent_trial.imu_trace = parent_trial.project_imu_trace(parent_joint_center_offset)
        child_trial.imu_trace = child_trial.project_imu_trace(child_joint_center_offset)

        da_parent = np.diff(parent_trial.imu_trace.acc, axis=0) + np.cross(parent_trial.imu_trace.gyro[1:], parent_trial.imu_trace.acc[1:])
        da_child = np.diff(child_trial.imu_trace.acc, axis=0) + np.cross(child_trial.imu_trace.gyro[1:], child_trial.imu_trace.acc[1:])
        o_parent = np.cross(parent_trial.imu_trace.acc[1:], da_parent)
        o_child = np.cross(child_trial.imu_trace.acc[1:], da_child)
        observability_metric = np.minimum(np.linalg.norm(o_parent, axis=1),
                                        np.linalg.norm(o_child, axis=1))
        observability_metric = np.concatenate(([0.0], observability_metric))

        plt.plot(parent_trial.imu_trace.timestamps, observability_metric, label=f'Joint {joint}')
        print(f"Joint {joint} observability metric (mean over trial): {np.mean(observability_metric)}")
    plt.legend()
    plt.show()
    # for trial in plate_trials:
    #     print(f"Loaded trial: {trial.name} with {len(trial)} frames.")
    #     raw_imu_trace = trial.imu_trace
    #     no_grav_imu_trace = raw_imu_trace.copy()
    #     no_grav_imu_trace.acc = [a + r.T @ np.array([0, 9.81, 0]) for a, r in zip(raw_imu_trace.acc, trial.world_trace.rotations)]

    #     raw_imu_jerk_estimate = np.diff(np.array(raw_imu_trace.acc), axis=0)  # Simple finite difference jerk estimate
    #     raw_imu_jerk_correction = np.cross(np.array(raw_imu_trace.gyro)[:-1], np.array(raw_imu_trace.acc)[:-1])  # Cross product correction term

    #     # Plot the acceleration difference for comparison
    #     # fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    #     # for i in range(3):
    #     #     ax[i].plot( np.array(raw_imu_trace.acc)[:, i], label='Raw IMU Acceleration', alpha=0.7)
    #     #     ax[i].plot( np.array(no_grav_imu_trace.acc)[:, i], label='No Gravity IMU Acceleration', alpha=0.7)
    #     #     ax[i].set_ylabel(['Acceleration X (m/s²)', 'Acceleration Y (m/s²)', 'Acceleration Z (m/s²)'][i])
    #     #     ax[i].legend()
    #     # ax[2].set_xlabel('Frame')
    #     # plt.suptitle(f'Acceleration Comparison for Trial: {trial.name}')
    #     # plt.tight_layout()
    #     # plt.show()

    #     # Plot the jerk estimates for comparison
    #     fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    #     for i in range(3):
    #         ax[i].plot(raw_imu_trace.timestamps[1:], raw_imu_jerk_estimate[:, i], label='Raw IMU Jerk Estimate', alpha=0.7)
    #         ax[i].plot(raw_imu_trace.timestamps[1:], raw_imu_jerk_correction[:, i] + raw_imu_jerk_estimate[:, i], label='Raw IMU Jerk with Correction', alpha=0.7)
    #         ax[i].set_ylabel(['Jerk X (m/s³)', 'Jerk Y (m/s³)', 'Jerk Z (m/s³)'][i])
    #         ax[i].legend()
    #     ax[2].set_xlabel('Time (s)')
    #     plt.suptitle(f'Jerk Estimates for Trial: {trial.name}')
    #     plt.tight_layout()
    #     plt.show()
