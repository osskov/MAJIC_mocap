import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy.spatial.transform import Rotation
from src.toolchest.PlateTrial import PlateTrial
from src.RelativeFilterPlus import RelativeFilter

# --- Configuration ---
SUBJECTS_TO_OPTIMIZE = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
TRIALS = ['walking', 'complexTasks']
THRESHOLDS_TO_TEST = np.linspace(0, 150, 10)
JOINT_SEGMENT_DICT = {
     'Lumbar': ('pelvis_imu', 'torso_imu'),
     'R_Hip': ('pelvis_imu', 'femur_r_imu'),
     'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
    'R_Ankle': ('tibia_r_imu', 'calcn_r_imu'),
    # 'L_Hip': ('pelvis_imu', 'femur_l_imu'),
    # 'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
    # 'L_Ankle': ('tibia_l_imu', 'calcn_l_imu'),
}

# --- Core Functions ---

def preprocess_trial_data(plate_trials: List[PlateTrial]) -> Dict[str, Dict[str, Any]]:
    """
    Performs all expensive, one-time data processing for a given trial.
    
    This includes IMU projection and observability metric calculation.
    """
    preprocessed_data = {}
    for joint_name, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
        parent_trial = next((p for p in plate_trials if parent_name in p.name), None)
        child_trial = next((p for p in plate_trials if child_name in p.name), None)

        if not parent_trial or not child_trial:
            continue
            
        parent_trial = parent_trial.copy()
        child_trial = child_trial.copy()

        # --- One-time expensive operations ---
        # 1. Project IMU data to the joint center
        parent_offset, child_offset, _ = parent_trial.world_trace.get_joint_center(child_trial.world_trace)
        parent_trial.imu_trace = parent_trial.project_imu_trace(parent_offset)
        child_trial.imu_trace = child_trial.project_imu_trace(child_offset)

        # 2. Calculate the observability metric
        da_parent = np.diff(parent_trial.imu_trace.acc, axis=0) + np.cross(parent_trial.imu_trace.gyro[1:], parent_trial.imu_trace.acc[1:])
        da_child = np.diff(child_trial.imu_trace.acc, axis=0) + np.cross(child_trial.imu_trace.gyro[1:], child_trial.imu_trace.acc[1:])
        o_parent = np.cross(parent_trial.imu_trace.acc[1:], da_parent)
        o_child = np.cross(child_trial.imu_trace.acc[1:], da_child)
        observability_metric = np.minimum(np.linalg.norm(o_parent, axis=1), np.linalg.norm(o_child, axis=1))
        observability_metric = np.concatenate(([0.0], observability_metric))

        # 3. Store all necessary data for the filter loop
        preprocessed_data[joint_name] = {
            'parent_gyro': parent_trial.imu_trace.gyro,
            'parent_acc': parent_trial.imu_trace.acc,
            'parent_mag_original': parent_trial.imu_trace.mag,
            'child_gyro': child_trial.imu_trace.gyro,
            'child_acc': child_trial.imu_trace.acc,
            'child_mag_original': child_trial.imu_trace.mag,
            'observability_metric': observability_metric,
            'dt': parent_trial.imu_trace.timestamps[1] - parent_trial.imu_trace.timestamps[0],
            'q_wp_initial': Rotation.from_matrix(parent_trial.world_trace.rotations[0]),
            'q_wc_initial': Rotation.from_matrix(child_trial.world_trace.rotations[0]),
            'num_frames': len(parent_trial)
        }
    return preprocessed_data

def run_filter_with_preprocessed_data(
    joint_data: Dict[str, Any], 
    threshold: float,
    gyro_std: float = np.sqrt(0.01),
    acc_std: float = np.sqrt(0.05),
    mag_std: float = np.sqrt(0.05)
) -> List[np.ndarray]:
    """
    Runs the RelativeFilter update loop using pre-processed data and a new threshold.
    This function is fast as it only performs masking and the filter update.
    """
    # --- Fast, threshold-dependent operations ---
    # 1. Create magnetometer mask
    high_indexes = joint_data['observability_metric'] > threshold
    
    # 2. Apply the mask to the original magnetometer data
    parent_mag_masked = [np.zeros(3) if high else mag for high, mag in zip(high_indexes, joint_data['parent_mag_original'])]
    child_mag_masked = [np.zeros(3) if high else mag for high, mag in zip(high_indexes, joint_data['child_mag_original'])]

    # 3. Initialize the filter
    joint_filter = RelativeFilter(
        gyro_std_parent=np.ones(3) * gyro_std, gyro_std_child=np.ones(3) * gyro_std,
        vector_sensor_stds_parent=[np.ones(3) * acc_std, np.ones(3) * mag_std],
        vector_sensor_stds_child=[np.ones(3) * acc_std, np.ones(3) * mag_std]
    )
    joint_filter.set_qs(joint_data['q_wp_initial'], joint_data['q_wc_initial'])
    
    # 4. Run the filter update loop
    R_pc = []
    for t in range(joint_data['num_frames']):
        joint_filter.update(
            joint_data['parent_gyro'][t], joint_data['child_gyro'][t],
            [joint_data['parent_acc'][t], parent_mag_masked[t]],
            [joint_data['child_acc'][t], child_mag_masked[t]],
            joint_data['dt']
        )
        R_pc.append(joint_filter.get_R_pc())
        
    return R_pc
    
def get_ground_truth_joint_rotations(plate_trials: List[PlateTrial]) -> Dict[str, List[np.ndarray]]:
    ground_truth_joints = {}
    for joint_name, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
        parent_trial = next((p for p in plate_trials if parent_name in p.name), None)
        child_trial = next((p for p in plate_trials if child_name in p.name), None)
        if parent_trial and child_trial:
            r_parent_world = Rotation.from_matrix(parent_trial.world_trace.rotations)
            r_child_world = Rotation.from_matrix(child_trial.world_trace.rotations)
            r_joint = r_parent_world.inv() * r_child_world
            ground_truth_joints[joint_name] = r_joint.as_matrix()
    return ground_truth_joints

def calculate_joint_rotation_rmse(
    estimated_joint_rots: Dict[str, List[np.ndarray]],
    ground_truth_joint_rots: Dict[str, List[np.ndarray]]
) -> float:
    all_errors_deg = []
    for joint_name, est_rots in estimated_joint_rots.items():
        if joint_name in ground_truth_joint_rots:
            gt_rots = ground_truth_joint_rots[joint_name]
            min_len = min(len(est_rots), len(gt_rots))
            r_est = Rotation.from_matrix(est_rots[:min_len])
            r_gt = Rotation.from_matrix(gt_rots[:min_len])
            error_rotation = r_gt * r_est.inv()
            error_angles_deg = np.linalg.norm(error_rotation.as_rotvec(), axis=1) * 180.0 / np.pi
            all_errors_deg.extend(error_angles_deg)
    return np.median(all_errors_deg) if all_errors_deg else np.inf

# --- Main Execution Logic ---
if __name__ == "__main__":
    optimal_thresholds = {}
    print("--- Starting Efficient Threshold Optimization for Relative Filter (Joint Space) ---")
    
    for subject_num in SUBJECTS_TO_OPTIMIZE:
        print(f"\n======= Processing Subject {subject_num} =======")
        
        # --- Load and Pre-process Data ONCE for the subject ---
        subject_data_cache = {}
        for activity in TRIALS:
            try:
                folder = os.path.abspath(os.path.join("data", f"Subject{subject_num}", activity))
                plate_trials = PlateTrial.load_trial_from_folder(folder, align_plate_trials=True)
                if not plate_trials: continue
                
                plate_trials = [trial[:10000] for trial in plate_trials]

                subject_data_cache[activity] = {
                    "preprocessed_data": preprocess_trial_data(plate_trials),
                    "ground_truth_rotations": get_ground_truth_joint_rotations(plate_trials)
                }
                print(f"  Loaded and pre-processed '{activity}' data.")
            except Exception as e:
                print(f"  Could not load or process '{activity}' data for Subject {subject_num}: {e}")
                continue
        
        if not subject_data_cache:
            print(f"No data could be loaded for Subject {subject_num}. Skipping.")
            continue

        # --- Fast loop to test thresholds ---
        subject_results = []
        for threshold in THRESHOLDS_TO_TEST:
            trial_rmses = []
            
            for activity, cached_data in subject_data_cache.items():
                estimated_joint_rotations = {}
                for joint_name, joint_data in cached_data["preprocessed_data"].items():
                    estimated_joint_rotations[joint_name] = run_filter_with_preprocessed_data(joint_data, threshold)
                
                rmse = calculate_joint_rotation_rmse(estimated_joint_rotations, cached_data["ground_truth_rotations"])
                trial_rmses.append(rmse)
            
            avg_rmse_for_threshold = np.mean(trial_rmses)
            subject_results.append((threshold, avg_rmse_for_threshold))
            print(f"  Threshold: {threshold:7.2f} -> Median Error: {avg_rmse_for_threshold:.4f} degrees")
            
        # Find the best threshold for the subject
        best_threshold, min_rmse = min(subject_results, key=lambda item: item[1])
        optimal_thresholds[subject_num] = best_threshold
        
        print(f"----------------------------------------------------")
        print(f">>> Optimal Threshold for Subject {subject_num}: {best_threshold:.2f} (RMSE: {min_rmse:.4f} deg)")
        print(f"----------------------------------------------------")

    print("\n\n--- Optimization Complete ---")
    print("Final Optimal Thresholds per Subject:")
    results_df = pd.DataFrame(list(optimal_thresholds.items()), columns=['Subject', 'Optimal Threshold'])
    print(results_df.to_string(index=False))