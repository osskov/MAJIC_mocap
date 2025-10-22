import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from scipy.spatial.transform import Rotation
import numpy as np
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import warnings
import pandas as pd
import os


from paper_data_pipeline.generate_kinematics_from_trial import JOINT_SEGMENT_DICT

# --- Global Constants and Helper Functions ---

# This function must be used to resync the timing of a given set of PlateTrials to target WorldTraces
def resync_traces(plate_trials: List['PlateTrial'],
                  target_world_traces: Dict[str, 'WorldTrace']) -> List['PlateTrial']:
    
    imu_slice: slice = slice(0,0)
    world_slice: slice = slice(0,0)

    # A list to hold the newly resynced and aligned trials
    new_plate_trials: List['PlateTrial'] = [] 

    for i, trial in enumerate(plate_trials):
        try:
            target_world_trace = target_world_traces[trial.name]
        except KeyError:
            print(f"Warning: No target world trace found for trial {trial.name}. Skipping.")
            continue
        trial.world_trace = target_world_trace
        if abs(trial.imu_trace.get_sample_frequency() - trial.world_trace.get_sample_frequency()) > 0.2:
            print(f"Sample frequency mismatch for {trial.name}: IMU {trial.imu_trace.get_sample_frequency()} Hz, World {trial.world_trace.get_sample_frequency()} Hz")
            trial.imu_trace = trial.imu_trace.resample(float(trial.world_trace.get_sample_frequency()))

        if imu_slice == slice(0, 0) and world_slice == slice(0, 0):
            imu_slice, world_slice = PlateTrial._sync_traces(trial.imu_trace, trial.world_trace)
        synced_imu_trace = trial.imu_trace[imu_slice].re_zero_timestamps()
        synced_world_trace = trial.world_trace[world_slice].re_zero_timestamps()
        new_plate_trial = PlateTrial(trial.name, synced_imu_trace, synced_world_trace)

        new_plate_trials.append(new_plate_trial)
        
    return new_plate_trials

# This function calculates the joint traces (R_parent_world^T * R_child_world) from PlateTrials
def get_joint_traces(plate_trials: List['PlateTrial']) -> Dict[str, 'WorldTrace']:
    """Calculates joint rotations (child relative to parent) from WorldTraces."""
    joint_traces: Dict[str, 'WorldTrace'] = {}
    for joint_name, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
        parent_trial = next((p for p in plate_trials if parent_name in p.name), None)
        child_trial = next((p for p in plate_trials if child_name in p.name), None)
        
        if not parent_trial or not child_trial:
            continue
            
        joint_rotations = [Rwp.T @ Rwc for Rwp, Rwc in zip(parent_trial.world_trace.rotations, child_trial.world_trace.rotations)]
        
        # Assuming WorldTrace is a class that can be initialized like this
        joint_traces[joint_name] = WorldTrace( 
            timestamps=parent_trial.imu_trace.timestamps,
            positions=[np.zeros(3) for _ in range(len(parent_trial))], 
            rotations=joint_rotations
        )
    return joint_traces

def get_joint_traces_df(plate_trials: List['PlateTrial'], euler_order: str = 'ZYX') -> pd.DataFrame:
    """
    Calculates joint rotations (child relative to parent) from PlateTrials
    and returns them in a long-form pandas DataFrame using Euler angles.
    
    Args:
        plate_trials (List['PlateTrial']): Assumes all trials in this list 
                                          have been resynced and trimmed.
        euler_order (str): The order of intrinsic Euler angle rotations 
                           (e.g., 'ZYX', 'YXZ').
                           Angles are in RADIANS.

    Returns:
        pd.DataFrame: A DataFrame with columns for metadata and Euler angles
                      (e.g., 'euler_Z_rad', 'euler_Y_rad', 'euler_X_rad').
    """
    all_joint_data = [] # List of dictionaries for DataFrame creation
    
    for joint_name, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
        # Find the parent and child trials
        parent_trial = next((p for p in plate_trials if parent_name in p.name), None)
        child_trial = next((p for p in plate_trials if child_name in p.name), None)
        
        if not parent_trial or not child_trial:
            continue
            
        timestamps = parent_trial.imu_trace.timestamps
        
        try:
            # Calculate joint rotations: R_joint = R_parent_world.T @ R_child_world
            joint_rotations_mat = [Rwp.T @ Rwc for Rwp, Rwc in zip(parent_trial.world_trace.rotations, child_trial.world_trace.rotations)]
            
            if not joint_rotations_mat:
                print(f"Warning: No rotations calculated for {joint_name}. Skipping.")
                continue

            # Convert 3x3 matrices to Euler angles (in radians)
            joint_rotations_euler = Rotation.from_matrix(joint_rotations_mat).as_euler(euler_order) # (N, 3)
        
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error processing rotations for {joint_name} ({parent_name}/{child_name}): {e}. Skipping.")
            continue

        # Create a dictionary for this joint's data
        # Dynamically name columns based on the Euler order
        joint_df_data = {
            'timestamp': timestamps,
            f'euler_{euler_order[0]}_rad': joint_rotations_euler[:, 0],
            f'euler_{euler_order[1]}_rad': joint_rotations_euler[:, 1],
            f'euler_{euler_order[2]}_rad': joint_rotations_euler[:, 2],
        }
        
        joint_df = pd.DataFrame(joint_df_data)
        joint_df['joint_name'] = joint_name
        
        all_joint_data.append(joint_df)
    
    if not all_joint_data:
        return pd.DataFrame() # Return empty DF if no joints were processed

    final_df = pd.concat(all_joint_data, ignore_index=True)
    
    return final_df

# This function loads all joint traces for a given subject and trial type
def load_joint_traces_for_subject(subject_id: str, trial_type: str) -> Dict[str, Dict[str, 'WorldTrace']]:
    """
    Load joint traces for a specific subject and trial type.

    Args:
        subject_id (str): The ID of the subject.
        trial_type (str): The type of trial (e.g., "marker", "madgwick", etc.).

    Returns:
        Dict[str, Dict[str, 'WorldTrace']]: A dictionary mapping trial names to their corresponding WorldTraces.
    """
    try:
        # Load Marker trials
        plate_trials_marker: List['PlateTrial'] = PlateTrial.load_trial_from_folder(f"data/ODay_Data/{subject_id}/{trial_type}", align_plate_trials = True)

        # Load fresh sets of PlateTrials for each IMU method
        plate_trials_never_project: List['PlateTrial'] = [trial.copy() for trial in plate_trials_marker]
        plate_trials_madgwick: List['PlateTrial'] = [trial.copy() for trial in plate_trials_marker]
        plate_trials_mag_free: List['PlateTrial'] = [trial.copy() for trial in plate_trials_marker]

        # Load WorldTraces (Orientations)
        never_project_world_traces: Dict[str, 'WorldTrace'] = WorldTrace.load_from_sto_file(
            f"data/ODay_Data/{subject_id}/{trial_type}/IMU/never project/{trial_type}_orientations.sto")
        madgwick_world_traces: Dict[str, 'WorldTrace'] = WorldTrace.load_WorldTraces_from_folder(
            f"data/ODay_Data/{subject_id}/{trial_type}/IMU")
        mag_free_world_traces: Dict[str, 'WorldTrace'] = WorldTrace.load_from_sto_file(
            f"data/ODay_Data/{subject_id}/{trial_type}/IMU/mag free/{trial_type}_orientations.sto")
    except Exception as e:
        print(f"Error loading data for {subject_id}: {e}. Skipping subject.")
        return {} # Return empty dict on failure
    assert len(plate_trials_marker) == len(plate_trials_madgwick) == len(plate_trials_mag_free) == len(plate_trials_never_project), "Mismatch in number of trials loaded."

    # Resync Traces
    plate_trials_never_project = resync_traces(plate_trials_never_project, never_project_world_traces)
    plate_trials_madgwick = resync_traces(plate_trials_madgwick, madgwick_world_traces)
    plate_trials_mag_free = resync_traces(plate_trials_mag_free, mag_free_world_traces)

    
    # Trim to Minimum Length (Essential for alignment)
    all_trials_lists = [plate_trials_marker, plate_trials_madgwick, plate_trials_mag_free, plate_trials_never_project]
    all_lengths = [len(trial) for trials_list in all_trials_lists for trial in trials_list]

    if not all_lengths:
        print(f"No valid trials found for {subject_id} after resync/trim. Skipping subject.")
        return {}

    current_min_length = min(all_lengths)
    
    for trials_list in all_trials_lists:
        for i in range(len(trials_list)):
            trials_list[i] = trials_list[i][:current_min_length]

    # Calculate Joint Traces
    marker_joint_traces = get_joint_traces(plate_trials_marker)
    never_projected_joint_traces = get_joint_traces(plate_trials_never_project)
    madgwick_joint_traces = get_joint_traces(plate_trials_madgwick)
    mag_free_joint_traces = get_joint_traces(plate_trials_mag_free)

    
    # Consolidate all joint traces for plotting
    all_joint_traces_for_plotting = {
        'Marker': marker_joint_traces,
        'Madgwick': madgwick_joint_traces,
        'Mag-Free': mag_free_joint_traces,
        'Never Project': never_projected_joint_traces,
    }
    return all_joint_traces_for_plotting

def load_joint_traces_for_subject_df(subject_id: str, 
                                     trial_type: str, 
                                     euler_order: str = 'ZYX') -> pd.DataFrame:
    """
    Loads, resyncs, and processes joint traces for a specific subject and
    trial type, returning a consolidated pandas DataFrame with Euler angles.

    Args:
        subject_id (str): The subject identifier.
        trial_type (str): The trial type (e.g., "Gait").
        euler_order (str): The order of intrinsic Euler angle rotations 
                           (e.g., 'ZYX', 'YXZ') to be passed to get_joint_traces_df.

    Returns:
        pd.DataFrame: A multi-indexed DataFrame containing all joint data.
    """
    try:
        plate_trials_marker: List['PlateTrial'] = PlateTrial.load_trial_from_folder(
            f"data/ODay_Data/{subject_id}/{trial_type}"
        )
        plate_trials_never_project: List['PlateTrial'] = [trial.copy() for trial in plate_trials_marker]
        plate_trials_madgwick: List['PlateTrial'] = [trial.copy() for trial in plate_trials_marker]
        plate_trials_unprojected: List['PlateTrial'] = [trial.copy() for trial in plate_trials_marker]
        plate_trials_mag_free: List['PlateTrial'] = [trial.copy() for trial in plate_trials_marker]

        never_project_world_traces: Dict[str, 'WorldTrace'] = WorldTrace.load_from_sto_file(
            f"data/ODay_Data/{subject_id}/{trial_type}/IMU/never project/{trial_type}_orientations.sto")
        madgwick_world_traces: Dict[str, 'WorldTrace'] = WorldTrace.load_WorldTraces_from_folder(
            f"data/ODay_Data/{subject_id}/{trial_type}/IMU")
        unprojected_world_traces: Dict[str, 'WorldTrace'] = WorldTrace.load_from_sto_file(
            f"data/ODay_Data/{subject_id}/{trial_type}/IMU/unprojected/{trial_type}_orientations.sto")
        mag_free_world_traces: Dict[str, 'WorldTrace'] = WorldTrace.load_from_sto_file(
            f"data/ODay_Data/{subject_id}/{trial_type}/IMU/mag free/{trial_type}_orientations.sto")
    
    except Exception as e:
        print(f"Error loading data for {subject_id}: {e}. Skipping subject.")
        return pd.DataFrame() 
    
    # --- Resync Traces ---
    plate_trials_never_project = resync_traces(plate_trials_never_project, never_project_world_traces)
    plate_trials_madgwick = resync_traces(plate_trials_madgwick, madgwick_world_traces)
    plate_trials_mag_free = resync_traces(plate_trials_mag_free, mag_free_world_traces)
    plate_trials_unprojected = resync_traces(plate_trials_unprojected, unprojected_world_traces)
    
    # --- Trim to Minimum Length ---
    all_trials_lists = [plate_trials_marker, plate_trials_madgwick, plate_trials_mag_free, plate_trials_never_project, plate_trials_unprojected]

    all_lengths = [len(trial) for trials_list in all_trials_lists for trial in trials_list]
    all_lengths.append(60000)

    if not all_lengths or min(all_lengths) == 0:
        print(f"No valid trials found for {subject_id} after resync/trim. Skipping subject.")
        return pd.DataFrame()

    current_min_length = min(all_lengths)
    
    all_trials_lists_trimmed = []
    for trials_list in all_trials_lists:
        new_list = []
        for trial in trials_list:
            if len(trial) >= current_min_length:
                new_list.append(trial[-current_min_length:])
            else:
                print(f"Skipping trial {trial.name} as it is shorter ({len(trial)}) than min_length ({current_min_length}).")
        all_trials_lists_trimmed.append(new_list)
        
    (plate_trials_marker_trimmed, 
     plate_trials_madgwick_trimmed, 
     plate_trials_mag_free_trimmed, 
     plate_trials_never_project_trimmed,
     plate_trials_unprojected_trimmed) = all_trials_lists_trimmed

    # --- Calculate Joint Traces (using new DF function) ---
    method_data = [
        ('Marker', plate_trials_marker_trimmed),
        ('Madgwick', plate_trials_madgwick_trimmed),
        ('Mag-Free', plate_trials_mag_free_trimmed),
        ('Never Project', plate_trials_never_project_trimmed),
        ('Unprojected', plate_trials_unprojected_trimmed)
    ]
    
    all_method_dfs = []
    for method_name, trials_list in method_data:
        if not trials_list:
            print(f"No valid trimmed trials for method {method_name}. Skipping.")
            continue
        
        # Pass the euler_order argument
        joint_df = get_joint_traces_df(trials_list, euler_order=euler_order)
        
        if not joint_df.empty:
            joint_df['method'] = method_name
            all_method_dfs.append(joint_df)

    if not all_method_dfs:
        print(f"No joint data processed for {subject_id}. Returning empty DataFrame.")
        return pd.DataFrame()
        
    # --- Consolidate into a single DataFrame ---
    final_df = pd.concat(all_method_dfs, ignore_index=True)
    
    final_df['subject_id'] = subject_id
    final_df['trial_type'] = trial_type
    
    # Dynamically define column order
    meta_cols = ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']
    euler_cols = [f'euler_{c}_rad' for c in euler_order] # e.g., ['euler_Z_rad', 'euler_Y_rad', 'euler_X_rad']
    
    columns_order = meta_cols + euler_cols
    
    # Reorder columns
    final_df = final_df[columns_order]

    # Set a final multi-index for easy access
    final_df = final_df.set_index(
        ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']
    ).sort_index()

    return final_df

# --- RMSE Calculation Functions ---

def calculate_euler_rmse_matrix(truth_trace: 'WorldTrace', test_trace: 'WorldTrace') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the RMSE of Euler angles (xyz) between two WorldTraces,
    correctly accounting for 360-degree angle wrapping.

    Returns a 3-element array [rmse_x, rmse_y, rmse_z] and a
    3-element array for the standard deviation of the absolute angular error.
    """
    # Get Euler angles in degrees
    truth_angles = np.array([Rotation.from_matrix(r).as_euler('xyz', degrees=True) for r in truth_trace.rotations])
    test_angles = np.array([Rotation.from_matrix(r).as_euler('xyz', degrees=True) for r in test_trace.rotations])

    if truth_angles.shape != test_angles.shape:
        warnings.warn("Trace shapes mismatch, truncating to shortest.")
        min_len = min(truth_angles.shape[0], test_angles.shape[0])
        truth_angles = truth_angles[:min_len]
        test_angles = test_angles[:min_len]

    # Calculate the difference in angles
    diff = truth_angles - test_angles

    # Wrap the difference to the range [-180, 180] to get the shortest angle
    wrapped_diff = (diff + 180) % 360 - 180
    # --------------------------------------------------------

    # Calculate squared error from the wrapped difference
    sq_error = wrapped_diff ** 2
    
    # Mean squared error along the time axis (axis=0)
    mean_sq_error = np.mean(sq_error, axis=0)
    
    # Root mean squared error
    rmse = np.sqrt(mean_sq_error)
    
    # Calculate standard deviation of the absolute angular error
    abs_error = np.abs(wrapped_diff)
    std = np.std(abs_error, axis=0)

    return rmse, std

def calculate_euler_rmse_by_joint(
    all_subjects_joint_traces: List[Dict[str, Dict[str, 'WorldTrace']]],
    method_names: List[str],
    joints: List[str]
) -> Dict[str, Dict[str, np.ndarray]]: # <-- 1. FIX: Changed return type hint
    """
    Calculates combined RMSE results for multiple subjects, organized by method and joint.
    
    Returns a dict where each value is a (2, 3) np.ndarray:
    [[mean_rmse_x, mean_rmse_y, mean_rmse_z],
     [mean_std_x,  mean_std_y,  mean_std_z]]
    """
    # --- Initial Data Collection Loop ---
    results_list: Dict[str, Dict[str, List[np.ndarray]]] = \
        {method: {joint: [] for joint in joints} for method in method_names}

    for subject_traces in all_subjects_joint_traces:
        if not subject_traces:
            continue

        for method in method_names:
            for joint in joints:
                try:
                    truth_trace = subject_traces['Marker'][joint]
                    test_trace = subject_traces[method][joint]
                    rmse_values, std_values = calculate_euler_rmse_matrix(truth_trace, test_trace)
                    # Appending a (2, 3) array for each subject
                    results_list[method][joint].append(np.array([rmse_values, std_values]))
                except KeyError:
                    print(f"Warning: Missing data for RMSE calculation: {method} - {joint}")
                    continue

    # --- Final Processing Loop (with fixes) ---
    
    # This will be the final, type-stable dictionary
    final_results: Dict[str, Dict[str, np.ndarray]] = \
        {method: {} for method in method_names}

    for method in method_names:
        for joint in joints:
            
            subject_data_list = results_list[method][joint]
            
            if subject_data_list:
                # Convert list of (2, 3) arrays into one (N, 2, 3) array
                # N = number of subjects
                rmses_and_stds_array = np.array(subject_data_list)
                
                # Calculate the mean along the "subject" axis (axis=0)
                # This results in a (2, 3) array
                mean_rmses = np.mean(rmses_and_stds_array[:, 0, :], axis=0)
                std_rmses = np.std(rmses_and_stds_array[:, 0, :], axis=0)
                final_results[method][joint] = np.array([mean_rmses, std_rmses])
            else:
                # Handle the case where no data was found
                # Instead of an empty list, store a (2, 3) array of NaNs
                # to maintain type consistency.
                final_results[method][joint] = np.full((2, 3), np.nan)
    return final_results

def calculate_euler_rmse_by_axis(
    all_subjects_joint_traces: List[Dict[str, Dict[str, 'WorldTrace']]],
    method_names: List[str],
    joints: List[str],
) -> Dict[str, np.ndarray]: # <-- FIX: Changed return type to match description
    """
    Calculates combined RMSE results for multiple subjects, organized by method.
    
    This function averages across ALL subjects AND ALL joints for a given method.
    
    Returns a dict where each value is a (2, 3) np.ndarray:
    [[mean_rmse_x, mean_rmse_y, mean_rmse_z],
     [mean_std_x,  mean_std_y,  mean_std_z]]
    """
    
    # --- Initial Data Collection Loop ---
    # --- 1. Get results per joint (averaged over subjects) ---
    joint_results = calculate_euler_rmse_by_joint(
        all_subjects_joint_traces, 
        method_names, 
        joints
    )

    # --- 2. Aggregate joint results for each method ---
    final_results: Dict[str, np.ndarray] = {}

    for method in method_names:
        
        # Collect all (2, 3) mean arrays for this method, one for each joint.
        # This list may contain (2, 3) arrays of np.nan from the previous step
        all_joint_mean_rmse_for_method: List[np.ndarray] = []
        if method in joint_results:
            for joint in joints:
                if joint in joint_results[method]:
                    all_joint_mean_rmse_for_method.append(joint_results[method][joint])

        if all_joint_mean_rmse_for_method:
            # Convert list of (2, 3) arrays into one (M, 2, 3) array
            # M = number of joints
            joint_means_array = np.array(all_joint_mean_rmse_for_method)
            
            # Calculate the nan-safe mean along the "joint" axis (axis=0)
            # This averages all the joint means, ignoring any NaN entries
            # from joints that had no data.
            mean_of_joint_means = np.nanmean(joint_means_array[:, 0, :], axis=0)
            std_of_joint_means = np.std(joint_means_array[:, 0, :], axis=0)
            
            # np.nanmean returns a single np.nan if all values in a slice
            # (e.g., all x-axis RMSEs) were np.nan, which is the desired behavior.
            final_results[method] = np.array([mean_of_joint_means, std_of_joint_means])
        else:
            # Handle the case where no data was found for this method
            final_results[method] = np.full((2, 3), np.nan)
            
    return final_results

def calculate_rmse_and_std(all_data_df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """
    Calculates RMSE and STD of the error between all methods and the 'Marker' method.

    Args:
        all_data_df (pd.DataFrame): The input DataFrame, indexed by 
            ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp'].
            Columns are the Euler angles (e.g., 'euler_X_rad', 'euler_Y_rad').
        group_by (List[str]): A list of index level names to group by. 
            Can include 'subject_id', 'trial_type', 'joint_name', and 'axis'.

    Returns:
        pd.DataFrame: A DataFrame with RMSE and STD values, grouped as specified,
                      with methods as the first column level and metrics (RMSE, STD)
                      as the second.
    """
    
    # 1. Stack Euler angle columns (e.g., 'euler_X_rad') to be a new index level 'axis'
    # This makes grouping by axis possible.
    original_index_names = all_data_df.index.names
    data_stacked = all_data_df.stack().rename_axis(original_index_names + ['axis'])
    
    # 2. Unstack by 'method' to get methods as columns for comparison
    # Index: ['subject_id', 'trial_type', 'joint_name', 'timestamp', 'axis']
    # Columns: ['Marker', 'Madgwick', 'Mag-Free', ...]
    try:
        data_wide = data_stacked.unstack('method')
    except KeyError:
        print("Error: 'method' not found in the DataFrame's index.")
        return pd.DataFrame()

    # 3. Check if 'Marker' column exists
    if 'Marker' not in data_wide.columns:
        print("Warning: 'Marker' method not found. Cannot calculate error.")
        return pd.DataFrame()

    # 4. Calculate error (all methods - 'Marker')
    # We subtract the 'Marker' Series from the entire DataFrame
    # axis=0 (or 'index') ensures alignment on the multi-level index
    error_df = data_wide.subtract(data_wide['Marker'], axis=0)
    
    # Drop the 'Marker' column itself (which is now all zeros)
    error_df = error_df.drop('Marker', axis=1, errors='ignore')

    # 5. Calculate Squared Error for RMSE
    squared_error_df = error_df ** 2

    # 6. Group by the specified levels
    # Filter for group_by levels that actually exist in the index
    valid_group_by = [level for level in group_by if level in error_df.index.names]
    
    if not valid_group_by:
        # If group_by is empty, calculate metrics over the entire dataset
        grouped_error = error_df
        grouped_sq_error = squared_error_df
    else:
        try:
            grouped_error = error_df.groupby(level=valid_group_by)
            grouped_sq_error = squared_error_df.groupby(level=valid_group_by)
        except KeyError as e:
            print(f"Error: Grouping level not found: {e}. Available levels are: {error_df.index.names}")
            return pd.DataFrame()

    # 7. Calculate final metrics
    # RMSE = sqrt(mean(squared_error))
    rmse = np.sqrt(grouped_sq_error.mean())
    
    # STD = std(error)
    std_dev = grouped_error.std()

    # 8. Combine into a final DataFrame with multi-index columns
    results_df = pd.concat(
        {'RMSE': rmse, 'STD': std_dev},
        axis=1  # Concatenate as columns
    )
    
    # Reorder column levels to have method first, then metric
    if not results_df.empty:
        results_df = results_df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    return results_df

# --- Single Subject Plotting and Error Calculation Functions ---

def plot_subject_time_series(
    all_joint_traces: Dict[str, Dict[str, 'WorldTrace']], 
    subject_id: str, 
    trial_type: str,
    show_plots: bool,
    save_plots: bool
):
    """
    Plots the time-series joint angles for all methods against the Marker truth.
    Creates one figure per subject, with subplots for each joint and axis.
    """
    print(f"Generating time-series plots for {subject_id}...")
    
    methods = ['Marker', 'Madgwick', 'Mag-Free', 'Never Project']
    colors = {'Marker': 'k', 'Madgwick': 'r', 'Mag-Free': 'g', 'Never Project': 'b'}
    linestyles = {'Marker': '--', 'Madgwick': '-', 'Mag-Free': '-', 'Never Project': '-'}
    
    # Get joint list from Marker data
    joints = list(all_joint_traces.get('Marker', {}).keys())
    if not joints:
        print(f"No joints found for {subject_id}. Skipping time-series plot.")
        return

    num_joints = len(joints)
    axes = ['X-axis (Add/Ab)', 'Y-axis (Int/Ext Rot)', 'Z-axis (Flex/Ext)']
    
    fig, axs = plt.subplots(num_joints, 3, 
                            figsize=(20, 5 * num_joints), 
                            sharex=True)
    
    # Ensure axs is always a 2D array
    if num_joints == 1:
        axs = np.array([axs])

    for i, joint_name in enumerate(joints):
        for j, axis_name in enumerate(axes):
            ax = axs[i, j]
            
            for method_name in methods:
                try:
                    trace = all_joint_traces[method_name][joint_name]
                    timestamps = trace.timestamps
                    
                    # Get Euler angles and extract the correct axis
                    angles_deg = np.array([Rotation.from_matrix(r).as_euler('xyz', degrees=True) for r in trace.rotations])
                    axis_data = angles_deg[:, j]
                    
                    ax.plot(timestamps, axis_data, 
                            label=method_name, 
                            color=colors[method_name], 
                            linestyle=linestyles[method_name],
                            linewidth=1.5)
                    
                except KeyError:
                    print(f"Warning: No data for {method_name} - {joint_name}")
            
            ax.set_title(f"{joint_name} - {axis_name}")
            ax.set_ylabel("Angle (degrees)")
            if i == num_joints - 1:
                ax.set_xlabel("Time (s)")
            ax.grid(True, linestyle=':', alpha=0.6)

    # Create a single legend for the whole figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize='large')
    
    fig.suptitle(f"Joint Angle Time Series for {subject_id} ({trial_type})", 
                 fontsize=20, y=1.02)
    plt.tight_layout(rect=(0., 0., 1., 0.98))
    if save_plots:
        plt.savefig(f"{subject_id}_{trial_type}_time_series.png", dpi=150)
    if show_plots:
        plt.show()

def plot_subject_euler_error_bar_chart(
    all_joint_traces: Dict[str, Dict[str, 'WorldTrace']], 
    method_names: List[str], 
    subject_id: str, 
    trial_type: str,
    show_plots: bool,
    save_plots: bool
):
    """
    Plots a grid of bar charts: one row per joint, with cols for X, Y, Z axes.
    Each chart compares the RMSE of different methods.
    Labels for RMSE and STD are printed above each bar.
    """
    print(f"Generating error bar charts for {subject_id}...")
    
    try:
        truth_traces = all_joint_traces['Marker']
    except KeyError:
        print("Error: 'Marker' data not found. Cannot calculate RMSE.")
        return

    joints = list(truth_traces.keys())
    if not joints:
        print(f"No joints found for {subject_id}. Skipping RMSE plot.")
        return

    # Store results: {method: {joint: ([rmse_x, rmse_y, rmse_z], [std_x, std_y, std_z])}}
    all_rmses: Dict[str, Dict[str, np.ndarray]] = {}

    for method in method_names:
        method_rmses = {}
        for joint in joints:
            try:
                truth_trace = truth_traces[joint]
                test_trace = all_joint_traces[method][joint]
                # Returns (array[x,y,z], array[x,y,z])
                rmse_values, rmse_std = calculate_euler_rmse_matrix(truth_trace, test_trace) 
                method_rmses[joint] = (rmse_values, rmse_std) 
                
            except KeyError:
                print(f"Warning: Missing data for RMSE: {method} - {joint}")
                method_rmses[joint] = (np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]))
        all_rmses[method] = method_rmses

    # --- Plotting ---
    num_joints = len(joints)
    num_methods = len(method_names)
    
    axes_names = ['X-axis (Flex/Ext)', 'Y-axis (Ad/Ab)', 'Z-axis (Int/Ext Rot)']
    
    # NEW: Create a grid of subplots (rows=joints, cols=axes)
    # figsize height is scaled by the number of joints
    # squeeze=False ensures axs is always a 2D array, even if num_joints=1
    fig, axs = plt.subplots(num_joints, 3, figsize=(22, 6 * num_joints), squeeze=False)

    # Define consistent colors for each method
    colors = plt.get_cmap('tab10', num_methods)
    method_colors = {method: colors(i) for i, method in enumerate(method_names)}

    for j_idx, joint in enumerate(joints):
        for a_idx, axis_name in enumerate(axes_names):
            
            ax = axs[j_idx, a_idx]
            
            # Prepare data for this specific subplot (this joint, this axis)
            rmse_data = [all_rmses[method][joint][0][a_idx] for method in method_names]
            std_data = [all_rmses[method][joint][1][a_idx] for method in method_names]
            bar_colors = [method_colors[method] for method in method_names]

            # Plot the bars for each method
            bars = ax.bar(method_names, rmse_data, yerr=std_data, 
                          capsize=5, color=bar_colors)

            # NEW: Add labels above bars
            labels = [f'{r:.2f}\n±{s:.2f}' if not np.isnan(r) else '' 
                      for r, s in zip(rmse_data, std_data)]
            ax.bar_label(bars, labels=labels, fontsize=9, padding=3, zorder=10)

            # --- Formatting ---
            # 1. Set the tick locations explicitly
            ax.set_xticks(np.arange(len(method_names))) 
            # 2. Now set the labels and rotation
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.grid(True, linestyle=':', alpha=0.6, axis='y')
            
            # Give 20% extra space at the top for the labels
            if not all(np.isnan(rmse_data)):
                ax.set_ylim(top=np.nanmax(rmse_data) * 1.25)
            else:
                ax.set_ylim(top=1.0) # Default if all data is NaN

            # Set titles only for the top row
            if j_idx == 0:
                ax.set_title(axis_name, fontsize=14, pad=15)
                
            # Set Y-axis labels only for the left-most column
            if a_idx == 0:
                ax.set_ylabel(f"{joint}\nRMSE (degrees)", fontsize=12, labelpad=15)

    fig.suptitle(f"Joint Angle RMSE vs. Marker for {subject_id} ({trial_type})", 
                 fontsize=20, y=1.0) # y=1.0 is often cleaner with tight_layout
    
    # Adjust layout to prevent overlap and fit suptitle
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    
    if save_plots:
        plt.savefig(f"{subject_id}_{trial_type}_rmse_barchart.png", dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()

# --- By Joint, Combined Subjects Plotting ---

def plot_by_joint_rmse(
    all_results: Dict[str, Dict[str, np.ndarray]],
    method_names: List[str],
    joints: List[str],
    title: str,
    filename: str = "multi_subject_rmse.png",
    show_plots: bool = True,
    save_plots: bool = True
):
    """
    Plots a grid of bar charts for multi-subject mean RMSE.
    
    The input `all_results` is the direct output from
    `calculate_multi_subject_euler_rmse_by_joint`.
    
    The plot layout has one row per joint and one column per axis (X, Y, Z).
    """
    print("Generating multi-subject mean RMSE plot...")
    
    num_joints = len(joints)
    num_methods = len(method_names)
    
    if num_joints == 0 or num_methods == 0:
        print("No joints or methods to plot. Skipping.")
        return

    axes_names = ['X-axis (Flex/Ext)', 'Y-axis (Ad/Ab)', 'Z-axis (Int/Ext Rot)']
    
    # Create a grid of subplots (rows=joints, cols=axes)
    fig, axs = plt.subplots(num_joints, 3, 
                          figsize=(20, 5 * num_joints), 
                          squeeze=False)

    # Define consistent colors for each method
    colors = plt.get_cmap('tab10', num_methods)
    method_colors = {method: colors(i) for i, method in enumerate(method_names)}

    for j_idx, joint in enumerate(joints):
        for a_idx, axis_name in enumerate(axes_names):
            
            ax = axs[j_idx, a_idx]
            
            # Prepare data for this specific subplot (this joint, this axis)
            rmse_data = []
            std_data = []
            bar_colors = []
            
            for method in method_names:
                # Get the (2, 3) array for this method and joint
                # [[rmse_x, rmse_y, rmse_z],
                #  [std_x,  std_y,  std_z]]
                data_array = all_results[method].get(joint, np.full((2, 3), np.nan))
                
                # Get the RMSE for this axis (a_idx)
                rmse_data.append(data_array[0, a_idx]) 
                # Get the STD for this axis (a_idx)
                std_data.append(data_array[1, a_idx])
                
                bar_colors.append(method_colors[method])

            # Plot the bars for each method
            bars = ax.bar(method_names, rmse_data, yerr=std_data, 
                          capsize=5, color=bar_colors)

            # Add labels above bars (RMSE ± STD)
            labels = [f'{r:.2f}\n±{s:.2f}' if not np.isnan(r) else 'N/A' 
                      for r, s in zip(rmse_data, std_data)]
            ax.bar_label(bars, labels=labels, fontsize=9, padding=3, zorder=10)

            # --- Formatting ---
            # 1. Set the tick locations explicitly
            ax.set_xticks(np.arange(len(method_names))) 
            # 2. Now set the labels and rotation
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.grid(True, linestyle=':', alpha=0.6, axis='y')
            
            # Give 20% extra space at the top for the labels
            if not all(np.isnan(rmse_data)):
                ax.set_ylim(top=np.nanmax(rmse_data) * 1.25)
            else:
                ax.set_ylim(top=1.0) # Default if all data is NaN

            # Set titles only for the top row
            if j_idx == 0:
                ax.set_title(axis_name, fontsize=14, pad=15)
                
            # Set Y-axis labels only for the left-most column
            if a_idx == 0:
                ax.set_ylabel(f"{joint}\nMean RMSE (degrees)", fontsize=12, labelpad=15)

    fig.suptitle(title, fontsize=20, y=1.0)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))

    if save_plots:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()

# --- By Axis, Combined Subjects Plotting ---

def plot_by_axis_rmse(
    all_results: Dict[str, np.ndarray],
    method_names: List[str],
    title: str,
    filename: str = "multi_subject_axis_rmse.png",
    show_plots: bool = True,
    save_plots: bool = True
):
    """
    Plots a bar chart for multi-subject mean RMSE, grouped by axis.
    
    The input `all_results` is the direct output from
    `calculate_euler_rmse_by_axis`.
    
    The plot layout has one row with three subplots: X-axis, Y-axis, Z-axis.
    """
    print("Generating multi-subject mean RMSE (by axis) plot...")
    
    num_methods = len(method_names)
    if num_methods == 0:
        print("No methods to plot. Skipping.")
        return

    # Use generic axis names, as 'xyz' doesn't always map to biomechanics
    axes_names = ['X-axis (Roll)', 'Y-axis (Pitch)', 'Z-axis (Yaw)']
    
    # Create a grid of subplots (1 row, 3 columns)
    fig, axs = plt.subplots(1, 3, figsize=(20, 7), squeeze=False)
    axs = axs.flatten() # Make sure axs is a 1D array

    # Define consistent colors for each method
    colors = plt.get_cmap('tab10', num_methods)
    method_colors = {method: colors(i) for i, method in enumerate(method_names)}

    for a_idx, axis_name in enumerate(axes_names):
        
        ax = axs[a_idx]
        
        # Prepare data for this specific subplot (this axis)
        rmse_data = []
        std_data = []
        bar_colors = []
        
        for method in method_names:
            # Get the (2, 3) array for this method
            # [[mean_x, mean_y, mean_z],
            #  [std_x,  std_y,  std_z]]
            data_array = all_results.get(method, np.full((2, 3), np.nan))
            
            # Get the RMSE for this axis (a_idx)
            rmse_data.append(data_array[0, a_idx]) 
            # Get the STD for this axis (a_idx)
            std_data.append(data_array[1, a_idx])
            
            bar_colors.append(method_colors[method])

        # Plot the bars for each method
        bars = ax.bar(method_names, rmse_data, yerr=std_data, 
                      capsize=5, color=bar_colors)

        # Add labels above bars (RMSE ± STD)
        labels = [f'{r:.2f}\n±{s:.2f}' if not np.isnan(r) else 'N/A' 
                  for r, s in zip(rmse_data, std_data)]
        ax.bar_label(bars, labels=labels, fontsize=10, padding=3, zorder=10)

        # --- Formatting ---
        ax.set_title(axis_name, fontsize=14, pad=15)
        # 1. Set the tick locations explicitly
        ax.set_xticks(np.arange(len(method_names))) 
        # 2. Now set the labels and rotation
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        
        # Give 20% extra space at the top for the labels
        if not all(np.isnan(rmse_data)):
            ax.set_ylim(top=np.nanmax(rmse_data) * 1.25)
        else:
            ax.set_ylim(top=1.0) # Default if all data is NaN

        # Set Y-axis label only for the left-most plot
        if a_idx == 0:
            ax.set_ylabel("Mean RMSE (degrees)\n(Averaged over all subjects & joints)", 
                          fontsize=12, labelpad=15)

    fig.suptitle(title, fontsize=20, y=1.05)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    if save_plots:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()

# --- By Method, Combined Subjects Plotting ---

def plot_by_method_rmse():
    per_method_errors = {}
    for subject in range(len(all_subjects_results)):
        for method in method_names_for_error:
            if method not in per_method_errors:
                per_method_errors[method] = []
            for joint in joints:
                try:
                    truth_trace = all_subjects_results[subject]['Marker'][joint]
                    test_trace = all_subjects_results[subject][method][joint]
                    rmse_values, rmse_std = calculate_euler_rmse_matrix(truth_trace, test_trace)
                    # Store or plot as needed
                    per_method_errors[method].append(rmse_values)
                except KeyError:
                    print(f"Warning: Missing data for RMSE calculation: Subject {subject}, {method} - {joint}")
                    continue
# --- Main Execution ---

if __name__ == "__main__":
    # 1. Define the parameters
    # Change this list to include all your subjects!
    SUBJECT_IDS: List[str] = ["Subject01", "Subject02", "Subject03", "Subject04", "Subject05", "Subject06", "Subject07", "Subject08", "Subject09", "Subject10", "Subject11"]
    trial_type = "complexTasks"
    method_names_for_error = ['Madgwick', 'Mag-Free', 'Never Project']
    joints = list(JOINT_SEGMENT_DICT.keys()) # Assumes JOINT_SEGMENT_DICT is available
    show_plots = False
    save_plots = True

    # Assuming your DataFrame is named 'all_subjects_gait_df'
    file_path = f"data/ODay_Data/all_subject_data_{trial_type}.pkl"

    if os.path.exists(file_path):
        print(f"Loading existing DataFrame from {file_path}...")
        all_data_df = pd.read_pickle(file_path)
    else:
        # 2. Loop, load, and collect
        all_subject_dfs = []
        for subject_id in SUBJECT_IDS:
            print(f"--- Processing {subject_id} ---")
            subject_df = load_joint_traces_for_subject_df(
                subject_id=subject_id,
                trial_type=trial_type,
                euler_order='XYZ'
            )
            
            if not subject_df.empty:
                all_subject_dfs.append(subject_df)
            else:
                print(f"No data returned for {subject_id}, skipping.")

        # 3. Concatenate all DataFrames
        if not all_subject_dfs:
            print("No data was loaded for any subject. Exiting.")
            exit()

        print("--- Concatenating all subjects ---")
        all_data_df = pd.concat(all_subject_dfs)
        
        # Assuming your DataFrame is named 'all_subjects_gait_df'
        file_path = f"output/all_subject_data_{trial_type}.pkl"

        # Save the DataFrame to a pickle file
        all_data_df.to_pickle(file_path)

        print(f"DataFrame saved to {file_path}")
        all_subjects_results = [{} for _ in range(12)]
    print("Data loading and concatenation complete.")

    by_joint_rmse = calculate_rmse_and_std(
        all_data_df,
        group_by=['joint_name', 'axis']
    )
    print(by_joint_rmse.head())

    by_subject_rmse = calculate_rmse_and_std(
        all_data_df,
        group_by=['subject_id'])
    print(by_subject_rmse.head())