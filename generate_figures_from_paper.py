import matplotlib.pyplot as plt
import matplotlib.container as mpc
from typing import List, Dict, Tuple
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from scipy.spatial.transform import Rotation
import numpy as np
import scipy.stats as stats
import pandas as pd
import os


from generate_orientations_from_data import JOINT_SEGMENT_DICT

# --- Data Loading Functions ---

# This function calculates joint rotations from PlateTrials and returns a DataFrame
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

# This function loads, resyncs, trims, and processes joint traces for a subject and trial type
def load_joint_traces_for_subject_df(subject_id: str, 
                                     trial_type: str,
                                     methods: List[str] = ['Marker', 'Madgwick', 'Mag Free', 'Never Project'],
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
        plate_trials_by_method: Dict[str, List['PlateTrial']] = {}
        plate_trials_by_method['Marker'] = PlateTrial.load_trial_from_folder(
                        os.path.join("data", "data", subject_id, trial_type)
                    )
        imu_traces = {trial.name: trial.imu_trace.copy() for trial in plate_trials_by_method['Marker']}
        min_length = min(min(len(trial) for trial in plate_trials_by_method['Marker']), 60000)  # Cap at 60000 frames
        
        for method in methods:
            if method == 'Marker':
                continue
            else:
                if method == 'Madgwick (Al Borno)':
                    world_traces = WorldTrace.load_WorldTraces_from_folder(
                        os.path.abspath(os.path.join("data", "data", subject_id, trial_type, "madgwick (al borno)"))
                    )
                else:
                    world_traces = WorldTrace.load_from_sto_file(
                        os.path.abspath(os.path.join("data", "data", subject_id, trial_type, f"{trial_type}_orientations_{method.replace(' ', '_').lower()}.sto"))
                    )
                plate_trials_by_method[method] = PlateTrial.generate_plate_from_traces(imu_traces, world_traces, align_plate_trials=False)
                min_length = min(min_length, min(len(trial) for trial in plate_trials_by_method[method]))
                print(f"Loaded {len(plate_trials_by_method[method])} trials for method {method}.")
    except Exception as e:
        print(f"Error loading data for {subject_id}: {e}. Skipping subject.")
        return pd.DataFrame() 
    # --- Trim Trials to Common Length ---
    for method, trials in plate_trials_by_method.items():
        for i, trial in enumerate(trials):
            plate_trials_by_method[method][i] = trial[-min_length:]

    # --- Calculate Joint Traces (using new DF function) ---
    all_method_dfs = []
    for method_name, trials_list in plate_trials_by_method.items():
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

# --- RMSE Calculation and Plotting Functions ---

# This function calculates RMSE and STD between all methods and the 'Marker' method. 
# The RMSE can be grouped by various levels: subject_id, trial_type, joint_name, axis.
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
    # Subtract the 'Marker' series from each method column in data_wide,
    # using DataFrame.sub with an explicit Series to avoid type confusion.
    marker_series = data_wide['Marker']
    # Ensure marker_series is aligned to the DataFrame index, then subtract
    error_df = data_wide.sub(marker_series, axis=0)
    
    # Drop the 'Marker' column itself (which is now all zeros)
    error_df = error_df.drop('Marker', axis=1, errors='ignore')
    error_df = error_df * 180 / np.pi  # Convert radians to degrees

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
            # _perform_statistical_tests(grouped_error)
        except KeyError as e:
            print(f"Error: Grouping level not found: {e}. Available levels are: {error_df.index.names}")
            return pd.DataFrame()

    # 7. Calculate final metrics
    # RMSE = sqrt(mean(squared_error))
    rmse = np.sqrt(grouped_sq_error.mean())
    
    # STD = std(error)
    std_dev = grouped_error.std()
    mean = grouped_error.mean()
    median = grouped_error.median()
    q_75 = grouped_error.quantile(0.75)
    q_25 = grouped_error.quantile(0.25)

    # 8. Combine into a final DataFrame with multi-index columns
    results_df = pd.concat(
        {
            'RMSE': rmse,
            'MAE': grouped_error.apply(lambda x: x.abs().mean()),
            'Bias': mean,
            'STD': std_dev,
            'MAD': grouped_error.apply(lambda x: (x - x.mean()).abs().median()),
            'Skew': grouped_error.apply(stats.skew),
            'Kurtosis': grouped_error.apply(stats.kurtosis),
            'Q25': q_25,
            'Median': median,
            'Q75': q_75,
        },
        axis=1
    )
    
    # Reorder column levels to have method first, then metric
    if not results_df.empty and isinstance(results_df.columns, pd.MultiIndex):
        results_df = results_df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    return results_df


# --- Main Execution ---

if __name__ == "__main__":
    # 1. Define the parameters
    # Change this list to include all your subjects!
    SUBJECT_IDS: List[str] = ["Subject01", "Subject02", "Subject03", "Subject04", "Subject05", "Subject06",
                              "Subject07", "Subject08", "Subject09", "Subject10", "Subject11"]
    METHODS: List[str] = ['Marker', 
                          'Madgwick (Al Borno)', 'EKF', 'Mahony', 'Madgwick',
                          'Never Project', 'Mag Free', 'Cascade', 'Unprojected'] # Capitalization is sensitive!
    joints = list(JOINT_SEGMENT_DICT.keys()) # Assumes JOINT_SEGMENT_DICT is available
    show_plots = False
    save_plots = True
    drop_ankles = False

    file_path = os.path.abspath(os.path.join("data", "data", f"all_subject_data.pkl"))
    os.mkdir("plots") if not os.path.exists("plots") else None
    if os.path.exists(file_path):
        print(f"Loading existing DataFrame from {file_path}...")
        all_data_df = pd.read_pickle(file_path)
    else:
        # 2. Loop, load, and collect
        all_subject_dfs = []
        for trial_type in ['walking', 'complexTasks']:
            for subject_id in SUBJECT_IDS:
                print(f"--- Processing {subject_id} ---")
                try:
                    subject_df = load_joint_traces_for_subject_df(
                        subject_id=subject_id,
                        methods=METHODS,
                        trial_type=trial_type,
                        euler_order='XYZ'
                    )
                except Exception as e:
                    print(f"Error processing {subject_id}: {e}. Skipping subject.")
                    continue
                
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
        
        # Save the DataFrame to a pickle file
        all_data_df.to_pickle(file_path)

        print(f"DataFrame saved to {file_path}")
    print("Data loading and concatenation complete.")

    # --- Get RMSE by all groups ---
    file_path = os.path.abspath(os.path.join("data", "data", f"all_subject_statistics.pkl"))
    if os.path.exists(file_path):
        print(f"Loading existing statistics DataFrame from {file_path}...")
        by_all_group_rmse = pd.read_pickle(file_path)
    else:
        print("Calculating RMSE and STD by all groups...")
        by_all_group_rmse = calculate_rmse_and_std(
            all_data_df,
            group_by=['joint_name', 'axis', 'subject_id', 'axis']
        )
        by_all_group_rmse.to_pickle(file_path)

    # --- Filter Dataset as Needed ---
    if drop_ankles or METHODS != all_data_df.index.get_level_values('method').unique().tolist() or SUBJECT_IDS != all_data_df.index.get_level_values('subject_id').unique().tolist():
        print("Filtering dataset based on specified criteria...")
        if drop_ankles:
            print("Dropping ankle joints from the dataset...")
            idx_to_drop = all_data_df.index.get_level_values('joint_name').isin(['L_Ankle', 'R_Ankle'])
            all_data_df = all_data_df[~idx_to_drop]
        
        if METHODS != all_data_df.index.get_level_values('method').unique().tolist():
            # Drop any methods not in the METHODS list
            print("Filtering methods to match the specified METHODS list...")
            idx_to_keep = all_data_df.index.get_level_values('method').isin(METHODS)
            all_data_df = all_data_df[idx_to_keep]

        if SUBJECT_IDS != all_data_df.index.get_level_values('subject_id').unique().tolist():
            # Drop any subjects not in the SUBJECT_IDS list
            print("Filtering subjects to match the specified SUBJECT_IDS list...")
            idx_to_keep = all_data_df.index.get_level_values('subject_id').isin(SUBJECT_IDS)
            all_data_df = all_data_df[idx_to_keep]
        print("Dataset filtering complete.")