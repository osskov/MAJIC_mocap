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
import seaborn as sns
import itertools
import json


# --- Assume JOINT_SEGMENT_DICT is defined here ---
# Example:
JOINT_SEGMENT_DICT = {
    'R_Ankle': ('R_Tibia', 'R_Foot'),
    'L_Ankle': ('L_Tibia', 'L_Foot'),
    'R_Knee': ('R_Femur', 'R_Tibia'),
    'L_Knee': ('L_Femur', 'L_Tibia'),
    'R_Hip': ('Pelvis', 'R_Femur'),
    'L_Hip': ('Pelvis', 'L_Femur'),
    'Pelvis_Trunk': ('Pelvis', 'Trunk')
}

ALL_SUBJECTS = ["Subject01"]#, "Subject02", "Subject03", "Subject04", "Subject05", "Subject06",
              #  "Subject07", "Subject08", "Subject09", "Subject10", "Subject11"]

ALL_METHODS = ['Marker', 
               'Madgwick (Al Borno)', 'EKF', 'Mahony', 'Madgwick',
               'Never Project', 'Mag Free', 'Cascade', 'Unprojected']

ALL_JOINTS = ['R_Ankle', 'L_Ankle', 'R_Knee', 'L_Knee', 'R_Hip', 'L_Hip', 'Lumbar']

ALL_TRIAL_TYPES = ['walking', 'complexTasks']

# ------------------------------------------------

# --- Data Loading Functions ---

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
        parent_trial = next((p for p in plate_trials if parent_name in p.name), None)
        child_trial = next((p for p in plate_trials if child_name in p.name), None)
        
        if not parent_trial or not child_trial:
            continue
            
        timestamps = parent_trial.imu_trace.timestamps
        
        try:
            joint_rotations_mat = [Rwp.T @ Rwc for Rwp, Rwc in zip(parent_trial.world_trace.rotations, child_trial.world_trace.rotations)]
            
            if not joint_rotations_mat:
                print(f"Warning: No rotations calculated for {joint_name}. Skipping.")
                continue

            joint_rotations_euler = Rotation.from_matrix(joint_rotations_mat).as_euler(euler_order) # (N, 3)
        
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error processing rotations for {joint_name} ({parent_name}/{child_name}): {e}. Skipping.")
            continue

        joint_df_data = {
            'timestamp': timestamps,
            f'euler_{euler_order[0]}_rad': joint_rotations_euler[:, 0],
            f'euler_{euler_order[1]}_rad': joint_rotations_euler[:, 1],
            f'euler_{euler_order[2]}_rad': joint_rotations_euler[:, 2],
            f'angle_axis_magnitude_rad': [rot.magnitude() for rot in Rotation.from_matrix(joint_rotations_mat)]
        }
        
        joint_df = pd.DataFrame(joint_df_data)
        joint_df['joint_name'] = joint_name
        
        all_joint_data.append(joint_df)
    
    if not all_joint_data:
        return pd.DataFrame() 

    final_df = pd.concat(all_joint_data, ignore_index=True)
    
    return final_df

def load_joint_traces_for_subject_df(subject_id: str, 
                                     trial_type: str,
                                     methods: List[str] = ['Marker', 'Madgwick', 'Mag Free', 'Never Project'],
                                     euler_order: str = 'ZYX') -> pd.DataFrame:
    """
    Loads, resyncs, and processes joint traces for a specific subject and
    trial type, returning a consolidated pandas DataFrame with Euler angles.
    """
    try:
        plate_trials_by_method: Dict[str, List['PlateTrial']] = {}
        plate_trials_by_method['Marker'] = PlateTrial.load_trial_from_folder(
                        os.path.join("data", "data", subject_id, trial_type)
                    )
        imu_traces = {trial.name: trial.imu_trace.copy() for trial in plate_trials_by_method['Marker']}
        min_length = min(min(len(trial) for trial in plate_trials_by_method['Marker']), 60000)
        
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
    
    for method, trials in plate_trials_by_method.items():
        for i, trial in enumerate(trials):
            plate_trials_by_method[method][i] = trial[-min_length:]

    all_method_dfs = []
    for method_name, trials_list in plate_trials_by_method.items():
        if not trials_list:
            print(f"No valid trimmed trials for method {method_name}. Skipping.")
            continue
        
        joint_df = get_joint_traces_df(trials_list, euler_order=euler_order)
        
        if not joint_df.empty:
            joint_df['method'] = method_name
            all_method_dfs.append(joint_df)

    if not all_method_dfs:
        print(f"No joint data processed for {subject_id}. Returning empty DataFrame.")
        return pd.DataFrame()
        
    final_df = pd.concat(all_method_dfs, ignore_index=True)
    
    final_df['subject_id'] = subject_id
    final_df['trial_type'] = trial_type
    
    meta_cols = ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']
    euler_cols = [f'euler_{c}_rad' for c in euler_order] 
    
    columns_order = meta_cols + euler_cols
    final_df = final_df[columns_order]

    final_df = final_df.set_index(
        ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']
    ).sort_index()

    return final_df

# --- RMSE Calculation Function ---

def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a comprehensive set of error statistics and correlation 
    between all methods and the 'Marker' method. The grouping for statistics
    is internally fixed to ['method', 'joint_name', 'axis'].
    
    Args:
        all_data_df (pd.DataFrame): The DataFrame containing joint rotation data
                                    from all methods, including 'Marker'.
                                    Expected MultiIndex: 
                                    ['subject_id', 'trial_type', 'method', 
                                     'joint_name', 'timestamp']

    Returns:
        pd.DataFrame: A DataFrame with error statistics and Pearson correlation,
                      indexed by ['method', 'joint_name', 'axis'].
    """
    print("--- [get_summary_statistics] Starting statistics calculation... ---")

    # --- 1. Define Fixed Grouping ---
    # The statistics will always be calculated based on these levels.
    valid_group_by = ['method', 'joint_name', 'axis']
    corr_group_levels = [lvl for lvl in df.index.names if lvl != 'timestamp'] # Used for the initial correlation grouping
    
    # --- 2. Separate Reference (Marker) and Test (IMU) Data ---
    try:
        # 'Marker' is our ground truth.
        marker_df = df.xs('Marker', level='method')
        
        # 'imu_df' contains all other methods to be tested.
        imu_df = df.drop('Marker', level='method')
        print(f"   > Separated 'Marker' ({len(marker_df)} rows) from IMU methods ({len(imu_df)} rows).")

    except KeyError:
        print("Error: 'Marker' method not found in DataFrame. Cannot calculate error.")
        return pd.DataFrame()
        
    if imu_df.empty:
        print("Error: No IMU methods found to compare against 'Marker'.")
        return pd.DataFrame()

    # --- 3. Calculate Error via Merge (to avoid ambiguous join) ---
    
    # Get the column names for euler angles
    euler_cols = [col for col in df.columns if col.startswith('euler_') or col == 'angle_axis_magnitude_rad']
    if not euler_cols:
        print("Error: No Euler angle columns (e.g., 'euler_X_rad') found.")
        return pd.DataFrame()
        
    # Get index names to join on (all levels in marker_df)
    join_levels = marker_df.index.names
    
    # Reset indices for merge
    marker_reset = marker_df.reset_index()
    imu_reset = imu_df.reset_index()
    
    print(f"   > Merging IMU and Marker data on {len(join_levels)} levels...")
    # Perform a left join: keep all imu_df rows, find matching marker_df rows
    merged_df = pd.merge(
        imu_reset, 
        marker_reset, 
        on=list(join_levels), 
        suffixes=('_imu', '_marker')
    )
    
    # Check if merge was successful
    if merged_df.empty:
        print("Error: Merge between IMU and Marker data resulted in an empty DataFrame.")
        return pd.DataFrame()
    
    print(f"   > Merge complete. Result has {len(merged_df)} rows.")

    # Calculate error for each euler angle
    error_cols_dict = {}
    for col in euler_cols:
        # e.g., 'error_X'
        error_col_name = col.replace('euler_', 'error_').replace('_rad', '') if col != 'angle_axis_magnitude_rad' else 'error_angle_axis_magnitude'
        merged_df[error_col_name] = merged_df[f'{col}_imu'] - merged_df[f'{col}_marker']
        # 'X', 'Y', 'Z', 'MAG'
        error_cols_dict[error_col_name] = col.split('_')[1] if col != 'angle_axis_magnitude_rad' else 'MAG'
    
    print(f"   > Calculated error columns: {list(error_cols_dict.keys())}.")

    # --- 3.5: Calculate Pearson Correlation ---
    
    # Identify the IMU and Marker column pairs
    corr_cols = {
        col: (f'{col}_imu', f'{col}_marker') for col in euler_cols
    }
    
    # Levels to group by for the correlation (all non-time/sample levels)
    # The 'method' column is already in the grouped object via its index.
    corr_group_levels = [lvl for lvl in join_levels if lvl != 'timestamp']
    
    print(f"   > Calculating Pearson correlation grouped by {corr_group_levels}...")
    
    def calculate_pair_correlation(group, col_pairs):
        results = {}
        for original_col, (imu_col, marker_col) in col_pairs.items():
            # Calculate Pearson R and map it to the axis name (e.g., 'X', 'Y', 'Z', 'MAG')
            axis_name = error_cols_dict[original_col.replace('euler_', 'error_').replace('_rad', '') if original_col != 'angle_axis_magnitude_rad' else 'error_angle_axis_magnitude']
            results[axis_name] = group[imu_col].corr(group[marker_col], method='pearson')
        return pd.Series(results, name='PearsonR')

    # Apply the function to the grouped data
    corr_grouped = merged_df.groupby(corr_group_levels).apply(lambda x: calculate_pair_correlation(x, corr_cols))
    
    # Stack the axis results to match the final summary_df index levels
    correlation_df = corr_grouped.stack().rename('PearsonR').to_frame()
    final_corr_index = corr_group_levels + ['axis']
    correlation_df.index.names = final_corr_index
    print("   > Pearson correlation calculated and formatted.")

    # --- 4. Reshape to Long Format (Melt) ---
    
    # Metadata columns to keep
    meta_cols = list(imu_df.index.names) 
    
    print("   > Reshaping DataFrame (melting) from wide to long format...")
    error_long = pd.melt(
        merged_df,
        id_vars=meta_cols,
        value_vars=list(error_cols_dict.keys()),
        var_name='axis_name',
        value_name='error_rad'
    )
    print(f"   > Melt complete. Result has {len(error_long)} rows.")
    
    # --- 5. Clean up 'axis' Names and Set Index ---
    error_long['axis'] = error_long['axis_name'].map(error_cols_dict)
    
    all_levels = meta_cols + ['axis']
    error_long = error_long.set_index(all_levels).drop(columns=['axis_name'])
    print("   > Set new MultiIndex including 'axis'.")
    
    # --- 6. Group and Calculate Statistics ---
    
    # Use the fixed grouping for the final statistics
    grouped = error_long['error_rad'].groupby(level=valid_group_by)
    print(f"   > Grouping data by fixed levels: {valid_group_by}...")

    # Define the aggregation functions
    def q25(x):
        return x.quantile(0.25)

    def q75(x):
        return x.quantile(0.75)
        
    def rmse(x):
        return np.sqrt(np.mean(x**2))

    def mae(x):
        return x.abs().mean()
        
    def mad(x):
        # Median Absolute Deviation from the median
        return (x - x.median()).abs().median()

    print("   > Calculating aggregate statistics (RMSE, MAE, Mean, etc.)...")
    summary_df = grouped.agg(
        RMSE = rmse,
        MAE = mae,
        Mean = 'mean',
        STD = 'std',
        MAD = mad,
        Skew = 'skew',
        Kurtosis = lambda x: x.kurtosis(),
        Q25 = q25,
        Median = 'median',
        Q75 = q75,
    ).sort_index()
    print("   > Aggregation complete.")
    
    # --- 7. Merge Correlation Data ---
    print("   > Merging Pearson correlation data.")
    
    # The correlation_df index (subject, trial_type, method, joint_name, axis) is a 
    # superset of the summary_df index (method, joint_name, axis).
    # Group the correlation data to match the final summary_df index levels
    corr_to_merge = correlation_df.groupby(level=valid_group_by).min() 
    
    summary_df = summary_df.merge(
        corr_to_merge, 
        how='left', 
        left_index=True, 
        right_index=True
    )

    # --- 8. Convert to Degrees and Final Formatting ---
    rad_cols = ['RMSE', 'MAE', 'Mean', 'STD', 'MAD', 'Q25', 'Median', 'Q75']
    deg_cols = [f'{col}_deg' for col in rad_cols]
    
    summary_df[deg_cols] = summary_df[rad_cols].apply(np.rad2deg)

    rad_rename_dict = {col: f'{col}_rad' for col in rad_cols}
    summary_df = summary_df.rename(columns=rad_rename_dict)
    print("   > Converted statistics to degrees and renamed columns.")
    
    final_cols = []
    for col in rad_cols:
        final_cols.append(f'{col}_rad')
        final_cols.append(f'{col}_deg')
    
    final_cols.append('Skew')
    final_cols.append('Kurtosis')
    final_cols.append('PearsonR') 
    
    final_cols = [c for c in final_cols if c in summary_df.columns]
    
    summary_df = summary_df[final_cols]

    print("--- [get_summary_statistics] Finished. Returning summary DataFrame. ---")
    return summary_df

# --- Main Execution ---

if __name__ == "__main__":
    
    # 1. Define the parameters
    SUBJECT_IDS: List[str] = ALL_SUBJECTS
    METHODS: List[str] = ALL_METHODS
    JOINTS = ALL_JOINTS
    TRIAL_TYPES = ALL_TRIAL_TYPES

    # Create directories if they don't exist
    os.makedirs("plots", exist_ok=True)
    os.makedirs(os.path.join("data", "data"), exist_ok=True)

    data_file_path = os.path.abspath(os.path.join("data", "data", f"all_subject_data.pkl"))
    
    for trial_type in TRIAL_TYPES:
        for subject_id in SUBJECT_IDS:
            print(f"--- Processing {subject_id} - {trial_type} ---")
            pickle_path = os.path.abspath(os.path.join("data", "data", subject_id, trial_type, f"{subject_id}_{trial_type}_data.pkl"))
            if os.path.exists(pickle_path):
                print(f"Loading existing data for {subject_id} from {pickle_path}...")
                subject_df = pd.read_pickle(pickle_path)
            else:
                try:
                    subject_df = load_joint_traces_for_subject_df(
                        subject_id=subject_id,
                        methods=METHODS,
                        trial_type=trial_type,
                        euler_order='XYZ'
                    )
                    subject_df.to_pickle(pickle_path)
                except Exception as e:
                    print(f"Error processing {subject_id}: {e}. Skipping subject.")
                    continue
            stats_pickle_path = os.path.abspath(os.path.join("data", "data", subject_id, trial_type, f"{subject_id}_{trial_type}_statistics.pkl"))
            if os.path.exists(stats_pickle_path):
                print(f"Loading existing statistics for {subject_id} from {stats_pickle_path}...")
                summary_stats = pd.read_pickle(stats_pickle_path)
            else:
                summary_stats = get_summary_statistics(
                    subject_df,
                )
                summary_stats.to_pickle(stats_pickle_path)
                print(summary_stats.head())
                
            print(f"Processed {subject_id} - {trial_type} successfully.")
