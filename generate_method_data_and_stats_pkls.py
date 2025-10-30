from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from scipy.spatial.transform import Rotation
from typing import List, Dict
import numpy as np
import pandas as pd
import os
from generate_method_orientation_sto_files import JOINT_SEGMENT_DICT


ALL_SUBJECTS = ["Subject01"]#, "Subject02", "Subject03", "Subject04", "Subject05", "Subject06",
                #"Subject07", "Subject08", "Subject09", "Subject10", "Subject11"]

ALL_METHODS = ['Marker', 
            #    'Madgwick (Al Borno)', 'EKF', 'Mahony', 'Madgwick',
               'Mag On', 'Mag Off', 'Mag Adapt', 'Unprojected']

ALL_JOINTS = ['R_Ankle', 'L_Ankle', 'R_Knee', 'L_Knee', 'R_Hip', 'L_Hip', 'Lumbar']

ALL_TRIAL_TYPES = ['walking', 'complexTasks']

REGENERATE_FILES = True
BASE_DATA_PATH = os.path.abspath(os.path.join("data", "data"))

# ------------------------------------------------

# --- Data Loading Functions (Remain Unchanged) ---

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
        print(f"Processing joint: {joint_name} ({parent_name}/{child_name}) with {len(timestamps)} samples.")
        try:
            joint_rotations_mat = [Rwp.T @ Rwc for Rwp, Rwc in zip(parent_trial.world_trace.rotations, child_trial.world_trace.rotations)]
            
            if not joint_rotations_mat:
                print(f"Warning: No rotations calculated for {joint_name}. Skipping.")
                continue

            joint_rotations_euler = Rotation.from_matrix(joint_rotations_mat).as_euler(euler_order) # (N, 3)
            joint_rotations_rotvec = Rotation.from_matrix(joint_rotations_mat).as_rotvec() # (N, 3)
        
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error processing rotations for {joint_name} ({parent_name}/{child_name}): {e}. Skipping.")
            continue
        print(f"   > Calculated {len(joint_rotations_euler)} joint rotations for {joint_name}.")
        joint_df_data = {
            'timestamp': timestamps,
            f'euler_{euler_order[0]}_rad': joint_rotations_euler[:, 0],
            f'euler_{euler_order[1]}_rad': joint_rotations_euler[:, 1],
            f'euler_{euler_order[2]}_rad': joint_rotations_euler[:, 2],
            f'angle_axis_x_rad': joint_rotations_rotvec[:, 0],
            f'angle_axis_y_rad': joint_rotations_rotvec[:, 1],
            f'angle_axis_z_rad': joint_rotations_rotvec[:, 2],
        }
        print(f"   > Prepared data for DataFrame for joint {joint_name}.")
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
        print(f"Processing joint traces for method {method_name}...")
        joint_df = get_joint_traces_df(trials_list, euler_order=euler_order)
        
        if not joint_df.empty:
            joint_df['method'] = method_name
            print(f"   > Processed joint traces for method {method_name}, resulting in {len(joint_df)} rows.")
            all_method_dfs.append(joint_df)

    if not all_method_dfs:
        print(f"No joint data processed for {subject_id}. Returning empty DataFrame.")
        return pd.DataFrame()
        
    final_df = pd.concat(all_method_dfs, ignore_index=True)
    
    final_df['subject_id'] = subject_id
    final_df['trial_type'] = trial_type
    
    meta_cols = ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']
    data_cols = [f'euler_{c}_rad' for c in euler_order] + ['angle_axis_x_rad', 'angle_axis_y_rad', 'angle_axis_z_rad']
    
    columns_order = meta_cols + data_cols
    # Ensure all columns exist before reordering
    final_df = final_df.reindex(columns=columns_order)

    final_df = final_df.set_index(
        ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']
    ).sort_index()
    print(f"Final DataFrame for {subject_id} has {len(final_df)} rows and columns: {final_df.columns.tolist()}")
    return final_df

# --- Calculation Functions (Remain Unchanged) ---
def get_summary_statistics(all_data_df: pd.DataFrame, 
                           group_by: List[str], 
                           euler_order: str = 'XYZ') -> pd.DataFrame:
    """
    Calculates a comprehensive set of error statistics 
    between all methods and the 'Marker' method. (Refactored for performance)
    """
    print("--- [get_summary_statistics] Starting error statistics calculation... ---")
    
    if 'subject' in group_by and 'subject_id' in all_data_df.index.names:
        df = all_data_df.rename_axis(index={'subject_id': 'subject'})
    else:
        df = all_data_df.copy()

    try:
        marker_df = df.xs('Marker', level='method')
        imu_df = df.drop('Marker', level='method')
        print(f"   > Separated 'Marker' ({len(marker_df)} rows) from IMU methods ({len(imu_df)} rows).")
    except KeyError:
        print("Error: 'Marker' method not found in DataFrame. Cannot calculate error.")
        return pd.DataFrame()
        
    if imu_df.empty:
        print("Error: No IMU methods found to compare against 'Marker'.")
        return pd.DataFrame()

    join_levels = [name for name in imu_df.index.names if name != 'method']
    
    print(f"   > Merging IMU and Marker data on levels: {join_levels}...")
    merged_df = pd.merge(
        imu_df.reset_index(), 
        marker_df.reset_index(), 
        on=join_levels, 
        suffixes=('_imu', '_marker')
    )
    
    if merged_df.empty:
        print("Error: Merge between IMU and Marker data resulted in an empty DataFrame.")
        return pd.DataFrame()
    print(f"   > Merge complete. Result has {len(merged_df)} rows.")

    print("   > Reconstructing rotations from Euler angles in a single batch operation...")
    try:
        euler_cols_imu = [f'euler_{c}_rad_imu' for c in euler_order]
        euler_cols_marker = [f'euler_{c}_rad_marker' for c in euler_order]
        imu_euler_angles = merged_df[euler_cols_imu].values
        marker_euler_angles = merged_df[euler_cols_marker].values

        R_imu = Rotation.from_euler(euler_order, imu_euler_angles)
        R_marker = Rotation.from_euler(euler_order, marker_euler_angles)
        
        print("   > Calculating error rotation (R_err = R_marker.inv() * R_imu)...")
        R_err = R_marker.inv() * R_imu
    except Exception as e:
        print(f"   > CRITICAL Error: Failed to calculate rotational error: {e}.")
        print(f"   > Check if these columns exist in merged_df: {euler_cols_imu} and {euler_cols_marker}")
        return pd.DataFrame()

    print("   > Decomposing R_err into 3D magnitude ('MAG')...")
    error_magnitude = R_err.magnitude()
    MAGNITUDE_COL_NAME = 'angle_axis_error_magnitude_rad'
    merged_df[MAGNITUDE_COL_NAME] = error_magnitude
    
    error_cols_dict = {MAGNITUDE_COL_NAME: 'MAG'}

    print(f"   > Decomposing R_err into Euler angles (order: {euler_order})...")
    error_euler_angles = R_err.as_euler(euler_order)
    
    for i, char in enumerate(euler_order):
        error_col_name = f'error_euler_{char}_rad'
        merged_df[error_col_name] = error_euler_angles[:, i]
        error_cols_dict[error_col_name] = char
    
    print(f"   > Calculated error columns: {list(error_cols_dict.keys())}.")
    
    meta_cols = [lvl for lvl in join_levels] 
    if 'method' not in meta_cols:
        meta_cols.append('method')
        
    error_columns_to_melt = list(error_cols_dict.keys())
    
    print("   > Reshaping DataFrame (melting) from wide to long format...")
    id_vars = [col for col in meta_cols + ['timestamp'] if col in merged_df.columns]

    error_long = pd.melt(
        merged_df,
        id_vars=id_vars,
        value_vars=error_columns_to_melt,
        var_name='error_metric_name',
        value_name='error_rad'
    )
    
    error_long['axis'] = error_long['error_metric_name'].map(error_cols_dict)
    
    final_index_cols = [col for col in group_by if col in error_long.columns]
    final_index_cols.append('axis')
    final_index_cols = list(dict.fromkeys(final_index_cols))

    set_index_cols = [col for col in final_index_cols if col in error_long.columns]
    error_long = error_long.set_index(set_index_cols)

    valid_group_by = [lvl for lvl in group_by if lvl in error_long.index.names]
    valid_group_by.append('axis')
    valid_group_by = list(dict.fromkeys(valid_group_by))
    
    if not valid_group_by:
        print("Error: No valid group_by keys. Returning empty DataFrame.")
        return pd.DataFrame()

    print(f"   > Grouping data by {valid_group_by}...")
    grouped = error_long['error_rad'].dropna().groupby(level=valid_group_by)
    
    def q25(x): return x.quantile(0.25)
    def q75(x): return x.quantile(0.75)
    def rmse(x): return np.sqrt(np.mean(x**2))
    def mae(x): return x.abs().mean()
    def mad(x): return (x - x.median()).abs().median()

    print("   > Calculating aggregate statistics (RMSE, MAE, Mean, etc.)...")
    summary_df = grouped.agg(
        RMSE = rmse, MAE = mae, Mean = 'mean', STD = 'std', MAD = mad,
        Skew = 'skew', Kurtosis = lambda x: x.kurtosis(),
        Q25 = q25, Median = 'median', Q75 = q75,
    ).sort_index()
    
    rad_cols = ['RMSE', 'MAE', 'Mean', 'STD', 'MAD', 'Q25', 'Median', 'Q75']
    deg_cols = [f'{col}_deg' for col in rad_cols]
    
    summary_df[deg_cols] = summary_df[rad_cols].apply(np.rad2deg)

    rad_rename_dict = {col: f'{col}_rad' for col in rad_cols}
    summary_df = summary_df.rename(columns=rad_rename_dict)
    print("   > Converted statistics to degrees and renamed columns.")
    
    final_cols = [item for col in rad_cols for item in (f'{col}_rad', f'{col}_deg')]
    final_cols.extend(['Skew', 'Kurtosis'])
    
    final_cols = [c for c in final_cols if c in summary_df.columns]
    summary_df = summary_df[final_cols]

    print("--- [get_summary_statistics] Finished. Returning error summary DataFrame. ---")
    return summary_df

def get_pearson_correlation_summary(all_data_df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """
    Calculates the Pearson correlation (R) between the time series of each method 
    and the 'Marker' method for all Euler and Angle-Axis components.
    """
    print("--- [get_pearson_correlation_summary] Starting correlation calculation... ---")

    if 'subject' in group_by and 'subject_id' in all_data_df.index.names:
        df = all_data_df.rename_axis(index={'subject_id': 'subject'})
    else:
        df = all_data_df.copy()

    try:
        marker_df = df.xs('Marker', level='method')
        imu_df = df.drop('Marker', level='method')
    except KeyError:
        print("Error: 'Marker' method not found in DataFrame. Cannot calculate correlation.")
        return pd.DataFrame()
        
    euler_cols = [col for col in df.columns if col.startswith('euler_')]
    angle_axis_cols = [col for col in df.columns if col.startswith('angle_axis_')]
    all_correlation_cols = euler_cols + angle_axis_cols
        
    if not all_correlation_cols:
        print("Error: No angle columns found for correlation.")
        return pd.DataFrame()

    join_levels = marker_df.index.names
    corr_group_levels = [lvl for lvl in join_levels if lvl != 'timestamp']
    if 'method' not in corr_group_levels:
        corr_group_levels.append('method') 

    marker_reset = marker_df.reset_index()
    imu_reset = imu_df.reset_index()
    
    merged_df = pd.merge(
        imu_reset, 
        marker_reset, 
        on=list(join_levels), 
        suffixes=('_imu', '_marker')
    )
    if merged_df.empty:
        print("Error: Merge resulted in an empty DataFrame.")
        return pd.DataFrame()
    
    corr_cols = {
        col: (f'{col}_imu', f'{col}_marker') 
        for col in all_correlation_cols
    }
    
    print(f"   > Calculating Pearson correlation grouped by {corr_group_levels}...")
    
    def calculate_pair_correlation(group, col_pairs):
        results = {}
        for original_col, (imu_col, marker_col) in col_pairs.items():
            if original_col.startswith('euler_'):
                axis_name = original_col.split('_')[1].upper()
            else:
                axis_name = 'AA_' + original_col.split('_')[2].upper()
            results[axis_name] = group[imu_col].corr(group[marker_col], method='pearson')
        return pd.Series(results, name='PearsonR')

    corr_grouped = merged_df.groupby(corr_group_levels).apply(lambda x: calculate_pair_correlation(x, corr_cols))
    
    correlation_df = corr_grouped.stack().rename('PearsonR').to_frame()
    correlation_df.index.names = corr_group_levels + ['axis'] 

    valid_group_by = [lvl for lvl in group_by if lvl in correlation_df.index.names]
    
    if len(valid_group_by) < len(correlation_df.index.names):
        print(f"   > Re-grouping correlation data by requested levels: {valid_group_by}...")
        correlation_df = correlation_df.groupby(level=valid_group_by).min()
    
    print("--- [get_pearson_correlation_summary] Finished. Returning correlation DataFrame. ---")
    return correlation_df

# --- Main Execution ---

if __name__ == "__main__":
    
    # ------------------ CONTROL FLAG ------------------
    # Set to True to force regeneration of all data files from source.
    # Set to False to load existing .pkl files if they are available.
    # ----------------------------------------------------

    # 1. Define the parameters
    SUBJECT_IDS: List[str] = ALL_SUBJECTS
    METHODS: List[str] = ALL_METHODS
    JOINTS = ALL_JOINTS
    TRIAL_TYPES = ALL_TRIAL_TYPES

    # Create directories if they don't exist
    os.makedirs("plots", exist_ok=True)
    os.makedirs(os.path.join("data", "data"), exist_ok=True)

    data_file_path = os.path.join(BASE_DATA_PATH, f"all_subject_data.pkl")
    
    if not REGENERATE_FILES and os.path.exists(data_file_path):
        print(f"Loading existing DataFrame from {data_file_path}...")
        all_data_df = pd.read_pickle(data_file_path)
    else:
        if REGENERATE_FILES:
            print("--- REGENERATE_FILES is True. Forcing regeneration of all data. ---")
        
        # 2. Loop, load, and collect
        all_subject_dfs = []
        for trial_type in TRIAL_TYPES:
            for subject_id in SUBJECT_IDS:
                subject_pkl_path = os.path.abspath(os.path.join("data", "data", subject_id, trial_type, f"{subject_id}_{trial_type}_data.pkl"))
                print(f"--- Processing {subject_id} - {trial_type} ---")

                if not REGENERATE_FILES and os.path.exists(subject_pkl_path):
                    print(f"Data for {subject_id} - {trial_type} already exists. Loading from pickle...")
                    subject_df = pd.read_pickle(subject_pkl_path)
                else:
                    print(f"Loading and processing from source for {subject_id} - {trial_type}...")
                    try:
                        subject_df = load_joint_traces_for_subject_df(
                            subject_id=subject_id,
                            methods=METHODS,
                            trial_type=trial_type,
                            euler_order='XYZ' # Using 'XYZ' as per your original call
                        )
                        if not subject_df.empty:
                            subject_df.to_pickle(subject_pkl_path)
                    except Exception as e:
                        print(f"Error processing {subject_id}: {e}. Skipping subject.")
                        continue
                
                if not subject_df.empty:
                    all_subject_dfs.append(subject_df)
                else:
                    print(f"No data returned for {subject_id}, skipping.")

        if not all_subject_dfs:
            print("No data was loaded for any subject. Exiting.")
            exit()

        print("--- Concatenating all subjects ---")
        all_data_df = pd.concat(all_subject_dfs)
        all_data_df.to_pickle(data_file_path)
        print(f"DataFrame saved to {data_file_path}")  
    
    print("Data loading and concatenation complete.")

    # --- Get Summary Statistics ---
    stats_file_path = os.path.join(BASE_DATA_PATH, f"all_subject_statistics.pkl")
    if not REGENERATE_FILES and os.path.exists(stats_file_path):
        print(f"Loading existing summary statistics from {stats_file_path}...")
        summary_stats_df = pd.read_pickle(stats_file_path)
    else:
        if REGENERATE_FILES:
            print("--- REGENERATE_FILES is True. Forcing regeneration of summary statistics. ---")
        summary_stats_df = get_summary_statistics(
            all_data_df,
            group_by=['trial_type', 'method', 'joint_name', 'subject', 'axis']
        )
        summary_stats_df.to_pickle(stats_file_path)

    print("Summary statistics calculation complete.")
    summary_stats_df.to_csv(os.path.join(BASE_DATA_PATH, f"all_subject_statistics.csv"))

    # --- Get Pearson Correlation Summary ---
    pearson_corr_file_path = os.path.join(BASE_DATA_PATH, f"all_subject_pearson_correlation.pkl")
    if not REGENERATE_FILES and os.path.exists(pearson_corr_file_path):
        print(f"Loading existing Pearson correlation data from {pearson_corr_file_path}...")
        pearson_corr_df = pd.read_pickle(pearson_corr_file_path)
    else:
        if REGENERATE_FILES:
            print("--- REGENERATE_FILES is True. Forcing regeneration of Pearson correlation. ---")
        pearson_corr_df = get_pearson_correlation_summary(
            all_data_df,
            group_by=['trial_type', 'method', 'joint_name', 'subject', 'axis']
        )
        pearson_corr_df.to_pickle(pearson_corr_file_path)

    print("Pearson correlation calculation complete.")
    pearson_corr_df.to_csv(os.path.join(BASE_DATA_PATH, f"all_subject_pearson_correlation.csv"))