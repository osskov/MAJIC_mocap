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
               'Madgwick (Al Borno)', 'EKF', 'Mahony', 'Madgwick',
               'Mag On', 'Mag Off', 'Mag Adapt', 'Unprojected']

ALL_JOINTS = ['R_Ankle', 'L_Ankle', 'R_Knee', 'L_Knee', 'R_Hip', 'L_Hip', 'Lumbar']

ALL_TRIAL_TYPES = ['walking', 'complexTasks']

BASE_DATA_PATH = os.path.abspath(os.path.join("data", "data"))
REGENERATE_DATA_PKLS = True

# ------------------------------------------------

# --- Data Loading Functions (Copied from your prompt) ---

def get_joint_traces_df(plate_trials: List['PlateTrial']) -> pd.DataFrame:
    """
    Calculates joint rotations (child relative to parent) from PlateTrials
    and returns them in a long-form pandas DataFrame, storing the
    scipy.spatial.transform.Rotation object directly.
    
    Args:
        plate_trials (List['PlateTrial']): Assumes all trials in this list 
                                          have been resynced and trimmed.
    Returns:
        pd.DataFrame: A DataFrame with columns for metadata and 'rotation'.
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

            # --- SIMPLIFIED LOGIC ---
            # 1. Create the Rotation objects
            joint_rotations = Rotation.from_matrix(joint_rotations_mat)
        
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error processing rotations for {joint_name} ({parent_name}/{child_name}): {e}. Skipping.")
            continue
            
        print(f"   > Calculated {len(joint_rotations)} joint rotations for {joint_name}.")
        
        # 2. Store the objects directly in the DataFrame
        # Note: The column will have dtype='object'
        joint_df_data = {
            'timestamp': timestamps,
            'rotation': joint_rotations, # This is an array of Rotation objects
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
                                     methods: List[str] = ['Marker', 'Madgwick', 'Mag Free', 'Never Project']) -> pd.DataFrame:
    """
    Loads, resyncs, and processes joint traces for a specific subject and
    trial type, returning a consolidated pandas DataFrame with Rotation objects.
    """
    try:
        # ... [Unchanged loading logic] ...
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
        
        # --- MODIFIED CALL (no euler_order) ---
        joint_df = get_joint_traces_df(trials_list)
        
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
    
    # --- SIMPLIFIED COLUMNS ---
    meta_cols = ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']
    data_cols = ['rotation'] # The only data column
    
    # Ensure all columns exist, especially 'rotation' in case of empty data
    columns_order = [c for c in (meta_cols + data_cols) if c in final_df.columns]
    final_df = final_df[columns_order]

    final_df = final_df.set_index(
        ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']
    ).sort_index()
    print(f"Final DataFrame for {subject_id} has {len(final_df)} rows and columns: {final_df.columns.tolist()}")
    return final_df

# --- RMSE Calculation Function ---
def get_summary_statistics(all_data_df: pd.DataFrame, 
                           group_by: List[str], 
                           euler_order: str = 'XYZ') -> pd.DataFrame:
    """
    Calculates a comprehensive set of error statistics 
    between all methods and the 'Marker' method using the 'rotation' object column.
    
    Args:
        all_data_df (pd.DataFrame): Must contain a 'rotation' column.
        group_by (List[str]): List of index levels to group by.
        euler_order (str): The euler order (e.g., 'XYZ') to use
                           for decomposing the error rotation.
    """
    print("--- [get_summary_statistics] Starting error statistics calculation... ---")
    
    # --- 1. Prepare DataFrame ---
    if 'subject' in group_by and 'subject_id' in all_data_df.index.names:
        print("   > Renaming index 'subject_id' to 'subject'.")
        df = all_data_df.rename_axis(index={'subject_id': 'subject'})
    else:
        df = all_data_df.copy()

    # --- 2. Separate Reference (Marker) and Test (IMU) Data ---
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

    # --- 3. Merge Data ---
    join_levels = marker_df.index.names
    marker_reset = marker_df.reset_index()
    imu_reset = imu_df.reset_index()
    
    print(f"   > Merging IMU and Marker data on {len(join_levels)} levels...")
    merged_df = pd.merge(
        imu_reset, 
        marker_reset, 
        on=list(join_levels), 
        suffixes=('_imu', '_marker')
    )
    
    if merged_df.empty:
        print("Error: Merge between IMU and Marker data resulted in an empty DataFrame.")
        return pd.DataFrame()
    
    print(f"   > Merge complete. Result has {len(merged_df)} rows.")
    
    if 'rotation_imu' not in merged_df.columns:
        print("Error: 'rotation_imu' column not found after merge. Check data loading.")
        return pd.DataFrame()

    # --- 4. Calculate Rotational Error (R_err = R_marker.inv() * R_imu) ---
    print("   > Accessing rotation objects from DataFrame columns...")
    try:
        # --- MODIFIED LOGIC ---
        # 1. Access the Rotation objects directly from the 'object' columns
        #    and stack them into a single (N,) Rotation object for batch processing.
        R_imu = Rotation(list(merged_df['rotation_imu'].values))
        R_marker = Rotation(list(merged_df['rotation_marker'].values))
        
        # 2. Calculate the error rotation
        print("   > Calculating error rotation (R_err = R_marker.inv() * R_imu)...")
        R_err = R_marker.inv() * R_imu
        
    except Exception as e:
        print(f"   > CRITICAL Error: Failed to calculate rotational error: {e}.")
        return pd.DataFrame()

    error_cols_dict = {}

    # --- 5. Decompose R_err into 3D Error Magnitude (MAG) ---
    print("   > Decomposing R_err into 3D magnitude ('MAG')...")
    error_magnitude = R_err.magnitude()
    MAGNITUDE_COL_NAME = 'angle_axis_error_magnitude_rad'
    merged_df[MAGNITUDE_COL_NAME] = error_magnitude
    error_cols_dict[MAGNITUDE_COL_NAME] = 'MAG'

    # --- 6. Decompose R_err into Euler Angle Components (X, Y, Z) ---
    print(f"   > Decomposing R_err into Euler angles (order: {euler_order})...")
    
    # Get the (N, 3) array of Euler angle errors
    error_euler_angles = R_err.as_euler(euler_order)
    
    for i, char in enumerate(euler_order): # e.g., 'X', 'Y', 'Z'
        error_col_name = f'error_euler_{char}_rad' # e.g., 'error_euler_X_rad'
        merged_df[error_col_name] = error_euler_angles[:, i]
        error_cols_dict[error_col_name] = char # e.g., 'X'
    
    print(f"   > Calculated error columns: {list(error_cols_dict.keys())}.")
    
    # --- 7. Reshape to Long Format (Melt) ---
    # ... [Unchanged] ...
    meta_cols = [lvl for lvl in join_levels if lvl != 'timestamp'] # subject, trial_type, method, joint_name
    if 'method' not in meta_cols:
        meta_cols.append('method') # Ensure 'method' is included
    error_columns_to_melt = list(error_cols_dict.keys())
    
    print("   > Reshaping DataFrame (melting) from wide to long format...")
    error_long = pd.melt(
        merged_df,
        id_vars=meta_cols + ['timestamp'], # Keep timestamp for potential later use
        value_vars=error_columns_to_melt,
        var_name='error_metric_name',
        value_name='error_rad'
    ).set_index(meta_cols + ['timestamp'])
    
    # --- 8. Clean up 'axis' Names ---
    # ... [Unchanged] ...
    error_long['axis'] = error_long['error_metric_name'].map(error_cols_dict)
    
    all_levels = meta_cols + ['timestamp', 'axis']
    error_long = error_long.reset_index().set_index(all_levels).drop(columns=['error_metric_name'])
    
    # --- 9. Group and Calculate Statistics ---
    # ... [Unchanged logic] ...
    valid_group_by = [lvl for lvl in group_by if lvl in error_long.index.names]
    if not valid_group_by:
        print("Error: No valid group_by keys. Returning empty DataFrame.")
        return pd.DataFrame()

    print(f"   > Grouping data by {valid_group_by} for columns {error_long.columns.tolist()}...")
    grouped = error_long['error_rad'].dropna().groupby(level=valid_group_by)
    
    def q25(x): return x.quantile(0.25)
    def q75(x): return x.quantile(0.75)
    def rmse(x): return np.sqrt(np.mean(x**2))
    def mae(x): return x.abs().mean()
    def mad(x): return (x - x.median()).abs().median()

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
    
    # --- 10. Final Formatting ---
    # ... [Unchanged] ...
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
    
    final_cols = [c for c in final_cols if c in summary_df.columns]
    summary_df = summary_df[final_cols]

    print("--- [get_summary_statistics] Finished. Returning error summary DataFrame. ---")
    return summary_df

def get_pearson_correlation_summary(all_data_df: pd.DataFrame, 
                                    group_by: List[str], 
                                    euler_order: str = 'XYZ') -> pd.DataFrame:
    """
    Calculates the Pearson correlation (R) between the time series of each method 
    and the 'Marker' method. It first decomposes the 'rotation' object column
    into Euler and Angle-Axis components.
    
    Args:
        all_data_df (pd.DataFrame): Must contain a 'rotation' column.
        group_by (List[str]): List of index levels to group by.
        euler_order (str): The euler order (e.g., 'XYZ') to use
                           for decomposing the rotation.
    """
    print("--- [get_pearson_correlation_summary] Starting correlation calculation... ---")

    # --- 1. Prepare DataFrame ---
    if 'subject' in group_by and 'subject_id' in all_data_df.index.names:
        df = all_data_df.rename_axis(index={'subject_id': 'subject'})
    else:
        df = all_data_df.copy()

    if 'rotation' not in df.columns:
        print("Error: 'rotation' column not found. Cannot decompose for correlation.")
        return pd.DataFrame()

    # --- NEW STEP: Decompose Rotation Objects (Vectorized) ---
    print("   > Vectorized decomposition of 'rotation' objects for correlation...")
    
    # 1. Reset index to get a flat DataFrame and access the 'rotation' column
    original_index_names = df.index.names
    df_reset = df.reset_index()
    
    # 2. Stack all rotation objects into one for batch processing
    all_rotations = Rotation(list(df_reset['rotation'].values))
    
    # 3. Decompose to Euler
    print(f"   > Decomposing to Euler (order: {euler_order})...")
    euler_angles = all_rotations.as_euler(euler_order)
    for i, char in enumerate(euler_order):
        df_reset[f'euler_{char}_rad'] = euler_angles[:, i]
        
    # 4. Decompose to Angle-Axis
    print("   > Decomposing to Angle-Axis...")
    rotvecs = all_rotations.as_rotvec()
    df_reset['angle_axis_x_rad'] = rotvecs[:, 0]
    df_reset['angle_axis_y_rad'] = rotvecs[:, 1]
    df_reset['angle_axis_z_rad'] = rotvecs[:, 2]
    
    # 5. Restore the original index
    df = df_reset.set_index(original_index_names)
    
    # --- 2. Separate Data and Get Columns ---
    # The rest of the function now proceeds exactly as before,
    # as the required columns (euler_... and angle_axis_...) now exist.
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

    # --- 3. Merge Data ---
    # ... [Unchanged] ...
    join_levels = marker_df.index.names
    corr_group_levels = [lvl for lvl in join_levels if lvl != 'timestamp']
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
    
    # --- 4. Calculate Correlation ---
    # ... [Unchanged] ...
    corr_cols = {
        col: (f'{col}_imu', f'{col}_marker') 
        for col in all_correlation_cols
    }
    
    print(f"   > Calculating Pearson correlation grouped by {corr_group_levels}...")
    
    def calculate_pair_correlation(group, col_pairs):
        results = {}
        for original_col, (imu_col, marker_col) in col_pairs.items():
            if original_col.startswith('euler_'):
                axis_name = original_col.split('_')[1]
            else: 
                axis_name = 'AA_' + original_col.split('_')[2].upper()
                
            results[axis_name] = group[imu_col].corr(group[marker_col], method='pearson')
        return pd.Series(results, name='PearsonR')

    corr_grouped = merged_df.groupby(corr_group_levels).apply(lambda x: calculate_pair_correlation(x, corr_cols))
    
    correlation_df = corr_grouped.stack().rename('PearsonR').to_frame()
    correlation_df.index.names = corr_group_levels + ['axis'] 

    # --- 5. Final Grouping (if necessary) ---
    # ... [Unchanged] ...
    valid_group_by = [lvl for lvl in group_by if lvl in correlation_df.index.names]
    
    if len(valid_group_by) < len(correlation_df.index.names):
        print(f"   > Re-grouping correlation data by requested levels: {valid_group_by}...")
        correlation_df = correlation_df.groupby(level=valid_group_by).min()
    
    print("--- [get_pearson_correlation_summary] Finished. Returning correlation DataFrame. ---")
    return correlation_df

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

    data_file_path = os.path.join(BASE_DATA_PATH, f"all_subject_data.pkl")
    
    if os.path.exists(data_file_path) and not REGENERATE_DATA_PKLS:
        print(f"Loading existing DataFrame from {data_file_path}...")
        all_data_df = pd.read_pickle(data_file_path)
    else:
        # 2. Loop, load, and collect
        all_subject_dfs = []
        for trial_type in TRIAL_TYPES:
            for subject_id in SUBJECT_IDS:
                subject_pkl_path = os.path.abspath(os.path.join("data", "data", subject_id, trial_type, f"{subject_id}_{trial_type}_data.pkl"))
                print(f"--- Processing {subject_id} - {trial_type} ---")
                if not os.path.exists(subject_pkl_path) or REGENERATE_DATA_PKLS:
                    print(f"Data for {subject_id} - {trial_type} not found. Loading and processing...")
                    try:
                        subject_df = load_joint_traces_for_subject_df(
                            subject_id=subject_id,
                            methods=METHODS,
                            trial_type=trial_type
                        )
                        subject_df.to_pickle(subject_pkl_path)  # Ensure DataFrame is valid
                    except Exception as e:
                        print(f"Error processing {subject_id}: {e}. Skipping subject.")
                        continue
                else:
                    print(f"Data for {subject_id} - {trial_type} already exists. Loading from pickle...")
                    subject_df = pd.read_pickle(subject_pkl_path)
                
                if len(subject_df) > 0:
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
    if os.path.exists(stats_file_path) and not REGENERATE_DATA_PKLS:
        print(f"Loading existing summary statistics from {stats_file_path}...")
        summary_stats_df = pd.read_pickle(stats_file_path)
    else:
        summary_stats_df = get_summary_statistics(
            all_data_df,
            group_by=['trial_type', 'method', 'joint_name', 'subject', 'axis']
        )
        summary_stats_df.to_pickle(stats_file_path)

    print("Summary statistics calculation complete.")
    summary_stats_df.to_csv(os.path.join(BASE_DATA_PATH, f"all_subject_statistics.csv"))

    # --- Get Pearson Correlation Summary ---
    pearson_corr_file_path = os.path.join(BASE_DATA_PATH, f"all_subject_pearson_correlation.pkl")
    if os.path.exists(pearson_corr_file_path) and not REGENERATE_DATA_PKLS:
        print(f"Loading existing Pearson correlation data from {pearson_corr_file_path}...")
        pearson_corr_df = pd.read_pickle(pearson_corr_file_path)
    else:
        pearson_corr_df = get_pearson_correlation_summary(
            all_data_df,
            group_by=['trial_type', 'method', 'joint_name', 'subject', 'axis']
        )
        pearson_corr_df.to_pickle(pearson_corr_file_path)

    print("Pearson correlation calculation complete.")
    pearson_corr_df.to_csv(os.path.join(BASE_DATA_PATH, f"all_subject_pearson_correlation.csv"))