import matplotlib.pyplot as plt
import matplotlib.container as mpc
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

# --- Data Loading Functions ---

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
    if not results_df.empty and isinstance(results_df.columns, pd.MultiIndex):
        results_df = results_df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    return results_df

def plot_rmse_with_std(stats_df: pd.DataFrame, title: str, y_label: str = "RMSE (units)", save_plot: bool = False, show_plot: bool = True):
    """
    Generates a bar plot of RMSE with STD as error bars.
    
    This function assumes the input DataFrame 'stats_df' has a 
    MultiIndex on the columns, with:
      - Level 0: Method name (e.g., 'Madgwick', 'Mag-Free')
      - Level 1: Metric name ('RMSE', 'STD')
      
    The DataFrame's index is used for the x-axis categories, allowing
    for varied groupings (e.g., by subject, by joint/axis).

    Args:
        stats_df (pd.DataFrame): The DataFrame containing the statistics.
        title (str): The title for the plot.
        y_label (str): The label for the y-axis.
        save_plot (bool): If True, saves the plot to a PNG file.
        show_plot (bool): If True, displays the plot.
    """
    
    # --- 1. Validate and Prepare Data ---
    if not isinstance(stats_df.columns, pd.MultiIndex) or stats_df.columns.nlevels < 2:
        print("Error: DataFrame columns must be a MultiIndex with at least 2 levels (method, metric).")
        return

    try:
        # Select the RMSE data for the bar heights
        rmse_data = stats_df.xs('RMSE', level=1, axis=1)
        
        # Select the STD data for the error bars
        std_data = stats_df.xs('STD', level=1, axis=1)
        
        # Ensure the columns match in case some methods don't have STD
        common_methods = rmse_data.columns.intersection(std_data.columns)
        rmse_data = rmse_data[common_methods]
        std_data = std_data[common_methods]
        
    except KeyError as e:
        print(f"Error: Could not find '{e.args[0]}' in column level 1. "
              "DataFrame must contain 'RMSE' and 'STD' metrics.")
        return
    except Exception as e:
        print(f"An error occurred during data selection: {e}")
        return

    if rmse_data.empty:
        print("No data left to plot after filtering for RMSE and STD.")
        return

    # --- 2. Create the Plot ---
    
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

    # Plot the bar chart
    rmse_data.plot(
        kind='bar',
        yerr=std_data,       # Use the STD data for error bars
        ax=ax,
        capsize=4,           # Adds caps to the error bars
        width=0.8,           # Adjust bar width
        edgecolor='black',   # Add edge color for clarity
        alpha=0.8
    )

    # --- 3. Customize and Style ---
    
    # --- 3a. Add Data Labels ---
    max_plot_height = 0
    try:
        methods = rmse_data.columns
        bar_containers = [c for c in ax.containers if isinstance(c, mpc.BarContainer)]

        for i, container in enumerate(bar_containers): # Iterate over the filtered list
            if i >= len(methods):
                print(f"Warning: Found more bar containers ({len(bar_containers)}) than methods ({len(methods)}). Skipping extra containers.")
                break
                
            method = methods[i]
            
            # Get the corresponding RMSE and STD data
            rmse_values = rmse_data[method]
            std_values = std_data[method]
            
            # Find the max height (bar + std) for this container
            current_max = (rmse_values + std_values).max()
            if current_max > max_plot_height:
                max_plot_height = current_max
            
            # Create custom labels: "RMSE ± STD"
            # Format to 2 decimal places (e.g., .2f)
            labels = [f"{r:.2f} ± {s:.2f}" for r, s in zip(rmse_values, std_values)]
            
            # Add the labels to the bars in this container
            ax.bar_label(
                container,
                labels=labels,
                padding=3,           # Space above the error bar
                fontsize=8,          # Smaller font size
                rotation=90,         # Rotate for readability
                fontweight='normal'
            )
    except (AttributeError, IndexError, KeyError) as e:
        # AttributeError if ax.containers doesn't exist
        print(f"Warning: Could not add bar labels. Error: {e}")

    # --- 3b. Adjust Y-Limit for Labels ---
    # Give 25% extra space above the highest error bar for the rotated labels
    if max_plot_height > 0:
        ax.set_ylim(top=max_plot_height * 1.25) 
    
    # --- 3c. Set titles and labels ---
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel(stats_df.index.name or 'Group', fontsize=12)
    
    # --- 3d. Customize legend ---
    ax.legend(title="Method", title_fontsize=11, fontsize=10, loc='upper right')
    
    # --- 3e. Customize grid and ticks ---
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True) # Put grid behind bars
    
    # --- 3f. Improve x-axis label readability ---
    num_idx_levels = stats_df.index.nlevels
    
    if len(stats_df.index) > 5 or num_idx_levels > 1:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    else:
        plt.setp(ax.get_xticklabels(), rotation=0)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- 4. Show and/or Save the Plot ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if save_plot:
            plt.savefig(f"{title.replace(' ', '_').lower()}_rmse_plot.png", dpi=150)
            print(f"Plot saved as {title.replace(' ', '_').lower()}_rmse_plot.png")
        if show_plot:
            plt.show()
        
        # If not showing, close the plot to free memory
        if not show_plot:
            plt.close(fig)

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
    match_al_borno_dataset = False

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
        
        # Save the DataFrame to a pickle file
        all_data_df.to_pickle(file_path)

        print(f"DataFrame saved to {file_path}")
        all_subjects_results = [{} for _ in range(12)]
    print("Data loading and concatenation complete.")

    if match_al_borno_dataset:
        # Drop the following: "R_Ankle" for Subject 1 5 6 9 10 and 11, "L_Ankle" for Subject 1 8 9 and 11
        al_borno_joints = [
            ("R_Ankle", [1, 5, 6, 9, 10, 11]),
            ("L_Ankle", [1, 8, 9, 11]),
        ]
        for joint_name, subject_nums in al_borno_joints:
            for subject_num in subject_nums:
                subject_id = f"Subject{subject_num:02d}"
                idx_to_drop = all_data_df.index.get_level_values('subject_id') == subject_id
                idx_to_drop &= all_data_df.index.get_level_values('joint_name') == joint_name
                all_data_df = all_data_df[~idx_to_drop]

    # --- PLOT BY JOINT AND AXIS RMSE ---
    by_joint_rmse_file_path = f"data/ODay_Data/rmse_by_joint_and_axis_{trial_type}_{match_al_borno_dataset}.pkl"
    if os.path.exists(by_joint_rmse_file_path):
        print(f"Loading existing RMSE by Joint and Axis DataFrame from {by_joint_rmse_file_path}...")
        by_joint_rmse = pd.read_pickle(by_joint_rmse_file_path)
    else:
        by_joint_rmse = calculate_rmse_and_std(
            all_data_df,
            group_by=['joint_name', 'axis']
        )
        by_joint_rmse.to_pickle(by_joint_rmse_file_path)
    plot_rmse_with_std(
        by_joint_rmse,
        title=f"RMSE by Joint and Axis for {trial_type} {'(AL Borno Matched)' if match_al_borno_dataset else ''}",
        y_label="RMSE (degrees)",
        save_plot=save_plots,
        show_plot=show_plots
    )

    # --- PLOT BY SUBJECT RMSE ---
    print("Calculating and plotting RMSE by Subject...")
    by_subject_rmse_file_path = f"data/ODay_Data/rmse_by_subject_{trial_type}_{match_al_borno_dataset}.pkl"
    if os.path.exists(by_subject_rmse_file_path):
        print(f"Loading existing RMSE by Subject DataFrame from {by_subject_rmse_file_path}...")
        by_subject_rmse = pd.read_pickle(by_subject_rmse_file_path)
    else:
        by_subject_rmse = calculate_rmse_and_std(
            all_data_df,
            group_by=['subject_id']
        )
        by_subject_rmse.to_pickle(by_subject_rmse_file_path)
    plot_rmse_with_std(
        by_subject_rmse,
        title=f"RMSE by Subject for {trial_type} {'(AL Borno Matched)' if match_al_borno_dataset else ''}",
        y_label="RMSE (degrees)",
        save_plot=save_plots,
        show_plot=show_plots
    )

    # --- PLOT RMSE ---
    print("Calculating and plotting Overall RMSE...")
    overall_rmse_file_path = f"data/ODay_Data/overall_rmse_{trial_type}_{match_al_borno_dataset}.pkl"
    if os.path.exists(overall_rmse_file_path):
        print(f"Loading existing Overall RMSE DataFrame from {overall_rmse_file_path}...")
        overall_rmse = pd.read_pickle(overall_rmse_file_path)
    else:
         overall_rmse = calculate_rmse_and_std(
            all_data_df,
            group_by=[]
        )
         overall_rmse.to_pickle(overall_rmse_file_path) 
    if not overall_rmse.empty:
        overall_rmse = overall_rmse.stack().to_frame().T
        overall_rmse.index = ['Overall']
        overall_rmse.index.name = 'Group'
    plot_rmse_with_std(
        overall_rmse,
        title=f"Overall RMSE Across All Subjects and Joints for {trial_type} {'(AL Borno Matched)' if match_al_borno_dataset else ''}",
        y_label="RMSE (degrees)",
        save_plot=save_plots,
        show_plot=show_plots
    )