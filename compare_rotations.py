import matplotlib.pyplot as plt
import matplotlib.container as mpc
from typing import List, Dict, Tuple
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from scipy.spatial.transform import Rotation
import numpy as np
import scipy.stats as stats
import itertools
import sys
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
    plate_trials_by_method: Dict[str, List['PlateTrial']] = {}
    world_traces_by_method: Dict[str, Dict[str, 'WorldTrace']] = {}
    plate_trials_by_method['Marker'] = PlateTrial.load_trial_from_folder(
                    f"data/ODay_Data/{subject_id}/{trial_type}"
                )
    min_length = min(len(trial) for trial in plate_trials_by_method['Marker'])
    min_length = min(min_length, 60000)  # Cap at 60000 frames to avoid excessive lengths
    
    try:
        for method in methods:
            if method == 'Marker':
                continue
            else:
                plate_trials_for_method = [trial.copy() for trial in plate_trials_by_method['Marker']]
                if method == 'Madgwick (ODAY)':
                    world_traces_by_method[method] = WorldTrace.load_WorldTraces_from_folder(
                        f"data/ODay_Data/{subject_id}/{trial_type}/IMU"
                    )
                else:
                    world_traces_by_method[method] = WorldTrace.load_from_sto_file(
                        f"data/ODay_Data/{subject_id}/{trial_type}/IMU/{method.lower()}/{trial_type}_orientations.sto"
                    )
                plate_trials_by_method[method] = resync_traces(plate_trials_for_method, world_traces_by_method[method])
                min_length = min(min_length, min(len(trial) for trial in plate_trials_by_method[method]))
                
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

def _perform_statistical_tests(grouped_error, 
                             output_file=sys.stdout, 
                             normality_alpha=0.05, 
                             min_samples=8,
                             normality_test_limit=5000): # <-- [NEW]
    """
    Performs and prints statistical tests on the grouped error data to a file.
    
    Skips slow normality tests for N > normality_test_limit and defaults
    to parametric tests, per the Central Limit Theorem.
    """
    print("\n" + "="*60, file=output_file)
    print(" STATISTICAL SIGNIFICANCE TESTS", file=output_file)
    print(f" (Normality test limit N={normality_test_limit})", file=output_file)
    print("="*60, file=output_file)
    
    if not hasattr(grouped_error, 'groups'):
        print("No groups to analyze. Skipping statistics.", file=output_file)
        print("="*60 + "\n", file=output_file)
        return

    for name, group in grouped_error:
        group = group.dropna(axis=0, how='all')
        if group.empty:
            continue
            
        methods = group.columns.tolist()
        if len(methods) < 1:
            continue

        print(f"\n--- Group: {name} ---", file=output_file)

        # --- 1. Bias Test (vs. Zero Error) ---
        print("  1. Bias Test (Error vs. Zero):", file=output_file)
        if len(methods) == 0:
            print("    - No methods to test.", file=output_file)
            
        for method in methods:
            data = group[method].dropna()
            n_samples = len(data)
            
            if n_samples < 2: 
                print(f"    - {method}: Insufficient data ({n_samples} samples).", file=output_file)
                continue
            
            run_parametric = False
            # [NEW] Logic to decide which test to run
            if n_samples >= normality_test_limit:
                run_parametric = True # Default to parametric for large N (CLT)
            elif 3 <= n_samples < normality_test_limit:
                # Only run Shapiro for "medium" N
                try:
                    _stat_s, p_s = stats.shapiro(data)
                    if n_samples >= min_samples:
                        run_parametric = p_s > normality_alpha
                except ValueError:
                    run_parametric = False # e.g., constant data
            # else (n_samples < 3, or < min_samples), default to non-parametric

            
            if run_parametric:
                # Parametric: One-sample T-test
                if n_samples >= normality_test_limit:
                     print(f"    - {method} (T-test, N={n_samples} > {normality_test_limit}): ", end="", file=output_file)
                else:
                     print(f"    - {method} (T-test, N={n_samples}, normal): ", end="", file=output_file)
                _stat_t, p_t = stats.ttest_1samp(data, 0)
                print(f"Mean={data.mean():.2f}°, p={p_t:.4f}", file=output_file)
            else:
                # Non-Parametric: Wilcoxon signed-rank test
                if n_samples < min_samples:
                    print(f"    - {method} (Wilcoxon, N={n_samples} < {min_samples}): ", end="", file=output_file)
                else:
                    print(f"    - {method} (Wilcoxon, N={n_samples}, non-normal): ", end="", file=output_file)
                try:
                    _stat_w, p_w = stats.wilcoxon(data, alternative='two-sided', zero_method='zsplit')
                    print(f"Median={data.median():.2f}°, p={p_w:.4f}", file=output_file)
                except ValueError as e:
                    print(f"Test failed. (e.g., all values are zero).", file=output_file)

        # --- 2. Pairwise Comparison (Method vs. Method) ---
        if len(methods) < 2:
            continue
            
        print("\n  2. Pairwise Comparison (Method vs. Method):", file=output_file)
        for m1, m2 in itertools.combinations(methods, 2):
            paired_data = group[[m1, m2]].dropna()
            data1 = paired_data[m1]
            data2 = paired_data[m2]
            n_samples = len(data1)
            
            if n_samples < 2:
                print(f"    - {m1} vs {m2}: Insufficient paired data ({n_samples} samples).", file=output_file)
                continue

            diff = data1 - data2
            
            run_parametric = False
            # [NEW] Logic to decide which test to run
            if n_samples >= normality_test_limit:
                run_parametric = True # Default to parametric for large N (CLT)
            elif 3 <= n_samples < normality_test_limit:
                # Only run Shapiro on the *differences*
                try:
                    _stat_s, p_s = stats.shapiro(diff)
                    if n_samples >= min_samples:
                        run_parametric = p_s > normality_alpha
                except ValueError:
                    run_parametric = False
            # else (n_samples < 3, or < min_samples), default to non-parametric

            if run_parametric:
                # Parametric: Paired T-test
                if n_samples >= normality_test_limit:
                    print(f"    - {m1} vs {m2} (Paired T-test, N={n_samples} > {normality_test_limit}): ", end="", file=output_file)
                else:
                    print(f"    - {m1} vs {m2} (Paired T-test, N={n_samples}, normal diff): ", end="", file=output_file)
                _stat_t, p_t = stats.ttest_rel(data1, data2)
                print(f"p={p_t:.4f}", file=output_file)
            else:
                # Non-Parametric: Wilcoxon signed-rank test
                if n_samples < min_samples:
                    print(f"    - {m1} vs {m2} (Wilcoxon, N={n_samples} < {min_samples}): ", end="", file=output_file)
                else:
                    print(f"    - {m1} vs {m2} (Wilcoxon, N={n_samples}, non-normal diff): ", end="", file=output_file)
                try:
                    _stat_w, p_w = stats.wilcoxon(data1, data2, alternative='two-sided', zero_method='zsplit')
                    print(f"p={p_w:.4f}", file=output_file)
                except ValueError as e:
                    print(f"Test failed. (e.g., all differences are zero).", file=output_file)

    print("="*60 + "\n", file=output_file)
# --- Main Execution ---

if __name__ == "__main__":
    # 1. Define the parameters
    # Change this list to include all your subjects!
    SUBJECT_IDS: List[str] = ["Subject01", "Subject02", "Subject03", "Subject04", "Subject05", "Subject06",
                              "Subject07", "Subject08", "Subject09", "Subject10", "Subject11"]
    METHODS: List[str] = ['Marker', 'Cascade', 'Never Project', 'Mag Free', 'Mahony', 'Unprojected', 'Madgwick', 'Madgwick (ODAY)'] # Capitalization is sensitive!
    trial_type = "complexTasks"
    joints = list(JOINT_SEGMENT_DICT.keys()) # Assumes JOINT_SEGMENT_DICT is available
    show_plots = True
    save_plots = False
    match_al_borno_dataset = False
    drop_ankles = False

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
        all_subjects_results = [{} for _ in range(12)]
    print("Data loading and concatenation complete.")

    # --- Get RMSE by all groups ---
    by_all_group_rmse = calculate_rmse_and_std(
        all_data_df,
        group_by=['joint_name', 'axis', 'subject_id', 'axis']
    )
    by_all_group_rmse_file_path = f"data/ODay_Data/rmse_by_all_groups_{trial_type}.pkl"
    by_all_group_rmse.to_pickle(by_all_group_rmse_file_path)

    # --- Filter Dataset as Needed ---
    if match_al_borno_dataset or drop_ankles or METHODS != all_data_df.index.get_level_values('method').unique().tolist() or SUBJECT_IDS != all_data_df.index.get_level_values('subject_id').unique().tolist():
        print("Filtering dataset based on specified criteria...")
        if match_al_borno_dataset:
            # Drop the following: "R_Ankle" for Subject 1 5 6 9 10 and 11, "L_Ankle" for Subject 1 8 9 and 11
            if trial_type == 'walking':
                al_borno_joints = [
                    ("R_Ankle", [1, 5, 6, 9, 10, 11]),
                    ("L_Ankle", [1, 5, 8, 9, 11]),
                ]
            else:
                al_borno_joints = [
                    ("R_Ankle", []),
                    ("L_Ankle", [1, 8, 9]),
                ]
            for joint_name, subject_nums in al_borno_joints:
                for subject_num in subject_nums:
                    subject_id = f"Subject{subject_num:02d}"
                    idx_to_drop = all_data_df.index.get_level_values('subject_id') == subject_id
                    idx_to_drop &= all_data_df.index.get_level_values('joint_name') == joint_name
                    all_data_df = all_data_df[~idx_to_drop]

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

    # --- PLOT BY JOINT AND AXIS RMSE ---
    by_joint_rmse = calculate_rmse_and_std(
        all_data_df,
        group_by=['joint_name', 'axis']
    )
    plot_rmse_with_std(
        by_joint_rmse,
        title=f"RMSE by Joint and Axis for {trial_type}",
        y_label="RMSE (degrees)",
        save_plot=save_plots,
        show_plot=show_plots
    )

    # --- PLOT BY SUBJECT RMSE ---
    print("Calculating and plotting RMSE by Subject...")
    by_subject_rmse = calculate_rmse_and_std(
        all_data_df,
        group_by=['subject_id']
    )
    plot_rmse_with_std(
        by_subject_rmse,
        title=f"RMSE by Subject for {trial_type}{' (AL Borno Matched)' if match_al_borno_dataset else ''}",
        y_label="RMSE (degrees)",
        save_plot=save_plots,
        show_plot=show_plots
    )

    # --- PLOT RMSE ---
    print("Calculating and plotting Overall RMSE...")
    overall_rmse_file_path = f"data/ODay_Data/overall_rmse_{trial_type}_{match_al_borno_dataset}.pkl"
    # if os.path.exists(overall_rmse_file_path):
    #     print(f"Loading existing Overall RMSE DataFrame from {overall_rmse_file_path}...")
    #     overall_rmse = pd.read_pickle(overall_rmse_file_path)
    # else:
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
        title=f"Overall RMSE Across All Subjects and Joints for {trial_type}{' (AL Borno Matched)' if match_al_borno_dataset else ''}",
        y_label="RMSE (degrees)",
        save_plot=save_plots,
        show_plot=show_plots
    )

    # --- PLOT SINGLE SUBJECT RMSE ---
    for subject_id in SUBJECT_IDS:
        try: 
            print(f"Calculating and plotting RMSE for {subject_id}...")
            all_subject_data_df = all_data_df.xs(subject_id, level='subject_id')
            subject_error_df = calculate_rmse_and_std(
                all_subject_data_df,
                group_by=['joint_name', 'axis']
            )
            plot_rmse_with_std(
                subject_error_df,
                title=f"RMSE for {subject_id} by Joint and Axis for {trial_type}",
                y_label="RMSE (degrees)",
                save_plot=save_plots,
                show_plot=show_plots
            )
        except KeyError:
            print(f"No data found for {subject_id}, skipping RMSE plot.")
            continue