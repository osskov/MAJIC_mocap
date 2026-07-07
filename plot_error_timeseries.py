import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns # Import seaborn
from typing import List, Dict

# --- Configuration (can be imported or defined here) ---
BASE_DATA_PATH = os.path.abspath(os.path.join("data"))
ALL_SUBJECTS = ["Subject01", "Subject02", "Subject03", "Subject04", "Subject05", "Subject06",
                "Subject07", "Subject08", "Subject09", "Subject10", "Subject11"]
ALL_METHODS = ['Marker', 
                'EKF', #'Madgwick (Al Borno)',  'Mahony', 'Madgwick',
               'Mag On', 'Mag Off', 'Mag Adapt']#, 'Unprojected']
ALL_JOINTS = ['R_Ankle', 'L_Ankle', 'R_Knee', 'L_Knee', 'R_Hip', 'L_Hip', 'Lumbar']
ALL_TRIAL_TYPES = ['walking', 'complexTasks']

# --- Plotting Function ---
def plot_multi_joint_error_time_series(
    all_data_df: pd.DataFrame,
    subject_id: str,
    trial_type: str,
    joint_names: List[str], # Now accepts a list of joint names
    methods_to_plot: List[str], # New parameter for specific methods to plot
    plot_duration_s: int = 60, # New parameter for plot duration
    output_dir: str = "plots"
):
    """
    Plots the time series of only the angle-axis error magnitude (IMU method relative to Marker)
    for a specified subject and trial type.
    Selected joints are plotted as subplots on the same figure.
    Only specified IMU methods' error magnitude are overlaid.
    The plot is limited to the middle `plot_duration_s` seconds of the trial.
    'Mag Adapt' method is renamed to 'MAJIC' for plotting.
    Uses Seaborn's 'Set2' palette for consistent colors.

    Args:
        all_data_df (pd.DataFrame): The DataFrame containing all joint angle-axis data.
                                    Expected to have 'subject_id', 'trial_type',
                                    'method', 'joint_name', 'timestamp' as index levels
                                    and 'angle_axis_x_rad', 'y', 'z' as columns.
        subject_id (str): The ID of the subject to plot.
        trial_type (str): The type of trial to plot (e.g., 'walking', 'complexTasks').
        joint_names (List[str]): A list of joint names to plot as subplots.
        methods_to_plot (List[str]): A list of specific IMU methods to include in the plot.
        plot_duration_s (int): The duration in seconds to plot, centered in the trial.
        output_dir (str): Directory to save the plots.
    """
    print(f"Generating multi-joint magnitude error time series plot for Subject: {subject_id}, Trial: {trial_type}, Joints: {joint_names}, Methods: {methods_to_plot}")
    print(f"DEBUG: Initial all_data_df index names: {all_data_df.index.names}")

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(len(joint_names), 1, figsize=(15, 5 * len(joint_names)), sharex=True)
    if len(joint_names) == 1: # Ensure axes is always an array for consistent indexing
        axes = [axes]
    
    fig.suptitle(f'{subject_id} - {trial_type} - Angle-Axis Magnitude Error Time Series (IMU vs. Marker)', fontsize=16)

    all_methods_plotted = set() # To collect all methods for a single legend

    # Set Seaborn palette
    colors = sns.color_palette('Set2', n_colors=len(methods_to_plot))
    method_color_map = {method: colors[i] for i, method in enumerate(methods_to_plot)}


    for i, joint_name in enumerate(joint_names):
        ax = axes[i]
        
        try:
            # Filter for the specific subject, trial, and joint from the MultiIndex
            # When specific values are selected, those levels are dropped from the index/columns
            filtered_data = all_data_df.loc[
                (subject_id, trial_type, slice(None), joint_name, slice(None))
            ].reset_index()

            # CRITICAL FIX: Re-add the subject_id, trial_type, and joint_name as columns
            # since they were dropped by .loc selection + .reset_index()
            filtered_data['subject_id'] = subject_id
            filtered_data['trial_type'] = trial_type
            filtered_data['joint_name'] = joint_name

            if filtered_data.empty:
                print(f"No data found for {subject_id}, {trial_type}, {joint_name}. Skipping subplot.")
                ax.set_title(f'{joint_name} (No Data)')
                continue

            # Ensure timestamps are numeric AFTER resetting the index
            filtered_data.loc[:, 'timestamp'] = pd.to_numeric(filtered_data['timestamp'])

            # --- Apply time window trimming ---
            min_time = filtered_data['timestamp'].min()
            max_time = filtered_data['timestamp'].max()
            total_duration = max_time - min_time

            if total_duration <= plot_duration_s:
                start_time_plot = min_time
                end_time_plot = max_time
                print(f"   > Plotting full duration for {joint_name} as total duration ({total_duration:.2f}s) <= {plot_duration_s}s.")
            else:
                middle_time = min_time + total_duration / 2
                start_time_plot = middle_time - plot_duration_s / 2
                end_time_plot = middle_time + plot_duration_s / 2
                # Ensure start/end times are within actual data range
                start_time_plot = max(min_time, start_time_plot)
                end_time_plot = min(max_time, end_time_plot)
                print(f"   > Trimming {joint_name} to middle {plot_duration_s}s: [{start_time_plot:.2f}s, {end_time_plot:.2f}s]")

            filtered_data = filtered_data[
                (filtered_data['timestamp'] >= start_time_plot) & 
                (filtered_data['timestamp'] <= end_time_plot)
            ]
            if filtered_data.empty:
                print(f"No data in the middle {plot_duration_s}s window for {joint_name}. Skipping subplot.")
                ax.set_title(f'{joint_name} (No Data in Window)')
                continue
            # --- End time window trimming ---

            # Now, separate Marker data and IMU data from this *already flattened* DataFrame
            marker_data = filtered_data[filtered_data['method'] == 'Marker'].copy()
            imu_data = filtered_data[filtered_data['method'] != 'Marker'].copy()
            
            # Further filter IMU data to only include specified methods
            imu_data = imu_data[imu_data['method'].isin(methods_to_plot)].copy()


            if marker_data.empty:
                print(f"No 'Marker' data found for {subject_id}, {trial_type}, {joint_name}. Cannot calculate error. Skipping subplot.")
                ax.set_title(f'{joint_name} (No Marker Data)')
                continue
            if imu_data.empty:
                print(f"No IMU data found for {subject_id}, {trial_type}, {joint_name} with selected methods. Skipping subplot.")
                ax.set_title(f'{joint_name} (No Selected IMU Data)')
                continue
                
            # Define columns for merging. These are now definitely columns in both marker_data and imu_data
            merge_cols = ['subject_id', 'trial_type', 'joint_name', 'timestamp']
            angle_axis_cols = [col for col in filtered_data.columns if col.startswith('angle_axis_')]

            marker_data_for_merge = marker_data[merge_cols + angle_axis_cols]
            imu_data_for_merge = imu_data[merge_cols + ['method'] + angle_axis_cols]

            merged_data = pd.merge(
                imu_data_for_merge,
                marker_data_for_merge,
                on=merge_cols,
                suffixes=('_imu', '_marker')
            )
            
            if merged_data.empty:
                print(f"Merged data for {subject_id}, {trial_type}, {joint_name} is empty. Skipping subplot.")
                ax.set_title(f'{joint_name} (Merge Empty)')
                continue

            # Calculate angle-axis error components (still needed for magnitude)
            for axis in ['x', 'y', 'z']:
                merged_data[f'error_aa_{axis}_rad'] = (
                    merged_data[f'angle_axis_{axis}_rad_imu'] - merged_data[f'angle_axis_{axis}_rad_marker']
                )
            
            # Calculate error magnitude
            merged_data['error_aa_magnitude_rad'] = np.linalg.norm(
                merged_data[[f'error_aa_{axis}_rad' for axis in ['x', 'y', 'z']]].values,
                axis=1
            )
            
            # Convert magnitude error to degrees for plotting
            merged_data['error_aa_magnitude_deg'] = np.rad2deg(merged_data['error_aa_magnitude_rad'])

            # Plot magnitude error for each method
            for method in merged_data['method'].unique():
                plot_method_name = method # Use original name by default
                if method == 'Mag Adapt':
                    plot_method_name = 'MAJIC' # Rename for plotting
                
                method_data = merged_data[merged_data['method'] == method]
                # Only add label if it hasn't been added yet for the legend
                line, = ax.plot(
                    method_data['timestamp'], 
                    method_data['error_aa_magnitude_deg'], 
                    label=plot_method_name,
                    color=method_color_map.get(method) # Use color from palette
                )
                all_methods_plotted.add((plot_method_name, line.get_color())) # Store method and its color

            ax.set_title(f'{joint_name} Angle-Axis Magnitude Error')
            ax.set_ylabel('Error (deg)')
            ax.grid(True, linestyle='--', alpha=0.7)

        except Exception as e:
            print(f"An error occurred while plotting for {subject_id}, {trial_type}, {joint_name}: {e}")
            ax.set_title(f'{joint_name} (Error during plot: {e})')
            continue

    # Create a single legend for the entire figure using collected labels and colors
    # This avoids duplicate legends on each subplot
    if all_methods_plotted:
        # Sort methods for consistent legend order if desired
        sorted_methods = sorted(list(all_methods_plotted), key=lambda x: x[0])
        legend_handles = [plt.Line2D([0], [0], color=color, lw=2, label=name) for name, color in sorted_methods]
        fig.legend(handles=legend_handles, title='Method', loc='center right', bbox_to_anchor=(0.98, 0.5))

    axes[-1].set_xlabel('Timestamp (s)') # Only the bottom subplot needs an x-label
    plt.tight_layout(rect=[0, 0, 0.88, 0.96]) # Adjust layout to make space for the legend on the right

    plot_filename = os.path.join(output_dir, f"{subject_id}_{trial_type}_multi_joint_magnitude_error_timeseries_selected_methods.png")
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Plot saved to {plot_filename}")

# --- Main Execution Block ---
if __name__ == "__main__":
    data_file_path = os.path.join(BASE_DATA_PATH, f"all_subject_data.pkl")

    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}.")
        print("Please run the previous script (the one you provided) to generate 'all_subject_data.pkl' first.")
        exit()

    print(f"Loading all_subject_data.pkl from {data_file_path}...")
    all_data_df = pd.read_pickle(data_file_path)
    print("DataFrame loaded.")
    print(f"DEBUG: all_data_df index names after loading: {all_data_df.index.names}")

    selected_subjects = ["Subject02"]
    selected_trial_types = ["walking"]
    selected_joints_to_plot = ["L_Ankle", "Lumbar"] # List of joints for subplots
    
    # New: Specify only the methods to plot
    methods_for_plotting = ['EKF', 'Mag Off', 'Mag On', 'Mag Adapt'] # 'Mag Adapt' will be renamed to 'MAJIC'

    for subject in selected_subjects:
        for trial in selected_trial_types:
            # Call the new function for multiple joints and selected methods
            plot_multi_joint_error_time_series(
                all_data_df, 
                subject, 
                trial, 
                selected_joints_to_plot, 
                methods_for_plotting, # Pass the new list of methods
                plot_duration_s=60
            )

    print("\nTime series plotting complete.")