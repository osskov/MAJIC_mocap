import matplotlib.pyplot as plt
from typing import List, Dict, Union

from matplotlib.lines import Line2D
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from scipy.spatial.transform import Rotation
import numpy as np
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import pandas as pd
import warnings


from paper_data_pipeline.generate_kinematics_from_trial import JOINT_SEGMENT_DICT

# --- Global Constants and Helper Functions ---

# This function must be used to resync traces for each subject/trial combination
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
            # print(f"Warning: No target world trace found for trial {trial.name}. Skipping.")
            continue
        
        # NOTE: The original code modified the world_trace of the input trials.
        # This is preserved here for consistency with the original script's logic flow.
        trial.world_trace = target_world_trace
        
        if abs(trial.imu_trace.get_sample_frequency() - target_world_trace.get_sample_frequency()) > 0.2:
            print(f"Sample frequency mismatch for {trial.name}: IMU {trial.imu_trace.get_sample_frequency()} Hz, World {target_world_trace.get_sample_frequency()} Hz")
            trial.imu_trace = trial.imu_trace.resample(target_world_trace.get_sample_frequency())
        
        # PlateTrial._sync_traces is assumed to be a static method of PlateTrial
        if imu_slice == slice(0, 0) and world_slice == slice(0, 0):
            imu_slice, world_slice = PlateTrial._sync_traces(trial.imu_trace, trial.world_trace)
        synced_imu_trace = trial.imu_trace[imu_slice].re_zero_timestamps()
        synced_world_trace = trial.world_trace[world_slice].re_zero_timestamps()
        new_plate_trial = PlateTrial(trial.name, synced_imu_trace, synced_world_trace)
        new_plate_trial = new_plate_trial._align_world_trace_to_imu_trace()
        
        new_plate_trials.append(new_plate_trial)
        
    return new_plate_trials

# This function calculates the joint traces (R_parent_world^T * R_child_world)
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

# This function calculates the error metrics
def calculate_joint_errors_rmse_std(
    marker_traces: Dict[str, 'WorldTrace'], 
    imu_traces_list: List[Dict[str, 'WorldTrace']], 
    method_names: List[str], 
    joints: List[str]
) -> Dict[str, Dict[str, Dict[str, tuple]]]:
    """
    Calculates the RMSE and STD of Euler angle error for each IMU method 
    relative to the Marker data, for each joint and axis.

    Returns:
        A nested dictionary: {MethodName: {JointName: {AxisName: (RMSE, STD)}}}
    """
    
    axis_names = ['X', 'Y', 'Z']
    all_results: Dict[str, Dict[str, Dict[str, tuple]]] = {
        method: {joint: {} for joint in joints} for method in method_names
    }

    for k, trace in enumerate(imu_traces_list):
        method_name = method_names[k]
        
        for joint in joints:
            if joint in trace and joint in marker_traces:
                imu_joint_trace = trace[joint]
                marker_joint_trace = marker_traces[joint]
                
                if len(imu_joint_trace) != len(marker_joint_trace):
                    continue
                
                # Calculate the error rotation R_error = R_marker @ R_imu.T
                error_rotations = [R_m @ R_i.T for R_m, R_i in zip(marker_joint_trace.rotations, imu_joint_trace.rotations)]
                
                # Convert error rotations to Euler angles (XYZ order)
                angles_rad_list = [Rotation.from_matrix(rot).as_euler('xyz', degrees=False) for rot in error_rotations]
                angles_deg = np.array(angles_rad_list) * (180 / np.pi) 

                for j in range(3):
                    axis_name = axis_names[j]
                    data_degrees = angles_deg[:, j]
                    
                    rmse = np.sqrt(np.mean(data_degrees**2))
                    std = np.std(data_degrees)
                    
                    all_results[method_name][joint][axis_name] = (rmse, std)

    return all_results

# --- NEW FUNCTIONS FOR MULTI-SUBJECT ANALYSIS ---

def get_subject_error_metrics(subject_id: str, trial_type: str) -> Dict[str, Dict[str, Dict[str, tuple]]]:
    """
    Loads data, resyncs, calculates joint traces, and computes RMSE/STD 
    for a single subject.
    """
    
    # 1. Load Data
    try:
        # Load Marker trials (with world trace = True for initial loading, 
        # then False for IMU methods to avoid re-using the same objects)
        plate_trials: List['PlateTrial'] = PlateTrial.load_trial_from_folder(f"data/ODay_Data/{subject_id}/{trial_type}", True)
        
        # Load fresh sets of PlateTrials for each IMU method
        plate_trials_madgwick: List['PlateTrial'] = plate_trials.copy()
        plate_trials_mag_free: List['PlateTrial'] = plate_trials.copy()
        plate_trials_never_project: List['PlateTrial'] = plate_trials.copy()

        # Load WorldTraces (Orientations)
        madgwick_world_traces: Dict[str, 'WorldTrace'] = WorldTrace.load_WorldTraces_from_folder(
            f"data/ODay_Data/{subject_id}/{trial_type}/IMU")
        mag_free_world_traces: Dict[str, 'WorldTrace'] = WorldTrace.load_from_sto_file(
            f"data/ODay_Data/{subject_id}/{trial_type}/IMU/mag free/{trial_type}_orientations.sto")
        never_project_world_traces: Dict[str, 'WorldTrace'] = WorldTrace.load_from_sto_file(
            f"data/ODay_Data/{subject_id}/{trial_type}/IMU/never project/{trial_type}_orientations.sto")
    except Exception as e:
        print(f"Error loading data for {subject_id}: {e}. Skipping subject.")
        return {} # Return empty dict on failure

    # 2. Resync Traces (assigning WorldTrace and aligning)
    plate_trials_madgwick = resync_traces(plate_trials_madgwick, madgwick_world_traces)
    plate_trials_mag_free = resync_traces(plate_trials_mag_free, mag_free_world_traces)
    plate_trials_never_project = resync_traces(plate_trials_never_project, never_project_world_traces)
    
    # 3. Trim to Minimum Length (Essential for alignment)
    all_trials_lists = [plate_trials, plate_trials_madgwick, plate_trials_mag_free, plate_trials_never_project]
    
    # Get lengths of all trials across all methods/lists
    all_lengths = [len(trial) for trials_list in all_trials_lists for trial in [plate_trials, plate_trials_madgwick, plate_trials_mag_free, plate_trials_never_project] for trial in trials_list]

    if not all_lengths:
        print(f"No valid trials found for {subject_id} after resync/trim. Skipping subject.")
        return {}

    current_min_length = min(all_lengths)
    
    # Perform trimming to the minimum length
    # Note: We loop over the number of trials in the original list for consistency
    for trials_list in all_trials_lists:
        for i in range(len(trials_list)):
            trials_list[i] = trials_list[i][:current_min_length]

    # 4. Calculate Joint Traces (R_parent_world^T * R_child_world)
    marker_joint_traces = get_joint_traces(plate_trials)
    madgwick_joint_traces = get_joint_traces(plate_trials_madgwick)
    mag_free_joint_traces = get_joint_traces(plate_trials_mag_free)
    never_projected_joint_traces = get_joint_traces(plate_trials_never_project)
    
    # 5. Calculate Error Metrics
    joints = list(JOINT_SEGMENT_DICT.keys())
    method_names_for_error = ['Madgwick', 'Mag-Free', 'Never Project']
    trace_list_for_error = [madgwick_joint_traces, mag_free_joint_traces, never_projected_joint_traces]
    
    rmse_std_results = calculate_joint_errors_rmse_std(
        marker_traces=marker_joint_traces,
        imu_traces_list=trace_list_for_error,
        method_names=method_names_for_error,
        joints=joints
    )
    
    return rmse_std_results

def aggregate_error_metrics(
    all_subjects_results: List[Dict[str, Dict[str, Dict[str, tuple]]]]
) -> Dict[str, Dict[str, Dict[str, tuple]]]:
    """
    Aggregates error metrics across multiple subjects by calculating the 
    mean RMSE and mean STD for each (Method, Joint, Axis) combination.
    """
    
    # Structure: {Method: {Joint: {Axis: [ (RMSE1, STD1), (RMSE2, STD2), ... ]}}}
    all_data = {} 
    
    # 1. Collect all data points
    for subject_results in all_subjects_results:
        for method, joint_results in subject_results.items():
            if method not in all_data:
                all_data[method] = {}
            for joint, axis_results in joint_results.items():
                if joint not in all_data[method]:
                    all_data[method][joint] = {}
                for axis, (rmse, std) in axis_results.items():
                    if axis not in all_data[method][joint]:
                        all_data[method][joint][axis] = []
                    all_data[method][joint][axis].append((rmse, std))

    # 2. Calculate mean RMSE and mean STD
    aggregated_results = {}
    for method, joint_results in all_data.items():
        aggregated_results[method] = {}
        for joint, axis_results in joint_results.items():
            aggregated_results[method][joint] = {}
            for axis, data_points in axis_results.items():
                # Separate RMSEs and STDs
                rmses = np.array([d[0] for d in data_points])
                stds = np.array([d[1] for d in data_points])
                
                # Calculate mean (combined) metrics
                mean_rmse = np.mean(rmses)
                mean_std = np.mean(stds)
                
                aggregated_results[method][joint][axis] = (mean_rmse, mean_std)
                
    return aggregated_results

def calculate_overall_mean_error(
    aggregated_rmse_std: Dict[str, Dict[str, Dict[str, tuple]]],
    method_names: List[str]
) -> Dict[str, tuple]:
    """
    Aggregates the mean RMSE and mean STD across all joints and all axes 
    for each method.
    
    Returns:
        A dictionary: {MethodName: (Overall Mean RMSE, Overall Mean STD)}
    """
    
    overall_results: Dict[str, tuple] = {}
    
    for method_name in method_names:
        method_rmses = []
        method_stds = []
        
        if method_name in aggregated_rmse_std:
            for joint in aggregated_rmse_std[method_name]:
                for axis in aggregated_rmse_std[method_name][joint]:
                    # These are the means already aggregated across all subjects
                    mean_rmse, mean_std = aggregated_rmse_std[method_name][joint][axis]
                    method_rmses.append(mean_rmse)
                    method_stds.append(mean_std)
        
        if method_rmses:
            # Calculate the mean of the mean-RMSEs and mean of the mean-STDs
            overall_mean_rmse = np.mean(method_rmses)
            overall_mean_std = np.mean(method_stds)
            overall_results[method_name] = (overall_mean_rmse, overall_mean_std)
            
    return overall_results

# --- New Plotting Function ---

def plot_overall_mean_comparison(
    overall_results: Dict[str, tuple], 
    method_names: List[str]
):
    """
    Creates a single bar plot comparing the Overall Mean RMSE and Overall Mean STD 
    of Euler angle error across all joints and axes, aggregated across subjects.
    Saves the plot to 'overall_mean_comparison.png'.
    """
    
    fig, ax = plt.subplots(figsize=(6, 8))
    plt.title('Overall Mean RMSE of Euler Angle Error (IMU vs. Marker)', fontsize=16)
    ax.set_ylabel('Overall Mean Error (Degrees)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

    method_colors = {'Madgwick': 'lightblue', 'Mag-Free': 'lightgreen', 'Never Project': 'orange'} 
    
    methods_to_plot = [m for m in method_names if m in overall_results]
    
    x_positions = np.arange(len(methods_to_plot))
    bar_heights = [overall_results[m][0] for m in methods_to_plot] # Overall Mean RMSE
    error_whiskers = [overall_results[m][1] for m in methods_to_plot] # Overall Mean STD
    bar_colors = [method_colors[m] for m in methods_to_plot]
    
    bar_width = 0.5 

    # Draw Bars and Error Bars
    ax.bar(x_positions, bar_heights, width=bar_width, color=bar_colors, edgecolor='black', zorder=3)
    ax.errorbar(x_positions, bar_heights, yerr=error_whiskers, fmt='none', ecolor='k', capsize=6, zorder=4)

    # Label the bars with Mean RMSE +/- Mean STD
    max_label_y = 0.0
    for i in range(len(x_positions)):
        x = x_positions[i]
        rmse = bar_heights[i]
        std = error_whiskers[i]
        
        label_string = f'{rmse:.2f} $\\pm$ {std:.2f}$^\circ$' 
        y_pos = rmse + error_whiskers[i] + 0.5
        max_label_y = max(max_label_y, y_pos)
        
        ax.text(x, y_pos, label_string, 
                ha='center', va='bottom', 
                fontsize=10, 
                rotation=0) 

    # Set x-ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods_to_plot, fontsize=12)
    
    # Set Y-axis limits
    if max_label_y > 0:
        ax.set_ylim(bottom=0, top=max_label_y * 1.05) 

    plt.tight_layout()
    plt.savefig("overall_mean_comparison.png")
# --- MODIFIED PLOTTING FUNCTION ---

def plot_mean_rmse_std_comparison(
    rmse_std_results: Dict[str, Dict[str, Dict[str, tuple]]], 
    method_names: List[str], 
    joints: List[str]
):
    """
    Creates a bar plot comparing the MEAN RMSE and MEAN STD of Euler angle error 
    for different IMU methods relative to Marker data, aggregated across subjects.
    Saves the plot to 'mean_rmse_std_comparison.png'.
    """
    
    num_joints = len(joints)
    fig, ax = plt.subplots(num_joints, 1, figsize=(12, 3 * num_joints), sharex=False)
    plt.suptitle('Mean RMSE of Euler Angle Error (IMU vs. Marker) by Joint and Axis', fontsize=16)

    axis_names = ['X', 'Y', 'Z']
    method_colors = {'Madgwick': 'lightblue', 'Mag-Free': 'lightgreen', 'Never Project': 'orange'} 

    # Ensure ax is an array even for a single joint
    if num_joints == 1:
        ax = np.array([ax])

    for i, joint in enumerate(joints):
        current_ax = ax[i]
        
        x_positions = []
        bar_heights = [] # Mean RMSE
        error_whiskers = [] # Mean STD
        bar_colors = []
        
        bar_width = 0.25 
        group_width = bar_width * len(method_names) 
        
        for axis_idx, axis_name in enumerate(axis_names):
            group_start_x = axis_idx * (group_width + bar_width) 
            
            for method_idx, method_name in enumerate(method_names):
                if method_name in rmse_std_results and joint in rmse_std_results[method_name] and axis_name in rmse_std_results[method_name][joint]:
                    
                    mean_rmse, mean_std = rmse_std_results[method_name][joint][axis_name]
                    
                    bar_x = group_start_x + (method_idx * bar_width)
                    
                    x_positions.append(bar_x)
                    bar_heights.append(mean_rmse)
                    error_whiskers.append(mean_std)
                    bar_colors.append(method_colors[method_name])

        if x_positions:
            current_ax.bar(x_positions, bar_heights, width=bar_width, color=bar_colors, edgecolor='black', zorder=3)
            current_ax.errorbar(x_positions, bar_heights, yerr=error_whiskers, fmt='none', ecolor='k', capsize=4, zorder=4)
            
            max_label_y = 0.0 
            for k in range(len(x_positions)):
                x = x_positions[k]
                rmse = bar_heights[k]
                std = error_whiskers[k]
                
                label_string = f'{rmse:.2f} $\\pm$ {std:.2f}' 
                
                y_pos = rmse + error_whiskers[k] + 0.5 
                max_label_y = max(max_label_y, y_pos)
                
                current_ax.text(x, y_pos, label_string, 
                                ha='center', va='bottom', 
                                fontsize=8, 
                                rotation=45) 
            
            if max_label_y > 0:
                current_ax.set_ylim(bottom=0, top=max_label_y * 1.05) 

            axis_center_x = [ (axis_idx * (group_width + bar_width)) + (group_width / 2) - (bar_width / 2) for axis_idx in range(len(axis_names)) ]
            current_ax.set_xticks(axis_center_x)
            current_ax.set_xticklabels(axis_names, fontsize=12)

            current_ax.set_title(f'{joint} Mean Error (RMSE $\\pm$ STD)', fontsize=14)
            current_ax.set_ylabel('Error (Degrees)', fontsize=12)
            
            current_ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            if i == 0:
                custom_lines = [Line2D([0], [0], color=method_colors[m], lw=8) for m in method_names]
                current_ax.legend(custom_lines, method_names, title='Method', loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("mean_rmse_std_comparison.png")
    # print("Saved plot to mean_rmse_std_comparison.png") # Removed print for final output

def plot_single_figure_all_metrics(
    rmse_std_results: Dict[str, Dict[str, Dict[str, tuple]]], 
    method_names: List[str], 
    joints: List[str]
):
    """
    Creates a single bar plot comparing the MEAN RMSE and MEAN STD of Euler angle 
    error for different IMU methods, across all joints and axes, aggregated 
    across subjects.
    Saves the plot to 'single_figure_all_metrics_comparison.png'.
    """
    
    # --- Setup Data for Plotting ---
    axis_names = ['X', 'Y', 'Z']
    method_colors = {'Madgwick': 'lightblue', 'Mag-Free': 'lightgreen', 'Never Project': 'orange'} 
    
    # Flatten the data structure
    plot_data: List[Dict[str, Union[str, float]]] = []
    
    for method_name in method_names:
        for joint in joints:
            for axis_name in axis_names:
                if method_name in rmse_std_results and \
                   joint in rmse_std_results[method_name] and \
                   axis_name in rmse_std_results[method_name][joint]:
                    
                    mean_rmse, mean_std = rmse_std_results[method_name][joint][axis_name]
                    
                    plot_data.append({
                        'method': method_name,
                        'joint_axis': f'{joint}-{axis_name}',
                        'rmse': mean_rmse,
                        'std': mean_std
                    })

    # Get the unique joint-axis labels for the x-axis
    joint_axis_labels = sorted(list(set([d['joint_axis'] for d in plot_data])))
    num_categories = len(joint_axis_labels)
    num_methods = len(method_names)
    
    if num_categories == 0:
        print("No data available to plot.")
        return

    # --- Plotting Configuration ---
    bar_width = 0.25 
    group_width = bar_width * num_methods
    spacing = bar_width * 0.5 
    
    fig, ax = plt.subplots(figsize=(4 + num_categories * 0.5, 7)) # Dynamic figure size
    
    plt.title('Mean RMSE of Euler Angle Error (IMU vs. Marker) Across All Joints and Axes', fontsize=16)
    ax.set_ylabel('Mean Error (Degrees)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)
    
    x_positions = []
    bar_heights = [] # Mean RMSE
    error_whiskers = [] # Mean STD
    bar_colors = []
    
    # --- Generate Plot Data Points ---
    
    # Calculate x-positions for each bar
    for category_idx, label in enumerate(joint_axis_labels):
        group_start_x = category_idx * (group_width + spacing)
        
        for method_idx, method_name in enumerate(method_names):
            
            # Find the data point for this method and joint-axis combination
            data_point = next((d for d in plot_data if d['method'] == method_name and d['joint_axis'] == label), None)
            
            if data_point:
                mean_rmse = data_point['rmse']
                mean_std = data_point['std']
                
                bar_x = group_start_x + (method_idx * bar_width)
                
                x_positions.append(bar_x)
                bar_heights.append(mean_rmse)
                error_whiskers.append(mean_std)
                bar_colors.append(method_colors[method_name])

    # --- Draw Bars and Error Bars ---
    ax.bar(x_positions, bar_heights, width=bar_width, color=bar_colors, edgecolor='black', zorder=3)
    ax.errorbar(x_positions, bar_heights, yerr=error_whiskers, fmt='none', ecolor='k', capsize=4, zorder=4)

    # --- Final Touches (Labels and Ticks) ---
    
    # Calculate the center x-position for each label group
    label_center_x = [ (idx * (group_width + spacing)) + (group_width / 2) - (bar_width / 2) 
                       for idx in range(num_categories) ]

    ax.set_xticks(label_center_x)
    ax.set_xticklabels(joint_axis_labels, rotation=45, ha='right', fontsize=10)
    
    # Create legend
    custom_lines = [Line2D([0], [0], color=method_colors[m], lw=8) for m in method_names]
    ax.legend(custom_lines, method_names, title='Method', loc='upper left')

    plt.tight_layout()
    plt.savefig("single_figure_all_metrics_comparison.png")
    # print("Saved single figure plot to single_figure_all_metrics_comparison.png") # Removed print for final output

# --- NEW FUNCTION FOR STATISTICAL ANALYSIS ---

def analyze_significance(
    all_subjects_results: List[Dict[str, Dict[str, Dict[str, tuple]]]],
    method_names: List[str],
    metric_type: str = 'rmse' # 'rmse' or 'std'
) -> Dict[str, Dict[str, Union[float, str, Dict[str, float]]]]:
    """
    Performs the Friedman test and Nemenyi post-hoc test on the error metric 
    (RMSE or STD) across the three IMU methods for each joint-axis combination.

    Args:
        all_subjects_results: List of results (one dict per subject).
        method_names: List of IMU method names.
        metric_type: The metric to test ('rmse' or 'std').

    Returns:
        A dictionary: {Joint-Axis: { 'Friedman_p': p_value, 
                                     'Friedman_sig': 'Significant'/'Not Significant',
                                     'Nemenyi_p_values': { 'MethodA vs MethodB': p_value, ... } }
    """
    
    # 1. Prepare the structure for the statistical results
    stats_results = {}
    
    # Define metric index: 0 for RMSE, 1 for STD
    metric_idx = 0 if metric_type.lower() == 'rmse' else 1
    
    # Determine all unique joint-axis combinations
    joint_axes = set()
    for subject_result in all_subjects_results:
        for method in subject_result:
            for joint in subject_result[method]:
                for axis in subject_result[method][joint]:
                    joint_axes.add(f"{joint}-{axis}")
    
    # 2. Loop over each Joint-Axis combination
    for ja_combo in sorted(list(joint_axes)):
        joint, axis = ja_combo.split('-')

        # 3. Collect Data: N_subjects x N_methods matrix
        # Columns will be the methods, rows will be the subjects
        method_data = {method: [] for method in method_names}
        
        # Collect the target metric from each subject's results
        for subject_result in all_subjects_results:
            
            # Find the metric value for each method
            found_all_methods = True
            for method in method_names:
                if method in subject_result and joint in subject_result[method] and axis in subject_result[method][joint]:
                    metric_value = subject_result[method][joint][axis][metric_idx]
                    method_data[method].append(metric_value)
                else:
                    # Skip this subject for this combination if data is missing for any method
                    found_all_methods = False
                    break
            
            # If data was missing, remove the incomplete entry added in the last loop
            if not found_all_methods:
                for method in method_names:
                    if method_data[method]:
                        method_data[method].pop()

        # Check if we have enough data (at least 5 subjects is generally preferred)
        n_subjects = len(list(method_data.values())[0]) if method_data and method_data[method_names[0]] else 0
        
        if n_subjects < 5 or any(len(v) != n_subjects for v in method_data.values()):
            # print(f"Skipping {ja_combo}: Not enough complete data points (N={n_subjects})")
            continue


        # Prepare data for SciPy (list of arrays, one array per method)
        data_for_friedman = [np.array(method_data[method]) for method in method_names]

        # 4. Perform Friedman Test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning) # Ignore warning about ties
            stat, p_friedman = friedmanchisquare(*data_for_friedman)
        
        is_significant = 'Significant' if p_friedman < 0.05 else 'Not Significant'
        
        current_ja_results = {
            'Friedman_p': p_friedman,
            'Friedman_sig': is_significant,
            'N_subjects': n_subjects
        }

        # 5. Perform Nemenyi Post-Hoc Test if Friedman is significant
        nemenyi_p_values = {}
        if is_significant == 'Significant':
            # Create a Pandas DataFrame for scikit_posthocs: Subjects x Methods
            df_for_posthoc = pd.DataFrame(method_data)
            
            # Nemenyi test (uses a table, not arrays like friedmanchisquare)
            nemenyi_results = posthoc_nemenyi_friedman(df_for_posthoc.values)
            
            # Rename columns/index to method names for clarity
            nemenyi_results.columns = method_names
            nemenyi_results.index = method_names
            
            # Extract the p-values for the pairs (only need the upper triangle)
            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    m1 = method_names[i]
                    m2 = method_names[j]
                    pair_key = f'{m1} vs {m2}'
                    nemenyi_p_values[pair_key] = nemenyi_results.loc[m1, m2]

        current_ja_results['Nemenyi_p_values'] = nemenyi_p_values
        
        stats_results[ja_combo] = current_ja_results
        
    return stats_results

if __name__ == "__main__":
    # 1. Define the parameters
    # Change this list to include all your subjects!
    SUBJECT_IDS: List[str] = ["Subject03"]
    trial_type = "walking"  
    method_names_for_error = ['Madgwick', 'Mag-Free', 'Never Project']
    joints = list(JOINT_SEGMENT_DICT.keys()) # Assumes JOINT_SEGMENT_DICT is available

    all_subjects_results = []

    # 2. Loop and Collect Results
    for subject_id in SUBJECT_IDS:
        print(f"Processing data for {subject_id}...")
        subject_results = get_subject_error_metrics(subject_id, trial_type)
        if subject_results:
            all_subjects_results.append(subject_results)
        
    print("\nAggregation complete. Calculating mean errors...")

    # 3. Aggregate the Error Metrics
    if all_subjects_results:
        aggregated_rmse_std = aggregate_error_metrics(all_subjects_results)

        # 4. Generate the Combined RMSE/STD Bar Plot
        plot_mean_rmse_std_comparison(
            rmse_std_results=aggregated_rmse_std,
            method_names=method_names_for_error,
            joints=joints
        )
        print("Combined RMSE/STD plot saved to 'mean_rmse_std_comparison.png'.")

        # # 5. Generate the Single Figure Plot
        # plot_single_figure_all_metrics(
        #     rmse_std_results=aggregated_rmse_std,
        #     method_names=method_names_for_error,
        #     joints=joints
        # )
        # print("Single figure plot saved to 'single_figure_all_metrics_comparison.png'.")

        # Assuming 'aggregated_rmse_std' contains the aggregated results from previous steps
        # and 'method_names_for_error' is ['Madgwick', 'Mag-Free', 'Never Project']

        overall_results = calculate_overall_mean_error(
            aggregated_rmse_std=aggregated_rmse_std,
            method_names=method_names_for_error
        )

        plot_overall_mean_comparison(
            overall_results=overall_results,
            method_names=method_names_for_error
        )
    else:
        print("No subjects processed successfully. Cannot generate plot.")