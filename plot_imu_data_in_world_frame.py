import os
from typing import Dict, List, Any
import pandas as pd
from src.toolchest.PlateTrial import PlateTrial
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import sys
import seaborn as sns

# Set print options for floats to 3 significant figures
np.set_printoptions(formatter={'float': lambda x: "%.3g" % x})

def load_and_process_imu_data(subject_num: str, trial_type: str, max_frames: int = 60000) -> List[Dict[str, Any]]:
    """
    Loads marker plate trials for a subject, rotates IMU data to the global
    frame, and returns the mean and std dev for accelerometer and 
    magnetometer vectors for each trial.
    """
    subject_id = f"Subject{subject_num}"
    print(f"--- Loading data for {subject_id}, {trial_type} ---")
    
    results = [] # Store results here
    
    try:
        data_folder_path = os.path.join("data", "data", subject_id, trial_type)
        data_folder_path = os.path.abspath(data_folder_path)
        
        plate_trials = PlateTrial.load_trial_from_folder(
            data_folder_path
        )
        print(f"Successfully loaded {len(plate_trials)} trials.")

        if len(plate_trials[0]) > max_frames:
            print(f"Trimming trials to {max_frames} frames for manageability.")
            plate_trials = [trial[-max_frames:] for trial in plate_trials]

        imu_traces_in_global_frame = {trial.name: trial.get_imu_trace_in_global_frame() for trial in plate_trials}
        
        for trial_name, imu_trace in imu_traces_in_global_frame.items():
            acc = np.array(imu_trace.acc)
            acc_magnitude = np.linalg.norm(acc, axis=1)
            mag = np.array(imu_trace.mag)
            mag_magnitude = np.linalg.norm(mag, axis=1)
            mean_acc = np.mean(acc, axis=0)
            mean_mag = np.mean(mag, axis=0)
            std_acc = np.std(acc, axis=0)  # <-- Intra-trial std
            std_mag = np.std(mag, axis=0)  # <-- Intra-trial std
            acc_magnitude_mean = np.mean(acc_magnitude)
            mag_magnitude_mean = np.mean(mag_magnitude)
            acc_magnitude_std = np.std(acc_magnitude)
            mag_magnitude_std = np.std(mag_magnitude)
            
            # Print the data as before
            print(f"Trial: {trial_name} | Acceleration: {mean_acc} ± {std_acc} m/s² | Magnetometer: {mean_mag} ± {std_mag} µT")

            # Store the data to be returned
            results.append({
                "subject": subject_id,
                "trial_type": trial_type,
                "segment": trial_name,
                # --- Store the per-trial mean ---
                "acc_x": mean_acc[0],
                "acc_y": mean_acc[1],
                "acc_z": mean_acc[2],
                "mag_x": mean_mag[0],
                "mag_y": mean_mag[1],
                "mag_z": mean_mag[2],
                # --- Store the per-trial std dev ---
                "acc_x_std": std_acc[0],
                "acc_y_std": std_acc[1],
                "acc_z_std": std_acc[2],
                "mag_x_std": std_mag[0],
                "mag_y_std": std_mag[1],
                "mag_z_std": std_mag[2],
                # --- Store the vector magnitudes ---
                "acc_magnitude": acc_magnitude_mean,
                "acc_magnitude_std": acc_magnitude_std,
                "mag_magnitude": mag_magnitude_mean,
                "mag_magnitude_std": mag_magnitude_std,
            })
        
        return results
    
    except Exception as e:
        print(f"An error occurred loading data for {subject_id} {trial_type}: {e}", file=sys.stderr)
        return [] # Return empty list on failure

def get_pooled_summary(df: pd.DataFrame, group_by_cols: List[str], data_cols: List[str]) -> pd.DataFrame:
    """
    Pools data using the Law of Total Variance.
    
    Args:
        df: The full DataFrame.
        group_by_cols: List of columns to group by (e.g., ['segment'] or ['segment', 'trial_type']).
        data_cols: List of data prefixes to pool (e.g., ['acc_x', 'mag_y']).
    """
    
    # First, calculate variance (std^2) for each trial
    var_cols = []
    for col in data_cols:
        mean_col = col
        std_col = f"{col}_std"
        var_col = f"{col}_var"
        
        # Ensure columns exist before trying to square them
        if std_col in df.columns:
            df[var_col] = df[std_col]**2
            var_cols.append(var_col)
        else:
            print(f"Warning: Missing std column {std_col}. Cannot calculate variance.")

    # Group by the specified columns (e.g., 'segment')
    grouped = df.groupby(group_by_cols)
    
    # Calculate the two parts of the Law of Total Variance
    mean_of_means = grouped[data_cols].mean()     # E[Y|X] -> new dot
    var_of_means = grouped[data_cols].var()       # Var(E[Y|X]) -> inter-trial variance
    mean_of_vars = grouped[var_cols].mean()     # E[Var(Y|X)] -> mean intra-trial variance
    
    # The mean_of_vars DataFrame will have '_var' suffixes. Rename for easy addition.
    mean_of_vars.columns = [col.replace('_var', '') for col in mean_of_vars.columns]
    
    # Calculate Total Variance and Total Std Dev
    total_var = var_of_means + mean_of_vars
    total_std = np.sqrt(total_var)
    
    # Combine into a new summary DataFrame
    summary_df = mean_of_means.rename(columns={c: f"{c}_mean" for c in data_cols})
    summary_std_df = total_std.rename(columns={c: f"{c}_std" for c in data_cols})
    
    return summary_df.join(summary_std_df)

def plot_subject_segment_dots(df: pd.DataFrame):
    """
    Generates 6 dot-and-whisker plots (one for each axis of acc/mag).
    
    In each plot:
    - X-axis: Body Segment (in a specific order)
    - Y-axis: Measurement
    - Dot: Mean value for a specific (Subject, Segment, Trial).
    - Whisker: Intra-trial Std Dev for that (Subject, Segment, Trial).
    - Color: Determined by Segment
    - Shape (Marker): Determined by Subject
    - Fill (Marker Fill): Determined by Trial Type
    """
    
    print("Plotting data per subject, segment, and trial...")

    # 1. Define the desired segment order
    segment_order = [
        'torso_imu', 'pelvis_imu', 
        'femur_r_imu', 'femur_l_imu', 
        'tibia_r_imu', 'tibia_l_imu', 
        'calcn_r_imu', 'calcn_l_imu'
    ]
    
    # Filter dataframe to only include segments in our desired list
    df = df[df['segment'].isin(segment_order)].copy()
    
    # Get unique subjects and trials
    subjects = sorted(df['subject'].unique())
    trial_types = sorted(df['trial_type'].unique())
    
    n_segments = len(segment_order)
    n_subjects = len(subjects)
    n_trials = len(trial_types)
    
    print(f"Generating plots for {n_segments} segments, {n_subjects} subjects, and {n_trials} trial types...")

    # 3. Create mappings
    
    # Color map for Segments
    if n_segments <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_segments))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_segments))
    segment_colors = {segment: colors[i] for i, segment in enumerate(segment_order)}
    
    # Marker map for Subjects
    markers_list = ['o', 's', '^', 'v', 'P', 'X', '*', 'D', 'p', '<', '>', 'h']
    subject_markers = {subject: markers_list[i % len(markers_list)] for i, subject in enumerate(subjects)}
    
    # Fill map for Trial Types
    trial_fills = {trial: 'full' for trial in trial_types}
    if n_trials > 1:
        trial_fills[trial_types[0]] = 'none' # e.g., 'complexTasks' = hollow
        trial_fills[trial_types[1]] = 'full' # e.g., 'walking' = solid

    # 4. Create the 6 subplots (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(18, 24), sharex=True)
    fig.suptitle('IMU Measurements by Segment and Subject\n(Dot = Trial Mean, Whisker = Intra-Trial Std Dev)', fontsize=16)
    
    ax_list = axes.flatten() # [ax0, ax1, ax2, ax3, ax4, ax5]
    
    # Define what to plot in each subplot
    # --- CHANGED: Using trial mean/std columns directly ---
    plot_specs = [
        {'col_mean': 'acc_x', 'col_std': 'acc_x_std', 'title': 'Acceleration (X-axis)', 'ylabel': 'm/s²'},
        {'col_mean': 'mag_x', 'col_std': 'mag_x_std', 'title': 'Magnetometer (X-axis)', 'ylabel': 'µT'},
        {'col_mean': 'acc_y', 'col_std': 'acc_y_std', 'title': 'Acceleration (Y-axis)', 'ylabel': 'm/s²'},
        {'col_mean': 'mag_y', 'col_std': 'mag_y_std', 'title': 'Magnetometer (Y-axis)', 'ylabel': 'µT'},
        {'col_mean': 'acc_z', 'col_std': 'acc_z_std', 'title': 'Acceleration (Z-axis)', 'ylabel': 'm/s²'},
        {'col_mean': 'mag_z', 'col_std': 'mag_z_std', 'title': 'Magnetometer (Z-axis)', 'ylabel': 'µT'},
    ]

    # 5. Main Plotting Loop
    
    # Calculate horizontal dodge positions
    # We have (n_subjects * n_trials) dots per segment
    n_total_dots = n_subjects * n_trials
    dodge = np.linspace(-0.35, 0.35, n_total_dots)
    
    for ax, spec in zip(ax_list, plot_specs):
        for i_seg, segment in enumerate(segment_order):
            dot_index = 0
            for subject in subjects:
                for trial_type in trial_types:
                    
                    x_pos = i_seg + dodge[dot_index]
                    dot_index += 1 # Increment for next dot
                    
                    # Find the single row for this data point
                    data_row = df[
                        (df['subject'] == subject) & 
                        (df['segment'] == segment) & 
                        (df['trial_type'] == trial_type)
                    ]
                    
                    # If this subject/segment/trial doesn't exist, skip
                    if data_row.empty:
                        continue
                        
                    data = data_row.iloc[0] # Get the series
                    mean_val = data[spec['col_mean']]
                    std_val = data[spec['col_std']]
                    
                    # Get the specific x position, color, and marker
                    color = segment_colors[segment]
                    marker = subject_markers[subject]
                    fillstyle = trial_fills[trial_type]
                    
                    ax.errorbar(
                        x=x_pos,
                        y=mean_val,
                        yerr=std_val,
                        marker=marker,
                        color=color,
                        fillstyle=fillstyle,
                        linestyle='none', # No connecting line
                        capsize=3,        # Width of the whisker cap
                        markersize=8,
                        alpha=0.7,
                        markeredgecolor=color, # Ensure hollow markers have colored edge
                    )
        
        # --- Configure each subplot ---
        ax.set_title(spec['title'])
        ax.set_ylabel(spec['ylabel'])
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5) # Zero line
        
        # Set shared x-ticks (only for the bottom plots)
        if ax in axes[2, :]: # If in the last row
            ax.set_xticks(range(n_segments))
            ax.set_xticklabels(segment_order, rotation=45, ha='right')
        
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xlim(-0.5, n_segments - 0.5) # Add padding to x-axis

        # Adjust Y-axis for Acc-Y plot
        if spec['col_mean'] == 'acc_y':
            # Find min/max of data *for this axis*
            data_means = df[spec['col_mean']]
            data_stds = df[spec['col_std']]
            
            # Filter out NaNs just in case
            valid_data = ~data_means.isna() & ~data_stds.isna()
            if valid_data.any():
                plot_min = (data_means[valid_data] - data_stds[valid_data]).min()
                plot_max = (data_means[valid_data] + data_stds[valid_data]).max()
                
                # Add 10% padding
                padding = (plot_max - plot_min) * 0.1
                if padding == 0: # Handle case with no variance
                    padding = 0.1 
                
                # Set the new limits
                ax.set_ylim(plot_min - padding, plot_max + padding)

    # 6. Create Custom Legends (outside the plots)
    
    # Color (Segment) legend
    segment_patches = [mpatches.Patch(color=color, label=segment) 
                       for segment, color in segment_colors.items()]
    fig.legend(handles=segment_patches, title='Body Segment (Color)', 
               bbox_to_anchor=(0.15, 0.75), loc='upper right', frameon=True)
    
    # Marker (Subject) legend
    subject_lines = [mlines.Line2D([], [], color='gray', marker=marker, linestyle='None',
                                  markersize=10, label=subject) 
                     for subject, marker in subject_markers.items()]
    fig.legend(handles=subject_lines, title='Subject (Shape)', 
               bbox_to_anchor=(0.15, 0.45), loc='upper right', frameon=True)
    
    # --- NEW: Trial Type (Fill) Legend ---
    trial_lines = [mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=10, label=trial, fillstyle=fill,
                                markeredgecolor='gray')
                   for trial, fill in trial_fills.items()]
    fig.legend(handles=trial_lines, title='Trial Type (Fill)',
               bbox_to_anchor=(0.15, 0.15), loc='upper right', frameon=True)
    
    # Adjust layout to make room on the *left*
    plt.tight_layout(rect=[0.15, 0, 1.0, 0.95])

def plot_pooled_data(df: pd.DataFrame):
    """
    Generates 6 dot-and-whisker plots (one for each axis of acc/mag)
    from a POOLED DataFrame.
    
    In each plot:
    - X-axis: Body Segment (in a specific order)
    - Y-axis: Measurement
    - Dot: Pooled Mean value
    - Whisker: Pooled Total Std Dev.
    - Color: Determined by Segment
    - Fill (Marker Fill): Determined by Trial Type (if 'trial_type' is a column)
    """
    
    print("Plotting pooled data...")

    # 1. Setup and Detect Groupings
    
    # CRITICAL: Reset index to turn 'segment' and 'trial_type' into columns
    df_plot = df.reset_index()

    segment_order = [
        'torso_imu', 'pelvis_imu', 
        'femur_r_imu', 'femur_l_imu', 
        'tibia_r_imu', 'tibia_l_imu', 
        'calcn_r_imu', 'calcn_l_imu'
    ]
    
    # Filter dataframe to only include segments in our desired list
    df_plot = df_plot[df_plot['segment'].isin(segment_order)].copy()
    
    # Automatically detect if we are plotting by trial_type
    if 'trial_type' in df_plot.columns:
        trial_types = sorted(df_plot['trial_type'].unique())
    else:
        trial_types = []
    
    n_segments = len(segment_order)
    n_trials = len(trial_types)
    
    print(f"Generating plots for {n_segments} segments.")
    if n_trials > 0:
        print(f"Plot will be dodged by {n_trials} trial types.")

    # 2. Create Mappings
    
    # Color map for Segments
    if n_segments <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_segments))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_segments))
    segment_colors = {segment: colors[i] for i, segment in enumerate(segment_order)}
    
    # Fill map for Trial Types
    trial_fills = {trial: 'full' for trial in trial_types}
    if n_trials > 1:
        trial_fills[trial_types[0]] = 'none' # e.g., 'complexTasks' = hollow
        trial_fills[trial_types[1]] = 'full' # e.g., 'walking' = solid

    # 3. Create the 6 subplots (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(6, 12), sharex=True)
    fig.suptitle('Mean and Standard Deviation of Sensor Measurements', fontsize=16)
    
    ax_list = axes.flatten()
    
    # Define what to plot in each subplot
    # --- CHANGED: Using the _mean and _std suffixes from get_pooled_summary ---
    plot_specs = [
        {'col_mean': 'acc_x_mean', 'col_std': 'acc_x_std', 'title': 'Acceleration (X-axis)', 'ylabel': 'm/s²'},
        {'col_mean': 'mag_x_mean', 'col_std': 'mag_x_std', 'title': 'Magnetometer (X-axis)', 'ylabel': 'µT'},
        {'col_mean': 'acc_y_mean', 'col_std': 'acc_y_std', 'title': 'Acceleration (Y-axis)', 'ylabel': 'm/s²'},
        {'col_mean': 'mag_y_mean', 'col_std': 'mag_y_std', 'title': 'Magnetometer (Y-axis)', 'ylabel': 'µT'},
        {'col_mean': 'acc_z_mean', 'col_std': 'acc_z_std', 'title': 'Acceleration (Z-axis)', 'ylabel': 'm/s²'},
        {'col_mean': 'mag_z_mean', 'col_std': 'mag_z_std', 'title': 'Magnetometer (Z-axis)', 'ylabel': 'µT'},
    ]

    # 4. Main Plotting Loop
    
    # Calculate horizontal dodge positions
    if n_trials > 1:
        dodge = np.linspace(-0.2, 0.2, n_trials)
    else:
        dodge = [0]
    
    for ax, spec in zip(ax_list, plot_specs):
        for i_seg, segment in enumerate(segment_order):
            
            # Get all data for this segment
            segment_data = df_plot[df_plot['segment'] == segment]
            
            if segment_data.empty:
                continue

            # Case 1: No trial types, plot one dot
            if n_trials <= 1:
                x_pos = i_seg
                data = segment_data.iloc[0] # Should only be one row
                fillstyle = 'full'
                
                ax.errorbar(
                    x=x_pos,
                    y=data[spec['col_mean']],
                    yerr=data[spec['col_std']],
                    marker='o', # Use a consistent marker
                    color=segment_colors[segment],
                    fillstyle=fillstyle,
                    linestyle='none',
                    capsize=5, # Wider capsize
                    markersize=10,
                    alpha=0.8,
                    markeredgecolor=segment_colors[segment],
                )
            
            # Case 2: Plot dodged dots for each trial type
            else:
                for i_trial, trial_type in enumerate(trial_types):
                    x_pos = i_seg + dodge[i_trial]
                    
                    # Find the single row for this segment/trial
                    data_row = segment_data[segment_data['trial_type'] == trial_type]
                    
                    if data_row.empty:
                        continue
                        
                    data = data_row.iloc[0] # Get the series
                    mean_val = data[spec['col_mean']]
                    std_val = data[spec['col_std']]
                    
                    color = segment_colors[segment]
                    fillstyle = trial_fills[trial_type]
                    
                    ax.errorbar(
                        x=x_pos,
                        y=mean_val,
                        yerr=std_val,
                        marker='o', # Use a consistent marker
                        color=color,
                        fillstyle=fillstyle,
                        linestyle='none',
                        capsize=5,
                        markersize=10,
                        alpha=0.8,
                        markeredgecolor=color,
                    )
        
        # 5. Configure Axes
        ax.set_title(spec['title'])
        ax.set_ylabel(spec['ylabel'])
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        
        if ax in axes[2, :]: # If in the last row
            ax.set_xticks(range(n_segments))
            ax.set_xticklabels(segment_order, rotation=45, ha='right')
        
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xlim(-0.5, n_segments - 0.5)

        # Adjust Y-axis for Acc-Y plot
        if spec['col_mean'] == 'acc_y_mean':
            data_means = df_plot[spec['col_mean']]
            data_stds = df_plot[spec['col_std']]
            
            valid_data = ~data_means.isna() & ~data_stds.isna()
            if valid_data.any():
                plot_min = (data_means[valid_data] - data_stds[valid_data]).min()
                plot_max = (data_means[valid_data] + data_stds[valid_data]).max()
                
                padding = (plot_max - plot_min) * 0.1
                if padding == 0: padding = 0.1 
                
                ax.set_ylim(plot_min - padding, plot_max + padding)

    # 6. Create Custom Legends
    
    # # Color (Segment) legend
    # segment_patches = [mpatches.Patch(color=color, label=segment) 
    #                    for segment, color in segment_colors.items()]
    # fig.legend(handles=segment_patches, title='Body Segment (Color)', 
    #            bbox_to_anchor=(0.15, 0.75), loc='upper right', frameon=True)
    
    # Trial Type (Fill) Legend (Only if we have trials)
    # if n_trials > 1:
    #     trial_lines = [mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
    #                                 markersize=10, label=trial, fillstyle=fill,
    #                                 markeredgecolor='gray')
    #                    for trial, fill in trial_fills.items()]
    #     fig.legend(handles=trial_lines, title='Trial Type (Fill)',
    #                bbox_to_anchor=(0.15, 0.45), loc='upper right', frameon=True)
    
    plt.tight_layout(rect=[0.15, 0, 1.0, 0.95])

def plot_pooled_magnitudes(df: pd.DataFrame):
    """
    Generates 2 dot-and-whisker plots (one for each magnitude of acc/mag)
    from a POOLED DataFrame. All dots are the same color.
    
    In each plot:
    - X-axis: Body Segment (in a specific order)
    - Y-axis: Measurement
    - Dot: Pooled Mean value
    - Whisker: Pooled Total Std Dev.
    - Color: Constant (e.g., 'C0')
    - Fill (Marker Fill): Determined by Trial Type (if 'trial_type' is a column)
    """
    
    print("Plotting pooled magnitude data...")

    # 1. Setup and Detect Groupings
    
    # CRITICAL: Reset index to turn 'segment' and 'trial_type' into columns
    df_plot = df.reset_index()

    segment_order = [
        'torso_imu', 'pelvis_imu', 
        'femur_r_imu', 'femur_l_imu', 
        'tibia_r_imu', 'tibia_l_imu', 
        'calcn_r_imu', 'calcn_l_imu'
    ]
    
    # Filter dataframe to only include segments in our desired list
    df_plot = df_plot[df_plot['segment'].isin(segment_order)].copy()
    
    # Automatically detect if we are plotting by trial_type
    if 'trial_type' in df_plot.columns:
        trial_types = sorted(df_plot['trial_type'].unique())
    else:
        trial_types = []
    
    n_segments = len(segment_order)
    n_trials = len(trial_types)
    
    print(f"Generating plots for {n_segments} segments.")
    if n_trials > 0:
        print(f"Plot will be dodged by {n_trials} trial types.")

    # 2. Create Mappings
    
    # Fill map for Trial Types
    trial_fills = {trial: 'full' for trial in trial_types}
    if n_trials > 1:
        trial_fills[trial_types[0]] = 'none' # e.g., 'complexTasks' = hollow
        trial_fills[trial_types[1]] = 'full' # e.g., 'walking' = solid

    # 3. Create the 2 subplots (2 rows, 1 column)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
    fig.suptitle('Mean and Standard Deviation of Sensor Magnitudes', fontsize=16)
    
    ax_list = axes.flatten()
    
    # Define what to plot in each subplot
    plot_specs = [
        {'col_mean': 'acc_magnitude_mean', 'col_std': 'acc_magnitude_std', 'title': 'Acceleration Magnitude', 'ylabel': 'm/s²'},
        {'col_mean': 'mag_magnitude_mean', 'col_std': 'mag_magnitude_std', 'title': 'Magnetometer Magnitude', 'ylabel': 'µT'},
    ]

    # 4. Main Plotting Loop
    
    # Calculate horizontal dodge positions
    if n_trials > 1:
        dodge = np.linspace(-0.2, 0.2, n_trials)
    else:
        dodge = [0]

    colors_list = sns.color_palette(n_colors=6)
    colors = colors_list[4:]
    
    for ax, spec in zip(ax_list, plot_specs):
        color = colors[0]  # Constant color for all dots
        for i_seg, segment in enumerate(segment_order):
            
            # Get all data for this segment
            segment_data = df_plot[df_plot['segment'] == segment]
            
            if segment_data.empty:
                continue

            # Case 1: No trial types, plot one dot
            if n_trials <= 1:
                x_pos = i_seg
                data = segment_data.iloc[0] # Should only be one row
                fillstyle = 'full'
                
                ax.errorbar(
                    x=x_pos,
                    y=data[spec['col_mean']],
                    yerr=data[spec['col_std']],
                    marker='o',
                    color=colors[0],
                    fillstyle=fillstyle,
                    linestyle='none',
                    capsize=5,
                    markersize=10,
                    alpha=0.8,
                )
            
            # Case 2: Plot dodged dots for each trial type
            else:
                for i_trial, trial_type in enumerate(trial_types):
                    x_pos = i_seg + dodge[i_trial]
                    
                    # Find the single row for this segment/trial
                    data_row = segment_data[segment_data['trial_type'] == trial_type]
                    
                    if data_row.empty:
                        continue
                        
                    data = data_row.iloc[0] # Get the series
                    mean_val = data[spec['col_mean']]
                    std_val = data[spec['col_std']]
                    
                    color = 'C0' # Use constant color
                    fillstyle = trial_fills[trial_type]
                    
                    ax.errorbar(
                        x=x_pos,
                        y=mean_val,
                        yerr=std_val,
                        marker='o',
                        color=colors[1],
                        fillstyle=fillstyle,
                        linestyle='none',
                        capsize=5,
                        markersize=10,
                        alpha=0.8,
                        markeredgecolor=color,
                    )
        color = colors[1]  # Switch color for next dot (if any)
        
        # 5. Configure Axes
        ax.set_title(spec['title'])
        ax.set_ylabel(spec['ylabel'])
        
        # Set x-ticks only for the bottom plot
        if ax == axes[1]: # If in the last row
            ax.set_xticks(range(n_segments))
            ax.set_xticklabels(segment_order, rotation=45, ha='right')
        
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_xlim(-0.5, n_segments - 0.5)

        # Auto-adjust Y-axis for all plots
        data_means = df_plot[spec['col_mean']]
        data_stds = df_plot[spec['col_std']]
        
        valid_data = ~data_means.isna() & ~data_stds.isna()
        if valid_data.any():
            plot_min = (data_means[valid_data] - data_stds[valid_data]).min()
            plot_max = (data_means[valid_data] + data_stds[valid_data]).max()
            
            padding = (plot_max - plot_min) * 0.1
            if padding == 0: padding = 0.1 
            
            # Magnitudes can't be negative, so set min to 0 or 
            ax.set_ylim(max(0, plot_min - padding), plot_max + padding)

    # 6. Create Custom Legends
    
    # Trial Type (Fill) Legend (Only if we have trials)
    if n_trials > 1:
        trial_lines = [mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                    markersize=10, label=trial, fillstyle=fill,
                                    markeredgecolor='gray')
                       for trial, fill in trial_fills.items()]
        fig.legend(handles=trial_lines, title='Trial Type (Fill)',
                   bbox_to_anchor=(0.15, 0.45), loc='upper right', frameon=True)
    
    plt.tight_layout(rect=[0.15, 0, 1.0, 0.95])

def plot_pooled_magnitudes_twin_ax(df: pd.DataFrame):
    """
    Generates 1 dot-and-whisker plot with two y-axes for
    acc and mag magnitudes from a POOLED DataFrame.
    
    - ax1 (Left): Acceleration (5th sns color)
    - ax2 (Right): Magnetometer (6th sns color)
    - Dots are smaller.
    - Top spines are hidden.
    """
    
    print("Plotting pooled magnitude data on twin axes...")

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    # 1. Setup and Detect Groupings
    df_plot = df.reset_index()

    segment_order = [
        'torso_imu', 'pelvis_imu', 
        'femur_r_imu', 'femur_l_imu', 
        'tibia_r_imu', 'tibia_l_imu', 
        'calcn_r_imu', 'calcn_l_imu'
    ]
    
    df_plot = df_plot[df_plot['segment'].isin(segment_order)].copy()
    
    if 'trial_type' in df_plot.columns:
        trial_types = sorted(df_plot['trial_type'].unique())
    else:
        trial_types = []
    
    n_segments = len(segment_order)
    n_trials = len(trial_types)

    # 2. Create Mappings
    
    # Get 5th and 6th colors
    colors_list = sns.color_palette(n_colors=10)
    acc_color = colors_list[4]  # 5th color
    mag_color = colors_list[8]  # 6th color
    
    # Fill map for Trial Types
    trial_fills = {trial: 'full' for trial in trial_types}
    if n_trials > 1:
        trial_fills[trial_types[0]] = 'none' 
        trial_fills[trial_types[1]] = 'full' 

    # 3. Create the 1 subplot and twin axis
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 3.5))
    # fig.suptitle('Mean and Standard Deviation of Sensor Magnitudes', fontsize=16)
    
    # Create the second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    
    # Define specs for clarity
    acc_spec = {'col_mean': 'acc_magnitude_mean', 'col_std': 'acc_magnitude_std', 'ylabel': 'Acceleration\nm/s²'}
    mag_spec = {'col_mean': 'mag_magnitude_mean', 'col_std': 'mag_magnitude_std', 'ylabel': 'µT\nMagnetometer'}

    # 4. Main Plotting Loop
    
    if n_trials > 1:
        dodge = np.linspace(-0.2, 0.2, n_trials)
    else:
        dodge = [0]
    
    # Loop by segment, then plot on both axes
    for i_seg, segment in enumerate(segment_order):
        segment_data = df_plot[df_plot['segment'] == segment]
        if segment_data.empty:
            continue

        # Case 1: No trial types, plot one dot
        if n_trials <= 1:
            x_pos = i_seg - 0.15
            data = segment_data.iloc[0]
            fillstyle = 'full'
            
            # --- Plot on ax1 (ACC) ---
            ax1.errorbar(
                x=x_pos, y=data[acc_spec['col_mean']], yerr=data[acc_spec['col_std']],
                marker='o', color=acc_color, fillstyle=fillstyle, linestyle='none',
                capsize=5, markersize=8, alpha=0.8, markeredgecolor=acc_color
            )
            
            x_pos += 0.3
            # --- Plot on ax2 (MAG) ---
            ax2.errorbar(
                x=x_pos, y=data[mag_spec['col_mean']], yerr=data[mag_spec['col_std']],
                marker='o', color=mag_color, fillstyle=fillstyle, linestyle='none',
                capsize=5, markersize=8, alpha=0.8, markeredgecolor=mag_color
            )
        
        # Case 2: Plot dodged dots for each trial type
        else:
            for i_trial, trial_type in enumerate(trial_types):
                x_pos = i_seg + dodge[i_trial]
                data_row = segment_data[segment_data['trial_type'] == trial_type]
                
                if data_row.empty:
                    continue
                    
                data = data_row.iloc[0]
                fillstyle = trial_fills[trial_type]
                
                # --- Plot on ax1 (ACC) ---
                ax1.errorbar(
                    x=x_pos, y=data[acc_spec['col_mean']], yerr=data[acc_spec['col_std']],
                    marker='o', color=acc_color, fillstyle=fillstyle, linestyle='none',
                    capsize=5, markersize=8, alpha=0.8, markeredgecolor=acc_color
                )
                
                # --- Plot on ax2 (MAG) ---
                ax2.errorbar(
                    x=x_pos, y=data[mag_spec['col_mean']], yerr=data[mag_spec['col_std']],
                    marker='o', color=mag_color, fillstyle=fillstyle, linestyle='none',
                    capsize=5, markersize=8, alpha=0.8, markeredgecolor=mag_color
                )

    # 5. Configure Axes
    segment_order = ["Torso", "Pelvis", "R Femur", "L Femur", 
                     "R Tibia", "L Tibia", "R Foot", "L Foot"]
    # X-Axis (configure on ax1, it's shared)
    ax1.set_xlim(-0.5, n_segments - 0.5)
    ax1.set_xticks(range(n_segments))
    ax1.set_xticklabels(segment_order, rotation=30, ha='left', fontsize=16)
    # ax1.set_xlabel('Body Segment')

    # Y-Axes (configure separately)
    ax1.set_ylabel(acc_spec['ylabel'], fontweight='bold', fontsize=16)
    ax2.set_ylabel(mag_spec['ylabel'], fontweight='bold', fontsize=16)

    # Set y-tick colors
    ax1.tick_params(axis='y', labelcolor=acc_color, labelsize=10)
    ax2.tick_params(axis='y', labelcolor=mag_color, labelsize=10)
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()
    # Spines
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False) # Hide original right spine
    ax2.spines['left'].set_visible(False)  # Hide original left spine
    ax1.spines['left'].set_visible(False)  # Hide left spine of ax1 for cleaner look
    ax2.spines['right'].set_visible(False) # Hide right spine of ax2 for cleaner look
    ax1.spines['bottom'].set_visible(False) # Hide bottom spine for cleaner look
    ax2.spines['bottom'].set_visible(False) # Hide bottom spine for cleaner look

    # Grid (only for ax1 to avoid clutter)
    # ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Auto-adjust Y-axis limits for BOTH axes
    for ax, spec in [(ax1, acc_spec), (ax2, mag_spec)]:
        data_means = df_plot[spec['col_mean']]
        data_stds = df_plot[spec['col_std']]
        
        valid_data = ~data_means.isna() & ~data_stds.isna()
        if valid_data.any():
            plot_min = (data_means[valid_data] - data_stds[valid_data]).min()
            plot_max = (data_means[valid_data] + data_stds[valid_data]).max()
            
            padding = (plot_max - plot_min) * 0.1
            if padding == 0: padding = 0.1 
            
            ax.set_ylim(max(0, plot_min - padding), plot_max + padding)

    # 6. Create Custom Legends
    if n_trials > 1:
        trial_lines = [mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                    markersize=8, label=trial, fillstyle=fill,
                                    markeredgecolor='gray')
                       for trial, fill in trial_fills.items()]
        # Place legend outside the plot
        fig.legend(handles=trial_lines, title='Trial Type (Fill)',
                   bbox_to_anchor=(0.98, 0.4), loc='center left', frameon=False)
    
    # Adjust layout
    plt.tight_layout(rect=[0.05, 0.1, 0.85, 0.9])
    plt.savefig('pooled_magnitudes_twin_ax.png', dpi=300)


if __name__ == "__main__":
    pkl_file = "imu_data_summary_.pkl"
    
    # Check for cached data
    if os.path.exists(pkl_file):
        print(f"Loading existing IMU data summary from '{pkl_file}'...")
        df = pd.read_pickle(pkl_file)
    else:
        print(f"No '{pkl_file}' found. Loading and processing all raw data...")
        all_results = [] # List to aggregate all data
        
        for subject_num in range(1, 12):
            SUBJECT_TO_LOAD = f"{subject_num:02d}"
            for TRIAL_TYPE_TO_LOAD in ['walking', 'complexTasks']:
                # Call the processing function and get the data
                processed_data = load_and_process_imu_data(SUBJECT_TO_LOAD, TRIAL_TYPE_TO_LOAD)
                all_results.extend(processed_data) # Add data to the master list

        # --- After all data is loaded, create DataFrame and Plot ---
        if not all_results:
            print("No data was successfully loaded. Exiting.")
            sys.exit()
            
        print(f"\n--- All data loaded. Caching to '{pkl_file}'... ---")
        
        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(all_results)
        df.to_pickle(pkl_file)  # Save the *new* data with std dev

    # Create the original visualization (optional, you can comment this out)
    # plot_subject_segment_dots(df)
    # plt.show()
    
    # # --- Generate Grouped Summary DataFrames ---
    # print("\nGenerating pooled summary for (X, Y, Z) components...")
    # pooled_by_segment = get_pooled_summary(
    #     df,
    #     group_by_cols=['segment'],
    #     data_cols=['acc_x', 'acc_y', 'acc_z', 'mag_x', 'mag_y', 'mag_z']
    # )

    # plot_pooled_data(pooled_by_segment)
    # plt.show()

    # --- NEW: Generate and plot pooled summary for magnitudes ---
    print("\nGenerating pooled summary for magnitudes...")
    pooled_magnitudes_by_segment = get_pooled_summary(
        df,
        group_by_cols=['segment'],
        # IMPORTANT: Include the magnitude columns here
        data_cols=['acc_magnitude', 'mag_magnitude'] 
    )

    # Call the new plotting function
    # plot_pooled_magnitudes(pooled_magnitudes_by_segment)
    plot_pooled_magnitudes_twin_ax(pooled_magnitudes_by_segment)
    plt.show()