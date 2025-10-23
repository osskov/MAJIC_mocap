import os
from typing import Dict, List, Any
import pandas as pd
from src.toolchest.PlateTrial import PlateTrial
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import sys

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
            mag = np.array(imu_trace.mag)
            mean_acc = np.mean(acc, axis=0)
            mean_mag = np.mean(mag, axis=0)
            std_acc = np.std(acc, axis=0)  # <-- Intra-trial std
            std_mag = np.std(mag, axis=0)  # <-- Intra-trial std
            
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
    fig, axes = plt.subplots(3, 2, figsize=(18, 24), sharex=True)
    fig.suptitle('Pooled IMU Measurements by Segment\n(Dot = Pooled Mean, Whisker = Pooled Total Std Dev)', fontsize=16)
    
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
    
    # Color (Segment) legend
    segment_patches = [mpatches.Patch(color=color, label=segment) 
                       for segment, color in segment_colors.items()]
    fig.legend(handles=segment_patches, title='Body Segment (Color)', 
               bbox_to_anchor=(0.15, 0.75), loc='upper right', frameon=True)
    
    # Trial Type (Fill) Legend (Only if we have trials)
    if n_trials > 1:
        trial_lines = [mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                    markersize=10, label=trial, fillstyle=fill,
                                    markeredgecolor='gray')
                       for trial, fill in trial_fills.items()]
        fig.legend(handles=trial_lines, title='Trial Type (Fill)',
                   bbox_to_anchor=(0.15, 0.45), loc='upper right', frameon=True)
    
    plt.tight_layout(rect=[0.15, 0, 1.0, 0.95])


if __name__ == "__main__":
    pkl_file = "imu_data_summary.pkl"
    
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

    # Create the new visualization
    # plot_subject_segment_dots(df)
    # plt.show()

    
    # --- Generate Grouped Summary DataFrames ---
    print("\nGenerating pooled summary DataFrames...")
    pooled_by_segment = get_pooled_summary(
        df,
        group_by_cols=['segment', 'trial_type'],
        data_cols=['acc_x', 'acc_y', 'acc_z', 'mag_x', 'mag_y', 'mag_z']
    )

    plot_pooled_data(pooled_by_segment)
    plt.show()