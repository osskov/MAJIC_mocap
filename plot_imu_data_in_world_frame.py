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

    # --- Add Mean Torso Magnetic Field ---
    torso_mag_data = df[df['segment'] == 'torso_imu'][['mag_x', 'mag_y', 'mag_z']]
    if not torso_mag_data.empty:
        # Calculate mean across all subjects and trials for torso
        mean_torso_mag_vec = torso_mag_data.mean().values
        
        print(f"\nOverall Mean Torso Magnetic Field: {mean_torso_mag_vec[0]:.3g}, {mean_torso_mag_vec[1]:.3g}, {mean_torso_mag_vec[2]:.3g} µT")
    else:
        print("\nNo 'torso_imu' data found to plot mean magnetic field.")
    
    # Create the new visualization
    plot_subject_segment_dots(df)
    
    # Finally, display the plot
    plt.show()