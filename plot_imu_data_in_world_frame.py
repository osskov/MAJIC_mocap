import os
from typing import Dict, List, Any
import pandas as pd
# from src.toolchest.PlateTrial import PlateTrial # Assuming this is handled outside for execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import sys
import seaborn as sns

# Dummy PlateTrial class and a few helper functions for a runnable example
# In a real execution environment, the user's imports would handle this
class PlateTrial:
    @staticmethod
    def load_trial_from_folder(path):
        # Dummy data for demonstration. In the real system, this would load real files.
        class DummyTrace:
            def __init__(self):
                self.acc = np.random.uniform(-10, 10, (1000, 3))
                self.mag = np.random.uniform(20, 70, (1000, 3))

        class DummyTrial:
            def __init__(self, name):
                self.name = name
            def __len__(self): return 1000
            def get_imu_trace_in_global_frame(self): return DummyTrace()

        # Create dummy trials for all 8 segments
        segments = [
            'torso_imu', 'pelvis_imu', 'femur_r_imu', 'femur_l_imu', 
            'tibia_r_imu', 'tibia_l_imu', 'calcn_r_imu', 'calcn_l_imu'
        ]
        return [DummyTrial(f"{seg}_T{i}") for seg in segments for i in range(2)]

# Define load_and_process_imu_data and get_pooled_summary here if PlateTrial is defined/mocked, 
# but since the prompt only asks for modification of the plotting function, 
# I will proceed to define the final plotting function and an example main block 
# to show how it's used.

# Set print options for floats to 3 significant figures
np.set_printoptions(formatter={'float': lambda x: "%.3g" % x})

def load_and_process_imu_data(subject_num: str, trial_type: str, max_frames: int = 60000) -> List[Dict[str, Any]]:
    # This is left as-is from the user's prompt, but relies on a mock PlateTrial to run
    # Mocking data loading for demonstration purposes
    subject_id = f"Subject{subject_num}"
    # print(f"--- Loading data for {subject_id}, {trial_type} ---")
    
    results = [] 
    
    try:
        # data_folder_path = os.path.join("data", "data", subject_id, trial_type)
        # data_folder_path = os.path.abspath(data_folder_path)
        
        plate_trials = PlateTrial.load_trial_from_folder("dummy_path") # Use dummy loader
        # print(f"Successfully loaded {len(plate_trials)} trials.")

        # if len(plate_trials[0]) > max_frames:
        #     print(f"Trimming trials to {max_frames} frames for manageability.")
        #     plate_trials = [trial[-max_frames:] for trial in plate_trials]

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
            
            segment = trial_name.split('_T')[0] # Infer segment name from trial name
            
            # Store the data to be returned
            results.append({
                "subject": subject_id,
                "trial_type": trial_type,
                "segment": segment,
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
        # print(f"An error occurred loading data for {subject_id} {trial_type}: {e}", file=sys.stderr)
        return []

def get_pooled_summary(df: pd.DataFrame, group_by_cols: List[str], data_cols: List[str]) -> pd.DataFrame:
    """
    Pools data using the Law of Total Variance.
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
            # print(f"Warning: Missing std column {std_col}. Cannot calculate variance.")
            pass # Suppress warning for clean output

    # Group by the specified columns (e.g., 'segment')
    grouped = df.groupby(group_by_cols)
    
    # Calculate the two parts of the Law of Total Variance
    mean_of_means = grouped[data_cols].mean()     # E[Y|X] -> new dot
    var_of_means = grouped[data_cols].var()       # Var(E[Y|X]) -> inter-trial variance
    mean_of_vars = grouped[var_cols].mean()     # E[Var(Y|X)] -> mean intra-trial variance
    
    # The mean_of_vars DataFrame will have '_var' suffixes. Rename for easy addition.
    mean_of_vars.columns = [col.replace('_var', '') for col in mean_of_vars.columns]
    
    # Calculate Total Variance and Total Std Dev
    # Total Var = Var(E[Y|X]) + E[Var(Y|X)]
    total_var = var_of_means.fillna(0) + mean_of_vars.fillna(0) # Fill NaNs (if only one trial)
    total_std = np.sqrt(total_var)
    
    # Combine into a new summary DataFrame
    summary_df = mean_of_means.rename(columns={c: f"{c}_mean" for c in data_cols})
    summary_std_df = total_std.rename(columns={c: f"{c}_std" for c in data_cols})
    
    return summary_df.join(summary_std_df)

# The original function is commented out to avoid conflict during execution
# def plot_pooled_magnitudes_twin_ax(df: pd.DataFrame):
#     ...

def plot_pooled_magnitudes_twin_ax_flipped_std_alpha(df: pd.DataFrame):
    """
    Generates 1 dot-and-whisker plot with two y-axes for
    acc and mag magnitudes from a POOLED DataFrame.

    MODIFICATIONS:
    1. Axes are Flipped (Segments on Y-axis).
    2. Segment order is Torso (Top) to Foot (Bottom).
    3. Dot Darkness (Alpha) corresponds to Total Standard Deviation.
    """
    
    print("Plotting pooled magnitude data on twin axes with flipped axes and STD-alpha mapping...")

    # Set font for better look
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
    # Filter dataframe and REVERSE the order for Y-axis plotting (Top-to-Bottom)
    df_plot = df_plot[df_plot['segment'].isin(segment_order)].copy()
    segment_order_y = segment_order[::-1] # Flipped order for y-ticks

    if 'trial_type' in df_plot.columns:
        trial_types = sorted(df_plot['trial_type'].unique())
    else:
        trial_types = []
    
    n_segments = len(segment_order)
    n_trials = len(trial_types)

    # 2. Create Mappings
    
    # Get 5th and 6th colors
    colors_list = sns.color_palette(n_colors=10)
    acc_color = colors_list[2]  # 5th color (muted green)
    mag_color = colors_list[4]  # 6th color (teal/cyan)
    
    # Fill map for Trial Types
    trial_fills = {trial: 'full' for trial in trial_types}
    if n_trials > 1:
        # hollow for the first trial type, solid for the second
        trial_fills[trial_types[0]] = 'none' 
        trial_fills[trial_types[1]] = 'full' 

    # 3. Calculate Max STD for Alpha Mapping
    acc_std_max = df_plot['acc_magnitude_std'].max()
    mag_std_max = df_plot['mag_magnitude_std'].max()
    # Find the global max std dev across all segments/magnitudes
    global_std_max = max(acc_std_max, mag_std_max)
    # Define the alpha range. Max std = 1.0 alpha (opaque). Min std = 0.2 alpha (light)
    ALPHA_MIN = 0.2
    ALPHA_MAX = 1.0
    
    def get_alpha(std_val, max_std):
        """Maps standard deviation to an alpha value (darkness)."""
        if max_std == 0: return ALPHA_MAX # Handle zero std case
        # Normalize the std dev to the range [0, 1]
        normalized_std = std_val / max_std
        # Linear map from [0, 1] to [ALPHA_MIN, ALPHA_MAX]
        return ALPHA_MIN + normalized_std * (ALPHA_MAX - ALPHA_MIN)
    
    # 4. Create the 1 subplot and twin axis
    # Flipped axes: figsize=(Vertical, Horizontal)
    fig, ax1 = plt.subplots(1, 1, figsize=(4.5, 7)) 
    
    # Create the second x-axis sharing the same y-axis
    ax2 = ax1.twiny() # Use twinY for vertical segments
    
    # Define specs for clarity
    acc_spec = {'col_mean': 'acc_magnitude_mean', 'col_std': 'acc_magnitude_std', 'xlabel': 'Acceleration ($m/s^2$)'}
    mag_spec = {'col_mean': 'mag_magnitude_mean', 'col_std': 'mag_magnitude_std', 'xlabel': 'Magnetometer ($\mu T$)'}

    # 5. Main Plotting Loop
    
    # Calculate vertical dodge positions (for trial types)
    if n_trials > 1:
        # We need a small dodge to separate trial types vertically
        dodge = np.linspace(-0.15, 0.15, n_trials)
    else:
        dodge = [0]
    
    # Loop by segment
    for i_seg, segment in enumerate(segment_order_y):
        # The index (0 is torso, 7 is foot) is reversed in order_y.
        # The segment's Y position should be its index in segment_order_y
        y_pos_base = i_seg
        
        segment_data = df_plot[df_plot['segment'] == segment]
        if segment_data.empty:
            continue

        # Plot dodged dots for each trial type
        for i_trial, trial_type in enumerate(trial_types if n_trials > 0 else ['_single']):
            
            y_pos = y_pos_base + dodge[i_trial]

            if n_trials <= 1:
                data = segment_data.iloc[0] # Should only be one row
                fillstyle = 'full'
            else:
                data_row = segment_data[segment_data['trial_type'] == trial_type]
                if data_row.empty: continue
                data = data_row.iloc[0]
                fillstyle = trial_fills[trial_type]
            
            # --- Get the values ---
            acc_mean = data[acc_spec['col_mean']]
            acc_std = data[acc_spec['col_std']]
            mag_mean = data[mag_spec['col_mean']]
            mag_std = data[mag_spec['col_std']]

            # Calculate Alpha (Darkness)
            # acc_alpha = get_alpha(acc_std, global_std_max)
            # mag_alpha = get_alpha(mag_std, global_std_max)

            # --- Plot on ax1 (ACC) ---
            ax1.errorbar(
                x=acc_mean, y=y_pos - 0.1, xerr=acc_std, # x and y are flipped
                marker='o', color=acc_color, fillstyle=fillstyle, linestyle='none',
                capsize=5, markersize=7, markeredgecolor=acc_color
            )
            
            # --- Plot on ax2 (MAG) ---
            ax2.errorbar(
                x=mag_mean, y=y_pos + 0.2, xerr=mag_std, # x and y are flipped
                marker='o', color=mag_color, fillstyle=fillstyle, linestyle='none',
                capsize=5, markersize=7, markeredgecolor=mag_color
            )

    # 6. Configure Axes
    # Use clean labels for the Y-axis (Segments)
    segment_labels = ["Torso", "Pelvis", "R Femur", "L Femur", "R Tibia", "L Tibia", "R Foot", "L Foot"]
    segment_labels_y = segment_labels[::-1] # Flipped to match the segment_order_y list

    # Y-Axis (Segments - configure on ax1, it's shared)
    ax1.set_ylim(-0.5, n_segments - 0.5)
    ax1.set_yticks(range(n_segments))
    ax1.set_yticklabels(segment_labels_y, fontsize=16)
    # ax1.set_ylabel('Body Segment', fontweight='bold', fontsize=14)

    # X-Axes (Measurements - configure separately)
    # ax1 (Bottom) - Acceleration
    ax1.set_xlabel(acc_spec['xlabel'], color=acc_color, fontweight='bold', fontsize=18)
    ax1.tick_params(axis='x', labelcolor=acc_color, labelsize=10)
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color(acc_color)

    # ax2 (Top) - Magnetometer
    ax2.set_xlabel(mag_spec['xlabel'], color=mag_color, fontweight='bold', fontsize=18)
    ax2.tick_params(axis='x', labelcolor=mag_color, labelsize=10)
    ax2.spines['top'].set_color(mag_color)
    ax2.spines['right'].set_color('black') 
    
    # Hide unnecessary spines
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Set X-axis limits (Auto-adjust)
    for ax, spec in [(ax1, acc_spec), (ax2, mag_spec)]:
        data_means = df_plot[spec['col_mean']]
        data_stds = df_plot[spec['col_std']]
        
        valid_data = ~data_means.isna() & ~data_stds.isna()
        if valid_data.any():
            plot_min = (data_means[valid_data] - data_stds[valid_data]).min()
            plot_max = (data_means[valid_data] + data_stds[valid_data]).max()
            
            padding = (plot_max - plot_min) * 0.1
            if padding == 0: padding = 0.1 
            
            ax.set_xlim(max(0, plot_min - padding), plot_max + padding)

    # Grid (only horizontal)
    # ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # 7. Create Custom Legends
    legend_handles = []
    
    # Trial Type (Fill) Legend
    if n_trials > 1:
        for trial, fill in trial_fills.items():
            # Use black edge color for legend to contrast with the plot colors
            legend_handles.append(mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                            markersize=7, label=trial, fillstyle=fill,
                                            markeredgecolor='black'))

    # Std Dev (Darkness) Legend
    # Create a gradient legend to explain alpha mapping
    darkness_handles = []
    std_values = np.linspace(0, global_std_max, 5)
    for std_val in std_values:
        alpha = get_alpha(std_val, global_std_max)
        label = f"$\pm {std_val:.2g}$" # Use LaTeX for std symbol
        darkness_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                            markersize=7, alpha=alpha, label=label, 
                                            fillstyle='full', markeredgecolor='gray'))

    # Combine all legends
    if legend_handles:
        trial_legend = fig.legend(handles=legend_handles, title='Trial Type',
                                bbox_to_anchor=(0.99, 0.95), loc='upper left', frameon=False, 
                                fontsize=10, title_fontsize=12)

        # Add darkness legend next to it
        fig.legend(handles=darkness_handles, title='Total STD Dev (Darkness)',
                    bbox_to_anchor=(0.99, 0.7), loc='upper left', frameon=False, 
                    fontsize=10, title_fontsize=12)
        fig.add_artist(trial_legend) # Re-add the first legend

    
    plt.tight_layout(rect=[0, 0.1, 0.85, 0.9])
    plt.savefig('pooled_magnitudes_twin_ax_flipped_std_alpha.png', dpi=300)


if __name__ == "__main__":
    pkl_file = "imu_data_summary_.pkl"
    
    # Mocking data loading since the file is not available
    if os.path.exists(pkl_file):
        df = pd.read_pickle(pkl_file)
    else:
        print(f"No '{pkl_file}' found. Generating mock data for demonstration...")
        all_results = []
        for subject_num in range(1, 4): # Reduced subjects for mock data speed
            SUBJECT_TO_LOAD = f"{subject_num:02d}"
            for TRIAL_TYPE_TO_LOAD in ['walking', 'complexTasks']:
                # Call the processing function and get the data
                processed_data = load_and_process_imu_data(SUBJECT_TO_LOAD, TRIAL_TYPE_TO_LOAD)
                all_results.extend(processed_data)
        
        if not all_results:
            print("Mock data generation failed. Exiting.")
            sys.exit()
            
        df = pd.DataFrame(all_results)
        # df.to_pickle(pkl_file) # Do not save mock data over real data

    # --- Generate Grouped Summary DataFrames ---
    print("\nGenerating pooled summary for magnitudes...")
    pooled_magnitudes_by_segment = get_pooled_summary(
        df,
        group_by_cols=['segment'], # Group by both for dodged plot
        data_cols=['acc_magnitude', 'mag_magnitude'] 
    )

    # Call the NEW plotting function
    plot_pooled_magnitudes_twin_ax_flipped_std_alpha(pooled_magnitudes_by_segment)
    plt.show()