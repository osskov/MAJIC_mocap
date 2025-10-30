import os
from typing import Dict, List, Any
import pandas as pd
from src.toolchest.PlateTrial import PlateTrial 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
import seaborn as sns

PKL_FILE = os.path.abspath(os.path.join("data", "all_subject_imu_data_summary.pkl"))
RENAME_SEGMENTS = {
    'torso_imu': 'Torso',
    'pelvis_imu': 'Pelvis',
    'femur_r_imu': 'Femur',
    'femur_l_imu': 'Femur',
    'tibia_r_imu': 'Tibia',
    'tibia_l_imu': 'Tibia',
    'calcn_r_imu': 'Foot',
    'calcn_l_imu': 'Foot',
}
SUBJECTS_TO_LOAD = [f"{i:02d}" for i in range(1, 12)]
TRIAL_TYPES_TO_LOAD = ['walking', 'complexTasks']
SHOW_PLOTS = True
SAVE_PLOTS = True
ACC_PALETTE = 'Reds'
MAG_PALETTE = 'Blues'

def load_and_process_imu_data(subject_num: str, trial_type: str, max_frames: int = 60000) -> List[Dict[str, Any]]:
    # This is left as-is from the user's prompt, but relies on a mock PlateTrial to run
    # Mocking data loading for demonstration purposes
    subject_id = f"Subject{subject_num}"
    # print(f"--- Loading data for {subject_id}, {trial_type} ---")
    
    results = [] 
    
    try:
        data_folder_path = os.path.join("data", subject_id, trial_type)
        data_folder_path = os.path.abspath(data_folder_path)
        
        plate_trials = PlateTrial.load_trial_from_folder(data_folder_path)

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
    Parameters:
    - df: DataFrame containing the data.
    - group_by_cols: Columns to group by (e.g., ['segment', 'trial_type', 'subject']).
    - data_cols: Columns for which to calculate pooled mean and std dev (e.g., ['acc_magnitude', 'mag_magnitude', 'acc_x', 'acc_y', 'acc_z', 'mag_x', 'mag_y', 'mag_z']).
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

def plot_pooled_imu_distributions(df: pd.DataFrame):
    """
    Plots the distribution of per-trial IMU data (pooled mean as dot, pooled std as whisker)
    using a dot-and-whisker plot with dual Y-axes for ACC and MAG data.
    
    Parameters:
    - df: DataFrame containing the pooled summary data ({col}_mean and {col}_std).
    """
    
    # 1. Infer/Validate columns and Prepare DataFrame
    group_by_cols = df.index.names if df.index.names != [None] else []
    
    plot_df = df.copy().reset_index(names=group_by_cols, drop=(df.index.names == [None]))
    
    mean_cols = [col for col in plot_df.columns if col.endswith('_mean')]
    std_cols = [col for col in plot_df.columns if col.endswith('_std')]
    
    acc_data_map = {col.replace('_mean', ''): (col, col.replace('_mean', '_std')) for col in mean_cols if col.startswith('acc_')}
    mag_data_map = {col.replace('_mean', ''): (col, col.replace('_mean', '_std')) for col in mean_cols if col.startswith('mag_')}

    if not acc_data_map and not mag_data_map:
        print("Error: No 'acc_' or 'mag_' mean/std columns found for plotting.")
        return

    # 2. Determine X-axis and HUE columns (MODIFIED)
    n_groups = len(group_by_cols)
    
    if n_groups == 0:
        # No grouping: X-axis is 'All Data', and HUE is a dummy for a single color
        x_col = 'X_Group'
        hue_col = 'HUE_Group'
        plot_df[x_col] = 'All Data'
        plot_df[hue_col] = '__single_hue__'
        print("Using dummy columns for X-axis and HUE.")
        
    elif n_groups == 1:
        # One grouping column: Use it for X-axis, and HUE is a dummy for a single color per X-group
        x_col = group_by_cols[0]
        hue_col = 'HUE_Group'
        plot_df[hue_col] = '__single_hue__'
        print(f"Using single group by column '{x_col}' for X-axis. Using dummy hue.")
        
    else: 
        # Two or more grouping columns: Use all but the last for X-axis (combined), last for HUE
        combined_group_name = 'X_Group'
        plot_df[combined_group_name] = plot_df[group_by_cols[:-1]].apply(
            lambda row: ' | '.join(row.values.astype(str)), axis=1
        )
        x_col = combined_group_name
        hue_col = group_by_cols[-1]
        print(f"Using combined groups for X-axis ('{x_col}') and '{hue_col}' for HUE.")

    # 3. Data Wrangling (Melting) for Errorbars
    
    # 3a. Prepare Long DataFrame for ACC
    acc_long_df = pd.DataFrame()
    for base_col, (mean_col, std_col) in acc_data_map.items():
        # Ensure we are using the resolved x_col and hue_col
        temp_df = plot_df[[x_col, hue_col, mean_col, std_col]].copy()
        temp_df.rename(columns={mean_col: 'Mean', std_col: 'StdDev'}, inplace=True)
        temp_df['Variable'] = base_col
        temp_df['Core_Variable'] = base_col.replace('acc_', '') 
        acc_long_df = pd.concat([acc_long_df, temp_df])
    acc_long_df['Sensor'] = 'ACC'

    # 3b. Prepare Long DataFrame for MAG
    mag_long_df = pd.DataFrame()
    for base_col, (mean_col, std_col) in mag_data_map.items():
        # Ensure we are using the resolved x_col and hue_col
        temp_df = plot_df[[x_col, hue_col, mean_col, std_col]].copy()
        temp_df.rename(columns={mean_col: 'Mean', std_col: 'StdDev'}, inplace=True)
        temp_df['Variable'] = base_col
        temp_df['Core_Variable'] = base_col.replace('mag_', '') 
        mag_long_df = pd.concat([mag_long_df, temp_df])
    mag_long_df['Sensor'] = 'MAG'

    all_long_df = pd.concat([acc_long_df, mag_long_df])

    # 4. Plotting Setup
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx() 
    
    # Get hue levels
    hue_levels = plot_df[hue_col].unique()
    n_hues = len(hue_levels)

    if hue_levels[0] == '__single_hue__':
        # Use single colors if only a dummy hue exists
        acc_hue_colors = {'__single_hue__': sns.color_palette(ACC_PALETTE, 1)[0]}
        mag_hue_colors = {'__single_hue__': sns.color_palette(MAG_PALETTE, 1)[0]}
        
        # Adjust hue_col for the legend title
        hue_legend_title = 'Sensor Color'
    else:
        # Create separate color palettes for ACC and MAG
        acc_palette = sns.color_palette(ACC_PALETTE, n_hues+1)
        mag_palette = sns.color_palette(MAG_PALETTE, n_hues+1)

        acc_hue_colors = dict(zip(hue_levels, acc_palette))
        mag_hue_colors = dict(zip(hue_levels, mag_palette))
        hue_legend_title = f'Group Color ({hue_col})'
    
    # Map shapes based on Core_Variable
    core_variables = all_long_df['Core_Variable'].unique() 
    available_markers = ['o', 's', 'D', '^'] 
    
    variable_shapes = {core_var: available_markers[i % len(available_markers)] 
                       for i, core_var in enumerate(core_variables)}

    # Setup for positioning (first level: X-axis categories)
    x_tick_labels = plot_df[x_col].unique()
    n_x_groups = len(x_tick_labels)
    x_pos_map = dict(zip(x_tick_labels, np.arange(n_x_groups)))

    # SECOND LEVEL DODGING SETUP (Sensor x Core_Variable x Hue)
    sensors = all_long_df['Sensor'].unique()
    n_dodged_elements = n_hues * len(core_variables) * len(sensors) # <-- MODIFIED: Include sensor

    total_dodge_width = 0.8 
    
    dot_spacing = total_dodge_width / n_dodged_elements
    
    base_offsets = np.linspace(
        -total_dodge_width / 2 + dot_spacing / 2, 
        total_dodge_width / 2 - dot_spacing / 2, 
        n_dodged_elements
    )

    combined_offset_map = {}
    i = 0
    
    # Loop over sensors first
    for s_level in sensors: 
        for h_level in hue_levels:
            for v_level in core_variables: 
                # Key now includes the Sensor
                combined_offset_map[(s_level, h_level, v_level)] = base_offsets[i] 
                i += 1
            
    # Store legend handles (for both color and shape)
    legend_proxies_color = {} 
    legend_proxies_shape = {} 

    # 5. Plot Errorbars (Dot-and-Whisker)
    for index, row in all_long_df.iterrows():
        
        sensor = row['Sensor'] # <-- Now used in offset calculation
        
        x_group = str(row[x_col])
        hue_level = str(row[hue_col])
        variable_level = row['Variable']
        core_variable = row['Core_Variable']

        # Calculate X position
        x_base = x_pos_map[x_group] 
        # MODIFIED: Use the new key including sensor
        x_offset = combined_offset_map[(sensor, hue_level, core_variable)] 
        x_pos = x_base + x_offset
        
        mean_val = row['Mean']
        std_dev = row['StdDev']
        
        # Determine color based on sensor and hue_level
        color = acc_hue_colors[hue_level] if sensor == 'ACC' else mag_hue_colors[hue_level]

        # Calculate the darker color for the edge and whisker
        try:
            rgb = mcolors.to_rgb(color)
            darker_color = tuple(c * 0.6 for c in rgb)
        except ValueError:
            darker_color = 'black' # Fallback
    
        # Determine marker based on Core_Variable
        marker = variable_shapes[core_variable]
        
        target_ax = ax1 if sensor == 'ACC' else ax2
        
        # Plot the error bar
        target_ax.errorbar(
            x=x_pos, 
            y=mean_val, 
            yerr=std_dev, 
            fmt=marker,        
            capsize=5,         
            elinewidth=1.5,    
            markerfacecolor=color,
            markeredgecolor=darker_color,
            markersize=7,
            color=darker_color,      
            zorder=10           
        )
        
        # Create legend proxies for colors (by hue_level and sensor) (MODIFIED)
        # Use the actual hue level unless it's the dummy one, then just use Sensor
        if hue_level == '__single_hue__':
             color_key = sensor
             label = f'Sensor: {sensor.replace("ACC", "Accelerometer").replace("MAG", "Magnetometer")}'
        else:
             color_key = (hue_level, sensor)
             label = f'{hue_level} ({sensor.replace("ACC", "Accelerometer").replace("MAG", "Magnetometer")} Color)'
             
        if color_key not in legend_proxies_color:
            proxy = plt.Line2D(
                [0], [0], 
                marker='s', 
                color='w',
                markeredgecolor=darker_color, 
                markerfacecolor=color, 
                markersize=8, 
                linestyle='',
                label=label
            )
            legend_proxies_color[color_key] = proxy
            
        # Create legend proxies for shapes (by core_variable)
        shape_key = core_variable
        if shape_key not in legend_proxies_shape:
            proxy = plt.Line2D(
                [0], [0], 
                marker=marker, 
                color='w',
                markeredgecolor='black', 
                markerfacecolor='black', 
                markersize=8, 
                linestyle='',
                label=f'{core_variable}' 
            )
            legend_proxies_shape[shape_key] = proxy
            
    # 6. Final Plot Customization and Legend 

    # Turn off the spines between the two y-axes
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # Set X-axis ticks and labels
    ax1.set_xticks(np.arange(n_x_groups))
    ax1.set_xticklabels(x_tick_labels, rotation=0, ha='center')
    ax1.set_xlabel("", fontsize=0)
    
    # Set Y-axis 1 (ACC) style
    acc_color = sns.color_palette(ACC_PALETTE, 1)[0]
    ax1.set_ylabel(r'Acceleration Mean ($\text{m/s}^2$)', color=acc_color, fontsize=12,)
    ax1.tick_params(axis='y', labelcolor=acc_color)
    # Calculate the darker color for the edge and whisker
    try:
        rgb = mcolors.to_rgb(acc_color)
        darker_color = tuple(c * 0.6 for c in rgb)
    except ValueError:
        darker_color = 'black' # Fallback
    ax1.spines['left'].set_color(darker_color)
    # ax1.grid(True, axis='y', linestyle='--', alpha=0.7) 

    # Set Y-axis 2 (MAG) style
    mag_color = sns.color_palette(MAG_PALETTE, 1)[0]
    ax2.set_ylabel(r'Magnetometer Mean ($\mu\text{T}$)', color=mag_color, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=mag_color)
        # Calculate the darker color for the edge and whisker
    try:
        rgb = mcolors.to_rgb(mag_color)
        darker_color = tuple(c * 0.6 for c in rgb)
    except ValueError:
        darker_color = 'black' # Fallback
    ax2.spines['right'].set_color(darker_color)
    ax2.grid(False) 
    
    ax1.set_title(f'Pooled IMU Distribution (Mean $\pm$ STD) by {x_col}', fontsize=14)
    
    # Combine and place the legend
    color_handles = sorted(legend_proxies_color.values(), key=lambda x: x.get_label())
    shape_handles = sorted(legend_proxies_shape.values(), key=lambda x: x.get_label())

    # Create the first legend for colors (Hue levels per sensor)
    legend1 = ax1.legend(
        handles=color_handles, 
        labels=[h.get_label() for h in color_handles], 
        title=hue_legend_title, # Use the derived title
        loc='upper left', 
        bbox_to_anchor=(1.1, 1) 
    )
    ax1.add_artist(legend1) 

    # 2. Get the bounding box of the legend in pixel coordinates (display coordinates).
    legend_bbox_display = legend1.get_window_extent()

    # 3. Convert the bounding box from pixel (display) coordinates to Figure coordinates (0 to 1).
    # We invert the figure's transform to map display pixels back to the figure's fractional space.
    legend_bbox_fig_coords = legend_bbox_display.transformed(fig.transFigure.inverted())

    # 4. Extract the bottom edge (ymin)
    legend_bottom_y_fig_coords = legend_bbox_fig_coords.ymin

    # Create the second legend for shapes (Core Variables)
    legend2 = ax1.legend(
        handles=shape_handles, 
        labels=[h.get_label() for h in shape_handles], 
        title='Sensor Axis', 
        loc='upper left', 
        # MODIFIED: Position based on the bottom of the first legend
        bbox_to_anchor=(1.1, legend_bottom_y_fig_coords - 0.05) # <-- Slightly adjusted y_pos
    )

    fig.tight_layout() 
    fig.subplots_adjust(right=0.7) 
    if SAVE_PLOTS:
        output_path = os.path.abspath(os.path.join("plots", f"pooled_imu_distribution_{x_col}_{hue_col}.png"))
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    if SHOW_PLOTS:
        plt.show()

if __name__ == "__main__":
    # Mocking data loading since the file is not available
    if os.path.exists(PKL_FILE):
        df = pd.read_pickle(PKL_FILE)
    else:
        print(f"No '{PKL_FILE}' found. Generating mock data for demonstration...")
        all_results = []
        for subject_num in SUBJECTS_TO_LOAD:
            for trial_type in TRIAL_TYPES_TO_LOAD:
                # Call the processing function and get the data
                processed_data = load_and_process_imu_data(subject_num, trial_type)
                all_results.extend(processed_data)
        
        if not all_results:
            print("Data generation failed. Exiting.")
            sys.exit()
            
        df = pd.DataFrame(all_results)
        df.to_pickle(PKL_FILE)

    # Order columns according to RENAME_SEGMENTS, then rename columns
    df['segment'] = df['segment'].replace(RENAME_SEGMENTS)
    
    # Get unique values while preserving order (Python 3.7+)
    segment_order = list(dict.fromkeys(RENAME_SEGMENTS.values()))
    
    df['segment'] = pd.Categorical(df['segment'], categories=segment_order, ordered=True)
    df = df.sort_values('segment')

    # Filter down to only subjects and trial types of interest
    df = df[df['subject'].isin([f"Subject{num}" for num in SUBJECTS_TO_LOAD])]
    df = df[df['trial_type'].isin(TRIAL_TYPES_TO_LOAD)]

    # --- Generate Grouped Summary DataFrames ---
    print("\nGenerating pooled summary for magnitudes...")
    pooled_magnitudes_by_segment = get_pooled_summary(
        df,
        group_by_cols=['segment'],
        data_cols=['acc_magnitude', 'mag_magnitude'] 
    )
    # Call the NEW plotting function
    plot_pooled_imu_distributions(pooled_magnitudes_by_segment)