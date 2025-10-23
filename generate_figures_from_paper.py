import matplotlib.pyplot as plt
import matplotlib.container as mpc
from typing import List, Dict, Tuple
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from scipy.spatial.transform import Rotation
import numpy as np
import scipy.stats as stats
import pandas as pd
import os
import seaborn as sns
import itertools
import json

# --- Assume JOINT_SEGMENT_DICT is defined here ---
# Example:
JOINT_SEGMENT_DICT = {
    'R_Ankle': ('R_Tibia', 'R_Foot'),
    'L_Ankle': ('L_Tibia', 'L_Foot'),
    'R_Knee': ('R_Femur', 'R_Tibia'),
    'L_Knee': ('L_Femur', 'L_Tibia'),
    'R_Hip': ('Pelvis', 'R_Femur'),
    'L_Hip': ('Pelvis', 'L_Femur'),
    'Pelvis_Trunk': ('Pelvis', 'Trunk')
}
# ------------------------------------------------

# --- Data Loading Functions (Copied from your prompt) ---

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
        parent_trial = next((p for p in plate_trials if parent_name in p.name), None)
        child_trial = next((p for p in plate_trials if child_name in p.name), None)
        
        if not parent_trial or not child_trial:
            continue
            
        timestamps = parent_trial.imu_trace.timestamps
        
        try:
            joint_rotations_mat = [Rwp.T @ Rwc for Rwp, Rwc in zip(parent_trial.world_trace.rotations, child_trial.world_trace.rotations)]
            
            if not joint_rotations_mat:
                print(f"Warning: No rotations calculated for {joint_name}. Skipping.")
                continue

            joint_rotations_euler = Rotation.from_matrix(joint_rotations_mat).as_euler(euler_order) # (N, 3)
        
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error processing rotations for {joint_name} ({parent_name}/{child_name}): {e}. Skipping.")
            continue

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
        return pd.DataFrame() 

    final_df = pd.concat(all_joint_data, ignore_index=True)
    
    return final_df

def load_joint_traces_for_subject_df(subject_id: str, 
                                     trial_type: str,
                                     methods: List[str] = ['Marker', 'Madgwick', 'Mag Free', 'Never Project'],
                                     euler_order: str = 'ZYX') -> pd.DataFrame:
    """
    Loads, resyncs, and processes joint traces for a specific subject and
    trial type, returning a consolidated pandas DataFrame with Euler angles.
    """
    try:
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
        
        joint_df = get_joint_traces_df(trials_list, euler_order=euler_order)
        
        if not joint_df.empty:
            joint_df['method'] = method_name
            all_method_dfs.append(joint_df)

    if not all_method_dfs:
        print(f"No joint data processed for {subject_id}. Returning empty DataFrame.")
        return pd.DataFrame()
        
    final_df = pd.concat(all_method_dfs, ignore_index=True)
    
    final_df['subject_id'] = subject_id
    final_df['trial_type'] = trial_type
    
    meta_cols = ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']
    euler_cols = [f'euler_{c}_rad' for c in euler_order] 
    
    columns_order = meta_cols + euler_cols
    final_df = final_df[columns_order]

    final_df = final_df.set_index(
        ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']
    ).sort_index()

    return final_df

# --- RMSE Calculation Function (Copied from your prompt) ---

def calculate_rmse_and_std(all_data_df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """
    Calculates RMSE and STD of the error between all methods and the 'Marker' method.
    """
    
    original_index_names = all_data_df.index.names
    data_stacked = all_data_df.stack().rename_axis(original_index_names + ['axis'])
    
    try:
        data_wide = data_stacked.unstack('method')
    except KeyError:
        print("Error: 'method' not found in the DataFrame's index.")
        return pd.DataFrame()

    if 'Marker' not in data_wide.columns:
        print("Warning: 'Marker' method not found. Cannot calculate error.")
        return pd.DataFrame()

    marker_series = data_wide['Marker']
    error_df = data_wide.sub(marker_series, axis=0)
    
    error_df = error_df.drop('Marker', axis=1, errors='ignore')
    error_df_deg = error_df * 180 / np.pi  # Convert radians to degrees

    squared_error_df_deg = error_df_deg ** 2

    valid_group_by = [level for level in group_by if level in error_df_deg.index.names]
    
    if not valid_group_by:
        grouped_error = error_df_deg
        grouped_sq_error = squared_error_df_deg
    else:
        try:
            grouped_error = error_df_deg.groupby(level=valid_group_by)
            grouped_sq_error = squared_error_df_deg.groupby(level=valid_group_by)
        except KeyError as e:
            print(f"Error: Grouping level not found: {e}. Available levels are: {error_df_deg.index.names}")
            return pd.DataFrame()

    rmse = np.sqrt(grouped_sq_error.mean())
    std_dev = grouped_error.std()
    mean = grouped_error.mean()
    median = grouped_error.median()
    q_75 = grouped_error.quantile(0.75)
    q_25 = grouped_error.quantile(0.25)

    results_df = pd.concat(
        {
            'RMSE': rmse,
            'MAE': grouped_error.apply(lambda x: x.abs().mean()),
            'Bias': mean,
            'STD': std_dev,
            'MAD': grouped_error.apply(lambda x: (x - x.mean()).abs().median()),
            'Skew': grouped_error.apply(stats.skew),
            'Kurtosis': grouped_error.apply(stats.kurtosis),
            'Q25': q_25,
            'Median': median,
            'Q75': q_75,
        },
        axis=1
    )
    
    if not results_df.empty and isinstance(results_df.columns, pd.MultiIndex):
        results_df = results_df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    return results_df

# --- Visualization Function (From your prompt) ---

def plot_metric_distribution(stats_df: pd.DataFrame, 
                             metric: str = 'RMSE', 
                             group_by: str = 'joint_name', 
                             title_prefix: str = '',
                             save_dir: str = 'plots'):
    """
    Generates and saves boxplots for a given metric, grouped by a specified category.

    Args:
        stats_df (pd.DataFrame): The summary statistics DataFrame 
                                 (output of calculate_rmse_and_std).
        metric (str): The metric to plot (e.g., 'RMSE', 'Bias', 'MAE').
        group_by (str): The index level to group by (e.g., 'joint_name', 'trial_type').
        title_prefix (str): Text to prepend to the plot title.
        save_dir (str): Directory to save plots.
    """
    print(f"Generating plot for {metric} grouped by {group_by}...")
    
    # 1. Select the metric
    try:
        metric_df = stats_df.xs(metric, level=1, axis=1)
    except KeyError:
        print(f"Error: Metric '{metric}' not found in stats_df columns.")
        return

    # 2. Melt the DataFrame for Seaborn
    # Index becomes columns, 'method' is stacked
    metric_df.columns.name = 'method'
    
    metric_long_df = metric_df.stack().rename(metric).reset_index()

    # 3. Create the plot
    g = sns.catplot(
        data=metric_long_df,
        x='method',
        y=metric,
        col=group_by,
        kind='box',
        col_wrap=4,
        sharey=False,
        showfliers=False # Hide outliers for a cleaner plot
    )
    
    # 4. Improve formatting
    title = f'{title_prefix} ({metric}) by {group_by} (deg)'
    g.fig.suptitle(title, y=1.03)
    g.set_xticklabels(rotation=45, ha='right')
    g.set_titles("{col_name}")
    g.set_axis_labels("Method", f"{metric} (degrees)")
    
    plt.tight_layout()
    
    # 5. Save the figure
    save_path = os.path.join(save_dir, f"{title_prefix.lower()}_{metric.lower()}_by_{group_by}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close() # Close plot to free memory

# --- Statistical Analysis Function (From your prompt) ---

def perform_statistical_analysis(error_df: pd.DataFrame, 
                                 group_by: str = 'joint_name',
                                 p_alpha: float = 0.05):
    """
    Performs statistical analysis on the raw error time-series data.

    Tests performed:
    1.  Bias vs. Zero: One-sample Wilcoxon signed-rank test (is median error different from 0?)
    2.  Overall Comparison: Kruskal-Wallis H-test (are any methods different from each other?)
    3.  Pairwise Comparison: Paired Wilcoxon test with Bonferroni correction
        (which specific pairs are different?)

    Args:
        error_df (pd.DataFrame): The raw error time-series (in degrees).
                                 Index should be multi-index, columns are methods.
        group_by (str): The index level to group by (e.g., 'joint_name', 'trial_type', None).
        p_alpha (float): The significance level.

    Returns:
        dict: A dictionary containing the test results.
    """
    
    all_results = {}
    methods = error_df.columns.tolist()

    if group_by is None:
        groups_to_iterate = [('Overall', error_df)]
    else:
        if group_by not in error_df.index.names:
            print(f"Error: Grouping level '{group_by}' not in error_df index. Skipping stats.")
            return {}
        groups_to_iterate = error_df.groupby(level=group_by)

    print(f"\n--- Performing Statistical Analysis (Grouped by: {group_by}) ---")

    for name, group_data in groups_to_iterate:
        group_results = {
            'Bias_vs_Zero': {},
            'Overall_Comparison_Kruskal': {},
            'Pairwise_Comparison_Wilcoxon': {}
        }
        
        # 1. Bias vs. Zero (One-sample Wilcoxon)
        for method in methods:
            data = group_data[method].dropna()
            if len(data) < 10: # Not enough data to test
                continue
            try:
                stat, p_val = stats.wilcoxon(data)
                group_results['Bias_vs_Zero'][method] = {
                    'median_error': data.median(),
                    'p_value': p_val,
                    'significant': p_val < p_alpha
                }
            except ValueError:
                 group_results['Bias_vs_Zero'][method] = {
                    'median_error': data.median(),
                    'p_value': 1.0, # Occurs if all values are zero
                    'significant': False
                }


        # 2. Overall Comparison (Kruskal-Wallis)
        data_for_kruskal = [group_data[method].dropna() for method in methods]
        data_for_kruskal = [d for d in data_for_kruskal if len(d) > 0] # Remove empty
        
        if len(data_for_kruskal) < 2:
            continue # Can't compare
            
        try:
            stat, p_val = stats.kruskal(*data_for_kruskal)
            group_results['Overall_Comparison_Kruskal'] = {
                'statistic': stat,
                'p_value': p_val,
                'significant': p_val < p_alpha
            }
        except ValueError as e:
            print(f"  Kruskal test failed for group {name}: {e}")
            group_results['Overall_Comparison_Kruskal'] = None

        # 3. Pairwise Post-hoc (Paired Wilcoxon w/ Bonferroni)
        if group_results['Overall_Comparison_Kruskal'].get('significant', False):
            comparisons = list(itertools.combinations(methods, 2))
            num_comparisons = len(comparisons)
            # Bonferroni-corrected p-value threshold
            corrected_alpha = p_alpha / num_comparisons 
            
            post_hoc_results = {}
            for m1, m2 in comparisons:
                data1 = group_data[m1].dropna()
                data2 = group_data[m2].dropna()
                
                # We need paired data, so align and drop NaNs
                paired_data = pd.DataFrame({'m1': data1, 'm2': data2}).dropna()
                
                if len(paired_data) < 10:
                    continue
                
                try:
                    stat, p_val = stats.wilcoxon(paired_data['m1'], paired_data['m2'])
                    post_hoc_results[f"{m1}_vs_{m2}"] = {
                        'p_value': p_val,
                        'significant_bonferroni': p_val < corrected_alpha
                    }
                except ValueError as e:
                    # This can happen if all differences are zero
                    post_hoc_results[f"{m1}_vs_{m2}"] = {
                        'p_value': 1.0,
                        'significant_bonferroni': False
                    }
            group_results['Pairwise_Comparison_Wilcoxon'] = post_hoc_results

        all_results[name] = group_results
        
    return all_results

# ---------------------------------------------------------------------
# --- NEW PLOTTING FUNCTIONS ------------------------------------------
# ---------------------------------------------------------------------

def plot_overall_metric_distribution(stats_df: pd.DataFrame, 
                                     metric: str = 'RMSE', 
                                     title_prefix: str = '',
                                     save_dir: str = 'plots'):
    """
    Generates and saves a single boxplot for a given metric, 
    showing the overall performance of each method.

    Args:
        stats_df (pd.DataFrame): The summary statistics DataFrame.
        metric (str): The metric to plot (e.g., 'RMSE', 'Bias').
        title_prefix (str): Text to prepend to the plot title.
        save_dir (str): Directory to save plots.
    """
    print(f"Generating overall plot for {metric}...")
    
    try:
        metric_df = stats_df.xs(metric, level=1, axis=1)
    except KeyError:
        print(f"Error: Metric '{metric}' not found in stats_df columns.")
        return

    metric_df.columns.name = 'method'
    metric_long_df = metric_df.stack().rename(metric).reset_index()

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=metric_long_df, x='method', y=metric, showfliers=False)
    
    title = f'{title_prefix} Overall {metric} Distribution (deg)'
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Method")
    plt.ylabel(f"{metric} (degrees)")
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{title_prefix.lower()}_overall_{metric.lower()}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_metric_heatmap(stats_df: pd.DataFrame, 
                        metric: str = 'RMSE', 
                        breakdown_by: str = 'joint_name', 
                        title_prefix: str = '',
                        save_dir: str = 'plots'):
    """
    Generates a heatmap showing the average of a metric, 
    broken down by method and another category (e.g., joint).

    Args:
        stats_df (pd.DataFrame): The summary statistics DataFrame.
        metric (str): The metric to plot (e.g., 'RMSE', 'MAE').
        breakdown_by (str): The index level to use for the y-axis (e.g., 'joint_name').
        title_prefix (str): Text to prepend to the plot title.
        save_dir (str): Directory to save plots.
    """
    print(f"Generating heatmap for {metric} by {breakdown_by}...")

    try:
        metric_df = stats_df.xs(metric, level=1, axis=1)
    except KeyError:
        print(f"Error: Metric '{metric}' not found in stats_df columns.")
        return
        
    if breakdown_by not in metric_df.index.names:
        print(f"Error: Breakdown level '{breakdown_by}' not in stats_df index. Skipping heatmap.")
        return

    # Calculate the mean for each method and breakdown group
    heatmap_data = metric_df.groupby(level=breakdown_by).mean()
    
    plt.figure(figsize=(14, 10))
    # Use a sequential colormap where lower is better (e.g., 'viridis_r' or 'rocket_r')
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt=".2f", 
                cmap="viridis_r", 
                linewidths=.5)
    
    title = f'{title_prefix} Mean {metric} by Method and {breakdown_by.replace("_", " ").title()} (deg)'
    plt.title(title)
    plt.xlabel("Method")
    plt.ylabel(breakdown_by.replace('_', ' ').title())
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{title_prefix.lower()}_{metric.lower()}_heatmap_by_{breakdown_by}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_error_distribution_shape(stats_df: pd.DataFrame, 
                                  title_prefix: str = '',
                                  save_dir: str = 'plots'):
    """
    Generates a scatter plot of Skew vs. Kurtosis to visualize 
    the shape of the error distribution for each method.

    Args:
        stats_df (pd.DataFrame): The summary statistics DataFrame.
        title_prefix (str): Text to prepend to the plot title.
        save_dir (str): Directory to save plots.
    """
    print(f"Generating Skew vs. Kurtosis scatter plot...")
    
    try:
        skew_df = stats_df.xs('Skew', level=1, axis=1)
        kurt_df = stats_df.xs('Kurtosis', level=1, axis=1)
    except KeyError:
        print("Error: 'Skew' or 'Kurtosis' not found in stats_df. Skipping shape plot.")
        return
        
    skew_df.columns.name = 'method'
    kurt_df.columns.name = 'method'

    skew_long = skew_df.stack().rename('Skew').reset_index()
    kurt_long = kurt_df.stack().rename('Kurtosis').reset_index()
    
    # Define primary keys for merging
    # This should be all index levels + 'method'
    merge_keys = skew_long.columns.tolist()
    merge_keys.remove('Skew')

    plot_df = pd.merge(skew_long, kurt_long[merge_keys + ['Kurtosis']], on=merge_keys)

    plt.figure(figsize=(12, 8))
    g = sns.scatterplot(data=plot_df, 
                        x='Skew', 
                        y='Kurtosis', 
                        hue='method', 
                        alpha=0.5, 
                        s=15)
    
    plt.axvline(0, c='k', ls='--', lw=1, zorder=0)
    plt.axhline(0, c='k', ls='--', lw=1, zorder=0)
    
    plt.title(f'{title_prefix} Error Distribution Shape (Skew vs. Kurtosis)')
    plt.xlabel("Skew (Symmetry)")
    plt.ylabel("Kurtosis (Peakedness)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{title_prefix.lower()}_skew_kurtosis_scatter.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_metric_correlation(stats_df: pd.DataFrame, 
                              title_prefix: str = '',
                              save_dir: str = 'plots'):
    """
    Generates a correlation heatmap of all calculated error metrics.

    Args:
        stats_df (pd.DataFrame): The summary statistics DataFrame.
        title_prefix (str): Text to prepend to the plot title.
        save_dir (str): Directory to save plots.
    """
    print(f"Generating metric correlation heatmap...")
    
    # Stack the 'method' level (level 0) to get a single DataFrame
    # with all metrics as columns
    try:
        stacked_df = stats_df.stack(level=0)
    except Exception as e:
        print(f"Error stacking DataFrame: {e}. Skipping correlation plot.")
        return

    corr_matrix = stacked_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap="vlag",  # Diverging colormap centered at 0
                center=0)
    
    plt.title(f'{title_prefix} Correlation Matrix of Error Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{title_prefix.lower()}_metric_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


# --- Main Execution ---

if __name__ == "__main__":
    
    # 1. Define the parameters
    SUBJECT_IDS: List[str] = ["Subject01", "Subject02", "Subject03", "Subject04", "Subject05", "Subject06",
                              "Subject07", "Subject08", "Subject09", "Subject10", "Subject11"]
    METHODS: List[str] = ['Marker', 
                          'Madgwick (Al Borno)', 'EKF', 'Mahony', 'Madgwick',
                          'Never Project', 'Mag Free', 'Cascade', 'Unprojected'] 
    joints = list(JOINT_SEGMENT_DICT.keys()) 
    drop_ankles = False

    # Create directories if they don't exist
    os.makedirs("plots", exist_ok=True)
    os.makedirs(os.path.join("data", "data"), exist_ok=True)

    data_file_path = os.path.abspath(os.path.join("data", "data", f"all_subject_data.pkl"))
    
    if os.path.exists(data_file_path):
        print(f"Loading existing DataFrame from {data_file_path}...")
        all_data_df = pd.read_pickle(data_file_path)
    else:
        # 2. Loop, load, and collect
        all_subject_dfs = []
        for trial_type in ['walking', 'complexTasks']:
            for subject_id in SUBJECT_IDS:
                print(f"--- Processing {subject_id} - {trial_type} ---")
                try:
                    subject_df = load_joint_traces_for_subject_df(
                        subject_id=subject_id,
                        methods=METHODS,
                        trial_type=trial_type,
                        euler_order='XYZ' # Using 'XYZ' as per your original call
                    )
                except Exception as e:
                    print(f"Error processing {subject_id}: {e}. Skipping subject.")
                    continue
                
                if not subject_df.empty:
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
    stats_file_path = os.path.abspath(os.path.join("data", "data", f"all_subject_statistics.pkl"))
    if os.path.exists(stats_file_path):
        print(f"Loading existing statistics DataFrame from {stats_file_path}...")
        all_stats_df = pd.read_pickle(stats_file_path)
    else:
        print("Calculating summary statistics by all groups...")
        # *** MODIFIED THIS LINE ***
        # Added 'trial_type' to the grouping for more detailed analysis
        all_stats_df = calculate_rmse_and_std(
            all_data_df,
            group_by=['joint_name', 'axis', 'subject_id', 'trial_type'] 
        )
        all_stats_df.to_pickle(stats_file_path)
        print(f"Statistics DataFrame saved to {stats_file_path}")

    # --- Filter Dataset as Needed (from your original code) ---
    if drop_ankles or METHODS != all_data_df.index.get_level_values('method').unique().tolist() or SUBJECT_IDS != all_data_df.index.get_level_values('subject_id').unique().tolist():
        print("Filtering dataset based on specified criteria...")
        if drop_ankles:
            print("Dropping ankle joints from the dataset...")
            idx_to_drop = all_data_df.index.get_level_values('joint_name').isin(['L_Ankle', 'R_Ankle'])
            all_data_df = all_data_df[~idx_to_drop]
        
        if METHODS != all_data_df.index.get_level_values('method').unique().tolist():
            print("Filtering methods to match the specified METHODS list...")
            idx_to_keep = all_data_df.index.get_level_values('method').isin(METHODS)
            all_data_df = all_data_df[idx_to_keep]

        if SUBJECT_IDS != all_data_df.index.get_level_values('subject_id').unique().tolist():
            print("Filtering subjects to match the specified SUBJECT_IDS list...")
            idx_to_keep = all_data_df.index.get_level_values('subject_id').isin(SUBJECT_IDS)
            all_data_df = all_data_df[idx_to_keep]
        print("Dataset filtering complete.")
        
        # Recalculate stats if we filtered the data
        print("Recalculating summary statistics on filtered data...")
        all_stats_df = calculate_rmse_and_std(
            all_data_df,
            group_by=['joint_name', 'axis', 'subject_id', 'trial_type'] 
        )


    # --- Generate Raw Error DataFrame (for statistical testing) ---
    print("Generating raw error DataFrame for statistical tests...")
    
    # 1. Stack Euler angle columns
    original_index_names = all_data_df.index.names
    data_stacked = all_data_df.stack().rename_axis(original_index_names + ['axis'])
    
    # 2. Unstack by 'method'
    data_wide = data_stacked.unstack('method')

    # 3. Calculate error (all methods - 'Marker') and convert to degrees
    marker_series = data_wide['Marker']
    all_errors_deg_df = data_wide.sub(marker_series, axis=0)
    all_errors_deg_df = all_errors_deg_df.drop('Marker', axis=1, errors='ignore')
    all_errors_deg_df = all_errors_deg_df * 180 / np.pi  # Convert radians to degrees
    
    print("Raw error DataFrame complete.")

    # --- Run Visualizations ---
    print("\n--- Running Visualizations ---")
    
    # Plot 1: Overall Method Performance (Box Plots)
    plot_overall_metric_distribution(all_stats_df, 
                                     metric='RMSE', 
                                     title_prefix='Overall')
    plot_overall_metric_distribution(all_stats_df, 
                                     metric='Bias', 
                                     title_prefix='Overall')

    # Plot 2: Performance Breakdown (Heatmaps)
    plot_metric_heatmap(all_stats_df, 
                        metric='RMSE', 
                        breakdown_by='joint_name', 
                        title_prefix='RMSE')
    
    plot_metric_heatmap(all_stats_df, 
                        metric='RMSE', 
                        breakdown_by='trial_type', 
                        title_prefix='RMSE')

    # Plot 3: Error Distribution Analysis (Scatter Plot)
    plot_error_distribution_shape(all_stats_df, 
                                  title_prefix='Overall')

    # Plot 4: Metric Relationship Analysis (Correlation Heatmap)
    plot_metric_correlation(all_stats_df, 
                            title_prefix='Overall')

    # Original Breakdown Plots (still useful!)
    print("Generating original breakdown plots...")
    plot_metric_distribution(all_stats_df, 
                             metric='RMSE', 
                             group_by='joint_name', 
                             title_prefix='RMSE')
    
    plot_metric_distribution(all_stats_df, 
                             metric='Bias', 
                             group_by='joint_name', 
                             title_prefix='Bias')

    plot_metric_distribution(all_stats_df, 
                             metric='RMSE', 
                             group_by='trial_type', 
                             title_prefix='RMSE')

    # --- Run Statistical Analysis ---
    
    # Analyze by joint
    stats_by_joint = perform_statistical_analysis(all_errors_deg_df, 
                                                group_by='joint_name')
    print("\n\n--- STATISTICAL ANALYSIS (BY JOINT) ---")
    print(json.dumps(stats_by_joint, indent=2))
    
    # Analyze by trial_type
    stats_by_trial = perform_statistical_analysis(all_errors_deg_df, 
                                                group_by='trial_type')
    print("\n\n--- STATISTICAL ANALYSIS (BY TRIAL TYPE) ---")
    print(json.dumps(stats_by_trial, indent=2))
    
    # Analyze overall
    stats_overall = perform_statistical_analysis(all_errors_deg_df, 
                                               group_by=None)
    print("\n\n--- STATISTICAL ANALYSIS (OVERALL) ---")
    print(json.dumps(stats_overall, indent=2))

    print("\n\n--- Analysis Complete ---")
    print(f"Plots saved to '{os.path.abspath('plots')}' directory.")
    print(f"Statistics results printed above.")