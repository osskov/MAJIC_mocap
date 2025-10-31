import os
from typing import List, Any, Literal
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import numpy as np

# --- Global Configuration ---
# Subjects to plot can be "Subject01", "Subject02", ..., "Subject11"
SUBJECTS_TO_PLOT = ['Subject01', 'Subject02', 'Subject03', 'Subject04', 'Subject05',
                    'Subject06', 'Subject07', 'Subject08', 'Subject09', 'Subject10', 'Subject11']

# Methods to plot can be 'EKF' (Global EKF), 'Madgwick (Al Borno)' (Loaded from Al Borno files), 'Madgwick' (GLobal Madgwick), Mahony (Global Mahony),
# 'Mag On' (Magnetometer always used), 'Mag Off' (Magnetometer never integrated), 'Unprojected' (No acceleration projection), 'Mag Adapt' (Magnetometer used adaptively).
METHODS_TO_PLOT = ['EKF', 'Mag Off', 'Mag On', 'Mag Adapt']

# Which summary metrics to plot from: RMSE, MAE, Mean, STD, Kurtosis, Skewness, Pearson, Median, Q25, Q75, MAD.
# Default is in radians, for degrees use '_deg' suffix, e.g., 'RMSE_deg'.
METRIC_TO_PLOT = 'RMSE_deg' 

# If you do not want data to be combined across left/right sides, you can specify them individually as R_{joint} and L_{joint}
# in both RENAME_JOINTS.
RENAME_JOINTS = {
    'R_Hip': 'Hip',
    'L_Hip': 'Hip',
    'R_Knee': 'Knee',   
    'L_Knee': 'Knee',
    'R_Ankle': 'Ankle',
    'L_Ankle': 'Ankle',
}
JOINT_PLOT_ORDER = ['Lumbar', 'Hip', 'Knee', 'Ankle']

# Plot style can be 'strip' (strip + box-whisker for median + iqr), 'bar' (mean + std error bars), or 'box' (standard boxplot + strip)
PLOT_STYLE = 'bar'


DATA_FILE_PATH = os.path.join("data", "all_subject_statistics.pkl")
PLOTS_DIRECTORY = "plots"

SHOW_PLOTS = True
SAVE_PLOTS = True

PALETTE = "Set2"  # Can be a seaborn palette name or a dict mapping method names to colors

# --- Helper Functions ---

def _run_statistical_analysis(
    df: pd.DataFrame,
    metric: str,
    methods_order: List[str],
    alpha: float = 0.05,
    verbose: bool = True  # <-- MODIFICATION: Added verbose flag
) -> List[tuple[str, str]]:
    """
    Performs statistical analysis using Friedman and Wilcoxon post-hoc tests.
    ...
    """
    if verbose:
        print(f"--- a. Dependent-Sample Test (Friedman Test for '{metric}') ---")
    significant_pairs_list: List[tuple[str, str]] = []

    try:
        # Define the columns that create a unique "block" for repeated measures.
        block_cols = [
            col for col in ['trial_type', 'joint_name', 'subject', 'axis']
            if col in df.columns
        ]
        if not block_cols:
            if verbose:
                print("  > Error: Could not find block columns for Friedman test.")
            return significant_pairs_list

        # Create a unique block_id for each combination of conditions.
        df = df.copy()
        df['block_id'] = df[block_cols].apply(
            lambda row: '_'.join(row.values.astype(str)), axis=1
        )

        pivot_df = df.pivot_table(
            index='block_id', columns='method', values=metric
        )
        pivot_df_clean = pivot_df.dropna()
        valid_methods = [m for m in methods_order if m in pivot_df_clean.columns]

        if pivot_df_clean.shape[0] < 2 or len(valid_methods) < 2:
            if verbose:
                print(f"  > Skipping stats: Insufficient data (N_blocks={pivot_df_clean.shape[0]}, N_methods={len(valid_methods)}).")
            return significant_pairs_list

        if verbose:
            print(f"  > Using {pivot_df_clean.shape[0]} complete blocks for {len(valid_methods)} methods.")
        groups_for_test = [pivot_df_clean[col] for col in valid_methods]

        # --- b. Run Friedman Test ---
        _, p_friedman = stats.friedmanchisquare(*groups_for_test)
        if verbose:
            print(f"  Friedman Test (overall comparison): p-value = {p_friedman:.4e}")

        # --- c. Run Post-hoc Test if overall difference is significant ---
        if p_friedman >= alpha:
            if verbose:
                print("  > No significant overall difference found between methods (p >= alpha).")
        else:
            if verbose:
                print("  > Significant difference detected. Running post-hoc (Wilcoxon Signed-Rank)...")
            p_uncorrected_list = []
            method_pairs = []

            for i in range(len(valid_methods)):
                for j in range(i + 1, len(valid_methods)):
                    method1, method2 = valid_methods[i], valid_methods[j]
                    method_pairs.append((method1, method2))
                    try:
                        _, p_val = stats.wilcoxon(
                                                    pivot_df_clean[method1], 
                                                    pivot_df_clean[method2],
                                                    alternative='two-sided', # Explicitly state you're checking for *any* difference
                                                    zero_method='zsplit'   # A robust method for handling zero-differences
                                                )
                        p_uncorrected_list.append(p_val)
                    except ValueError:
                        p_uncorrected_list.append(1.0)

            # Apply Holm-Bonferroni correction
            if p_uncorrected_list:
                sort_indices = np.argsort(p_uncorrected_list)
                p_sorted = np.array(p_uncorrected_list)[sort_indices]
                num_comparisons = len(p_sorted)
                multipliers = np.arange(num_comparisons, 0, -1)
                p_adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(p_sorted * multipliers))
                p_adjusted = np.empty_like(p_adjusted_sorted)
                p_adjusted[sort_indices] = p_adjusted_sorted
            else:
                p_adjusted = []
                
            if verbose:
                print("  Significant pairs (p_adj < alpha):")
            significant_pairs_count = 0
            for (method1, method2), p_adj in zip(method_pairs, p_adjusted):
                if p_adj < alpha:
                    if verbose:
                        print(f"    - {method1} vs. {method2}: p_adj = {p_adj:.4e}")
                    significant_pairs_list.append((method1, method2)) # <-- Add to list
                    significant_pairs_count += 1

            if significant_pairs_count == 0 and verbose:
                print("    - None (after p-value correction).")

    except (ValueError, AttributeError) as e:
        if verbose:
            print(f"  > An error occurred during statistical testing: {e}")
    
    return significant_pairs_list # <-- RETURN THE LIST

def _finalize_and_save_plot(
    figure: plt.Figure,
    plot_title: str,
    filename: str,
    epilog: str = None  # <-- MODIFICATION: Added epilog
) -> None:
    """
    Applies final touches to a plot and saves it to disk.
    """
    figure.suptitle(plot_title, fontsize=16, y=1.02, fontweight='bold')
    
    # --- START MODIFICATION ---
    # Add an epilog text to the bottom-right of the figure
    if epilog:
        figure.text(
            0.99, 0.01, epilog, 
            ha='right', va='bottom', fontsize=10, 
            fontstyle='italic', transform=figure.transFigure
        )
    # --- END MODIFICATION ---
    
    plt.tight_layout()

    # Adjust layout to make space for title and epilog if they exist
    top_margin = 0.97 if plot_title else 1.0
    bottom_margin = 0.03 if epilog else 0.0
    if epilog or plot_title:
        plt.tight_layout(rect=[0, bottom_margin, 1, top_margin])


    if SAVE_PLOTS:
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        print(f"Saved plot to {save_path}")

    if SHOW_PLOTS:
        plt.show()

    plt.close(figure)

def _remove_outliers(
    df: pd.DataFrame,
    metric: str,
    group_cols: List[str]
) -> pd.DataFrame:
    """
    Removes outliers from a DataFrame based on the 1.5 IQR rule, applied per group.
    """
    if not group_cols:
        q1 = df[metric].quantile(0.25)
        q3 = df[metric].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return df[(df[metric] >= lower_bound) & (df[metric] <= upper_bound)]
    
    grouped = df.groupby(group_cols)[metric]
    q1 = grouped.transform('quantile', 0.25)
    q3 = grouped.transform('quantile', 0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_df = df[(df[metric] >= lower_bound) & (df[metric] <= upper_bound)]
    
    return filtered_df

# --- Main Plotting Functions ---

def _add_strip_whisker_and_stats(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    order: List[str],
    significant_pairs: List[tuple[str, str]], # Simplified: passed directly
    ax: plt.Axes,
    show_labels: bool = True,
    **kwargs: Any
) -> None:
    """
    Draws a stripplot, custom whisker lines (median/quartiles), and significance
    brackets onto a given Axes. Consolidated logic from previous two whisker/bar functions.
    
    MODIFIED: Also prints a table of median and quartiles.
    """
    jitter = kwargs.pop('jitter', 0.1)
    color_map = {method: color for method, color in zip(order, sns.color_palette(PALETTE, n_colors=len(order)))}

    # 1. Plot the strip plot
    sns.stripplot(
        data=data, x=x_col, y=y_col, order=order, palette=PALETTE,
        legend=False, ax=ax, alpha=0.5, jitter=jitter, zorder=1
    )
    
    # 2. Add whisker lines (Median/Quartiles)
    stats_df = data.groupby(x_col)[y_col].quantile([0.25, 0.5, 0.75]).unstack().reindex(order)

    # --- START OF MODIFICATION ---
    # Print the stats table
    stats_df_print = stats_df.copy()
    stats_df_print.columns = ['Q1 (25%)', 'Median (50%)', 'Q3 (75%)']
    print(f"\n--- Stats for {y_col} (Median/Quartiles) ---")
    print(stats_df_print.to_string(float_format="%.3f"))
    print("--------------------------------------------------\n")
    # --- END OF MODIFICATION ---
    
    if show_labels:
        line_width = kwargs.get('line_width', 0.8)
        whisker_width = kwargs.get('whisker_width', 0.7)
        for i, method in enumerate(order):
            if method not in stats_df.index or stats_df.loc[method].isnull().any():
                continue
            q1, median, q3 = stats_df.loc[method, 0.25], stats_df.loc[method, 0.5], stats_df.loc[method, 0.75]
            method_color = color_map.get(method, 'gray')
            darker_color = sns.set_hls_values(method_color, l=0.4)
            x_offset = 0.1
            
            ax.hlines(q1, i - whisker_width/2, i + whisker_width/2, color=darker_color, linestyle='--', linewidth=1.5, zorder=10)
            ax.hlines(q3, i - whisker_width/2, i + whisker_width/2, color=darker_color, linestyle='--', linewidth=1.5, zorder=10)
            ax.hlines(median, i - whisker_width/2, i + line_width/2 + x_offset, color=darker_color, linestyle='-', linewidth=2, zorder=10)
            ax.text(i + whisker_width/2 + x_offset, median, f'{median:.2f}', ha='center', va='bottom', fontsize=12, color='black', fontweight='bold', zorder=11)

    # 3. Add Significance Brackets (Logic from the original _add_strip_and_whisker_elements)
    if significant_pairs:
        # Get max y-value from the data or quartile to position the brackets
        max_val = stats_df[0.75].max() if not stats_df[0.75].empty and pd.notna(stats_df[0.75].max()) else data[y_col].max()
        max_val = max_val if pd.notna(max_val) else data[y_col].mean() + data[y_col].std() # Failsafe
        
        # Determine the step size for brackets dynamically
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_step = y_range * 0.05 
        current_y_level = max_val + y_step 

        sorted_pairs = sorted(significant_pairs, key=lambda p: abs(order.index(p[0]) - order.index(p[1])))
        
        for method1, method2 in sorted_pairs:
            if method1 not in order or method2 not in order: continue
            x1, x2 = order.index(method1), order.index(method2)
            bar_y, text_y = current_y_level, current_y_level + (y_step * 0.1)
            
            ax.plot([x1, x1, x2, x2], [bar_y, bar_y + y_step, bar_y + y_step, bar_y], lw=1.5, c='black')
            ax.text((x1 + x2) * 0.5, bar_y + y_step, '*', ha='center', va='bottom', fontsize=20, fontweight='bold', c='black')
            current_y_level += 2 * y_step 
        
        ax.set_ylim(ax.get_ylim()[0], current_y_level + y_step)


def _add_bar_ci_and_stats(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    order: List[str],
    significant_pairs: List[tuple[str, str]], # Simplified: passed directly
    ax: plt.Axes,
    show_labels: bool = True,
    **kwargs: Any
) -> None:
    """
    Draws a barplot with mean and CI error bars, plus significance brackets.
    
    MODIFIED: Also prints a table of mean and 95% CI.
    """
    # 1. Plot the bar plot with CI error bars
    sns.barplot(
        data=data, x=x_col, y=y_col, order=order, palette=PALETTE,
        legend=False, ax=ax, errorbar=('ci', 95), capsize=0.1, zorder=2
    )
    
    # 2. Calculate mean and CI for label/bracket positioning
    # Calculate stats needed for labels and bracket positioning
    ci_stats = data.groupby(x_col)[y_col].agg(
        ['mean', lambda x: x.std() / np.sqrt(len(x)) * stats.t.ppf(1 - 0.05/2, len(x) - 1)]
    ).reindex(order)
    ci_stats.columns = ['mean', 'ci_half']
    ci_stats['ci_lower'] = ci_stats['mean'] - ci_stats['ci_half']
    ci_stats['ci_upper'] = ci_stats['mean'] + ci_stats['ci_half']
    
    # --- START OF MODIFICATION ---
    # Print the stats table
    # Select and reorder columns for a clean print
    stats_df_print = ci_stats[['mean', 'ci_lower', 'ci_upper']].copy()
    stats_df_print.columns = ['Mean', 'CI (2.5%)', 'CI (97.5%)']
    print(f"\n--- Stats for {y_col} (Mean + 95% CI) ---")
    print(stats_df_print.to_string(float_format="%.3f"))
    print("------------------------------------------------\n")
    # --- END OF MODIFICATION ---

    # 3. Add text labels for the mean
    if show_labels:
        for i, method in enumerate(order):
            if method not in ci_stats.index or pd.isna(ci_stats.loc[method, 'mean']):
                continue
            
            mean_val = ci_stats.loc[method, 'mean']
            ci_upper = ci_stats.loc[method, 'ci_upper']
            y_pos = ci_upper * 1.05 # Position text above the CI bar
            
            ax.text(
                i, y_pos, f'{mean_val:.2f}',
                ha='center', va='bottom', fontsize=12,
                color='black', fontweight='bold', zorder=11
            )
        
    # 4. Add Significance Brackets (Similar logic to strip/whisker)
    if significant_pairs:
        # Get max y-value from the CI upper bound
        max_val = ci_stats['ci_upper'].max() if not ci_stats['ci_upper'].empty and pd.notna(ci_stats['ci_upper'].max()) else data[y_col].max()
        max_val = max_val if pd.notna(max_val) else data[y_col].mean() + data[y_col].std() # Failsafe
        
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_step = y_range * 0.05 
        current_y_level = max_val + y_step 

        sorted_pairs = sorted(significant_pairs, key=lambda p: abs(order.index(p[0]) - order.index(p[1])))
        
        for method1, method2 in sorted_pairs:
            if method1 not in order or method2 not in order: continue
            x1, x2 = order.index(method1), order.index(method2)
            bar_y = current_y_level
            
            ax.plot([x1, x1, x2, x2], [bar_y, bar_y + y_step, bar_y + y_step, bar_y], lw=1.5, c='black')
            ax.text((x1 + x2) * 0.5, bar_y + y_step, '*', ha='center', va='bottom', fontsize=20, fontweight='bold', c='black')
            current_y_level += 2 * y_step 
        
        ax.set_ylim(ax.get_ylim()[0], current_y_level + y_step)
    
    # Finalize style
    sns.despine(left=False, bottom=False, top=True, right=True, ax=ax)


def _add_boxplot_and_stats(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    order: List[str],
    significant_pairs: List[tuple[str, str]],
    ax: plt.Axes,
    show_labels: bool = True,
    **kwargs: Any
) -> None:
    """
    Draws a standard boxplot, an overlying stripplot, and significance brackets.
    
    MODIFIED: Also prints a table of median and quartiles.
    """
    # 1. Plot the box plot
    boxplot_kwargs = {
        'palette': PALETTE,
        'legend': False,
        'showfliers': False,  # We will show all points with a stripplot
        'zorder': 5,
        'width': 0.7
    }
    boxplot_kwargs.update(kwargs)
    
    sns.boxplot(
        data=data, x=x_col, y=y_col, order=order, ax=ax,
        **boxplot_kwargs
    )
    
    # 2. Add the strip plot on top
    sns.stripplot(
        data=data, x=x_col, y=y_col, order=order, palette=PALETTE,
        legend=False, ax=ax, alpha=0.2, jitter=0.15, zorder=3 
    )

    # --- START OF MODIFICATION ---
    # 2a. Calculate and Print Stats
    # This calculation is now done here to be printed and re-used by show_labels
    stats_df_quantiles = data.groupby(x_col)[y_col].quantile([0.25, 0.5, 0.75]).unstack().reindex(order)
    
    # Format and print the table
    stats_df_print = stats_df_quantiles.copy()
    stats_df_print.columns = ['Q1 (25%)', 'Median (50%)', 'Q3 (75%)']
    print(f"\n--- Stats for {y_col} (Median/Quartiles) ---")
    print(stats_df_print.to_string(float_format="%.3f"))
    print("--------------------------------------------------\n")

    # 2b. Add median labels
    if show_labels:
        # Re-use the stats calculation from above
        stats_df = stats_df_quantiles[0.5] # Get just the median Series
        box_width = boxplot_kwargs.get('width', 0.7) 
        
        for i, method in enumerate(order):
            if method not in stats_df.index or pd.isna(stats_df.loc[method]):
                continue
            median = stats_df.loc[method]
            
            # Position text in the center of the box
            x_pos = i

            ax.text(
                x_pos, median + 0.1, f'{median:.2f}', 
                ha='center', va='bottom', fontsize=12, color='black', 
                fontweight='bold', zorder=11
            )
    # --- END OF MODIFICATION ---

    # 3. Add Significance Brackets (Logic from the other functions)
    if significant_pairs:
        # Find max y-value. The 1.5*IQR whisker is a good reference.
        # We need to find the max value *within* the 1.5 IQR range,
        # as this is where the whiskers are drawn.
        
        try:
            # Use the calculated quantiles
            q1_series = stats_df_quantiles[0.25]
            q3_series = stats_df_quantiles[0.75]
            
            # Map q1/q3 back to the original dataframe
            q1_map = data[x_col].map(q1_series)
            q3_map = data[x_col].map(q3_series)
            iqr = q3_map - q1_map
            upper_bound = q3_map + 1.5 * iqr
            
            # Find the max data point that is *at or below* the upper_bound
            data_within_whiskers = data[data[y_col] <= upper_bound]
            max_val = data_within_whiskers[y_col].max()
        except Exception:
             max_val = data[y_col].max() # Fallback

        max_val = max_val if pd.notna(max_val) else data[y_col].mean() + data[y_col].std() # Failsafe
        
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_step = y_range * 0.05 
        current_y_level = max_val + y_step * 2 # Add a bit more padding

        sorted_pairs = sorted(significant_pairs, key=lambda p: abs(order.index(p[0]) - order.index(p[1])))
        
        for method1, method2 in sorted_pairs:
            if method1 not in order or method2 not in order: continue
            x1, x2 = order.index(method1), order.index(method2)
            bar_y = current_y_level
            
            ax.plot([x1, x1, x2, x2], [bar_y, bar_y + y_step, bar_y + y_step, bar_y], lw=1.5, c='black')
            ax.text((x1 + x2) * 0.5, bar_y + y_step, '*', ha='center', va='bottom', fontsize=20, fontweight='bold', c='black')
            current_y_level += 2 * y_step 
        
        ax.set_ylim(ax.get_ylim()[0], current_y_level + y_step)
    
    # Finalize style
    sns.despine(left=False, bottom=False, top=True, right=True, ax=ax)

# --- Unified Plotting Function ---
def plot_summary_data(
    summary_df: pd.DataFrame,
    group_cols: List[str],
    plot_type: Literal['bar', 'strip', 'box'],
    metric: str,
    method_order: List[str],
    show_labels: bool = True,
) -> None:
    """
    DISPATCHER: Generates statistical plots, splitting for Axes vs. Magnitude.
    
    Checks if 'axis' column contains both 'MAG' and 'X'/'Y'/'Z'.
    If so, it splits the data and calls _generate_single_plot() twice:
    1. Once for 'MAG' data.
    2. Once for 'X', 'Y', 'Z' data.
    
    If no split is needed, it calls _generate_single_plot() once.
    """
    if plot_type not in ['bar', 'strip', 'box']:
        print(f"Error: Unknown plot_type '{plot_type}'. Skipping plot for {metric}.")
        return

    data = summary_df.reset_index()

    if 'method' not in data.columns or metric not in data.columns:
        print("Error: DataFrame must contain 'method' and the specified metric.")
        return
    
    # Apply categorical ordering if 'joint_name' is a grouping column
    if 'joint_name' in data.columns and 'joint_name' in group_cols:
        # Filter the global order to only joints present in the data
        present_joints = data['joint_name'].unique()
        
        # Start with the desired order
        ordered_categories = [j for j in JOINT_PLOT_ORDER if j in present_joints]
        
        # Add any other joints (not in JOINT_PLOT_ORDER) to the end
        for joint in present_joints:
            if joint not in ordered_categories:
                ordered_categories.append(joint)
                
        data['joint_name'] = pd.Categorical(
            data['joint_name'],
            categories=ordered_categories,
            ordered=True
        )
        print(f"  > Applied custom sort order to 'joint_name': {ordered_categories}")

    # Filter data to only include specified methods
    data = data[data['method'].isin(method_order)]
    
    # 1. Check if we need to split data into Axes and Magnitude
    if 'axis' in data.columns and len(data['axis'].unique()) > 1:
        unique_axes = set(data['axis'].unique())
        has_mag = 'MAG' in unique_axes
        has_axes = any(ax in unique_axes for ax in ['X', 'Y', 'Z'])

        if has_mag and has_axes:
            print(f"--- Splitting data into 'Axes (X,Y,Z)' and 'Magnitude' for {metric} ---")
            
            # 1. Plot Magnitude Data
            data_mag = data[data['axis'] == 'MAG'].copy()
            if not data_mag.empty:
                print("\n=== Generating MAGNITUDE Plot ===")
                _generate_single_plot(
                    data=data_mag,
                    group_cols=group_cols,
                    plot_type=plot_type,
                    metric=metric,
                    method_order=method_order,
                    show_labels=show_labels,
                    data_type_title="Magnitude"
                )
            else:
                print("\n=== No 'MAG' data found. Skipping Magnitude plot. ===")

            # 2. Plot Axes Data
            data_axes = data[data['axis'].isin(['X', 'Y', 'Z'])].copy()
            if not data_axes.empty:
                print("\n=== Generating AXES (X,Y,Z) Plot ===")
                _generate_single_plot(
                    data=data_axes,
                    group_cols=group_cols,
                    plot_type=plot_type,
                    metric=metric,
                    method_order=method_order,
                    show_labels=show_labels,
                    data_type_title="Axes"
                )
            else:
                print("\n=== No 'X,Y,Z' data found. Skipping Axes plot. ===")
                
            return # We are done after splitting

        elif has_mag:
            data_type_title = "Magnitude"
            print(f"--- Only Magnitude data found for {metric}. ---")
        elif has_axes:
            data_type_title = "Axes"
            print(f"--- Only Axes (X,Y,Z) data found for {metric}. ---")
        else:
            data_type_title = "UnknownAxis"
            print(f"--- 'axis' column has multiple values, but not X/Y/Z or MAG. Plotting as single group. ---")
            
    else:
        data_type_title = "Overall"
        print(f"--- No 'axis' column detected or single axis value. Plotting as single group. ---")

    # If no split was needed, run the plot once on the full (filtered) data
    _generate_single_plot(
        data=data,
        group_cols=group_cols,
        plot_type=plot_type,
        metric=metric,
        method_order=method_order,
        show_labels=show_labels,
        data_type_title=data_type_title
    )


def _generate_single_plot(
    data: pd.DataFrame,
    group_cols: List[str],
    plot_type: Literal['bar', 'strip', 'box'],
    metric: str,
    method_order: List[str],
    show_labels: bool,
    data_type_title: str # New arg: e.g., "Magnitude", "Axes", "Overall"
) -> None:
    """
    WORKER: Generates a single statistical plot for a given data subset.
    (This function contains the logic from the original plot_summary_data)
    """
    
    print(f"\n--- Generating {plot_type.title()} Plot for {metric} ({data_type_title}) ---")

    plot_order = [m for m in method_order if m in data['method'].unique()]
    
    # Create a modifiable list of grouping columns for this plot
    plot_group_cols = group_cols.copy()

    # 1. Handle axis faceting based on data_type_title
    if data_type_title == "Magnitude" or data_type_title == "Overall":
        # This data is magnitude-only or has no axis, remove 'axis' faceting.
        if 'axis' in plot_group_cols:
             plot_group_cols.remove('axis')
        if 'axis_group' in plot_group_cols:
             plot_group_cols.remove('axis_group')
        data = data.drop(columns=['axis', 'axis_group'], errors='ignore')
        print(f"  > Plotting {data_type_title} data.")

    elif data_type_title == "Axes":
        # This data is X, Y, Z. Check if user *wants* to facet by axis.
        if 'axis' in plot_group_cols:
            print(f"  > 'axis' requested. Faceting by original 'axis' column (X, Y, Z).")
        else:
            print(f"  > 'axis' not requested. Pooling X, Y, Z data.")
            # We drop the 'axis' column so it's not used in stats/faceting.
            if 'axis' in data.columns:
                data = data.drop(columns='axis', errors='ignore')
    
    # Handle Outliers
    original_count = len(data)
    
    # --- START OF FIX (from original code) ---
    outlier_group_cols = plot_group_cols.copy()
    if 'method' not in outlier_group_cols:
        outlier_group_cols.append('method')
    
    filtered_data = _remove_outliers(data, metric, group_cols=outlier_group_cols)
    
    removed_count = original_count - len(filtered_data)
    if removed_count > 0:
        print(f"  > Removed {removed_count} outliers (beyond 1.5 IQR per method/group) for '{metric}'.")
    # --- END OF FIX ---
    
    if filtered_data.empty:
        print("Warning: No data remaining after filtering. Skipping plot.")
        return

    # Determine the faceting columns based on the *final* plot_group_cols
    facet_cols = [col for col in plot_group_cols if col in filtered_data.columns]
    
    print(f"--- Starting Statistical Analysis for {metric} (Facets: {facet_cols or 'Overall'}) ---")

    # 2. Run Statistical Analysis for ALL facets
    stats_results = {}
    if not facet_cols:
        stats_results['_overall_'] = _run_statistical_analysis(filtered_data, metric, plot_order, alpha=0.05)
        facet_levels = ['_overall_']
    else:
        try:
            facet_levels_iter = filtered_data.groupby(facet_cols, dropna=True).groups.keys()
            facet_levels = list(facet_levels_iter)
        except Exception as e:
            print(f"  > Error during groupby for facets: {e}. Aborting plot.")
            return

        for level_tuple in facet_levels:
            key_tuple = level_tuple if isinstance(level_tuple, tuple) else (level_tuple,)
            level_key = '_'.join(map(str, key_tuple))
            
            level_data = filtered_data.copy()
            query_parts = []
            for col, level in zip(facet_cols, key_tuple):
                if pd.isna(level):
                    query_parts.append(f"`{col}`.isnull()")
                else:
                    query_parts.append(f"`{col}` == {repr(level)}")
            
            try:
                level_data = level_data.query(' & '.join(query_parts))
            except Exception as e:
                print(f"  > Error querying for facet {level_key}: {e}")
                continue
                
            print(f"\n--- Testing for Facet: {level_key} ---")
            stats_results[level_key] = _run_statistical_analysis(level_data, metric, plot_order, alpha=0.05)
            
    print(f"--- End of Statistical Analysis for {metric} ---")
    
    # 3. Create the Plot: FacetGrid or Single Ax
    
    if len(facet_levels) > 1:
        facet_col_str = '_'.join(facet_cols) 
        plot_data = filtered_data.copy()
        valid_facet_keys = set(stats_results.keys())
        
        def create_facet_key(row):
            key_parts = [row[col] for col in facet_cols]
            return '_'.join(map(str, key_parts))
            
        plot_data[facet_col_str] = plot_data.apply(create_facet_key, axis=1)
        plot_data = plot_data[plot_data[facet_col_str].isin(valid_facet_keys)]
        
        if plot_data.empty:
            print("Warning: No data left for plotting after filtering for valid facets. Skipping.")
            return

        facet_key_order = [
            '_'.join(map(str, k if isinstance(k, tuple) else (k,))) for k in facet_levels
        ]

        col_wrap = min(len(facet_levels), 7)
        g = sns.FacetGrid(
            plot_data,
            col=facet_col_str,
            col_order=facet_key_order,
            col_wrap=col_wrap,
            height=6,
            aspect=0.8,
            sharey=True
        )
        
        def _facet_plot_helper(data, **kwargs):
            ax = plt.gca()
            if data.empty:
                return
            current_facet_key = data[facet_col_str].iloc[0] 
            sig_pairs = stats_results.get(current_facet_key, []) if 'Pearson' not in metric else []
            
            if plot_type == 'strip':
                _add_strip_whisker_and_stats(
                    data=data, x_col='method', y_col=metric, order=plot_order,
                    significant_pairs=sig_pairs, ax=ax, show_labels=show_labels,
                )
            elif plot_type == 'bar':
                _add_bar_ci_and_stats(
                    data=data, x_col='method', y_col=metric, order=plot_order,
                    significant_pairs=sig_pairs, ax=ax, show_labels=show_labels,
                )
            elif plot_type == 'box':
                _add_boxplot_and_stats(
                    data=data, x_col='method', y_col=metric, order=plot_order,
                    significant_pairs=sig_pairs, ax=ax,
                    show_labels=show_labels, # <-- MODIFICATION: Pass show_labels
                    width=0.8 # Set a consistent width
                )

        g.map_dataframe(_facet_plot_helper)
        
        g.set_axis_labels(x_var="", y_var=f"{metric.replace('_deg', '')} (Degrees)" if "_deg" in metric else metric)
        g.set_xticklabels(rotation=45, ha='right', fontweight='black')
        # Modify the 'Mag Adapt' to 'MAJIC' in xtick labels if present
        for ax in g.axes.flatten():
            xtick_labels = [label.get_text().replace('Mag Adapt', 'MAJIC') for label in ax.get_xticklabels()]
            ax.set_xticklabels(xtick_labels)
        g.set_titles(f"{{col_name}}") 
        g.fig.subplots_adjust(wspace=0.1, hspace=1.0)
        fig = g.fig
        
        clean_facet_title = facet_col_str.replace('_', ' ').title()
        # MODIFIED: Use data_type_title in title and filename
        # Figure out if its a magnitude or axes plot for the title
        if "mag" in data_type_title.lower():
            data_type_title = "Joint Angle Magnitude "
        elif "axis" in data_type_title.lower():
            data_type_title = "Joint Angle Axes "
        else:
            data_type_title = "Joint Angle "
        if plot_type == 'strip' or plot_type == 'box':
            plot_type_full = "(Median + Quartiles)"
        else:
            plot_type_full = "(Mean + 95% CI)"
        plot_title = f"{data_type_title} Performance for {metric.upper()} by {clean_facet_title} {plot_type_full}"
        filename = f"by_{facet_col_str.lower()}_{metric.lower()}_{data_type_title.lower()}_{plot_type}.png"
        
    else:
        fig, ax = plt.subplots(figsize=(4, 7))
        
        single_facet_name = ""
        if facet_cols and len(facet_levels) == 1:
            single_facet_name = f" ({facet_levels[0]})"
            
        sig_pairs = stats_results.get('_overall_' if not facet_levels else '_'.join(map(str, facet_levels[0] if isinstance(facet_levels[0], tuple) else (facet_levels[0],))), [])
        
        if plot_type == 'strip':
            _add_strip_whisker_and_stats(
                data=filtered_data, x_col='method', y_col=metric, order=plot_order,
                significant_pairs=sig_pairs, ax=ax, show_labels=show_labels, jitter=0.27
            )
            plot_detail = "(Median/Quartiles)"
        elif plot_type == 'bar':
            _add_bar_ci_and_stats(
                data=filtered_data, x_col='method', y_col=metric, order=plot_order,
                significant_pairs=sig_pairs, ax=ax, show_labels=show_labels
            )
            plot_detail = "(Mean $\pm$ 95% CI)"
        elif plot_type == 'box':
            _add_boxplot_and_stats(
                data=filtered_data, x_col='method', y_col=metric, order=plot_order,
                significant_pairs=sig_pairs, ax=ax,
                show_labels=show_labels, # <-- MODIFICATION: Pass show_labels
                width=0.8 # Set a consistent width
            )
            plot_detail = "(Boxplot)"
        
        ax.set_ylabel(f"{metric.replace('_deg', '')}\n(Degrees)" if "_deg" in metric else f"{metric}\n(Radians)")
        ax.set_xlabel("", fontsize=12) 
        ax.tick_params(axis='x', rotation=25, labelsize=16)
        # Modify the 'Mag Adapt' to 'MAJIC' in xtick labels if present
        xtick_labels = [label.get_text().replace('Mag Adapt', 'MAJIC') for label in ax.get_xticklabels()]
        ax.set_xticklabels(xtick_labels)
        ax.tick_params(axis='y', labelsize=16)
        
        if "mag" in data_type_title.lower():
            data_type_title = "Joint Angle Magnitude "
        elif "axis" in data_type_title.lower():
            data_type_title = "Joint Angle Axes "
        else:
            data_type_title = "Joint Angle "
        if plot_type == 'strip' or plot_type == 'box':
            plot_type_full = "(Median + Quartiles)"
        else:
            plot_type_full = "(Mean + 95% CI)"

        # MODIFIED: Use data_type_title in title and filename
        plot_title = f"{data_type_title} \n{metric.replace('_deg', '')} {plot_type_full}"
        filename = f"overall{single_facet_name.replace(' ', '_').lower()}_{metric.lower()}_{data_type_title.lower()}_{plot_type}.png"
    
    _finalize_and_save_plot(fig, plot_title, filename)

def _generate_single_heatmap(
    data: pd.DataFrame,
    metric: str,
    method_order: List[str],
    joint_order: List[str],
    data_type_title: str,
    **heatmap_kwargs: Any
) -> None:
    """
    WORKER: Generates a single heatmap for a given data subset (e.g., "Axes" or "Magnitude").
    (MODIFIED: Puts joints on Y-axis, methods on X-axis, and adds significance)
    """
    print(f"--- Generating Heatmap for {metric} ({data_type_title}) ---")

    # Filter data to only included methods and joints
    data = data[data['method'].isin(method_order) & data['joint_name'].isin(joint_order)]

    if data.empty:
        print(f"  > No data found for {data_type_title}. Skipping heatmap.")
        return

    # Create the pivot table: mean metric for each joint/method
    try:
        pivot_data = data.groupby(['joint_name', 'method'])[metric].mean().unstack()
    except Exception as e:
        print(f"  > Error creating pivot table: {e}. Skipping heatmap.")
        return

    # Order the rows (joints) and columns (methods)
    valid_joints = [j for j in joint_order if j in pivot_data.index]
    valid_methods = [m for m in method_order if m in pivot_data.columns]
    
    if not valid_joints or not valid_methods:
        print(f"  > No valid data after pivoting. Skipping heatmap.")
        return

    pivot_data = pivot_data.reindex(index=valid_joints, columns=valid_methods)

    # --- START OF MODIFICATION (Run Stats & Create Annot Labels) ---

    # 4. Run Statistical Analysis for each Joint (Row)
    stats_results = {}
    print(f"--- Running stats for heatmap ({data_type_title}) ---")
    for joint in valid_joints:
        joint_data = data[data['joint_name'] == joint]
        sig_pairs = _run_statistical_analysis(
            joint_data, metric, valid_methods, alpha=0.05, verbose=False # Run silently
        )
        stats_results[joint] = sig_pairs
    print("--- Stats complete ---")

    # 5. Determine Annotation Labels with Significance
    is_error_metric = any(err in metric.lower() for err in ['rmse', 'mae', 'std', 'mad'])
    is_corr_metric = 'pearson' in metric.lower()
    
    # Set formatting
    fmt_str = '.2f'
    if is_corr_metric:
        fmt_str = '.3f'

    # Create base annotation labels (the numbers)
    annot_labels = pivot_data.applymap(lambda x: f"{x:{fmt_str}}" if pd.notna(x) else "")

    # Add asterisks
    for joint in annot_labels.index:
        joint_mean_values = pivot_data.loc[joint]
        if joint_mean_values.isnull().all():
            continue

        # Find the "best" method for this joint
        best_method = ""
        if is_error_metric:
            best_method = joint_mean_values.idxmin()
        elif is_corr_metric:
            best_method = joint_mean_values.idxmax()
        else:
            # Default to min for unknown metrics (like 'Mean')
            best_method = joint_mean_values.idxmin() 
        
        if pd.isna(best_method): continue # Skip if best method is NaN

        sig_pairs = stats_results.get(joint, [])

        for method in annot_labels.columns:
            if method == best_method or pd.isna(pivot_data.loc[joint, method]):
                continue
            
            # Check if (best, method) or (method, best) is in the significant list
            is_significant = False
            for p1, p2 in sig_pairs:
                if (p1 == best_method and p2 == method) or (p1 == method and p2 == best_method):
                    is_significant = True
                    break
            
            if is_significant:
                annot_labels.loc[joint, method] += "*"

    # --- END OF MODIFICATION ---

    # Set up plot
    fig_height = len(valid_joints) * 1.8
    fig_width = len(valid_methods) * 2.1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Set default heatmap arguments, allowing override
    cbar_label = f"Mean {metric.replace('_deg', '')} (Degrees)" if "_deg" in metric else f"Mean {metric}"
    default_kwargs = {
        'annot': annot_labels,  # <-- MODIFICATION: Use custom labels
        'annot_kws': {'size': 14, 'weight': 'bold'},
        'fmt': '',  # <-- MODIFICATION: Disable default formatting
        'cmap': 'Reds',
        'linewidths': 0.5, 
        'cbar_kws': {'label': cbar_label, 'shrink': 0.8}
    }
    
    # Use a diverging colormap for Pearson correlation
    if 'pearson' in metric.lower():
         default_kwargs['cmap'] = 'vlag' 
         default_kwargs['vmin'] = -1.0
         default_kwargs['vmax'] = 1.0

    default_kwargs.update(heatmap_kwargs)

    sns.heatmap(pivot_data, ax=ax, **default_kwargs)

    # --- (Axis labeling - same as your original) ---
    ax.set_ylabel("Joint", fontsize=16)
    ax.set_xlabel("Method", fontsize=16)
    
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    xtick_labels = [label.get_text().replace('Mag Adapt', 'MAJIC') for label in ax.get_xticklabels()]
    ax.set_xticklabels(xtick_labels)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.tick_params(axis='y', rotation=0, labelsize=14)
    # --- (End of axis labeling) ---

    # Finalize
    plot_title = f"Mean Joint Angle {metric.replace('_deg', '')}\nby Method and Joint"
    filename = f"heatmap_joint_vs_meth_{metric.lower()}_{data_type_title.lower()}.png"
    
    # --- MODIFICATION: Pass epilog to save function ---
    epilog_text = "* Significantly different from best method in row (p < 0.05, Wilcoxon)"
    _finalize_and_save_plot(fig, plot_title, filename, epilog=epilog_text)

def plot_metric_vs_joint_heatmap(
    summary_df: pd.DataFrame,
    metric: str,
    method_order: List[str],
    joint_order: List[str],
    **heatmap_kwargs: Any
) -> None:
    """
    DISPATCHER: Generates heatmaps, splitting for Axes vs. Magnitude.
    
    Checks if 'axis' column contains both 'MAG' and 'X'/'Y'/'Z'.
    If so, it splits the data and calls _generate_single_heatmap() twice.
    """
    data = summary_df.reset_index()

    if 'method' not in data.columns or 'joint_name' not in data.columns or metric not in data.columns:
        print("Error: DataFrame must contain 'method', 'joint_name', and the specified metric for heatmap.")
        return

    # 1. Check if we need to split data into Axes and Magnitude
    if 'axis' in data.columns and len(data['axis'].unique()) > 1:
        unique_axes = set(data['axis'].unique())
        has_mag = 'MAG' in unique_axes
        has_axes = any(ax in unique_axes for ax in ['X', 'Y', 'Z'])

        if has_mag and has_axes:
            print(f"--- Splitting data into 'Axes (X,Y,Z)' and 'Magnitude' for {metric} Heatmap ---")
            
            # Only Plot Magnitude Data
            data_mag = data[data['axis'] == 'MAG'].copy()
            _generate_single_heatmap(
                data=data_mag,
                metric=metric,
                method_order=method_order,
                joint_order=joint_order,
                data_type_title="Magnitude",
                **heatmap_kwargs
            )
            return # We are done after splitting

        elif has_mag:
            data_type_title = "Magnitude"
            data_to_plot = data[data['axis'] == 'MAG'].copy()
            print(f"--- Only Magnitude data found for {metric}. ---")
        elif has_axes:
            data_type_title = "Axes"
            data_to_plot = data[data['axis'].isin(['X', 'Y', 'Z'])].copy()
            print(f"--- Only Axes (X,Y,Z) data found for {metric}. ---")
        else:
            data_type_title = "Overall"
            data_to_plot = data.copy()
            print(f"--- 'axis' column has multiple values, but not X/Y/Z or MAG. Plotting as single group. ---")
    else:
        data_type_title = "Overall"
        data_to_plot = data.copy()
        print(f"--- No 'axis' column detected or single axis value. Plotting as single group. ---")

    # If no split was needed (or only one type was found), run the plot once
    _generate_single_heatmap(
        data=data_to_plot,
        metric=metric,
        method_order=method_order,
        joint_order=joint_order,
        data_type_title=data_type_title,
        **heatmap_kwargs
    )

# --- Main Execution ---

def main():
    """
    Main function to load data and generate all plots and analyses.
    """
    # --- Set Global Matplotlib Font Settings ---
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.weight': 'light',
        'axes.labelweight': 'black',
        'axes.titleweight': 'black',
        'axes.titlesize': 26,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
    })

    # --- 2. Load Data ---
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Statistics file not found at {DATA_FILE_PATH}")
        return
    else:
        print(f"Loading summary statistics from {DATA_FILE_PATH}...")
        try:
            summary_stats_df = pd.read_pickle(DATA_FILE_PATH)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return
    print("Data loaded successfully.")

    # Rename joints for clarity
    summary_stats_df = summary_stats_df.rename(index=RENAME_JOINTS, level='joint_name')

    # Drop subjects and methods not in the analysis
    summary_stats_df = summary_stats_df[summary_stats_df.index.get_level_values('method').isin(METHODS_TO_PLOT)]
    summary_stats_df = summary_stats_df[summary_stats_df.index.get_level_values('subject').isin(SUBJECTS_TO_PLOT)]

    print(summary_stats_df.head())
    
    plot_summary_data(
        summary_df=summary_stats_df,
        group_cols=['joint_name'],  # You can add more grouping columns as needed
        plot_type=PLOT_STYLE,  # 'bar', 'strip', or 'box'
        metric=METRIC_TO_PLOT,  # e.g., 'rmse_deg'
        method_order=METHODS_TO_PLOT,
        show_labels=True,
    )

    plot_metric_vs_joint_heatmap(
        summary_df=summary_stats_df,
        metric=METRIC_TO_PLOT,
        method_order=METHODS_TO_PLOT,
        joint_order=JOINT_PLOT_ORDER
    )

    # --- 2. Load Data ---
    if not os.path.exists(DATA_FILE_PATH.replace("statistics", "pearson_correlation")):
        print(f"Error: Statistics file not found at {DATA_FILE_PATH.replace("statistics", "pearson_correlation")}")
        return
    else:
        print(f"Loading summary statistics from {DATA_FILE_PATH.replace("statistics", "pearson_correlation")}...")
        try:
            pearson_stats_df = pd.read_pickle(DATA_FILE_PATH.replace("statistics", "pearson_correlation"))
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return
    print("Data loaded successfully.")

    # Rename joints for clarity
    pearson_stats_df = pearson_stats_df.rename(index=RENAME_JOINTS, level='joint_name')

    # Drop subjects and methods not in the analysis
    pearson_stats_df = pearson_stats_df[pearson_stats_df.index.get_level_values('method').isin(METHODS_TO_PLOT)]
    pearson_stats_df = pearson_stats_df[pearson_stats_df.index.get_level_values('subject').isin(SUBJECTS_TO_PLOT)]
    print(pearson_stats_df.head())

    plot_summary_data(
        summary_df=pearson_stats_df,
        group_cols=['joint_name'],  # You can add more grouping columns as needed
        plot_type=PLOT_STYLE,  # 'bar', 'strip', or 'box'
        metric='PearsonR',  # e.g., 'pearson'
        method_order=METHODS_TO_PLOT,
        show_labels=True,
    )
    print("\n--- All plotting complete ---")

if __name__ == "__main__":
    main()