import os
from typing import Dict, List, Optional, Union, Any, Literal

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
METHODS_TO_PLOT = ['EKF', 'Mag On', 'Mag Off', 'Mag Adapt']

# Which summary metrics to plot from: RMSE, MAE, Mean, STD, Kurtosis, Skewness, Pearson, Median, Q25, Q75, MAD.
# Default is in radians, for degrees use '_deg' suffix, e.g., 'RMSE_deg'.
METRICS_TO_PLOT = ['RMSE_deg'] 

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

# Plot style can be 'strip' (strip + box-whisker for median + iqr) or 'bar' (mean + std error bars)
PLOT_STYLE = 'strip'


DATA_FILE_PATH = os.path.join("data", "data", "all_subject_statistics.pkl")
PLOTS_DIRECTORY = "plots"

SHOW_PLOTS = True
SAVE_PLOTS = True

PALETTE = "Set2"  # Can be a seaborn palette name or a dict mapping method names to colors

# --- Helper Functions ---

def _run_statistical_analysis(
    df: pd.DataFrame,
    metric: str,
    methods_order: List[str],
    alpha: float = 0.05
) -> List[tuple[str, str]]:
    """
    Performs statistical analysis using Friedman and Wilcoxon post-hoc tests.

    Args:
        ... (same as before) ...

    Returns:
        List[tuple[str, str]]: A list of (method1, method2) tuples
                               for pairs that are significantly different.
    """
    print(f"--- a. Dependent-Sample Test (Friedman Test for '{metric}') ---")
    significant_pairs_list: List[tuple[str, str]] = []

    try:
        # Define the columns that create a unique "block" for repeated measures.
        block_cols = [
            col for col in ['trial_type', 'joint_name', 'subject', 'axis']
            if col in df.columns
        ]
        if not block_cols:
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
            print(f"  > Skipping stats: Insufficient data (N_blocks={pivot_df_clean.shape[0]}, N_methods={len(valid_methods)}).")
            return significant_pairs_list

        print(f"  > Using {pivot_df_clean.shape[0]} complete blocks for {len(valid_methods)} methods.")
        groups_for_test = [pivot_df_clean[col] for col in valid_methods]

        # --- b. Run Friedman Test ---
        _, p_friedman = stats.friedmanchisquare(*groups_for_test)
        print(f"  Friedman Test (overall comparison): p-value = {p_friedman:.4e}")

        # --- c. Run Post-hoc Test if overall difference is significant ---
        if p_friedman >= alpha:
            print("  > No significant overall difference found between methods (p >= alpha).")
        else:
            print("  > Significant difference detected. Running post-hoc (Wilcoxon Signed-Rank)...")
            p_uncorrected_list = []
            method_pairs = []

            for i in range(len(valid_methods)):
                for j in range(i + 1, len(valid_methods)):
                    method1, method2 = valid_methods[i], valid_methods[j]
                    method_pairs.append((method1, method2))
                    try:
                        _, p_val = stats.wilcoxon(pivot_df_clean[method1], pivot_df_clean[method2])
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

            print("  Significant pairs (p_adj < alpha):")
            significant_pairs_count = 0
            for (method1, method2), p_adj in zip(method_pairs, p_adjusted):
                if p_adj < alpha:
                    print(f"    - {method1} vs. {method2}: p_adj = {p_adj:.4e}")
                    significant_pairs_list.append((method1, method2)) # <-- Add to list
                    significant_pairs_count += 1

            if significant_pairs_count == 0:
                print("    - None (after p-value correction).")

    except (ValueError, AttributeError) as e:
        print(f"  > An error occurred during statistical testing: {e}")
    
    return significant_pairs_list # <-- RETURN THE LIST

def _finalize_and_save_plot(
    figure: plt.Figure,
    plot_title: str,
    filename: str,
) -> None:
    """
    Applies final touches to a plot and saves it to disk.
    """
    figure.suptitle(plot_title, fontsize=16, y=1.02)
    plt.tight_layout()

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
            ax.text(i + whisker_width/2 + x_offset, median, f'{median:.1f}', ha='center', va='bottom', fontsize=12, color='black', fontweight='bold', zorder=11)

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
    """
    # 1. Plot the bar plot with CI error bars
    sns.barplot(
        data=data, x=x_col, y=y_col, order=order, palette=PALETTE,
        legend=False, ax=ax, errorbar=('ci', 95), capsize=0.1, zorder=2
    )
    
    # 2. Calculate mean and CI for label/bracket positioning
    # Calculate stats needed for labels and bracket positioning
    ci_stats = data.groupby(x_col)[y_col].agg(['mean', lambda x: x.std() / np.sqrt(len(x)) * stats.t.ppf(1 - 0.05/2, len(x) - 1)]).reindex(order)
    ci_stats.columns = ['mean', 'ci_half']
    ci_stats['ci_upper'] = ci_stats['mean'] + ci_stats['ci_half']
    
    # 3. Add text labels for the mean
    if show_labels:
        for i, method in enumerate(order):
            if method not in ci_stats.index or pd.isna(ci_stats.loc[method, 'mean']):
                continue
            
            mean_val = ci_stats.loc[method, 'mean']
            ci_upper = ci_stats.loc[method, 'ci_upper']
            y_pos = ci_upper * 1.05 # Position text above the CI bar
            
            ax.text(
                i, y_pos, f'{mean_val:.1f}',
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


# --- Unified Plotting Function ---

def plot_summary_data(
    summary_df: pd.DataFrame,
    group_cols: List[str],
    plot_type: Literal['bar', 'strip'],
    metric: str,
    method_order: List[str],
    show_labels: bool = True,
) -> None:
    """
    Generates a statistical plot (bar or strip/whisker) faceted by the given
    grouping columns.
    ... (args unchanged) ...
    """
    if plot_type not in ['bar', 'strip']:
        print(f"Error: Unknown plot_type '{plot_type}'. Skipping plot for {metric}.")
        return

    data = summary_df.reset_index()

    if 'method' not in data.columns or metric not in data.columns:
        print("Error: DataFrame must contain 'method' and the specified metric.")
        return

    print(f"\n--- Generating {plot_type.title()} Plot for {metric} ---")
    
    # Filter data to only include specified methods
    data = data[data['method'].isin(method_order)]
    plot_order = [m for m in method_order if m in data['method'].unique()]
    
    # Create a modifiable list of grouping columns for this plot
    plot_group_cols = group_cols.copy()

    # 1. Differentiate AngleAxis vs. Magnitude and handle axis-pooling
    if 'axis' in data.columns and len(data['axis'].unique()) > 1:
        data_type = 'AngleAxis'
        
        if 'axis' in plot_group_cols:
            print(f"  > 'axis' requested. Faceting by original 'axis' column (X, Y, Z, MAG).")
        
        else:
            print(f"  > 'axis' not requested. Defaulting to faceting by 'axis_group' (AngleAxis_Pooled vs. MAG).")
            data['axis_group'] = data['axis'].apply(lambda x: 'MAG' if x == 'MAG' else 'AngleAxis_Pooled')
            
            if 'axis_group' not in plot_group_cols:
                plot_group_cols.append('axis_group')
                
            data = data.drop(columns='axis', errors='ignore')
    
    else:
        data_type = 'Magnitude'
        print("  > Magnitude data detected (or 'axis' column missing/single-value).")
        if 'axis' in plot_group_cols:
             plot_group_cols.remove('axis')
        data = data.drop(columns='axis', errors='ignore')
    
    print(f"  > Plotting {data_type} data.")

    # Handle Outliers
    original_count = len(data)
    
    # --- START OF FIX ---
    # Create a list of columns to group by for outlier removal.
    # This MUST include 'method' plus any faceting columns.
    outlier_group_cols = plot_group_cols.copy()
    if 'method' not in outlier_group_cols:
        outlier_group_cols.append('method')
    
    # Group by the faceting columns AND method to remove outliers
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
        # No facets (overall plot)
        stats_results['_overall_'] = _run_statistical_analysis(filtered_data, metric, plot_order, alpha=0.05)
        facet_levels = ['_overall_']
    else:
        # Faceted plot: calculate stats for each combination of facet levels
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
            sharey=False 
        )
        
        def _facet_plot_helper(data, **kwargs):
            ax = plt.gca()
            if data.empty:
                return
            current_facet_key = data[facet_col_str].iloc[0] 
            sig_pairs = stats_results.get(current_facet_key, [])
            
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

        g.map_dataframe(_facet_plot_helper)
        
        g.set_axis_labels(x_var="", y_var=f"{metric.replace('_deg', '')} (Degrees)" if "_deg" in metric else metric)
        g.set_xticklabels(rotation=45, ha='right')
        g.set_titles(f"{{col_name}}") 
        g.fig.subplots_adjust(wspace=0.1, hspace=1.0)
        fig = g.fig
        
        clean_facet_title = facet_col_str.replace('_', ' ').title()
        plot_title = f"Performance for {metric} by {clean_facet_title} ({plot_type})"
        filename = f"unified_by_{facet_col_str.lower()}_{metric.lower()}_{plot_type}.png"
        
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
        
        ax.set_ylabel(f"{metric.replace('_deg', '')}\n(Degrees)" if "_deg" in metric else f"{metric}\n(Radians)")
        ax.set_xlabel("", fontsize=12) 
        ax.tick_params(axis='x', rotation=25, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        plot_title = f"Overall Method Performance{single_facet_name}\n{metric.replace('_deg', '').title()} {plot_detail}"
        filename = f"unified_overall{single_facet_name.replace(' ', '_').lower()}_{metric.lower()}_{plot_type}.png"
    
    _finalize_and_save_plot(fig, plot_title, filename)
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
        group_cols=[],  # You can add more grouping columns as needed
        plot_type=PLOT_STYLE,  # 'bar' or 'strip'
        metric=METRICS_TO_PLOT[0],  # e.g., 'rmse_deg'
        method_order=METHODS_TO_PLOT,
        show_labels=True,
    )

    plot_summary_data(
        summary_df=summary_stats_df,
        group_cols=['axis'],  # You can add more grouping columns as needed
        plot_type=PLOT_STYLE,  # 'bar' or 'strip'
        metric=METRICS_TO_PLOT[0],  # e.g., 'rmse_deg'
        method_order=METHODS_TO_PLOT,
        show_labels=True,
    )

    # plot_summary_data(
    #     summary_df=summary_stats_df,
    #     group_cols=['joint_name'],  # You can add more grouping columns as needed
    #     plot_type=PLOT_STYLE,  # 'bar' or 'strip'
    #     metric=METRICS_TO_PLOT[0],  # e.g., 'rmse_deg'
    #     method_order=METHODS_TO_PLOT,
    #     show_labels=True,
    # )

    # plot_summary_data(
    #     summary_df=summary_stats_df,
    #     group_cols=['subject'],  # You can add more grouping columns as needed
    #     plot_type=PLOT_STYLE,  # 'bar' or 'strip'
    #     metric=METRICS_TO_PLOT[0],  # e.g., 'rmse_deg'
    #     method_order=METHODS_TO_PLOT,
    #     show_labels=True,
    # )

    print("\n--- All plotting complete ---")

if __name__ == "__main__":
    main()