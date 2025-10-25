# -*- coding: utf-8 -*-
"""
plot_paper_figures.py

This script is designed to generate publication-quality figures and perform
statistical analysis for comparing the performance of various estimation methods.
It loads a pre-processed pandas DataFrame containing performance metrics and
produproduces three main types of visualizations:

1.  **Overall Performance Plots (Whisker)**: Compares the distribution of key metrics
    (e.g., RMSE, MAE) across all methods using a combined strip and
    box-whisker plot (median, quartiles). Includes a robust statistical 
    analysis using the Friedman test, followed by a Wilcoxon signed-rank 
    post-hoc test.
    **Annotations for significant pairs are drawn on the plot.**

2.  **Overall Performance Plots (Bar)**: Compares the mean and standard 
    deviation of key metrics across all methods using a bar plot.

3.  **Performance by Factor Plots**: Creates faceted plots (either whisker
    or bar) to compare method performance side-by-side for each level of a
    given experimental factor (e.g., trial type, joint, axis).
    **Annotations for significant pairs are drawn on each facet for 
    whisker plots.**

4.  **Interaction Heatmaps**: Visualizes the mean performance of each method
    across different experimental conditions (e.g., by joint or subject).

The script is structured to be easily configurable by modifying the variables
within the `main()` function.

Dependencies:
    - pandas
    - matplotlib
    - seaborn
    - scipy
    - numpy
"""

import os
from typing import Dict, List, Optional, Union, Any

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import numpy as np

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
    show_plots: bool,
    save_plots: bool
) -> None:
    """
    Applies final touches to a plot and saves it to disk.
    """
    figure.suptitle(plot_title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_plots:
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        print(f"Saved plot to {save_path}")

    if show_plots:
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

def _add_strip_and_whisker_elements(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    order: List[str],
    palette: Dict[str, str],
    show_labels: bool = True,
    ax: Optional[plt.Axes] = None,
    # --- NEW ARGUMENTS ---
    stats_dict: Optional[Dict] = None,
    factor_col_name: Optional[str] = None,
    # ---
    **kwargs: Any
) -> None:
    """
    Draws a stripplot, custom whisker lines (median/quartiles), and significance
    brackets onto a given Axes.
    """
    if ax is None:
        ax = plt.gca()
    jitter = kwargs.pop('jitter', 0.1)

    # --- 0. Get significance pairs for this specific facet ---
    significant_pairs = []
    if stats_dict and factor_col_name and not data.empty:
        # Get the name of this facet from the data
        facet_name = data[factor_col_name].iloc[0]
        significant_pairs = stats_dict.get(facet_name, [])

    # --- 1. Plot the strip plot ---
    sns.stripplot(
        data=data,
        x=x_col,
        y=y_col,
        order=order,
        palette=palette,
        hue=hue_col,
        legend=False,
        ax=ax,
        alpha=0.5,
        jitter=jitter,
        zorder=1
    )
    
    # --- 2. Add whisker lines and labels (Median/Quartiles) ---
    stats_df = data.groupby(x_col)[y_col].quantile([0.25, 0.5, 0.75]).unstack().reindex(order)
    if show_labels:
        line_width = kwargs.get('line_width', 0.8)
        whisker_width = kwargs.get('whisker_width', 0.7)
        
        for i, method in enumerate(order):
            if method not in stats_df.index:
                continue
            q1, median, q3 = stats_df.loc[method, 0.25], stats_df.loc[method, 0.5], stats_df.loc[method, 0.75]
            
            if pd.notna(q1) and pd.notna(median) and pd.notna(q3):
                method_color = palette.get(method, 'black')
                darker_color = sns.set_hls_values(method_color, l=0.4)
                x_offset = 0.1
                
                ax.hlines(q1, i - whisker_width/2, i + whisker_width/2, color=darker_color, linestyle='--', linewidth=1.5, zorder=10)
                ax.hlines(q3, i - whisker_width/2, i + whisker_width/2, color=darker_color, linestyle='--', linewidth=1.5, zorder=10)
                ax.hlines(median, i - whisker_width/2, i + line_width/2 + x_offset, color=darker_color, linestyle='-', linewidth=2, zorder=10)
                ax.text(i + whisker_width/2 + x_offset, median, f'{median:.1f}', ha='center', va='bottom', fontsize=16, color='black', fontweight='bold', zorder=11)

    # --- 3. Add Significance Brackets ---
    if significant_pairs:
        # Get the current top of the plot
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        
        # Find the highest data point (or whisker) to draw above
        all_q3s = stats_df[0.75]
        # Filter for methods present in this plot
        valid_q3s = all_q3s.reindex(order).dropna()
        if valid_q3s.empty:
            max_val = ylim[0] # Failsafe
        else:
            max_val = valid_q3s.max()
            
        # Add a text label, find its height to adjust
        # This is a bit of a hack to find the text height
        temp_text = ax.text(0, max_val, f'{max_val:.1f}', ha='center', va='bottom', fontsize=16, fontweight='bold', zorder=11)
        text_bbox = temp_text.get_window_extent(ax.figure.canvas.get_renderer())
        text_height_data_coords = (text_bbox.height / ax.figure.dpi) * (y_range / (ax.get_position().height * ax.figure.get_figheight()))
        temp_text.remove()
        
        # Start drawing brackets above the highest point + text
        y_step = text_height_data_coords * 1.5 # Height for one bracket
        current_y_level = max_val + text_height_data_coords * 2.0
        
        # Sort pairs by span (narrowest first) to draw them neatly
        sorted_pairs = sorted(
            significant_pairs, 
            key=lambda p: abs(order.index(p[0]) - order.index(p[1]))
        )
        
        for method1, method2 in sorted_pairs:
            if method1 not in order or method2 not in order:
                continue
                
            x1 = order.index(method1)
            x2 = order.index(method2)
            
            bar_y = current_y_level
            text_y = bar_y + (y_step * 0.1)
            
            # Draw the bracket
            ax.plot([x1, x1, x2, x2], [bar_y, text_y, text_y, bar_y], lw=1.5, c='black')
            # Draw the star
            ax.text((x1 + x2) * 0.5, text_y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold', c='black')
            
            current_y_level += y_step # Move up for the next bracket
        
        # Update the y-limit to make space for the brackets
        ax.set_ylim(ylim[0], current_y_level + (y_step * 0.5))


# --- NEW ---
def _add_bar_and_std_elements(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    order: List[str],
    palette: Dict[str, str],
    show_labels: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any
) -> None:
    """
    Draws a barplot with mean and std error bars onto a given Axes.
    """
    if ax is None:
        ax = plt.gca()

    # --- 1. Plot the bar plot with std error bars ---
    sns.barplot(
        data=data,
        x=x_col,
        y=y_col,
        order=order,
        palette=palette,
        hue=hue_col,
        legend=False,
        ax=ax,
        errorbar='ci',  # Use confidence interval for error bars
        capsize=0.1,
        zorder=2
    )
    ax.set_xticklabels(['Traditional', 'Proposed'], rotation=0, fontsize=16)
    sns.despine(left=True, bottom=True, top=True, right=True, ax=ax)

    # --- 2. Add text labels for the mean ---
    if show_labels:
        # Calculate stats needed for labels
        stats_df = data.groupby(x_col)[y_col].agg(['mean', 'std']).reindex(order)

        for i, method in enumerate(order):
            if method not in stats_df.index:
                continue
            
            mean_val = stats_df.loc[method, 'mean']
            std_val = stats_df.loc[method, 'std']
            
            if pd.notna(mean_val):
                # Position text above the std error bar
                y_pos = mean_val + (std_val / 8 if pd.notna(std_val) else 0)
                
                ax.text(
                    i,  # x-coordinate for a barplot is its index
                    y_pos,
                    f'{mean_val:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=16,
                    color='black',
                    fontweight='bold',
                    zorder=11
                )
        
        # Adjust Y-limit slightly to make room for text
        max_y_with_err = (stats_df['mean'] + stats_df['std'].fillna(0)).max() * 0.6
        current_ylim = ax.get_ylim()
        if pd.notna(max_y_with_err):
            ax.set_ylim(current_ylim[0], max(current_ylim[1], max_y_with_err * 1.0))

# --- MODIFIED ---
def plot_method_performance(
    summary_df: pd.DataFrame,
    metrics: List[str],
    plot_type: str = 'whisker', # --- NEW ---
    method_order: Optional[List[str]] = None,
    palette: Optional[Union[str, Dict[str, str]]] = None,
    show_labels: bool = True,
    show_plots: bool = False,
    save_plots: bool = True
) -> None:
    """
    Generates combined strip and box-whisker plots (overall performance).
    """
    if plot_type not in ['whisker', 'bar']:
        print(f"Error: Unknown plot_type '{plot_type}'. Defaulting to 'whisker'.")
        plot_type = 'whisker'
        
    print(f"\n--- Generating Overall Performance ({plot_type}) Plots ---")
    data = summary_df.reset_index()

    if method_order:
        data = data[data['method'].isin(method_order)]
        plot_order = ['EKF', 'Cascade']
    else:
        plot_order = sorted(data['method'].unique())

    if data.empty or len(plot_order) < 2:
        print("Warning: Not enough data or methods to generate plots. Skipping.")
        return

    color_map = {}
    if isinstance(palette, dict):
        color_map = palette
    elif isinstance(palette, str):
        colors = sns.color_palette(palette, n_colors=len(plot_order))
        color_map = dict(zip(plot_order, colors))
    else: 
        colors = sns.color_palette(n_colors=len(plot_order))
        color_map = dict(zip(plot_order, colors))

    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: Metric '{metric}' not found. Skipping.")
            continue
        
        print(f"\n--- Processing Metric: {metric} ({plot_type} Plot) ---")
        
        # --- 1. Run stats (always run, just print to console) ---
        sig_pairs = _run_statistical_analysis(data, metric, plot_order, alpha=0.05)
        
        fig, ax = plt.subplots(figsize=(4, 7))
        
        # --- 2. Call the correct plotting helper based on plot_type ---
        if plot_type == 'whisker':
            # Create a dummy dict and column for the helper function
            stats_dict = {'_overall_': sig_pairs}
            data['_overall_'] = '_overall_' # Dummy column
            
            _add_strip_and_whisker_elements(
                ax=ax,
                data=data,
                x_col='method',
                y_col=metric,
                hue_col='method',
                order=plot_order,
                palette=color_map,
                show_labels=show_labels,
                jitter=0.27,
                # Pass the stats info
                stats_dict=stats_dict,
                factor_col_name='_overall_'
            )
            data.drop(columns=['_overall_'], inplace=True, errors='ignore')
            ax.set_title(f"Overall Method Performance\n{metric} (Median/Quartiles)")

        elif plot_type == 'bar':
            _add_bar_and_std_elements(
                ax=ax,
                data=data,
                x_col='method',
                y_col=metric,
                hue_col='method',
                order=plot_order,
                palette=color_map,
                show_labels=show_labels
            )
            ax.set_title(f"Joint Angle Estimate \n{metric.replace('_deg', '')} by Method\n(Mean ± CI)")


        # --- 3. Finalize plot-specific labels ---
        ax.set_ylabel(f"Root Mean Squared Error\n(Degrees)" if "_deg" in metric else metric)
        ax.set_xlabel("", fontsize=12) 
        ax.tick_params(axis='x', rotation=25, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        # --- MODIFIED ---
        filename = f"overall_performance_{metric.lower()}_{plot_type}.png"
        _finalize_and_save_plot(fig, "", filename, show_plots, save_plots)


# --- MODIFIED ---
def plot_performance_by_factor(
    summary_df: pd.DataFrame,
    factor: str,
    metrics: List[str],
    plot_type: str = 'whisker', # --- NEW ---
    method_order: Optional[List[str]] = None,
    palette: Optional[Union[str, Dict[str, str]]] = None,
    show_labels: bool = True,
    show_plots: bool = False,
    save_plots: bool = True
) -> None:
    """
    Generates faceted plots (whisker or bar) by a given factor.
    """
    if plot_type not in ['whisker', 'bar']:
        print(f"Error: Unknown plot_type '{plot_type}'. Defaulting to 'whisker'.")
        plot_type = 'whisker'

    print(f"\n--- Generating Performance ({plot_type}) Plots by {factor.title()} ---")
    data = summary_df.reset_index()

    if factor not in data.columns:
        print(f"Error: '{factor}' column not found. Skipping factor-specific plots.")
        return

    if method_order:
        data = data[data['method'].isin(method_order)]
        plot_order = [m for m in method_order if m in data['method'].unique()]
    else:
        plot_order = sorted(data['method'].unique())

    if data.empty or len(plot_order) < 2:
        print("Warning: Not enough data or methods to generate plots. Skipping.")
        return
        
    factor_levels = sorted(data[factor].unique())

    color_map = {}
    if isinstance(palette, dict):
        color_map = palette
    elif isinstance(palette, str):
        colors = sns.color_palette(palette, n_colors=len(plot_order))
        color_map = dict(zip(plot_order, colors))
    else: 
        colors = sns.color_palette(n_colors=len(plot_order))
        color_map = dict(zip(plot_order, colors))

    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: Metric '{metric}' not found. Skipping.")
            continue

        print(f"\n--- Processing Metric: {metric} by {factor.title()} ({plot_type} Plot) ---")
        
        original_count = len(data)
        filtered_data =  _remove_outliers(data, metric, group_cols=[factor, 'method'])
        removed_count = original_count - len(filtered_data)
        if removed_count > 0:
            print(f"  > Removed {removed_count} outliers (beyond 1.5 IQR) for '{metric}'.")

        # --- 1. Pre-calculate stats for ALL facets (always run) ---
        stats_results = {}
        for level in factor_levels:
            print(f"\n--- Testing for {factor.title()} = {level} ---")
            level_data = filtered_data[filtered_data[factor] == level]
            # Run stats and store the list of significant pairs
            stats_results[level] = _run_statistical_analysis(level_data, metric, plot_order, alpha=0.05)
        print(f"--- End of Statistical Analysis for {metric} ---")
        
        
        # --- 2. Create the FacetGrid structure ---
        col_wrap = min(len(factor_levels), 7)
        g = sns.FacetGrid(
            filtered_data,
            col=factor,
            col_wrap=col_wrap,
            height=6,
            aspect=0.8,
            sharey=True # Must be False for per-facet brackets OR different bar heights
        )

        # --- 3. Map the correct plotting helper to each facet ---
        if plot_type == 'whisker':
            g.map_dataframe(
                _add_strip_and_whisker_elements,
                x_col='method',
                y_col=metric,
                hue_col='method',
                order=plot_order,
                palette=color_map,
                show_labels=show_labels,
                line_width=1.3,
                whisker_width=0.9,
                jitter=0.15,
                # Pass the full stats dict and factor column name
                stats_dict=stats_results,
                factor_col_name=factor
            )
        elif plot_type == 'bar':
            g.map_dataframe(
                _add_bar_and_std_elements,
                x_col='method',
                y_col=metric,
                hue_col='method',
                order=plot_order,
                palette=color_map,
                show_labels=show_labels
            )
        
        # --- 4. Finalize grid-level labels ---
        g.set_axis_labels(x_var="", y_var=f"{metric.replace('_deg', '')} (Degrees)" if "_deg" in metric else metric)
        g.set_xticklabels(rotation=45, ha='right')
        g.set_titles(f"{{col_name}}")
        
        g.fig.subplots_adjust(wspace=0.1, hspace=1)
        
        fig = g.fig 
        plot_title = f"Method Performance for {metric} by {factor.title()} ({plot_type})"
        # --- MODIFIED ---
        filename = f"by_{factor}_performance_{metric.lower()}_{plot_type}.png"
        _finalize_and_save_plot(fig, plot_title, filename, show_plots, save_plots)

def plot_interaction_heatmap(
    summary_df: pd.DataFrame,
    metrics: List[str],
    interaction_factor: str,
    method_order: Optional[List[str]] = None,
    factor_order: Optional[List[str]] = None,  # <-- ADD THIS
    cmap: str = 'Reds',
    show_plots: bool = False,
    save_plots: bool = True
) -> None:
    """
    Generates heatmaps to analyze performance by an interaction factor.
    (This function does not call the strip/whisker plot helper and
     is not modified to include brackets).
    """
    print(f"\n--- Generating Interaction Heatmaps for Method vs. {interaction_factor.title()} ---")
    data = summary_df.reset_index()

    if interaction_factor not in data.columns:
        print(f"Error: Interaction factor '{interaction_factor}' not found. Skipping.")
        return

    if method_order:
        data = data[data['method'].isin(method_order)]
        y_axis_order = [m for m in method_order if m in data['method'].unique()]
    else:
        y_axis_order = sorted(data['method'].unique())

    if data.empty or len(y_axis_order) < 2:
        print("Warning: Not enough data or methods to generate plots. Skipping.")
        return

    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: Metric '{metric}' not found. Skipping.")
            continue
        print(f"\n--- Processing Metric: {metric} ---")

        original_count = len(data)
        filtered_data = _remove_outliers(data, metric, group_cols=[interaction_factor, 'method'])
        removed_count = original_count - len(filtered_data)
        if removed_count > 0:
            print(f"  > Removed {removed_count} outliers (beyond 1.5 IQR) for '{metric}'.")

        # --- Statistics are still run and printed to console ---
        factor_levels = sorted(filtered_data[interaction_factor].unique())
        for level in factor_levels:
            print(f"\n--- Testing for {interaction_factor.title()} = {level} ---")
            level_data = filtered_data[filtered_data[interaction_factor] == level]
            _run_statistical_analysis(level_data, metric, y_axis_order, alpha=0.05)
        print(f"--- End of Statistical Analysis for {metric} ---")

        try:
            heatmap_data = filtered_data.groupby(['method', interaction_factor])[metric].mean().unstack()
            heatmap_data = heatmap_data.reindex(y_axis_order)

            # --- START: MODIFIED LOGIC ---
            if factor_order:
                # Use the provided order, but only for columns that exist in the data
                valid_factor_order = [f for f in factor_order if f in heatmap_data.columns]
                heatmap_data = heatmap_data.reindex(valid_factor_order, axis=1)
                heatmap_data = heatmap_data.reindex(['EKF', 'Cascade'], axis=0)
            else:
                # Default to alphabetical sorting if no order is given
                heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
            # --- END: MODIFIED LOGIC ---

            fig_width = len(heatmap_data.columns) * 1.0
            fig_height = len(heatmap_data.index) * 1.2
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            y_label_cbar = f"Mean {metric.replace('_deg', '')}\n(Degrees)" if "_deg" in metric else f"Mean {metric}"
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap, linewidths=.5, cbar_kws={'label': y_label_cbar}, ax=ax)
            
            ax.set_ylabel("", fontsize=12)
            ax.set_yticklabels(['Traditional', 'Proposed'], rotation=0, fontsize=16)
            ax.set_xlabel("", fontsize=12)

            ax.set_xticklabels(['Lumbar', 'Right Hip', 'Left Hip', 'Right Knee', 'Left Knee', 'Right Ankle', 'Left Ankle'], rotation=30, ha='right', fontsize=16)

            plot_title = f""
            filename = f"interaction_heatmap_{metric.lower()}_vs_{interaction_factor.lower()}.png"
            _finalize_and_save_plot(fig, plot_title, filename, show_plots, save_plots)
        except Exception as e:
            print(f"    Error plotting heatmap for {metric}: {e}")
            plt.close()

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
    
    # --- 1. Configuration ---
    DATA_FILE_PATH = os.path.join("data", "data", "all_subject_statistics.pkl")
    PLOTS_DIRECTORY = "plots"
    
    SHOW_PLOTS = True
    SAVE_PLOTS = True

    # --- NEW ---
    # Set the desired plot style:
    # 'whisker' = strip plot with median/quartile whiskers and sig. brackets
    # 'bar'     = bar plot with mean and standard deviation error bars
    PLOT_STYLE = 'bar' # <-- CHANGE THIS to 'whisker' to get the old plots
    # ---

    METRICS_TO_PLOT = ['RMSE_deg'] 
    METHODS_IN_ORDER = [
        'EKF',
        # 'Madgwick (Al Borno)',
        # 'Mag Free',
        # 'Never Project',
        'Cascade',
    ]

    # --- 2. Load Data ---
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Statistics file not found at {DATA_FILE_PATH}")
        print("Creating a dummy dataframe for demonstration.")
        num_entries = 200
        data = {
            'method': np.random.choice(METHODS_IN_ORDER, num_entries),
            'trial_type': np.random.choice(['Slow', 'Medium', 'Fast'], num_entries),
            'joint_name': np.random.choice(['Knee', 'Elbow'], num_entries),
            'subject': np.random.choice(['S1', 'S2', 'S3'], num_entries),
            'axis': np.random.choice(['X', 'Y', 'Z'], num_entries),
            'RMSE_deg': np.random.rand(num_entries) * 10,
            'MAE_deg': np.random.rand(num_entries) * 8,
            'Mean_deg': np.random.randn(num_entries) * 5,
            'STD_deg': np.random.rand(num_entries) * 3
        }
        # Add some offsets to make dummy stats more interesting
        method_offsets = {m: np.random.rand() * 5 for m in METHODS_IN_ORDER}
        joint_offsets = {'Knee': 2, 'Elbow': -2}
        
        summary_stats_df = pd.DataFrame(data)
        summary_stats_df['RMSE_deg'] += summary_stats_df['method'].map(method_offsets)
        summary_stats_df['RMSE_deg'] += summary_stats_df['joint_name'].map(joint_offsets)
        summary_stats_df['STD_deg'] += summary_stats_df['method'].map(method_offsets) / 2.0
        summary_stats_df['RMSE_deg'] = summary_stats_df['RMSE_deg'].clip(lower=0.5)
        summary_stats_df['STD_deg'] = summary_stats_df['STD_deg'].clip(lower=0.1)

    else:
        print(f"Loading summary statistics from {DATA_FILE_PATH}...")
        try:
            summary_stats_df = pd.read_pickle(DATA_FILE_PATH)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return

    # --- 3. Generate Overall Performance Plots (Strip + Whisker) ---
    # --- MODIFIED ---
    plot_method_performance(
        summary_df=summary_stats_df, metrics=METRICS_TO_PLOT,
        plot_type=PLOT_STYLE, # <-- Pass the style
        method_order=METHODS_IN_ORDER, palette="Set2",
        show_labels=True, show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS
    )
    
    # # --- 4. Generate Performance Plots Faceted by Various Factors ---
    # # --- MODIFIED ---
    # plot_performance_by_factor(
    #     summary_df=summary_stats_df, factor='joint_name', metrics=METRICS_TO_PLOT,
    #     plot_type=PLOT_STYLE, # <-- Pass the style
    #     method_order=METHODS_IN_ORDER, palette="Set2",
    #     show_labels=True, show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS
    # )
    
    # --- 5. Generate Interaction Heatmaps for various factors ---
    FACTOR_ORDERS = {
            'joint_name': ['Lumbar', 'R_Hip', 'L_Hip', 'R_Knee', 'L_Knee', 'R_Ankle', 'L_Ankle'], # <-- DEFINE YOUR CUSTOM ORDER HERE
            # 'trial_type': ['Slow', 'Medium', 'Fast']
            # Add any other factor orders you might need
        }
    
    interaction_factors = ['joint_name']
    for factor in interaction_factors:
        
        # Get the custom order for this factor, or None if not defined
        custom_factor_order = FACTOR_ORDERS.get(factor)
        
        plot_interaction_heatmap(
            summary_df=summary_stats_df, metrics=METRICS_TO_PLOT,
            interaction_factor=factor, 
            method_order=METHODS_IN_ORDER,
            factor_order=custom_factor_order, # <-- PASS THE NEW ARGUMENT
            cmap='Reds', show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS
        )

    print("\n--- All plotting complete ---")

if __name__ == "__main__":
    main()