# -*- coding: utf-8 -*-
"""
plot_paper_figures.py

This script is designed to generate publication-quality figures and perform
statistical analysis for comparing the performance of various estimation methods.
It loads a pre-processed pandas DataFrame containing performance metrics and
produces three main types of visualizations:

1.  **Overall Performance Plots**: Compares the distribution of key metrics
    (e.g., RMSE, MAE) across all methods using box, strip, or swarm plots.
    Includes a robust statistical analysis using the Friedman test, followed
    by a Wilcoxon signed-rank post-hoc test if significant.

2.  **Performance by Trial Plots**: Creates a faceted plot to compare
    method performance side-by-side for each experimental trial type. Also
    supports box, strip, and swarm plots.

3.  **Interaction Heatmaps**: Visualizes the mean performance of each method
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
from typing import Dict, List, Optional, Union

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
) -> None:
    """
    Performs statistical analysis using Friedman and Wilcoxon post-hoc tests.

    This is a helper function designed to test for significant differences
    between multiple methods on dependent samples. It pivots the data to align
    measurements from the same "block" (e.g., the same subject/joint/axis)
    and then runs the appropriate non-parametric tests.

    Args:
        df (pd.DataFrame): The long-format DataFrame containing the data to test.
            Must contain 'method', the specified metric column, and block columns.
        metric (str): The name of the performance metric column to analyze.
        methods_order (List[str]): The list of method names to include in the test.
        alpha (float): The significance level for the tests.
    """
    print(f"--- a. Dependent-Sample Test (Friedman Test for '{metric}') ---")

    try:
        # Define the columns that create a unique "block" for repeated measures.
        block_cols = [
            col for col in ['trial_type', 'joint_name', 'subject', 'axis']
            if col in df.columns
        ]
        if not block_cols:
            print("  > Error: Could not find block columns for Friedman test.")
            return

        # Create a unique block_id for each combination of conditions.
        df = df.copy()
        df['block_id'] = df[block_cols].apply(
            lambda row: '_'.join(row.values.astype(str)), axis=1
        )

        # Pivot the data so each row is a block and each column is a method.
        # This is WIDE-FORMAT data.
        pivot_df = df.pivot_table(
            index='block_id', columns='method', values=metric
        )

        # The Friedman test requires complete blocks (no missing values for any method).
        pivot_df_clean = pivot_df.dropna()

        # Ensure the methods being tested are present in the cleaned data.
        valid_methods = [m for m in methods_order if m in pivot_df_clean.columns]

        # Check if we have enough data to run the test.
        if pivot_df_clean.shape[0] < 2 or len(valid_methods) < 2:
            print(f"  > Skipping stats: Insufficient data (N_blocks={pivot_df_clean.shape[0]}, N_methods={len(valid_methods)}).")
            return

        print(f"  > Using {pivot_df_clean.shape[0]} complete blocks for {len(valid_methods)} methods.")

        # Prepare the list of data series for the test.
        groups_for_test = [pivot_df_clean[col] for col in valid_methods]

        # --- b. Run Friedman Test ---
        _, p_friedman = stats.friedmanchisquare(*groups_for_test)
        print(f"  Friedman Test (overall comparison): p-value = {p_friedman:.4e}")

        # --- c. Run Post-hoc Test if overall difference is significant ---
        if p_friedman >= alpha:
            print("  > No significant overall difference found between methods (p >= alpha).")
        else:
            print("  > Significant difference detected. Running post-hoc (Wilcoxon Signed-Rank)...")

            # Perform pairwise Wilcoxon signed-rank tests for all method pairs.
            p_uncorrected_list = []
            method_pairs = []

            # Iterate over unique pairs of methods.
            for i in range(len(valid_methods)):
                for j in range(i + 1, len(valid_methods)):
                    method1, method2 = valid_methods[i], valid_methods[j]
                    method_pairs.append((method1, method2))
                    try:
                        _, p_val = stats.wilcoxon(pivot_df_clean[method1], pivot_df_clean[method2])
                        p_uncorrected_list.append(p_val)
                    except ValueError:
                        p_uncorrected_list.append(1.0)

            # Apply Holm-Bonferroni correction for multiple comparisons.
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

            # Create the final results matrix.
            p_adj_matrix = pd.DataFrame(
                np.ones((len(valid_methods), len(valid_methods))),
                index=valid_methods, columns=valid_methods
            )
            for (method1, method2), p_adj in zip(method_pairs, p_adjusted):
                p_adj_matrix.loc[method1, method2] = p_adj
                p_adj_matrix.loc[method2, method1] = p_adj

            print("  Significant pairs (p_adj < alpha):")
            significant_pairs = 0
            for i in range(len(p_adj_matrix.columns)):
                for j in range(i + 1, len(p_adj_matrix.columns)):
                    method1, method2 = p_adj_matrix.columns[i], p_adj_matrix.columns[j]
                    p_adj = p_adj_matrix.iloc[i, j]
                    if p_adj < alpha:
                        print(f"    - {method1} vs. {method2}: p_adj = {p_adj:.4e}")
                        significant_pairs += 1

            if significant_pairs == 0:
                print("    - None (after p-value correction).")

    except (ValueError, AttributeError) as e:
        print(f"  > An error occurred during statistical testing: {e}")

def _finalize_and_save_plot(
    figure: plt.Figure,
    plot_title: str,
    filename: str,
    show_plots: bool,
    save_plots: bool
) -> None:
    """
    Applies final touches to a plot and saves it to disk.

    Args:
        figure (plt.Figure): The figure object to finalize.
        plot_title (str): The title for the plot.
        filename (str): The filename for the saved plot image.
        show_plots (bool): If True, display the plot interactively.
        save_plots (bool): If True, save the plot to the 'plots' directory.
    """
    figure.suptitle(plot_title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_plots:
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show_plots:
        plt.show()

    plt.close(figure) # Close the figure to free up memory.


# --- Main Plotting Functions ---

def plot_method_performance(
    summary_df: pd.DataFrame,
    metrics: List[str],
    plot_type: str = 'box',
    method_order: Optional[List[str]] = None,
    palette: Optional[Union[str, Dict[str, str]]] = None,
    run_stats: bool = True,
    alpha: float = 0.05,
    show_labels: bool = True,
    show_plots: bool = False,
    save_plots: bool = True
) -> None:
    """
    Generates plots to compare the overall performance of different methods.

    Args:
        ...
        plot_type (str): Type of plot: 'box', 'violin', 'strip', 'swarm',
            or 'histogram'.
        ...
        show_labels (bool): If True, adds Q1, median, and Q3 labels
            to 'box' plots, or median/IQR lines to 'strip'/'swarm' plots.
    """
    print(f"\n--- Generating Overall Performance {plot_type.title()} Plots ---")
    data = summary_df.reset_index()

    if method_order:
        data = data[data['method'].isin(method_order)]
        plot_order = [m for m in method_order if m in data['method'].unique()]
    else:
        plot_order = sorted(data['method'].unique())

    if data.empty or len(plot_order) < 2:
        print("Warning: Not enough data or methods to generate plots. Skipping.")
        return

    # --- Create a consistent color map ---
    color_map = {}
    if isinstance(palette, dict):
        color_map = palette
    elif isinstance(palette, str):
        colors = sns.color_palette(palette, n_colors=len(plot_order))
        color_map = dict(zip(plot_order, colors))
    else: # Default palette
        colors = sns.color_palette(n_colors=len(plot_order))
        color_map = dict(zip(plot_order, colors))

    # Define the mapping from plot_type string to the Seaborn function
    plot_funcs = {
        'box': sns.boxplot,
        'violin': sns.violinplot,
        'strip': sns.stripplot,
        'swarm': sns.swarmplot
    }
    if plot_type not in plot_funcs and plot_type != 'histogram':
        print(f"Error: Unsupported plot_type '{plot_type}'. Choose from {list(plot_funcs.keys()) + ['histogram']}.")
        return

    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: Metric '{metric}' not found. Skipping.")
            continue
        print(f"\n--- Processing Metric: {metric} ({plot_type.title()} Plot) ---")
        if run_stats and plot_type != 'histogram': # Stats are for comparing methods
            _run_statistical_analysis(data, metric, plot_order, alpha)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if plot_type in plot_funcs:
            plot_func = plot_funcs[plot_type]
            # --- MODIFIED: Added zorder to stripplot/swarmplot ---
            plot_kwargs = {'data': data, 'x': 'method', 'y': metric, 'order': plot_order, 'palette': color_map, 'hue': 'method', 'legend': False, 'ax': ax}
            if plot_type in ['strip', 'swarm']:
                plot_kwargs['zorder'] = 1 # Draw points behind markers
            
            plot_func(**plot_kwargs)
            
            # --- MODIFIED: Re-structured logic to add labels or markers ---
            if show_labels:
                # Calculate stats for all methods
                stats_df = data.groupby('method')[metric].quantile([0.25, 0.5, 0.75]).unstack().reindex(plot_order)
                
                # Add quantile labels, but ONLY for box plots.
                if plot_type == 'box':
                    ylim = ax.get_ylim()
                    y_offset = (ylim[1] - ylim[0]) * 0.015
                    for i, method in enumerate(plot_order):
                        if method not in stats_df.index: continue
                        q1, median, q3 = stats_df.loc[method, 0.25], stats_df.loc[method, 0.5], stats_df.loc[method, 0.75]
                        if pd.notna(q1) and pd.notna(median) and pd.notna(q3):
                            ax.text(i, q3 + y_offset, f'{q3:.1f}', ha='center', va='bottom', fontsize=8)
                            ax.text(i, median, f'{median:.1f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
                            ax.text(i, q1 - y_offset, f'{q1:.1f}', ha='center', va='top', fontsize=8)

                # --- NEW: Add median/IQR lines for strip/swarm plots ---
                elif plot_type in ['strip', 'swarm']:
                    line_width = 0.4    # Total width of the median line
                    whisker_width = 0.2 # Total width of the Q1/Q3 whiskers
                    
                    for i, method in enumerate(plot_order):
                        if method not in stats_df.index: continue
                        q1, median, q3 = stats_df.loc[method, 0.25], stats_df.loc[method, 0.5], stats_df.loc[method, 0.75]
                        
                        if pd.notna(q1) and pd.notna(median) and pd.notna(q3):
                            # Draw the vertical IQR line
                            ax.vlines(i, q1, q3, color='black', linestyle='-', linewidth=1, zorder=10)
                            # Draw the Q1 whisker
                            ax.hlines(q1, i - whisker_width/2, i + whisker_width/2, color='black', linestyle='-', linewidth=1, zorder=10)
                            # Draw the Q3 whisker
                            ax.hlines(q3, i - whisker_width/2, i + whisker_width/2, color='black', linestyle='-', linewidth=1, zorder=10)
                            # Draw the median line (make it stand out)
                            ax.hlines(median, i - line_width/2, i + line_width/2, color='red', linestyle='-', linewidth=2, zorder=10)
                # --- END NEW ---

            ax.set_ylabel(f"{metric} (Degrees)" if "_deg" in metric else metric, fontsize=12)
            ax.set_xlabel("Method", fontsize=12)
            ax.tick_params(axis='x', rotation=45)

        elif plot_type == 'histogram':
            # ... (histogram logic remains unchanged) ...
            for method in plot_order:
                method_data = data[data['method'] == method][metric].dropna()
                if method_data.empty:
                    continue
                sns.histplot(
                    method_data,
                    label=method,
                    ax=ax,
                    kde=True,
                    stat="density",
                    element="step",
                    alpha=0.5,
                    color=color_map.get(method)
                )
            ax.legend(title="Method")
            ax.set_xlabel(f"{metric} (Degrees)" if "_deg" in metric else metric, fontsize=12)
            ax.set_ylabel("Density", fontsize=12)

        ax.set_title(f"Overall Method Performance: {metric}", fontsize=16)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        filename = f"overall_performance_{metric.lower()}_{plot_type}.png"
        _finalize_and_save_plot(fig, "", filename, show_plots, save_plots)

def plot_performance_by_trial(
    summary_df: pd.DataFrame,
    metrics: List[str],
    plot_type: str = 'box',
    method_order: Optional[List[str]] = None,
    palette: Optional[Union[str, Dict[str, str]]] = None,
    run_stats: bool = True,
    alpha: float = 0.05,
    show_labels: bool = True,
    show_plots: bool = False,
    save_plots: bool = True
) -> None:
    """
    Generates faceted plots to compare method performance for each trial type.

    Args:
        ...
        plot_type (str): Type of plot: 'box', 'violin', 'strip', 'swarm',
            or 'histogram'.
        ...
        show_labels (bool): If True, adds Q1, median, and Q3 labels
            to 'box' plots, or median/IQR lines to 'strip'/'swarm' plots.
    """
    print(f"\n--- Generating Performance {plot_type.title()} Plots by Trial Type ---")
    data = summary_df.reset_index()

    if 'trial_type' not in data.columns:
        print("Error: 'trial_type' column not found. Skipping trial-specific plots.")
        return

    if method_order:
        data = data[data['method'].isin(method_order)]
        plot_order = [m for m in method_order if m in data['method'].unique()]
    else:
        plot_order = sorted(data['method'].unique())

    if data.empty or len(plot_order) < 2:
        print("Warning: Not enough data or methods to generate plots. Skipping.")
        return
        
    trial_types = sorted(data['trial_type'].unique())

    # --- Create a consistent color map ---
    color_map = {}
    if isinstance(palette, dict):
        color_map = palette
    elif isinstance(palette, str):
        colors = sns.color_palette(palette, n_colors=len(plot_order))
        color_map = dict(zip(plot_order, colors))
    else: # Default palette
        colors = sns.color_palette(n_colors=len(plot_order))
        color_map = dict(zip(plot_order, colors))

    for metric in metrics:
        if metric not in data.columns:
            print(f"Warning: Metric '{metric}' not found. Skipping.")
            continue

        print(f"\n--- Processing Metric: {metric} by Trial Type ({plot_type.title()} Plot) ---")
        
        if run_stats:
            for trial in trial_types:
                print(f"\n--- Testing for Trial Type = {trial} ---")
                trial_data = data[data['trial_type'] == trial]
                _run_statistical_analysis(trial_data, metric, plot_order, alpha)
            print(f"--- End of Statistical Analysis for {metric} ---")
        
        catplot_kinds = ['box', 'violin', 'strip', 'swarm', 'boxen', 'point', 'bar', 'count']

        if plot_type in catplot_kinds:
            # --- MODIFIED: Added zorder to stripplot/swarmplot ---
            plot_kwargs = {}
            if plot_type in ['strip', 'swarm']:
                plot_kwargs['zorder'] = 1 # Draw points behind markers
            
            g = sns.catplot(
                data=data,
                x='method',
                y=metric,
                col='trial_type',
                order=plot_order,
                kind=plot_type,
                palette=color_map,
                height=6,
                aspect=1.2,
                sharey=False,
                hue='method',
                legend=False,
                **plot_kwargs
            )

            g.set_axis_labels("Method", f"{metric} (Degrees)" if "_deg" in metric else metric)
            g.set_xticklabels(rotation=45, ha='right')
            g.set_titles("Trial: {col_name}")

            # --- MODIFIED: Re-structured loop to handle all plot types ---
            for i, ax in enumerate(g.axes.flat):
                # Stop if we run out of trial types (e.g., 3 trials on a 2x2 grid)
                if i >= len(trial_types):
                    break
                
                trial_name = trial_types[i]
                trial_data = data[data['trial_type'] == trial_name]

                if show_labels:
                    # Calculate stats for this specific facet
                    stats_df = trial_data.groupby('method')[metric].quantile([0.25, 0.5, 0.75]).unstack().reindex(plot_order)
                    
                    if plot_type == 'box':
                        ylim = ax.get_ylim()
                        y_offset = (ylim[1] - ylim[0]) * 0.015
                        
                        for j, method in enumerate(plot_order):
                            if method not in stats_df.index: continue
                            q1, median, q3 = stats_df.loc[method, 0.25], stats_df.loc[method, 0.5], stats_df.loc[method, 0.75]
                            if pd.notna(q1) and pd.notna(median) and pd.notna(q3):
                                ax.text(j, q3 + y_offset, f'{q3:.1f}', ha='center', va='bottom', fontsize=8)
                                ax.text(j, median, f'{median:.1f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
                                ax.text(j, q1 - y_offset, f'{q1:.1f}', ha='center', va='top', fontsize=8)
                    
                    # --- NEW: Add median/IQR lines for strip/swarm plots ---
                    elif plot_type in ['strip', 'swarm']:
                        line_width = 0.4    # Total width of the median line
                        whisker_width = 0.2 # Total width of the Q1/Q3 whiskers
                        
                        for j, method in enumerate(plot_order): # 'j' is the x-position index
                            if method not in stats_df.index: continue
                            q1, median, q3 = stats_df.loc[method, 0.25], stats_df.loc[method, 0.5], stats_df.loc[method, 0.75]
                            
                            if pd.notna(q1) and pd.notna(median) and pd.notna(q3):
                                # Draw the vertical IQR line
                                ax.vlines(j, q1, q3, color='black', linestyle='-', linewidth=1, zorder=10)
                                # Draw the Q1 whisker
                                ax.hlines(q1, j - whisker_width/2, j + whisker_width/2, color='black', linestyle='-', linewidth=1, zorder=10)
                                # Draw the Q3 whisker
                                ax.hlines(q3, j - whisker_width/2, j + whisker_width/2, color='black', linestyle='-', linewidth=1, zorder=10)
                                # Draw the median line (make it stand out)
                                ax.hlines(median, j - line_width/2, j + line_width/2, color='red', linestyle='-', linewidth=2, zorder=10)
                    # --- END NEW ---

                # Add grid to all axes
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            # --- END MODIFIED BLOCK ---
            
            fig = g.fig # Get the figure object from the FacetGrid
            plot_title = f"Method Performance for {metric} by Trial Type"

        elif plot_type == 'histogram':
            # ... (histogram logic remains unchanged) ...
            n_trials = len(trial_types)
            ncols = min(n_trials, 3)
            nrows = int(np.ceil(n_trials / ncols))
            
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 5), squeeze=False, sharex=True)
            axes_flat = axes.flatten()
            
            metric_min = data[metric].min()
            metric_max = data[metric].max()
            bins = np.linspace(metric_min, metric_max, 50)

            for i, trial in enumerate(trial_types):
                ax = axes_flat[i]
                trial_data = data[data['trial_type'] == trial]
                
                for method in plot_order:
                    method_data = trial_data[trial_data['method'] == method][metric].dropna()
                    if method_data.empty:
                        continue
                    sns.histplot(
                        method_data,
                        label=method,
                        ax=ax,
                        kde=True,
                        stat="density",
                        element="step",
                        alpha=0.5,
                        color=color_map.get(method),
                        bins=bins
                    )
                
                ax.set_title(f"Trial: {trial}")
                ax.set_ylabel("Density")
                ax.grid(axis='y', linestyle='--', alpha=0.7)

            handles, labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, title="Method", loc='upper right')
            fig.text(0.5, 0.01, f"{metric} (Degrees)" if "_deg" in metric else metric, ha='center', va='bottom', fontsize=12)
            
            for j in range(i + 1, len(axes_flat)):
                axes_flat[j].axis('off')

            plot_title = f"Method Performance for {metric} by Trial Type"
            
        else:
            print(f"Error: Unsupported plot_type '{plot_type}'. Choose from {catplot_kinds + ['histogram']}.")
            continue 

        filename = f"by_trial_performance_{metric.lower()}_{plot_type}.png"
        _finalize_and_save_plot(fig, plot_title, filename, show_plots, save_plots)

def plot_interaction_heatmap(
    summary_df: pd.DataFrame,
    metrics: List[str],
    interaction_factor: str,
    method_order: Optional[List[str]] = None,
    cmap: str = 'Reds',
    run_stats: bool = True,
    alpha: float = 0.05,
    show_plots: bool = False,
    save_plots: bool = True
) -> None:
    """
    Generates heatmaps to analyze performance by an interaction factor.
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

        if run_stats:
            factor_levels = sorted(data[interaction_factor].unique())
            for level in factor_levels:
                print(f"\n--- Testing for {interaction_factor.title()} = {level} ---")
                level_data = data[data[interaction_factor] == level]
                _run_statistical_analysis(level_data, metric, y_axis_order, alpha)
            print(f"--- End of Statistical Analysis for {metric} ---")

        try:
            heatmap_data = data.groupby(['method', interaction_factor])[metric].mean().unstack()
            heatmap_data = heatmap_data.reindex(y_axis_order)
            heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)

            fig_width = max(12, len(heatmap_data.columns) * 1.5)
            fig_height = max(8, len(heatmap_data.index) * 0.8)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            y_label_cbar = f"Mean {metric} (Degrees)" if "_deg" in metric else f"Mean {metric}"
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap, linewidths=.5, cbar_kws={'label': y_label_cbar}, ax=ax)
            
            ax.set_ylabel("Method", fontsize=12)
            ax.set_xlabel(interaction_factor.title(), fontsize=12)

            plot_title = f"Mean {metric} by Method and {interaction_factor.title()}"
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
    # --- 1. Configuration ---
    DATA_FILE_PATH = os.path.join("data", "data", "all_subject_statistics.pkl")
    PLOTS_DIRECTORY = "plots"
    
    # Plotting settings
    SHOW_PLOTS = True
    SAVE_PLOTS = True
    RUN_STATS = True

    # Define the metrics and methods to analyze.
    METRICS_TO_PLOT = ['RMSE_deg', 'Mean_deg', 'STD_deg'] # Reduced for quicker demo
    METHODS_IN_ORDER = [
        'EKF',
        'Madgwick (Al Borno)',
        'Mag Free',
        'Never Project',
        'Cascade',
    ]
    
    # --- 2. Load Data ---
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Statistics file not found at {DATA_FILE_PATH}")
        # Create a dummy dataframe for demonstration purposes if file not found
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
        summary_stats_df = pd.DataFrame(data)
    else:
        print(f"Loading summary statistics from {DATA_FILE_PATH}...")
        try:
            summary_stats_df = pd.read_pickle(DATA_FILE_PATH)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return

    # --- 3. Generate Overall Performance Plots (Box and Swarm) ---
    # plot_method_performance(
    #     summary_df=summary_stats_df, metrics=METRICS_TO_PLOT, plot_type='box',
    #     method_order=METHODS_IN_ORDER, palette="Set2", run_stats=RUN_STATS,
    #     show_labels=True, show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS
    # )
    plot_method_performance(
        summary_df=summary_stats_df, metrics=METRICS_TO_PLOT, plot_type='strip',
        method_order=METHODS_IN_ORDER, palette="Set2", run_stats=False, # Stats already run
        show_labels=True, show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS
    )
    
    # --- 4. Generate Performance Plots Faceted by Trial (Box and Strip) ---
    # plot_performance_by_trial(
    #     summary_df=summary_stats_df, metrics=METRICS_TO_PLOT, plot_type='box',
    #     method_order=METHODS_IN_ORDER, palette="Set2", run_stats=RUN_STATS,
    #     show_labels=True, show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS
    # )
    plot_performance_by_trial(
        summary_df=summary_stats_df, metrics=METRICS_TO_PLOT, plot_type='strip',
        method_order=METHODS_IN_ORDER, palette="Set2", run_stats=False, # Stats already run
        show_labels=True, show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS
    )

    # --- 5. Generate Interaction Heatmaps for various factors ---
    # interaction_factors = ['joint_name', 'subject', 'trial_type', 'axis'] # Reduced for quicker demo
    # for factor in interaction_factors:
    #     plot_interaction_heatmap(
    #         summary_df=summary_stats_df, metrics=METRICS_TO_PLOT,
    #         interaction_factor=factor, method_order=METHODS_IN_ORDER,
    #         cmap='Reds', run_stats=RUN_STATS, show_plots=SHOW_PLOTS, save_plots=SAVE_PLOTS
    #     )

    print("\n--- All plotting complete ---")

if __name__ == "__main__":
    main()