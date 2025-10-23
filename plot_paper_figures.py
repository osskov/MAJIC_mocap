from typing import List, Dict, Union, Optional
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scikit_posthocs as sp

# --- Plotting Functions ---
def plot_method_performance(
    summary_df: pd.DataFrame, 
    metrics: List[str], 
    plot_type: str = 'box',
    method_filter: Optional[List[str]] = None, # <-- This now controls order too
    palette: Optional[Union[str, Dict[str, str]]] = None,
    show_plots: bool = False,
    save_plots: bool = True,
    run_stats: bool = True, # Default is True as in your last version
    alpha: float = 0.05,
    show_labels: bool = True
) -> None:
    """
    Generates and saves box plots or violin plots for specified metrics
    to compare overall method performance.
    ... [rest of docstring] ...
    Args:
        ...
        show_labels (bool, optional): If True, adds Q1, Median, Q3 labels to box plots.
    """
    print(f"--- Generating {plot_type} plots for {', '.join(metrics)} ---")
    
    # Ensure the plot directory exists
    if save_plots:
        os.makedirs("plots", exist_ok=True)
        
    # Reset index to make 'method' and other index levels available as columns
    data_for_plot = summary_df.reset_index()
    
    # --- 1. Determine method_order and apply filter ---
    plot_order_arg = None
    
    if method_filter:
        # Filter the DataFrame to only include methods from the list
        data_for_plot = data_for_plot[data_for_plot['method'].isin(method_filter)]
        if data_for_plot.empty:
            print(f"Warning: Filtering by methods {method_filter} resulted in an empty DataFrame. Skipping plots.")
            return
            
        # Set the plot order to be the order of the filter list
        plot_order_arg = [m for m in method_filter if m in data_for_plot['method'].unique()]
    else:
        # Default to alphabetical order if no filter is provided
        plot_order_arg = sorted(data_for_plot['method'].unique())
    
    # Make sure we're only testing data that will be plotted
    # This is important if method_filter was used
    stats_data_df = data_for_plot[data_for_plot['method'].isin(plot_order_arg)]

    for metric in metrics:
        if metric not in data_for_plot.columns:
            print(f"Warning: Metric '{metric}' not found in summary DataFrame. Skipping plot.")
            continue
            
        # --- 2. [UPDATED] Run statistical tests if requested ---
        if run_stats:
            print(f"\n--- Statistical Analysis for {metric} (alpha={alpha}) ---")
            
            # Create a list of data arrays for each group (method)
            groups_data = [
                stats_data_df[stats_data_df['method'] == method][metric].dropna() 
                for method in plot_order_arg
            ]

            # Filter out empty groups (can happen with aggressive filtering)
            valid_groups_data = [g for g in groups_data if not g.empty]

            if len(valid_groups_data) <= 1:
                print("  > Skipping stats, only one group with data to plot.")
                continue # Skip to the plotting section for this metric

            # --- a. [NEW] Normality Check (Shapiro-Wilk) ---
            print("--- a. Normality Check (Shapiro-Wilk) ---")
            all_groups_are_normal = True
            for method_name, group_data in zip(plot_order_arg, groups_data):
                if group_data.empty:
                    continue
                if len(group_data) < 3:
                    print(f"  > {method_name}: Skipping (N={len(group_data)} < 3). Cannot test normality.")
                    all_groups_are_normal = False # Be conservative
                    continue
                
                try:
                    shapiro_stat, p_shapiro = stats.shapiro(group_data)
                    print(f"  > {method_name}: p-value = {p_shapiro:.4e} (N={len(group_data)})")
                    if p_shapiro < alpha:
                        print("     - Data is NOT normally distributed (p < alpha).")
                        all_groups_are_normal = False
                    else:
                        print("     - Data IS normally distributed (p >= alpha).")
                except ValueError as e:
                    print(f"  > {method_name}: Error during Shapiro-Wilk test: {e}")
                    all_groups_are_normal = False # Be conservative

            # --- b. [NEW] Choose test based on normality ---
            try:
                if all_groups_are_normal:
                    # --- Parametric Test: ANOVA + Tukey ---
                    print("\n--- b. Parametric Test (ANOVA) ---")
                    f_stat, p_anova = stats.f_oneway(*valid_groups_data)
                    print(f"ANOVA (overall): p-value = {p_anova:.4e}")
                    
                    if p_anova >= alpha:
                        print("  > No significant difference found between any groups (p >= alpha).")
                    else:
                        print("  > Significant difference detected. Running post-hoc (Tukey's HSD)...")
                        
                        # Pairwise post-hoc test (Tukey's HSD)
                        tukey_results = sp.posthoc_tukey(
                            stats_data_df, 
                            val_col=metric, 
                            group_col='method'
                        )
                        
                        # Filter for only the methods we're plotting
                        tukey_results = tukey_results.loc[plot_order_arg, plot_order_arg]
                        
                        print("  Significant pairs (p_adj < alpha):")
                        significant_pairs = 0
                        # Iterate over the upper triangle of the matrix to avoid repeats
                        for i in range(len(tukey_results.columns)):
                            for j in range(i + 1, len(tukey_results.columns)):
                                method1 = tukey_results.columns[i]
                                method2 = tukey_results.columns[j]
                                p_adj = tukey_results.iloc[i, j]
                                
                                if p_adj < alpha:
                                    print(f"    - {method1} vs. {method2}: p_adj = {p_adj:.4e}")
                                    significant_pairs += 1
                                    
                        if significant_pairs == 0:
                            print("    - None (after p-value correction)")
                
                else:
                    # --- Non-Parametric Test: Kruskal-Wallis + Dunn ---
                    print("\n--- b. Non-Parametric Test (Kruskal-Wallis) ---")
                    h_stat, p_kruskal = stats.kruskal(*valid_groups_data)
                    print(f"Kruskal-Wallis H-test (overall): p-value = {p_kruskal:.4e}")
                    
                    if p_kruskal >= alpha:
                        print("  > No significant difference found between any groups (p >= alpha).")
                    else:
                        print("  > Significant difference detected. Running post-hoc (Dunn's) tests...")
                        
                        # p_adjust='holm' applies the robust Holm-Bonferroni correction
                        dunn_results = sp.posthoc_dunn(
                            stats_data_df, 
                            val_col=metric, 
                            group_col='method', 
                            p_adjust='holm'
                        )
                        
                        # Filter for only the methods we're plotting
                        dunn_results = dunn_results.loc[plot_order_arg, plot_order_arg]
                        
                        print("  Significant pairs (p_adj < alpha):")
                        significant_pairs = 0
                        # Iterate over the upper triangle of the matrix to avoid repeats
                        for i in range(len(dunn_results.columns)):
                            for j in range(i + 1, len(dunn_results.columns)):
                                method1 = dunn_results.columns[i]
                                method2 = dunn_results.columns[j]
                                p_adj = dunn_results.iloc[i, j]
                                
                                if p_adj < alpha:
                                    print(f"    - {method1} vs. {method2}: p_adj = {p_adj:.4e}")
                                    significant_pairs += 1
                                    
                        if significant_pairs == 0:
                            print("    - None (after p-value correction)")

            except ValueError as e:
                print(f"  > Error during statistical test: {e}. This can happen if a group has identical values.")

        # --- 3. Plotting ---
        plt.figure(figsize=(14, 8))
        ax = plt.gca() # Get current axes to add text later
        
        if plot_type == 'box':
            sns.boxplot(
                data=data_for_plot, 
                x='method', 
                y=metric, 
                order=plot_order_arg, 
                palette=palette,
                hue='method',
                legend=False,
                ax=ax # Pass the axes
            )

        elif plot_type == 'violin':
            sns.violinplot(
                data=data_for_plot, 
                x='method', 
                y=metric, 
                order=plot_order_arg, 
                palette=palette,
                hue='method',
                legend=False,
                ax=ax # Pass the axes
            )
        else:
            print(f"Warning: Unknown plot_type '{plot_type}'. Skipping plot for {metric}.")
            plt.close() # Close the empty figure
            continue
            
        # --- 4. [NEW] Add data labels if requested ---
        if plot_type == 'box' and show_labels:
            print("   > Adding data labels to boxplot...")
            
            # Calculate stats for all groups at once
            stats_df = data_for_plot.groupby('method')[metric].quantile([0.25, 0.5, 0.75]).unstack()
            stats_df = stats_df.reindex(plot_order_arg) # Ensure correct order
            
            # Get y-axis range to calculate a dynamic offset
            ylim = ax.get_ylim()
            y_offset = (ylim[1] - ylim[0]) * 0.015 # 1.5% of y-axis height

            # Iterate over each box (method) by its x-position
            for x_pos, method_name in enumerate(plot_order_arg):
                if method_name not in stats_df.index:
                    continue
                    
                # Get the pre-calculated stats
                q1 = stats_df.loc[method_name, 0.25]
                median = stats_df.loc[method_name, 0.5]
                q3 = stats_df.loc[method_name, 0.75]
                
                if pd.isna(q1) or pd.isna(median) or pd.isna(q3):
                    continue

                # Add text labels
                # Q3: Above the box
                ax.text(x_pos, q3 + y_offset, f'{q3:.1f}', ha='center', va='bottom', fontsize=8, color='black')
                
                # Median: Above the median line
                ax.text(x_pos, median + y_offset, f'{median:.1f}', ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

                # Q1: Below the box
                ax.text(x_pos, q1 - y_offset, f'{q1:.1f}', ha='center', va='top', fontsize=8, color='black')

        # --- 5. Finalize Plot ---
        plot_title = f"Overall Method Performance: {metric}"
        y_label = f"{metric}"
        if "_deg" in metric:
            y_label += " (Degrees)"
        elif "_rad" in metric:
            y_label += " (Radians)"

        ax.set_title(plot_title, fontsize=16)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlabel("Method", fontsize=12)
        
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_plots:
            filename = f"method_performance_{metric.lower()}_{plot_type}.png"
            if method_filter:
                filename = f"filtered_{filename}"
            save_path = os.path.join("plots", filename)
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        
        if show_plots:
            plt.show()
            
        plt.close() # Close the figure to free memory


# --- [UPDATED FUNCTION] ---
def plot_interaction_heatmap(
    summary_df: pd.DataFrame,
    metrics: List[str],
    interaction_factor: str,
    method_filter: Optional[List[str]] = None,
    cmap: str = 'Reds',
    annot: bool = True,
    fmt: str = ".2f",
    run_stats: bool = False, # <-- New parameter
    alpha: float = 0.05,     # <-- New parameter
    show_plots: bool = False,
    save_plots: bool = True
) -> None:
    """
    Generates and saves heatmaps for specified metrics to compare
    method performance broken down by an interaction factor (e.g., 'joint', 'subject').
    
    [NEW] If run_stats=True, performs a 1-Way (ANOVA or Kruskal-Wallis)
    test for each level of the interaction_factor, comparing all methods.

    Args:
        ... [all previous args] ...
        run_stats (bool, optional): If True, run 1-way tests for each
                                  level of the interaction factor. Defaults to False.
        alpha (float, optional): Significance level for stats. Defaults to 0.05.
    """
    print(f"\n--- Generating interaction heatmaps for Method vs. {interaction_factor.title()} ---")

    # Ensure the plot directory exists
    if save_plots:
        os.makedirs("plots", exist_ok=True)

    # Reset index to make 'method' and other index levels available as columns
    data_for_plot_full = summary_df.reset_index()

    # Check if interaction_factor exists
    if interaction_factor not in data_for_plot_full.columns:
        print(f"Warning: Interaction factor '{interaction_factor}' not found in DataFrame columns.")
        print(f"Available columns are: {data_for_plot_full.columns.tolist()}")
        return

    # --- 1. Determine method_order and apply filter ---
    y_axis_order = None # This is the 'method_filter' or all methods
    data_for_plot = data_for_plot_full.copy()
    
    if method_filter:
        # Filter the DataFrame to only include methods from the list
        data_for_plot = data_for_plot[data_for_plot['method'].isin(method_filter)]
        if data_for_plot.empty:
            print(f"Warning: Filtering by methods {method_filter} resulted in an empty DataFrame. Skipping heatmaps.")
            return
        # Set the plot order (Y-axis) to be the order of the filter list
        y_axis_order = [m for m in method_filter if m in data_for_plot['method'].unique()]
    else:
        # Default to alphabetical order if no filter is provided
        y_axis_order = sorted(data_for_plot['method'].unique())

    # --- 2. Loop through each metric and plot ---
    for metric in metrics:
        if metric not in data_for_plot.columns:
            print(f"Warning: Metric '{metric}' not found. Skipping heatmap.")
            continue

        print(f"\n  > Plotting {metric}...")
        
        # --- [NEW] 3. Run Statistical Tests (if requested) ---
        if run_stats:
            print(f"--- Statistical Analysis for {metric} by {interaction_factor.title()} (alpha={alpha}) ---")
            
            # Get all unique levels of the interaction factor
            factor_levels = sorted(data_for_plot[interaction_factor].unique())
            
            if len(y_axis_order) <= 1:
                print("  > Skipping stats, only one method group to test.")
            else:
                # --- Loop over each factor level (e.g., each joint) ---
                for level in factor_levels:
                    print(f"\n--- Testing for {interaction_factor.title()} = {level} ---")
                    
                    # Filter data for only this level
                    df_level = data_for_plot[data_for_plot[interaction_factor] == level]
                    
                    # Create a list of data arrays for each method
                    groups_data = [
                        df_level[df_level['method'] == method][metric].dropna() 
                        for method in y_axis_order
                    ]
                    
                    # Filter out empty groups for this level
                    valid_groups_data = [g for g in groups_data if not g.empty]
                    valid_group_names = [name for name, g in zip(y_axis_order, groups_data) if not g.empty]

                    if len(valid_groups_data) <= 1:
                        print(f"  > Skipping stats, only one group with data for this level ({level}).")
                        continue

                    # a. Normality Check (Shapiro-Wilk)
                    all_groups_are_normal = True
                    for method_name, group_data in zip(valid_group_names, valid_groups_data):
                        if len(group_data) < 3:
                            print(f"  > {method_name}: Skipping (N={len(group_data)} < 3).")
                            all_groups_are_normal = False
                            continue
                        
                        try:
                            _, p_shapiro = stats.shapiro(group_data)
                            if p_shapiro < alpha:
                                all_groups_are_normal = False
                        except ValueError:
                            all_groups_are_normal = False
                    
                    if all_groups_are_normal:
                        print("  > All groups normal (or N<3). Using Parametric Test (ANOVA)...")
                    else:
                        print("  > At least one group non-normal. Using Non-Parametric Test (Kruskal-Wallis)...")

                    # b. Choose test based on normality
                    try:
                        if all_groups_are_normal:
                            # --- Parametric Test: ANOVA + Tukey ---
                            f_stat, p_anova = stats.f_oneway(*valid_groups_data)
                            print(f"  ANOVA (overall): p-value = {p_anova:.4e}")
                            
                            if p_anova >= alpha:
                                print("  > No significant difference found between methods for this level.")
                            else:
                                print("  > Significant difference detected. Running post-hoc (Tukey's HSD)...")
                                tukey_results = sp.posthoc_tukey(df_level, val_col=metric, group_col='method')
                                # Filter to only relevant methods
                                tukey_results = tukey_results.reindex(index=valid_group_names, columns=valid_group_names)

                                print("    Significant pairs (p_adj < alpha):")
                                significant_pairs = 0
                                for i in range(len(tukey_results.columns)):
                                    for j in range(i + 1, len(tukey_results.columns)):
                                        method1 = tukey_results.columns[i]
                                        method2 = tukey_results.columns[j]
                                        p_adj = tukey_results.iloc[i, j]
                                        
                                        if p_adj < alpha:
                                            print(f"      - {method1} vs. {method2}: p_adj = {p_adj:.4e}")
                                            significant_pairs += 1
                                if significant_pairs == 0:
                                    print("      - None (after p-value correction)")
                        
                        else:
                            # --- Non-Parametric Test: Kruskal-Wallis + Dunn ---
                            h_stat, p_kruskal = stats.kruskal(*valid_groups_data)
                            print(f"  Kruskal-Wallis H-test (overall): p-value = {p_kruskal:.4e}")
                            
                            if p_kruskal >= alpha:
                                print("  > No significant difference found between methods for this level.")
                            else:
                                print("  > Significant difference detected. Running post-hoc (Dunn's)...")
                                dunn_results = sp.posthoc_dunn(df_level, val_col=metric, group_col='method', p_adjust='holm')
                                # Filter to only relevant methods
                                dunn_results = dunn_results.reindex(index=valid_group_names, columns=valid_group_names)
                                
                                print("    Significant pairs (p_adj < alpha):")
                                significant_pairs = 0
                                for i in range(len(dunn_results.columns)):
                                    for j in range(i + 1, len(dunn_results.columns)):
                                        method1 = dunn_results.columns[i]
                                        method2 = dunn_results.columns[j]
                                        p_adj = dunn_results.iloc[i, j]
                                        
                                        if p_adj < alpha:
                                            print(f"      - {method1} vs. {method2}: p_adj = {p_adj:.4e}")
                                            significant_pairs += 1
                                if significant_pairs == 0:
                                    print("      - None (after p-value correction)")

                    except ValueError as e:
                        print(f"  > Error during statistical test: {e}. This can happen if a group has identical values.")
            
            print(f"\n--- End of Statistical Analysis for {metric} ---")

        # --- 4. Aggregate and pivot data for heatmap ---
        try:
            # Calculate the average metric for each Method and Factor combination
            grouped_data = data_for_plot.groupby(['method', interaction_factor])[metric].mean()
            
            # Pivot the data to create a 2D matrix for the heatmap
            # Index = method, Columns = interaction_factor
            heatmap_data = grouped_data.unstack(level=interaction_factor)

            # --- 5. Order the axes ---
            # Order Y-axis (methods) based on filter or alphabetically
            heatmap_data = heatmap_data.reindex(y_axis_order)
            
            # Order X-axis (factor) alphabetically for consistency
            x_axis_order = sorted(heatmap_data.columns.unique())
            heatmap_data = heatmap_data[x_axis_order]

            # --- 6. Plotting ---
            # Adjust figsize based on the number of columns (factors)
            fig_width = max(12, len(x_axis_order) * 1.5)
            fig_height = max(8, len(y_axis_order) * 0.8)
            plt.figure(figsize=(fig_width, fig_height))

            y_label = f"{metric}"
            if "_deg" in metric:
                y_label += " (Degrees)"
            elif "_rad" in metric:
                y_label += " (Radians)"

            ax = sns.heatmap(
                heatmap_data,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                linewidths=.5,
                cbar_kws={'label': f'Mean {y_label}'}
            )

            # --- 7. Finalize Plot ---
            plot_title = f"Mean {metric} by Method and {interaction_factor.title()}"
            ax.set_title(plot_title, fontsize=16)
            ax.set_xlabel(interaction_factor.title(), fontsize=12)
            ax.set_ylabel("Method", fontsize=12)
            
            # --- [FIXED] ---
            # Rotate x-axis labels if they are long
            # Removed ha='right' as it's not a valid keyword for ax.tick_params
            ax.tick_params(axis='x', rotation=45) 
            ax.tick_params(axis='y', rotation=0)
            
            plt.tight_layout()

            if save_plots:
                filename = f"interaction_heatmap_{metric.lower()}_vs_{interaction_factor.lower()}.png"
                save_path = os.path.join("plots", filename)
                plt.savefig(save_path)
                print(f"    Saved heatmap to {save_path}")

            if show_plots:
                plt.show()
                
            plt.close() # Close the figure to free memory
        
        except Exception as e:
            print(f"    Error plotting heatmap for {metric}: {e}")
            plt.close() # Ensure plot is closed on error


# --- Main execution ---
if __name__ == "__main__":
    
    # 1. Define the correct path
    stats_file_path = os.path.join("data", "data", "all_subject_statistics.pkl")

    # 2. Check if the file exists and load it
    if not os.path.exists(stats_file_path):
        print(f"Error: Statistics file not found at {stats_file_path}")
        print("Please run the data generation script first.")
        exit()
        
    try:
        print(f"Loading summary statistics from {stats_file_path}...")
        summary_stats_df = pd.read_pickle(stats_file_path)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        exit()
    
    show_plots = True
    save_plots = True # Set to True to save images

    print("\nGenerating performance plots...")
    # Define metrics to plot. Using Mean_deg as 'Bias'
    # --- [FIXED] ---
    # Changed 'Std_deg' to 'STD_deg' to match the column name in your DataFrame
    metrics_to_plot = ['RMSE_deg', 'MAE_deg', 'Mean_deg', 'STD_deg']
    
    # --- Find the outlier ---
    # Check if 'EKF' is in the index before trying to access it
    if 'EKF' in summary_stats_df.index.get_level_values('method'):
        ekf_df = summary_stats_df.xs('EKF', level='method')
        ekf_sorted = ekf_df.sort_values(by='RMSE_deg', ascending=False)
        print("\n--- Top 5 Largest 'EKF' Errors ---")
        print(ekf_sorted.head(5))
    else:
        print("\n--- 'EKF' method not found in data, skipping outlier check ---")


    # --- Define your methods and their desired order ---
    # NOTE: Your log shows 'Madgwick (Al Borno)' was not found in the data.
    # The code will still work, but it will only plot the methods that *are* found.
    my_methods_in_order = [
        'EKF',
        'Madgwick (Al Borno)', # This method seems to be missing from your data
        'Mag Free',
        'Never Project',
        'Cascade', 
    ]
    
    my_palette = "Set2"
    
    print("\n--- Generating FILTERED Box Plots (in custom order) ---")
    plot_method_performance(
        summary_df=summary_stats_df,
        metrics=metrics_to_plot,
        plot_type='box',
        method_filter=my_methods_in_order,
        palette=my_palette,
        show_plots=show_plots,
        save_plots=save_plots,
        run_stats=True,        # <-- Make sure stats are on
        show_labels=True       # <-- SET TO TRUE TO SHOW LABELS
    )
    
    # --- [NEW] Call the interaction heatmap function ---
    print("\n--- Generating Interaction Heatmaps (in custom order) ---")
    
    # 1. Method vs. Joint
    plot_interaction_heatmap(
        summary_df=summary_stats_df,
        metrics=['RMSE_deg', 'MAE_deg'], # Plot mean RMSE and MAE
        # --- [FIXED] ---
        # Changed 'joint' to 'joint_name' to match your DataFrame column
        interaction_factor='joint_name',     # X-axis 
        method_filter=my_methods_in_order, # Y-axis
        cmap='Reds', # 'Reds' is good for error (high = bad = dark red)
        run_stats=True,                  # <-- STATS ARE NOW ON
        show_plots=show_plots,
        save_plots=save_plots
    )
    
    # 2. Method vs. Subject
    plot_interaction_heatmap(
        summary_df=summary_stats_df,
        metrics=['RMSE_deg'], # Just RMSE is probably fine for this one
        interaction_factor='subject',    # X-axis
        method_filter=my_methods_in_order, # Y-axis
        cmap='Reds',
        run_stats=True,                  # <-- STATS ARE NOW ON
        show_plots=show_plots,
        save_plots=save_plots
    )

    # 3. Method vs. Axis
    plot_interaction_heatmap(
        summary_df=summary_stats_df,
        metrics=['RMSE_deg'], # Just RMSE
        interaction_factor='axis',       # X-axis
        method_filter=my_methods_in_order, # Y-axis
        cmap='Reds',
        run_stats=True,                  # <-- STATS ARE NOW ON
        show_plots=show_plots,
        save_plots=save_plots
    )

    print("\n--- All plotting complete ---")