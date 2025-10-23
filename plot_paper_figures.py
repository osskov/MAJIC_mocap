from typing import List, Dict, Union, Optional
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scikit_posthocs as sp

# --- Plotting Functions ---

# --- [UPDATED FUNCTION] ---
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
    
    [NEW] Statistics are now run using the Friedman test, which
    accounts for the dependent-sample nature of the data (multiple
    methods tested on the same subject/joint/axis 'block').
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
    stats_data_df_all = data_for_plot[data_for_plot['method'].isin(plot_order_arg)]

    for metric in metrics:
        if metric not in stats_data_df_all.columns:
            print(f"Warning: Metric '{metric}' not found in summary DataFrame. Skipping plot.")
            continue
            
        # --- 2. [UPDATED] Run statistical tests (Friedman + Wilcoxon) ---
        if run_stats:
            print(f"\n--- Statistical Analysis for {metric} (alpha={alpha}) ---")
            print("--- a. Dependent-Sample Test (Friedman Test) ---")
            
            # Make a copy to avoid SettingWithCopyWarning
            stats_data_df = stats_data_df_all.copy()

            try:
                # --- b. Pivot data for repeated-measures test ---
                block_cols = [
                    col for col in ['trial_type', 'joint_name', 'subject', 'axis'] 
                    if col in stats_data_df.columns
                ]
                if not block_cols:
                    print("  > Error: Could not find block columns (subject, joint_name, etc.) to run Friedman test.")
                    continue

                # Create a unique block_id for the post-hoc test
                stats_data_df['block_id'] = stats_data_df[block_cols].apply(
                    lambda row: '_'.join(row.values.astype(str)), axis=1
                )

                pivot_df = stats_data_df.pivot_table(
                    index='block_id', # Use new unique block_id
                    columns='method', 
                    values=metric
                )
                
                # Friedman test requires complete blocks (no NaNs)
                pivot_df_clean = pivot_df.dropna()
                
                # Filter for methods we actually have in the cleaned data
                valid_methods = [m for m in plot_order_arg if m in pivot_df_clean.columns]

                if pivot_df_clean.shape[0] < 2:
                    print(f"  > Skipping stats: Insufficient complete blocks (N={pivot_df_clean.shape[0]}) for Friedman test.")
                    continue
                if len(valid_methods) < 2:
                    print(f"  > Skipping stats: Insufficient method groups (N={len(valid_methods)}) for Friedman test.")
                    continue

                print(f"  > Using {pivot_df_clean.shape[0]} complete blocks (out of {pivot_df.shape[0]} total) for {len(valid_methods)} methods.")

                # Get the data columns for the test
                groups_for_test = [pivot_df_clean[col] for col in valid_methods]
                
                f_stat, p_friedman = stats.friedmanchisquare(*groups_for_test)
                print(f"  Friedman Test (overall): p-value = {p_friedman:.4e}")

                # --- c. Post-hoc testing ---
                if p_friedman >= alpha:
                    print("  > No significant difference found between any groups (p >= alpha).")
                else:
                    print("  > Significant difference detected. Running post-hoc (Wilcoxon Signed-Rank)...")
                    
                    # --- [FIX] Pass the WIDE-FORMAT DataFrame directly ---
                    # This is compatible with older scikit-posthocs versions
                    # that do not have the 'block_col' argument.
                    wilcoxon_results = sp.posthoc_wilcoxon(
                        pivot_df_clean[valid_methods], 
                        p_adjust='holm'
                    )
                    
                    # Filter for only the methods we're plotting
                    wilcoxon_results = wilcoxon_results.loc[valid_methods, valid_methods]

                    print("  Significant pairs (p_adj < alpha):")
                    significant_pairs = 0
                    # Iterate over the upper triangle of the matrix to avoid repeats
                    for i in range(len(wilcoxon_results.columns)):
                        for j in range(i + 1, len(wilcoxon_results.columns)):
                            method1 = wilcoxon_results.columns[i]
                            method2 = wilcoxon_results.columns[j]
                            p_adj = wilcoxon_results.iloc[i, j]
                            
                            if p_adj < alpha:
                                print(f"    - {method1} vs. {method2}: p_adj = {p_adj:.4e}")
                                significant_pairs += 1
                                
                    if significant_pairs == 0:
                        print("    - None (after p-value correction)")
            
            except ValueError as e:
                print(f"  > Error during statistical test: {e}")
            except AttributeError as e:
                 print(f"  > Error during post-hoc test: {e}. This can happen with scikit-posthocs versioning.")


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
            # ... (violin plot code, unchanged) ...
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
            # ... (label code, unchanged) ...
            stats_df = data_for_plot.groupby('method')[metric].quantile([0.25, 0.5, 0.75]).unstack()
            stats_df = stats_df.reindex(plot_order_arg)
            ylim = ax.get_ylim()
            y_offset = (ylim[1] - ylim[0]) * 0.015
            for x_pos, method_name in enumerate(plot_order_arg):
                if method_name not in stats_df.index: continue
                q1 = stats_df.loc[method_name, 0.25]
                median = stats_df.loc[method_name, 0.5]
                q3 = stats_df.loc[method_name, 0.75]
                if pd.isna(q1) or pd.isna(median) or pd.isna(q3): continue
                ax.text(x_pos, q3 + y_offset, f'{q3:.1f}', ha='center', va='bottom', fontsize=8, color='black')
                ax.text(x_pos, median + y_offset, f'{median:.1f}', ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
                ax.text(x_pos, q1 - y_offset, f'{q1:.1f}', ha='center', va='top', fontsize=8, color='black')

        # --- 5. Finalize Plot ---
        plot_title = f"Overall Method Performance: {metric}"
        y_label = f"{metric}"
        if "_deg" in metric: y_label += " (Degrees)"
        elif "_rad" in metric: y_label += " (Radians)"

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
    run_stats: bool = False, 
    alpha: float = 0.05,     
    show_plots: bool = False,
    save_plots: bool = True
) -> None:
    """
    Generates and saves heatmaps for specified metrics to compare
    method performance broken down by an interaction factor.
    
    [NEW] If run_stats=True, performs a Friedman test for each
    level of the interaction_factor, comparing all methods.
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
        data_for_plot = data_for_plot[data_for_plot['method'].isin(method_filter)]
        if data_for_plot.empty:
            print(f"Warning: Filtering by methods {method_filter} resulted in an empty DataFrame. Skipping heatmaps.")
            return
        y_axis_order = [m for m in method_filter if m in data_for_plot['method'].unique()]
    else:
        y_axis_order = sorted(data_for_plot['method'].unique())

    # --- 2. Loop through each metric and plot ---
    for metric in metrics:
        if metric not in data_for_plot.columns:
            print(f"Warning: Metric '{metric}' not found. Skipping heatmap.")
            continue

        print(f"\n  > Plotting {metric}...")
        
        # --- [NEW] 3. Run Statistical Tests (Friedman + Wilcoxon) ---
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
                    print("  > Using Dependent-Sample Test (Friedman Test)...")
                    
                    try:
                        # Filter data for only this level
                        df_level = data_for_plot[data_for_plot[interaction_factor] == level].copy() # Use .copy()
                        
                        # --- b. Pivot data for repeated-measures test ---
                        all_block_cols = ['trial_type', 'joint_name', 'subject', 'axis']
                        block_cols = [
                            col for col in all_block_cols 
                            if col in df_level.columns and col != interaction_factor
                        ]
                        
                        if not block_cols:
                            print("  > Error: Could not find block columns to run Friedman test.")
                            continue

                        # Create a unique block_id for the post-hoc test
                        df_level['block_id'] = df_level[block_cols].apply(
                            lambda row: '_'.join(row.values.astype(str)), axis=1
                        )

                        pivot_df = df_level.pivot_table(
                            index='block_id', # Use new unique block_id
                            columns='method', 
                            values=metric
                        )
                        
                        # Friedman test requires complete blocks (no NaNs)
                        pivot_df_clean = pivot_df.dropna()

                        # Filter for methods we actually have in the cleaned data
                        valid_methods = [m for m in y_axis_order if m in pivot_df_clean.columns]

                        if pivot_df_clean.shape[0] < 2:
                            print(f"  > Skipping stats: Insufficient complete blocks (N={pivot_df_clean.shape[0]}) for Friedman test.")
                            continue
                        if len(valid_methods) < 2:
                            print(f"  > Skipping stats: Insufficient method groups (N={len(valid_methods)}) for Friedman test.")
                            continue

                        print(f"  > Using {pivot_df_clean.shape[0]} complete blocks (out of {pivot_df.shape[0]} total) for {len(valid_methods)} methods.")
                        
                        # Get the data columns for the test
                        groups_for_test = [pivot_df_clean[col] for col in valid_methods]
                        
                        f_stat, p_friedman = stats.friedmanchisquare(*groups_for_test)
                        print(f"  Friedman Test (overall): p-value = {p_friedman:.4e}")

                        # --- c. Post-hoc testing ---
                        if p_friedman >= alpha:
                            print("  > No significant difference found between methods for this level.")
                        else:
                            print("  > Significant difference detected. Running post-hoc (Wilcoxon Signed-Rank)...")
                            
                            # --- [FIX] Pass the WIDE-FORMAT DataFrame directly ---
                            # This is compatible with older scikit-posthocs versions
                            wilcoxon_results = sp.posthoc_wilcoxon(
                                pivot_df_clean[valid_methods], 
                                p_adjust='holm'
                            )
                            
                            # Filter for only the methods we're plotting
                            wilcoxon_results = wilcoxon_results.loc[valid_methods, valid_methods]

                            print("    Significant pairs (p_adj < alpha):")
                            significant_pairs = 0
                            # Iterate over the upper triangle of the matrix to avoid repeats
                            for i in range(len(wilcoxon_results.columns)):
                                for j in range(i + 1, len(wilcoxon_results.columns)):
                                    method1 = wilcoxon_results.columns[i]
                                    method2 = wilcoxon_results.columns[j]
                                    p_adj = wilcoxon_results.iloc[i, j]
                                    
                                    if p_adj < alpha:
                                        print(f"      - {method1} vs. {method2}: p_adj = {p_adj:.4e}")
                                        significant_pairs += 1
                                if significant_pairs == 0:
                                    print("      - None (after p-value correction)")

                    except ValueError as e:
                        print(f"  > Error during statistical test: {e}. This can happen if a group has identical values.")
                    except AttributeError as e:
                        print(f"  > Error during post-hoc test: {e}. This can happen with scikit-posthocs versioning.")
            
            print(f"\n--- End of Statistical Analysis for {metric} ---")

        # --- 4. Aggregate and pivot data for heatmap ---
        try:
            # ... (Heatmap aggregation code, unchanged) ...
            grouped_data = data_for_plot.groupby(['method', interaction_factor])[metric].mean()
            heatmap_data = grouped_data.unstack(level=interaction_factor)

            # --- 5. Order the axes ---
            # ... (Heatmap ordering code, unchanged) ...
            heatmap_data = heatmap_data.reindex(y_axis_order)
            x_axis_order = sorted(heatmap_data.columns.unique())
            heatmap_data = heatmap_data[x_axis_order]

            # --- 6. Plotting ---
            # ... (Heatmap plotting code, unchanged) ...
            fig_width = max(12, len(x_axis_order) * 1.5)
            fig_height = max(8, len(y_axis_order) * 0.8)
            plt.figure(figsize=(fig_width, fig_height))
            y_label = f"{metric}"
            if "_deg" in metric: y_label += " (Degrees)"
            elif "_rad" in metric: y_label += " (Radians)"
            ax = sns.heatmap(
                heatmap_data,
                annot=annot,
                fmt=fmt,
                cmap=cmap,
                linewidths=.5,
                cbar_kws={'label': f'Mean {y_label}'}
            )

            # --- 7. Finalize Plot ---
            # ... (Heatmap finalize code, unchanged) ...
            plot_title = f"Mean {metric} by Method and {interaction_factor.title()}"
            ax.set_title(plot_title, fontsize=16)
            ax.set_xlabel(interaction_factor.title(), fontsize=12)
            ax.set_ylabel("Method", fontsize=12)
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
            plt.close()
        
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
    metrics_to_plot = ['RMSE_deg', 'MAE_deg', 'Mean_deg', 'STD_deg']
    
    # --- Find the outlier ---
    if 'EKF' in summary_stats_df.index.get_level_values('method'):
        ekf_df = summary_stats_df.xs('EKF', level='method')
        ekf_sorted = ekf_df.sort_values(by='RMSE_deg', ascending=False)
        print("\n--- Top 5 Largest 'EKF' Errors ---")
        print(ekf_sorted.head(5))
    else:
        print("\n--- 'EKF' method not found in data, skipping outlier check ---")


    # --- Define your methods and their desired order ---
    my_methods_in_order = [
        'EKF',
        'Madgwick (Al Borno)', 
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
        run_stats=True,        
        show_labels=True       
    )
    
    # --- [NEW] Call the interaction heatmap function ---
    print("\n--- Generating Interaction Heatmaps (in custom order) ---")
    
    # 1. Method vs. Joint
    plot_interaction_heatmap(
        summary_df=summary_stats_df,
        metrics=metrics_to_plot, # Using all metrics
        interaction_factor='joint_name',     
        method_filter=my_methods_in_order, 
        cmap='Reds',
        run_stats=True,                  
        show_plots=show_plots,
        save_plots=save_plots
    )
    
    # 2. Method vs. Subject
    plot_interaction_heatmap(
        summary_df=summary_stats_df,
        metrics=metrics_to_plot, # Using all metrics
        interaction_factor='subject',    
        method_filter=my_methods_in_order, 
        cmap='Reds',
        run_stats=True,                  
        show_plots=show_plots,
        save_plots=save_plots
    )

    # 3. Method vs. Axis
    plot_interaction_heatmap(
        summary_df=summary_df,
        metrics=metrics_to_plot, # Using all metrics
        interaction_factor='axis',       
        method_filter=my_methods_in_order, 
        cmap='Reds',
        run_stats=True,                  
        show_plots=show_plots,
        save_plots=save_plots
    )

    # 4. Method vs. Trial Type
    plot_interaction_heatmap(
        summary_df=summary_stats_df,
        metrics=metrics_to_plot, # Using all metrics
        interaction_factor='trial_type', 
        method_filter=my_methods_in_order,
        cmap='Reds',
        run_stats=True,                  
        show_plots=show_plots,
        save_plots=save_plots
    )
    print("\n--- All plotting complete ---")