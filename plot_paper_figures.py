from typing import List
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Plotting Functions (if any) would go here ---

def plot_method_performance(summary_df: pd.DataFrame, 
                            metrics: List[str], 
                            plot_type: str = 'box',
                            show_plots: bool = False,
                            save_plots: bool = True) -> None:
    """
    Generates and saves box plots or violin plots for specified metrics
    to compare overall method performance.

    Args:
        summary_df (pd.DataFrame): The summary statistics DataFrame.
                                   Expected to have a MultiIndex including
                                   'method' and columns for each metric.
        metrics (List[str]): A list of metric column names to plot 
                             (e.g., ['RMSE_deg', 'MAE_deg']).
        plot_type (str, optional): Type of plot to generate. 
                                   'box' (default) or 'violin'.
    """
    print(f"--- Generating {plot_type} plots for {', '.join(metrics)} ---")
    
    # Ensure the plot directory exists
    if save_plots:
        os.makedirs("plots", exist_ok=True) # <-- ADDED
        
    # Reset index to make 'method' and other index levels available as columns
    # This is correct for your summary_df
    data_for_plot = summary_df.reset_index()
    
    # Get a sorted list of unique method names
    # This ensures a consistent order on the x-axis
    method_order = sorted(data_for_plot['method'].unique())

    for metric in metrics:
        if metric not in data_for_plot.columns:
            print(f"Warning: Metric '{metric}' not found in summary DataFrame. Skipping plot.")
            continue
        
        plt.figure(figsize=(14, 8))
        
        if plot_type == 'box':
            sns.boxplot(data=data_for_plot, x='method', y=metric, order=method_order)
        elif plot_type == 'violin':
            sns.violinplot(data=data_for_plot, x='method', y=metric, order=method_order)
        else:
            print(f"Warning: Unknown plot_type '{plot_type}'. Skipping plot for {metric}.")
            continue
            
        plot_title = f"Overall Method Performance: {metric}"
        y_label = f"{metric}"
        if "_deg" in metric:
            y_label += " (Degrees)"
        elif "_rad" in metric:
            y_label += " (Radians)"

        plt.title(plot_title, fontsize=16)
        plt.ylabel(y_label, fontsize=12)
        plt.xlabel("Method", fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_plots:
            # Save the figure
            save_path = os.path.join("plots", f"method_performance_{metric.lower()}_{plot_type}.png")
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        
        if show_plots:
            plt.show()
            
        plt.close() # Close the figure to free memory

if __name__ == "__main__":
    
    # --- UPDATED SECTION ---
    
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
    
    # --- END UPDATED SECTION ---

    show_plots = True
    save_plots = False

    print("\nGenerating performance plots...")
    # Define metrics to plot. Using Mean_deg as 'Bias'
    metrics_to_plot = ['RMSE_deg', 'MAE_deg', 'Mean_deg', 'STD_deg']
    
    plot_method_performance(
        summary_df=summary_stats_df,
        metrics=metrics_to_plot,
        plot_type='box',
        show_plots=show_plots,
        save_plots=save_plots
    )
    
    # You could also generate violin plots
    plot_method_performance(
        summary_df=summary_stats_df,
        metrics=metrics_to_plot,
        plot_type='violin',
        show_plots=show_plots,
        save_plots=save_plots
    )
    
    print("--- Plotting complete ---")