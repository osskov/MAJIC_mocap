import os
import argparse  # Import argparse
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import seaborn as sns
from src.toolchest.PlateTrial import PlateTrial
from generate_method_orientation_sto_files import JOINT_SEGMENT_DICT

TRIAL_TYPES = ['walking', 'complexTasks']
SUBJECTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
FILE_NAME = "joint_observability_summary.csv"  # Define file name as a constant

def get_bilateral_parts(joint_name: str) -> (str, str):
    """Helper function to split a joint name into side and base."""
    if joint_name.startswith('L_'):
        return 'L', joint_name[2:]
    elif joint_name.startswith('R_'):
        return 'R', joint_name[2:]
    else:
        # Handle non-bilateral joints
        return 'N/A', joint_name

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Load or calculate joint observability metrics.")
    # Changed flag: Now we check for --recalculate. Default is to load.
    parser.add_argument(
        '--recalculate', 
        action='store_true', 
        help=f"Force recalculation of data and overwrite {FILE_NAME}."
    )
    args = parser.parse_args()

    joint_observability = {}
    loaded_successfully = False # Flag to track if loading worked

    # --- Data Loading / Calculation ---
    
    # 1. Try to load by default, unless --recalculate is specified
    if not args.recalculate and os.path.exists(FILE_NAME):
        print(f"Loading data from {FILE_NAME}...")
        try:
            observability_df = pd.read_csv(FILE_NAME)
            # Convert wide DataFrame back to dictionary of lists
            joint_observability = {
                col: observability_df[col].dropna().tolist() 
                for col in observability_df.columns
            }
            if not joint_observability:
                print(f"Warning: {FILE_NAME} is empty. Recalculating...")
            else:
                 # Ensure keys match the canonical dictionary order
                joint_observability = {key: joint_observability.get(key, []) for key in JOINT_SEGMENT_DICT.keys()}
                loaded_successfully = True # Set flag on success

        except pd.errors.EmptyDataError:
            print(f"Warning: {FILE_NAME} is empty. Recalculating...")
        except Exception as e:
            print(f"Error loading {FILE_NAME}: {e}. Recalculating...")

    # 2. Calculate if --recalculate was passed OR loading failed
    if args.recalculate or not loaded_successfully:
        if args.recalculate:
            print("Recalculate flag set. Forcing data calculation...")
        else:
            print(f"Could not load data from {FILE_NAME}. Calculating data...")
            
        joint_observability = {joint: [] for joint in JOINT_SEGMENT_DICT.keys()}

        for subject_num in SUBJECTS:
            for trial_type in TRIAL_TYPES:
                subject_id = f"Subject{subject_num}"
                try:
                    plate_trials = PlateTrial.load_trial_from_folder(f"data/data/{subject_id}/{trial_type}")

                except FileNotFoundError:
                    print(f"Could not find data for {trial_type} {subject_id}")
                    continue

                for joint, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
                    parent_plate = next((p for p in plate_trials if parent_name in p.name), None)
                    child_plate = next((p for p in plate_trials if child_name in p.name), None)

                    if not parent_plate or not child_plate:
                        print(f"Warning: Could not find plates for joint '{joint}' in {subject_id} {trial_type}.")
                        continue
                    
                    # Assuming get_joint_center is a method of WorldTrace
                    parent_joint_center_offset, child_joint_center_offset, error = parent_plate.world_trace.get_joint_center(child_plate.world_trace)

                    if np.mean(np.linalg.norm(error, axis=1)) > 0.05:
                        print(f"Warning: High joint center error for {joint} in {subject_id} {trial_type}.")

                    projected_parent_imu = parent_plate.project_imu_trace(parent_joint_center_offset)
                    projected_child_imu = child_plate.project_imu_trace(child_joint_center_offset)

                    da_parent = np.diff(projected_parent_imu.acc, axis=0) + np.cross(projected_parent_imu.gyro[1:], projected_parent_imu.acc[1:])
                    da_child = np.diff(projected_child_imu.acc, axis=0) + np.cross(projected_child_imu.gyro[1:], projected_child_imu.acc[1:])
                    
                    o_parent = np.cross(projected_parent_imu.acc[1:], da_parent)
                    o_child = np.cross(projected_child_imu.acc[1:], da_child)
                    
                    observability_metric = np.minimum(np.linalg.norm(o_parent, axis=1),
                                                    np.linalg.norm(o_child, axis=1))
                    
                    mean_observability = np.mean(observability_metric)
                    joint_observability[joint].append(mean_observability)

        # Save the observability data to a CSV file
        observability_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in joint_observability.items()]))
        observability_df.to_csv(FILE_NAME, index=False)
        print(f"Observability data calculated and saved to {FILE_NAME}")


    # --- Generate Plots ---

    print("Generating plots...")

    # --- Data Preparation for Plotting ---
    
    # Convert dictionary to a long-form DataFrame for Seaborn
    df_data = []
    for joint, values in joint_observability.items():
        if not values: # Handle cases where a joint had no data
             print(f"Warning: No data found for joint '{joint}'. It will not be plotted.")
             continue
        for value in values:
            df_data.append({'Joint': joint, 'Mean Observability': value})
    
    if not df_data:
        print("Error: No data available to plot. Exiting.")
        return

    df = pd.DataFrame(df_data)

    # Add 'Side' (L/R) and 'Joint_Base' (e.g., 'hip') columns
    df[['Side', 'Joint_Base']] = df['Joint'].apply(lambda x: pd.Series(get_bilateral_parts(x)))

    # Get unique base joints, maintaining order from JOINT_SEGMENT_DICT
    joint_order = []
    for j in JOINT_SEGMENT_DICT.keys():
        base = get_bilateral_parts(j)[1]
        if base not in joint_order:
            joint_order.append(base)

    # --- Bar plot for Mean and Standard Deviation (Bilateral) ---
    plt.figure(figsize=(14, 8))
    sns.barplot(
        x='Joint_Base', 
        y='Mean Observability', 
        hue='Side', 
        data=df, 
        order=joint_order,
        errorbar='sd',  # Use standard deviation for error bars
        capsize=0.1
    )
    plt.ylabel('Mean Observability')
    plt.title('Mean and Standard Deviation of Joint Observability (Bilateral)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Side')
    plt.tight_layout()
    plt.show()

    # --- Strip plot for Median and IQR (Bilateral) ---
    plt.figure(figsize=(14, 8))
    
    # Plot the boxplot first to set up categories and hues
    sns.boxplot(
        x='Joint_Base', 
        y='Mean Observability', 
        hue='Side',
        data=df, 
        order=joint_order,
        showfliers=False, 
        boxprops=dict(alpha=0.4), 
        whiskerprops=dict(alpha=0.4), 
        capprops=dict(alpha=0.4),
        medianprops=dict(color='black', linewidth=2),
        dodge=True
    )
    
    # Overlay the stripplot
    sns.stripplot(
        x='Joint_Base', # Corrected from 'Joint_Bases'
        y='Mean Observability', 
        hue='Side', 
        data=df, 
        order=joint_order,
        jitter=True, 
        alpha=0.7, 
        dodge=True, # This splits the L/R points
        legend=False # Avoid duplicate legend
    )
    
    plt.ylabel('Mean Observability')
    plt.title('Distribution of Mean Joint Observability (Bilateral)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Handle legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Keep only unique legend entries (stripplot can add duplicates)
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Side')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()