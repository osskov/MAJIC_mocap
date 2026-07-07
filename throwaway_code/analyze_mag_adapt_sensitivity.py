import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from src.toolchest.PlateTrial import PlateTrial
from generate_method_orientation_sto_files import _get_joint_orientations_from_plate_trials_, JOINT_SEGMENT_DICT
import warnings
import concurrent.futures
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration
THRESHOLDS = [0, 25, 50, 75, 100, 125, 150, 175, 200, 1000000]
MAX_WORKERS = 10  # Use 10 cores out of 12 for the pool

def process_threshold(args):
    threshold, parent_trial_orig, child_trial_orig, observability_metric, R_gt_matrices, subject, activity, joint_name = args
    parent_trial = parent_trial_orig.copy()
    child_trial = child_trial_orig.copy()
    
    try:
        R_pc_list = _get_joint_orientations_from_plate_trials_(
            parent_trial, 
            child_trial, 
            condition='mag adapt',
            mag_adapt_threshold=threshold,
            project_imu=False,
            precomputed_observability_metric=observability_metric
        )
        
        gt_rotations = Rotation.from_matrix(R_gt_matrices)
        est_rotations = Rotation.from_matrix(R_pc_list)
        error_rotations = gt_rotations.inv() * est_rotations
        error_magnitudes = error_rotations.magnitude()
        rmse = np.sqrt(np.mean(error_magnitudes**2))
        
        return {
            'Subject': subject,
            'Activity': activity,
            'Joint': joint_name,
            'Threshold': threshold,
            'RMSE': np.rad2deg(rmse)
        }
        
    except Exception as e:
        return {'error': f"Failed for joint {joint_name}, threshold {threshold}: {e}"}

def main():
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    results = []
    
    data_dir = os.path.abspath("data")
    subjects = [d for d in os.listdir(data_dir) if d.startswith("Subject") and os.path.isdir(os.path.join(data_dir, d))]
    subjects.sort()

    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        activities = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]
        
        for activity in activities:
            print(f"\n--- Loading data for {subject}, Activity: {activity} ---")
            subject_activity_folder = os.path.join(subject_dir, activity)
            
            try:
                plate_trials = PlateTrial.load_trial_from_folder(
                    subject_activity_folder,
                    align_plate_trials=True
                )
            except Exception as e:
                print(f"Failed to load data for {subject} {activity}: {e}")
                continue

            # # Trim to 2000 frames to make the sensitivity analysis faster
            # num_frames = 2000
            # if len(plate_trials) > 0 and len(plate_trials[0]) > num_frames:
            #     plate_trials = [p[:num_frames] for p in plate_trials]

            activity_tasks = []

            for joint_name, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
                parent_trial_orig = next((p for p in plate_trials if parent_name in p.name), None)
                child_trial_orig = next((p for p in plate_trials if child_name in p.name), None)

                if not parent_trial_orig or not child_trial_orig:
                    print(f"  Skipping joint {joint_name} (Missing parent or child trial)")
                    continue

                # Calculate Marker Ground Truth Joint Rotations
                R_gt_matrices = [R_wp.T @ R_wc for R_wp, R_wc in zip(parent_trial_orig.world_trace.rotations, child_trial_orig.world_trace.rotations)]

                # Pre-project IMU traces
                parent_joint_center_offset, child_joint_center_offset, _ = parent_trial_orig.world_trace.get_joint_center(child_trial_orig.world_trace)
                parent_trial_orig.imu_trace = parent_trial_orig.project_imu_trace(parent_joint_center_offset)
                child_trial_orig.imu_trace = child_trial_orig.project_imu_trace(child_joint_center_offset)

                # Pre-calculate observability metric
                da_parent = np.diff(parent_trial_orig.imu_trace.acc, axis=0) + np.cross(parent_trial_orig.imu_trace.gyro[1:], parent_trial_orig.imu_trace.acc[1:])
                da_child = np.diff(child_trial_orig.imu_trace.acc, axis=0) + np.cross(child_trial_orig.imu_trace.gyro[1:], child_trial_orig.imu_trace.acc[1:])
                o_parent = np.cross(parent_trial_orig.imu_trace.acc[1:], da_parent)
                o_child = np.cross(child_trial_orig.imu_trace.acc[1:], da_child)
                observability_metric = np.minimum(np.linalg.norm(o_parent, axis=1), np.linalg.norm(o_child, axis=1))
                observability_metric = np.concatenate(([0.0], observability_metric))

                for threshold in THRESHOLDS:
                    activity_tasks.append((threshold, parent_trial_orig, child_trial_orig, observability_metric, R_gt_matrices, subject, activity, joint_name))

            if activity_tasks:
                print(f"  Running Sensitivity Analysis ({len(activity_tasks)} tasks) in parallel...")
                with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    for res in tqdm(executor.map(process_threshold, activity_tasks), total=len(activity_tasks), desc="Processing thresholds"):
                        if 'error' in res:
                            print(f"    {res['error']}")
                        else:
                            results.append(res)

    # Process and Plot the results using Plotly
    df = pd.DataFrame(results)
    if df.empty:
        print("No valid results computed.")
        return

    joints = list(JOINT_SEGMENT_DICT.keys())
    # Only keep joints that actually have data
    joints = [j for j in joints if j in df['Joint'].unique()]
    num_joints = len(joints)
    
    cols = min(3, num_joints)
    rows = (num_joints + cols - 1) // cols
    
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=joints, shared_xaxes=True)

    unique_subjects = sorted(df['Subject'].unique())
    unique_activities = sorted(df['Activity'].unique())
    
    colors = px.colors.qualitative.Plotly
    # Use different markers for activities
    markers = ['circle', 'triangle-up', 'square', 'diamond', 'cross', 'x']

    added_legends = set()

    for i, joint in enumerate(joints):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        df_joint = df[df['Joint'] == joint]
        
        for s_idx, subj in enumerate(unique_subjects):
            color = colors[s_idx % len(colors)]
            df_sub = df_joint[df_joint['Subject'] == subj]
            
            for a_idx, act in enumerate(unique_activities):
                marker = markers[a_idx % len(markers)]
                df_act = df_sub[df_sub['Activity'] == act]
                
                if df_act.empty:
                    continue
                
                df_act = df_act.sort_values(by='Threshold')
                name = f"{subj} - {act}"
                
                # Only show each Subject-Activity combination in the legend once
                showlegend = name not in added_legends
                if showlegend:
                    added_legends.add(name)
                
                fig.add_trace(go.Scatter(
                    x=df_act['Threshold'],
                    y=df_act['RMSE'],
                    mode='lines+markers',
                    name=name,
                    legendgroup=name,
                    line=dict(color=color),
                    marker=dict(symbol=marker, size=6),
                    showlegend=showlegend
                ), row=row, col=col)
                
    fig.update_layout(
        height=300*rows, 
        title_text="Mag Adapt Sensitivity Analysis by Joint", 
        hovermode="x unified"
    )
    
    # Update axes titles
    for i in range(cols):
        fig.update_xaxes(title_text="Observability Metric Threshold", row=rows, col=i+1)
    for i in range(rows):
        fig.update_yaxes(title_text="RMSE (degrees)", row=i+1, col=1)

    output_html = os.path.join(data_dir, 'mag_adapt_sensitivity_all.html')
    fig.write_html(output_html)
    print(f"\nInteractive plot saved to {output_html}")
    print("Open this HTML file in your browser to interactively toggle subjects/activities.")

if __name__ == "__main__":
    main()
