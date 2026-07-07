import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.toolchest.PlateTrial import PlateTrial
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def find_plate_by_name(plate_trials, name_part):
    """Helper function to find a PlateTrial in a list by a substring of its name."""
    return next((p for p in plate_trials if name_part in p.name), None)

# Focused only on Knees
JOINT_SEGMENT_DICT = {
    'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
    'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
}

def main():
    """
    Loads data from all subjects 'complexTasks' for markers, loads all joint relative
    rotations over time, converts them to angle-axis form, and performs PCA to extract
    the primary, secondary, and tertiary functional rotation axes. Finally plots
    the % variance explained by each PC across all subjects and joints.
    """
    subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
    trial_type = 'complexTasks'

    # Store results for plotting
    timeseries_data = {joint: [] for joint in JOINT_SEGMENT_DICT.keys()}

    for subject_num in subjects:
        subject_id = f"Subject{subject_num}"
        print(f"\n{'='*25} Processing {subject_id} {'='*25}")
        
        # --- 1. Load Marker (Ground Truth) Data ---
        try:
            data_folder_path = os.path.abspath(os.path.join("data", subject_id, trial_type))
            plate_trials_marker = PlateTrial.load_trial_from_folder(data_folder_path, align_plate_trials=False)
            
        except Exception as e:
            print(f"Could not load data for {subject_id}, {trial_type}. Error: {e}. Skipping.")
            continue
            
        # --- 2. Process Joints ---
        for joint, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
            parent_marker = find_plate_by_name(plate_trials_marker, parent_name)
            child_marker = find_plate_by_name(plate_trials_marker, child_name)

            if not parent_marker or not child_marker:
                print(f"  [{joint}] Missing plate data. Skipping.")
                continue
                
            R_parent = np.stack(parent_marker.world_trace.rotations)
            R_child = np.stack(child_marker.world_trace.rotations)
            
            # R_rel = R_parent.T @ R_child
            R_rel = np.einsum('nij,njk->nik', np.transpose(R_parent, (0, 2, 1)), R_child)
            
            rotvec = Rotation.from_matrix(R_rel).as_rotvec()
            angles = np.linalg.norm(rotvec, axis=1)
            
            normalized_axes = np.zeros_like(rotvec)
            valid_idx = angles > 1e-8
            normalized_axes[valid_idx] = rotvec[valid_idx] / angles[valid_idx, np.newaxis]
            
            if np.any(valid_idx):
                first_valid_axis = normalized_axes[valid_idx][0]
                for i in range(len(normalized_axes)):
                    if valid_idx[i] and np.dot(normalized_axes[i], first_valid_axis) < 0:
                        normalized_axes[i] *= -1
                        
            # Uncentered PCA
            valid_axes = normalized_axes[valid_idx]
            if len(valid_axes) > 0:
                scatter_matrix = valid_axes.T @ valid_axes
                eigenvalues, eigenvectors = np.linalg.eigh(scatter_matrix)
                
                sort_idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[sort_idx]
                eigenvectors = eigenvectors[:, sort_idx]
                
                variance_explained = (eigenvalues / np.sum(eigenvalues)) * 100
                
                # Project rotation vectors onto the PCA axes
                projections = rotvec @ eigenvectors # (N, 3)
                projections_deg = np.degrees(projections)
                
                timeseries_data[joint].append({
                    'subject': subject_id,
                    'time': parent_marker.world_trace.timestamps,
                    'pc1': projections_deg[:, 0],
                    'pc2': projections_deg[:, 1],
                    'pc3': projections_deg[:, 2]
                })
                
                print(f"  [{joint}] PCA Results: PC1={variance_explained[0]:.1f}%, PC2={variance_explained[1]:.1f}%, PC3={variance_explained[2]:.1f}%")
            else:
                print(f"  [{joint}] No valid rotation axes found.")

    # --- 3. Visualization Phase ---
    if not any(timeseries_data.values()):
        print("\nNo data generated to plot.")
        return

    print("\nGenerating Time-Series Plots...")
    fig1, axes1 = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    
    plot_joints = ['L_Knee', 'R_Knee']
    pcs = ['pc1', 'pc2', 'pc3']
    
    # Fix colormap warning and ensure enough colors
    try:
        cmap_tab20 = cm.get_cmap('tab20')
    except AttributeError:
        cmap_tab20 = plt.get_cmap('tab20')
        
    colors = cmap_tab20(np.linspace(0, 1, len(subjects)))
    subject_to_color = {f"Subject{s}": colors[i] for i, s in enumerate(subjects)}

    for col, joint in enumerate(plot_joints):
        for row, pc_key in enumerate(pcs):
            ax = axes1[row, col]
            for entry in timeseries_data[joint]:
                s_id = entry['subject']
                ax.plot(entry['time'], entry[pc_key], color=subject_to_color[s_id], alpha=0.7, linewidth=1, label=s_id if col==0 and row==0 else None)
            
            ax.set_ylabel(f'{pc_key.upper()} [deg]')
            ax.grid(True, linestyle='--', alpha=0.5)
            if row == 0:
                ax.set_title(f"{joint} Rotation Components", fontsize=14, fontweight='bold')
            if col == 0:
                ax.annotate(f"{pc_key.upper()}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', fontweight='bold')

    # Shared X-axis label
    for ax in axes1[2, :]:
        ax.set_xlabel('Time [s]', fontsize=12)

    # Add legend to the first subplot or a global legend
    fig1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title="Subjects", fontsize='small')
    fig1.tight_layout(rect=[0, 0, 0.95, 1])

    # --- 4. Mean and SD Visualization ---
    print("Generating Mean and SD Summary Plots...")
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
    
    for i, joint in enumerate(plot_joints):
        ax = axes2[i]
        
        # Prepare data for grouped bar chart
        subject_labels = []
        means = {pc: [] for pc in pcs}
        stds = {pc: [] for pc in pcs}
        
        # Extract subjects that actually have data for this joint
        joint_entries = timeseries_data[joint]
        for entry in joint_entries:
            subject_labels.append(entry['subject'])
            for pc in pcs:
                data = np.abs(entry[pc])
                means[pc].append(np.mean(data))
                stds[pc].append(np.std(data))
        
        if not subject_labels:
            continue
            
        x = np.arange(len(subject_labels))
        width = 0.25
        
        for j, pc in enumerate(pcs):
            ax.bar(x + (j-1)*width, means[pc], width, yerr=stds[pc], label=pc.upper() if i==0 else None, capsize=4, alpha=0.8)
            
        ax.set_ylabel('Mean Abs Magnitude [deg]')
        ax.set_title(f'Mean Rotation Magnitude for {joint}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subject_labels, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(title="Axes")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
