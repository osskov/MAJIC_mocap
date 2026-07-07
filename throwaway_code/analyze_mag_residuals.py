import os
import argparse
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from src.toolchest.PlateTrial import PlateTrial
from generate_method_orientation_sto_files import JOINT_SEGMENT_DICT

def main():
    parser = argparse.ArgumentParser(description="Analyze mag residuals.")
    parser.add_argument('--visualize', action='store_true', help="Show visualization of the residuals")
    args = parser.parse_args()

    data_dir = os.path.abspath("data")
    subjects = [d for d in os.listdir(data_dir) if d.startswith("Subject") and os.path.isdir(os.path.join(data_dir, d))]
    subjects.sort()
    
    # Limit to the first 3 subjects to keep execution reasonably fast, as in the sensitivity analysis
    # subjects = subjects[:3]
    
    # Store by joint
    joint_results = {joint: {'local': [], 'global_p': [], 'global_c': [],
                             'mag_local': [], 'mag_global_p': [], 'mag_global_c': []} 
                     for joint in JOINT_SEGMENT_DICT.keys()}
                     
    sum_local_angle = []
    sum_global_angle = []
    sum_local_mag = []
    sum_global_mag = []
    
    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject)
        activities = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]
        
        for activity in activities:
            print(f"Processing {subject} - {activity}...")
            subject_activity_folder = os.path.join(subject_dir, activity)
            try:
                plate_trials = PlateTrial.load_trial_from_folder(
                    subject_activity_folder,
                    align_plate_trials=True
                )
            except Exception as e:
                print(f"  Skipping {subject} {activity}: {e}")
                continue
                
            if not plate_trials:
                continue
                
            # Trim to 2000 frames to make the analysis faster (like in analyze_mag_adapt_sensitivity)
            # num_frames = 2000
            # if len(plate_trials) > 0 and len(plate_trials[0]) > num_frames:
            #     plate_trials = [p[:num_frames] for p in plate_trials]
                
            # 1. Rotate all IMU traces to global frame
            global_mag_traces = {}
            for trial in plate_trials:
                global_trace = trial.get_imu_trace_in_global_frame()
                global_mag_traces[trial.name] = np.array(global_trace.mag)
            
            # Find the minimum length in case they somehow differ slightly
            min_len = min(len(trace) for trace in global_mag_traces.values())
            
            # 2. Calculate m_median for each timestep
            # Stack all mags: shape (num_imus, N, 3)
            all_mags = []
            for name, mag in global_mag_traces.items():
                all_mags.append(mag[:min_len])
            all_mags = np.stack(all_mags, axis=0) # (num_imus, N, 3)
            
            # Median across IMUs
            m_median = np.median(all_mags, axis=0) # (N, 3)
            # Normalize m_median
            m_med_norm = np.linalg.norm(m_median, axis=1, keepdims=True)
            m_med_norm[m_med_norm == 0] = 1e-9
            m_median_unit = m_median / m_med_norm
            
            # Calculate sum of global errors for all timestamps in this trial
            all_mags_norm = np.linalg.norm(all_mags, axis=2, keepdims=True)
            all_mags_norm[all_mags_norm == 0] = 1e-9
            all_mags_unit = all_mags / all_mags_norm
            
            dot_global_all = np.sum(all_mags_unit * m_median_unit[np.newaxis, :, :], axis=2)
            dot_global_all = np.clip(dot_global_all, -1.0, 1.0)
            angle_global_all = np.arccos(dot_global_all)
            mag_global_all = np.linalg.norm(all_mags - m_median[np.newaxis, :, :], axis=2)
            
            sum_global_angle_trial = np.sum(angle_global_all, axis=0)
            sum_global_mag_trial = np.sum(mag_global_all, axis=0)
            
            local_angle_trial = np.zeros(min_len)
            local_mag_trial = np.zeros(min_len)
            
            # 3. Calculate residuals per joint
            for joint_name, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
                # Find matching trial names
                parent_trial_name = next((name for name in global_mag_traces.keys() if parent_name in name), None)
                child_trial_name = next((name for name in global_mag_traces.keys() if child_name in name), None)
                
                if not parent_trial_name or not child_trial_name:
                    continue
                    
                m_p = global_mag_traces[parent_trial_name][:min_len]
                m_c = global_mag_traces[child_trial_name][:min_len]
                
                # Normalize
                norm_p = np.linalg.norm(m_p, axis=1, keepdims=True)
                norm_p[norm_p == 0] = 1e-9
                m_p_unit = m_p / norm_p
                
                norm_c = np.linalg.norm(m_c, axis=1, keepdims=True)
                norm_c[norm_c == 0] = 1e-9
                m_c_unit = m_c / norm_c
                
                # Dot products (clip to [-1, 1] to avoid arccos nan)
                dot_local = np.sum(m_p_unit * m_c_unit, axis=1)
                dot_local = np.clip(dot_local, -1.0, 1.0)
                delta_theta_local = np.arccos(dot_local)
                
                dot_global_p = np.sum(m_p_unit * m_median_unit, axis=1)
                dot_global_p = np.clip(dot_global_p, -1.0, 1.0)
                delta_theta_global_p = np.arccos(dot_global_p)
                
                dot_global_c = np.sum(m_c_unit * m_median_unit, axis=1)
                dot_global_c = np.clip(dot_global_c, -1.0, 1.0)
                delta_theta_global_c = np.arccos(dot_global_c)
                
                # Magnitude residuals (on unnormalized vectors)
                mag_res_local = np.linalg.norm(m_p - m_c, axis=1)
                mag_res_global_p = np.linalg.norm(m_p - m_median, axis=1)
                mag_res_global_c = np.linalg.norm(m_c - m_median, axis=1)
                
                joint_results[joint_name]['local'].extend(delta_theta_local)
                joint_results[joint_name]['global_p'].extend(delta_theta_global_p)
                joint_results[joint_name]['global_c'].extend(delta_theta_global_c)
                joint_results[joint_name]['mag_local'].extend(mag_res_local)
                joint_results[joint_name]['mag_global_p'].extend(mag_res_global_p)
                joint_results[joint_name]['mag_global_c'].extend(mag_res_global_c)
                
                local_angle_trial += delta_theta_local
                local_mag_trial += mag_res_local
                
            sum_local_angle.extend(local_angle_trial)
            sum_global_angle.extend(sum_global_angle_trial)
            sum_local_mag.extend(local_mag_trial)
            sum_global_mag.extend(sum_global_mag_trial)

    print("\n=== Paired T-Test Results ===")
    print("Testing H0: Δθ_local >= Δθ_global vs H1: Δθ_local < Δθ_global\n")
    for joint_name, res in joint_results.items():
        if not res['local']:
            continue
            
        local_array = np.array(res['local'])
        global_p_array = np.array(res['global_p'])
        global_c_array = np.array(res['global_c'])
        
        # Test for parent
        t_stat_p, p_val_p = stats.ttest_rel(local_array, global_p_array, alternative='less')
        # Test for child
        t_stat_c, p_val_c = stats.ttest_rel(local_array, global_c_array, alternative='less')
        
        print(f"Joint: {joint_name}")
        print(f"  Mean Local Residual:    {np.rad2deg(np.mean(local_array)):.2f} deg")
        print(f"  Mean Global Residual P: {np.rad2deg(np.mean(global_p_array)):.2f} deg")
        print(f"  Mean Global Residual C: {np.rad2deg(np.mean(global_c_array)):.2f} deg")
        print(f"  T-Test (Local vs Global_P): T={t_stat_p:.2f}, p={p_val_p:.2e}")
        print(f"  T-Test (Local vs Global_C): T={t_stat_c:.2f}, p={p_val_c:.2e}")
        print("")

    print("\n=== Magnitude Paired T-Test Results ===")
    print("Testing H0: ||m_local_diff|| >= ||m_global_diff|| vs H1: ||m_local_diff|| < ||m_global_diff||\n")
    for joint_name, res in joint_results.items():
        if not res['mag_local']:
            continue
            
        local_array = np.array(res['mag_local'])
        global_p_array = np.array(res['mag_global_p'])
        global_c_array = np.array(res['mag_global_c'])
        
        # Test for parent
        t_stat_p, p_val_p = stats.ttest_rel(local_array, global_p_array, alternative='less')
        # Test for child
        t_stat_c, p_val_c = stats.ttest_rel(local_array, global_c_array, alternative='less')
        
        print(f"Joint: {joint_name}")
        print(f"  Mean Local Magnitude Residual:    {np.mean(local_array):.4f}")
        print(f"  Mean Global Magnitude Residual P: {np.mean(global_p_array):.4f}")
        print(f"  Mean Global Magnitude Residual C: {np.mean(global_c_array):.4f}")
        print(f"  T-Test (Local vs Global_P): T={t_stat_p:.2f}, p={p_val_p:.2e}")
        print(f"  T-Test (Local vs Global_C): T={t_stat_c:.2f}, p={p_val_c:.2e}")
        print("")

    print("\n=== Sum of Errors Comparison ===")
    print("Testing H0: Sum(Local) >= Sum(Global) vs H1: Sum(Local) < Sum(Global)\n")
    
    loc_ang_arr = np.array(sum_local_angle)
    glob_ang_arr = np.array(sum_global_angle)
    t_stat_ang, p_val_ang = stats.ttest_rel(loc_ang_arr, glob_ang_arr, alternative='less')
    
    print(f"Angle Errors:")
    print(f"  Mean Sum Local (7 joints):          {np.rad2deg(np.mean(loc_ang_arr)):.2f} deg")
    print(f"  Mean Overall Local (per joint):     {np.rad2deg(np.mean(loc_ang_arr))/7:.2f} deg")
    print(f"  Mean Sum Global (8 segments):       {np.rad2deg(np.mean(glob_ang_arr)):.2f} deg")
    print(f"  Mean Overall Global (per segment):  {np.rad2deg(np.mean(glob_ang_arr))/8:.2f} deg")
    print(f"  T-Test (on Sums): T={t_stat_ang:.2f}, p={p_val_ang:.2e}\n")
    
    loc_mag_arr = np.array(sum_local_mag)
    glob_mag_arr = np.array(sum_global_mag)
    t_stat_mag, p_val_mag = stats.ttest_rel(loc_mag_arr, glob_mag_arr, alternative='less')
    
    print(f"Magnitude Errors:")
    print(f"  Mean Sum Local (7 joints):          {np.mean(loc_mag_arr):.4f}")
    print(f"  Mean Overall Local (per joint):     {np.mean(loc_mag_arr)/7:.4f}")
    print(f"  Mean Sum Global (8 segments):       {np.mean(glob_mag_arr):.4f}")
    print(f"  Mean Overall Global (per segment):  {np.mean(glob_mag_arr)/8:.4f}")
    print(f"  T-Test (on Sums): T={t_stat_mag:.2f}, p={p_val_mag:.2e}\n")

    if args.visualize:
        plot_residuals(joint_results)
        plot_kinematic_chain(joint_results)

def plot_kinematic_chain(joint_results):
    if not joint_results['Lumbar']['mag_local']:
        return
        
    def subsample(arr, max_pts=5000):
        if len(arr) > max_pts:
            return np.random.choice(arr, max_pts, replace=False)
        return arr
        
    r_chain_names = ['torso', 'Lumbar', 'pelvis', 'R_Hip', 'femur_r', 'R_Knee', 'tibia_r', 'R_Ankle', 'calcn_r']
    r_chain_mag = [
        subsample(joint_results['Lumbar']['mag_global_c']),
        subsample(joint_results['Lumbar']['mag_local']),
        subsample(joint_results['Lumbar']['mag_global_p']),
        subsample(joint_results['R_Hip']['mag_local']),
        subsample(joint_results['R_Hip']['mag_global_c']),
        subsample(joint_results['R_Knee']['mag_local']),
        subsample(joint_results['R_Knee']['mag_global_c']),
        subsample(joint_results['R_Ankle']['mag_local']),
        subsample(joint_results['R_Ankle']['mag_global_c']),
    ]
    r_chain_ang = [
        np.rad2deg(subsample(joint_results['Lumbar']['global_c'])),
        np.rad2deg(subsample(joint_results['Lumbar']['local'])),
        np.rad2deg(subsample(joint_results['Lumbar']['global_p'])),
        np.rad2deg(subsample(joint_results['R_Hip']['local'])),
        np.rad2deg(subsample(joint_results['R_Hip']['global_c'])),
        np.rad2deg(subsample(joint_results['R_Knee']['local'])),
        np.rad2deg(subsample(joint_results['R_Knee']['global_c'])),
        np.rad2deg(subsample(joint_results['R_Ankle']['local'])),
        np.rad2deg(subsample(joint_results['R_Ankle']['global_c'])),
    ]
    
    l_chain_names = ['torso', 'Lumbar', 'pelvis', 'L_Hip', 'femur_l', 'L_Knee', 'tibia_l', 'L_Ankle', 'calcn_l']
    l_chain_mag = [
        subsample(joint_results['Lumbar']['mag_global_c']),
        subsample(joint_results['Lumbar']['mag_local']),
        subsample(joint_results['Lumbar']['mag_global_p']),
        subsample(joint_results['L_Hip']['mag_local']),
        subsample(joint_results['L_Hip']['mag_global_c']),
        subsample(joint_results['L_Knee']['mag_local']),
        subsample(joint_results['L_Knee']['mag_global_c']),
        subsample(joint_results['L_Ankle']['mag_local']),
        subsample(joint_results['L_Ankle']['mag_global_c']),
    ]
    l_chain_ang = [
        np.rad2deg(subsample(joint_results['Lumbar']['global_c'])),
        np.rad2deg(subsample(joint_results['Lumbar']['local'])),
        np.rad2deg(subsample(joint_results['Lumbar']['global_p'])),
        np.rad2deg(subsample(joint_results['L_Hip']['local'])),
        np.rad2deg(subsample(joint_results['L_Hip']['global_c'])),
        np.rad2deg(subsample(joint_results['L_Knee']['local'])),
        np.rad2deg(subsample(joint_results['L_Knee']['global_c'])),
        np.rad2deg(subsample(joint_results['L_Ankle']['local'])),
        np.rad2deg(subsample(joint_results['L_Ankle']['global_c'])),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Angles
    axes[0, 0].boxplot(r_chain_ang, tick_labels=r_chain_names, patch_artist=True, boxprops=dict(facecolor='#1f77b4', alpha=0.7))
    axes[0, 0].set_title("Right Leg Chain (Angle)")
    axes[0, 0].set_ylabel("Angle Error (deg)")
    
    axes[1, 0].boxplot(l_chain_ang, tick_labels=l_chain_names, patch_artist=True, boxprops=dict(facecolor='#ff7f0e', alpha=0.7))
    axes[1, 0].set_title("Left Leg Chain (Angle)")
    axes[1, 0].set_ylabel("Angle Error (deg)")
    axes[1, 0].set_xlabel("Segment / Joint")
    
    # Magnitudes
    axes[0, 1].boxplot(r_chain_mag, tick_labels=r_chain_names, patch_artist=True, boxprops=dict(facecolor='#1f77b4', alpha=0.7))
    axes[0, 1].set_title("Right Leg Chain (Magnitude)")
    axes[0, 1].set_ylabel("Magnitude Error")
    
    axes[1, 1].boxplot(l_chain_mag, tick_labels=l_chain_names, patch_artist=True, boxprops=dict(facecolor='#ff7f0e', alpha=0.7))
    axes[1, 1].set_title("Left Leg Chain (Magnitude)")
    axes[1, 1].set_ylabel("Magnitude Error")
    axes[1, 1].set_xlabel("Segment / Joint")
    
    fig.suptitle("Residuals along Kinematic Chain")
    plt.tight_layout()
    plt.show()

def plot_residuals(joint_results):
    joints = [j for j, res in joint_results.items() if res['local']]
    if not joints:
        print("No valid results to plot.")
        return
        
    num_joints = len(joints)
    
    fig, axes = plt.subplots(2, num_joints, figsize=(3 * num_joints, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    def subsample(arr, max_pts=5000):
        if len(arr) > max_pts:
            return np.random.choice(arr, max_pts, replace=False)
        return arr
        
    for i, joint in enumerate(joints):
        res = joint_results[joint]
        
        ax_ang = axes[0, i] if num_joints > 1 else axes[0]
        ax_mag = axes[1, i] if num_joints > 1 else axes[1]
        
        # Row 1: Angles
        data_ang = [np.rad2deg(subsample(res['local'])), np.rad2deg(subsample(res['global_p'])), np.rad2deg(subsample(res['global_c']))]
        bp1 = ax_ang.boxplot(data_ang, tick_labels=['Local', 'Global P', 'Global C'], patch_artist=True)
        ax_ang.set_title(f"{joint} Angle (deg)")
        
        # Row 2: Magnitudes
        data_mag = [subsample(res['mag_local']), subsample(res['mag_global_p']), subsample(res['mag_global_c'])]
        bp2 = ax_mag.boxplot(data_mag, tick_labels=['Local', 'Global P', 'Global C'], patch_artist=True)
        ax_mag.set_title(f"{joint} Magnitude")
        
        for bp in [bp1, bp2]:
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                
    fig.suptitle("Magnetic Residuals Comparison")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
