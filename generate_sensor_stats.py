import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from src.toolchest.PlateTrial import PlateTrial

def get_sensor_type(sensor_name):
    if 'pelvis' in sensor_name.lower():
        return 'pelvis'
    elif 'femur' in sensor_name.lower():
        return 'thigh'
    elif 'tibia' in sensor_name.lower():
        return 'shank'
    elif 'calcn' in sensor_name.lower():
        return 'foot'
    return 'other'

# Parent/child sensor pairs, keyed by the joint between them
JOINTS = {
    'Lumbar': ('pelvis_imu', 'torso_imu'),
    'R_Hip':  ('pelvis_imu', 'femur_r_imu'),
    'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
    'R_Ankle':('tibia_r_imu', 'calcn_r_imu'),
    'L_Hip':  ('pelvis_imu', 'femur_l_imu'),
    'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
    'L_Ankle':('tibia_l_imu', 'calcn_l_imu'),
}

def get_base_joint_name(joint_name):
    if joint_name.startswith('R_') or joint_name.startswith('L_'):
        return joint_name[2:]
    return joint_name

def cosine_similarity(v1, v2):
    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    return dot / (norm1 * norm2)

def generate_sensor_stats():
    subjects = [f"{i:02d}" for i in range(1, 12)]
    
    # Storage for Task 1: Mag magnitude std per sensor
    mag_distortions = defaultdict(list)
    
    # Storage for Task 2: World mag median per subject
    subject_world_mags = {}
    
    # Storage for Task 3: Noise std per sensor type during sitting
    noise_stats = defaultdict(lambda: {'gyro': [], 'acc': [], 'mag': []})

    # Storage for Task 4: Parent-vs-global local consistency of the magnetic field, per joint
    joint_consistency_stats = defaultdict(lambda: {
        'cos_sim_child_global': [], 'cos_sim_child_parent': [],
        'var_reduction': []
    })
    
    print("Processing subjects to generate sensor stats...")
    
    for subject in subjects:
        folder = Path(f"data/Subject{subject}/complexTasks")
        if not folder.exists():
            continue
            
        try:
            plates = PlateTrial.from_folder(folder)
        except Exception as e:
            print(f"Error loading Subject {subject}: {e}. Skipping.")
            continue
            
        if 'pelvis_imu' not in plates:
            print(f"No pelvis_imu found for Subject {subject}. Skipping.")
            continue
            
        # Task 1: Mag magnitude std per sensor (distortions)
        for sensor_name, plate in plates.items():
            mag_norm = np.linalg.norm(plate.imu_trace.mag, axis=1)
            mag_norm_std = np.std(mag_norm)
            mag_distortions[sensor_name].append(mag_norm_std)
            
        # Find sitting intervals using pelvis
        pelvis_plate = plates['pelvis_imu']
        timestamps = pelvis_plate.imu_trace.timestamps
        dt = np.mean(np.diff(timestamps))
        fs = 1.0 / dt
        
        pelvis_y = pelvis_plate.world_trace.positions[:, 1]
        pelvis_y_vel = np.abs(np.gradient(pelvis_y, dt))
        window_size = int(0.5 * fs)
        smoothed_vel = pd.Series(pelvis_y_vel).rolling(window=window_size, center=True).mean().ffill().bfill().values
        
        pelvis_gyro_norm = np.linalg.norm(pelvis_plate.imu_trace.gyro, axis=1)
        smoothed_gyro = pd.Series(pelvis_gyro_norm).rolling(window=window_size, center=True).mean().ffill().bfill().values
        
        # Sitting criteria
        is_quiet_sitting_strict = (pelvis_y < 0.75) & (smoothed_vel < 0.01) & (smoothed_gyro < 0.05)
        
        num_samples = len(timestamps)
        min_samples = int(3.0 * fs)
        
        def find_intervals(mask):
            intervals = []
            in_interval = False
            start_idx = 0
            for i in range(num_samples):
                if mask[i] and not in_interval:
                    in_interval = True
                    start_idx = i
                elif not mask[i] and in_interval:
                    in_interval = False
                    if (i - start_idx) >= min_samples:
                        intervals.append((start_idx, i))
            if in_interval:
                if (num_samples - start_idx) >= min_samples:
                    intervals.append((start_idx, num_samples))
            return intervals

        intervals = find_intervals(is_quiet_sitting_strict)
        if not intervals:
            is_quiet_sitting_loose = (pelvis_y < 0.75) & (smoothed_vel < 0.03) & (smoothed_gyro < 0.15)
            intervals = find_intervals(is_quiet_sitting_loose)
            
        if not intervals:
            print(f"Subject {subject}: No quiet sitting periods found.")
            continue
            
        # Task 2: World mag median per subject (using all data)
        world_mags_all = []
        for sensor_name, plate in plates.items():
            rotations = plate.world_trace.rotations
            mags = plate.imu_trace.mag
            world_mags = (rotations @ mags[..., None])[..., 0]
            world_mags_all.append(world_mags)
        
        if world_mags_all:
            all_world_mags = np.concatenate(world_mags_all, axis=0)
            subject_world_mags[subject] = np.median(all_world_mags, axis=0)

        # Task 4: Parent-vs-global local consistency of the magnetic field, per joint
        if subject in subject_world_mags:
            global_mag = subject_world_mags[subject]
            for joint_name, (parent_name, child_name) in JOINTS.items():
                if parent_name not in plates or child_name not in plates:
                    continue

                p_mag_world = plates[parent_name].get_imu_trace_in_global_frame().mag
                c_mag_world = plates[child_name].get_imu_trace_in_global_frame().mag
                min_len = min(len(p_mag_world), len(c_mag_world))
                p_mag_world = p_mag_world[:min_len]
                c_mag_world = c_mag_world[:min_len]
                global_mag_array = np.tile(global_mag, (min_len, 1))

                cos_sim_child_global = cosine_similarity(c_mag_world, global_mag_array)
                cos_sim_child_parent = cosine_similarity(c_mag_world, p_mag_world)

                # Variance-reduction ratio: how much of the child's temporal variability is
                # explained away by referencing the parent instead of the constant global field
                resid_global = c_mag_world - global_mag_array
                resid_parent = c_mag_world - p_mag_world
                var_global = np.sum(np.var(resid_global, axis=0))
                var_parent = np.sum(np.var(resid_parent, axis=0))
                var_reduction = 1 - (var_parent / var_global) if var_global > 0 else np.nan

                base_joint = get_base_joint_name(joint_name)
                joint_consistency_stats[base_joint]['cos_sim_child_global'].append(np.mean(cos_sim_child_global))
                joint_consistency_stats[base_joint]['cos_sim_child_parent'].append(np.mean(cos_sim_child_parent))
                joint_consistency_stats[base_joint]['var_reduction'].append(var_reduction)

        # Task 3: Noise std per sensor type during sitting
        for sensor_name, plate in plates.items():
            sensor_type = get_sensor_type(sensor_name)
            for start, end in intervals:
                # We want std of the sensor frame readings to determine sensor noise parameters
                gyro_std = np.std(plate.imu_trace.gyro[start:end], axis=0)
                acc_std = np.std(plate.imu_trace.acc[start:end], axis=0)
                mag_std = np.std(plate.imu_trace.mag[start:end], axis=0)
                
                # Weight by interval length
                length = end - start
                noise_stats[sensor_type]['gyro'].append((gyro_std, length))
                noise_stats[sensor_type]['acc'].append((acc_std, length))
                noise_stats[sensor_type]['mag'].append((mag_std, length))

    # Print Results
    print("\n" + "="*80)
    print("1. MAGNETIC FIELD DISTORTIONS PER SENSOR SEGMENT")
    print("   (Standard deviation of the mag vector magnitude across the whole trial)")
    print("="*80)
    avg_distortions = {}
    for sensor, stds in mag_distortions.items():
        avg_distortions[sensor] = np.mean(stds)
        
    sorted_distortions = sorted(avg_distortions.items(), key=lambda x: x[1])
    for sensor, std in sorted_distortions:
        print(f"{sensor:<20}: {std:.4f} uT")
        
    print(f"\n-> Segment with LOWEST distortion: {sorted_distortions[0][0]} ({sorted_distortions[0][1]:.4f} uT)")
    print(f"-> Segment with HIGHEST distortion: {sorted_distortions[-1][0]} ({sorted_distortions[-1][1]:.4f} uT)")

    print("\n" + "="*80)
    print("2. GLOBAL MAGNETIC FIELD VARIATION ACROSS SUBJECTS")
    print("   (Median world-frame magnetic field per subject across the entire trial)")
    print("="*80)
    world_mags_array = np.array(list(subject_world_mags.values()))
    if len(world_mags_array) > 0:
        global_median = np.median(world_mags_array, axis=0)
        global_std = np.std(world_mags_array, axis=0)
        global_norm_std = np.std(np.linalg.norm(world_mags_array, axis=1))
        
        for subj, mag in subject_world_mags.items():
            print(f"Subject {subj}: [{mag[0]:>7.2f}, {mag[1]:>7.2f}, {mag[2]:>7.2f}] uT (Norm: {np.linalg.norm(mag):.2f} uT)")
            
        print(f"\nAcross all subjects:")
        print(f"  Mean of Medians : [{np.mean(world_mags_array, axis=0)[0]:.2f}, {np.mean(world_mags_array, axis=0)[1]:.2f}, {np.mean(world_mags_array, axis=0)[2]:.2f}] uT")
        print(f"  Std across Subj : [{global_std[0]:.2f}, {global_std[1]:.2f}, {global_std[2]:.2f}] uT")
        print(f"  Std of Magnitude: {global_norm_std:.4f} uT")
    else:
        print("No subject data available.")

    print("\n" + "="*80)
    print("3. SENSOR NOISE PARAMETERS BY SENSOR TYPE")
    print("   (Average standard deviation during quiet sitting intervals)")
    print("="*80)
    
    for sensor_type, stats in noise_stats.items():
        if sensor_type != 'pelvis':
            continue
            
        print(f"\n{sensor_type.upper()} SENSORS:")
        
        for modality in ['gyro', 'acc', 'mag']:
            data = stats[modality]
            if not data:
                continue
            
            # Weighted average
            total_len = sum(length for _, length in data)
            weighted_std = np.sum([std * length for std, length in data], axis=0) / total_len
            
            unit = "rad/s" if modality == 'gyro' else ("m/s^2" if modality == 'acc' else "uT   ")
            mean_val = np.mean(weighted_std)
            print(f"  {modality.capitalize()} Noise Std ({unit}): {mean_val:.6f} | [x={weighted_std[0]:.5f}, y={weighted_std[1]:.5f}, z={weighted_std[2]:.5f}]")

    print("\n" + "="*80)
    print("4. MAGNETIC FIELD LOCAL CONSISTENCY: CHILD-VS-PARENT vs. CHILD-VS-GLOBAL")
    print("   (World-frame mag vectors; per-subject-trial means, averaged across subjects)")
    print("="*80)
    print(f"{'Joint':<10}{'CosSim Child-Global':>22}{'CosSim Child-Parent':>22}{'Var-Reduction Ratio':>22}")

    joint_order = ['Lumbar', 'Hip', 'Knee', 'Ankle']
    for joint in joint_order:
        if joint not in joint_consistency_stats:
            continue
        stats = joint_consistency_stats[joint]
        mean_cos_sim_global = np.mean(stats['cos_sim_child_global'])
        mean_cos_sim_parent = np.mean(stats['cos_sim_child_parent'])
        mean_var_reduction = np.nanmean(stats['var_reduction'])
        print(f"{joint:<10}{mean_cos_sim_global:>22.3f}{mean_cos_sim_parent:>22.3f}{mean_var_reduction:>22.3f}")

    print("\nDone.")

if __name__ == "__main__":
    generate_sensor_stats()
