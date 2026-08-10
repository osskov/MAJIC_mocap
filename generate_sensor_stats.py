import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import paths
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.IMUTrace import IMUTrace
from experiments.experiment_utils import segment_observability

PLOTS_DIR = paths.plots_dir("sensor_stats")

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

def get_base_joint_name(joint_name):
    if joint_name.startswith('R_') or joint_name.startswith('L_'):
        return joint_name[2:]
    return joint_name

def cosine_similarity(v1, v2):
    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    return dot / (norm1 * norm2)

def calculate_magnetic_distortions(plates, mag_distortions):
    """Task 1: Calculate magnetic magnitude standard deviation per sensor."""
    for sensor_name, plate in plates.items():
        mag_norm = np.linalg.norm(plate.imu_trace.mag, axis=1)
        mag_norm_std = np.std(mag_norm)
        mag_distortions[sensor_name].append(mag_norm_std)

def calculate_world_mags(plates, subject, subject_world_mags):
    """Task 2: Calculate median world-frame magnetic field per subject."""
    world_mags_all = []
    for sensor_name, plate in plates.items():
        rotations = plate.world_trace.rotations
        mags = plate.imu_trace.mag
        world_mags = (rotations @ mags[..., None])[..., 0]
        world_mags_all.append(world_mags)
    
    if world_mags_all:
        all_world_mags = np.concatenate(world_mags_all, axis=0)
        subject_world_mags[subject] = np.median(all_world_mags, axis=0)

def calculate_joint_consistency(plates, global_mag, joint_consistency_stats):
    """Task 4: Parent-vs-global local consistency of the magnetic field."""
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

        resid_global = c_mag_world - global_mag_array
        resid_parent = c_mag_world - p_mag_world
        var_global = np.sum(np.var(resid_global, axis=0))
        var_parent = np.sum(np.var(resid_parent, axis=0))
        var_reduction = 1 - (var_parent / var_global) if var_global > 0 else np.nan

        base_joint = get_base_joint_name(joint_name)
        joint_consistency_stats[base_joint]['cos_sim_child_global'].append(np.mean(cos_sim_child_global))
        joint_consistency_stats[base_joint]['cos_sim_child_parent'].append(np.mean(cos_sim_child_parent))
        joint_consistency_stats[base_joint]['var_reduction'].append(var_reduction)

def find_stationary_intervals(raw_imus):
    """Finds periods where both feet are completely stationary."""
    foot_sensors = [s for s in ['calcn_l_imu', 'calcn_r_imu'] if s in raw_imus]
    if not foot_sensors:
        return None, foot_sensors, 0
        
    foot_raw = raw_imus[foot_sensors[0]]
    timestamps = foot_raw.timestamps
    dt = np.mean(np.diff(timestamps))
    fs = 1.0 / dt
    window_size = int(1.0 * fs)
    threshold = 0.05
    
    min_len = min(len(raw_imus[s].gyro) for s in foot_sensors)
    is_stationary_feet = np.ones(min_len, dtype=bool)
    
    for sensor_name in foot_sensors:
        trace = raw_imus[sensor_name]
        gyro_norm = np.linalg.norm(trace.gyro[:min_len], axis=1)
        rolling_std = pd.Series(gyro_norm).rolling(window=window_size, center=True).std().values
        is_stationary = (rolling_std < threshold) & (~np.isnan(rolling_std))
        is_stationary_feet &= is_stationary
    
    diffs = np.diff(is_stationary_feet.astype(int), prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    min_samples = int(20.0 * fs)
    margin_samples = int(3.0 * fs)
    intervals = []
    for s, e in zip(starts, ends):
        if e - s >= min_samples:
            intervals.append((s + margin_samples, e - margin_samples))
            
    return intervals, foot_sensors, min_len

def plot_stationary_intervals(subject, raw_imus, foot_sensors, intervals, min_len):
    """Plots the detected stationary intervals."""
    import matplotlib.pyplot as plt
    longest = max(intervals, key=lambda x: x[1] - x[0])
    s, e = longest
    
    foot_raw = raw_imus[foot_sensors[0]]
    timestamps = foot_raw.timestamps
    
    max_gyro_norm = np.zeros(min_len)
    for sensor_name in foot_sensors:
        trace = raw_imus[sensor_name]
        max_gyro_norm = np.maximum(max_gyro_norm, np.linalg.norm(trace.gyro[:min_len], axis=1))
        
    acc_norm = np.linalg.norm(foot_raw.acc[:min_len], axis=1)
    mag_norm = np.linalg.norm(foot_raw.mag[:min_len], axis=1)
    
    time_axis = timestamps[:min_len] - timestamps[0]
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f'Detected Stationary Period (Foot Anchored) - Subject {subject}', fontsize=16)
    
    start_sec = time_axis[s]
    end_sec = time_axis[e]
    
    axs[0].plot(time_axis, acc_norm, color='blue', alpha=0.7, label='Foot Acc Magnitude')
    axs[0].axvspan(start_sec, end_sec, color='green', alpha=0.3, label='Detected Stationary Period')
    axs[0].set_ylabel('Acc (m/s²)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    axs[1].plot(time_axis, max_gyro_norm, color='orange', alpha=0.7, label='Max Gyr Magnitude (Feet)')
    axs[1].axvspan(start_sec, end_sec, color='green', alpha=0.3, label='Detected Stationary Period')
    axs[1].set_ylabel('Gyr (rad/s)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    axs[2].plot(time_axis, mag_norm, color='purple', alpha=0.7, label='Foot Mag Magnitude')
    axs[2].axvspan(start_sec, end_sec, color='green', alpha=0.3, label='Detected Stationary Period')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].set_ylabel('Mag (a.u.)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    plt.tight_layout()
    output_path = paths.ensure_parent(PLOTS_DIR / f"stationary_intervals_Subject{subject}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_path}")

def detect_sitting_sections(plates, subject, generate_plots=False):
    """Detects sitting sections in the complex tasks trial using the pelvis gyroscope."""
    if 'pelvis_imu' not in plates:
        return
        
    pelvis = plates['pelvis_imu'].imu_trace
    timestamps = pelvis.timestamps
    dt = np.mean(np.diff(timestamps))
    fs = 1.0 / dt
    window_size = int(1.0 * fs)
    threshold = 0.05
    
    gyro_norm = np.linalg.norm(pelvis.gyro, axis=1)
    rolling_std = pd.Series(gyro_norm).rolling(window=window_size, center=True).std().values
    
    is_stationary = (rolling_std < threshold) & (~np.isnan(rolling_std))
    diffs = np.diff(is_stationary.astype(int), prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    # Sitting usually lasts a while. We use > 10 seconds.
    min_samples = int(10.0 * fs) 
    intervals = []
    for s, e in zip(starts, ends):
        if e - s >= min_samples:
            intervals.append((s, e))
            
    if not intervals:
        print(f"Subject {subject}: No sitting sections > 10s found in trial.")
        return
        
    if generate_plots:
        import matplotlib.pyplot as plt
        acc_norm = np.linalg.norm(pelvis.acc, axis=1)
        mag_norm = np.linalg.norm(pelvis.mag, axis=1)
        
        # Calculate observability for the pelvis. Shared with the pipeline rather than
        # reimplemented here — this was a copy of the formula, and it carried the same
        # missing-dt bug the pipeline copy did.
        observability = segment_observability(pelvis)

        from scipy.signal import butter, filtfilt
        b, a = butter(4, 30.0 / (fs / 2), btype='low')
        observability = filtfilt(b, a, observability)
        
        time_axis = timestamps - timestamps[0]
        fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f'Detected Sitting Sections (Pelvis) - Subject {subject}', fontsize=16)
        
        axs[0].plot(time_axis, acc_norm, color='blue', alpha=0.7, label='Pelvis Acc Magnitude')
        axs[0].set_ylabel('Acc (m/s²)')
        axs[0].legend(loc='upper right')
        axs[0].grid(True)

        axs[1].plot(time_axis, gyro_norm, color='orange', alpha=0.7, label='Pelvis Gyr Magnitude')
        axs[1].set_ylabel('Gyr (rad/s)')
        axs[1].legend(loc='upper right')
        axs[1].grid(True)

        axs[2].plot(time_axis, mag_norm, color='purple', alpha=0.7, label='Pelvis Mag Magnitude')
        axs[2].set_ylabel('Mag (a.u.)')
        axs[2].legend(loc='upper right')
        axs[2].grid(True)
        
        axs[3].plot(time_axis, observability, color='red', alpha=0.7, label='Pelvis Observability')
        axs[3].set_xlabel('Time (seconds)')
        axs[3].set_ylabel('Observability')
        axs[3].set_ylim(0, 5000)
        axs[3].legend(loc='upper right')
        axs[3].grid(True)

        for (s, e) in intervals:
            start_sec = time_axis[s]
            end_sec = time_axis[e]
            axs[0].axvspan(start_sec, end_sec, color='green', alpha=0.3)
            axs[1].axvspan(start_sec, end_sec, color='green', alpha=0.3)
            axs[2].axvspan(start_sec, end_sec, color='green', alpha=0.3)
            axs[3].axvspan(start_sec, end_sec, color='green', alpha=0.3)

        plt.tight_layout()
        output_path = paths.ensure_parent(PLOTS_DIR / f"sitting_sections_Subject{subject}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved sitting sections plot to {output_path}")

def calculate_baseline_noise(raw_imus, foot_sensors, intervals, noise_stats):
    """Task 3: Baseline Intrinsic Sensor Noise using anchored foot sensors."""
    for sensor_name in foot_sensors:
        raw_trace = raw_imus[sensor_name]
        for start, end in intervals:
            s = min(start, len(raw_trace.gyro))
            e = min(end, len(raw_trace.gyro))
            if e <= s:
                continue
            gyro_std = np.std(raw_trace.gyro[s:e], axis=0)
            acc_std = np.std(raw_trace.acc[s:e], axis=0)
            mag_std = np.std(raw_trace.mag[s:e], axis=0)
            
            length = e - s
            noise_stats['baseline']['gyro'].append((gyro_std, length))
            noise_stats['baseline']['acc'].append((acc_std, length))
            noise_stats['baseline']['mag'].append((mag_std, length))

def print_results(mag_distortions, subject_world_mags, noise_stats, joint_consistency_stats):
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
        
    if sorted_distortions:
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
    print("3. INTRINSIC SENSOR NOISE PARAMETERS")
    print("   (Average standard deviation of foot sensors anchored to the ground)")
    print("="*80)
    
    for sensor_type, stats in noise_stats.items():
        print(f"\n{sensor_type.upper()} NOISE:")
        
        for modality in ['gyro', 'acc', 'mag']:
            data = stats[modality]
            if not data:
                continue
            
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

def generate_sensor_stats(generate_plots=True):
    subjects = [f"{i:02d}" for i in range(1, 12)]
    
    mag_distortions = defaultdict(list)
    subject_world_mags = {}
    noise_stats = defaultdict(lambda: {'gyro': [], 'acc': [], 'mag': []})
    joint_consistency_stats = defaultdict(lambda: {
        'cos_sim_child_global': [], 'cos_sim_child_parent': [], 'var_reduction': []
    })
    
    print("Processing subjects to generate sensor stats...")
    
    for subject in subjects:
        folder = paths.raw_trial_dir(subject, "complexTasks")
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
            
        calculate_magnetic_distortions(plates, mag_distortions)
        calculate_world_mags(plates, subject, subject_world_mags)
        
        if subject in subject_world_mags:
            global_mag = subject_world_mags[subject]
            calculate_joint_consistency(plates, global_mag, joint_consistency_stats)
            
        detect_sitting_sections(plates, subject, generate_plots)
            
        imu_folder = folder / "imu data"
        if not imu_folder.exists():
            print(f"Subject {subject}: No raw IMU folder found.")
            continue
            
        try:
            raw_imus = IMUTrace.from_folder(imu_folder)
        except Exception as e:
            print(f"Subject {subject}: Error loading raw IMUs: {e}")
            continue
            
        intervals, foot_sensors, min_len = find_stationary_intervals(raw_imus)
        
        if not intervals:
            if foot_sensors:
                print(f"Subject {subject}: No stationary periods > 20s found in raw data across foot sensors.")
            else:
                print(f"Subject {subject}: No foot sensors found.")
            continue
            
        if generate_plots:
            plot_stationary_intervals(subject, raw_imus, foot_sensors, intervals, min_len)
            
        calculate_baseline_noise(raw_imus, foot_sensors, intervals, noise_stats)

    print_results(mag_distortions, subject_world_mags, noise_stats, joint_consistency_stats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", action="store_true", help="Generate stationary detection plots")
    args = parser.parse_args()
    generate_sensor_stats(generate_plots=args.plots)

if __name__ == "__main__":
    main()
