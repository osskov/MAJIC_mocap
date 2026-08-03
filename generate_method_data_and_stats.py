import os
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from src.toolchest.dataset_loaders import DataLoader, parse_sto_file, parse_npz_file
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from src.RelativeFilterPlus import RelativeFilter
from concurrent.futures import ProcessPoolExecutor

JOINT_SEGMENT_DICT = {'Lumbar': ('pelvis_imu', 'torso_imu'),
                      'R_Hip': ('pelvis_imu', 'femur_r_imu'),
                      'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
                      'R_Ankle': ('tibia_r_imu', 'calcn_r_imu'),
                      'L_Hip': ('pelvis_imu', 'femur_l_imu'),
                      'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
                      'L_Ankle': ('tibia_l_imu', 'calcn_l_imu'),
                      }

SUBJECTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
METHODS = ['Marker', 'Mag On', 'Mag Adapt', 'Mag Off', 'Unprojected', 'EKF']
TRIALS = ['walking', 'complexTasks']
BASE_DATA_PATH = os.path.abspath("data")

# ==============================================================================
# PART 1: Orientation Estimation and NPZ Generation
# ==============================================================================

def _setup_ekf_ground_plate_(plate_trials: List[PlateTrial]) -> PlateTrial:
    """Precomputes global expected gravity and magnetic field to build a virtual parent ground plate."""
    base_plate = plate_trials[0]
    expected_gravity = np.array([0.0, 9.81, 0.0]) # Expected gravity in Z-axis
    
    # Precompute expected magnetic field as the median of all global magnetic field readings
    all_global_mags = [
        np.einsum('tij,tj->ti', plate.world_trace.rotations, plate.imu_trace.mag)
        for plate in plate_trials
    ]
    expected_mag = np.median(np.concatenate(all_global_mags, axis=0), axis=0)
    
    ground_plate = base_plate.copy()
    ground_plate.name = "ground_virtual_parent"
    ground_plate.world_trace.rotations = np.tile(np.eye(3), (len(base_plate), 1, 1))
    ground_plate.imu_trace.gyro = np.zeros_like(base_plate.imu_trace.gyro)
    ground_plate.imu_trace.acc = np.tile(expected_gravity, (len(base_plate), 1))
    ground_plate.imu_trace.mag = np.tile(expected_mag, (len(base_plate), 1))
    return ground_plate

def _calculate_observability_metric_(parent_trial: PlateTrial, child_trial: PlateTrial) -> np.ndarray:
    """Computes the observability metric between parent and child IMU traces."""
    da_parent = np.diff(parent_trial.imu_trace.acc, axis=0) + np.cross(parent_trial.imu_trace.gyro[1:], parent_trial.imu_trace.acc[1:])
    da_child = np.diff(child_trial.imu_trace.acc, axis=0) + np.cross(child_trial.imu_trace.gyro[1:], child_trial.imu_trace.acc[1:])
    o_parent = np.cross(parent_trial.imu_trace.acc[1:], da_parent)
    o_child = np.cross(child_trial.imu_trace.acc[1:], da_child)
    observability_metric = np.minimum(np.linalg.norm(o_parent, axis=1),
                                      np.linalg.norm(o_child, axis=1))
    return np.concatenate(([0.0], observability_metric))

def calculate_joint_angle_from_segments(parent_trial: PlateTrial,
                                        child_trial: PlateTrial,
                                        condition: str = 'mag on',
                                        gyro_std_parent: float = np.sqrt(0.01),
                                        acc_std_parent: float = np.sqrt(0.05),
                                        mag_std_parent: float = np.sqrt(0.05),
                                        gyro_std_child: float = np.sqrt(0.01),
                                        acc_std_child: float = np.sqrt(0.05),
                                        mag_std_child: float = np.sqrt(0.05),
                                        mag_adapt_threshold: float = 150.0,
                                        warmup_steps: int = 0) -> List[np.ndarray]:
    """Estimates joint orientations between parent and child trials using specified filter conditions."""
    parent_trial = parent_trial.copy()
    child_trial = child_trial.copy()
    
    # 1. IMU projection
    if condition not in ['unprojected', 'ekf']:
        parent_offset, child_offset, error = parent_trial.world_trace.get_joint_center(child_trial.world_trace)
        if np.mean(np.linalg.norm(error, axis=1)) > 0.05:
            print(f"Warning: High joint center error ({np.mean(np.linalg.norm(error, axis=1))} m) "
                  f"between {parent_trial.name} and {child_trial.name}. Check marker placement.")
        parent_trial.imu_trace = parent_trial.project_imu_trace(parent_offset)
        child_trial.imu_trace = child_trial.project_imu_trace(child_offset)

    # 2. Magnetometer modifications
    if condition == 'mag adapt':
        obs = _calculate_observability_metric_(parent_trial, child_trial)
        high_idx = obs > mag_adapt_threshold
        parent_trial.imu_trace.mag[high_idx] = 0.0
        child_trial.imu_trace.mag[high_idx] = 0.0
    elif condition == 'mag off':
        parent_trial.imu_trace.mag = np.zeros_like(parent_trial.imu_trace.mag)
        child_trial.imu_trace.mag = np.zeros_like(child_trial.imu_trace.mag)
    elif condition not in ['mag on', 'unprojected', 'ekf']:
        raise ValueError(f"Unknown condition '{condition}' specified for joint orientation estimation.")

    # 3. Filter execution
    joint_filter = RelativeFilter(
        gyro_std_parent=np.ones(3) * gyro_std_parent,
        gyro_std_child=np.ones(3) * gyro_std_child,
        vector_sensor_stds_parent=[np.ones(3) * acc_std_parent, np.ones(3) * mag_std_parent],
        vector_sensor_stds_child=[np.ones(3) * acc_std_child, np.ones(3) * mag_std_child]
    )
    joint_filter.set_qs(Rotation.from_matrix(parent_trial.world_trace.rotations[0]), Rotation.from_matrix(child_trial.world_trace.rotations[0]))
    dt = np.mean(parent_trial.imu_trace.timestamps[1:] - parent_trial.imu_trace.timestamps[:-1])

    # Warmup loop
    for _ in range(warmup_steps):
        joint_filter.update(parent_trial.imu_trace.gyro[0], child_trial.imu_trace.gyro[0],
                            [parent_trial.imu_trace.acc[0], parent_trial.imu_trace.mag[0]],
                            [child_trial.imu_trace.acc[0], child_trial.imu_trace.mag[0]], dt)

    # Main update loop
    R_pc = []
    for t in range(len(parent_trial)):
        joint_filter.update(parent_trial.imu_trace.gyro[t], child_trial.imu_trace.gyro[t],
                            [parent_trial.imu_trace.acc[t], parent_trial.imu_trace.mag[t]],
                            [child_trial.imu_trace.acc[t], child_trial.imu_trace.mag[t]], dt)
        R_pc.append(joint_filter.get_R_pc())

    return R_pc

def generate_joint_angle_npz(output_directory: str,
                              plate_trials: List[PlateTrial],
                              num_frames: int,
                              condition: str = 'Never Project') -> Tuple[float, List[str]]:
    """Generates a .npz file containing root orientations and precalculated joint angles directly."""
    num_frames = num_frames if num_frames > 0 else len(plate_trials[0])
    plate_trials = [plate[:num_frames] for plate in plate_trials]
    timestamps = plate_trials[0].imu_trace.timestamps
    data_dict = {}

    def get_plate(name_pattern):
        return next((p for p in plate_trials if name_pattern in p.name), None)

    pelvis_plate = get_plate('pelvis_imu')

    if condition == 'marker':
        if pelvis_plate:
            data_dict['pelvis_imu'] = pelvis_plate.world_trace.rotations
            
        for joint_name, (parent, child) in JOINT_SEGMENT_DICT.items():
            parent_plate = get_plate(parent)
            child_plate = get_plate(child)
            if parent_plate and child_plate:
                R_joint = np.einsum('tji,tjk->tik', parent_plate.world_trace.rotations, child_plate.world_trace.rotations)
                data_dict[joint_name] = Rotation.from_matrix(R_joint).as_rotvec()

    elif condition == 'ekf':
        segment_orientations = {}
        ground_plate = _setup_ekf_ground_plate_(plate_trials)
            
        for plate in plate_trials:
            segment_orientations[plate.name] = calculate_joint_angle_from_segments(
                ground_plate, plate, condition='ekf',
                gyro_std_parent=1e-4, acc_std_parent=1e-4, mag_std_parent=1e-4,
                project_imu=False, warmup_steps=2000
            )
            
        pelvis_key = next((k for k in segment_orientations if 'pelvis_imu' in k), None)
        if pelvis_key:
            data_dict['pelvis_imu'] = np.array(segment_orientations[pelvis_key])

        for joint_name, (parent, child) in JOINT_SEGMENT_DICT.items():
            parent_key = next((k for k in segment_orientations if parent in k), None)
            child_key = next((k for k in segment_orientations if child in k), None)
            if parent_key and child_key:
                R_parent = np.array(segment_orientations[parent_key])
                R_child = np.array(segment_orientations[child_key])
                R_joint = np.einsum('tji,tjk->tik', R_parent, R_child)
                data_dict[joint_name] = Rotation.from_matrix(R_joint).as_rotvec()

    else:
        if pelvis_plate:
            data_dict['pelvis_imu'] = pelvis_plate.world_trace.rotations

        for joint_name, (parent, child) in JOINT_SEGMENT_DICT.items():
            parent_plate = get_plate(parent)
            child_plate = get_plate(child)
            if parent_plate and child_plate:
                R_pc = calculate_joint_angle_from_segments(parent_plate, child_plate, condition)
                data_dict[joint_name] = Rotation.from_matrix(R_pc).as_rotvec()

    output_path = os.path.join(
        output_directory,
        f'{"walking" if "walking" in output_directory else "complexTasks"}_orientations_{condition.lower().replace(" ", "_")}.npz'
    )
    os.makedirs(output_directory, exist_ok=True)
    np.savez_compressed(output_path, timestamps=timestamps, **data_dict)
    return timestamps[-1], list(data_dict.keys())

def process_subject_activity(subject_num: str, activity: str, num_frames: int):
    print(f"-------Processing Subject {subject_num}, Activity {activity}...--------")
    try:
        subject_activity_folder = os.path.abspath(os.path.join("data", f"Subject{subject_num}", activity))
        plate_trials = DataLoader(subject_activity_folder).load_plate_trials(align_plate_trials=True)

        print(f"Loaded {len(plate_trials)} plate trials for Subject {subject_num}, {activity}.")
        print(f"Identified segments: {[plate.name for plate in plate_trials]}")

        for condition in METHODS:
            print(f"Generating NPZ file for Subject {subject_num}, {activity}, condition: {condition}...")
            generate_joint_angle_npz(subject_activity_folder, plate_trials, num_frames, condition.lower())
    except Exception as e:
        print(f"Failed to process Subject {subject_num}, Activity {activity}: {e}")

# ==============================================================================
# PART 2: Statistics Analysis & Downstream Aggregation
# ==============================================================================

def get_joint_traces_from_world_traces(world_traces: Dict[str, WorldTrace]) -> pd.DataFrame:
    """Calculates joint rotations (child relative to parent) from WorldTraces."""
    all_joint_data = []
    any_trace = next(iter(world_traces.values()))
    timestamps = any_trace.timestamps
    
    for joint_name, (parent_name, child_name) in JOINT_SEGMENT_DICT.items():
        parent_key = next((k for k in world_traces if parent_name in k), None)
        child_key = next((k for k in world_traces if child_name in k), None)
        
        if not parent_key or not child_key:
            continue
            
        parent_trace = world_traces[parent_key]
        child_trace = world_traces[child_key]
        
        # R_joint = R_parent.T @ R_child
        R_joint = np.einsum('tji,tjk->tik', parent_trace.rotations, child_trace.rotations)
        joint_rotations_rotvec = Rotation.from_matrix(R_joint).as_rotvec()
        
        joint_df = pd.DataFrame({
            'timestamp': timestamps,
            'angle_axis_x_rad': joint_rotations_rotvec[:, 0],
            'angle_axis_y_rad': joint_rotations_rotvec[:, 1],
            'angle_axis_z_rad': joint_rotations_rotvec[:, 2],
        })
        joint_df['joint_name'] = joint_name
        all_joint_data.append(joint_df)
        
    if not all_joint_data:
        return pd.DataFrame()
    return pd.concat(all_joint_data, ignore_index=True)

def load_joint_traces_for_subject_df(subject_id: str, 
                                     trial_type: str,
                                     methods: List[str]) -> pd.DataFrame:
    """Loads, resyncs, and processes joint traces for a specific subject/trial from precalculated NPZs."""
    plate_trials_by_method = {}
    
    try:
        # Load Marker first to define primary timestamps and get length
        marker_npz_path = os.path.abspath(os.path.join("data", subject_id, trial_type, f"{trial_type}_orientations_marker.npz"))
        if not os.path.exists(marker_npz_path):
            print(f"Error: Marker NPZ not found for {subject_id} {trial_type}. Skipping.")
            return pd.DataFrame()
            
        marker_data = np.load(marker_npz_path)
        min_length = len(marker_data['timestamps'])
        
        # Load and verify lengths for all other methods
        loaded_methods = {}
        for method in methods:
            npz_path = os.path.abspath(os.path.join("data", subject_id, trial_type, f"{trial_type}_orientations_{method.replace(' ', '_').lower()}.npz"))
            sto_path = os.path.abspath(os.path.join("data", subject_id, trial_type, f"{trial_type}_orientations_{method.replace(' ', '_').lower()}.sto"))
            
            if os.path.exists(npz_path):
                # Try loading baked joint angles from NPZ directly for speed
                data = np.load(npz_path)
                loaded_methods[method] = (data, 'npz')
                min_length = min(min_length, len(data['timestamps']))
            elif os.path.exists(sto_path):
                # Fallback to parsing STO
                world_traces = parse_sto_file(sto_path)
                loaded_methods[method] = (world_traces, 'sto')
                any_trace = next(iter(world_traces.values()))
                min_length = min(min_length, len(any_trace.timestamps))
                
        # Aggregate joint angles for each method trimmed to min_length
        all_method_dfs = []
        for method, (data, file_type) in loaded_methods.items():
            if file_type == 'npz':
                timestamps = data['timestamps'][-min_length:]
                joint_dfs = []
                for joint_name in JOINT_SEGMENT_DICT:
                    if joint_name in data:
                        rotvec = data[joint_name][-min_length:]
                        df = pd.DataFrame({
                            'timestamp': timestamps,
                            'angle_axis_x_rad': rotvec[:, 0],
                            'angle_axis_y_rad': rotvec[:, 1],
                            'angle_axis_z_rad': rotvec[:, 2],
                        })
                        df['joint_name'] = joint_name
                        joint_dfs.append(df)
                if joint_dfs:
                    joint_df = pd.concat(joint_dfs, ignore_index=True)
                else:
                    joint_df = pd.DataFrame()
            else: # sto fallback
                # Slices all world trace rotations to match min_length
                sliced_world_traces = {}
                for key, trace in data.items():
                    sliced_world_traces[key] = WorldTrace(
                        timestamps=trace.timestamps[-min_length:],
                        positions=trace.positions[-min_length:],
                        rotations=trace.rotations[-min_length:]
                    )
                joint_df = get_joint_traces_from_world_traces(sliced_world_traces)
                
            if not joint_df.empty:
                joint_df['method'] = method
                all_method_dfs.append(joint_df)

        if not all_method_dfs:
            return pd.DataFrame()
            
        final_df = pd.concat(all_method_dfs, ignore_index=True)
        final_df['subject_id'] = subject_id
        final_df['trial_type'] = trial_type
        
        # Columns normalization
        meta_cols = ['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']
        data_cols = ['angle_axis_x_rad', 'angle_axis_y_rad', 'angle_axis_z_rad']
        columns_order = [col for col in meta_cols + data_cols if col in final_df.columns]
        
        final_df = final_df.reindex(columns=columns_order)
        final_df = final_df.set_index(['subject_id', 'trial_type', 'method', 'joint_name', 'timestamp']).sort_index()
        return final_df
        
    except Exception as e:
        print(f"Error loading joint data for {subject_id}: {e}. Skipping subject.")
        return pd.DataFrame()

def get_summary_statistics(all_data_df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """Calculates summary statistics of error vectors against Marker ground truth."""
    if 'subject' in group_by and 'subject_id' in all_data_df.index.names:
        df = all_data_df.rename_axis(index={'subject_id': 'subject'})
    else:
        df = all_data_df.copy()

    index_levels = df.index.names
    try:
        df_reset = df.reset_index()
        marker_df_full = df_reset[df_reset['method'] == 'Marker']
        imu_df_full = df_reset[df_reset['method'] != 'Marker']
    except KeyError:
        return pd.DataFrame()
        
    if imu_df_full.empty:
        return pd.DataFrame()

    join_levels = [name for name in index_levels if name != 'method']
    merged_df = pd.merge(imu_df_full, marker_df_full, on=join_levels, suffixes=('_imu', '_marker'))
    
    if merged_df.empty:
        return pd.DataFrame()

    aa_cols_imu = ['angle_axis_x_rad_imu', 'angle_axis_y_rad_imu', 'angle_axis_z_rad_imu']
    aa_cols_marker = ['angle_axis_x_rad_marker', 'angle_axis_y_rad_marker', 'angle_axis_z_rad_marker']

    imu_vecs = merged_df[aa_cols_imu].values
    marker_vecs = merged_df[aa_cols_marker].values
    error_vecs = imu_vecs - marker_vecs

    error_magnitude = np.linalg.norm(error_vecs, axis=1)
    MAGNITUDE_COL_NAME = 'angle_axis_error_magnitude_rad'
    merged_df[MAGNITUDE_COL_NAME] = error_magnitude
    merged_df['error_aa_x_rad'] = error_vecs[:, 0]
    merged_df['error_aa_y_rad'] = error_vecs[:, 1]
    merged_df['error_aa_z_rad'] = error_vecs[:, 2]

    error_cols_dict = {
        MAGNITUDE_COL_NAME: 'MAG',
        'error_aa_x_rad': 'X',
        'error_aa_y_rad': 'Y',
        'error_aa_z_rad': 'Z'
    }
    
    meta_cols = [lvl for lvl in join_levels]
    if 'method_imu' in merged_df.columns:
        merged_df = merged_df.rename(columns={'method_imu': 'method'})
    
    if 'method' not in meta_cols:
        meta_cols.append('method')
        
    id_vars = [col for col in meta_cols if col in merged_df.columns]
    error_long = pd.melt(merged_df, id_vars=id_vars, value_vars=list(error_cols_dict.keys()), var_name='error_metric_name', value_name='error_rad')
    error_long['axis'] = error_long['error_metric_name'].map(error_cols_dict)
    
    final_index_cols = [col for col in group_by if col in error_long.columns]
    final_index_cols.append('axis')
    final_index_cols = list(dict.fromkeys(final_index_cols))

    set_index_cols = [col for col in final_index_cols if col in error_long.columns]
    error_long_indexed = error_long.set_index(set_index_cols)

    def q25(x): return x.quantile(0.25)
    def q75(x): return x.quantile(0.75)
    def rmse(x): return np.sqrt(np.mean(x**2))
    def mae(x): return x.abs().mean()
    def mad(x): return (x - x.median()).abs().median()

    summary_df = error_long_indexed['error_rad'].groupby(level=set_index_cols).agg(
        [np.mean, np.std, rmse, mae, mad, np.min, q25, np.median, q75, np.max]
    )
    summary_df.columns = ['mean_rad', 'std_rad', 'rmse_rad', 'mae_rad', 'mad_rad', 'min_rad', 'q25_rad', 'median_rad', 'q75_rad', 'max_rad']
    return summary_df

def get_pearson_correlation_summary(all_data_df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """Generates a Pearson correlation summary comparing predicted joint angles vs Marker."""
    if 'subject' in group_by and 'subject_id' in all_data_df.index.names:
        df = all_data_df.rename_axis(index={'subject_id': 'subject'})
    else:
        df = all_data_df.copy()

    index_levels = df.index.names
    try:
        df_reset = df.reset_index()
        marker_df_full = df_reset[df_reset['method'] == 'Marker']
        imu_df_full = df_reset[df_reset['method'] != 'Marker']
    except KeyError:
        return pd.DataFrame()

    join_levels = [name for name in index_levels if name != 'method']
    merged_df = pd.merge(imu_df_full, marker_df_full, on=join_levels, suffixes=('_imu', '_marker'))

    if merged_df.empty:
        return pd.DataFrame()

    if 'method_imu' in merged_df.columns:
        merged_df = merged_df.rename(columns={'method_imu': 'method'})
        
    meta_cols = [lvl for lvl in join_levels]
    if 'method' not in meta_cols:
        meta_cols.append('method')

    groupby_cols = [col for col in group_by if col in merged_df.columns]
    
    def calculate_pair_correlation(group, col_pairs):
        corrs = {}
        for axis_name, (col_imu, col_marker) in col_pairs.items():
            if len(group) > 1:
                corr_matrix = np.corrcoef(group[col_imu], group[col_marker])
                corrs[axis_name] = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
            else:
                corrs[axis_name] = 0.0
        return pd.Series(corrs)

    col_pairs = {
        'X': ('angle_axis_x_rad_imu', 'angle_axis_x_rad_marker'),
        'Y': ('angle_axis_y_rad_imu', 'angle_axis_y_rad_marker'),
        'Z': ('angle_axis_z_rad_imu', 'angle_axis_z_rad_marker')
    }

    corr_df = merged_df.groupby(groupby_cols).apply(calculate_pair_correlation, col_pairs=col_pairs)
    corr_df_stacked = corr_df.stack().to_frame()
    corr_df_stacked.index.names = groupby_cols + ['axis']
    corr_df_stacked.columns = ['pearson_r']
    return corr_df_stacked

# ==============================================================================
# PART 3: Core Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Segment Orientation and Statistics Pipeline.")
    parser.add_argument("--stats-only", action="store_true", help="Skip orientation generation and only compile statistics.")
    args = parser.parse_args()

    # 1. Orientation NPZ Generation Phase
    if not args.stats_only:
        num_frames = -1
        tasks = []
        for subject_num in SUBJECTS:
            for activity in TRIALS:
                tasks.append((subject_num, activity, num_frames))

        print(f"Starting parallel generation of NPZ files for {len(tasks)} tasks...")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_subject_activity, *task) for task in tasks]
            for future in futures:
                future.result()
        print("NPZ generation phase complete.\n")

    # 2. Aggregation & Statistics Phase
    print("--- Starting Statistics Aggregation Phase ---")
    data_file_path = os.path.join(BASE_DATA_PATH, "all_subject_data.pkl")
    
    tasks = []
    # Note: We aggregate across ALL subjects (Subject01-Subject11) and trials
    for subject_num in SUBJECTS:
        subject_id = f"Subject{subject_num}"
        for activity in TRIALS:
            tasks.append((subject_id, activity, METHODS))

    all_subject_dfs = []
    print(f"Starting parallel load of precalculated joint data for {len(tasks)} tasks...")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(load_joint_traces_for_subject_df, *task) for task in tasks]
        for future in futures:
            subject_df = future.result()
            if not subject_df.empty:
                all_subject_dfs.append(subject_df)

    if not all_subject_dfs:
        print("Error: No data was loaded for any subject. Exiting.")
        return

    print("--- Concatenating all subject trials ---")
    all_data_df = pd.concat(all_subject_dfs)
    all_data_df.to_pickle(data_file_path)
    print(f"Concatenated DataFrame saved to {data_file_path}")

    # Generate and save summary statistics
    print("\n--- Generating summary statistics... ---")
    stats_file_path = os.path.join(BASE_DATA_PATH, "all_subject_statistics.pkl")
    summary_stats_df = get_summary_statistics(
        all_data_df,
        group_by=['trial_type', 'method', 'joint_name', 'subject']
    )
    summary_stats_df.to_pickle(stats_file_path)
    summary_stats_df.to_csv(os.path.join(BASE_DATA_PATH, "all_subject_statistics.csv"))
    print(f"Summary statistics saved to {stats_file_path}")

    # Generate and save Pearson Correlation summary
    print("\n--- Generating Pearson correlation summary... ---")
    pearson_corr_file_path = os.path.join(BASE_DATA_PATH, "all_subject_pearson_correlation.pkl")
    pearson_corr_df = get_pearson_correlation_summary(
        all_data_df,
        group_by=['trial_type', 'method', 'joint_name', 'subject']
    )
    pearson_corr_df.to_pickle(pearson_corr_file_path)
    pearson_corr_df.to_csv(os.path.join(BASE_DATA_PATH, "all_subject_pearson_correlation.csv"))
    print(f"Pearson correlations saved to {pearson_corr_file_path}")
    print("\nPipeline finished successfully!")

if __name__ == '__main__':
    main()
