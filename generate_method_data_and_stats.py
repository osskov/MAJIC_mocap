import os
from pathlib import Path
os.environ["DISABLE_TQDM"] = "True"
import argparse
import itertools
import time
import multiprocessing
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from src.toolchest.PlateTrial import PlateTrial
from src.RelativeFilterPlus import RelativeFilter
from concurrent.futures import ProcessPoolExecutor
from rich.live import Live
from rich.table import Table

# ==============================================================================
# CONFIGURATION
# ==============================================================================

JOINTS = {
    'Lumbar': ('pelvis_imu', 'torso_imu'),
    'R_Hip':  ('pelvis_imu', 'femur_r_imu'),
    'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
    'R_Ankle':('tibia_r_imu', 'calcn_r_imu'),
    'L_Hip':  ('pelvis_imu', 'femur_l_imu'),
    'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
    'L_Ankle':('tibia_l_imu', 'calcn_l_imu'),
}

METHODS = {
    'marker':      {'kind': 'marker'},
    'mag_on':      {'kind': 'filter', 'project': True,  'mag_mode': 'on'},
    'mag_off':     {'kind': 'filter', 'project': True,  'mag_mode': 'off'},
    'mag_adapt':   {'kind': 'filter', 'project': True,  'mag_mode': 'adapt'},
    # 'unprojected': {'kind': 'filter', 'project': False, 'mag_mode': 'on'},
    'ekf':         {'kind': 'ekf'},
}

# Add threshold methods for sensitivity study
# for th in np.logspace(np.log10(10), np.log10(200), 8):
#     METHODS[f'mag_adapt_{th:.2f}'] = {'kind': 'filter', 'project': True, 'mag_mode': 'adapt', 'mag_adapt_threshold': th}

SUBJECTS = [f'{i:02d}' for i in range(1, 12)]
ACTIVITIES = ['walking', 'complexTasks']
BASE_DATA_PATH = Path("data").resolve()

# ==============================================================================
# STAGE 1 — Load raw subject sensor data
# ==============================================================================

def load_raw_data(subject: str, activity: str) -> Dict[str, PlateTrial]:
    folder = BASE_DATA_PATH / f"Subject{subject}" / activity
    plates = PlateTrial.from_folder(folder)
    return plates

# ==============================================================================
# STAGE 2 — Compute joint angles for a method
# ==============================================================================

def _setup_ekf_ground_plate_(plate_trials: List[PlateTrial]) -> PlateTrial:
    """Precomputes global expected gravity and magnetic field to build a virtual parent ground plate."""
    base_plate = plate_trials[0]
    expected_gravity = np.array([0.0, 9.81, 0.0])  # Expected gravity in Z-axis
    
    # Precompute expected magnetic field as the median of all global magnetic field readings
    all_global_mags = [
        (plate.world_trace.rotations @ plate.imu_trace.mag[..., None])[..., 0]
        for plate in plate_trials if 'pelvis' in plate.name
    ]
    expected_mag = np.median(np.concatenate(all_global_mags, axis=0), axis=0)
    
    ground_plate = base_plate.copy()
    ground_plate.name = "ground"
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

def _run_relative_filter(parent_trial: PlateTrial,
                         child_trial: PlateTrial,
                         project: bool,
                         mag_mode: str,
                         gyro_std_parent: float = 0.008,
                         acc_std_parent: float = 0.052,
                         mag_std_parent: float = 0.058,
                         gyro_std_child: float = 0.008,
                         acc_std_child: float = 0.052,
                         mag_std_child: float = 0.058,
                         mag_adapt_threshold: float = 150.0) -> List[np.ndarray]:
    """Estimates joint orientations between parent and child trials using specified filter configurations."""
    parent_trial = parent_trial.copy()
    child_trial = child_trial.copy()

    # Save raw accelerometer readings before any projection is applied
    acc_p_raw = parent_trial.imu_trace.acc.copy()
    acc_c_raw = child_trial.imu_trace.acc.copy()
    
    # 1. IMU projection
    parent_offset, child_offset = None, None
    if project:
        parent_offset, child_offset, error = parent_trial.world_trace.get_joint_center(child_trial.world_trace)
        if np.mean(np.linalg.norm(error, axis=1)) > 0.05:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Warning: High joint center error ({np.mean(np.linalg.norm(error, axis=1))} m) "
                      f"between {parent_trial.name} and {child_trial.name}. Check marker placement.")
        parent_trial.imu_trace = parent_trial.project_imu_trace(parent_offset)
        child_trial.imu_trace = child_trial.project_imu_trace(child_offset)

    # 2. Magnetometer modifications
    if mag_mode == 'adapt':
        obs = _calculate_observability_metric_(parent_trial, child_trial)
        high_idx = obs > mag_adapt_threshold
        parent_trial.imu_trace.mag[high_idx] = 0.0
        child_trial.imu_trace.mag[high_idx] = 0.0
    elif mag_mode == 'off':
        parent_trial.imu_trace.mag = np.zeros_like(parent_trial.imu_trace.mag)
        child_trial.imu_trace.mag = np.zeros_like(child_trial.imu_trace.mag)
    elif mag_mode != 'on':
        raise ValueError(f"Unknown mag_mode '{mag_mode}' specified.")

    # 3. Filter execution
    joint_filter = RelativeFilter(
        gyro_std_parent=np.ones(3) * gyro_std_parent,
        gyro_std_child=np.ones(3) * gyro_std_child,
        vector_sensor_stds_parent=[np.ones(3) * acc_std_parent, np.ones(3) * mag_std_parent],
        vector_sensor_stds_child=[np.ones(3) * acc_std_child, np.ones(3) * mag_std_child],
        r_parent=parent_offset,
        r_child=child_offset
    )
    joint_filter.set_qs(Rotation.from_matrix(parent_trial.world_trace.rotations[0]), Rotation.from_matrix(child_trial.world_trace.rotations[0]))
    dt = np.mean(parent_trial.imu_trace.timestamps[1:] - parent_trial.imu_trace.timestamps[:-1])

    # Main update loop
    N = len(parent_trial)
    R_pc = np.empty((N, 3, 3), dtype=np.float64)

    for t in range(N):
        joint_filter.update(
            parent_trial.imu_trace.gyro[t], child_trial.imu_trace.gyro[t],
            [parent_trial.imu_trace.acc[t], parent_trial.imu_trace.mag[t]],
            [child_trial.imu_trace.acc[t], child_trial.imu_trace.mag[t]], dt,
            acc_p_raw=acc_p_raw[t],
            acc_c_raw=acc_c_raw[t]
        )
        R_pc[t] = joint_filter.get_R_pc()
    return R_pc

def _joint_angles_from_marker(plates: Dict[str, PlateTrial]) -> pd.DataFrame:
    all_joint_data = []
    any_plate = next(iter(plates.values()))
    timestamps = any_plate.imu_trace.timestamps
    
    for joint_name, (parent, child) in JOINTS.items():
        if parent not in plates or child not in plates:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Skipping joint {joint_name}: parent '{parent}' or child '{child}' not in loaded plates.")
            continue
        parent_plate = plates[parent]
        child_plate = plates[child]
        R_joint = np.einsum('tji,tjk->tik', parent_plate.world_trace.rotations, child_plate.world_trace.rotations)
        rotvec = Rotation.from_matrix(R_joint).as_rotvec()
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'joint_name': joint_name,
            'rx': rotvec[:, 0],
            'ry': rotvec[:, 1],
            'rz': rotvec[:, 2],
        })
        all_joint_data.append(df)
        
    return pd.concat(all_joint_data, ignore_index=True) if all_joint_data else pd.DataFrame()

def _joint_angles_from_filter(plates: Dict[str, PlateTrial], project: bool, mag_mode: str, mag_adapt_threshold: float = 150.0) -> pd.DataFrame:
    all_joint_data = []
    any_plate = next(iter(plates.values()))
    timestamps = any_plate.imu_trace.timestamps
    
    for joint_name, (parent, child) in JOINTS.items():
        if parent not in plates or child not in plates:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Skipping joint {joint_name}: parent '{parent}' or child '{child}' not in loaded plates.")
            continue
        parent_plate = plates[parent]
        child_plate = plates[child]
        R_pc = _run_relative_filter(parent_plate, child_plate, project=project, mag_mode=mag_mode, mag_adapt_threshold=mag_adapt_threshold)
        rotvec = Rotation.from_matrix(R_pc).as_rotvec()
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'joint_name': joint_name,
            'rx': rotvec[:, 0],
            'ry': rotvec[:, 1],
            'rz': rotvec[:, 2],
        })
        all_joint_data.append(df)
        
    return pd.concat(all_joint_data, ignore_index=True) if all_joint_data else pd.DataFrame()

def _joint_angles_from_ekf(plates: Dict[str, PlateTrial]) -> pd.DataFrame:
    plate_trials = list(plates.values())
    ground_plate = _setup_ekf_ground_plate_(plate_trials)
    
    segment_orientations = {}
    for plate_name, plate in plates.items():
        segment_orientations[plate_name] = _run_relative_filter(
            ground_plate, plate, project=False, mag_mode='on'    
        )
        
    all_joint_data = []
    timestamps = plate_trials[0].imu_trace.timestamps
    
    for joint_name, (parent, child) in JOINTS.items():
        if parent not in segment_orientations or child not in segment_orientations:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Skipping joint {joint_name}: parent '{parent}' or child '{child}' not in segment orientations.")
            continue
        R_parent = np.array(segment_orientations[parent])
        R_child = np.array(segment_orientations[child])
        R_joint = np.einsum('tji,tjk->tik', R_parent, R_child)
        rotvec = Rotation.from_matrix(R_joint).as_rotvec()
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'joint_name': joint_name,
            'rx': rotvec[:, 0],
            'ry': rotvec[:, 1],
            'rz': rotvec[:, 2],
        })
        all_joint_data.append(df)
        
    return pd.concat(all_joint_data, ignore_index=True) if all_joint_data else pd.DataFrame()

def compute_joint_angles(plates: Dict[str, PlateTrial], method: str) -> pd.DataFrame:
    spec = METHODS[method]
    if spec['kind'] == 'marker':
        return _joint_angles_from_marker(plates)
    if spec['kind'] == 'ekf':
        return _joint_angles_from_ekf(plates)
    return _joint_angles_from_filter(
        plates,
        project=spec['project'],
        mag_mode=spec['mag_mode'],
        mag_adapt_threshold=spec.get('mag_adapt_threshold', 150.0)
    )

# ==============================================================================
# STAGE 2 — Intermediate save/load
# ==============================================================================

def joint_angles_path(subject: str, activity: str, method: str) -> Path:
    return BASE_DATA_PATH / f"Subject{subject}" / activity / f"{method}.parquet"

def save_joint_angles(df: pd.DataFrame, subject: str, activity: str, method: str):
    path = joint_angles_path(subject, activity, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine='pyarrow')

def load_joint_angles(subject: str, activity: str, method: str) -> Optional[pd.DataFrame]:
    path = joint_angles_path(subject, activity, method)
    return pd.read_parquet(path, engine='pyarrow') if path.exists() else None

# ==============================================================================
# STAGE 3 — Statistics on joint angles
# ==============================================================================

def load_all_joint_angles(subjects: List[str], activities: List[str], methods: List[str]) -> pd.DataFrame:
    frames = []
    for subject, activity, method in itertools.product(subjects, activities, methods):
        df = load_joint_angles(subject, activity, method)
        if df is None:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Warning: missing joint angles for Subject{subject}/{activity}/{method} — skipping")
            continue
        df = df.assign(subject=f"Subject{subject}", trial_type=activity, method=method)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def compute_error_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates summary statistics using fast wide-format C-aggregations."""
    if df.empty:
        return pd.DataFrame()
        
    marker_df = df[df['method'] == 'marker']
    imu_df = df[df['method'] != 'marker']
    
    if marker_df.empty or imu_df.empty:
        return pd.DataFrame()
        
    join_cols = ['subject', 'trial_type', 'joint_name', 'timestamp']
    merged_df = pd.merge(imu_df, marker_df, on=join_cols, suffixes=('_imu', '_marker'))
    
    if merged_df.empty:
        return pd.DataFrame()

    rotvec_imu = merged_df[['rx_imu', 'ry_imu', 'rz_imu']].to_numpy()
    rotvec_marker = merged_df[['rx_marker', 'ry_marker', 'rz_marker']].to_numpy()

    r_imu = R.from_rotvec(rotvec_imu)
    r_marker = R.from_rotvec(rotvec_marker)
    r_error = r_imu * r_marker.inv()
    rotvec_error = r_error.as_rotvec()

    merged_df['X'] = rotvec_error[:, 0]
    merged_df['Y'] = rotvec_error[:, 1]
    merged_df['Z'] = rotvec_error[:, 2]
    merged_df['MAG'] = np.linalg.norm(rotvec_error, axis=1)
    
    merged_df = merged_df.rename(columns={'method_imu': 'method'})
    group_cols = ['trial_type', 'method', 'joint_name', 'subject']
    target_cols = ['MAG', 'X', 'Y', 'Z']

    # Pre-calculate squared & absolute columns for vectorized MAE/RMSE
    for col in target_cols:
        merged_df[f'{col}_sq'] = merged_df[col] ** 2
        merged_df[f'{col}_abs'] = merged_df[col].abs()

    # Fast built-in aggregations across wide format
    grouped = merged_df.groupby(group_cols)
    
    means = grouped[target_cols].mean()
    stds = grouped[target_cols].std()
    mins = grouped[target_cols].min()
    maxs = grouped[target_cols].max()
    medians = grouped[target_cols].median()
    q25s = grouped[target_cols].quantile(0.25)
    q75s = grouped[target_cols].quantile(0.75)
    maes = grouped[[f'{c}_abs' for c in target_cols]].mean().rename(columns=lambda c: c.replace('_abs', ''))
    rmses = np.sqrt(grouped[[f'{c}_sq' for c in target_cols]].mean()).rename(columns=lambda c: c.replace('_sq', ''))
    
    # MAD: median absolute deviation from median
    mads = grouped[target_cols].apply(lambda g: (g - g.median()).abs().median())

    # Build multi-index summary and melt at the very end
    summary_list = []
    metric_map = {
        'mean_rad': means, 'std_rad': stds, 'rmse_rad': rmses, 
        'mae_rad': maes, 'mad_rad': mads, 'min_rad': mins, 
        'q25_rad': q25s, 'median_rad': medians, 'q75_rad': q75s, 'max_rad': maxs
    }
    
    for metric_name, metric_df in metric_map.items():
        melted = metric_df.reset_index().melt(
            id_vars=group_cols, value_vars=target_cols, var_name='axis', value_name=metric_name
        )
        summary_list.append(melted.set_index(group_cols + ['axis']))

    summary_df = pd.concat(summary_list, axis=1).reset_index()
    return summary_df

def compute_correlation_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Generates a Pearson correlation summary using fully vectorized operations."""
    if df.empty:
        return pd.DataFrame()
        
    marker_df = df[df['method'] == 'marker']
    imu_df = df[df['method'] != 'marker']
    
    if marker_df.empty or imu_df.empty:
        return pd.DataFrame()
        
    join_cols = ['subject', 'trial_type', 'joint_name', 'timestamp']
    merged_df = pd.merge(imu_df, marker_df, on=join_cols, suffixes=('_imu', '_marker'))
    
    if merged_df.empty:
        return pd.DataFrame()
        
    merged_df = merged_df.rename(columns={'method_imu': 'method'})
    groupby_cols = ['trial_type', 'method', 'joint_name', 'subject']

    # Vectorized mean-centering across all rows
    aggs = {}
    for axis in ['x', 'y', 'z']:
        col_imu = f'r{axis}_imu'
        col_marker = f'r{axis}_marker'
        
        imu_mean = merged_df.groupby(groupby_cols)[col_imu].transform('mean')
        marker_mean = merged_df.groupby(groupby_cols)[col_marker].transform('mean')
        
        dx = merged_df[col_imu] - imu_mean
        dy = merged_df[col_marker] - marker_mean
        
        merged_df[f'cov_{axis}'] = dx * dy
        merged_df[f'var_imu_{axis}'] = dx ** 2
        merged_df[f'var_marker_{axis}'] = dy ** 2
        
        aggs[f'cov_{axis}'] = 'sum'
        aggs[f'var_imu_{axis}'] = 'sum'
        aggs[f'var_marker_{axis}'] = 'sum'

    # Single vectorized aggregation pass
    grouped = merged_df.groupby(groupby_cols).agg(aggs)

    # Compute correlation per axis
    corr_cols = {}
    for axis, upper_axis in [('x', 'X'), ('y', 'Y'), ('z', 'Z')]:
        denom = np.sqrt(grouped[f'var_imu_{axis}'] * grouped[f'var_marker_{axis}'])
        corr_cols[upper_axis] = np.where(denom > 1e-12, grouped[f'cov_{axis}'] / denom, 0.0)

    res_df = pd.DataFrame(corr_cols, index=grouped.index)
    corr_df_stacked = res_df.stack().to_frame()
    corr_df_stacked.index.names = groupby_cols + ['axis']
    corr_df_stacked.columns = ['pearson_r']
    return corr_df_stacked

# ==============================================================================
# CLI / ORCHESTRATOR
# ==============================================================================

def process_subject_activity(subject: str, activity: str, methods: List[str], shared_state: Dict):
    t_start_load = time.time()
    shared_state[(subject, activity, 'load')] = "Running"
    try:
        plates = load_raw_data(subject, activity)
        shared_state[(subject, activity, 'load_time')] = time.time() - t_start_load
        shared_state[(subject, activity, 'load')] = "Success"
        
        for method in methods:
            t_start_method = time.time()
            shared_state[(subject, activity, method)] = "Running"
            try:
                df = compute_joint_angles(plates, method)
                if df is not None and not df.empty:
                    save_joint_angles(df, subject, activity, method)
                    shared_state[(subject, activity, f"{method}_time")] = time.time() - t_start_method
                    shared_state[(subject, activity, method)] = "Success"
                else:
                    shared_state[(subject, activity, method)] = "Skipped"
            except Exception as e:
                shared_state[(subject, activity, method)] = f"Failed ({e})"
    except Exception as e:
        shared_state[(subject, activity, 'load')] = "Failed"
        for method in methods:
            shared_state[(subject, activity, method)] = "Failed"

def make_table(subjects: List[str], activities: List[str], methods: List[str], shared_state: Dict) -> Table:
    table = Table(
        title="[bold magenta]🔮 MAJIC MOCAP PIPELINE STATUS 🔮[/bold magenta]", 
        show_header=True, 
        header_style="bold cyan",
        border_style="bold blue"
    )
    table.add_column("Subject", style="bold white", justify="center")
    table.add_column("Activity", style="bold white", justify="center")
    table.add_column("Load Raw", justify="center")
    for method in methods:
        table.add_column(method, justify="center")
    table.add_column("Stats", justify="center")
    
    for subject in subjects:
        for activity in activities:
            row = [subject, activity]
            
            # Step 1: Load Data
            load_status = shared_state.get((subject, activity, 'load'), 'Pending')
            load_time = shared_state.get((subject, activity, 'load_time'), None)
            time_suffix = f" [dim]({load_time:.1f}s)[/dim]" if load_time is not None else ""
            if load_status == "Pending":
                row.append("[bold white]■[/bold white]")
            elif load_status == "Running":
                row.append("[bold yellow]■[/bold yellow]")
            elif load_status == "Success":
                row.append(f"[bold green]■[/bold green]{time_suffix}")
            else:
                row.append("[bold red]■[/bold red]")
                
            # Step 2: Generate Joint Angles for each method
            for method in methods:
                status = shared_state.get((subject, activity, method), 'Pending')
                m_time = shared_state.get((subject, activity, f"{method}_time"), None)
                m_time_suffix = f" [dim]({m_time:.1f}s)[/dim]" if m_time is not None else ""
                if status == "Pending":
                    row.append("[bold white]■[/bold white]")
                elif status == "Running":
                    row.append("[bold yellow]■[/bold yellow]")
                elif status == "Success":
                    row.append(f"[bold green]■[/bold green]{m_time_suffix}")
                elif status == "Skipped":
                    row.append("[bold yellow]■[/bold yellow]")
                else:
                    row.append("[bold red]■[/bold red]")
                    
            # Step 3: Generate Statistics
            stats_status = shared_state.get((subject, activity, 'stats'), 'Pending')
            stats_time = shared_state.get((subject, activity, 'stats_time'), None)
            s_time_suffix = f" [dim]({stats_time:.1f}s)[/dim]" if stats_time is not None else ""
            if stats_status == "Pending":
                row.append("[bold white]■[/bold white]")
            elif stats_status == "Running":
                row.append("[bold yellow]■[/bold yellow]")
            elif stats_status == "Success":
                row.append(f"[bold green]■[/bold green]{s_time_suffix}")
            else:
                row.append("[bold red]■[/bold red]")
                
            table.add_row(*row)
    return table

def main():
    parser = argparse.ArgumentParser(description="Unified Segment Orientation and Statistics Pipeline.")
    parser.add_argument("--subjects", nargs='+', default=SUBJECTS, help="Subject IDs to process.")
    parser.add_argument("--activities", nargs='+', default=ACTIVITIES, help="Activities/trials to process.")
    parser.add_argument("--methods", nargs='+', default=list(METHODS.keys()), help="Methods to process.")
    parser.add_argument("--stats-only", action="store_true", help="Skip orientation generation and only compile statistics.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    # Validate up front
    for m in args.methods:
        if m not in METHODS:
            print(f"Error: Unknown method '{m}'. Allowed: {list(METHODS.keys())}")
            return
    for a in args.activities:
        if a not in ACTIVITIES:
            print(f"Error: Unknown activity '{a}'. Allowed: {ACTIVITIES}")
            return

    # We need a shared state manager
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    
    # Initialize state
    for subject in args.subjects:
        for activity in args.activities:
            shared_state[(subject, activity, 'load')] = "Pending"
            for method in args.methods:
                shared_state[(subject, activity, method)] = "Pending"
            shared_state[(subject, activity, 'stats')] = "Pending"

    # 1. Orientation NPZ/Pickle Generation Phase
    if not args.stats_only:
        print(f"Starting parallel generation of joint angles for {len(args.subjects) * len(args.activities)} tasks using {args.workers} workers...")
        
        with Live(make_table(args.subjects, args.activities, args.methods, shared_state), refresh_per_second=4) as live:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_subject_activity, subject, activity, args.methods, shared_state) 
                           for subject in args.subjects for activity in args.activities]
                
                while any(not f.done() for f in futures):
                    time.sleep(0.25)
                    live.update(make_table(args.subjects, args.activities, args.methods, shared_state))
                
                # Retrieve results to propagate exceptions if any
                for f in futures:
                    f.result()
                    
        print("Joint angles generation phase complete.\n")

    # 2. Aggregation & Statistics Phase
    print("--- Starting Statistics Aggregation Phase ---")
    with Live(make_table(args.subjects, args.activities, args.methods, shared_state), refresh_per_second=4) as live:
        for subject in args.subjects:
            for activity in args.activities:
                # Stats fails if any of the requested methods failed or load failed
                load_success = shared_state.get((subject, activity, 'load')) == "Success"
                any_method_success = any(shared_state.get((subject, activity, method)) == "Success" for method in args.methods)
                if not load_success or not any_method_success:
                    shared_state[(subject, activity, 'stats')] = "Failed"
                    live.update(make_table(args.subjects, args.activities, args.methods, shared_state))
                    continue
                    
                t_start_stats = time.time()
                shared_state[(subject, activity, 'stats')] = "Running"
                live.update(make_table(args.subjects, args.activities, args.methods, shared_state))
                
                try:
                    frames = []
                    for method in args.methods:
                        df = load_joint_angles(subject, activity, method)
                        if df is not None:
                            df = df.assign(subject=f"Subject{subject}", trial_type=activity, method=method)
                            frames.append(df)
                    
                    if frames:
                        all_df = pd.concat(frames, ignore_index=True)
                        
                        # Save single subject stats
                        stats_df = compute_error_stats(all_df)
                        if not stats_df.empty:
                            stats_path = BASE_DATA_PATH / f"Subject{subject}" / activity / "subject_statistics.parquet"
                            stats_df.to_parquet(stats_path, engine='pyarrow')
                            
                        # Save single subject pearson
                        pearson_df = compute_correlation_stats(all_df)
                        if not pearson_df.empty:
                            pearson_path = BASE_DATA_PATH / f"Subject{subject}" / activity / "subject_pearson_correlation.parquet"
                            pearson_df.to_parquet(pearson_path, engine='pyarrow')
                            
                        shared_state[(subject, activity, 'stats_time')] = time.time() - t_start_stats
                        shared_state[(subject, activity, 'stats')] = "Success"
                    else:
                        shared_state[(subject, activity, 'stats')] = "Failed"
                except Exception:
                    shared_state[(subject, activity, 'stats')] = "Failed"
                    
                live.update(make_table(args.subjects, args.activities, args.methods, shared_state))

    # 3. Global Aggregation & Statistics Phase
    print("\n--- Starting Global Aggregation & Statistics Phase ---")
    data_file_path = BASE_DATA_PATH / "all_subject_data.parquet"
    
    all_data_df = load_all_joint_angles(args.subjects, args.activities, args.methods)
    if all_data_df.empty:
        print("Error: No data was loaded for any subject. Exiting.")
        return

    print("--- Saving concatenated all subject joint angles ---")
    BASE_DATA_PATH.mkdir(parents=True, exist_ok=True)
    all_data_df.to_parquet(BASE_DATA_PATH / "all_subject_joint_angles.parquet", engine='pyarrow')
    # Also save to old-expected path for compatibility if any scripts read it
    all_data_df.to_parquet(data_file_path, engine='pyarrow')
    print(f"Concatenated DataFrame saved to {data_file_path}")

    # Generate and save summary statistics
    print("\n--- Generating summary statistics... ---")
    stats_file_path = BASE_DATA_PATH / "all_subject_statistics.parquet"
    summary_stats_df = compute_error_stats(all_data_df)
    if not summary_stats_df.empty:
        summary_stats_df.to_parquet(stats_file_path, engine='pyarrow')
        print(f"Summary statistics saved to {stats_file_path}")
    else:
        print("Warning: Summary statistics DataFrame is empty.")

    # Generate and save Pearson Correlation summary
    print("\n--- Generating Pearson correlation summary... ---")
    pearson_corr_file_path = BASE_DATA_PATH / "all_subject_pearson_correlation.parquet"
    pearson_corr_df = compute_correlation_stats(all_data_df)
    if not pearson_corr_df.empty:
        pearson_corr_df.to_parquet(pearson_corr_file_path, engine='pyarrow')
        print(f"Pearson correlations saved to {pearson_corr_file_path}")
    else:
        print("Warning: Pearson correlation DataFrame is empty.")
        
    print("\nPipeline finished successfully!")

if __name__ == '__main__':
    main()
