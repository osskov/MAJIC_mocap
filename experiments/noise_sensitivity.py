"""
Sensitivity study for RelativeFilter noise parameters (gyro/acc/mag std), swept
across a small log-spaced grid and run through the standard relative filter for
mag_on and mag_off. mag_off skips the mag_std sweep entirely, since zeroing the
magnetometer before the filter runs drives its Jacobian block and residual to
zero (see RelativeFilter._get_measurement_update) — mag_std is provably inert.
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import argparse
import itertools
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import paths
from experiments.experiment_utils import (
    JOINTS, SUBJECTS, ACTIVITIES,
    load_raw_data, _run_relative_filter, _joint_angles_from_marker,
    compute_error_stats, save_statistics, run_tracked_grid,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

METHOD_MAG_MODES = {'mag_on': 'on', 'mag_off': 'off'}
FIXED_MAG_STD = 0.05  # passed through for mag_off, but mathematically inert

SINGLE_TEST_SUBJECT = '01'

N_STEPS = 3
STD_MIN, STD_MAX = 0.001, 0.1
GYRO_STDS = np.logspace(np.log10(STD_MIN), np.log10(STD_MAX), N_STEPS)
ACC_STDS = np.logspace(np.log10(STD_MIN), np.log10(STD_MAX), N_STEPS)
MAG_STDS = np.logspace(np.log10(STD_MIN), np.log10(STD_MAX), N_STEPS)

OUT_DIR = paths.experiment_dir("noise_sensitivity")


def build_combos(method: str) -> List[Tuple[float, float, Optional[float]]]:
    """Parent and child sensors share the same std for a given combo. mag_std is
    None (unswept) for methods that don't use the magnetometer."""
    if method == 'mag_on':
        return list(itertools.product(GYRO_STDS, ACC_STDS, MAG_STDS))
    if method == 'mag_off':
        return [(g, a, None) for g in GYRO_STDS for a in ACC_STDS]
    raise ValueError(f"Unsupported method '{method}'. Must be one of {list(METHOD_MAG_MODES)}.")


def combo_tag(gyro_std: float, acc_std: float, mag_std: Optional[float]) -> str:
    tag = f"gyro{gyro_std:.0e}_acc{acc_std:.0e}"
    if mag_std is not None:
        tag += f"_mag{mag_std:.0e}"
    return tag


def joint_angles_path(subject: str, activity: str, method: str, tag: str):
    return OUT_DIR / "joint_angles" / f"Subject{subject}" / activity / f"{method}__{tag}.parquet"

# ==============================================================================
# Joint-angle generation for one noise combo
# ==============================================================================

def joint_angles_with_noise(plates: Dict, mag_mode: str, gyro_std: float, acc_std: float, mag_std: float) -> pd.DataFrame:
    all_joint_data = []
    any_plate = next(iter(plates.values()))
    timestamps = any_plate.imu_trace.timestamps

    for joint_name, (parent, child) in JOINTS.items():
        if parent not in plates or child not in plates:
            continue
        R_pc = _run_relative_filter(
            plates[parent], plates[child], project=True, mag_mode=mag_mode,
            gyro_std_parent=gyro_std, acc_std_parent=acc_std, mag_std_parent=mag_std,
            gyro_std_child=gyro_std, acc_std_child=acc_std, mag_std_child=mag_std,
        )
        rotvec = Rotation.from_matrix(R_pc).as_rotvec()
        df = pd.DataFrame({
            'timestamp': timestamps,
            'joint_name': joint_name,
            'rx': rotvec[:, 0], 'ry': rotvec[:, 1], 'rz': rotvec[:, 2],
        })
        all_joint_data.append(df)

    return pd.concat(all_joint_data, ignore_index=True) if all_joint_data else pd.DataFrame()

# ==============================================================================
# Live-table worker: one row per (subject, activity), one column per noise combo
# ==============================================================================

def _combo_worker(row_key, stage_labels: List[str], shared_state: Dict, method: str, mag_mode: str,
                   tag_to_combo: Dict[str, Tuple[float, float, Optional[float]]]) -> pd.DataFrame:
    """Each combo is its own filter pass over all 7 joints (~minutes), so this is
    called with run_tracked_grid(per_cell=True) — one process per (subject,
    activity, combo) triple, matching how fine the original flat task list was.
    Reloading raw data per combo (~2s) is negligible next to that."""
    subject, activity = row_key

    try:
        plates = load_raw_data(subject, activity)
        marker_df = _joint_angles_from_marker(plates)
    except Exception as e:
        for tag in stage_labels:
            shared_state[(row_key, tag)] = f"Failed ({e})"
        return pd.DataFrame()
    if marker_df.empty:
        for tag in stage_labels:
            shared_state[(row_key, tag)] = "Failed"
        return pd.DataFrame()
    marker_df = marker_df.assign(subject=f"Subject{subject}", trial_type=activity, method='marker')

    row_stats = []
    for tag in stage_labels:
        gyro_std, acc_std, mag_std = tag_to_combo[tag]
        t_start = time.time()
        shared_state[(row_key, tag)] = "Running"
        try:
            effective_mag_std = mag_std if mag_std is not None else FIXED_MAG_STD
            imu_df = joint_angles_with_noise(plates, mag_mode, gyro_std, acc_std, effective_mag_std)
            if imu_df.empty:
                shared_state[(row_key, tag)] = "Skipped"
                continue

            path = joint_angles_path(subject, activity, method, tag)
            path.parent.mkdir(parents=True, exist_ok=True)
            imu_df.to_parquet(path, engine='pyarrow')

            imu_tagged = imu_df.assign(subject=f"Subject{subject}", trial_type=activity, method=method)
            stats = compute_error_stats(pd.concat([marker_df, imu_tagged], ignore_index=True))
            if not stats.empty:
                stats = stats.assign(gyro_std=gyro_std, acc_std=acc_std, mag_std=effective_mag_std,
                                      mag_swept=mag_std is not None, combo_tag=tag)
                row_stats.append(stats)

            shared_state[(row_key, f"{tag}_time")] = time.time() - t_start
            shared_state[(row_key, tag)] = "Success"
        except Exception as e:
            shared_state[(row_key, tag)] = f"Failed ({e})"

    return pd.concat(row_stats, ignore_index=True) if row_stats else pd.DataFrame()

# ==============================================================================
# Ranking
# ==============================================================================

def rank_combos(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Ranks noise combos best-to-worst by mean RMSE on the MAG (rotation-error norm) axis."""
    mag_df = stats_df[stats_df['axis'] == 'MAG']
    ranking = (
        mag_df.groupby(['combo_tag', 'gyro_std', 'acc_std', 'mag_std', 'mag_swept'], as_index=False)
        .agg(mean_rmse_mag_rad=('rmse_rad', 'mean'),
             mean_mae_mag_rad=('mae_rad', 'mean'),
             n_rows=('rmse_rad', 'size'))
        .sort_values('mean_rmse_mag_rad', ascending=True)
        .reset_index(drop=True)
    )
    ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))
    return ranking

# ==============================================================================
# CLI / ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity study for RelativeFilter noise parameters (gyro/acc/mag std)."
    )
    parser.add_argument('--all', action='store_true', help="Run on all subjects (default: single test subject).")
    parser.add_argument('--subject', default=SINGLE_TEST_SUBJECT, help="Subject to run if not --all.")
    parser.add_argument('--activities', nargs='+', default=ACTIVITIES, help="Activities to include.")
    parser.add_argument('--method', choices=list(METHOD_MAG_MODES), default='mag_on',
                         help="Filter method to sweep. 'mag_off' skips the mag_std sweep entirely "
                              "since the magnetometer measurement is zeroed before the filter runs.")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Parallel workers across all tasks.")
    args = parser.parse_args()

    mag_mode = METHOD_MAG_MODES[args.method]
    combos = build_combos(args.method)
    tags = [combo_tag(g, a, m) for g, a, m in combos]
    print(f"Method: {args.method} ({len(combos)} combos, gyro/acc std in {STD_MIN}..{STD_MAX}, {N_STEPS} log steps each"
          f"{', mag std swept too' if args.method == 'mag_on' else ''}):")
    for tag in tags:
        print(f"  {tag}")

    subjects = SUBJECTS if args.all else [args.subject]
    row_keys = [(subject, activity) for subject in subjects for activity in args.activities]

    tag_to_combo = dict(zip(tags, combos))
    worker = partial(_combo_worker, method=args.method, mag_mode=mag_mode, tag_to_combo=tag_to_combo)
    _, results = run_tracked_grid(row_keys, ['Subject', 'Activity'], tags, worker, args.workers,
                                   title=f"NOISE SENSITIVITY ({args.method})", per_cell=True)

    row_frames = [df for df in results.values() if df is not None and not df.empty]
    if not row_frames:
        print("No results produced.")
        return
    combined_stats = pd.concat(row_frames, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_tag = "all" if args.all else args.subject

    for (subject_name, activity), group in combined_stats.groupby(['subject', 'trial_type']):
        out_path = paths.ensure_parent(
            OUT_DIR / "per_subject_stats" / subject_name / activity / f"{args.method}.parquet")
        group.to_parquet(out_path, engine='pyarrow')

    # Saved under data/statistics/ like every other experiment's summary stats, but
    # named per (method, subject-scope) since mag_on/mag_off and single-subject/--all
    # runs are independent sweeps that shouldn't clobber each other.
    stats_name = f"noise_sensitivity_{args.method}_{run_tag}"
    stats_path = save_statistics(combined_stats, stats_name)
    print(f"\nSaved combined stats ({args.method}, {run_tag}) to {stats_path}")

    ranking = rank_combos(combined_stats)
    ranking_csv_path = paths.ensure_parent(OUT_DIR / f"ranking_{args.method}_{run_tag}.csv")
    ranking.to_csv(ranking_csv_path, index=False)
    paths.write_manifest(ranking_csv_path, method=args.method, run_tag=run_tag,
                         gyro_stds=GYRO_STDS.tolist(), acc_stds=ACC_STDS.tolist(),
                         mag_stds=MAG_STDS.tolist() if args.method == 'mag_on' else None)
    print(f"Saved ranking ({args.method}, {run_tag}) to {ranking_csv_path}")

    print(f"\n=== Noise Parameter Sensitivity Ranking ({args.method}, {run_tag}) — best to worst, mean RMSE MAG ===")
    print(ranking.to_string(index=False))


if __name__ == '__main__':
    main()
