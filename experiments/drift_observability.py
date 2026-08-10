"""
Segment-and-reset drift-rate estimation, run on a single subject. Within contiguous
segments where the filter has little-to-no rotational information available (quiet
sitting, or an o^J threshold), the joint-angle error is re-zeroed at the segment start
and its growth is fit against elapsed time, both linearly (bias-driven) and as sqrt(t)
(random-walk-driven). Comparing fits and drift rates across joints, mag_mode, and
segment sources (including a high-observability control) is a diagnostic for how much
of the mag-off error is genuine accumulating drift, as opposed to bounded sensor noise.

Uses the already-computed joint angles in
results/joint_angles/Subject{S}/{activity}/{method}.parquet
(produced by the main pipeline) rather than re-running the filter: this only needs
R_pc_est(t) and R_pc_true(t), not internal filter state, so there's no reason to
duplicate that computation. o^J still requires the raw IMU data (for the
obs_threshold/high_obs segment sources, and quiet_sitting's pelvis detection), so
load_raw_data is still used for that, but the filter's per-timestep EKF loop is not.

This is diagnostic/exploratory code, not a rigorous production analysis — see the
caveat printed at the end.
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import sys
import time
import argparse
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths
from experiments.experiment_utils import load_raw_data, JOINTS, _calculate_observability_metric_, run_tracked_grid

OUT_DIR = paths.experiment_dir("drift_observability")

PROXIMAL_JOINTS = {'Lumbar', 'R_Hip', 'L_Hip'}  # expected low observability / high drift

# ==============================================================================
# Loading precomputed joint angles / observability
# ==============================================================================

def load_joint_angles(subject, activity, method, joint_name):
    """Reads results/joint_angles/Subject{subject}/{activity}/{method}.parquet (method
    is 'mag_on', 'mag_off', or 'marker') and returns (timestamps, R) for the given
    joint, R being the (N,3,3) rotation matrices reconstructed from the stored
    rx/ry/rz rotvec."""
    path = paths.joint_angles_path(subject, activity, method)
    df = pd.read_parquet(path, engine='pyarrow')
    df = df[df['joint_name'] == joint_name].sort_values('timestamp').reset_index(drop=True)
    timestamps = df['timestamp'].to_numpy()
    R = Rotation.from_rotvec(df[['rx', 'ry', 'rz']].to_numpy()).as_matrix()
    return timestamps, R


def compute_obs_metric(parent_trial, child_trial):
    """Mirrors the projection step in experiment_utils._run_relative_filter
    (project=True), then computes o^J — the part of the filter pass that this analysis
    actually needs, without running the EKF itself."""
    parent_trial = parent_trial.copy()
    child_trial = child_trial.copy()
    parent_offset, child_offset, error = parent_trial.world_trace.get_joint_center(child_trial.world_trace)
    parent_trial.imu_trace = parent_trial.project_imu_trace(parent_offset)
    child_trial.imu_trace = child_trial.project_imu_trace(child_offset)
    return _calculate_observability_metric_(parent_trial, child_trial)

# ==============================================================================
# Live-table worker: one row per joint (load precomputed joint angles + o^J)
# ==============================================================================

def _joint_data_worker(row_key: str, stage_labels: List[str], shared_state: Dict,
                        plates: Dict, subject: str, activity: str, mag_mode: str):
    joint_name = row_key
    stage = stage_labels[0]
    parent_name, child_name = JOINTS[joint_name]
    if parent_name not in plates or child_name not in plates:
        shared_state[(row_key, stage)] = "Failed"
        return None

    t_start = time.time()
    shared_state[(row_key, stage)] = "Running"
    try:
        method = f"mag_{mag_mode}"
        timestamps, R_pc_est = load_joint_angles(subject, activity, method, joint_name)
        timestamps_true, R_pc_true = load_joint_angles(subject, activity, 'marker', joint_name)
        if len(timestamps) != len(timestamps_true) or not np.allclose(timestamps, timestamps_true):
            n = min(len(timestamps), len(timestamps_true))
            timestamps, R_pc_est, R_pc_true = timestamps[:n], R_pc_est[:n], R_pc_true[:n]

        obs_metric = compute_obs_metric(plates[parent_name], plates[child_name])
        if len(obs_metric) != len(timestamps):
            n = min(len(obs_metric), len(timestamps))
            obs_metric = obs_metric[:n]
            timestamps, R_pc_est, R_pc_true = timestamps[:n], R_pc_est[:n], R_pc_true[:n]

        shared_state[(row_key, f"{stage}_time")] = time.time() - t_start
        shared_state[(row_key, stage)] = "Success"
        return {'timestamps': timestamps, 'R_pc_est': R_pc_est, 'R_pc_true': R_pc_true, 'obs_metric': obs_metric}
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
        return None

# ==============================================================================
# Segment-and-reset drift-rate estimation
# ==============================================================================

def detect_quiet_sitting_segments(pelvis_plate, min_duration_sec=3.0, strict=True):
    """Detects quiet-sitting intervals from the pelvis plate alone (low height, low
    pelvis linear velocity, low pelvis gyro norm), independent of o^J. Using an
    independently-defined "true" low-observability period (the body isn't
    accelerating, so the accelerometer carries near-zero rotational information)
    avoids the circularity of using o^J to both define and explain the segments here.

    Returns a list of (start, end) sample-index tuples, falling back to looser
    thresholds if the strict criteria find nothing.
    """
    timestamps = pelvis_plate.imu_trace.timestamps
    dt = np.mean(np.diff(timestamps))
    fs = 1.0 / dt

    pelvis_y = pelvis_plate.world_trace.positions[:, 1]
    pelvis_y_vel = np.abs(np.gradient(pelvis_y, dt))
    window_n = int(0.5 * fs)
    smoothed_vel = pd.Series(pelvis_y_vel).rolling(window_n, center=True).mean().ffill().bfill().to_numpy()

    pelvis_gyro_norm = np.linalg.norm(pelvis_plate.imu_trace.gyro, axis=1)
    smoothed_gyro = pd.Series(pelvis_gyro_norm).rolling(window_n, center=True).mean().ffill().bfill().to_numpy()

    min_n = int(min_duration_sec * fs)

    def find_intervals(mask):
        intervals = []
        start = None
        for i, val in enumerate(mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if i - start >= min_n:
                    intervals.append((start, i))
                start = None
        if start is not None and len(mask) - start >= min_n:
            intervals.append((start, len(mask)))
        return intervals

    is_quiet_strict = (pelvis_y < 0.75) & (smoothed_vel < 0.01) & (smoothed_gyro < 0.05)
    intervals = find_intervals(is_quiet_strict)
    if not intervals and not strict:
        is_quiet_loose = (pelvis_y < 0.75) & (smoothed_vel < 0.03) & (smoothed_gyro < 0.15)
        intervals = find_intervals(is_quiet_loose)
    return intervals


def find_low_obs_segments(obs_metric, threshold, min_n, above=False):
    """obs_metric should already be smoothed by the caller: the raw per-sample o^J
    (a finite-difference-of-acceleration quantity) is extremely noisy sample-to-sample
    (median raw run length ~0.05s at 100Hz for a walking trial) and never sustains a
    stretch anywhere near typical segment durations without smoothing first.

    above=True finds high-o^J segments instead (obs_metric > threshold) — the control:
    if reset-then-track error growth is genuinely driven by low observability, the same
    fit applied to high-observability segments should show a much smaller/less
    confident drift, not the same clean linear growth.
    """
    mask = obs_metric > threshold if above else obs_metric < threshold
    segments = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_n:
                segments.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_n:
        segments.append((start, len(mask)))
    return segments


def smooth_obs(obs_metric, timestamps, smooth_sec):
    dt = np.mean(np.diff(timestamps))
    win_n = max(int(round(smooth_sec / dt)), 1)
    return pd.Series(obs_metric).rolling(win_n, center=True, min_periods=1).mean().to_numpy()


def segment_drift(timestamps, R_pc_est, R_pc_true, segments, obs_metric=None):
    """Fits drift growth within each pre-defined (start, end) sample-index segment.
    Segments can come from any source — o^J thresholding (find_low_obs_segments) or an
    independent criterion like detected quiet-sitting intervals. obs_metric, if given, is
    only used to report each segment's mean o^J for cross-checking against the segment
    source (e.g. verifying that quiet-sitting periods do in fact have low o^J)."""
    rows = []
    for (s, e) in segments:
        t0 = timestamps[s]
        t_rel = timestamps[s:e] - t0
        n = e - s
        R_est0_inv = R_pc_est[s].T
        R_true0_inv = R_pc_true[s].T
        drift_angle = np.empty(n)
        for k in range(n):
            delta_est = R_est0_inv @ R_pc_est[s + k]
            delta_true = R_true0_inv @ R_pc_true[s + k]
            drift_angle[k] = Rotation.from_matrix(delta_est.T @ delta_true).magnitude()

        # Linear (bias-driven) and sqrt(t) (random-walk-driven) fits through the origin
        sq_t = np.sum(t_rel ** 2)
        k_lin = np.sum(t_rel * drift_angle) / sq_t if sq_t > 0 else np.nan
        sqrt_t = np.sqrt(t_rel)
        sq_sqrt = np.sum(sqrt_t ** 2)
        k_sqrt = np.sum(sqrt_t * drift_angle) / sq_sqrt if sq_sqrt > 0 else np.nan

        ss_tot = np.sum((drift_angle - np.mean(drift_angle)) ** 2)
        r2_lin = 1 - np.sum((drift_angle - k_lin * t_rel) ** 2) / ss_tot if ss_tot > 0 else np.nan
        r2_sqrt = 1 - np.sum((drift_angle - k_sqrt * sqrt_t) ** 2) / ss_tot if ss_tot > 0 else np.nan

        rows.append({
            'start_t': t0, 'duration_s': t_rel[-1],
            'drift_rate_linear_rad_s': k_lin, 'r2_linear': r2_lin,
            'drift_rate_sqrt_rad_sqrts': k_sqrt, 'r2_sqrt': r2_sqrt,
            'mean_obs': np.mean(obs_metric[s:e]) if obs_metric is not None else np.nan,
        })
    columns = ['start_t', 'duration_s', 'drift_rate_linear_rad_s', 'r2_linear',
               'drift_rate_sqrt_rad_sqrts', 'r2_sqrt', 'mean_obs']
    return pd.DataFrame(rows, columns=columns)

# ==============================================================================
# CLI / ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', default='01')
    parser.add_argument('--activity', default='walking')
    parser.add_argument('--joints', nargs='+', default=list(JOINTS.keys()))
    parser.add_argument('--mag-mode', choices=['on', 'off'], default='off',
                         help="'off' (default) zeroes the magnetometer, matching the Mag-Off method this "
                              "analysis was designed around. 'on' uses the real magnetometer reading, "
                              "as a control for whether a given drift-rate result is specific to losing "
                              "the magnetometer.")
    parser.add_argument('--obs-threshold', type=float, default=None,
                         help="Observability threshold for the obs_threshold/high_obs segment sources. "
                              "Default (None): use each joint's own median o^J for this trial, since o^J's "
                              "scale varies a lot by joint and a fixed value (e.g. mag_adapt's 150.0) can "
                              "land entirely on one side of a joint's distribution, producing a single "
                              "degenerate segment.")
    parser.add_argument('--min-segment-sec', type=float, default=1.5, help="Minimum segment duration.")
    parser.add_argument('--smooth-sec', type=float, default=0.5,
                         help="(obs_threshold/high_obs sources only): rolling-mean window applied to o^J "
                              "before thresholding/segmenting. Raw per-sample o^J is a noisy finite-difference "
                              "quantity that rarely sustains a stretch on its own (median raw run length "
                              "~0.05s at 100Hz); smoothing surfaces the slower postural-phase structure the "
                              "segment analysis actually wants.")
    parser.add_argument('--segment-source', choices=['obs_threshold', 'quiet_sitting', 'high_obs'], default='obs_threshold',
                         help="How segments are defined. 'obs_threshold' (default) thresholds each joint's "
                              "own (smoothed) o^J below its median — same segments used to define AND "
                              "explain the drift, a real circularity concern. 'quiet_sitting' instead detects "
                              "quiet-sitting intervals from the pelvis (low height, low velocity, low gyro) "
                              "independent of o^J entirely, and uses the SAME intervals for every joint. "
                              "'high_obs' is the control for either: same smoothed-o^J segmentation as "
                              "obs_threshold, but ABOVE each joint's median instead of below — if the drift "
                              "fit found in low-observability segments is genuine, the same fit on "
                              "high-observability segments should show much weaker/less confident growth.")
    parser.add_argument('--min-sitting-sec', type=float, default=3.0,
                         help="Minimum duration for a detected quiet-sitting interval (quiet_sitting source only).")
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    if args.segment_source == 'quiet_sitting' and args.activity != 'complexTasks':
        print(f"Warning: quiet_sitting segments are expected in 'complexTasks', not '{args.activity}' — "
              f"this activity may contain no sitting periods at all.")

    out_dir = OUT_DIR / f"Subject{args.subject}" / args.activity / f"mag_{args.mag_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Subject{args.subject}/{args.activity}...")
    plates = load_raw_data(args.subject, args.activity)

    shared_sitting_segments = None
    if args.segment_source == 'quiet_sitting':
        if 'pelvis_imu' not in plates:
            print("Error: 'pelvis_imu' plate not found — cannot detect quiet-sitting segments.")
            return
        shared_sitting_segments = detect_quiet_sitting_segments(plates['pelvis_imu'],
                                                                  min_duration_sec=args.min_sitting_sec, strict=True)
        if not shared_sitting_segments:
            shared_sitting_segments = detect_quiet_sitting_segments(plates['pelvis_imu'],
                                                                      min_duration_sec=args.min_sitting_sec, strict=False)
            print("No strict quiet-sitting intervals found; falling back to looser thresholds.")
        ts = plates['pelvis_imu'].imu_trace.timestamps
        print(f"Detected {len(shared_sitting_segments)} quiet-sitting interval(s), "
              f"total {sum(ts[e - 1] - ts[s] for s, e in shared_sitting_segments):.1f}s:")
        for (s, e) in shared_sitting_segments:
            print(f"  [{ts[s]:.1f}s - {ts[e - 1]:.1f}s] ({ts[e - 1] - ts[s]:.1f}s)")
        if not shared_sitting_segments:
            print("No quiet-sitting segments found even with loose thresholds — output will be empty for all joints.")

    joints_to_run = [j for j in args.joints if j in JOINTS]
    for j in args.joints:
        if j not in JOINTS:
            print(f"Skipping unknown joint '{j}'")

    worker = partial(_joint_data_worker, plates=plates, subject=args.subject, activity=args.activity,
                      mag_mode=args.mag_mode)
    _, joint_results = run_tracked_grid(joints_to_run, ['Joint'], ['load'], worker, args.workers,
                                         title=f"DRIFT OBSERVABILITY (mag_{args.mag_mode})")

    exp3_summary = []

    for joint_name in joints_to_run:
        result = joint_results.get(joint_name)
        if result is None:
            print(f"Skipping {joint_name}: failed to load precomputed joint angles / o^J")
            continue
        timestamps, R_pc_est, R_pc_true, obs_metric = (
            result['timestamps'], result['R_pc_est'], result['R_pc_true'], result['obs_metric'])

        threshold_used = np.nan
        if args.segment_source == 'quiet_sitting':
            segments = shared_sitting_segments
        else:
            # Threshold and segmentation both operate on the smoothed o^J: the raw per-sample
            # metric is too noisy to sustain any stretch near min_segment_sec (see smooth_obs).
            obs_smoothed = smooth_obs(obs_metric, timestamps, args.smooth_sec)
            threshold_used = args.obs_threshold if args.obs_threshold is not None else float(np.median(obs_smoothed))
            min_n = max(int(round(args.min_segment_sec / np.mean(np.diff(timestamps)))), 5)
            segments = find_low_obs_segments(obs_smoothed, threshold_used, min_n,
                                              above=(args.segment_source == 'high_obs'))

        exp3_df = segment_drift(timestamps, R_pc_est, R_pc_true, segments, obs_metric=obs_metric)
        exp3_df.to_parquet(paths.ensure_parent(out_dir / f"segments_{args.segment_source}_{joint_name}.parquet"),
                           engine='pyarrow')
        frac_time_in_segments = sum(e - s for s, e in segments) / len(timestamps) if segments else 0.0
        summary_row = {
            'joint_name': joint_name, 'segment_source': args.segment_source, 'threshold_used': threshold_used,
            'n_segments': len(exp3_df), 'frac_time_in_segments': frac_time_in_segments,
            'mean_obs_in_segments': exp3_df['mean_obs'].mean() if not exp3_df.empty else np.nan,
        }
        if not exp3_df.empty:
            summary_row.update({
                'median_drift_rate_linear_rad_s': exp3_df['drift_rate_linear_rad_s'].median(),
                'median_drift_rate_sqrt_rad_sqrts': exp3_df['drift_rate_sqrt_rad_sqrts'].median(),
                'mean_r2_linear': exp3_df['r2_linear'].mean(),
                'mean_r2_sqrt': exp3_df['r2_sqrt'].mean(),
            })
        else:
            summary_row.update({
                'median_drift_rate_linear_rad_s': np.nan, 'median_drift_rate_sqrt_rad_sqrts': np.nan,
                'mean_r2_linear': np.nan, 'mean_r2_sqrt': np.nan,
            })
        exp3_summary.append(summary_row)

    exp3_summary_df = pd.DataFrame(exp3_summary)
    summary_path = paths.ensure_parent(out_dir / f"summary_{args.segment_source}.csv")
    exp3_summary_df.to_csv(summary_path, index=False)
    paths.write_manifest(summary_path, subject=f"Subject{args.subject}", activity=args.activity,
                         mag_mode=args.mag_mode, segment_source=args.segment_source,
                         obs_threshold=args.obs_threshold, min_segment_sec=args.min_segment_sec,
                         smooth_sec=args.smooth_sec)

    print(f"\n=== Segment-and-reset drift rate (segment_source={args.segment_source}) ===")
    print(exp3_summary_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = exp3_summary_df.dropna(subset=['median_drift_rate_linear_rad_s'])
    colors = ['tab:red' if j in PROXIMAL_JOINTS else 'tab:blue' for j in plot_df['joint_name']]
    ax.bar(plot_df['joint_name'], np.degrees(plot_df['median_drift_rate_linear_rad_s']) * 60, color=colors)
    ax.set_ylabel('Median drift rate (deg/min, linear model)')
    ax.set_title(f'Segment drift rate by joint ({args.segment_source}, '
                 f'Subject{args.subject}, {args.activity}, mag_{args.mag_mode})')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig_path = paths.ensure_parent(
        paths.plots_dir("drift_observability")
        / f"drift_rate_by_joint_{args.segment_source}_Subject{args.subject}_{args.activity}_mag_{args.mag_mode}.png")
    fig.savefig(fig_path, dpi=150)

    print(f"\nSaved data to {out_dir}")
    print(f"Saved figure to {fig_path}")
    print("\nCaveat: the linear-vs-sqrt(t) comparison is only meaningful for segments with enough samples "
          "(duration >> dt); very short segments will have noisy R^2 values.")


if __name__ == '__main__':
    main()
