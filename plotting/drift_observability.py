"""
Paper figures for the segment-and-reset drift-observability diagnostic
(experiments/drift_observability.py): whether Mag-Off joint-angle error is
genuine accumulating drift (vs. bounded noise), and whether it's specific to
low-observability conditions.

The low-observability arm is quiet sitting (detected from the pelvis alone, independent of
o^J); the control arm is its complement, everything that isn't quiet sitting. An earlier
version split on o^J above/below its per-joint median instead, but during gait o^J
oscillates at stride frequency and recrosses its own median roughly twice a second, so
almost no run survived a multi-second duration floor — the ankle was down to 7 segments
pooled over every subject. Sitting vs not-sitting is a postural criterion rather than a
per-sample one, so its stretches are long by construction.

Both arms are chunked into equal-length windows (WINDOW_SEC) before fitting, since R^2
depends on the length of the window it is computed over and raw not-sitting stretches run
minutes against sitting bouts of ~15s.

All figures pool across every subject with data for the given activity. Figures 1 and 3 are
also produced in a _by_side variant that keeps left and right separate instead of folding
them into one Hip/Knee/Ankle row, which is what shows whether an effect is bilateral.
  1. dumbbell_r2_by_joint.png   - R^2 (linear fit) for sitting vs not-sitting windows,
                                   grouped by joint, mag-off vs mag-on, all subjects.
  2. segment_examples.png       - best-, median-, and worst-fitting windows ranked across
                                   all subjects, so the three panels may come from three
                                   different ones (drift vs. time, mag-off vs mag-on,
                                   with linear fits).
  3. heatmap_r2.png             - mean linear-fit R^2 by joint x condition, all subjects,
                                   red-blue diverging scale (R^2 can be negative).

Uses experiments/drift_observability.py's precomputed-joint-angle loaders (reads
results/joint_angles/Subject{S}/{activity}/{method}.parquet) rather than re-running the EKF.
"""
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from plotting import utils as plot_utils  # noqa: F401  (applies the shared paper rcParams on import)
import paths
from experiments.experiment_utils import load_raw_data, JOINTS, SUBJECTS
from experiments.drift_observability import (
    load_joint_angles, compute_obs_metric, detect_quiet_sitting_segments,
    invert_segments, chunk_segments, clip_segments, segment_drift,
)

PLOTS_DIR = paths.plots_dir("drift_observability")
JOINT_ORDER = ['Lumbar', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle']
# Pools left/right into one group, matching plot_paper_figures.py's RENAME_JOINTS convention.
RENAME_JOINTS = {
    'R_Hip': 'Hip', 'L_Hip': 'Hip',
    'R_Knee': 'Knee', 'L_Knee': 'Knee',
    'R_Ankle': 'Ankle', 'L_Ankle': 'Ankle',
}
JOINT_GROUP_ORDER = ['Lumbar', 'Hip', 'Knee', 'Ankle']
# Every fit runs over a window of exactly this length, in both arms. A through-the-origin
# R^2 on a window only a few samples long is dominated by noise, and R^2 also depends on
# window length, so equal-length windows are what make sitting and not-sitting comparable.
WINDOW_SEC = 10.0
# Sitting bouts shorter than one window would contribute nothing after chunking.
MIN_SITTING_SEC = WINDOW_SEC

COLOR_OFF, COLOR_ON = sns.color_palette('Set2', 2)

# One caption per figure. A figure travels into a slide or a draft without the code that made
# it, so each says what is plotted, how to read it, and what it does NOT support.
CAPTIONS = {
    'dumbbell':
        'Whether Mag-Off joint-angle error is genuine accumulating DRIFT rather than bounded '
        'noise, and whether it is specific to low-observability conditions. Each row is a '
        f'joint; each dumbbell joins the mean R^2 of a through-the-origin linear fit over '
        f'{int(WINDOW_SEC)}s windows of quiet sitting to the same quantity over everything '
        'that is not quiet sitting, for Mag-Off and Mag-On separately. HIGH R^2 MEANS THE '
        'ERROR GROWS LINEARLY IN TIME, which is drift; low R^2 means it wanders without a '
        'trend, which is noise. Windows are equal length in both arms because R^2 depends on '
        'the length of the window it is computed over, and raw not-sitting stretches run '
        'minutes against sitting bouts of about 15 s. Sitting is detected from the pelvis '
        'alone, so the split is independent of o^J and of the joint being scored. NOT SHOWN: '
        'the SIZE of the error. R^2 measures how linear the drift is, not how large, so a '
        'joint can sit high here and still be accurate.',
    'segment_examples':
        'The individual windows behind the pooled R^2, so the summary can be checked against '
        'what it is summarizing. Three panels: the best-, median- and worst-fitting quiet-'
        f'sitting windows of {int(WINDOW_SEC)}s, RANKED ACROSS ALL SUBJECTS — so the three '
        'may come from three different people and three different joints, and the panel '
        'titles name which. Each shows joint-angle drift against time for Mag-Off and Mag-On '
        'with their linear fits. The worst panel is the important one: it is what a low R^2 '
        'actually looks like, and whether it is noise or a non-linear trend the linear fit '
        'cannot represent. NOT SHOWN: anything typical. These are order statistics chosen for '
        'being extreme, and the median panel is the only one that represents the population.',
    'heatmap':
        'The magnitude the R^2 figures deliberately leave out. Rows are joints, columns are '
        'the four condition-by-magnetometer combinations, and the cell value is the MEDIAN '
        'LINEAR DRIFT RATE in deg/s while the colour is the mean R^2 on a red-blue diverging '
        'scale. Read the two together: a large rate with a high R^2 is real sustained drift, '
        'a large rate with a low R^2 is a fit through noise and the rate is not meaningful. '
        'R^2 can be negative here because the fit is through the origin, so the scale is '
        'diverging rather than sequential. NOT SHOWN: within-condition spread. Each cell is a '
        'median over every window pooled across every subject, so a cell can be moderate '
        'while individual subjects sit at both extremes.',
}


def subjects_with_data(activity: str):
    return [s for s in SUBJECTS if paths.joint_angles_path(s, activity, "mag_off").exists()]


def compute_for_subject(subject: str, activity: str):
    print(f"Loading Subject{subject}/{activity}...")
    plates = load_raw_data(subject, activity)

    sitting_segments = detect_quiet_sitting_segments(plates['pelvis_imu'], min_duration_sec=MIN_SITTING_SEC, strict=True)
    if not sitting_segments:
        sitting_segments = detect_quiet_sitting_segments(plates['pelvis_imu'], min_duration_sec=MIN_SITTING_SEC, strict=False)
    print(f"Found {len(sitting_segments)} quiet-sitting segments")

    results = {}
    for joint_name in JOINT_ORDER:
        parent_name, child_name = JOINTS[joint_name]
        obs_metric = compute_obs_metric(plates[parent_name], plates[child_name])
        results[joint_name] = {}
        for mag_mode in ['off', 'on']:
            timestamps, R_pc_est = load_joint_angles(subject, activity, f"mag_{mag_mode}", joint_name)
            timestamps_true, R_pc_true = load_joint_angles(subject, activity, 'marker', joint_name)
            n = min(len(timestamps), len(timestamps_true), len(obs_metric))
            timestamps, R_pc_est, R_pc_true = timestamps[:n], R_pc_est[:n], R_pc_true[:n]
            obs_metric_j = obs_metric[:n]

            window_n = max(int(round(WINDOW_SEC / np.mean(np.diff(timestamps)))), 5)
            sitting = clip_segments(sitting_segments, n)
            sit_windows = chunk_segments(sitting, window_n)
            move_windows = chunk_segments(invert_segments(sitting, n, window_n), window_n)

            sit_df = segment_drift(timestamps, R_pc_est, R_pc_true, sit_windows, obs_metric=obs_metric_j)
            move_df = segment_drift(timestamps, R_pc_est, R_pc_true, move_windows, obs_metric=obs_metric_j)

            results[joint_name][mag_mode] = {
                'sit_df': sit_df, 'move_df': move_df, 'sit_windows': sit_windows,
                'timestamps': timestamps, 'R_pc_est': R_pc_est, 'R_pc_true': R_pc_true,
            }
            # mean_obs is reported for both arms as the check that this postural split really
            # does separate observability — the o^J split it replaced did so by construction.
            print(f"{joint_name} mag_{mag_mode}: "
                  f"sitting n={len(sit_df)} r2={sit_df['r2_linear'].mean():.3f} "
                  f"obs={sit_df['mean_obs'].mean():.1f} | "
                  f"not-sitting n={len(move_df)} r2={move_df['r2_linear'].mean():.3f} "
                  f"obs={move_df['mean_obs'].mean():.1f}")
    return results


def compute_all(subjects, activity):
    """Runs compute_for_subject once per subject. Loading the raw IMU data dominates the
    runtime, so this is kept separate from pooling — the same per-subject results can then
    be aggregated several ways (see pool) without paying for the load again."""
    all_results, used_subjects = {}, []
    for subject in subjects:
        print(f"\n--- Subject{subject} ---")
        try:
            all_results[subject] = compute_for_subject(subject, activity)
        except Exception as e:
            print(f"Skipping Subject{subject}: {e}")
            continue
        used_subjects.append(subject)
    return all_results, used_subjects


def pool(all_results, group_order, group_map):
    """Concatenates each group x mag_mode's per-window rows (sitting / not-sitting) across
    subjects into one DataFrame.

    group_map decides what a group is. Pass RENAME_JOINTS to fold left and right into a
    single Hip/Knee/Ankle row; pass {} to keep the two sides separate, which is what shows
    whether an effect is bilateral or one-sided."""
    pooled = {group: {mag_mode: {'sit_dfs': [], 'move_dfs': []} for mag_mode in ['off', 'on']}
              for group in group_order}
    for results in all_results.values():
        for joint in JOINT_ORDER:
            group = group_map.get(joint, joint)
            for mag_mode in ['off', 'on']:
                pooled[group][mag_mode]['sit_dfs'].append(results[joint][mag_mode]['sit_df'])
                pooled[group][mag_mode]['move_dfs'].append(results[joint][mag_mode]['move_df'])

    for group in group_order:
        for mag_mode in ['off', 'on']:
            pooled[group][mag_mode]['sit_df'] = pd.concat(pooled[group][mag_mode]['sit_dfs'], ignore_index=True)
            pooled[group][mag_mode]['move_df'] = pd.concat(pooled[group][mag_mode]['move_dfs'], ignore_index=True)
    return pooled


# ==============================================================================
# Figure 1: dumbbell plot of R^2 (linear), sitting vs not-sitting, by joint x mag mode
# ==============================================================================

def plot_dumbbell_r2(results, subject_label, activity, joint_order=JOINT_ORDER,
                     filename="dumbbell_r2_by_joint.png", save=True, show=False):
    fig, ax = plt.subplots(figsize=(7, 1.4 * len(joint_order) + 1))
    row_labels, row_ypos = [], []
    y = 0
    for joint in joint_order:
        for mag_mode, base_color in [('on', COLOR_ON), ('off', COLOR_OFF)]:
            sit_r2 = results[joint][mag_mode]['sit_df']['r2_linear'].mean()
            move_r2 = results[joint][mag_mode]['move_df']['r2_linear'].mean()
            light = sns.set_hls_values(base_color, l=0.8)
            dark = sns.set_hls_values(base_color, l=0.35)
            ax.plot([sit_r2, move_r2], [y, y], color=base_color, lw=1.5, zorder=1)
            ax.scatter([sit_r2], [y], color=light, edgecolor=dark, s=70, zorder=2)
            ax.scatter([move_r2], [y], color=dark, edgecolor=dark, s=70, zorder=2)
            row_labels.append(f"{joint}  ({'Mag On' if mag_mode == 'on' else 'Mag Off'})")
            row_ypos.append(y)
            y += 1
        y += 0.5

    ax.axvline(0, color='#999999', linestyle='--', linewidth=1, zorder=0)
    ax.set_yticks(row_ypos)
    ax.set_yticklabels(row_labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$R^2$ (linear fit)")
    sns.despine(ax=ax)

    handles = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=sns.set_hls_values(COLOR_OFF, l=0.8),
                   markeredgecolor=sns.set_hls_values(COLOR_OFF, l=0.35), markersize=9, label='Mag Off, sitting'),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=sns.set_hls_values(COLOR_OFF, l=0.35),
                   markeredgecolor=sns.set_hls_values(COLOR_OFF, l=0.35), markersize=9, label='Mag Off, not sitting'),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=sns.set_hls_values(COLOR_ON, l=0.8),
                   markeredgecolor=sns.set_hls_values(COLOR_ON, l=0.35), markersize=9, label='Mag On, sitting'),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=sns.set_hls_values(COLOR_ON, l=0.35),
                   markeredgecolor=sns.set_hls_values(COLOR_ON, l=0.35), markersize=9, label='Mag On, not sitting'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

    plot_utils.finalize_and_save_plot(
        fig, f"Segment-and-reset $R^2$: sitting vs not sitting "
             f"({int(WINDOW_SEC)}s windows, {subject_label}, {activity})",
        filename, PLOTS_DIR, save=save, show=show,
        caption=CAPTIONS['dumbbell'],
    )


# ==============================================================================
# Figure 2: best- and worst-fitting example segments
# ==============================================================================

def extract_segment_timeseries(subject_results, joint_name, seg_idx):
    out = {}
    for mag_mode in ['off', 'on']:
        r = subject_results[joint_name][mag_mode]
        s, e = r['sit_windows'][seg_idx]
        timestamps, R_pc_est, R_pc_true = r['timestamps'], r['R_pc_est'], r['R_pc_true']
        t_rel = timestamps[s:e] - timestamps[s]
        n = e - s
        R_est0_inv, R_true0_inv = R_pc_est[s].T, R_pc_true[s].T
        drift_deg = np.empty(n)
        for k in range(n):
            delta_est = R_est0_inv @ R_pc_est[s + k]
            delta_true = R_true0_inv @ R_pc_true[s + k]
            drift_deg[k] = np.degrees(Rotation.from_matrix(delta_est.T @ delta_true).magnitude())
        row = r['sit_df'].iloc[seg_idx]
        out[mag_mode] = {'t': t_rel, 'y': drift_deg,
                          'k_lin_deg_s': np.degrees(row['drift_rate_linear_rad_s']), 'r2_lin': row['r2_linear']}
    return out


def find_best_median_worst(all_results):
    """Ranks every (subject, joint, sitting window) triple for mag-off by its linear-fit
    R^2 and returns the best-, median-, and worst-fitting one as an illustrative spread
    rather than just the two extremes.

    Ranking across all subjects rather than within one means the three panels can come
    from three different subjects — the point is to show the range of fit quality the
    pooled statistics are averaging over, and restricting that to one subject understates
    it."""
    all_r2 = []
    for subject, results in all_results.items():
        for j in JOINT_ORDER:
            for idx, row in results[j]['off']['sit_df'].iterrows():
                if not np.isnan(row['r2_linear']):
                    all_r2.append((row['r2_linear'], subject, j, idx))
    all_r2.sort(key=lambda x: x[0])

    def entry(rec):
        r2, subject, joint, seg_idx = rec
        return {'r2': r2, 'subject': subject, 'joint': joint, 'seg_idx': seg_idx}

    return entry(all_r2[-1]), entry(all_r2[len(all_r2) // 2]), entry(all_r2[0]), len(all_r2)


def plot_segment_examples(all_results, subject_label, activity, save=True, show=False):
    best, median, worst, n_windows = find_best_median_worst(all_results)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=False)
    panels = [(axes[0], best, 'Best fit'), (axes[1], median, 'Median fit'), (axes[2], worst, 'Worst fit')]
    for ax, info, title in panels:
        ts = extract_segment_timeseries(all_results[info['subject']], info['joint'], info['seg_idx'])
        for mag_mode, color in [('off', COLOR_OFF), ('on', COLOR_ON)]:
            d = ts[mag_mode]
            label = 'Mag Off' if mag_mode == 'off' else 'Mag On'
            ax.plot(d['t'], d['y'], color=color, lw=1.2, alpha=0.85, label=f"{label} (observed)")
            ax.plot(d['t'], d['k_lin_deg_s'] * d['t'], color=color, lw=2, linestyle='--',
                     label=f"{label} (linear fit)")
        # Two lines: joint + subject + R^2 on one line overruns the panel width and the
        # three titles collide.
        ax.set_title(f"{title}: {info['joint']}\nSubject{info['subject']} ($R^2$={info['r2']:.2f})")
        ax.set_xlabel("Time (s)")
        sns.despine(ax=ax)
    axes[0].set_ylabel("Drift (deg)")
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=11)

    plot_utils.finalize_and_save_plot(
        fig, f"Best-, median-, and worst-fitting {int(WINDOW_SEC)}s sitting windows "
             f"of {n_windows} ({subject_label}, {activity})",
        "segment_examples.png", PLOTS_DIR, save=save, show=show,
        caption=CAPTIONS['segment_examples'],
    )


# ==============================================================================
# Figure 3: heatmap of median linear drift rate (deg/s), colored by mean R^2
# ==============================================================================

def plot_heatmap_r2(results, subject_label, activity, joint_order=JOINT_ORDER,
                    filename="heatmap_r2.png", save=True, show=False):
    columns = ['Mag Off\nsitting', 'Mag On\nsitting', 'Mag Off\nnot sitting', 'Mag On\nnot sitting']
    rate_data = np.full((len(joint_order), len(columns)), np.nan)
    r2_data = np.full((len(joint_order), len(columns)), np.nan)
    for i, joint in enumerate(joint_order):
        for j, (mag_mode, obs_key) in enumerate([('off', 'sit_df'), ('on', 'sit_df'),
                                                   ('off', 'move_df'), ('on', 'move_df')]):
            df = results[joint][mag_mode][obs_key]
            rate_data[i, j] = np.degrees(df['drift_rate_linear_rad_s']).median()
            r2_data[i, j] = df['r2_linear'].mean()

    rate_pivot = pd.DataFrame(rate_data, index=joint_order, columns=columns)
    r2_pivot = pd.DataFrame(r2_data, index=joint_order, columns=columns)
    annot = rate_pivot.map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    vmax = np.nanmax(np.abs(r2_pivot.to_numpy()))

    fig, ax = plt.subplots(figsize=(2.1 * len(columns), 1.8 * len(joint_order)))
    sns.heatmap(r2_pivot, ax=ax, annot=annot, fmt='', annot_kws={'size': 13, 'weight': 'bold'},
                cmap='RdBu', center=0, vmin=-vmax, vmax=vmax,
                cbar_kws={'label': r'Mean $R^2$ (linear fit)', 'shrink': 0.8})
    ax.grid(False)
    ax.set_ylabel("Joint")
    ax.set_xlabel("Condition")
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=0)

    plot_utils.finalize_and_save_plot(
        fig, f"Median linear drift rate (deg/s), colored by $R^2$ "
             f"({int(WINDOW_SEC)}s windows, {subject_label}, {activity})",
        filename, PLOTS_DIR, save=save, show=show,
        caption=CAPTIONS['heatmap'],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activity', default='complexTasks')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    subjects = subjects_with_data(args.activity)
    print(f"Subjects with {args.activity} data: {subjects}")
    all_results, used_subjects = compute_all(subjects, args.activity)
    subject_label = f"All subjects (N={len(used_subjects)})"

    # Two aggregations of the same per-subject results: sides folded together, and sides
    # kept apart. The split-sides version is what shows whether an effect is bilateral.
    for group_order, group_map, suffix in [
        (JOINT_GROUP_ORDER, RENAME_JOINTS, ""),
        (JOINT_ORDER, {}, "_by_side"),
    ]:
        pooled = pool(all_results, group_order, group_map)
        plot_dumbbell_r2(pooled, subject_label, args.activity, joint_order=group_order,
                         filename=f"dumbbell_r2_by_joint{suffix}.png", show=args.show)
        plot_heatmap_r2(pooled, subject_label, args.activity, joint_order=group_order,
                        filename=f"heatmap_r2{suffix}.png", show=args.show)

    print(f"\n--- Segment examples: all subjects ({args.activity}) ---")
    plot_segment_examples(all_results, subject_label, args.activity, show=args.show)


if __name__ == '__main__':
    main()
