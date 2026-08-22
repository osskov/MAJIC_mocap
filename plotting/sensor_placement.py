"""
Figures for experiments/sensor_placement.py: what moving the IMU along the segment costs.

Reads ONLY the tables that experiment wrote. Nothing here reloads a trial, reruns a filter or
refits a joint centre, so no number in a figure can disagree with the parquet beside it — the
same tier discipline plotting/joint_dof.py and plotting/build_quality.py follow, and for the same
reason: a figure that recomputes is a second implementation nobody diffs.

    python -m plotting.sensor_placement --dataset imove
    python -m plotting.sensor_placement --dataset imove --figures grid lever

Five figures, each answering one question the console report answers in prose:

  grid      The substitution itself. One 3x3 panel per knee — parent placement against child
            placement — under each projection, on a shared colour scale so the collapse from
            'none' to 'mocap' is a change in the picture and not a change in the legend.

  lever     Error against distance from sensor to joint centre, one point per cell, with the
            within-trial slope. This is the mechanism figure: if the lines flatten as the
            projection improves, the lever arm was the mechanism.

  flip      The falsification test. The same segment's three placements ranked against the joint
            above it and the joint below it. Lines that cross are the lever arm; parallel lines
            would be a sensor-quality story.

  spread    What a practitioner pays: per joint, the within-trial best-to-worst gap under each
            projection, with the cohort spread behind it.

  mag       The magnetometer's benefit against how high off the floor the sensor sat, which is
            where this lab's distortion comes from.

Every figure is drawn from the per-trial tables rather than from the pooled summary, because the
quantity that matters — best minus worst placement in the SAME trial — is destroyed by pooling
first (see `experiments.sensor_placement._spread_per_cell`).
"""
import argparse
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
from experiments.sensor_placement import (METRICS, PROJECTIONS, SINGLE_PLACEMENT,
                                          _segment_roles, _spread_per_cell, dataset_dir,
                                          load_trial_table)
from experiments.global_assumptions import DATASETS
from plotting import utils as plot_utils

PLOTS_DIR = paths.plots_dir("sensor_placement")

FIGURES = ('grid', 'lever', 'flip', 'spread', 'mag')

# One colour per projection, ordered as the arms are: less information about the joint centre,
# warmer. Fixed here rather than drawn from a palette so a projection is the same colour in every
# figure and in every dataset.
PROJECTION_COLORS = {'none': '#b4432f', 'inertial': '#d99a2b', 'mocap': '#3f6fa8'}
PROJECTION_LABELS = {'none': 'no projection',
                     'inertial': 'projected, IMU-only joint centre',
                     'mocap': 'projected, marker joint centre'}

# Proximal to distal within a segment, matching the physical layout. SINGLE_PLACEMENT last: it is
# the pelvis and the feet, which have no height to speak of.
PLACEMENT_ORDER = ['High', 'Mid', 'Low', SINGLE_PLACEMENT]


def _joint_order(frame: pd.DataFrame) -> List[str]:
    """Proximal to distal, then anything unrecognised, so every figure sorts the same way."""
    order = ['R_Hip', 'L_Hip', 'R_Knee', 'L_Knee', 'R_Ankle', 'L_Ankle']
    present = list(dict.fromkeys(frame['joint']))
    return [j for j in order if j in present] + sorted(set(present) - set(order))


def _n_trials(frame: pd.DataFrame) -> int:
    return frame.groupby(['subject', 'trial'], observed=True).ngroups


def _order_placements(values) -> List[str]:
    present = set(values)
    return [p for p in PLACEMENT_ORDER if p in present] + sorted(present - set(PLACEMENT_ORDER))

# ==============================================================================
# 1. The substitution grid
# ==============================================================================


def plot_grid(dataset: str, stats: pd.DataFrame, metric: str, mag: str = 'on',
              save: bool = True, show: bool = False) -> None:
    """Parent placement x child placement, one panel per (joint, projection).

    Only joints with a choice on BOTH sides get a panel — a hip or an ankle is a 1x3 strip, which
    is a bar chart wearing a heatmap's clothes and belongs in `spread` instead.
    """
    frame = stats[stats.mag == mag]
    both_sided = [joint for joint, rows in frame.groupby('joint', observed=True)
                  if rows.parent_placement.nunique() > 1 and rows.child_placement.nunique() > 1]
    joints = [j for j in _joint_order(frame) if j in both_sided]
    projections = [p for p in PROJECTIONS if p in set(frame.projection)]
    if not joints or not projections:
        print("  grid: no joint has a choice of placement on both sides.")
        return

    # GridSpec with a narrow trailing column for the colourbar, rather than a figure-level
    # `fig.colorbar(ax=axes)`: that kind of colourbar is invisible to tight_layout, which
    # `finalize_and_save_plot` calls, so it warns and then draws the bar straight through the
    # last panel. Given its own axes it is laid out like any other.
    fig = plt.figure(figsize=(3.4 * len(projections) + 1.6, 3.1 * len(joints)))
    grid = fig.add_gridspec(len(joints), len(projections) + 1,
                            width_ratios=[1] * len(projections) + [0.09])
    axes = [[fig.add_subplot(grid[row, column]) for column in range(len(projections))]
            for row in range(len(joints))]
    colorbar_axis = fig.add_subplot(grid[:, -1])
    # ONE colour scale across every panel. Per-panel scaling would renormalise the collapse away:
    # the projected panels would look exactly as varied as the unprojected one, which is the
    # opposite of the finding.
    pivots: Dict[tuple, pd.DataFrame] = {}
    for joint in joints:
        for projection in projections:
            cell = frame[(frame.joint == joint) & (frame.projection == projection)]
            pivots[(joint, projection)] = cell.pivot_table(
                index='parent_placement', columns='child_placement', values=metric,
                aggfunc='median', observed=True)
    finite = [v.to_numpy() for v in pivots.values() if not v.empty]
    vmax = float(np.nanpercentile(np.concatenate([a.ravel() for a in finite]), 98)) if finite else 1

    for row, joint in enumerate(joints):
        for column, projection in enumerate(projections):
            ax = axes[row][column]
            pivot = pivots[(joint, projection)]
            if pivot.empty:
                ax.set_axis_off()
                continue
            pivot = pivot.reindex(index=_order_placements(pivot.index),
                                  columns=_order_placements(pivot.columns))
            image = ax.imshow(pivot.to_numpy(), cmap='Reds', vmin=0, vmax=vmax)
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.to_numpy()[i, j]
                    if np.isfinite(value):
                        ax.text(j, i, f"{value:.1f}", ha='center', va='center', fontsize=10,
                                color='white' if value > 0.6 * vmax else '#222222')
            ax.set_xticks(range(pivot.shape[1]), pivot.columns, fontsize=9)
            ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=9)
            if row == 0:
                ax.set_title(PROJECTION_LABELS[projection], fontsize=11)
            if column == 0:
                ax.set_ylabel(f"{joint}\nparent placement", fontsize=10)
            if row == len(joints) - 1:
                ax.set_xlabel('child placement', fontsize=10)
            ax.grid(False)
    fig.colorbar(image, cax=colorbar_axis, label=f"median {metric}")

    plot_utils.finalize_and_save_plot(
        fig, f"Every placement of both sensors — {dataset}",
        f"sensor_placement_grid_{dataset}_{mag}.png", PLOTS_DIR, save=save, show=show,
        epilog=f"median over {_n_trials(frame)} trials; magnetometer {mag}; shared colour scale",
        caption=(
            f"Joint-angle error for every combination of where the two IMUs sat on their "
            f"segments. Rows are the proximal segment's sensor, columns the distal one's; the "
            f"diagonal is the placement-matched setup a careful technician would produce. All "
            f"panels share one colour scale, so the flattening from left to right is the "
            f"projection removing the lever-arm term and not a rescaled legend. The reference "
            f"for every cell is the markers on that cell's own two sensors, so the comparison "
            f"is controlled: same subject, same trial, same motion, same filter, different "
            f"mounting point. NOT shown: the absolute level, which is inflated for every cell "
            f"by an accelerometer std tuned on a different dataset — read the contrast between "
            f"cells, not the numbers themselves."))

# ==============================================================================
# 2. The mechanism: error against lever arm
# ==============================================================================


def plot_lever(dataset: str, merged: pd.DataFrame, metric: str, save: bool = True,
               show: bool = False) -> None:
    """Error against total distance from the two sensors to the joint centre."""
    if merged.empty:
        print("  lever: no geometry rows matched the stats rows.")
        return
    projections = [p for p in PROJECTIONS if p in set(merged.projection)]
    fig, axes = plt.subplots(1, len(projections), figsize=(4.4 * len(projections), 4.4),
                             squeeze=False, sharey=True, sharex=True)
    frame = merged[merged.mag == 'on']

    for column, projection in enumerate(projections):
        ax = axes[0][column]
        rows = frame[frame.projection == projection]
        if rows.empty:
            ax.set_axis_off()
            continue
        ax.scatter(rows['lever_sum_mm'], rows[metric], s=4, alpha=0.12,
                   color=PROJECTION_COLORS[projection], edgecolors='none')
        # Binned median rather than a regression line: the cloud is heteroscedastic and heavily
        # overplotted at the low end, and a least-squares line through it would be dragged by the
        # tail of cells whose joint-centre fit failed.
        bins = np.linspace(rows['lever_sum_mm'].quantile(0.01),
                           rows['lever_sum_mm'].quantile(0.99), 12)
        grouped = rows.groupby(pd.cut(rows['lever_sum_mm'], bins), observed=True)[metric]
        centres = [interval.mid for interval in grouped.median().index]
        ax.plot(centres, grouped.median().to_numpy(), color='#222222', linewidth=2,
                marker='o', markersize=4, label='binned median')
        ax.set_title(PROJECTION_LABELS[projection], fontsize=11)
        if column == len(projections) // 2:
            ax.set_xlabel('|r parent| + |r child| (mm)')
        if column == 0:
            ax.set_ylabel(metric)
            ax.legend(frameon=False, fontsize=9)
    # The tail is the unprojected arm's failures; clipping keeps the projected panels readable
    # without hiding that they exist, since the scatter still runs off the top.
    axes[0][0].set_ylim(0, float(np.nanpercentile(frame[metric], 99)))

    plot_utils.finalize_and_save_plot(
        fig, f"Is it the lever arm? — {dataset}",
        f"sensor_placement_lever_{dataset}.png", PLOTS_DIR, save=save, show=show,
        epilog=f"one point per (trial, joint, placement pair); {_n_trials(frame)} trials; "
               f"magnetometer on; y clipped at p99",
        caption=(
            f"Every cell's error against how far its two sensors sat from the joint centre. An "
            f"unprojected accelerometer carries alpha x r + omega x (omega x r) as pure error, "
            f"so the left panel is the prediction that placement matters at all. The projection "
            f"subtracts exactly that term: if it works, the slope flattens, and the residual "
            f"slope on the right is what placement costs for reasons OTHER than the lever arm. "
            f"The middle panel is the deployable case — the same projection with the joint "
            f"centre estimated from the IMUs alone — and it sits between them because a distal "
            f"sensor makes that estimate worse at the same time as it makes the correction "
            f"matter more. The lever arm is measured from the markers in all three panels, "
            f"including the one that does not use it."))

# ==============================================================================
# 3. The falsification test
# ==============================================================================


def plot_flip(dataset: str, merged: pd.DataFrame, metric: str, save: bool = True,
              show: bool = False) -> None:
    """The same segment's placements ranked against the joint above and the joint below."""
    if merged.empty:
        print("  flip: no geometry rows matched the stats rows.")
        return
    frame = merged[merged.mag == 'on']
    records = []
    for (subject, trial), cell in frame.groupby(['subject', 'trial'], observed=True):
        for projection in PROJECTIONS:
            arm = cell[cell.projection == projection]
            if arm.empty:
                continue
            for segment, proximal, distal in _segment_roles(arm):
                for role, rows in (('proximal joint', proximal), ('distal joint', distal)):
                    for placement, value in rows.groupby('placement',
                                                         observed=True)[metric].mean().items():
                        records.append({'segment': segment, 'projection': projection,
                                        'role': role, 'placement': placement, metric: value,
                                        'subject': subject, 'trial': trial})
    if not records:
        print("  flip: no segment appears on both sides of a joint.")
        return
    table = pd.DataFrame(records)
    segments = sorted(set(table.segment))
    projections = [p for p in PROJECTIONS if p in set(table.projection)]

    fig, axes = plt.subplots(len(projections), len(segments),
                             figsize=(2.9 * len(segments), 3.0 * len(projections)),
                             squeeze=False, sharey='row')
    for row, projection in enumerate(projections):
        for column, segment in enumerate(segments):
            ax = axes[row][column]
            subset = table[(table.projection == projection) & (table.segment == segment)]
            for role, style in (('proximal joint', dict(marker='o', linestyle='-')),
                                ('distal joint', dict(marker='s', linestyle='--'))):
                rows = subset[subset.role == role]
                if rows.empty:
                    continue
                order = _order_placements(rows.placement)
                medians = rows.groupby('placement', observed=True)[metric].median().reindex(order)
                ax.plot(range(len(order)), medians.to_numpy(),
                        color=PROJECTION_COLORS[projection], alpha=1.0 if role.startswith('prox')
                        else 0.55, label=role if (row == 0 and column == 0) else None, **style)
                ax.set_xticks(range(len(order)), order, fontsize=9)
            if column == 0:
                ax.set_ylabel(f"{PROJECTION_LABELS[projection]}\n{metric}", fontsize=9)
            if row == 0:
                ax.set_title(segment, fontsize=11)
    axes[0][0].legend(frameon=False, fontsize=9)

    plot_utils.finalize_and_save_plot(
        fig, f"The same sensor, judged by the joint above and the joint below — {dataset}",
        f"sensor_placement_flip_{dataset}.png", PLOTS_DIR, save=save, show=show,
        epilog=f"median over {_n_trials(frame)} trials; magnetometer on",
        caption=(
            f"A thigh sensor's distance to the hip and its distance to the knee move in "
            f"OPPOSITE directions as it slides down the segment, so a lever-arm mechanism "
            f"predicts that the best placement for one joint is the worst for the other — the "
            f"two lines in a panel should cross. Every rival explanation (that sensor is "
            f"noisier, that spot has more soft tissue, that tape job was worse) predicts "
            f"parallel lines instead, because a bad sensor is bad at both ends. The crossing is "
            f"therefore the whole test, and it needs no extra data: the same six numbers "
            f"settle it. Rows are the three projections, so the test is also a check on the "
            f"correction — the crossing should disappear as the lever arm is subtracted out."))

# ==============================================================================
# 4. What a practitioner pays
# ==============================================================================


def plot_spread(dataset: str, stats: pd.DataFrame, metric: str, save: bool = True,
                show: bool = False) -> None:
    """Within-trial best-to-worst gap, per joint, per projection."""
    spread = _spread_per_cell(stats[stats.mag == 'on'], metric)
    if spread.empty:
        print("  spread: no cell has more than one placement.")
        return
    joints = _joint_order(spread)
    projections = [p for p in PROJECTIONS if p in set(spread.projection)]
    fig, ax = plt.subplots(figsize=(1.7 * len(joints) + 3, 5.0))
    width = 0.8 / len(projections)
    x = np.arange(len(joints))

    for index, projection in enumerate(projections):
        offset = (index - (len(projections) - 1) / 2) * width
        values, spreads = [], []
        for joint in joints:
            rows = spread[(spread.joint == joint)
                          & (spread.projection == projection)]['spread_deg'].dropna()
            values.append(rows.median() if len(rows) else np.nan)
            # IQR across trials, not a standard error: these are repeated trials of the same
            # subjects, so they are not independent and an error bar implying they were would be
            # a claim this figure has no basis for.
            spreads.append([rows.median() - rows.quantile(0.25) if len(rows) else 0,
                            rows.quantile(0.75) - rows.median() if len(rows) else 0])
        ax.bar(x + offset, values, width=width, color=PROJECTION_COLORS[projection],
               label=PROJECTION_LABELS[projection], yerr=np.array(spreads).T, capsize=2,
               error_kw=dict(elinewidth=1.0, ecolor='#333333'))
    ax.set_xticks(x, joints, rotation=30, ha='right')
    ax.set_ylabel(f"worst - best placement (deg)")
    ax.legend(frameon=False, fontsize=10)

    plot_utils.finalize_and_save_plot(
        fig, f"What getting the placement wrong costs — {dataset}",
        f"sensor_placement_spread_{dataset}.png", PLOTS_DIR, save=save, show=show,
        epilog=f"bars = median over trials, whiskers = IQR; n={_n_trials(stats)} trials; "
               f"magnetometer on",
        caption=(
            f"The gap between the best and the worst placement WITHIN each trial and joint, "
            f"pooled afterwards. Computed in that order deliberately: taking the spread of "
            f"cohort medians instead averages away every trial in which placement mattered and "
            f"returns a much smaller number. This is what one subject would see by moving the "
            f"sensor and re-running, not a difference between groups of people. NOT shown: "
            f"which placement won — that is in the console report, and it is not the same "
            f"placement at every joint."))

# ==============================================================================
# 5. The magnetometer and the floor
# ==============================================================================


def plot_mag(dataset: str, merged: pd.DataFrame, metric: str, save: bool = True,
             show: bool = False) -> None:
    """The magnetometer's benefit against how high off the floor the two sensors sat."""
    if merged.empty:
        print("  mag: no geometry rows matched the stats rows.")
        return
    index = ['subject', 'trial', 'joint', 'pair', 'projection']
    pivot = merged.pivot_table(index=index, columns='mag', values=metric,
                               observed=True).dropna()
    if 'on' not in pivot.columns or 'off' not in pivot.columns:
        print("  mag: both magnetometer arms are needed and only one is present.")
        return
    benefit = (pivot['off'] - pivot['on']).rename('benefit_deg').reset_index()
    heights = merged.groupby(index, observed=True)[
        ['parent_sensor_height_m', 'child_sensor_height_m']].mean().reset_index()
    benefit = benefit.merge(heights, on=index)
    benefit['mean_height_m'] = benefit[['parent_sensor_height_m',
                                        'child_sensor_height_m']].mean(axis=1)

    projections = [p for p in PROJECTIONS if p in set(benefit.projection)]
    fig, axes = plt.subplots(1, len(projections), figsize=(4.2 * len(projections), 4.6),
                             squeeze=False, sharey=True)
    for column, projection in enumerate(projections):
        ax = axes[0][column]
        rows = benefit[benefit.projection == projection]
        if rows.empty:
            ax.set_axis_off()
            continue
        bins = np.linspace(rows['mean_height_m'].quantile(0.01),
                           rows['mean_height_m'].quantile(0.99), 10)
        grouped = rows.groupby(pd.cut(rows['mean_height_m'], bins), observed=True)['benefit_deg']
        centres = [interval.mid for interval in grouped.median().index]
        ax.scatter(rows['mean_height_m'], rows['benefit_deg'], s=4, alpha=0.10,
                   color=PROJECTION_COLORS[projection], edgecolors='none')
        ax.plot(centres, grouped.median().to_numpy(), color='#222222', linewidth=2, marker='o',
                markersize=4)
        ax.axhline(0.0, color='#888888', linewidth=1, linestyle=':')
        ax.set_title(PROJECTION_LABELS[projection], fontsize=11)
        # One x label, on the middle panel: three copies of this sentence overlap end to end.
        if column == len(projections) // 2:
            ax.set_xlabel('mean sensor height above the floor (m)')
        if column == 0:
            ax.set_ylabel('magnetometer benefit (deg, off - on)')
    limit = float(np.nanpercentile(np.abs(benefit['benefit_deg']), 97))
    axes[0][0].set_ylim(-limit, limit)

    plot_utils.finalize_and_save_plot(
        fig, f"Does the magnetometer pay off better away from the floor? — {dataset}",
        f"sensor_placement_mag_{dataset}.png", PLOTS_DIR, save=save, show=show,
        epilog=f"one point per (trial, joint, placement pair); {_n_trials(merged)} trials; "
               f"y clipped at +/- p97 of |benefit|",
        caption=(
            f"How much turning the magnetometer on improved the joint angle, against how high "
            f"the two sensors sat. Positive is better with the magnetometer. This lab's magnetic "
            f"distortion comes from the FLOOR — it scales with sensor height and is localised in "
            f"lab coordinates — so a Low placement reads a more disturbed field than a High one "
            f"on the same segment, and the prediction is an upward trend. Height is measured "
            f"along the world frame's gravity axis, taken from the accelerometers rather than "
            f"assumed. NOT a controlled comparison in the way the other figures are: height "
            f"covaries with the lever arm, so the projected panels are the ones to read — there "
            f"the lever-arm term has already been subtracted."))

# ==============================================================================
# CLI
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='imove', choices=sorted(DATASETS))
    parser.add_argument('--metric', default='rms_deg', choices=sorted(METRICS))
    parser.add_argument('--figures', nargs='+', choices=FIGURES, default=list(FIGURES))
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    stats = load_trial_table(args.dataset, 'placement_stats')
    if stats.empty:
        print(f"No tables under {dataset_dir(args.dataset)}. Run:\n"
              f"  python -m experiments.sensor_placement --dataset {args.dataset}")
        return 1
    geometry = load_trial_table(args.dataset, 'placement_geometry')
    merged = (stats.merge(geometry, on=['subject', 'trial', 'joint', 'pair'], suffixes=('', '_g'))
              if not geometry.empty else pd.DataFrame())
    print(f"Loaded {len(stats):,} cells over {_n_trials(stats)} trials.")

    save, show = not args.no_save, args.show
    if 'grid' in args.figures:
        plot_grid(args.dataset, stats, args.metric, save=save, show=show)
    if 'lever' in args.figures:
        plot_lever(args.dataset, merged, args.metric, save=save, show=show)
    if 'flip' in args.figures:
        plot_flip(args.dataset, merged, args.metric, save=save, show=show)
    if 'spread' in args.figures:
        plot_spread(args.dataset, stats, args.metric, save=save, show=show)
    if 'mag' in args.figures:
        plot_mag(args.dataset, merged, args.metric, save=save, show=show)
    print(f"\nFigures under {PLOTS_DIR}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
