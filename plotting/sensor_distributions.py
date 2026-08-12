"""
Figures for experiments/sensor_distributions.py: what the IMUs on each body segment
actually measure, and how observable each joint is.

Three metrics, all per-sample and all read back from the per-trial tables that experiment
wrote (nothing here reloads or re-projects raw data):

    linacc   linear acceleration magnitude, gravity removed, world frame   (m/s^2)
    magdev   magnetic field deviation from the subject's global field      (a.u., see MAG_UNIT)
    obs      o^J at the joint center, the quantity mag_adapt gates on      ((m/s^2)(m/s^3))

Figures
-------
Pooled distributions, one per metric x plot style. The three styles answer different
questions and are all cheap, so all three are drawn rather than one being picked:
  box    - medians and IQRs, the numbers that go in the text (fliers hidden; the
           impulsive-motion tail would otherwise set the axis and flatten every box)
  joy    - the SHAPE of each distribution, which is the point for magdev in particular:
           a bimodal segment means the field it sits in has two regimes, and no box plot
           shows that
  strip  - the raw sample cloud behind the summary, subsampled per category, with
           median/IQR lines computed from the FULL data

Observability is drawn two ways because it is a property of a joint, not a segment:
by joint as min(parent, child) — the joint's own o^J — and by segment split by
bordering joint, which is what shows WHICH of the two sensors limits each joint.

Time series around one sitting bout (default Subject06, complexTasks), which is where the
three metrics visibly decouple: sitting collapses observability while leaving magnetic
distortion untouched, so it is the case that motivates gating on observability rather than
on any magnetometer-based check.

Optional per-trial diagnostics (--diagnostics) for the two interval detectors, so that a
labeling failure is visible rather than silently shifting a shaded region.

Display truncations (joyplot percentiles, x-limits) are declared in JOY_LIMITS and apply
to the DRAWING only; every number quoted in text comes from
results/statistics/sensor_distributions_statistics.parquet, which is computed over
untruncated samples.
"""
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paths
from plotting import utils as plot_utils  # noqa: F401  (applies the shared paper rcParams on import)
from experiments.experiment_utils import ACTIVITIES, JOINTS, SUBJECTS
from experiments.sensor_distributions import (EXPERIMENT_DIR, MAG_UNIT, OBS_FILTER_CUTOFF_HZ,
                                              SEGMENT_ORDER, load_trial_table, segment_joint_roles)

PLOTS_DIR = paths.plots_dir("sensor_distributions")
DIAGNOSTICS_DIR = PLOTS_DIR / "diagnostics"

EXAMPLE_SUBJECT = '06'
EXAMPLE_ACTIVITY = 'complexTasks'
SITTING_PAD_S = 20.0  # context shown either side of the sitting bout

# metric -> (distribution x-axis label, time-series y-axis label, distribution title, time-series title)
# Magnetic axes are labeled in Xsens's normalized units, not µT — see MAG_UNIT.
# The time-series label is the short form on purpose: it sits on a rotated y-axis, where the
# full label runs longer than the axes are tall and gets clipped off the top of the figure.
METRICS = {
    'linacc': ('Linear acceleration magnitude (m/s²)',
               'Linear acceleration\n(m/s²)',
               'Linear acceleration by segment (gravity removed)',
               'Linear acceleration'),
    'magdev': (f'Magnetic field deviation magnitude ({MAG_UNIT}, 1 ≈ Earth field)',
               f'Field deviation\n({MAG_UNIT})',
               "Magnetic field deviation from the subject's global field, by segment",
               'Magnetic field deviation'),
}
# The filter note lives in the figure footer rather than the axis label: as an axis label it
# ran longer than the axes are tall and got clipped off the top of the time-series figures.
OBS_LABEL = 'Observability $o^J$ at joint center'
OBS_MIN_LABEL = 'Observability $o^J$\nmin(parent, child)'
OBS_EPILOG = (f"$o^J$ low-pass filtered at {OBS_FILTER_CUTOFF_HZ:.0f} Hz, zero-lag "
              f"(see sensor_distributions.smooth_observability)")

# Display-only truncation, per metric: the KDE is fit to samples below `percentile` and the
# axis capped at `xlim`. Without it a single metric's long right tail (footfall impulses for
# linacc, observability spikes) compresses every ridge into the leftmost few percent of the
# axis. Quoted statistics come from the summary parquet, which is untruncated.
JOY_LIMITS = {
    'linacc': {'percentile': 0.975, 'xlim': 15},
    'magdev': {'percentile': 0.85, 'xlim': None},
    'obs': {'percentile': 0.90, 'xlim': 300},
}

SIT_SHADE_COLOR = '#c8890a'  # amber
STAND_SHADE_COLOR = '#c9436b'  # rose — kept out of viridis's purple/blue/teal/green/yellow range
INTERVAL_COLORS = {'sitting': SIT_SHADE_COLOR, 'standing': STAND_SHADE_COLOR,
                   'foot_stationary': '#2a9d5c'}

MAX_STRIP_POINTS = 5000  # per category; see plot_stripplot
# Strip-plot x-axis clip, per metric — a display bound only: the median/IQR lines and every
# quoted statistic use all samples, and whatever falls beyond is drawn, counted and annotated
# at the axis edge (see plot_stripplot). Set per metric because the tails are not comparably
# heavy: magdev's max is ~2x its p99, while observability's is ~12x, so one shared quantile
# either wastes most of the magdev axis or leaves observability unreadable. --strip-clip
# overrides all three at once.
STRIP_CLIP_QUANTILE = 0.99
STRIP_CLIP = {'linacc': 0.99, 'magdev': 0.999, 'obs': 0.98}

# ==============================================================================
# Naming / ordering / color
# ==============================================================================

def joint_label(joint: str) -> str:
    """'R_Knee' -> 'Knee R'. Side goes last so that a sorted axis groups by joint level
    (every Knee together) instead of by side."""
    return f"{joint[2:]} {joint[0]}" if joint[:2] in ('R_', 'L_') else joint


JOINT_LABELS = {joint: joint_label(joint) for joint in JOINTS}
JOINT_ORDER = [JOINT_LABELS[joint] for joint in JOINTS]


def side_filter(names: Sequence[str], sides: str) -> List[str]:
    """Restricts an ordered segment/joint list to one side, keeping midline entries.

    Default is right-side-only: the two sides are near-mirror images (the benchmark's own
    ICC measurement puts left-vs-right of a joint at 0.51-0.56), so drawing both doubles
    every figure's rows to show the same thing twice."""
    if sides == 'both':
        return list(names)
    keep = 'R' if sides == 'right' else 'L'
    return [n for n in names if not n.endswith((' R', ' L')) or n.endswith(f" {keep}")]


def proximity_color(index: int, n: int, cmap_name: str = 'viridis',
                    lo: float = 0.08, hi: float = 0.92):
    """Maps an ordinal index (proximal=0 ... distal=n-1) to a perceptually uniform colormap.

    Not a hue sweep in HSL: equal-degree hue steps are perceptually uneven (steps through
    green look far smaller than the same steps through orange), so equally-spaced anatomical
    levels came out looking unevenly spaced. Viridis is built to avoid exactly that."""
    frac = index / (n - 1) if n > 1 else 0.0
    return plt.get_cmap(cmap_name)(lo + (hi - lo) * frac)


def color_map(order: Sequence[str]) -> Dict[str, tuple]:
    return {name: proximity_color(i, len(order)) for i, name in enumerate(order)}

# ==============================================================================
# Data reshaping
# ==============================================================================

def observability_by_segment(joint_df: pd.DataFrame, segments: Sequence[str]) -> pd.DataFrame:
    """Long-form (segment, joint, value): each segment's own observability at each joint it
    borders, taken from that joint's parent or child column as appropriate.

    This is the view that identifies the LIMITING sensor. A joint's o^J is the minimum of
    its two segments, so a low value on its own does not say which side caused it; the femur
    appearing low at the hip but high at the knee does."""
    roles = segment_joint_roles()
    frames = []
    for segment in segments:
        for joint, role in roles.get(segment, []):
            values = joint_df.loc[joint_df['joint'] == joint, f'obs_{role}']
            if values.empty:
                continue
            frames.append(pd.DataFrame({'segment': segment, 'joint': JOINT_LABELS[joint],
                                        'value': values.to_numpy()}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def as_long(df: pd.DataFrame, group_col: str, value_col: str, groups: Sequence[str]) -> pd.DataFrame:
    subset = df[df[group_col].isin(groups)]
    return pd.DataFrame({group_col: subset[group_col].to_numpy(), 'value': subset[value_col].to_numpy()})


def wide_series(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """One column per group, indexed by timestamp — for the time-series figures."""
    return df.pivot(index='timestamp', columns=group_col, values=value_col)

# ==============================================================================
# Distribution figures
# ==============================================================================

def plot_boxplot(data: pd.DataFrame, cat_col: str, order: List[str], xlabel: str, title: str,
                 filename: str, colors: Dict[str, tuple], hue_col: Optional[str] = None,
                 hue_colors: Optional[Dict[str, tuple]] = None, epilog: Optional[str] = None,
                 save: bool = True, show: bool = False) -> None:
    """Horizontal box plot: category on y, value on x, matching the joyplot's orientation.

    showfliers=False on purpose — matplotlib scales the axis to the drawn whiskers, so the
    impulsive tail of every one of these metrics would otherwise squash the boxes into a
    few pixels. The tail is not hidden from the reader: the strip plot shows it, and the
    summary parquet carries p90/p99/max."""
    if data.empty:
        print(f"No data for {filename}, skipping.")
        return

    grouped = hue_col is not None and hue_col != cat_col
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.9 * len(order) + (1.2 if grouped else 0.6))))
    if grouped:
        hue_order = [h for h in hue_colors if h in set(data[hue_col])]
        sns.boxplot(data=data, x='value', y=cat_col, order=order, hue=hue_col, hue_order=hue_order,
                    palette=hue_colors, dodge=True, showfliers=False, ax=ax, width=0.7)
        ax.legend(title=hue_col.title(), bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    else:
        sns.boxplot(data=data, x='value', y=cat_col, order=order, hue=cat_col, palette=colors,
                    dodge=False, showfliers=False, ax=ax, width=0.6, legend=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    ax.grid(axis='y', visible=False)
    sns.despine(ax=ax)
    plot_utils.finalize_and_save_plot(fig, title, filename, PLOTS_DIR, epilog=epilog,
                                      save=save, show=show)


def plot_joyplot(data: pd.DataFrame, cat_col: str, order: List[str], xlabel: str, title: str,
                 filename: str, colors: Dict[str, tuple], percentile: float = 0.85,
                 xlim: Optional[float] = None, epilog: Optional[str] = None,
                 save: bool = True, show: bool = False) -> None:
    """Ridgeline (joy) plot: one KDE per category, deliberately overlapping.

    Not routed through plot_utils.finalize_and_save_plot: the overlap comes from a negative
    subplot hspace, and that helper's tight_layout would immediately undo it. The rcParams
    override is applied through rc_context rather than sns.set_theme so that the transparent
    facecolor needed for the overlap does not leak into every later figure in the process —
    the previous version reset the theme by hand afterwards and dropped the shared paper
    style on the floor when it did.
    """
    if data.empty:
        print(f"No data for {filename}, skipping.")
        return
    # Truncate to BOTH the percentile and the x-limit, and clip the KDE's support to match.
    # The x-limit part is not cosmetic: clip_on=False (which is what lets a ridge overflow
    # vertically into its neighbour) also stops matplotlib clipping horizontally, so a curve
    # drawn out to a tail of 10^4 stays in the artist extents that bbox_inches='tight'
    # measures — which silently produced a 32-inch-wide figure with all the content squeezed
    # into its left eighth.
    upper = data['value'].quantile(percentile)
    if xlim is not None:
        upper = min(upper, xlim)
    plot_data = data[data['value'] <= upper]

    with plt.rc_context({'axes.facecolor': (0, 0, 0, 0), 'axes.grid': False,
                         'axes.spines.left': False, 'axes.spines.bottom': True}):
        grid = sns.FacetGrid(plot_data, row=cat_col, hue=cat_col, row_order=order, aspect=8,
                             height=0.8, palette=colors, sharey=True)
        # The fill and its edge stroke are separate artists in seaborn's kdeplot, and the
        # stroke ignores `alpha`, so this gives each ridge a thin fully-saturated outline in
        # its own hue over a softer fill body.
        grid.map(sns.kdeplot, 'value', bw_adjust=.5, clip=(0.0, upper), clip_on=False,
                 fill=True, alpha=0.75, linewidth=1.0)
        grid.figure.subplots_adjust(hspace=-0.7, left=0.25)
        grid.set_titles("")
        grid.set(yticks=[], ylabel="")
        grid.despine(bottom=True, left=True)
        if xlim is not None:
            grid.set(xlim=(0, xlim))
        for ax, name in zip(grid.axes.flat, order):
            ax.text(-0.05, 0.2, name, fontweight='bold', color='black', ha='right', va='center',
                    transform=ax.transAxes, fontsize=13)
        grid.set_axis_labels(xlabel, "")
        grid.figure.suptitle(title, y=1.02, fontsize=18, fontweight='bold')
        footer = (f"KDE fit to samples below {upper:.3g} "
                  f"({percentile:.0%} quantile{', x-limit' if xlim is not None and upper == xlim else ''})")
        grid.figure.text(0.99, -0.02, f"{footer}; {epilog}" if epilog else footer, ha='right',
                         va='top', fontsize=10, fontstyle='italic')

        if save:
            path = paths.ensure_parent(PLOTS_DIR / filename)
            grid.figure.savefig(path, dpi=600, bbox_inches='tight')
            print(f"Saved plot to {path}")
        if show:
            plt.show()
        plt.close(grid.figure)


def plot_stripplot(data: pd.DataFrame, cat_col: str, order: List[str], xlabel: str, title: str,
                   filename: str, colors: Dict[str, tuple], max_points: int = MAX_STRIP_POINTS,
                   seed: int = 0, clip_quantile: float = STRIP_CLIP_QUANTILE,
                   epilog: Optional[str] = None, save: bool = True, show: bool = False) -> None:
    """Horizontal strip plot with per-category downsampling — a category holding millions of
    samples is an unreadable smear and takes minutes to render.

    The median/IQR lines are computed from the FULL data, before downsampling, so the summary
    stays exact even though the cloud behind it is a subsample. Seeded, so the same inputs give
    the same figure.

    The x-axis is clipped at `clip_quantile` of the pooled samples, because all three of these
    metrics have tails running one to two orders of magnitude past their own IQR (observability
    reaches ~7e4 against a median near 300), and an axis sized to the largest sample compresses
    every distribution in the figure into its leftmost pixels. Nothing is dropped silently: the
    off-scale samples are drawn as carets pinned at the axis edge, each row is annotated with
    how many of its samples lie beyond, and the true maximum goes in the footer.
    """
    if data.empty:
        print(f"No data for {filename}, skipping.")
        return

    grouped = data.groupby(cat_col)['value']
    stats = grouped.quantile([0.25, 0.5, 0.75]).unstack()
    totals = grouped.size()
    limit = float(data['value'].quantile(clip_quantile))
    beyond = data[data['value'] > limit].groupby(cat_col)['value'].agg(['size', 'max'])
    sampled = pd.concat([group.sample(n=min(len(group), max_points), random_state=seed)
                         for _, group in data.groupby(cat_col)], ignore_index=True)
    in_range = sampled[sampled['value'] <= limit]
    off_scale = sampled[sampled['value'] > limit].assign(value=limit)

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.9 * len(order) + 0.6)))
    sns.stripplot(data=in_range, x='value', y=cat_col, order=order, hue=cat_col, palette=colors,
                  dodge=False, jitter=0.35, size=2, alpha=0.25, linewidth=0, ax=ax, legend=False)
    if not off_scale.empty:
        sns.stripplot(data=off_scale, x='value', y=cat_col, order=order, hue=cat_col, palette=colors,
                      dodge=False, jitter=0.3, size=5, alpha=0.55, linewidth=0, marker='>', ax=ax,
                      legend=False)

    for i, name in enumerate(order):
        if name not in stats.index:
            continue
        q1, median, q3 = stats.loc[name, 0.25], stats.loc[name, 0.5], stats.loc[name, 0.75]
        ax.vlines([q1, q3], i - 0.35, i + 0.35, color='black', linewidth=1.2, alpha=0.8, zorder=5)
        ax.vlines(median, i - 0.4, i + 0.4, color='black', linewidth=2.4, zorder=5)
        if name in beyond.index:
            count, largest = int(beyond.loc[name, 'size']), beyond.loc[name, 'max']
            ax.text(limit, i - 0.46, f"{count / totals[name]:.2%} beyond, max {largest:,.0f}  ",
                    ha='right', va='bottom', fontsize=9, fontstyle='italic', color='#555555')

    ax.set_xlim(min(0.0, float(data['value'].min())), limit * 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    ax.grid(axis='y', visible=False)
    sns.despine(ax=ax)
    handles = [Line2D([0], [0], color='black', linewidth=2.4, label='median (all samples)'),
               Line2D([0], [0], color='black', linewidth=1.2, alpha=0.8, label='IQR (all samples)')]
    if not off_scale.empty:
        handles.append(Line2D([0], [0], marker='>', color='none', markerfacecolor='#555555',
                              markersize=8, label='sample beyond axis'))
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)

    footer = (f"scatter subsampled to {max_points:,} samples per category; x-axis clipped at the "
              f"{clip_quantile:.1%} quantile ({limit:,.0f}), true max {data['value'].max():,.0f}")
    plot_utils.finalize_and_save_plot(fig, title, filename, PLOTS_DIR,
                                      epilog=f"{footer}; {epilog}" if epilog else footer,
                                      save=save, show=show)

# ==============================================================================
# Time series figures
# ==============================================================================

def add_fading_span(ax: plt.Axes, x0: float, x1: float, color: str, ylim: Tuple[float, float],
                    max_alpha: float = 0.28, fade: float = 1.5, n: int = 300) -> None:
    """Shades [x0, x1] with alpha ramping 0 -> max_alpha -> 0 across a `fade`-second margin
    at each edge, rather than as a hard-edged axvspan. The edges are what the fade is
    honest about: these boundaries come from a threshold on a rolling statistic, so they are
    accurate to about a window, and a crisp rectangle would assert a precision the detector
    does not have."""
    xs = np.linspace(x0 - fade, x1 + fade, n)
    ramp_in = (xs - (x0 - fade)) / fade
    ramp_out = ((x1 + fade) - xs) / fade
    rgba = np.zeros((1, n, 4))
    rgba[0, :, :3] = mcolors.to_rgb(color)
    rgba[0, :, 3] = np.clip(np.minimum(ramp_in, ramp_out), 0, 1) * max_alpha
    ax.imshow(rgba, extent=(x0 - fade, x1 + fade, ylim[0], ylim[1]), aspect='auto', zorder=0,
              interpolation='bilinear')


def shade_intervals(ax: plt.Axes, regions: Sequence[Tuple[float, float, str]]) -> List[Patch]:
    """Draws every (start, end, label) region as a fading span, holding the data limits
    fixed (imshow would otherwise expand them), and returns one legend patch per label."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    seen = []
    for start, end, label in regions:
        add_fading_span(ax, start, end, INTERVAL_COLORS.get(label, 'gray'), ylim)
        if label not in seen:
            seen.append(label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return [Patch(facecolor=INTERVAL_COLORS.get(label, 'gray'), alpha=0.45, label=label)
            for label in seen]


def plot_timeseries(rel_time: np.ndarray, series: pd.DataFrame, order: List[str],
                    colors: Dict[str, tuple], ylabel: str, xlabel: str, title: str, filename: str,
                    regions: Sequence[Tuple[float, float, str]] = (), epilog: Optional[str] = None,
                    save: bool = True, show: bool = False) -> None:
    """Lines are drawn in reverse of `order` so the most proximal (lowest-amplitude) trace
    ends up on top instead of buried under a distal one, while the legend is rebuilt in the
    original order so it still reads proximal-to-distal."""
    fig, ax = plt.subplots(figsize=(11, 5))
    lines = {}
    for name in reversed(order):
        if name not in series.columns:
            continue
        lines[name], = ax.plot(rel_time, series[name].to_numpy(), color=colors[name],
                               linewidth=1.6, label=name, alpha=0.9)

    handles = [lines[name] for name in order if name in lines]
    labels = [name for name in order if name in lines]
    for patch in shade_intervals(ax, regions):
        handles.append(patch)
        labels.append(patch.get_label())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    sns.despine(ax=ax)
    plot_utils.finalize_and_save_plot(fig, title, filename, PLOTS_DIR, epilog=epilog,
                                      save=save, show=show)


def select_sitting_window(intervals: pd.DataFrame, timestamps: np.ndarray
                          ) -> Optional[Tuple[float, float, float, List[Tuple[float, float, str]]]]:
    """Picks the median-ranked sitting bout and returns (sit_start, t0, t1, regions): the
    bout's absolute onset time, then the padded window bounds and shaded regions, both in
    seconds RELATIVE to that onset.

    Median rather than longest: the longest bout is the least representative one, and the
    figure is making a claim about what sitting does generally. `regions` carries the
    sitting bout plus any standing bout that falls inside the padded window, for shading.
    """
    sitting = intervals[(intervals['label'] == 'sitting') & (intervals['time_base'] == 'synced')]
    sitting = sitting.sort_values('start_time').reset_index(drop=True)
    if sitting.empty:
        return None

    chosen = sitting.iloc[len(sitting) // 2]
    sit_start, sit_end = chosen['start_time'], chosen['end_time']
    print(f"Using sitting bout {len(sitting) // 2 + 1} of {len(sitting)}: "
          f"t={sit_start:.1f}-{sit_end:.1f}s ({sit_end - sit_start:.1f}s)")

    t0 = max(timestamps[0], sit_start - SITTING_PAD_S) - sit_start
    t1 = min(timestamps[-1], sit_end + SITTING_PAD_S) - sit_start
    regions = [(0.0, sit_end - sit_start, 'sitting')]
    standing = intervals[(intervals['label'] == 'standing') & (intervals['time_base'] == 'synced')]
    for _, row in standing.iterrows():
        start, end = row['start_time'] - sit_start, row['end_time'] - sit_start
        if end >= t0 and start <= t1:
            regions.append((start, end, 'standing'))
    if len(regions) == 1:
        print("Note: no standing bout falls inside the padded window, so only sitting is shaded.")
    return sit_start, t0, t1, regions

# ==============================================================================
# Interval-detector diagnostics
# ==============================================================================

def plot_activity_labels(segment_df: pd.DataFrame, joint_df: pd.DataFrame, intervals: pd.DataFrame,
                         subject: str, activity: str, save: bool = True, show: bool = False) -> None:
    """Diagnostic for label_activity_intervals: the three pelvis signal norms plus pelvis
    observability, with the detected intervals shaded. This is how a labeling failure
    becomes visible — a fragmented sitting bout or a missed standing bout shows up here as
    shading that plainly disagrees with the quiet stretch underneath it.

    The observability panel takes the MINIMUM over the joint centers the pelvis borders
    (lumbar and both hips), because that is the pelvis's worst case and therefore what
    would gate the magnetometer at any of its joints."""
    pelvis = segment_df[segment_df['segment'] == 'Pelvis'].sort_values('timestamp')
    if pelvis.empty:
        print(f"No pelvis samples for Subject{subject}/{activity}, skipping activity diagnostic.")
        return

    pelvis_joints = [joint for joint, (parent, _) in JOINTS.items() if parent == 'pelvis_imu']
    pelvis_obs = joint_df[joint_df['joint'].isin(pelvis_joints)]
    obs_series = (pelvis_obs.pivot(index='timestamp', columns='joint', values='obs_parent').min(axis=1)
                  if not pelvis_obs.empty else None)

    time = pelvis['timestamp'].to_numpy() - pelvis['timestamp'].iloc[0]
    regions = [(row['start_time'] - pelvis['timestamp'].iloc[0],
                row['end_time'] - pelvis['timestamp'].iloc[0], row['label'])
               for _, row in intervals[intervals['time_base'] == 'synced'].iterrows()
               if row['label'] in ('sitting', 'standing')]

    panels = [('acc_norm', 'Acc magnitude (m/s²)', '#3b6ea5'),
              ('gyro_norm', 'Gyro magnitude (rad/s)', '#c9721f'),
              ('mag_norm', f'Mag magnitude ({MAG_UNIT})', '#7b4fa8')]
    n_panels = len(panels) + (1 if obs_series is not None else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 3.1 * n_panels), sharex=True)

    for ax, (column, ylabel, color) in zip(axes, panels):
        ax.plot(time, pelvis[column].to_numpy(), color=color, alpha=0.85, linewidth=1.0)
        ax.set_ylabel(ylabel)
        shade_intervals(ax, regions)

    if obs_series is not None:
        ax = axes[-1]
        ax.plot(obs_series.index.to_numpy() - pelvis['timestamp'].iloc[0], obs_series.to_numpy(),
                color='#b5323c', alpha=0.85, linewidth=1.0)
        ax.set_ylabel('Pelvis $o$, min over\nbordering joints')
        shade_intervals(ax, regions)

    handles = [Patch(facecolor=INTERVAL_COLORS[label], alpha=0.45, label=label)
               for label in dict.fromkeys(label for _, _, label in regions)]
    if handles:
        axes[0].legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
    axes[-1].set_xlabel('Time (s)')
    for ax in axes:
        sns.despine(ax=ax)

    plot_utils.finalize_and_save_plot(
        fig, f"Detected activity intervals — Subject{subject}, {activity}",
        f"activity_labels_Subject{subject}_{activity}.png", DIAGNOSTICS_DIR, save=save, show=show)


def plot_foot_stationary(foot_df: pd.DataFrame, intervals: pd.DataFrame, subject: str,
                         activity: str, save: bool = True, show: bool = False) -> None:
    """Diagnostic for find_foot_stationary_intervals: raw foot signal norms with the
    detected stationary periods shaded. Time base is the RAW IMU clock, not the
    mocap-synced one — these periods usually sit outside the captured mocap window, which
    is exactly why the noise floor is measured on the untrimmed traces."""
    stationary = intervals[(intervals['label'] == 'foot_stationary') & (intervals['time_base'] == 'raw')]
    if foot_df.empty or stationary.empty:
        print(f"No stationary foot periods for Subject{subject}/{activity}, skipping diagnostic.")
        return

    t_start = foot_df['timestamp'].min()
    regions = [(row['start_time'] - t_start, row['end_time'] - t_start, 'foot_stationary')
               for _, row in stationary.iterrows()]
    sensors = sorted(foot_df['sensor'].unique())
    sensor_colors = color_map(sensors)

    panels = [('acc_norm', 'Acc magnitude (m/s²)'), ('gyro_norm', 'Gyro magnitude (rad/s)'),
              ('mag_norm', f'Mag magnitude ({MAG_UNIT})')]
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 3.1 * len(panels)), sharex=True)
    for ax, (column, ylabel) in zip(axes, panels):
        for sensor in sensors:
            trace = foot_df[foot_df['sensor'] == sensor].sort_values('timestamp')
            ax.plot(trace['timestamp'].to_numpy() - t_start, trace[column].to_numpy(),
                    color=sensor_colors[sensor], alpha=0.8, linewidth=1.0, label=sensor)
        ax.set_ylabel(ylabel)
        shade_intervals(ax, regions)
        sns.despine(ax=ax)

    handles = [Line2D([0], [0], color=sensor_colors[s], linewidth=2, label=s) for s in sensors]
    handles.append(Patch(facecolor=INTERVAL_COLORS['foot_stationary'], alpha=0.45, label='both feet stationary'))
    axes[0].legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
    axes[-1].set_xlabel('Time (s, raw IMU clock)')

    plot_utils.finalize_and_save_plot(
        fig, f"Ground-anchored stationary periods — Subject{subject}, {activity}",
        f"foot_stationary_Subject{subject}_{activity}.png", DIAGNOSTICS_DIR, save=save, show=show)

# ==============================================================================
# Figure sets
# ==============================================================================

def plot_distributions(segment_df: pd.DataFrame, joint_df: pd.DataFrame, segments: List[str],
                       joint_order: List[str], plot_types: Sequence[str], caption: str,
                       strip_clip: Optional[float] = None, show: bool = False) -> None:
    clip = lambda metric: strip_clip if strip_clip is not None else STRIP_CLIP[metric]
    segment_colors = color_map(segments)
    joint_colors = color_map(joint_order)
    obs_segment_df = observability_by_segment(joint_df, segments)
    obs_min_df = as_long(joint_df.assign(joint=joint_df['joint'].map(JOINT_LABELS)),
                         'joint', 'obs_min', joint_order)

    for metric, (xlabel, _, title, _) in METRICS.items():
        data = as_long(segment_df, 'segment', metric, segments)
        full_title = f"{title}\n{caption}"
        if 'box' in plot_types:
            plot_boxplot(data, 'segment', segments, xlabel, full_title,
                         f"{metric}_by_segment_box.png", segment_colors, show=show)
        if 'joy' in plot_types:
            plot_joyplot(data, 'segment', segments, xlabel, full_title,
                         f"{metric}_by_segment_joy.png", segment_colors,
                         **JOY_LIMITS[metric], show=show)
        if 'strip' in plot_types:
            plot_stripplot(data, 'segment', segments, xlabel, full_title,
                           f"{metric}_by_segment_strip.png", segment_colors,
                           clip_quantile=clip(metric), show=show)

    if 'box' in plot_types:
        plot_boxplot(obs_segment_df, 'segment', segments, OBS_LABEL,
                     f"Observability by segment, split by bordering joint\n{caption}",
                     "observability_by_segment_box.png", segment_colors,
                     hue_col='joint', hue_colors=joint_colors, epilog=OBS_EPILOG, show=show)
    obs_title = f"Joint observability, min(parent, child)\n{caption}"
    obs_xlabel = OBS_MIN_LABEL.replace('\n', ', ')
    if 'joy' in plot_types:
        plot_joyplot(obs_min_df, 'joint', joint_order, obs_xlabel, obs_title,
                     "observability_by_joint_joy.png", joint_colors, **JOY_LIMITS['obs'],
                     epilog=OBS_EPILOG, show=show)
    if 'strip' in plot_types:
        plot_stripplot(obs_min_df, 'joint', joint_order, obs_xlabel, obs_title,
                       "observability_by_joint_strip.png", joint_colors,
                       clip_quantile=clip('obs'), epilog=OBS_EPILOG, show=show)


def plot_sitting_timeseries(subject: str, activity: str, segments: List[str],
                            joint_order: List[str], show: bool = False) -> None:
    """The three metrics through one sitting bout, on a shared window."""
    segment_df = load_trial_table('segment_samples', [subject], [activity])
    joint_df = load_trial_table('joint_samples', [subject], [activity])
    intervals = load_trial_table('intervals', [subject], [activity])
    if segment_df.empty or intervals.empty:
        print(f"No per-trial tables for Subject{subject}/{activity}; skipping time series.")
        return

    timestamps = np.sort(segment_df['timestamp'].unique())
    window = select_sitting_window(intervals, timestamps)
    if window is None:
        print(f"No sitting bout detected for Subject{subject}/{activity}; skipping time series.")
        return
    sit_start, t0, t1, regions = window

    segment_colors, joint_colors = color_map(segments), color_map(joint_order)
    label = f"Subject{subject}, {activity}"
    pad = int(SITTING_PAD_S)

    def in_window(df: pd.DataFrame) -> pd.DataFrame:
        """Re-references timestamps to sitting onset and clips to the padded window."""
        shifted = df.assign(timestamp=df['timestamp'] - sit_start)
        return shifted[(shifted['timestamp'] >= t0) & (shifted['timestamp'] <= t1)]

    seg_window = in_window(segment_df)
    joint_window = in_window(joint_df.assign(joint=joint_df['joint'].map(JOINT_LABELS)))

    for metric, (_, ylabel, _, series_title) in METRICS.items():
        series = wide_series(seg_window, 'segment', metric)
        plot_timeseries(series.index.to_numpy(), series, segments, segment_colors, ylabel,
                        'Time relative to sitting onset (s)',
                        f"{series_title} through a sitting bout (±{pad}s) — {label}",
                        f"timeseries_sitting_{metric}_Subject{subject}_{activity}.png",
                        regions=regions, show=show)

    obs_series = wide_series(joint_window, 'joint', 'obs_min')
    plot_timeseries(obs_series.index.to_numpy(), obs_series, joint_order, joint_colors,
                    OBS_MIN_LABEL, 'Time relative to sitting onset (s)',
                    f"Joint observability through a sitting bout (±{pad}s) — {label}",
                    f"timeseries_sitting_observability_Subject{subject}_{activity}.png",
                    regions=regions, epilog=OBS_EPILOG, show=show)


def plot_diagnostics(subjects: List[str], activities: List[str], show: bool = False) -> None:
    for subject in subjects:
        for activity in activities:
            intervals = load_trial_table('intervals', [subject], [activity])
            if intervals.empty:
                continue
            segment_df = load_trial_table('segment_samples', [subject], [activity])
            joint_df = load_trial_table('joint_samples', [subject], [activity])
            foot_df = load_trial_table('foot_samples', [subject], [activity])
            plot_activity_labels(segment_df, joint_df, intervals, subject, activity, show=show)
            plot_foot_stationary(foot_df, intervals, subject, activity, show=show)

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--activities', nargs='+', default=ACTIVITIES)
    parser.add_argument('--sides', choices=['right', 'left', 'both'], default='right',
                        help="Which side's segments/joints to draw (default: right; see side_filter).")
    parser.add_argument('--plot-types', nargs='+', choices=['box', 'joy', 'strip'],
                        default=['box', 'joy', 'strip'])
    parser.add_argument('--example-subject', default=EXAMPLE_SUBJECT,
                        help="Subject for the sitting-bout time series.")
    parser.add_argument('--example-activity', default=EXAMPLE_ACTIVITY)
    parser.add_argument('--strip-clip', type=float, default=None, metavar='QUANTILE',
                        help="Override the strip plots' x-axis clip for every metric (defaults are "
                             "per metric, see STRIP_CLIP). Samples beyond the clip are still drawn "
                             "at the axis edge and counted in the annotation, so tightening this "
                             "hides nothing.")
    parser.add_argument('--diagnostics', action='store_true',
                        help="Also draw the per-trial interval-detector diagnostics.")
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    segments = side_filter(SEGMENT_ORDER, args.sides)
    joint_order = side_filter(JOINT_ORDER, args.sides)

    print(f"Loading per-sample tables from {EXPERIMENT_DIR}...")
    segment_df = load_trial_table('segment_samples', args.subjects, args.activities)
    joint_df = load_trial_table('joint_samples', args.subjects, args.activities)
    if segment_df.empty and joint_df.empty:
        print(f"No per-trial tables found under {EXPERIMENT_DIR}. "
              f"Run `python -m experiments.sensor_distributions` first.")
        return

    trials = segment_df[['subject', 'activity']].drop_duplicates()
    n_subjects = trials['subject'].nunique()
    caption = (f"n={n_subjects} subject{'s' if n_subjects != 1 else ''}, "
               f"{' + '.join(sorted(trials['activity'].unique()))}"
               f"{'' if args.sides == 'both' else f', {args.sides} side'}")
    print(f"Loaded {len(trials)} trial(s), {len(segment_df):,} segment-samples, "
          f"{len(joint_df):,} joint-samples.")

    plot_distributions(segment_df, joint_df, segments, joint_order, args.plot_types, caption,
                       strip_clip=args.strip_clip, show=args.show)
    plot_sitting_timeseries(args.example_subject, args.example_activity, segments, joint_order,
                            show=args.show)
    if args.diagnostics:
        plot_diagnostics(args.subjects, args.activities, show=args.show)


if __name__ == '__main__':
    main()
