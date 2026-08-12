"""
Supplementary figure for experiments/threshold_sensitivity.py: how much the mag_adapt
observability threshold actually matters, and what a value of it means physically.

The claim the figure has to support is that the shipped threshold is not a tuned number
the result depends on. That takes three things, one per row of the argument:

  WHAT THE THRESHOLD MEANS (panel A).  o^J is in (m/s^2)(m/s^3) and nobody has an
  intuition for it, so the sweep's own duty-cycle table is drawn first: the fraction of
  samples each threshold would gate, per joint. It is the same axis as every other panel,
  so a reader can carry "1000 gates a third of the ankle's samples and 3% of the lumbar's"
  into the accuracy panels. Panels B and D also carry the POOLED duty cycle as a second
  x-axis on top, for the same reason.

  HOW FLAT THE CURVE IS (panels B, C).  RMSE against threshold, with mag_off and mag_on
  drawn as the two limits the sweep must converge to — mag_adapt gates on o^J > threshold,
  so a threshold below every sample IS mag_off and one above every sample IS mag_on. Both
  limits come from the same statistics file as the sweep (see the experiment's docstring),
  so their agreement with the curve's ends is a check on the whole pipeline and not a
  comparison across runs. Panel C splits by joint because the two ends of the body do not
  want the same threshold: the magnetometer helps where the field is clean (proximal) and
  hurts where it is not (distal), so a pooled curve alone would hide the trade the
  threshold is making.

  WHETHER THE DEFAULT IS LUCKY (panel D).  The per-cell optimum. For every subject x joint
  block the sweep has a best threshold and a best RMSE; what matters for the paper is the
  PENALTY for using one fixed default instead of that per-cell optimum. A flat curve with
  a scattered argmin and a small penalty is the honest version of "the threshold is not
  tuned"; a tight argmin cluster at the default would mean the opposite.

Everything is read back from results/statistics/observability_threshold_statistics.parquet
and the experiment's per-trial duty-cycle tables. Nothing here re-runs a filter, so no
number in the figure can disagree with the sweep that produced it.

Significance is tested across the swept thresholds as one Holm family, blocked by
subject x joint (see plotting/utils.py for why that is the replicate), and written to
<figure>_stats.csv beside the figure.
"""
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

import paths
from plotting import utils as plot_utils  # noqa: F401  (applies the shared paper rcParams on import)
from experiments.experiment_utils import DEFAULT_MAG_ADAPT_THRESHOLD, load_statistics
from experiments.threshold_sensitivity import (REFERENCE_METHODS, STATS_NAME,
                                               constants_disagreements, describe_disagreements,
                                               load_trial_table, sweep_arms,
                                               threshold_from_method)

PLOTS_DIR = paths.plots_dir("threshold_sensitivity")

AXIS = 'MAG'      # rotation-vector magnitude: the total joint-angle error
METRIC = 'rmse_deg'

# One colour per ARM, shared with the other supplements' convention that a colour means a
# method and never a joint or a subject. The sweep is the blue used for "what MAJIC does"
# in plotting/relative_vs_absolute.py; the two limits keep warm/cool apart so the curve can
# be seen approaching one and leaving the other.
ARM_COLORS = {
    'mag_adapt': '#1f6f8b',   # the swept arm
    'mag_on':    '#d1603d',   # threshold -> inf: never gate
    'mag_off':   '#8c6bb1',   # threshold -> 0: always gate
    'default':   '#111111',   # the shipped DEFAULT_MAG_ADAPT_THRESHOLD
}
REFERENCE_LABELS = {'mag_on': 'Mag on (never gate)', 'mag_off': 'Mag off (always gate)'}

# Left/right pooled into a joint type, matching plotting/paper_figures.py. Required by the
# significance test: the two sides of a joint correlate at ICC ~0.5, so blocking on them
# separately would overstate precision by ~1.5x (see plotting/utils.py).
RENAME_JOINTS = {'R_Hip': 'Hip', 'L_Hip': 'Hip', 'R_Knee': 'Knee', 'L_Knee': 'Knee',
                 'R_Ankle': 'Ankle', 'L_Ankle': 'Ankle', 'Lumbar': 'Lumbar'}
JOINT_ORDER = ['Lumbar', 'Hip', 'Knee', 'Ankle']  # proximal -> distal, which is the axis
                                                  # the threshold's trade-off runs along

# "Close enough to its own optimum" in panel D. Half a degree, because that is roughly the
# run-to-run spread this pipeline shows between adjacent thresholds and well inside what any
# downstream biomechanical measure resolves — a tolerance chosen from the measurement, not
# from what makes the count look best.
PENALTY_TOLERANCE_DEG = 0.5

# ==============================================================================
# Loading
# ==============================================================================

def load_sweep() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(swept, references) accuracy tables, in degrees, with joints collapsed to type.

    `swept` carries a 'threshold' column; `references` is the mag_on/mag_off pair, whose
    threshold is NaN by construction. Both are one row per subject x activity x joint.
    """
    df = load_statistics(STATS_NAME)
    if df is None:
        return pd.DataFrame(), pd.DataFrame()

    df = df[df['axis'] == AXIS].copy()
    df[METRIC] = np.degrees(df['rmse_rad'])
    df['joint_name'] = df['joint_name'].replace(RENAME_JOINTS)
    # Recomputed rather than trusted: an older statistics file may predate the column, and
    # deriving it from the method name means the two can never disagree.
    df['threshold'] = threshold_from_method(df['method'])

    return (df[df['threshold'].notna()].copy(),
            df[df['method'].isin(REFERENCE_METHODS)].copy())


def load_duty_cycle() -> pd.DataFrame:
    """Per-trial, per-joint fraction of samples gated, with joints collapsed to type."""
    gating = load_trial_table('gating')
    if gating.empty:
        return gating
    gating = gating.copy()
    gating['joint_name'] = gating['joint'].replace(RENAME_JOINTS)
    return gating


def load_observability() -> pd.DataFrame:
    """Per-trial, per-joint o^J percentile grid, with joints collapsed to type."""
    obs = load_trial_table('observability')
    if obs.empty:
        return obs
    obs = obs.copy()
    obs['joint_name'] = obs['joint'].replace(RENAME_JOINTS)
    return obs


def pooled_duty_cycle(gating: pd.DataFrame) -> pd.Series:
    """Fraction gated against threshold, averaged over every trial and joint.

    A plain mean over trial-joints, not a sample-weighted one: the second x-axis it labels
    sits above panels whose y-value is also a mean over trial-joints, and mixing the two
    weightings between an axis and the curve above it would be a quiet inconsistency."""
    if gating.empty:
        return pd.Series(dtype=float)
    return gating.groupby('threshold')['fraction_gated'].mean()


def describe_coverage(swept: pd.DataFrame, gating: pd.DataFrame) -> str:
    """One line naming what is behind the figure, for the footer. A supplementary figure
    that does not say how much data it rests on is not checkable."""
    if swept.empty:
        return "no sweep data on disk"
    trials = swept[['subject', 'trial_type']].drop_duplicates()
    thresholds = np.sort(swept['threshold'].unique())
    duty = pooled_duty_cycle(gating)
    span = (f", gating {duty.max():.0%} down to {duty.min():.1%} of samples"
            if not duty.empty else "")
    return (f"{len(thresholds)} thresholds spanning {thresholds[0]:.0f}-{thresholds[-1]:.0f} "
            f"(m/s²)(m/s³){span}; {len(trials)} trials, {swept['subject'].nunique()} subjects, "
            f"{swept['joint_name'].nunique()} joint types")

# ==============================================================================
# Shared axis furniture
# ==============================================================================

def _threshold_axis(ax: plt.Axes, thresholds: Sequence[float], mark_default: bool = True) -> None:
    """Log x-axis over the swept range, with the shipped default marked.

    Drawn on every panel so the four can be read as one axis: a threshold is at the same
    horizontal position in the duty-cycle panel as in the accuracy panels."""
    ax.set_xscale('log')
    if len(thresholds):
        ax.set_xlim(min(thresholds) * 0.8, max(thresholds) * 1.25)
    if mark_default:
        ax.axvline(DEFAULT_MAG_ADAPT_THRESHOLD, color=ARM_COLORS['default'], linewidth=1.2,
                   linestyle=(0, (5, 3)), zorder=1, alpha=0.7)
    ax.set_xlabel('Observability threshold $o^J$  (m/s²)(m/s³)')


def _duty_cycle_axis(ax: plt.Axes, duty: pd.Series) -> None:
    """A second x-axis on top labelled in fraction-of-samples-gated.

    Placed by interpolating the measured duty cycle, not by a formula: the mapping from
    threshold to duty cycle is a property of this dataset's o^J distribution, so it is read
    off panel A's own table. Ticks are chosen at round duty-cycle values and land wherever
    the data puts them."""
    if duty.empty:
        return
    thresholds = duty.index.to_numpy(dtype=float)
    fractions = duty.to_numpy(dtype=float)
    order = np.argsort(fractions)  # np.interp needs an increasing x

    targets = [f for f in (0.9, 0.6, 0.4, 0.2, 0.1, 0.03, 0.01)
               if fractions.min() <= f <= fractions.max()]
    positions = np.interp(targets, fractions[order], thresholds[order])

    top = ax.secondary_xaxis('top')
    top.set_xticks(positions)
    top.set_xticklabels([f'{f:.0%}' for f in targets], fontsize=11)
    top.set_xlabel('Samples with the magnetometer gated off', fontsize=12, labelpad=5)


def _reference_lines(ax: plt.Axes, interval: pd.DataFrame, label: bool = True) -> None:
    """mag_off and mag_on as horizontal bands, from the same within-block interval table as
    the curve beside them.

    Horizontal because neither depends on the threshold — that is precisely what makes them
    the limits. The band, rather than a bare line, keeps a reader from over-reading a curve
    end that sits a hundredth of a degree off its limit."""
    for method in REFERENCE_METHODS:
        if method not in interval.index:
            continue
        row = interval.loc[method]
        color = ARM_COLORS[method]
        ax.axhspan(row['lower'], row['upper'], color=color, alpha=0.14, linewidth=0, zorder=0)
        ax.axhline(row['mean'], color=color, linewidth=1.6, linestyle=':', zorder=2,
                   label=REFERENCE_LABELS[method] if label else None)

# ==============================================================================
# Blocking
# ==============================================================================
# Every accuracy number in this figure is a mean over subject x joint blocks, with side and
# activity averaged into the block first. Same replicate as the significance test — see
# plotting/utils.py's DEFAULT_BLOCK_COLS — so the drawn means and the p-values describe the
# same quantity rather than two different poolings of it.

BLOCK_COLS = ['subject', 'joint_name']

# The intervals are WITHIN-BLOCK (Cousineau-Morey), not the ordinary CI of each arm's mean.
# This is not cosmetic. Blocks differ enormously in absolute error — the lumbar sits near 10°
# and the hip near 12° while both knees and ankles sit near 6°, so the standard deviation of a
# block's mean across the grid is ~3.6°, against a within-block spread across thresholds of
# 0.6-3.5°. Plotted as ordinary CIs every arm's band would span the whole panel and overlap
# every other one, asserting "nothing is distinguishable" while the Friedman test on the same
# data returns p=1e-8. That is not conservatism, it is the wrong interval: every arm is
# measured on the SAME blocks, so between-block variance is common to all of them and cancels
# in the comparison the figure is about.
#
# So each block's own mean across arms is removed and the grand mean added back, which leaves
# the arm means exactly where they were and rescales only the spread. Morey's sqrt(k/(k-1))
# corrects the bias that centring introduces. The result is a COMPARISON interval: overlap
# between two arms means they are not distinguishable, which is what a reader tries to do with
# these bands anyway. Stated on the panel, because a within-block band silently swapped for a
# between-block one is a real way to overstate a result.


def within_block_interval(wide: pd.DataFrame, z: float = 1.96) -> pd.DataFrame:
    """Mean and Cousineau-Morey within-block CI for each column of a block x arm table.

    Blocks missing any arm are dropped, for the same reason Friedman needs complete blocks:
    a block present for some arms and not others would move the arm means against each other
    and the centring step would be comparing a block to a mean it did not contribute to."""
    wide = wide.dropna()
    n, k = wide.shape
    if n == 0 or k == 0:
        return pd.DataFrame(columns=['mean', 'lower', 'upper', 'n_blocks'])

    mean = wide.mean()
    if n < 2 or k < 2:
        return pd.DataFrame({'mean': mean, 'lower': mean, 'upper': mean, 'n_blocks': n})

    normalized = wide.sub(wide.mean(axis=1), axis=0) + wide.to_numpy().mean()
    half = z * normalized.std(ddof=1) / np.sqrt(n) * np.sqrt(k / (k - 1.0))
    return pd.DataFrame({'mean': mean, 'lower': mean - half, 'upper': mean + half,
                         'n_blocks': n})


def arm_table(swept: pd.DataFrame, references: pd.DataFrame,
              joint: Optional[str] = None) -> pd.DataFrame:
    """block x arm wide table of the metric, arms being the swept thresholds (keyed by their
    float value) plus whichever of mag_off/mag_on are on disk.

    One table for the whole panel rather than one per arm: the within-block interval is only
    defined over a common set of blocks, so the references have to be centred against the same
    blocks as the curve or their bands would not be comparable to it."""
    frames = []
    if not swept.empty:
        subset = swept if joint is None else swept[swept['joint_name'] == joint]
        frames.append(subset.assign(arm=subset['threshold']))
    if not references.empty:
        subset = references if joint is None else references[references['joint_name'] == joint]
        frames.append(subset.assign(arm=subset['method']))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.pivot_table(index=BLOCK_COLS, columns='arm', values=METRIC, aggfunc='mean')


def block_curve(swept: pd.DataFrame, references: pd.DataFrame = None,
                joint: Optional[str] = None) -> pd.DataFrame:
    """The swept arms of `arm_table`, as a threshold-ordered curve with within-block CIs."""
    references = pd.DataFrame() if references is None else references
    wide = arm_table(swept, references, joint)
    if wide.empty:
        return pd.DataFrame(columns=['threshold', 'mean', 'lower', 'upper', 'n_blocks'])

    interval = within_block_interval(wide)
    numeric = [arm for arm in interval.index if not isinstance(arm, str)]
    # Columns named even when empty: every caller either sorts on 'threshold' or reads
    # 'mean', and a bare DataFrame() would turn "no data for this joint" into a KeyError.
    return (interval.loc[numeric].rename_axis('threshold').reset_index()
                    .astype({'threshold': float}).sort_values('threshold'))


def block_table(swept: pd.DataFrame) -> pd.DataFrame:
    """The long table plot_utils.test_panels blocks on, with the threshold as a string
    group label (the group column has to be categorical for a Friedman/Wilcoxon panel)."""
    if swept.empty:
        return swept
    out = swept[['subject', 'joint_name', 'threshold', METRIC]].copy()
    out['threshold_label'] = out['threshold'].map(lambda t: f'{t:g}')
    return out


def threshold_order(swept: pd.DataFrame) -> List[str]:
    return [f'{t:g}' for t in np.sort(swept['threshold'].unique())]

# ==============================================================================
# Panel A: what a threshold means
# ==============================================================================

def panel_duty_cycle(ax: plt.Axes, gating: pd.DataFrame) -> None:
    """Fraction of samples gated against threshold, one curve per joint type.

    This is the panel that gives the x-axis a unit a reader can hold. It also carries the
    figure's other half-argument: the curves are decades apart, so ONE threshold cannot mean
    the same duty cycle everywhere on the body. That is not a defect of the threshold — o^J
    is a per-joint measurement and gating a joint that is genuinely observable more often is
    the intended behaviour — but a figure that showed only the pooled curve would let a
    reader assume a uniform 20%.
    """
    if gating.empty:
        return
    joints = [j for j in JOINT_ORDER if j in set(gating['joint_name'])]
    palette = dict(zip(joints, sns.color_palette('crest', n_colors=max(len(joints), 1))))

    for joint in joints:
        curve = gating[gating['joint_name'] == joint].groupby('threshold')['fraction_gated']
        median = curve.median()
        q25, q75 = curve.quantile(0.25), curve.quantile(0.75)
        ax.fill_between(median.index, q25, q75, color=palette[joint], alpha=0.15, linewidth=0)
        ax.plot(median.index, median, color=palette[joint], linewidth=2.4, marker='o',
                markersize=4.5, label=joint, zorder=3)

    pooled = pooled_duty_cycle(gating)
    ax.plot(pooled.index, pooled, color='black', linewidth=2.0, linestyle='--', zorder=4,
            label='All joints')

    default_duty = float(np.interp(DEFAULT_MAG_ADAPT_THRESHOLD, pooled.index, pooled.to_numpy()))
    ax.annotate(f'default gates {default_duty:.0%}\nof samples overall',
                xy=(DEFAULT_MAG_ADAPT_THRESHOLD, default_duty),
                xytext=(0.62, 0.82), textcoords='axes fraction', fontsize=11,
                fontweight='bold', color=ARM_COLORS['default'], ha='left',
                arrowprops={'arrowstyle': '->', 'color': ARM_COLORS['default'], 'linewidth': 1.2})

    _threshold_axis(ax, gating['threshold'].unique())
    ax.set_yscale('log')
    # A log axis cannot show "gated nothing", and at the top of the sweep some trial-joints
    # reach exactly zero. The floor is set from the smallest POSITIVE median so the curves are
    # not clipped short of it, and the count that falls off the bottom is stated rather than
    # left as a line that stops for no visible reason.
    medians = gating.groupby(['joint_name', 'threshold'])['fraction_gated'].median()
    positive = medians[medians > 0]
    floor = float(positive.min()) * 0.6 if len(positive) else 1e-3
    ax.set_ylim(floor, 1.4)
    ticks = [t for t in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0) if t >= floor]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t * 100:g}%' for t in ticks])
    ax.set_ylabel('Samples gated off')
    ax.legend(loc='lower left', fontsize=10, title='Joint', title_fontsize=10, ncol=2)
    off_scale = int((medians <= floor).sum())
    note = 'line: median trial, band: IQR'
    if off_scale:
        note += f'\n{off_scale} joint × threshold medians at or below {floor * 100:.2g}%, off scale'
    ax.text(0.99, 0.97, note, transform=ax.transAxes, ha='right', va='top', fontsize=9,
            fontstyle='italic', color='#666666')
    sns.despine(ax=ax)

# ==============================================================================
# Panel B: the pooled curve
# ==============================================================================

def panel_pooled(ax: plt.Axes, swept: pd.DataFrame, references: pd.DataFrame,
                 duty: pd.Series) -> None:
    """Pooled RMSE against threshold, with both limits.

    The y-axis is deliberately NOT zero-based. The whole content of the panel is the size of
    the variation across the threshold's range, and a zero-based axis would compress it into a
    flat line and make the figure prove its point by omission."""
    interval = within_block_interval(arm_table(swept, references))
    curve = block_curve(swept, references)
    if curve.empty:
        return

    _reference_lines(ax, interval)
    ax.fill_between(curve['threshold'], curve['lower'], curve['upper'],
                    color=ARM_COLORS['mag_adapt'], alpha=0.18, linewidth=0, zorder=2)
    ax.plot(curve['threshold'], curve['mean'], color=ARM_COLORS['mag_adapt'], linewidth=2.8,
            marker='o', markersize=6, zorder=3, label='mag_adapt (swept)')

    best = curve.loc[curve['mean'].idxmin()]
    at_default = curve.loc[(curve['threshold'] - DEFAULT_MAG_ADAPT_THRESHOLD).abs().idxmin()]
    ax.scatter([best['threshold']], [best['mean']], s=150, marker='*',
               color=ARM_COLORS['mag_adapt'], edgecolor='white', linewidth=1.0, zorder=5)
    ax.text(0.03, 0.96,
            f"span across the sweep: {curve['mean'].max() - curve['mean'].min():.2f}°\n"
            f"default costs {at_default['mean'] - best['mean']:.2f}° vs the pooled best "
            f"({best['threshold']:.0f})",
            transform=ax.transAxes, ha='left', va='top', fontsize=11, fontweight='bold',
            color=ARM_COLORS['mag_adapt'])

    _threshold_axis(ax, curve['threshold'])
    _duty_cycle_axis(ax, duty)
    ax.set_ylabel(f'RMSE, {AXIS} (deg)')
    ax.legend(loc='upper right', fontsize=10)
    ax.text(0.99, 0.02, f"mean ± within-block 95% CI, n={int(curve['n_blocks'].max())} "
                        f"subject × joint blocks; overlapping bands are not distinguishable",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontstyle='italic',
            color='#666666')
    sns.despine(ax=ax)

# ==============================================================================
# Panel C: by joint
# ==============================================================================

def panel_by_joint(ax: plt.Axes, swept: pd.DataFrame, references: pd.DataFrame) -> None:
    """One curve per joint type, each against its own mag_off and mag_on limits.

    This is where the threshold's trade-off is visible. The proximal joints sit in a clean
    field, so gating more (a lower threshold) costs them; the distal ones sit in the floor's
    distortion, so gating more helps them. Drawn as a difference from that joint's OWN
    mag_off, because the joints differ in absolute error by several degrees and a shared
    linear axis would render the lumbar curve as a flat line at the bottom.

    Sign convention: negative is better than mag_off. mag_on is drawn as a dotted line at
    that joint's own mag_on - mag_off gap, so the vertical distance between the curve and
    the dotted line of the same colour is how much the gating buys over never gating.
    """
    joints = [j for j in JOINT_ORDER if j in set(swept['joint_name'])]
    if not joints:
        return
    palette = dict(zip(joints, sns.color_palette('crest', n_colors=max(len(joints), 1))))

    for joint in joints:
        # Differenced WITHIN a block before averaging, not after: a block present in one arm
        # and missing from the other would otherwise shift the two means against each other.
        wide = arm_table(swept, references, joint)
        if wide.empty or 'mag_off' not in wide.columns:
            continue
        deltas = wide.drop(columns=['mag_off']).sub(wide['mag_off'], axis=0)
        interval = within_block_interval(deltas)
        if interval.empty:
            continue

        # Cast the index: arm_table's columns hold thresholds and method names together, so
        # the index comes back as object dtype and matplotlib cannot use it as an x-axis.
        curve = interval.loc[[a for a in interval.index if not isinstance(a, str)]]
        thresholds = curve.index.to_numpy(dtype=float)
        order = np.argsort(thresholds)
        thresholds, curve = thresholds[order], curve.iloc[order]
        ax.fill_between(thresholds, curve['lower'], curve['upper'], color=palette[joint],
                        alpha=0.13, linewidth=0)
        ax.plot(thresholds, curve['mean'], color=palette[joint], linewidth=2.4, marker='o',
                markersize=4.5, label=joint, zorder=3)

        if 'mag_on' in interval.index:
            ax.axhline(float(interval.loc['mag_on', 'mean']), color=palette[joint],
                       linewidth=1.3, linestyle=':', zorder=2)

    ax.axhline(0.0, color=ARM_COLORS['mag_off'], linewidth=1.8, zorder=1)
    ax.text(0.01, 0.0, ' mag_off', transform=ax.get_yaxis_transform(), color=ARM_COLORS['mag_off'],
            fontsize=11, fontweight='bold', va='bottom', ha='left')

    _threshold_axis(ax, swept['threshold'].unique())
    ax.set_ylabel('RMSE − mag_off RMSE (deg)')
    handles = [Line2D([], [], color=palette[j], linewidth=2.4, marker='o', markersize=4.5, label=j)
               for j in joints if j in palette]
    handles.append(Line2D([], [], color='#666666', linewidth=1.3, linestyle=':',
                          label='that joint\'s mag_on'))
    ax.legend(handles=handles, loc='best', fontsize=10, title='Joint', title_fontsize=10, ncol=2)
    ax.text(0.02, 0.02, 'mean ± within-block 95% CI, n=11 subjects per joint',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=9, fontstyle='italic',
            color='#666666')
    sns.despine(ax=ax)

# ==============================================================================
# Panel D: is the default lucky?
# ==============================================================================

def per_cell_optimum(swept: pd.DataFrame) -> pd.DataFrame:
    """One row per subject x joint block: its best threshold, its best RMSE, its RMSE at
    the shipped default, and the penalty for using the default instead.

    The penalty is the number the paper needs. It is non-negative by construction — the
    default is one of the swept points — so it is a bound on what per-cell tuning could buy,
    which is the honest way round: a per-cell optimum picked on the same data it is
    evaluated on is not an achievable method, only a ceiling on one."""
    if swept.empty:
        return pd.DataFrame()
    cells = swept.groupby([*BLOCK_COLS, 'threshold'])[METRIC].mean().reset_index()
    thresholds = np.sort(cells['threshold'].unique())
    default = float(thresholds[np.argmin(np.abs(thresholds - DEFAULT_MAG_ADAPT_THRESHOLD))])

    rows = []
    for block, group in cells.groupby(BLOCK_COLS):
        at_default = group.loc[group['threshold'] == default, METRIC]
        if at_default.empty:
            continue
        best = group.loc[group[METRIC].idxmin()]
        rows.append({'subject': block[0], 'joint_name': block[1],
                     'best_threshold': float(best['threshold']),
                     'best_rmse': float(best[METRIC]),
                     'default_rmse': float(at_default.iloc[0]),
                     'penalty': float(at_default.iloc[0] - best[METRIC])})
    return pd.DataFrame(rows)


def panel_optimum(ax: plt.Axes, optima: pd.DataFrame, swept: pd.DataFrame,
                  duty: pd.Series) -> None:
    """Where each block's optimum sits, and what the default costs it.

    The argmin is plotted against the penalty rather than as a histogram: a histogram would
    say where the optima are but not whether being far from the default matters, and those
    are the two halves of the same question. A block far to the left or right with a penalty
    near zero is a block whose curve is flat — which is the result, not noise around it."""
    if optima.empty:
        return
    joints = [j for j in JOINT_ORDER if j in set(optima['joint_name'])]
    palette = dict(zip(joints, sns.color_palette('crest', n_colors=max(len(joints), 1))))
    markers = dict(zip(joints, ('o', 's', '^', 'D', 'v')))

    # Jittered on the log axis, deterministically: several blocks share an argmin exactly
    # (the grid has nine points), and overplotted markers would hide the count.
    rng = np.random.default_rng(0)
    for joint in joints:
        subset = optima[optima['joint_name'] == joint]
        jitter = 10 ** (rng.uniform(-0.06, 0.06, len(subset)))
        ax.scatter(subset['best_threshold'] * jitter, subset['penalty'], s=62,
                   marker=markers[joint], color=palette[joint], edgecolor='white',
                   linewidth=0.8, alpha=0.9, label=joint, zorder=3)

    median_penalty = float(optima['penalty'].median())
    ax.axhline(median_penalty, color=ARM_COLORS['default'], linewidth=1.4, linestyle='-.',
               zorder=2)
    within = int((optima['penalty'] <= PENALTY_TOLERANCE_DEG).sum())
    at_default = int(np.isclose(optima['best_threshold'], DEFAULT_MAG_ADAPT_THRESHOLD).sum())
    ax.text(0.03, 0.97,
            f'median penalty for the default: {median_penalty:.2f}°\n'
            f'{within}/{len(optima)} blocks within {PENALTY_TOLERANCE_DEG:g}° of their own optimum\n'
            f'the default IS the optimum in {at_default}/{len(optima)} blocks',
            transform=ax.transAxes, ha='left', va='top', fontsize=11, fontweight='bold',
            color=ARM_COLORS['default'])

    _threshold_axis(ax, swept['threshold'].unique())
    _duty_cycle_axis(ax, duty)
    ax.set_xlabel('Best threshold for that block')
    ax.set_ylabel('Penalty for using the default (deg)')
    # Headroom for the annotation, and enough above the largest point that the worst block —
    # the one that most undercuts the "the default is fine" reading — is not clipped.
    ax.set_ylim(-0.05, max(float(optima['penalty'].max()) * 1.55, 1.0))
    ax.legend(loc='upper right', fontsize=10, title='Joint', title_fontsize=10, ncol=2)
    sns.despine(ax=ax)

# ==============================================================================
# Significance
# ==============================================================================

def test_sweep(swept: pd.DataFrame) -> Dict[str, plot_utils.PanelTest]:
    """Friedman across the swept thresholds + pairwise Wilcoxon, blocked by subject x joint.

    Two panels, tested as one Holm family: the pooled comparison, and the same restricted to
    the ankle, which panel C shows is where the threshold does most of its work. The pooled
    panel answering "flat" while the ankle panel answers "not flat" is a real result and the
    reason both are tested rather than only the first."""
    blocks = block_table(swept)
    if blocks.empty:
        return {}
    panels = {'All joints': blocks}
    for joint in ('Ankle', 'Lumbar'):
        subset = blocks[blocks['joint_name'] == joint]
        if not subset.empty:
            panels[joint] = subset
    return plot_utils.test_panels(panels, METRIC, 'threshold_label', threshold_order(swept))


def significance_epilog(results: Dict[str, plot_utils.PanelTest]) -> str:
    """One line summarizing what the test found, for the figure footer."""
    testable = [r for r in results.values() if r.skipped_reason is None]
    if not testable:
        return "No significance testing (insufficient blocks or thresholds)"
    verdicts = [f"{r.key}: {'differs' if r.friedman_p_adj < 0.05 else 'flat'} "
                f"(Friedman p={r.friedman_p_adj:.3g}, W={r.kendalls_w:.2f})" for r in testable]
    return ("Friedman across thresholds + pairwise Wilcoxon, Holm-corrected across all panels; "
            f"blocked by subject × joint, n={testable[0].n_blocks} — " + "; ".join(verdicts))

# ==============================================================================
# Composed figure
# ==============================================================================

# Panels B and D carry a duty-cycle axis on top, which needs room between the plot frame and
# the title. Everything above the frame — title and panel letter — is offset by the same
# amount so the letters stay on one line across the figure instead of stepping.
TITLE_PAD = {False: 10, True: 34}
LETTER_Y = {False: 1.06, True: 1.19}


def _label_panel(ax: plt.Axes, letter: str, has_duty_axis: bool = False) -> None:
    ax.text(-0.12, LETTER_Y[has_duty_axis], letter, transform=ax.transAxes, fontsize=20,
            fontweight='bold', va='bottom', ha='left')


def figure_supplement(swept: pd.DataFrame, references: pd.DataFrame, gating: pd.DataFrame,
                      optima: pd.DataFrame, results: Dict[str, plot_utils.PanelTest],
                      filename: str = 'threshold_sensitivity.png',
                      save: bool = True, show: bool = False) -> None:
    """The four-panel supplement.

    Constrained layout rather than plot_utils.finalize_and_save_plot's tight_layout: panels B
    and D carry a secondary x-axis on top, which tight_layout collides with the panel titles
    above them."""
    duty = pooled_duty_cycle(gating)
    fig = plt.figure(figsize=(17, 12.5), layout='constrained')
    axes = fig.subplots(2, 2)

    for ax, letter, title, has_duty_axis, draw in (
            (axes[0][0], 'A', 'What a threshold means', False,
             lambda ax: panel_duty_cycle(ax, gating)),
            (axes[0][1], 'B', 'Accuracy across the swept range', True,
             lambda ax: panel_pooled(ax, swept, references, duty)),
            (axes[1][0], 'C', 'The trade the threshold is making, by joint', False,
             lambda ax: panel_by_joint(ax, swept, references)),
            (axes[1][1], 'D', 'What the fixed default costs each block', True,
             lambda ax: panel_optimum(ax, optima, swept, duty))):
        draw(ax)
        ax.set_title(title, fontsize=15, pad=TITLE_PAD[has_duty_axis])
        _label_panel(ax, letter, has_duty_axis)

    fig.suptitle('Sensitivity of MAJIC to the magnetometer-gating threshold\n'
                 + describe_coverage(swept, gating), fontsize=21, fontweight='bold')
    # Constrained layout does not reserve space for a bare fig.text, so the footer gets its
    # own strip at the bottom of the layout rectangle rather than being drawn over panel D's
    # x-label.
    fig.get_layout_engine().set(rect=(0.0, 0.035, 1.0, 0.965))
    fig.text(0.99, 0.008, significance_epilog(results), ha='right', va='bottom', fontsize=9,
             fontstyle='italic')
    _save(fig, filename, save, show)


def _save(fig: plt.Figure, filename: str, save: bool, show: bool) -> None:
    if save:
        path = paths.ensure_parent(PLOTS_DIR / filename)
        fig.savefig(path, dpi=400, bbox_inches='tight')
        print(f"Saved plot to {path}")
    if show:
        plt.show()
    plt.close(fig)


def figures_standalone(swept: pd.DataFrame, references: pd.DataFrame, gating: pd.DataFrame,
                       optima: pd.DataFrame, save: bool = True, show: bool = False) -> None:
    """Each panel again at full size, one file each, for dropping into the manuscript."""
    duty = pooled_duty_cycle(gating)
    for draw, name, title, has_duty_axis in (
            (lambda ax: panel_duty_cycle(ax, gating), 'duty_cycle.png',
             'Fraction of samples gated, by threshold and joint', False),
            (lambda ax: panel_pooled(ax, swept, references, duty), 'pooled_rmse.png',
             'Joint-angle RMSE against the gating threshold', True),
            (lambda ax: panel_by_joint(ax, swept, references), 'rmse_by_joint.png',
             'Gating gain over mag_off, by joint', False),
            (lambda ax: panel_optimum(ax, optima, swept, duty), 'per_cell_optimum.png',
             'Per-block optimum and the cost of a fixed default', True)):
        fig, ax = plt.subplots(figsize=(9, 6.5), layout='constrained')
        draw(ax)
        ax.set_title(title, fontsize=15, pad=TITLE_PAD[has_duty_axis])
        _save(fig, name, save, show)

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--composed-only', action='store_true',
                        help='Write only the composed figure, not the per-panel files.')
    parser.add_argument('--show', action='store_true', help='Also display the figures.')
    parser.add_argument('--ignore-mixed-arms', action='store_true',
                        help="Plot even if the arms on disk came from different filter "
                             "configurations. Only for inspecting a known-mixed set.")
    args = parser.parse_args()

    swept, references = load_sweep()
    if swept.empty:
        print(f"Error: no mag_adapt_th* rows in {paths.statistics_path(STATS_NAME)}. "
              f"Run `python -m experiments.threshold_sensitivity` first.")
        return
    if references.empty:
        print("Warning: no mag_on/mag_off rows in the statistics file — the limit lines will "
              "be missing. Re-run the experiment without --skip-references.")

    # Checked before anything is drawn: a mixed set still loads, still plots, and its only
    # symptom is a number that should be impossible (see constants_disagreements).
    groups = constants_disagreements(
        sweep_arms(np.sort(swept['threshold'].unique()), with_references=not references.empty))
    if len(groups) > 1:
        print(describe_disagreements(groups))
        if not args.ignore_mixed_arms:
            print("\nRefusing to plot. Pass --ignore-mixed-arms to override.")
            return
        print("\n--ignore-mixed-arms given; plotting anyway. The figure is NOT publishable.")

    gating = load_duty_cycle()
    if gating.empty:
        print("Warning: no duty-cycle tables on disk — panel A will be empty. Run "
              "`python -m experiments.threshold_sensitivity --gating-only`.")

    print(describe_coverage(swept, gating))
    optima = per_cell_optimum(swept)
    results = test_sweep(swept)

    figure_supplement(swept, references, gating, optima, results, show=args.show)
    if results:
        plot_utils._emit_significance_report(plot_utils.significance_report(results, METRIC),
                                             'threshold_sensitivity.png', PLOTS_DIR)
    if not args.composed_only:
        figures_standalone(swept, references, gating, optima, show=args.show)
    print(f"\nFigures under {PLOTS_DIR}")


if __name__ == '__main__':
    main()
