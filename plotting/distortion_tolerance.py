"""
Supplementary figure for experiments/distortion_tolerance.py: how much magnetic distortion
the magnetometer-on filter tolerates before it stops being worth using.

The paper's main result says mag_on helps the proximal joints and hurts the distal ones in
this lab. That is only transferable if it comes with a dose, so the figure's job is to turn
"this lab's floor" into a number a reader can compare their own environment against. Four
panels, one per step of that argument:

  WHAT A SCALE IS (panel A).  The x-axis of every other panel is a multiplier on the
  estimated distortion, which means nothing on its own. Panel A converts it to degrees: the
  angle between the parent's and the child's world-frame magnetic field, which is the
  disagreement a RELATIVE magnetometer correction actually suffers from (a distortion common
  to both sensors cancels in the relative update). The measured field is a = 1, marked, so
  the reader can see where this dataset sits on its own axis.

  THE POOLED CURVE (panel B).  RMSE against scale, with mag_off and mag_on as reference
  bands. mag_on is a redundancy check, not a limit — the a = 1 arm reproduces it exactly by
  construction — and the two coinciding is evidence the dial is wired up correctly.

  WHERE THE MAGNETOMETER STOPS PAYING (panel C).  The same curve per joint, differenced
  against that joint's OWN mag_off, so zero is "no better than not using the magnetometer at
  all" and the crossing point is the tolerance. This is the panel the figure exists for: the
  joints do not share a tolerance, and the ankle's is the one that binds.

  THE TOLERANCE IN DEGREES (panel D).  Each block's crossing, converted through panel A's
  dose curve into degrees of field disagreement, against the level this lab actually
  presents. A joint whose crossing sits above the measured level is one the magnetometer
  still helps here; below it, one where it already hurts.

Everything is read back from results/statistics/distortion_tolerance_statistics.parquet and
the experiment's per-trial dose tables. Nothing here re-runs a filter.
"""
import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

import paths
from plotting import utils as plot_utils  # noqa: F401  (applies the shared paper rcParams on import)
from experiments.experiment_utils import load_statistics
from experiments.distortion_tolerance import (REFERENCE_METHODS, STATS_NAME, UNITY_SCALE,
                                              load_trial_table, scale_from_method)

PLOTS_DIR = paths.plots_dir("distortion_tolerance")

AXIS = 'MAG'      # rotation-vector magnitude: the total joint-angle error
METRIC = 'rmse_deg'

# The per-sample summary of the dose used as the figure's physical axis. The median rather
# than the mean because the anomaly is a function of where the segment is and the
# distribution is one-sided (see experiments/distortion_tolerance.QUANTILES); the p95 is
# drawn alongside it in panel A so the tail is visible rather than averaged away.
DOSE_STAT = 'disagreement_deg_median'
DOSE_TAIL_STAT = 'disagreement_deg_p95'

# One colour per ARM, following the other supplements' convention that a colour means a
# method and never a joint or a subject.
ARM_COLORS = {
    'sweep':    '#1f6f8b',   # the swept distortion arms
    'mag_on':   '#d1603d',   # the real field: the a = 1 arm, and its redundant reference
    'mag_off':  '#8c6bb1',   # no magnetometer: the floor the sweep is measured against
    'measured': '#111111',   # the a = 1 marker, i.e. this lab
}
REFERENCE_LABELS = {'mag_on': 'Mag on (real field)', 'mag_off': 'Mag off (no magnetometer)'}

# Left/right pooled into a joint type, matching plotting/paper_figures.py.
RENAME_JOINTS = {'R_Hip': 'Hip', 'L_Hip': 'Hip', 'R_Knee': 'Knee', 'L_Knee': 'Knee',
                 'R_Ankle': 'Ankle', 'L_Ankle': 'Ankle', 'Lumbar': 'Lumbar'}
JOINT_TYPE_ORDER = ['Lumbar', 'Hip', 'Knee', 'Ankle']  # proximal -> distal, the axis the
                                                       # distortion gradient runs along
SIDE_ORDER = ['R', 'L', '']  # sideless joints (Lumbar) sort last within their type, of one

# TWO BLOCK DEFINITIONS, AND THE DIFFERENCE IS LOAD-BEARING (see --split-sides).
#
# Everything inferential — the pooled curve's within-block CIs and every p-value — blocks on
# the joint TYPE, with side and (where pooled) activity averaged into the block first. That is
# forced, not chosen: the two sides of a joint correlate at ICC ~0.5, so treating them as two
# replicates would overstate precision by ~1.5x and narrow every band and p-value by a factor
# nothing in the data supports (see plot_utils._warn_if_sides_split).
#
# The descriptive panels — the dose curve, the per-joint curves, and the per-block tolerance
# markers — may split sides, because a mean is a mean however it is grouped and the question
# "do the two legs behave differently" cannot be asked of a pooled number. With --split-sides
# off the two definitions are identical and nothing changes.
STAT_BLOCK_COLS = ['subject', 'joint_type', 'trial_type']
DISPLAY_BLOCK_COLS = ['subject', 'joint_name', 'trial_type']

# ACTIVITY IS PART OF THE BLOCK HERE, which is a deliberate departure from
# plot_utils.DEFAULT_BLOCK_COLS (subject x joint, with activity averaged in) that every other
# figure in this repo follows. The reason is measured, not stylistic: the ankle's response to
# distortion runs the OPPOSITE WAY in the two activities — on Subject 01 more distortion costs
# the ankle 0.6 deg over the swept range in walking and BUYS it 0.2 deg in complexTasks — so
# averaging the two into one block cancels the effect this figure exists to show and reports a
# flat curve as the result. That sign flip is consistent with the rest of the paper's finding
# that the mag_on/mag_off winner at a distal joint is decided by how much quiet time a trial
# has and by the local field, not by the joint alone.
#
# Pooling activities is still the right default where a figure's effect has the same sign in
# both; it is wrong here. Unlike the side split above, activity is in BOTH block definitions —
# the two activities are different conditions rather than two correlated replicates of one, so
# separating them is not a precision claim.

# Linestyle per activity, since colour is already spoken for by the joint.
ACTIVITY_STYLES = {'walking': '-', 'complexTasks': (0, (5, 2))}
ACTIVITY_LABELS = {'walking': 'walking', 'complexTasks': 'complex tasks'}

# The joint TYPES the experiment was set up to answer for. Called out in the panel titles and
# tested as their own significance panels; the others are drawn but not foregrounded. Matched on
# the type, so both sides of a focus joint stay in focus when sides are split.
FOCUS_JOINTS = ['Knee', 'Ankle']

# How much lighter the left side is drawn than the right, when sides are split. Shading within
# one hue rather than two hues per joint: a reader should see 'the two ankles' as a pair first
# and the sides second, which two unrelated colours would invert.
LEFT_LIGHTEN = 0.45

# ==============================================================================
# Joint names, with or without sides
# ==============================================================================
# A joint_name is either a pooled type ('Ankle') or a side-split name ('R_Ankle'). Every display
# helper takes whichever is in the frame, so the two modes share one code path and the pooled
# default cannot drift from the split one.

def joint_type(name: str) -> str:
    """'R_Ankle' -> 'Ankle'; 'Ankle' -> 'Ankle'."""
    return RENAME_JOINTS.get(name, name)


def joint_side(name: str) -> str:
    """'R_Ankle' -> 'R'; 'Ankle' -> '' (pooled, or a joint with no side)."""
    return name.split('_')[0] if name[:2] in ('R_', 'L_') else ''


def joint_label(name: str) -> str:
    """The name as it appears on the figure. 'R_Ankle' -> 'Right ankle'; 'Ankle' -> 'Ankle'.

    Spelled out rather than left as 'R_Ankle' because the labels also appear inside
    parenthesised phrases in the figure's footer, where a second bracket would read badly."""
    side = joint_side(name)
    if not side:
        return name
    return f"{'Right' if side == 'R' else 'Left'} {joint_type(name).lower()}"


def _sort_key(name: str) -> tuple:
    """Proximal -> distal, right before left within a type."""
    return (JOINT_TYPE_ORDER.index(joint_type(name)) if joint_type(name) in JOINT_TYPE_ORDER
            else len(JOINT_TYPE_ORDER), SIDE_ORDER.index(joint_side(name)))

# ==============================================================================
# Loading
# ==============================================================================

def load_sweep(split_sides: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(swept, references) accuracy tables, in degrees.

    Two joint columns come back, and which is which matters: 'joint_type' is always the pooled
    name (Ankle) and is what everything inferential blocks on; 'joint_name' is the pooled name
    too unless `split_sides`, in which case it keeps the side (R_Ankle) and the descriptive
    panels split on it. See STAT_BLOCK_COLS.

    `swept` carries a 'distortion_scale' column; `references` is the mag_off/mag_on pair,
    whose scale is NaN by construction."""
    df = load_statistics(STATS_NAME)
    if df is None:
        return pd.DataFrame(), pd.DataFrame()

    df = df[df['axis'] == AXIS].copy()
    df[METRIC] = np.degrees(df['rmse_rad'])
    df['joint_type'] = df['joint_name'].replace(RENAME_JOINTS)
    if not split_sides:
        df['joint_name'] = df['joint_type']
    # Recomputed rather than trusted: an older statistics file may predate the column, and
    # deriving it from the method name means the two can never disagree.
    df['distortion_scale'] = scale_from_method(df['method'])

    return (df[df['distortion_scale'].notna()].copy(),
            df[df['method'].isin(REFERENCE_METHODS)].copy())


def load_dose(split_sides: bool = False) -> pd.DataFrame:
    """Per-trial, per-joint inter-sensor field disagreement at each scale, keyed the same way as
    load_sweep so the two join on 'joint_name'."""
    dose = load_trial_table('joint_distortion')
    if dose.empty:
        return dose
    dose = dose.copy()
    dose['joint_type'] = dose['joint'].replace(RENAME_JOINTS)
    dose['joint_name'] = dose['joint'] if split_sides else dose['joint_type']
    return dose


def dose_curve(dose: pd.DataFrame, joint: Optional[str] = None,
               activity: Optional[str] = None, stat: str = DOSE_STAT) -> pd.Series:
    """scale -> degrees of field disagreement, averaged over trials (and over joints and
    activities, if none is named). This is the mapping every "in degrees" statement in the
    figure goes through.

    NOT GUARANTEED MONOTONE. The angle between the distorted field and the assumed one
    saturates as the anomaly comes to dominate, so at the ankle the top of the sweep can bend
    back a degree or two. It is monotone in the region the tolerances fall in; callers that
    invert it (_dose_axis) sort first and are only labelling an axis."""
    if dose.empty:
        return pd.Series(dtype=float)
    subset = dose if joint is None else dose[dose['joint_name'] == joint]
    if activity is not None and 'activity' in subset.columns:
        subset = subset[subset['activity'] == activity]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.groupby('distortion_scale')[stat].mean()


def describe_coverage(swept: pd.DataFrame, dose: pd.DataFrame) -> str:
    """One line naming what is behind the figure, for the footer. A supplementary figure
    that does not say how much data it rests on is not checkable."""
    if swept.empty:
        return "no sweep data on disk"
    trials = swept[['subject', 'trial_type']].drop_duplicates()
    scales = np.sort(swept['distortion_scale'].unique())
    pooled = dose_curve(dose)
    span = (f" = {pooled.min():.0f}–{pooled.max():.0f}° of field disagreement"
            if not pooled.empty else "")
    split = swept['joint_name'].nunique() > swept['joint_type'].nunique()
    joints = (f"{swept['joint_name'].nunique()} joints, sides apart" if split
              else f"{swept['joint_name'].nunique()} joint types, sides pooled")
    n_subjects = swept['subject'].nunique()
    return (f"{len(scales)} levels, {scales[0]:.0%}–{scales[-1]:.0%} of the measured anomaly"
            f"{span} · {len(trials)} trials, {n_subjects} subject"
            f"{'' if n_subjects == 1 else 's'}, {joints}")

# ==============================================================================
# Blocking
# ==============================================================================
# Same replicate as the significance test (plot_utils.DEFAULT_BLOCK_COLS): one block per
# subject x joint type, with side and activity averaged in first, so the drawn means and the
# p-values describe the same quantity. The intervals are WITHIN-BLOCK (Cousineau-Morey) for
# the reason spelled out in plotting/threshold_sensitivity.py: blocks differ by several
# degrees in absolute error while the effect across arms is under one, so ordinary CIs would
# span the panel and assert "nothing is distinguishable" about a comparison every arm shares
# its blocks with. Overlap between two bands therefore means not distinguishable.

def within_block_interval(wide: pd.DataFrame, z: float = 1.96) -> pd.DataFrame:
    """Mean and Cousineau-Morey within-block CI for each column of a block x arm table.

    Blocks missing any arm are dropped, for the same reason Friedman needs complete blocks:
    a block present for some arms and not others would move the arm means against each other
    and the centring step would compare a block to a mean it did not contribute to."""
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


def _select(df: pd.DataFrame, joint: Optional[str], activity: Optional[str]) -> pd.DataFrame:
    if joint is not None:
        df = df[df['joint_name'] == joint]
    if activity is not None:
        df = df[df['trial_type'] == activity]
    return df


def arm_table(swept: pd.DataFrame, references: pd.DataFrame, joint: Optional[str] = None,
              activity: Optional[str] = None) -> pd.DataFrame:
    """block x arm wide table of the metric, arms being the swept scales (keyed by their
    float value) plus whichever of mag_off/mag_on are on disk.

    One table for the whole panel rather than one per arm: the within-block interval is only
    defined over a common set of blocks, so the references have to be centred against the
    same blocks as the curve or their bands would not be comparable to it."""
    frames = []
    if not swept.empty:
        subset = _select(swept, joint, activity)
        frames.append(subset.assign(arm=subset['distortion_scale']))
    if not references.empty:
        subset = _select(references, joint, activity)
        frames.append(subset.assign(arm=subset['method']))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    # Indexed on the STAT blocks even when a side-split `joint` was passed: the filter has
    # already restricted the frame to one side, so the pooled index still identifies each row
    # uniquely, and the interval this feeds keeps counting joint types rather than sides.
    return combined.pivot_table(index=STAT_BLOCK_COLS, columns='arm', values=METRIC,
                                aggfunc='mean')


def _numeric_arms(interval: pd.DataFrame) -> pd.DataFrame:
    """The swept arms of an interval table as a scale-ordered curve.

    arm_table's columns hold scales and method names together, so the index comes back as
    object dtype and matplotlib cannot use it as an x-axis until it is split and cast."""
    numeric = [arm for arm in interval.index if not isinstance(arm, str)]
    if not numeric:
        return pd.DataFrame(columns=['distortion_scale', 'mean', 'lower', 'upper', 'n_blocks'])
    return (interval.loc[numeric].rename_axis('distortion_scale').reset_index()
                    .astype({'distortion_scale': float}).sort_values('distortion_scale'))


def block_curve(swept: pd.DataFrame, references: pd.DataFrame = None,
                joint: Optional[str] = None, activity: Optional[str] = None) -> pd.DataFrame:
    """The swept arms as a scale-ordered curve with within-block CIs."""
    references = pd.DataFrame() if references is None else references
    wide = arm_table(swept, references, joint, activity)
    if wide.empty:
        return pd.DataFrame(columns=['distortion_scale', 'mean', 'lower', 'upper', 'n_blocks'])
    return _numeric_arms(within_block_interval(wide))

# ==============================================================================
# The tolerance itself
# ==============================================================================

def breakeven(swept: pd.DataFrame, references: pd.DataFrame, dose: pd.DataFrame) -> pd.DataFrame:
    """One row per block (subject x joint x activity, see DISPLAY_BLOCK_COLS — so per side when
    sides are split): the distortion scale at which using the magnetometer stops beating not
    using it, and what that scale is in degrees.

    Found by linear interpolation on the block's own (scale, RMSE) curve against the block's
    own mag_off — not against a pooled mag_off. Blocks differ by several degrees in absolute
    error, so a shared floor would put the crossing wherever the block sits rather than where
    its curve turns.

    Three outcomes, kept apart rather than collapsed to a number:
      'crosses'      the curve rises through mag_off inside the swept range; `scale` is where.
      'never_helps'  the curve is already above mag_off at zero distortion, so the
                     magnetometer is not paying for itself even with a perfect field — that
                     is a statement about the filter's tuning, not about the field, and
                     reporting it as a tolerance of 0 would misattribute it.
      'always_helps' the curve is below mag_off at every scale tested, including the
                     amplified ones. The tolerance is beyond the sweep; `scale` is NaN and
                     the sweep's top end is a lower bound on it.
    """
    if swept.empty or references.empty:
        return pd.DataFrame()

    curves = swept.groupby([*DISPLAY_BLOCK_COLS, 'distortion_scale'])[METRIC].mean()
    floors = (references[references['method'] == 'mag_off']
              .groupby(DISPLAY_BLOCK_COLS)[METRIC].mean())

    rows = []
    for block, curve in curves.groupby(level=DISPLAY_BLOCK_COLS):
        if block not in floors.index:
            continue
        block_id = dict(zip(DISPLAY_BLOCK_COLS, block))
        scales = curve.index.get_level_values('distortion_scale').to_numpy(dtype=float)
        order = np.argsort(scales)
        scales = scales[order]
        delta = curve.to_numpy()[order] - float(floors.loc[block])

        if delta[0] > 0:
            outcome, crossing = 'never_helps', np.nan
        elif np.all(delta < 0):
            outcome, crossing = 'always_helps', np.nan
        else:
            # First upward crossing: the tolerance is where the magnetometer stops helping,
            # so a later re-crossing (the curves are not guaranteed monotone) must not be
            # reported as a higher tolerance than the first failure.
            i = int(np.argmax(delta >= 0))
            x0, x1, d0, d1 = scales[i - 1], scales[i], delta[i - 1], delta[i]
            outcome = 'crosses'
            crossing = float(x0 + (x1 - x0) * (-d0) / (d1 - d0)) if d1 != d0 else float(x1)

        # The dose is looked up on the SAME block, not pooled: the anomaly a segment sees
        # depends on where that trial took it, so a crossing converted through a pooled dose
        # curve would be quoted in degrees the block never experienced.
        joint_dose = dose_curve(_block_dose(dose, block_id), block_id['joint_name'])
        rows.append({
            **block_id, 'joint_type': joint_type(block_id['joint_name']), 'outcome': outcome,
            'breakeven_scale': crossing,
            'breakeven_deg': _scale_to_degrees(crossing, joint_dose),
            # What this lab presents at that joint, i.e. the level the tolerance has to be
            # compared against for the result to mean anything.
            'measured_deg': _scale_to_degrees(UNITY_SCALE, joint_dose),
            'max_swept_deg': _scale_to_degrees(scales[-1], joint_dose),
        })
    return pd.DataFrame(rows)


def _block_dose(dose: pd.DataFrame, block_id: Dict[str, str]) -> pd.DataFrame:
    """The dose table restricted to one block. The dose table names its activity column
    'activity' while the statistics name theirs 'trial_type', so the mapping is spelled out
    here rather than assumed — a silent mismatch would leave the filter as a no-op and quietly
    fall back to the pooled dose."""
    filters = {'subject': 'subject', 'joint_name': 'joint_name', 'trial_type': 'activity'}
    for block_col, dose_col in filters.items():
        if block_col in block_id and dose_col in dose.columns:
            dose = dose[dose[dose_col] == block_id[block_col]]
    return dose


def _scale_to_degrees(scale: float, curve: pd.Series) -> float:
    """A distortion scale in degrees of inter-sensor field disagreement, via the measured
    dose curve rather than a formula: the mapping is a property of this dataset's field, and
    it is nonlinear (the angle saturates as the anomaly comes to dominate the field), so it
    has to be interpolated from panel A's own table."""
    if not np.isfinite(scale) or curve.empty:
        return float('nan')
    return float(np.interp(scale, curve.index.to_numpy(dtype=float), curve.to_numpy()))

# ==============================================================================
# Shared axis furniture
# ==============================================================================

def _scale_axis(ax: plt.Axes, scales, mark_measured: bool = True) -> None:
    """Linear x-axis in fraction-of-the-measured-distortion, labelled as a percentage.

    Linear, not log, because 0 is a swept point and it is the most informative one — it is
    the mag oracle. Percentage labels because "50% of the distortion we measured" is the
    quantity that was asked for; the degrees equivalent goes on the top axis."""
    scales = np.asarray(sorted(set(np.asarray(scales, dtype=float))))
    if scales.size:
        pad = 0.03 * (scales[-1] - scales[0] or 1.0)
        ax.set_xlim(scales[0] - pad, scales[-1] + pad)
        ax.set_xticks(scales)
        ax.set_xticklabels([f'{s:.0%}' for s in scales])
    if mark_measured:
        ax.axvline(UNITY_SCALE, color=ARM_COLORS['measured'], linewidth=1.2,
                   linestyle=(0, (5, 3)), zorder=1, alpha=0.7)
    ax.set_xlabel('Magnetic distortion applied  (% of the measured anomaly)')


def _dose_axis(ax: plt.Axes, dose_line: pd.Series) -> None:
    """A second x-axis on top, labelled in degrees of inter-sensor field disagreement.

    Placed by interpolating the measured dose curve, not by a formula — the mapping from
    scale to degrees is a property of this dataset's field, so it is read off panel A's own
    table."""
    if dose_line.empty:
        return
    scales = dose_line.index.to_numpy(dtype=float)
    degrees = dose_line.to_numpy(dtype=float)
    order = np.argsort(degrees)  # np.interp needs an increasing x

    targets = [d for d in (2, 5, 10, 15, 20, 30, 40)
               if degrees.min() <= d <= degrees.max()]
    if not targets:
        return
    top = ax.secondary_xaxis('top')
    top.set_xticks(np.interp(targets, degrees[order], scales[order]))
    top.set_xticklabels([f'{d:g}°' for d in targets], fontsize=11)
    top.set_xlabel('Median field disagreement between the joint\'s two sensors',
                   fontsize=12, labelpad=5)


def _reference_lines(ax: plt.Axes, interval: pd.DataFrame, label: bool = True) -> None:
    """mag_off and mag_on as horizontal bands, from the same within-block interval table as
    the curve beside them. Horizontal because neither depends on the distortion scale."""
    for method in REFERENCE_METHODS:
        if method not in interval.index:
            continue
        row = interval.loc[method]
        color = ARM_COLORS[method]
        ax.axhspan(row['lower'], row['upper'], color=color, alpha=0.14, linewidth=0, zorder=0)
        ax.axhline(row['mean'], color=color, linewidth=1.6, linestyle=':', zorder=2,
                   label=REFERENCE_LABELS[method] if label else None)


def _joint_palette(joints: List[str]) -> Dict[str, tuple]:
    """One colour per joint, hue by TYPE and lightness by side.

    Keyed on the type rather than on the position in the list so that a joint keeps its colour
    whether or not sides are split, and so the two sides of a joint read as a pair. Two
    unrelated hues per joint would make 'left ankle vs right knee' the visual comparison
    instead of 'left ankle vs right ankle'."""
    types = [t for t in JOINT_TYPE_ORDER if any(joint_type(j) == t for j in joints)]
    base = dict(zip(types, sns.color_palette('crest', n_colors=max(len(types), 1))))
    return {j: (tuple(np.array(base[joint_type(j)])
                      + LEFT_LIGHTEN * (1.0 - np.array(base[joint_type(j)])))
                if joint_side(j) == 'L' else base[joint_type(j)])
            for j in joints}


def _present_joints(df: pd.DataFrame, column: str = 'joint_name') -> List[str]:
    return sorted(set(df[column]), key=_sort_key)


def _is_focus(joint: str) -> bool:
    return joint_type(joint) in FOCUS_JOINTS

# ==============================================================================
# Panel A: what a scale is, in degrees
# ==============================================================================

def panel_dose(ax: plt.Axes, dose: pd.DataFrame) -> None:
    """Inter-sensor field disagreement against distortion scale, one curve per joint type.

    This panel gives the x-axis a unit and, in passing, states the premise of the whole
    experiment: at the measured field the curves are already decades apart, so the ankle is
    not running the same experiment as the lumbar. The p95 dashed lines matter because the
    distortion is not stationary — a segment's dose depends on where it is in the stride, and
    the ankle's worst moments are several times its median.
    """
    if dose.empty:
        return
    joints = _present_joints(dose)
    palette = _joint_palette(joints)
    activities = ([a for a in ACTIVITY_STYLES if a in set(dose['activity'])]
                  if 'activity' in dose.columns else [None])

    for joint in joints:
        for activity in activities:
            median = dose_curve(dose, joint, activity)
            ax.plot(median.index, median, color=palette[joint], linewidth=2.4, marker='o',
                    markersize=4.5, linestyle=ACTIVITY_STYLES.get(activity, '-'), zorder=3)

    measured = {joint: _scale_to_degrees(UNITY_SCALE, dose_curve(dose, joint)) for joint in joints}
    ax.annotate('this lab, as measured',
                xy=(UNITY_SCALE, max(measured.values(), default=1.0)),
                xytext=(0.56, 0.30), textcoords='axes fraction', fontsize=11,
                fontweight='bold', color=ARM_COLORS['measured'], ha='left',
                arrowprops={'arrowstyle': '->', 'color': ARM_COLORS['measured'], 'linewidth': 1.2})

    _scale_axis(ax, dose['distortion_scale'].unique())
    ax.set_ylabel('Field disagreement between the joint\'s two sensors (deg)')
    handles = [Line2D([], [], color=palette[j], linewidth=2.4, marker='o', markersize=4.5,
                      label=f"{joint_label(j)}  ({measured[j]:.0f}°)") for j in joints]
    handles += [Line2D([], [], color='#666666', linewidth=1.8, linestyle=ACTIVITY_STYLES[a],
                       label=ACTIVITY_LABELS[a]) for a in activities if a is not None]
    ax.legend(handles=handles, loc='upper left', fontsize=9.5, ncol=2,
              title='Median over samples (° in brackets = at 100%, both activities)',
              title_fontsize=9.5)
    ax.set_ylim(bottom=0)
    # The curves bend back at the top for the ankle: the angle to the assumed field saturates
    # once the anomaly dominates it, so doubling the anomaly does not double the angle. Said on
    # the panel because a reader reading the x-axis as linear in degrees would otherwise
    # over-read the amplified arms.
    ax.text(0.99, 0.02, 'the angle saturates once the anomaly dominates the field, so the top\n'
                        'of the sweep is compressed and can bend back',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontstyle='italic',
            color='#666666')
    sns.despine(ax=ax)

# ==============================================================================
# Panel B: the pooled curve
# ==============================================================================

def panel_pooled(ax: plt.Axes, swept: pd.DataFrame, references: pd.DataFrame,
                 dose_line: pd.Series) -> None:
    """Pooled RMSE against distortion scale, with both references.

    The y-axis is deliberately NOT zero-based: the content of the panel is the size of the
    variation across the sweep, and a zero-based axis would compress it into a flat line and
    make the figure prove its point by omission."""
    interval = within_block_interval(arm_table(swept, references))
    curve = _numeric_arms(interval)
    if curve.empty:
        return

    _reference_lines(ax, interval)
    ax.fill_between(curve['distortion_scale'], curve['lower'], curve['upper'],
                    color=ARM_COLORS['sweep'], alpha=0.18, linewidth=0, zorder=2)
    ax.plot(curve['distortion_scale'], curve['mean'], color=ARM_COLORS['sweep'], linewidth=2.8,
            marker='o', markersize=6, zorder=3, label='mag_on at scaled distortion')

    at_zero = float(curve['mean'].iloc[0])
    at_unity = float(np.interp(UNITY_SCALE, curve['distortion_scale'], curve['mean']))
    # Boxed, because the reference bands run behind it and the numbers are the panel's point.
    ax.text(0.03, 0.955,
            f"a perfectly clean field would buy {at_unity - at_zero:.2f}°\n"
            f"across the whole sweep the error spans "
            f"{curve['mean'].max() - curve['mean'].min():.2f}°",
            transform=ax.transAxes, ha='left', va='top', fontsize=11, fontweight='bold',
            color=ARM_COLORS['sweep'],
            bbox={'facecolor': 'white', 'alpha': 0.82, 'edgecolor': 'none', 'pad': 3.0})

    _scale_axis(ax, curve['distortion_scale'])
    _dose_axis(ax, dose_line)
    ax.set_ylabel(f'RMSE, {AXIS} (deg)')
    # Headroom above the mag_off band so the annotation is not drawn over the curve or the
    # reference lines it is describing.
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + 0.42 * (top - bottom))
    ax.legend(loc='center right', fontsize=10)
    ax.text(0.99, 0.02, f"mean ± within-block 95% CI, n={int(curve['n_blocks'].max())} "
                        f"blocks; overlapping bands are not distinguishable. The top axis is "
                        f"pooled over joints —\nthe joints' own doses differ by 5x (panel A), "
                        f"so it labels the pooled curve only.",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontstyle='italic',
            color='#666666')
    sns.despine(ax=ax)

# ==============================================================================
# Panel C: where the magnetometer stops paying, per joint
# ==============================================================================

def panel_by_joint(ax: plt.Axes, swept: pd.DataFrame, references: pd.DataFrame,
                   marks: pd.DataFrame) -> None:
    """Each joint's curve as a difference from its OWN mag_off, so zero is the decision
    boundary: below it the magnetometer is earning its place, above it the joint would be
    better off without it, and the crossing is that joint's distortion tolerance.

    Differenced within a block before averaging, not after — a block present in one arm and
    missing from the other would otherwise shift the two means against each other. Drawn as a
    difference rather than in absolute RMSE for the same reason the threshold supplement is:
    the joints differ by several degrees in absolute error, so a shared linear axis would
    render the proximal curves as a flat line at the bottom.
    """
    joints = _present_joints(swept)
    if not joints:
        return
    palette = _joint_palette(joints)
    activities = [a for a in ACTIVITY_STYLES if a in set(swept['trial_type'])]

    for joint in joints:
        focus = _is_focus(joint)
        for activity in activities:
            wide = arm_table(swept, references, joint, activity)
            if wide.empty or 'mag_off' not in wide.columns:
                continue
            deltas = wide.drop(columns=['mag_off']).sub(wide['mag_off'], axis=0)
            curve = _numeric_arms(within_block_interval(deltas))
            if curve.empty:
                continue
            ax.plot(curve['distortion_scale'], curve['mean'], color=palette[joint],
                    linestyle=ACTIVITY_STYLES[activity], linewidth=3.0 if focus else 1.6,
                    marker='o' if focus else None, markersize=5.5,
                    alpha=1.0 if focus else 0.6, zorder=3)

    # The crossings, marked where they are read off. Only for blocks that actually cross
    # inside the sweep — a mark drawn at the edge for a block that never crosses would read as
    # a tolerance that was measured.
    if not marks.empty:
        crossings = (marks[marks['outcome'] == 'crosses']
                     .groupby(['joint_name', 'trial_type'])[['breakeven_scale', 'breakeven_deg']]
                     .mean())
        # Labelled only when there is ONE crossing. Splitting sides puts three or four of them
        # inside the first two thirds of the axis, and there is no free region of this panel big
        # enough for that much stacked text — it lands on the legend or on the curves either way.
        # The markers still say where every crossing is, and panel D, immediately to the right,
        # gives all of them in degrees against the level each block actually sees.
        crossings = crossings.sort_values('breakeven_scale')
        for (joint, activity), row in crossings.iterrows():
            if joint not in palette:
                continue
            ax.scatter([row['breakeven_scale']], [0.0], s=190, marker='v',
                       color=palette[joint], edgecolor='white', linewidth=1.0, zorder=6)
            if len(crossings) == 1:
                ax.annotate(f"{joint_label(joint)}, {ACTIVITY_LABELS[activity]}:\ntolerates "
                            f"{row['breakeven_deg']:.0f}°",
                            xy=(row['breakeven_scale'], 0.0), xytext=(12, 26),
                            textcoords='offset points', ha='left', fontsize=10,
                            fontweight='bold', color=palette[joint], zorder=6)

    ax.axhline(0.0, color=ARM_COLORS['mag_off'], linewidth=1.8, zorder=1)
    ax.text(0.99, 0.0, 'no better than mag_off ', transform=ax.get_yaxis_transform(),
            color=ARM_COLORS['mag_off'], fontsize=11, fontweight='bold', va='bottom', ha='right')

    _scale_axis(ax, swept['distortion_scale'].unique())
    ax.set_ylabel('RMSE − that joint\'s mag_off RMSE (deg)')
    handles = [Line2D([], [], color=palette[j], linewidth=3.0 if _is_focus(j) else 1.6,
                      marker='o' if _is_focus(j) else None, markersize=5.5, label=joint_label(j))
               for j in joints]
    handles += [Line2D([], [], color='#666666', linewidth=1.8, linestyle=ACTIVITY_STYLES[a],
                       label=ACTIVITY_LABELS[a]) for a in activities]
    handles.append(Line2D([], [], color='#666666', linewidth=0, marker='v', markersize=9,
                          label='tolerance (crossing)'))
    ax.legend(handles=handles, loc='upper left', fontsize=9.5, ncol=2)
    # No CI bands here, deliberately: eight curves with bands is unreadable, and the bands are
    # already shown on the pooled panel beside it. The tolerance markers are the quantity this
    # panel is read for, and panel D carries their spread across blocks.
    ax.text(0.99, 0.02, '▼ marks a crossing; panel D gives each one in degrees. Means over '
                        'blocks — see panel B for interval widths.',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontstyle='italic',
            color='#666666')
    sns.despine(ax=ax)

# ==============================================================================
# Panel D: the tolerance in degrees, against what this lab presents
# ==============================================================================

def panel_tolerance(ax: plt.Axes, marks: pd.DataFrame) -> None:
    """Each block's tolerance in degrees of field disagreement, against the level its joint
    actually sees here. The comparison is the result: a joint whose tolerance sits to the
    RIGHT of its measured level is one the magnetometer still helps at in this lab, and one to
    the left is a joint where it is already costing accuracy.

    Blocks that never cross inside the sweep are drawn as right-pointing arrows at the sweep's
    top end rather than dropped, since "the tolerance is above 40°" is a result and silently
    omitting those blocks would bias the visible tolerances downward.
    """
    if marks.empty:
        return
    joints = _present_joints(marks)
    palette = _joint_palette(joints)
    activities = [a for a in ACTIVITY_STYLES if a in set(marks['trial_type'])]

    # One row per joint x activity rather than per joint. The activity is on the axis because
    # it changes the ANSWER at the ankle, not just the number (see ACTIVITY_BLOCK_COLS): a row
    # per joint would average a "stops helping at 3 deg" against a "still helping at 16 deg"
    # and print their midpoint as a tolerance.
    rows = [(joint, activity) for joint in joints for activity in activities]
    positions = {row: i for i, row in enumerate(rows)}

    for (joint, activity), y in positions.items():
        subset = marks[(marks['joint_name'] == joint) & (marks['trial_type'] == activity)]
        if subset.empty:
            continue
        measured = float(subset['measured_deg'].mean())
        ax.plot([measured], [y], marker='|', markersize=26, markeredgewidth=3.0,
                color=ARM_COLORS['measured'], zorder=5)

        crosses = subset[subset['outcome'] == 'crosses']
        ax.scatter(crosses['breakeven_deg'], np.full(len(crosses), y), s=95, marker='o',
                   color=palette[joint], edgecolor='white', linewidth=0.9, zorder=4)
        # Beyond the sweep, in either direction.
        beyond = subset[subset['outcome'] == 'always_helps']
        ax.scatter(beyond['max_swept_deg'], np.full(len(beyond), y), s=130, marker='>',
                   color=palette[joint], edgecolor='white', linewidth=0.9, zorder=4)
        never = subset[subset['outcome'] == 'never_helps']
        ax.scatter(np.zeros(len(never)), np.full(len(never), y), s=130, marker='x',
                   color=palette[joint], linewidth=2.0, zorder=4)

    ax.set_yticks(list(positions.values()))
    ax.set_yticklabels([f"{joint_label(joint)} · {ACTIVITY_LABELS[activity]}"
                        for joint, activity in positions])
    ax.set_ylim(-0.6, len(positions) - 0.4)
    ax.invert_yaxis()  # proximal at the top, matching JOINT_ORDER
    ax.set_xlabel('Field disagreement between the joint\'s two sensors (deg)')
    # Room on the right for the legend, so it does not sit on top of the ankle's markers —
    # which are the furthest right and the ones the figure is read for.
    ax.set_xlim(0, max(float(marks[['breakeven_deg', 'measured_deg', 'max_swept_deg']]
                             .max().max()) * 1.12, 1.0))
    ax.grid(axis='x', alpha=0.25)

    handles = [
        Line2D([], [], color='#666666', marker='o', linewidth=0, markersize=9,
               label='tolerance (crosses mag_off)'),
        Line2D([], [], color='#666666', marker='>', linewidth=0, markersize=10,
               label='beyond the sweep (still helping)'),
        Line2D([], [], color='#666666', marker='x', linewidth=0, markersize=9, markeredgewidth=2,
               label='never helps, even at 0%'),
        Line2D([], [], color=ARM_COLORS['measured'], marker='|', linewidth=0, markersize=16,
               markeredgewidth=3, label='level measured in this lab'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=10,
              title='one marker per subject × joint × activity block', title_fontsize=9)
    sns.despine(ax=ax, left=True)

# ==============================================================================
# Significance
# ==============================================================================

def block_table(swept: pd.DataFrame) -> pd.DataFrame:
    """The long table plot_utils.test_panels blocks on, with the scale as a string group
    label (the group column has to be categorical for a Friedman/Wilcoxon panel)."""
    if swept.empty:
        return swept
    out = swept[[*STAT_BLOCK_COLS, 'distortion_scale', METRIC]].copy()
    out['scale_label'] = out['distortion_scale'].map(lambda s: f'{s:.0%}')
    return out


def scale_order(swept: pd.DataFrame) -> List[str]:
    return [f'{s:.0%}' for s in np.sort(swept['distortion_scale'].unique())]


def test_sweep(swept: pd.DataFrame) -> Dict[str, plot_utils.PanelTest]:
    """Friedman across the distortion levels + pairwise Wilcoxon, blocked as STAT_BLOCK_COLS says
    (subject x joint TYPE x activity), with the joints the experiment was set up for tested as
    their own panels, split by activity.

    Blocked on the type even under --split-sides: the sides are two correlated measurements of
    one joint, so testing them separately would inflate n and narrow every p-value on a
    precision claim the data does not support. The side split is a descriptive view only.

    All panels form one Holm family. The pooled panel answering "flat" while the ankle's
    walking panel answers "not flat" is the expected result, not a problem — the whole point of
    panel C is that the joints and the activities do not share a tolerance."""
    blocks = block_table(swept)
    if blocks.empty:
        return {}
    panels = {'All joints': blocks}
    for joint in FOCUS_JOINTS:
        for activity in [a for a in ACTIVITY_STYLES if a in set(blocks['trial_type'])]:
            subset = blocks[(blocks['joint_type'] == joint) & (blocks['trial_type'] == activity)]
            if not subset.empty:
                panels[f"{joint} · {ACTIVITY_LABELS[activity]}"] = subset
    return plot_utils.test_panels(panels, METRIC, 'scale_label', scale_order(swept),
                                 block_cols=STAT_BLOCK_COLS)


def significance_epilog(results: Dict[str, plot_utils.PanelTest]) -> str:
    """One line summarizing what the test found, for the figure footer."""
    testable = [r for r in results.values() if r.skipped_reason is None]
    if not testable:
        reasons = {r.skipped_reason for r in results.values() if r.skipped_reason}
        return f"No significance testing ({'; '.join(sorted(reasons)) or 'no data'})"
    verdicts = [f"{r.key}: {'differs' if r.friedman_p_adj < 0.05 else 'flat'} "
                f"(Friedman p={r.friedman_p_adj:.3g}, W={r.kendalls_w:.2f})" for r in testable]
    return ("Friedman across distortion levels + pairwise Wilcoxon, Holm-corrected across all "
            f"panels; blocked by subject × joint type × activity (sides averaged into the block, "
            f"they are not independent replicates), n={testable[0].n_blocks} — "
            + "; ".join(verdicts))


# How many block verdicts go on one line of the figure's footer. Wrapped rather than joined into
# one string because the footer is drawn with bbox_inches='tight': a 14-block single line does not
# get truncated, it stretches the saved canvas to several times the width of the panels.
EPILOG_PER_LINE = 4


def tolerance_epilog(marks: pd.DataFrame) -> str:
    """The figure's headline verdicts, generated from its own numbers so they cannot drift away
    from the panels. Newline-separated, at most EPILOG_PER_LINE per line."""
    if marks.empty:
        return "No tolerance estimated (the mag_off reference is missing from the sweep)."
    lines = []
    activities = [a for a in ACTIVITY_STYLES if a in set(marks['trial_type'])]
    for joint in _present_joints(marks):
        for activity in activities:
            subset = marks[(marks['joint_name'] == joint) & (marks['trial_type'] == activity)]
            if subset.empty:
                continue
            crosses = subset[subset['outcome'] == 'crosses']
            measured = float(subset['measured_deg'].mean())
            if len(crosses) == len(subset):
                verdict = f"tolerates {crosses['breakeven_deg'].mean():.0f}°"
            elif crosses.empty and (subset['outcome'] == 'always_helps').all():
                verdict = f"still helping past {subset['max_swept_deg'].mean():.0f}°"
            elif crosses.empty and (subset['outcome'] == 'never_helps').all():
                verdict = "never helps, even in a clean field"
            else:
                counts = subset['outcome'].value_counts().to_dict()
                verdict = "mixed " + ", ".join(f"{v}x {k}" for k, v in sorted(counts.items()))
            lines.append(f"{joint_label(joint)} ({ACTIVITY_LABELS[activity]}): {verdict}, "
                         f"sees {measured:.0f}° here")
    return "\n".join("  |  ".join(lines[i:i + EPILOG_PER_LINE])
                     for i in range(0, len(lines), EPILOG_PER_LINE))

# ==============================================================================
# Composed figure
# ==============================================================================

# Panel B carries a dose axis on top, which needs room between the plot frame and the title.
# Only the TITLE moves for it: the panel letters sit left of the frame (x = -0.12), where the
# secondary axis's ticks and label never reach, so they can stay on one line across the figure
# rather than stepping up over panel B — which at this figure's height would put the letter in
# the header.
TITLE_PAD = {False: 10, True: 34}
LETTER_Y = {False: 1.06, True: 1.06}


def _label_panel(ax: plt.Axes, letter: str, has_dose_axis: bool = False) -> None:
    ax.text(-0.12, LETTER_Y[has_dose_axis], letter, transform=ax.transAxes, fontsize=20,
            fontweight='bold', va='bottom', ha='left')


def figure_supplement(swept: pd.DataFrame, references: pd.DataFrame, dose: pd.DataFrame,
                      marks: pd.DataFrame, results: Dict[str, plot_utils.PanelTest],
                      filename: str = 'distortion_tolerance.png',
                      save: bool = True, show: bool = False) -> None:
    """The four-panel supplement.

    Constrained layout rather than plot_utils.finalize_and_save_plot's tight_layout: panel B
    carries a secondary x-axis on top, which tight_layout collides with the title above it."""
    dose_line = dose_curve(dose)
    fig = plt.figure(figsize=(17, 12.5), layout='constrained')
    axes = fig.subplots(2, 2)

    for ax, letter, title, has_dose_axis, draw in (
            (axes[0][0], 'A', 'What a distortion level is, in degrees', False,
             lambda ax: panel_dose(ax, dose)),
            (axes[0][1], 'B', 'Accuracy across the swept distortion range', True,
             lambda ax: panel_pooled(ax, swept, references, dose_line)),
            (axes[1][0], 'C', 'Where the magnetometer stops paying, by joint', False,
             lambda ax: panel_by_joint(ax, swept, references, marks)),
            (axes[1][1], 'D', 'Tolerance against the distortion this lab presents', False,
             lambda ax: panel_tolerance(ax, marks))):
        draw(ax)
        ax.set_title(title, fontsize=15, pad=TITLE_PAD[has_dose_axis])
        _label_panel(ax, letter, has_dose_axis)

    # Both lines in the suptitle so constrained layout reserves space for them, and at 16pt
    # rather than the 19 the other supplements use: the coverage line is long, and at 19pt it is
    # wider than the figure — bbox_inches='tight' then widens the saved canvas to fit it instead
    # of wrapping, which shrinks every panel to a third of its size.
    fig.suptitle('How much magnetic distortion MAJIC tolerates before the magnetometer '
                 'stops helping\n' + describe_coverage(swept, dose),
                 fontsize=16, fontweight='bold')
    # Constrained layout does not reserve space for a bare fig.text, so the footer gets its
    # own strip at the bottom of the layout rectangle rather than being drawn over panel D's
    # x-label. The strip grows with the number of verdict lines — with sides split there are
    # twice as many blocks to report, and a fixed strip would put them over the x-label.
    # The top is pulled well down: panel B's letter and its secondary dose axis both sit above
    # its frame, and at 0.935 they landed on the header.
    verdicts = tolerance_epilog(marks)
    strip = 0.022 + 0.016 * verdicts.count('\n')
    fig.get_layout_engine().set(rect=(0.0, strip + 0.022, 1.0, 0.895))
    fig.text(0.99, 0.020, verdicts, ha='right', va='bottom', fontsize=10, fontweight='bold',
             linespacing=1.5)
    fig.text(0.99, 0.004, significance_epilog(results), ha='right', va='bottom', fontsize=9,
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


def figures_standalone(swept: pd.DataFrame, references: pd.DataFrame, dose: pd.DataFrame,
                       marks: pd.DataFrame, suffix: str = '',
                       save: bool = True, show: bool = False) -> None:
    """Each panel again at full size, one file each, for dropping into the manuscript."""
    dose_line = dose_curve(dose)
    for draw, name, title, has_dose_axis in (
            (lambda ax: panel_dose(ax, dose), 'distortion_dose',
             'Field disagreement produced by each distortion level', False),
            (lambda ax: panel_pooled(ax, swept, references, dose_line), 'pooled_rmse',
             'Joint-angle RMSE against applied magnetic distortion', True),
            (lambda ax: panel_by_joint(ax, swept, references, marks), 'rmse_by_joint',
             'Gain over mag_off against applied distortion, by joint', False),
            (lambda ax: panel_tolerance(ax, marks), 'tolerance',
             'Distortion tolerance per joint, against the measured level', False)):
        # Taller when the rows are joint x side x activity, which is twice as many as pooled.
        height = 6.5 if not suffix else 8.0
        fig, ax = plt.subplots(figsize=(9, height), layout='constrained')
        draw(ax)
        ax.set_title(title, fontsize=15, pad=TITLE_PAD[has_dose_axis])
        _save(fig, f'{name}{suffix}.png', save, show)

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--composed-only', action='store_true',
                        help='Write only the composed figure, not the per-panel files.')
    parser.add_argument('--split-sides', action='store_true',
                        help="Keep the left and right legs apart in the descriptive panels (A, C, "
                             "D) instead of pooling them into a joint type, and write to "
                             "*_by_side.png so both versions coexist. The significance test and "
                             "panel B stay blocked on the joint type either way — the two sides "
                             "correlate at ICC ~0.5, so testing them separately would inflate n "
                             "(see STAT_BLOCK_COLS).")
    parser.add_argument('--show', action='store_true', help='Also display the figures.')
    args = parser.parse_args()

    suffix = '_by_side' if args.split_sides else ''
    swept, references = load_sweep(split_sides=args.split_sides)
    if swept.empty:
        print(f"Error: no mag_on_dist* rows in {paths.statistics_path(STATS_NAME)}. "
              f"Run `python -m experiments.distortion_tolerance` first.")
        return
    if references.empty:
        print("Warning: no mag_off/mag_on rows in the statistics file — panels C and D need "
              "the mag_off floor. Re-run the experiment without --skip-references.")

    dose = load_dose(split_sides=args.split_sides)
    if dose.empty:
        print("Warning: no dose tables on disk — panel A will be empty and no tolerance can "
              "be quoted in degrees. Run "
              "`python -m experiments.distortion_tolerance --distortion-only`.")

    print(describe_coverage(swept, dose))
    marks = breakeven(swept, references, dose)
    if not marks.empty:
        print("\n=== Distortion tolerance per block ===")
        print(marks.to_string(index=False))
    print("\n" + tolerance_epilog(marks))

    results = test_sweep(swept)
    figure_supplement(swept, references, dose, marks, results,
                      filename=f'distortion_tolerance{suffix}.png', show=args.show)
    if results:
        plot_utils._emit_significance_report(plot_utils.significance_report(results, METRIC),
                                             f'distortion_tolerance{suffix}.png', PLOTS_DIR)
    if not args.composed_only:
        figures_standalone(swept, references, dose, marks, suffix=suffix, show=args.show)
    print(f"\nFigures under {PLOTS_DIR}")


if __name__ == '__main__':
    main()
