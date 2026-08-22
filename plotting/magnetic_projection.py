"""
Figures for experiments/magnetic_projection.py: does a magnetometer reading transport to the
joint centre, and can an array of them do better than assuming it does?

Reads ONLY the tables that experiment wrote. Nothing here reloads a trial, re-projects or
re-fits, so no number in a figure can disagree with the parquet beside it — the same tier
discipline plotting/acceleration_projection.py and plotting/joint_dof.py follow, and for the same
reason: a figure that recomputes is a second implementation nobody diffs.

    python -m plotting.magnetic_projection --dataset alborno
    python -m plotting.magnetic_projection --dataset imove
    python -m plotting.magnetic_projection --dataset imove --figures agreement modes

THE THREE FAMILIES, which every figure keys on, and which are the same three the acceleration
projection uses so the two sets of figures can be read side by side:

    marker    the transported field at the joint centre against a field MODEL the markers
              support — a constant global vector, or a polynomial in marker position. The only
              external reference, and the only one that can catch an error both segments make
              together; its truth is a model, so its residual includes the model's own
              inadequacy.
    joint     parent and child both transported to one point, so their readings should be one
              vector in two frames. No field model enters, and this identity IS what a relative
              filter's magnetometer residual measures. Blind to a common-mode error.
    segment   two magnetometers on ONE rigid segment. The truth is another magnetometer and the
              frame transform between them is a constant, so this is the sharpest of the three
              and the only one that needs no per-sample mocap. IMoVE only.

AND THE TWO CHANNELS, which are different quantities and are never drawn on one axis:

    vector    both fields in one common frame, so the full 3-vector residual and its direction
              are defined.
    norm      magnitudes only. For a magnetometer this needs NO mocap whatever — |R m| = |m| —
              so it is scored over the whole record, and it is the part of a disagreement that
              no orientation estimate can ever absorb.

SIX FIGURES, each answering one question the console report answers in prose:

  agreement   The headline. All three families at the 0th order, the per-joint breakdown, the
              mocap-free magnitude channel, and disagreement against sensor separation. That last
              panel is the one to read first: a smooth field gradient produces a disagreement
              PROPORTIONAL to separation, and a per-sensor error does not scale with distance at
              all.

  modes       The question the experiment was built for: eight ways of transporting the reading,
              scored on cross-segment agreement AND on distance from the reference. Drawn as a
              two-axis scatter rather than one bar chart, because a mode that improves agreement
              while moving away from the reference has moved error around — and `body_model`,
              which scores perfect agreement by discarding the measurement, is on the plot to
              make that visible.

  mechanism   Gradient or sensors? The variance each candidate model explains, the fitted
              gradient magnitudes against the one the markers can see, and the physical
              (symmetric, traceless) constraint that separates a magnetic field from a
              nine-parameter regression.

  covariation The complementary question to every other figure: not how far two sensors disagree
              but how much of what they see is SHARED, which is what decides whether differencing
              them helps. Adjacency is the axis, and the far pair classes are the control.

  holdout     Two magnetometers on a segment predicting the third — the only fully out-of-sample
              test in the experiment — split by whether the prediction interpolates or
              extrapolates.

  reference   The marker-supported field models: per-segment deviation proximal to distal, and
              the lab map's in-sample against leave-one-sensor-out skill.

  calibration What a per-sensor calibration buys and whether it transfers between a subject's
              trials, which is what decides whether it is usable at all.

Every figure is pooled over every subject and trial present on disk, and its epilog says how many.
"""
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

import paths
from plotting import utils as plot_utils
from experiments.magnetic_projection import (CALIBRATIONS, DATASETS, FAMILIES, FAMILY_LABELS,
                                             LINE_MODE_ORDER, MAG_UNIT, MODES, PAIR_CLASSES,
                                             PRIMARY_CALIBRATION, PRIMARY_MODE, SLOW_BAND_HZ,
                                             dataset_dir, get_dataset, load_trial_table,
                                             statistics_path)

PLOTS_DIR = paths.plots_dir("magnetic_projection")

FIGURES = ('agreement', 'modes', 'mechanism', 'covariation', 'holdout', 'reference',
           'calibration')

# --- Colours ---------------------------------------------------------------------------------
# One colour per family, fixed here rather than taken from a palette so a family is the same
# colour in every panel and across datasets. Matches plotting/acceleration_projection.py's
# assignment exactly, which is the point of using the same family names.
FAMILY_COLORS = {'marker': '#8a4fa8', 'joint': '#2f6fb5', 'segment': '#4c9f70'}
FAMILY_SHORT = {'marker': 'vs field model', 'joint': 'parent vs child', 'segment': 'sensor vs sensor'}

# Hatching, not a third colour, for the magnitude channel: it is the SAME family measured a
# different way, so sharing the family's colour and marking the channel is the honest encoding.
CHANNEL_HATCH = {'vector': '', 'norm': '///'}

# One colour per mode. The 0th order is grey and heaviest because it is the baseline rather than a
# competitor; the degenerate control is red because it is a warning, not a proposal.
MODE_COLORS = {
    'sensor': '#4a4a4a',
    'mean': '#7aa6c2',
    'linear': '#2f6fb5',
    'quadratic': '#1b3f6b',
    'body_linear': '#e8973a',
    'trial_gradient': '#c77d2a',
    'trial_gradient_phys': '#4c9f70',
    'body_model': '#b4432f',
}
MODE_LABELS = {
    'sensor': '0th order (the reading)',
    'mean': 'segment mean',
    'linear': '1st order along the array',
    'quadratic': '2nd order along the array',
    'body_linear': 'whole-body gradient, per sample',
    'trial_gradient': 'whole-body gradient, per trial',
    'trial_gradient_phys': 'per trial, physical (sym. traceless)',
    'body_model': 'model only — the CONTROL',
}
# Short forms for TICK LABELS. The long ones above are for legends and annotations, where there is
# room; on an axis they are wide enough to run into the neighbouring panel, which is what they did.
MODE_SHORT = {
    'sensor': '0th order\n(the reading)',
    'mean': 'segment\nmean',
    'linear': '1st order\nalong array',
    'quadratic': '2nd order\nalong array',
    'body_linear': 'body G\nper sample',
    'trial_gradient': 'body G\nper trial',
    'trial_gradient_phys': 'body G\nphysical',
    'body_model': 'model only\nCONTROL',
}

CALIBRATION_COLORS = {'none': '#4a4a4a', 'hard_iron': '#2f6fb5', 'hard_iron_own': '#7aa6c2',
                      'affine': '#e8973a'}
CALIBRATION_LABELS = {'none': 'as recorded', 'hard_iron': 'hard iron (other trials)',
                      'hard_iron_own': 'hard iron (this trial — in sample)',
                      'affine': 'affine (other trials)'}

MECHANISM_COLORS = {'explained_body': '#b4432f', 'explained_world': '#7aa6c2',
                    'explained_gradient': '#e8973a', 'explained_gradient_physical': '#4c9f70',
                    'explained_combined': '#4a4a4a'}
MECHANISM_LABELS = {'explained_body': 'body-frame constant\n(hard iron / gain)',
                    'explained_world': 'world-frame constant',
                    'explained_gradient': 'gradient, 9 free',
                    'explained_gradient_physical': 'gradient, PHYSICAL (5 free)',
                    'explained_combined': 'body + gradient together'}

WHISKER_PCT = (5, 95)

# Anatomical ordering, proximal to distal, because every magnetic result in this repository
# degrades down the limb and a figure sorted alphabetically hides that.
LEVEL_ORDER = ('Lumbar', 'Torso', 'Pelvis', 'Hip', 'Thigh', 'Femur', 'Knee', 'Shank', 'Tibia',
               'Ankle', 'Foot', 'Calcn')


def _level_key(group: str) -> Tuple[int, str]:
    """Sort key placing a group at its anatomical level, proximal first, unknowns last."""
    for index, level in enumerate(LEVEL_ORDER):
        if level.lower() in str(group).lower():
            return index, str(group)
    return len(LEVEL_ORDER), str(group)


def order_groups(groups: Sequence[str], spec=None) -> List[str]:
    """The groups to draw, proximal to distal, restricted to the spec's PRIMARY joints if given.

    The restriction is not cosmetic. IMoVE's joint table has 18 pairs against 6 anatomical joints,
    because each thigh and shank carries three sensors and every placement borders the same two
    joints; drawing all 18 in a third-width panel smears the tick labels and says nothing the 6 do
    not. Every variant is still measured and still in the parquet — the placement comparison has
    its own panel, which is the only place it says something the joint labels cannot.
    """
    present = {str(g) for g in groups if str(g) != 'all'}
    if spec is not None and getattr(spec, 'primary_joints', None):
        primary = present & set(spec.primary_joints)
        if primary:
            present = primary
    return sorted(present, key=_level_key)

# ==============================================================================
# Loading
# ==============================================================================

def load_tables(dataset: str, figures: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Every table the requested figures need, and nothing else."""
    wanted = set(figures)
    tables: Dict[str, pd.DataFrame] = {}
    tables['stats'] = load_trial_table(dataset, 'agreement_stats')
    if 'covariation' in wanted:
        tables['covariation'] = load_trial_table(dataset, 'covariation')
    if wanted & {'mechanism'}:
        tables['mechanism'] = load_trial_table(dataset, 'mechanism')
        tables['gradients'] = load_trial_table(dataset, 'gradient_fits')
    if wanted & {'reference'}:
        tables['lab'] = load_trial_table(dataset, 'lab_field')
    if wanted & {'calibration', 'reference'}:
        tables['calibration'] = load_trial_table(dataset, 'calibration')
    return tables


def coverage(frame: pd.DataFrame) -> str:
    """One line naming what a figure is pooled over, for the epilog."""
    if frame.empty:
        return "no data"
    cells = frame[['subject', 'trial']].drop_duplicates()
    return f"{len(cells)} trials, {cells['subject'].nunique()} subjects"


def cells(stats: pd.DataFrame, **filters) -> pd.DataFrame:
    """The stats rows matching every filter. Values may be a scalar or a collection."""
    frame = stats
    for column, value in filters.items():
        if column not in frame.columns:
            return frame.iloc[0:0]
        if isinstance(value, (list, tuple, set)):
            frame = frame[frame[column].isin(list(value))]
        else:
            frame = frame[frame[column] == value]
    return frame

# ==============================================================================
# Shared panel pieces
# ==============================================================================

def _box_stats(values: pd.Series) -> Dict[str, float]:
    """5-number summary in matplotlib's bxp format, with p5/p95 whiskers."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lo, hi = np.percentile(values, WHISKER_PCT)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {'med': median, 'q1': q1, 'q3': q3, 'whislo': lo, 'whishi': hi, 'fliers': []}


def grouped_boxes(ax: plt.Axes, frame: pd.DataFrame, group_col: str, order: Sequence[str],
                  value_col: str, colors: Dict[str, str], ylabel: str = '',
                  labels: Optional[Dict[str, str]] = None, log: bool = False,
                  hatches: Optional[Dict[str, str]] = None, rotation: float = 30) -> None:
    """One box per group, drawn from precomputed quantiles.

    ax.bxp rather than seaborn.boxplot on melted data: the quantiles drawn are then exactly the
    ones the parquet reports, instead of whatever seaborn recomputes after its own filtering.
    """
    labels = labels or {}
    stats, positions, colours, patterns = [], [], [], []
    for index, group in enumerate(order):
        values = frame.loc[frame[group_col].astype(str) == group, value_col].dropna()
        if values.empty:
            continue
        stats.append({'label': labels.get(group, group), **_box_stats(values)})
        positions.append(index)
        colours.append(colors.get(group, '#888888'))
        patterns.append((hatches or {}).get(group, ''))
    if not stats:
        _no_data(ax, "nothing to draw")
        return
    artists = ax.bxp(stats, positions=positions, widths=0.6, showfliers=False, patch_artist=True,
                     medianprops={'color': 'black', 'linewidth': 1.6})
    for patch, colour, hatch in zip(artists['boxes'], colours, patterns):
        patch.set_facecolor(colour)
        patch.set_edgecolor(colour)
        patch.set_alpha(0.75)
        if hatch:
            patch.set_hatch(hatch)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels.get(g, g) for g in order], rotation=rotation,
                       ha='right' if rotation else 'center', fontsize=10)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')
    ax.grid(axis='x', visible=False)
    ax.text(0.99, 0.02, f'boxes: IQR, whiskers: p{WHISKER_PCT[0]}–p{WHISKER_PCT[1]}, '
                        f'one point per trial',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontstyle='italic',
            color='#666666')
    sns.despine(ax=ax)


def _label_panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.09, 1.06, letter, transform=ax.transAxes, fontsize=20, fontweight='bold',
            va='bottom', ha='right')


def _no_data(ax: plt.Axes, message: str) -> None:
    """A panel with nothing to draw says so, rather than rendering as empty axes.

    Which happens legitimately: the segment family exists only on IMoVE, and a partial run may
    have no trial with a long enough mocap window."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha='center', va='center', fontsize=12,
            color='#777777', wrap=True)
    ax.set_axis_off()

# ==============================================================================
# Figure: agreement — the headline
# ==============================================================================

def panel_families(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """One box per family and channel, at the 0th order.

    The panel a reader is most likely to use to compare two things that are not comparable, so
    the magnitude channel is hatched and the axis says what it is. A vector residual and a
    magnitude gap are different quantities; what IS comparable is each against the field's own
    magnitude, which is why the axis is a percentage of it.
    """
    frame = cells(stats, mode=PRIMARY_MODE, calibration=PRIMARY_CALIBRATION).copy()
    if frame.empty:
        _no_data(ax, "no comparisons")
        return
    frame['key'] = frame['family'].astype(str) + '|' + frame['channel'].astype(str)
    frame['relative'] = 100 * frame['err_p50'] / frame['truth_norm_p50']
    order = [f'{family}|{channel}' for family in FAMILIES for channel in ('vector', 'norm')
             if f'{family}|{channel}' in set(frame['key'])]
    grouped_boxes(
        ax, frame, 'key', order, 'relative',
        colors={key: FAMILY_COLORS[key.split('|')[0]] for key in order},
        hatches={key: CHANNEL_HATCH[key.split('|')[1]] for key in order},
        labels={key: f"{FAMILY_SHORT[key.split('|')[0]]}\n({key.split('|')[1]})" for key in order},
        ylabel=f'Disagreement, % of the field magnitude')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=FAMILY_COLORS[f], alpha=0.75,
                                     label=FAMILY_LABELS[f]) for f in FAMILIES
                       if any(k.startswith(f) for k in order)],
              loc='upper left', fontsize=9)
    ax.set_title('The three families, 0th order')


def panel_by_joint(ax: plt.Axes, stats: pd.DataFrame, spec) -> None:
    """Cross-segment direction disagreement per joint, proximal to distal."""
    frame = cells(stats, family='joint', mode=PRIMARY_MODE, calibration=PRIMARY_CALIBRATION,
                  channel='vector')
    if frame.empty:
        _no_data(ax, "no cross-segment comparisons")
        return
    order = order_groups(frame['group'].unique(), spec)
    grouped_boxes(ax, frame, 'group', order, 'ang_p50',
                  colors={g: FAMILY_COLORS['joint'] for g in order},
                  ylabel='Direction disagreement (deg)')
    ax.set_title('Parent vs child, by joint')


def panel_separation(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """Disagreement against how far apart the two sensors are.

    THE PANEL THAT DISCRIMINATES THE TWO MECHANISMS, and the reason both families are on one axis
    even though they measure different pairs. A smooth spatial field gradient produces a
    disagreement PROPORTIONAL to separation, so the same-segment points (7-19 cm) and the
    cross-joint points (25-47 cm) should fall on one line through the origin. A per-sensor error
    does not scale with distance at all, so the short baselines sit far above that line.

    The dashed line is what the markers can see: the lab field map's own gradient times the
    separation, i.e. the disagreement a real, marker-visible field gradient would produce over
    that distance.
    """
    frame = cells(stats, mode=PRIMARY_MODE, calibration=PRIMARY_CALIBRATION, channel='vector')
    frame = frame[frame['family'].isin(['joint', 'segment'])
                  & frame['sep_m'].notna() & (frame['sep_m'] > 0)]
    if frame.empty:
        _no_data(ax, "no paired comparisons with a measured separation")
        return
    for family, group in frame.groupby('family', observed=True):
        ax.scatter(1000 * group['sep_m'], group['err_p50'], s=14, alpha=0.35,
                   color=FAMILY_COLORS[str(family)], label=FAMILY_SHORT[str(family)],
                   edgecolors='none')
    # Median per separation decile, so the trend is readable through the scatter.
    binned = frame.assign(bin=pd.qcut(frame['sep_m'], q=min(8, frame['sep_m'].nunique()),
                                      duplicates='drop'))
    trend = binned.groupby('bin', observed=True).agg(sep=('sep_m', 'median'),
                                                     err=('err_p50', 'median')).dropna()
    ax.plot(1000 * trend['sep'], trend['err'], color='black', linewidth=2.0, marker='o',
            markersize=5, label='median per separation bin', zorder=5)
    ax.set_xlabel('Sensor separation (mm)')
    ax.set_ylabel(f'Field disagreement ({MAG_UNIT})')
    ax.set_title('Does the disagreement scale with distance?')
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(ax=ax)


def panel_gradient_line(ax: plt.Axes, stats: pd.DataFrame, gradients: pd.DataFrame) -> None:
    """The same data as a RATE — disagreement per 100 mm — which is flat if it is a gradient."""
    frame = cells(stats, mode=PRIMARY_MODE, calibration=PRIMARY_CALIBRATION, channel='vector')
    frame = frame[frame['family'].isin(['joint', 'segment'])
                  & frame['sep_m'].notna() & (frame['sep_m'] > 0)].copy()
    if frame.empty:
        _no_data(ax, "no paired comparisons with a measured separation")
        return
    frame['rate'] = frame['err_p50'] / (frame['sep_m'] * 10.0)
    order = [f for f in ('segment', 'joint') if f in set(frame['family'].astype(str))]
    grouped_boxes(ax, frame, 'family', order, 'rate',
                  colors=FAMILY_COLORS, labels=FAMILY_SHORT,
                  ylabel=f'Disagreement per 100 mm ({MAG_UNIT})')
    if not gradients.empty and 'lab_map_gradient_norm' in gradients.columns:
        lab = float(gradients['lab_map_gradient_norm'].median()) / 10.0
        ax.axhline(lab, color='#b4432f', linewidth=1.8, linestyle='--')
        ax.text(0.98, lab, f'  what the markers see\n  ({lab:.3f} per 100 mm)',
                transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=9,
                color='#b4432f')
    ax.set_title('The same, as a rate')


def plot_agreement(dataset: str, stats: pd.DataFrame, gradients: pd.DataFrame,
                   save: bool = True, show: bool = False) -> None:
    spec = get_dataset(dataset)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    panel_families(axes[0, 0], stats)
    panel_by_joint(axes[0, 1], stats, spec)
    panel_separation(axes[1, 0], stats)
    panel_gradient_line(axes[1, 1], stats, gradients)
    for ax, letter in zip(axes.ravel(), 'ABCD'):
        _label_panel(ax, letter)
    plot_utils.finalize_and_save_plot(
        fig, f"Transporting a magnetometer reading to the joint centre — {dataset}",
        f"agreement_{dataset}.png", PLOTS_DIR, epilog=coverage(stats), save=save, show=show,
        caption=(
            "How wrong the 0th-order magnetic projection — transporting the reading unchanged, "
            "which is what every filter in this repository does — actually is. (A) The three "
            "families as a percentage of the field's own magnitude; hatched boxes are the "
            "magnitude-only channel, which needs no mocap at all and is the part of a "
            "disagreement no orientation estimate can absorb. (B) Cross-segment direction "
            "disagreement per joint, proximal to distal — this is the residual the relative "
            "filter's magnetometer update consumes. (C) and (D) are the discriminating panels: a "
            "smooth spatial field gradient produces a disagreement proportional to separation, so "
            "the two families would fall on one line through the origin in C and on one level in "
            "D. They do not — the short same-segment baselines disagree far more per millimetre "
            "than the long cross-joint ones — which is the signature of a per-sensor, body-fixed "
            "term rather than of a field. The dashed line in D is the gradient the marker-based "
            "lab field map can actually see. One point per trial and comparison; boxes are IQR "
            "with p5-p95 whiskers."))

# ==============================================================================
# Figure: modes — does a higher-order projection help?
# ==============================================================================

def panel_mode_boxes(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """Cross-segment agreement under each mode, with the 0th order marked."""
    frame = cells(stats, family='joint', calibration=PRIMARY_CALIBRATION, channel='vector')
    if frame.empty:
        _no_data(ax, "no cross-segment comparisons")
        return
    order = [m for m in MODES if m in set(frame['mode'].astype(str))]
    grouped_boxes(ax, frame, 'mode', order, 'ang_p50', colors=MODE_COLORS, labels=MODE_SHORT,
                  ylabel='Cross-segment disagreement (deg)', rotation=0)
    baseline = frame.loc[frame['mode'] == PRIMARY_MODE, 'ang_p50'].median()
    if np.isfinite(baseline):
        ax.axhline(baseline, color=MODE_COLORS['sensor'], linewidth=1.4, linestyle=':')
        ax.text(0.02, baseline, ' 0th order', transform=ax.get_yaxis_transform(), ha='left',
                va='bottom', fontsize=9, color=MODE_COLORS['sensor'])
    ax.set_title('Agreement by projection mode')


def panel_mode_tradeoff(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """Agreement against distance from the reference, one point per mode.

    THE PANEL THAT STOPS THE METRIC BEING GAMED. Agreement alone can be driven to zero by
    discarding the measurement, which is exactly what `body_model` does — it sits on the left
    edge and at the top, and the arrow from the 0th order shows the trade every other mode is
    making. The bottom-left corner is the only place an improvement can be.
    """
    joint = cells(stats, family='joint', calibration=PRIMARY_CALIBRATION, channel='vector')
    marker = cells(stats, family='marker', calibration=PRIMARY_CALIBRATION, channel='vector')
    if joint.empty or marker.empty:
        _no_data(ax, "needs both the joint and marker families")
        return
    agreement = joint.groupby('mode', observed=True)['ang_p50'].median()
    reference = marker.groupby('mode', observed=True)['ang_p50'].median()
    for mode in MODES:
        if mode not in agreement.index or mode not in reference.index:
            continue
        ax.scatter(agreement[mode], reference[mode], s=190, color=MODE_COLORS[mode],
                   edgecolors='white', linewidth=1.5, zorder=3)
        ax.annotate(MODE_SHORT[mode].replace('\n', ' '), (agreement[mode], reference[mode]),
                    textcoords='offset points', xytext=(9, 5), fontsize=9)
    if PRIMARY_MODE in agreement.index and PRIMARY_MODE in reference.index:
        ax.axvline(agreement[PRIMARY_MODE], color='#bbbbbb', linewidth=1.0, linestyle=':')
        ax.axhline(reference[PRIMARY_MODE], color='#bbbbbb', linewidth=1.0, linestyle=':')
        ax.text(0.02, 0.02, 'better on both\n(nothing is here)', transform=ax.transAxes,
                fontsize=10, color='#777777', va='bottom', ha='left')
    ax.set_xlabel('Cross-segment disagreement (deg)\nlower is better →')
    ax.set_ylabel('Distance from the field model (deg)\nlower is better →')
    ax.set_title('Agreement is not accuracy')
    sns.despine(ax=ax)


def panel_mode_paired(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """Per-cell ratio against the 0th order, so a mode that wins on average but loses badly on
    some cells is visible as a tail rather than averaged away."""
    frame = cells(stats, family='joint', calibration=PRIMARY_CALIBRATION, channel='vector')
    if frame.empty:
        _no_data(ax, "no cross-segment comparisons")
        return
    wide = frame.pivot_table(index=['subject', 'trial', 'group'], columns='mode',
                             values='ang_p50', observed=True)
    # Pivoting on a categorical leaves a CategoricalIndex on the columns, which `.dropna()`
    # and `.join()` both raise on. Plain strings from here.
    wide.columns = [str(column) for column in wide.columns]
    if PRIMARY_MODE not in wide.columns:
        _no_data(ax, "no 0th-order baseline")
        return
    modes = [m for m in MODES if m in wide.columns and m not in (PRIMARY_MODE, 'body_model')]
    data, positions, colours = [], [], []
    for index, mode in enumerate(modes):
        pair = wide[[mode, PRIMARY_MODE]].dropna()
        if pair.empty:
            continue
        data.append(np.log10((pair[mode] / pair[PRIMARY_MODE]).to_numpy()))
        positions.append(index)
        colours.append(MODE_COLORS[mode])
    if not data:
        _no_data(ax, "no paired cells")
        return
    parts = ax.violinplot(data, positions=positions, widths=0.75, showmedians=True,
                          showextrema=False)
    for body, colour in zip(parts['bodies'], colours):
        body.set_facecolor(colour)
        body.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    ax.axhline(0.0, color='#b4432f', linewidth=1.6, linestyle='--')
    ax.text(0.99, 0.0, ' no change ', transform=ax.get_yaxis_transform(), ha='right',
            va='bottom', fontsize=9, color='#b4432f')
    ax.set_xticks(positions)
    ax.set_xticklabels([MODE_SHORT[modes[i]] for i in positions], fontsize=10)
    ax.set_ylabel('log10(mode / 0th order)\nper trial x joint')
    ax.set_title('Paired against the 0th order')
    ax.grid(axis='x', visible=False)
    sns.despine(ax=ax)


def panel_extrapolation(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """Error against how far past the end of the sensor array the target sits.

    The geometric explanation for the panel above: a segment-scale fit is an EXTRAPOLATION to the
    joint centre, and the higher the order the faster it diverges out there. Both the joint-centre
    projections and the held-out-sensor predictions are drawn, because they share one x-axis and
    the held-out points at low extrapolation are the only place a 1st-order fit is asked to
    interpolate.
    """
    frame = stats[stats['mode'].isin(list(LINE_MODE_ORDER))
                  & (stats['calibration'] == PRIMARY_CALIBRATION)].copy()
    columns = [c for c in ('extrapolation', 'parent_extrapolation', 'child_extrapolation')
               if c in frame.columns]
    if frame.empty or not columns:
        _no_data(ax, "no segment-scale fits (needs 2+ sensors on a segment)")
        return
    frame['x'] = frame[columns].mean(axis=1)
    frame = frame[frame['x'].notna()]
    if frame.empty:
        _no_data(ax, "no segment-scale fits with a measured extrapolation")
        return
    for mode, group in frame.groupby('mode', observed=True):
        ax.scatter(group['x'], group['ang_p50'], s=16, alpha=0.4, color=MODE_COLORS[str(mode)],
                   label=MODE_LABELS[str(mode)], edgecolors='none')
    ax.axvline(1.0, color='#999999', linewidth=1.0, linestyle=':')
    ax.text(1.02, 0.98, ' one full array span\n beyond the last sensor', transform=
            ax.get_xaxis_transform(), fontsize=9, color='#777777', va='top')
    ax.set_xlabel('Extrapolation: array spans beyond the last sensor\n(0 = interpolation)')
    ax.set_ylabel('Direction error (deg)')
    ax.set_yscale('log')
    ax.set_title('Why the higher orders lose')
    ax.legend(loc='upper left', fontsize=9)
    sns.despine(ax=ax)


def plot_modes(dataset: str, stats: pd.DataFrame, save: bool = True, show: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    panel_mode_boxes(axes[0, 0], stats)
    panel_mode_tradeoff(axes[0, 1], stats)
    panel_mode_paired(axes[1, 0], stats)
    panel_extrapolation(axes[1, 1], stats)
    for ax, letter in zip(axes.ravel(), 'ABCD'):
        _label_panel(ax, letter)
    plot_utils.finalize_and_save_plot(
        fig, f"Can several magnetometers beat the 0th-order projection? — {dataset}",
        f"modes_{dataset}.png", PLOTS_DIR, epilog=coverage(stats), save=save, show=show,
        caption=(
            "Eight ways of transporting a magnetometer reading to the joint centre, scored on the "
            "agreement between the two segments spanning it. (A) None of the spatial models beats "
            "the 0th order — the reading transported unchanged. (B) is why the answer cannot be "
            "read off A alone: agreement can be driven to zero by discarding the measurement, "
            "which is what the 'model only' control does, so every mode is also scored on how far "
            "its projection sits from the marker-supported field model. Only the bottom-left "
            "quadrant is an improvement and nothing is in it. (C) The same comparison paired per "
            "trial and joint, so a mode that wins on average while failing badly on some cells is "
            "visible as a tail. (D) The geometric reason: the joint centre sits about one array "
            "span beyond the last sensor, so every segment-scale fit is an extrapolation and the "
            "2nd order diverges fastest out there."))

# ==============================================================================
# Figure: mechanism — a gradient, or the sensors?
# ==============================================================================

def panel_mechanism(ax: plt.Axes, mechanism: pd.DataFrame) -> None:
    """How much of a same-segment disagreement each candidate model explains, alone."""
    if mechanism.empty:
        _no_data(ax, "needs two sensors on one segment\n(IMoVE only)")
        return
    frame = mechanism[mechanism['calibration'] == PRIMARY_CALIBRATION]
    columns = [c for c in MECHANISM_LABELS if c in frame.columns]
    data = [100 * frame[c].dropna().to_numpy() for c in columns]
    parts = ax.violinplot(data, positions=range(len(columns)), widths=0.75, showmedians=True,
                          showextrema=False)
    for body, column in zip(parts['bodies'], columns):
        body.set_facecolor(MECHANISM_COLORS[column])
        body.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([MECHANISM_LABELS[c] for c in columns], rotation=25, ha='right',
                       fontsize=10)
    ax.set_ylabel('Variance of the pair difference explained (%)')
    ax.set_ylim(0, 100)
    ax.set_title('What explains two sensors disagreeing?')
    ax.grid(axis='x', visible=False)
    sns.despine(ax=ax)


def panel_physical(ax: plt.Axes, mechanism: pd.DataFrame) -> None:
    """The unconstrained gradient fit against the physically admissible one.

    THE SHARPEST TEST IN THE EXPERIMENT. A real magnetostatic gradient is symmetric and traceless
    (curl B = 0, div B = 0), so constraining the fit to five parameters instead of nine should
    cost a genuine field gradient almost nothing. Points far below the diagonal are fits whose
    explanatory power came from directions no magnetic field can occupy.
    """
    if mechanism.empty or 'explained_gradient_physical' not in mechanism.columns:
        _no_data(ax, "needs two sensors on one segment\n(IMoVE only)")
        return
    frame = mechanism[mechanism['calibration'] == PRIMARY_CALIBRATION]
    ax.scatter(100 * frame['explained_gradient'], 100 * frame['explained_gradient_physical'],
               s=18, alpha=0.4, color=MECHANISM_COLORS['explained_gradient_physical'],
               edgecolors='none')
    limit = 100
    ax.plot([0, limit], [0, limit], color='#4a4a4a', linewidth=1.4, linestyle='--')
    ax.text(0.97, 0.97, 'a real field gradient\nwould sit on this line', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color='#4a4a4a')
    ax.set_xlabel('Explained by a 9-parameter gradient (%)')
    ax.set_ylabel('Explained by a PHYSICAL gradient (%)\nsymmetric and traceless, 5 parameters')
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_title('Is the fitted gradient even a magnetic field?')
    sns.despine(ax=ax)


def panel_gradient_scale(ax: plt.Axes, gradients: pd.DataFrame) -> None:
    """The three whole-body gradient estimates against the one the markers can see."""
    if gradients.empty:
        _no_data(ax, "no whole-body gradient fits")
        return
    frame = gradients[(gradients['scope'] == 'body')
                      & (gradients['calibration'] == PRIMARY_CALIBRATION)]
    columns = [('gradient_norm_p50', 'per sample\n(9 free)', '#e8973a'),
               ('trial_gradient_norm', 'per trial\n(9 free)', '#c77d2a'),
               ('trial_gradient_norm_physical', 'per trial, PHYSICAL\n(5 free)', '#4c9f70'),
               ('lab_map_gradient_norm', 'the marker-based\nlab field map', '#8a4fa8')]
    present = [(c, label, colour) for c, label, colour in columns if c in frame.columns]
    if frame.empty or not present:
        _no_data(ax, "no whole-body gradient fits")
        return
    data = [frame[c].dropna().to_numpy() for c, _, _ in present]
    parts = ax.violinplot(data, positions=range(len(present)), widths=0.7, showmedians=True,
                          showextrema=False)
    for body, (_, _, colour) in zip(parts['bodies'], present):
        body.set_facecolor(colour)
        body.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([label for _, label, _ in present], fontsize=10)
    ax.set_ylabel(f'Fitted gradient magnitude ({MAG_UNIT}/m)')
    ax.set_yscale('log')
    ax.set_title('How big is the gradient, and who says so?')
    ax.grid(axis='x', visible=False)
    sns.despine(ax=ax)


def panel_constancy(ax: plt.Axes, gradients: pd.DataFrame) -> None:
    """Is the fitted gradient constant in the room, or on the body?"""
    if gradients.empty or 'gradient_constancy_world' not in gradients.columns:
        _no_data(ax, "no whole-body gradient fits")
        return
    frame = gradients[(gradients['scope'] == 'body')
                      & (gradients['calibration'] == PRIMARY_CALIBRATION)]
    if frame.empty:
        _no_data(ax, "no whole-body gradient fits")
        return
    ax.scatter(100 * frame['gradient_constancy_world'], 100 * frame['gradient_constancy_body'],
               s=20, alpha=0.5, color='#2f6fb5', edgecolors='none')
    ax.plot([0, 100], [0, 100], color='#bbbbbb', linewidth=1.0, linestyle=':')
    ax.set_xlabel('Constant in the WORLD frame (%)\na property of the room')
    ax.set_ylabel('Constant in the BODY frame (%)\na property of the subject')
    ax.set_title('What is the per-sample fit tracking?')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax)


def plot_mechanism(dataset: str, mechanism: pd.DataFrame, gradients: pd.DataFrame,
                   save: bool = True, show: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    panel_mechanism(axes[0, 0], mechanism)
    panel_physical(axes[0, 1], mechanism)
    panel_gradient_scale(axes[1, 0], gradients)
    panel_constancy(axes[1, 1], gradients)
    for ax, letter in zip(axes.ravel(), 'ABCD'):
        _label_panel(ax, letter)
    plot_utils.finalize_and_save_plot(
        fig, f"Is the magnetometer disagreement a field gradient, or is it the sensors? — {dataset}",
        f"mechanism_{dataset}.png", PLOTS_DIR, epilog=coverage(mechanism), save=save, show=show,
        caption=(
            "The two candidate causes transform differently, so one difference series separates "
            "them. (A) A three-parameter constant in the SENSOR's own frame — what a hard-iron or "
            "gain error is — explains most of a same-segment disagreement. A nine-parameter "
            "world-frame gradient explains slightly more, which is what three times the freedom "
            "buys and not evidence of a field; restricted to the five parameters a real "
            "magnetostatic gradient has, it explains LESS than the sensor-fixed constant. (B) A real "
            "magnetostatic gradient tensor is symmetric and traceless, so constraining the fit "
            "from nine parameters to five should cost it nothing; points far below the diagonal "
            "are fits whose explanatory power came from directions no magnetic field can occupy. "
            "(C) The per-sample whole-body fit returns a gradient several times larger than the "
            "marker-based lab field map sees over the whole room — it is absorbing sensor error — "
            "while the physically constrained per-trial fit agrees with the markers. (D) A real "
            "room gradient would be constant in the world frame and sit at the bottom right; the "
            "per-sample fit is constant in neither."))

# ==============================================================================
# Figure: covariation — how much do neighbouring magnetometers share?
# ==============================================================================

# Ordered near to far, so every panel reads left-to-right as "further apart".
PAIR_LABELS = {'same_segment': 'same segment\n(7-19 cm)', 'across_joint': 'across a joint\n(25-47 cm)',
               'same_limb': 'same limb\n(not adjacent)', 'contralateral': 'other leg\n(same segment)',
               'distant': 'unrelated'}
PAIR_COLORS = {'same_segment': '#1b3f6b', 'across_joint': '#2f6fb5', 'same_limb': '#7aa6c2',
               'contralateral': '#e8973a', 'distant': '#b4432f'}


def _pair_order(frame: pd.DataFrame) -> List[str]:
    present = set(frame['pair_class'].astype(str))
    return [c for c in PAIR_CLASSES if c in present]


def panel_covary_adjacency(ax: plt.Axes, frame: pd.DataFrame) -> None:
    """Correlation of the world-frame field FLUCTUATION, raw and after debiasing.

    The mean is removed first, so this is the disturbance rather than the ambient field the two
    obviously share. The falloff across the x-axis is the test: a genuinely spatial disturbance is
    shared by neighbours and not by a sensor on the other leg.
    """
    order = _pair_order(frame)
    width = 0.38
    for offset, column, colour, label in ((-width / 2, 'rho', '#4a4a4a', 'as measured'),
                                          (width / 2, 'rho_debiased', '#4c9f70',
                                           'each sensor\'s own bias removed')):
        data, positions = [], []
        for index, pair in enumerate(order):
            values = frame.loc[frame['pair_class'].astype(str) == pair, column].dropna()
            if values.empty:
                continue
            data.append({'label': pair, **_box_stats(values)})
            positions.append(index + offset)
        if not data:
            continue
        artists = ax.bxp(data, positions=positions, widths=width * 0.9, showfliers=False,
                         patch_artist=True, medianprops={'color': 'black', 'linewidth': 1.5},
                         boxprops={'facecolor': colour, 'edgecolor': colour, 'alpha': 0.75},
                         whiskerprops={'color': colour}, capprops={'color': colour})
    ax.axhline(0.0, color='#999999', linewidth=0.8, linestyle=':')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([PAIR_LABELS[p] for p in order], fontsize=9)
    ax.set_ylabel('Correlation of the field fluctuation')
    ax.set_title('Does adjacency buy shared signal?')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.75, label=l)
                       for c, l in (('#4a4a4a', 'as measured'),
                                    ('#4c9f70', "each sensor's own bias removed"))],
              loc='lower left', fontsize=9)
    ax.grid(axis='x', visible=False)
    sns.despine(ax=ax)


def panel_covary_surviving(ax: plt.Axes, frame: pd.DataFrame) -> None:
    """THE DECISION PANEL: what fraction of the disturbance survives differencing the two sensors.

    A relative filter subtracts one segment's magnetometer from the other's, so a disturbance
    common to both cancels and a disturbance private to each adds. 1.0 is the break-even line —
    below it differencing removes disturbance, above it differencing manufactures it.
    """
    order = _pair_order(frame)
    grouped_boxes(ax, frame, 'pair_class', order, 'surviving', colors=PAIR_COLORS,
                  labels=PAIR_LABELS, ylabel='Fraction surviving the difference', rotation=0)
    ax.axhline(1.0, color='#b4432f', linewidth=1.8, linestyle='--')
    ax.text(0.99, 1.0, ' independent — differencing is neutral ',
            transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=9,
            color='#b4432f')
    ax.axhline(0.0, color='#999999', linewidth=0.8, linestyle=':')
    ax.set_title('Does differencing them help?')


def panel_covary_bands(ax: plt.Axes, frame: pd.DataFrame) -> None:
    """The same correlation split by timescale, which separates the two candidate mechanisms.

    A room-driven disturbance changes as the subject walks across the lab, so it is SLOW and
    should be shared by everything on the body at once. A body-fixed sensor bias sweeps the world
    frame at the limb's own rotation rate, so it lives in the GAIT band and is shared only by
    sensors that rotate together — which is to say, only within a segment.
    """
    order = _pair_order(frame)
    width = 0.38
    for offset, column, colour, label in (
            (-width / 2, 'rho_slow', '#8a4fa8', f'slow (< {SLOW_BAND_HZ:g} Hz)'),
            (width / 2, 'rho_gait', '#e8973a', f'gait band (> {SLOW_BAND_HZ:g} Hz)')):
        data, positions = [], []
        for index, pair in enumerate(order):
            values = frame.loc[frame['pair_class'].astype(str) == pair, column].dropna()
            if values.empty:
                continue
            data.append({'label': pair, **_box_stats(values)})
            positions.append(index + offset)
        if not data:
            continue
        ax.bxp(data, positions=positions, widths=width * 0.9, showfliers=False,
               patch_artist=True, medianprops={'color': 'black', 'linewidth': 1.5},
               boxprops={'facecolor': colour, 'edgecolor': colour, 'alpha': 0.75},
               whiskerprops={'color': colour}, capprops={'color': colour})
    ax.axhline(0.0, color='#999999', linewidth=0.8, linestyle=':')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([PAIR_LABELS[p] for p in order], fontsize=9)
    ax.set_ylabel('Correlation of the field fluctuation')
    ax.set_title('At what timescale do they covary?')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.75, label=l)
                       for c, l in (('#8a4fa8', f'slow (< {SLOW_BAND_HZ:g} Hz)'),
                                    ('#e8973a', f'gait band (> {SLOW_BAND_HZ:g} Hz)'))],
              loc='lower left', fontsize=9)
    ax.grid(axis='x', visible=False)
    sns.despine(ax=ax)


def panel_covary_separation(ax: plt.Axes, frame: pd.DataFrame) -> None:
    """The continuous version: correlation against how far apart the two sensors are."""
    for pair, group in frame.groupby('pair_class', observed=True):
        if str(pair) not in PAIR_COLORS:
            continue
        ax.scatter(1000 * group['sep_m'], group['rho'], s=10, alpha=0.25,
                   color=PAIR_COLORS[str(pair)],
                   label=PAIR_LABELS[str(pair)].replace('\n', ' '), edgecolors='none')
    binned = frame.assign(bin=pd.qcut(frame['sep_m'], q=10, duplicates='drop'))
    trend = binned.groupby('bin', observed=True).agg(sep=('sep_m', 'median'),
                                                     rho=('rho', 'median')).dropna()
    ax.plot(1000 * trend['sep'], trend['rho'], color='black', linewidth=2.0, marker='o',
            markersize=5, label='median per separation bin', zorder=5)
    ax.axhline(0.0, color='#999999', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Sensor separation (mm)')
    ax.set_ylabel('Correlation of the field fluctuation')
    ax.set_title('Covariation against distance')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    sns.despine(ax=ax)


def plot_covariation(dataset: str, covariation: pd.DataFrame, save: bool = True,
                     show: bool = False) -> None:
    if covariation.empty:
        print("  covariation: no rows on disk, skipping")
        return
    frame = covariation[covariation['calibration'] == PRIMARY_CALIBRATION]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    panel_covary_adjacency(axes[0, 0], frame)
    panel_covary_surviving(axes[0, 1], frame)
    panel_covary_bands(axes[1, 0], frame)
    panel_covary_separation(axes[1, 1], frame)
    for ax, letter in zip(axes.ravel(), 'ABCD'):
        _label_panel(ax, letter)
    plot_utils.finalize_and_save_plot(
        fig, f"How much do neighbouring magnetometers covary? — {dataset}",
        f"covariation_{dataset}.png", PLOTS_DIR, epilog=coverage(covariation), save=save,
        show=show,
        caption=(
            "Every other figure here measures how far two magnetometers DISAGREE; this one asks "
            "how much of what they see is SHARED, which is what decides whether a relative filter "
            "— which subtracts one segment's reading from the other's — is helped or hurt by the "
            "disturbance. Each sensor's time-mean is removed first, so this is the FLUCTUATION "
            "and not the standing offset the rest of the experiment is about. (A) Correlation "
            "falls off monotonically with adjacency, so part of the disturbance is genuinely "
            "spatial; removing each sensor's own body-fixed bias RAISES the same-segment "
            "correlation, so the bias is what decorrelates close neighbours rather than what "
            "couples them. (B) The fraction surviving the subtraction: below the dashed line "
            "differencing removes disturbance, above it differencing manufactures it. Everything "
            "sits below, so differencing never hurts — but only two sensors on ONE segment sit "
            "far below, and how much a relative filter gains across a joint turns out to depend "
            "on the protocol (Al Borno's roaming walking trials leave 29% after subtraction, "
            "IMoVE's mostly-on-the-spot tasks 72%), because it is the slow room-scale term that "
            "is shared. (C) That mechanism, by timescale — the slow band carries nearly all of "
            "the shared part, while the gait band is shared only within a segment, as two biases "
            "rotating together would be. (D) The continuous version. One point per trial and "
            "pair."))

# ==============================================================================
# Figure: holdout — predict the third sensor
# ==============================================================================

def panel_holdout(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """Prediction error by which sensors predicted which, and at what order."""
    frame = cells(stats, family='segment', target_kind='held_out',
                  calibration=PRIMARY_CALIBRATION)
    if frame.empty:
        _no_data(ax, "needs three sensors on one segment\n(IMoVE only)")
        return
    frame = frame.copy()
    frame['key'] = frame['variant'].astype(str) + ' | ' + frame['mode'].astype(str)
    variants = sorted(set(frame['variant'].astype(str)))
    order = [f'{v} | {m}' for v in variants for m in LINE_MODE_ORDER
             if f'{v} | {m}' in set(frame['key'])]
    grouped_boxes(ax, frame, 'key', order, 'ang_p50',
                  colors={k: MODE_COLORS[k.split(' | ')[1]] for k in order},
                  labels={k: k.replace(' | ', '\n') for k in order},
                  ylabel='Prediction error at the\nheld-out sensor (deg)', rotation=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=MODE_COLORS[m], alpha=0.75,
                                     label=MODE_LABELS[m]) for m in LINE_MODE_ORDER
                       if any(m == k.split(' | ')[1] for k in order)],
              loc='upper left', fontsize=9)
    ax.set_title('Two sensors predicting the third')


def panel_holdout_geometry(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """The same, against whether the prediction interpolated or extrapolated.

    The one clean statement this experiment can make in favour of a spatial model: where the
    target lies BETWEEN the two sources, a line does about as well as their mean. Every time it
    has to reach beyond them, it loses — and the joint centre is always beyond them.
    """
    frame = cells(stats, family='segment', target_kind='held_out',
                  calibration=PRIMARY_CALIBRATION)
    if frame.empty or 'extrapolation' not in frame.columns:
        _no_data(ax, "needs three sensors on one segment\n(IMoVE only)")
        return
    wide = frame.pivot_table(index=['subject', 'trial', 'group', 'target'], columns='mode',
                             values='ang_p50', observed=True)
    # Pivoting on a categorical leaves a CategoricalIndex on the columns, which `.dropna()`
    # and `.join()` both raise on. Plain strings from here.
    wide.columns = [str(column) for column in wide.columns]
    geometry = frame.groupby(['subject', 'trial', 'group', 'target'],
                             observed=True)['extrapolation'].median()
    if not {'mean', 'linear'} <= set(wide.columns):
        _no_data(ax, "needs both the mean and the line")
        return
    joined = wide.join(geometry.rename('extrapolation')).dropna()
    ratio = joined['linear'] / joined['mean']
    ax.scatter(joined['extrapolation'], ratio, s=18, alpha=0.4, color=MODE_COLORS['linear'],
               edgecolors='none')
    ax.axhline(1.0, color='#b4432f', linewidth=1.6, linestyle='--')
    ax.text(0.99, 1.0, ' the line and the mean tie ', transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', fontsize=9, color='#b4432f')
    ax.axvline(0.0, color='#999999', linewidth=1.0, linestyle=':')
    ax.set_yscale('log')
    ax.set_xlabel('Extrapolation: array spans beyond the sources\n(0 = the target is between them)')
    ax.set_ylabel('Error ratio, 1st order / mean')
    ax.set_title('A spatial model only survives inside the array')
    sns.despine(ax=ax)


def plot_holdout(dataset: str, stats: pd.DataFrame, save: bool = True, show: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    panel_holdout(axes[0], stats)
    panel_holdout_geometry(axes[1], stats)
    for ax, letter in zip(axes, 'AB'):
        _label_panel(ax, letter)
    plot_utils.finalize_and_save_plot(
        fig, f"Can two magnetometers on a segment predict the third? — {dataset}",
        f"holdout_{dataset}.png", PLOTS_DIR, epilog=coverage(stats), save=save, show=show,
        caption=(
            "The only fully out-of-sample test in the experiment: unlike the joint centre, where "
            "nothing is mounted, a real measurement sits at the point being predicted. (A) Each "
            "pair of sensors predicts the third, by their mean and by the line through them. (B) "
            "The line beats the mean only where the held-out sensor lies BETWEEN the two sources "
            "— an interpolation. Every extrapolation loses, and the joint centre is always an "
            "extrapolation, about one array span beyond the last sensor."))

# ==============================================================================
# Figure: reference — the marker-supported field models
# ==============================================================================

def panel_segment_deviation(ax: plt.Axes, stats: pd.DataFrame, spec) -> None:
    """Distance from the field model per joint, proximal to distal."""
    frame = cells(stats, family='marker', mode=PRIMARY_MODE, calibration=PRIMARY_CALIBRATION,
                  channel='vector')
    if frame.empty:
        _no_data(ax, "no marker-family comparisons")
        return
    order = order_groups(frame['group'].unique(), spec)
    width = 0.38
    for offset, column, colour, label in ((-width / 2, 'err_p50', '#8a4fa8', 'constant field'),
                                          (width / 2, 'err_alt_p50', '#4c9f70', 'lab field map')):
        if column not in frame.columns:
            continue
        values = [frame.loc[frame['group'].astype(str) == g, column].median() for g in order]
        ax.bar(np.arange(len(order)) + offset, values, width=width * 0.92, color=colour,
               alpha=0.8, label=label)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha='right')
    ax.set_ylabel(f'Distance from the model ({MAG_UNIT})')
    ax.set_title('Against what the markers can support')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='x', visible=False)
    sns.despine(ax=ax)


def panel_lab_map(ax: plt.Axes, lab: pd.DataFrame) -> None:
    """The lab field map's in-sample against leave-one-sensor-out skill, by order.

    The gap between the two bars is how much of the map's apparent skill was fitting a per-sensor
    bias rather than the room — and where the leave-one-out bar FALLS with order, the extra terms
    are fitting sensors, which is what decides the order used as a reference elsewhere.
    """
    if lab.empty:
        _no_data(ax, "no lab field map")
        return
    orders = sorted(lab['order'].unique())
    width = 0.38
    for offset, column, colour, label in ((-width / 2, 'r2', '#7aa6c2', 'in sample'),
                                          (width / 2, 'r2_loso', '#2f6fb5',
                                           'leave one sensor out')):
        values = [100 * lab.loc[lab['order'] == o, column].median() for o in orders]
        ax.bar(np.arange(len(orders)) + offset, values, width=width * 0.92, color=colour,
               alpha=0.85, label=label)
    ax.axhline(0.0, color='#4a4a4a', linewidth=1.0)
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([f'order {int(o)}' for o in orders])
    ax.set_ylabel('Variance of the world-frame field explained (%)')
    ax.set_title('A field map fitted from marker positions')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='x', visible=False)
    sns.despine(ax=ax)


def plot_reference(dataset: str, stats: pd.DataFrame, lab: pd.DataFrame,
                   save: bool = True, show: bool = False) -> None:
    spec = get_dataset(dataset)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    panel_segment_deviation(axes[0], stats, spec)
    panel_lab_map(axes[1], lab)
    for ax, letter in zip(axes, 'AB'):
        _label_panel(ax, letter)
    plot_utils.finalize_and_save_plot(
        fig, f"What the markers can say about a magnetic field — {dataset}",
        f"reference_{dataset}.png", PLOTS_DIR, epilog=coverage(stats), save=save, show=show,
        caption=(
            "Mocap cannot measure a magnetic field, so 'agreement with the markers' can only mean "
            "agreement with a field model the markers support. (A) Distance from a constant global "
            "field and from a polynomial in marker position, per joint, proximal to distal — the "
            "position-aware model helps exactly where the constant one fails worst, which is the "
            "distal segments nearest the lab floor. (B) The map's own skill, in sample and with "
            "the scored sensor left out of the fit. The gap is how much of its apparent skill was "
            "fitting per-sensor error rather than the room; where the leave-one-out bar falls with "
            "order, the extra terms are fitting sensors."))

# ==============================================================================
# Figure: calibration
# ==============================================================================

def panel_calibration_effect(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """Cross-segment agreement under each calibration arm."""
    frame = cells(stats, family='joint', mode=PRIMARY_MODE, channel='vector')
    if frame.empty:
        _no_data(ax, "no cross-segment comparisons")
        return
    order = [c for c in CALIBRATIONS if c in set(frame['calibration'].astype(str))]
    grouped_boxes(ax, frame, 'calibration', order, 'ang_p50', colors=CALIBRATION_COLORS,
                  labels=CALIBRATION_LABELS, ylabel='Cross-segment disagreement (deg)')
    ax.set_title('What a per-sensor calibration buys')


def panel_calibration_transfer(ax: plt.Axes, calibration: pd.DataFrame) -> None:
    """How repeatable a subject's fitted hard iron is between their own trials.

    THE COLUMN THAT DECIDES WHETHER THE CALIBRATION IS USABLE. A hard iron is fixed in the sensor
    frame, so a fit that is a property of the device repeats across a subject's trials and can be
    carried to a new one; a fit that was really the local field does not, and carrying it can only
    hurt. Points above the diagonal scatter by more than their own size.
    """
    if calibration.empty or 'hard_iron_norm' not in calibration.columns:
        _no_data(ax, "no calibration fits")
        return
    usable = calibration.dropna(subset=['hard_iron_norm'])
    columns = ['hard_iron_x', 'hard_iron_y', 'hard_iron_z']
    grouped = usable.groupby(['subject', 'sensor'], observed=True)[columns]
    spread = grouped.std().dropna()
    magnitude = grouped.mean().dropna()
    if spread.empty:
        _no_data(ax, "each subject has one trial, so there is nothing to compare across")
        return
    common = spread.index.intersection(magnitude.index)
    x = np.linalg.norm(magnitude.loc[common].to_numpy(), axis=1)
    y = np.linalg.norm(spread.loc[common].to_numpy(), axis=1)
    ax.scatter(x, y, s=22, alpha=0.55, color='#2f6fb5', edgecolors='none')
    limit = max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.05
    ax.plot([0, limit], [0, limit], color='#b4432f', linewidth=1.4, linestyle='--')
    ax.text(0.97, 0.03, 'below the line: the fit repeats,\nso it is a device property',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color='#4a4a4a')
    ax.set_xlabel(f'Fitted hard iron, mean over the subject\'s trials ({MAG_UNIT})')
    ax.set_ylabel(f'Its scatter between those trials ({MAG_UNIT})')
    ax.set_title('Does the calibration transfer?')
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    sns.despine(ax=ax)


def plot_calibration(dataset: str, stats: pd.DataFrame, calibration: pd.DataFrame,
                     save: bool = True, show: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    panel_calibration_effect(axes[0], stats)
    panel_calibration_transfer(axes[1], calibration)
    for ax, letter in zip(axes, 'AB'):
        _label_panel(ax, letter)
    plot_utils.finalize_and_save_plot(
        fig, f"A per-sensor magnetometer calibration, and whether it transfers — {dataset}",
        f"calibration_{dataset}.png", PLOTS_DIR, epilog=coverage(stats), save=save, show=show,
        caption=(
            "If the sensor-to-sensor disagreement is a per-device term rather than a field, a "
            "calibration should remove it. (A) Cross-segment agreement under each arm; the "
            "in-sample arm is shown beside the leave-one-trial-out one so the optimism of fitting "
            "and scoring on the same trial is a measured quantity rather than an assumption. (B) "
            "The test that decides whether any of this is usable online: a hard iron is fixed in "
            "the sensor frame, so a genuine device property repeats across a subject's trials and "
            "sits below the diagonal. Points above it scatter by more than their own size, which "
            "means the fit was measuring the room the subject happened to be standing in."))

# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='alborno', choices=sorted(DATASETS))
    parser.add_argument('--figures', nargs='+', choices=FIGURES, default=list(FIGURES),
                        metavar='FIGURE', help=f"Any of: {', '.join(FIGURES)}")
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    print(f"Loading tables for {args.dataset}...")
    tables = load_tables(args.dataset, args.figures)
    stats = tables.get('stats', pd.DataFrame())
    if stats.empty:
        print(f"No results under {dataset_dir(args.dataset)}. Run:\n"
              f"  python -m experiments.magnetic_projection --dataset {args.dataset}")
        return 1
    print(f"  {coverage(stats)}")

    save = not args.no_save
    gradients = tables.get('gradients', pd.DataFrame())
    if 'agreement' in args.figures:
        if gradients.empty:
            gradients = load_trial_table(args.dataset, 'gradient_fits')
        plot_agreement(args.dataset, stats, gradients, save=save, show=args.show)
    if 'modes' in args.figures:
        plot_modes(args.dataset, stats, save=save, show=args.show)
    if 'mechanism' in args.figures:
        plot_mechanism(args.dataset, tables.get('mechanism', pd.DataFrame()), gradients,
                       save=save, show=args.show)
    if 'covariation' in args.figures:
        plot_covariation(args.dataset, tables.get('covariation', pd.DataFrame()),
                         save=save, show=args.show)
    if 'holdout' in args.figures:
        plot_holdout(args.dataset, stats, save=save, show=args.show)
    if 'reference' in args.figures:
        plot_reference(args.dataset, stats, tables.get('lab', pd.DataFrame()),
                       save=save, show=args.show)
    if 'calibration' in args.figures:
        plot_calibration(args.dataset, stats, tables.get('calibration', pd.DataFrame()),
                         save=save, show=args.show)

    print(f"\nFigures under {PLOTS_DIR}")
    print(f"Numbers: {statistics_path(args.dataset)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
