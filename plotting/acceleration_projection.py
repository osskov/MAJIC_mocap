"""
Figures for experiments/acceleration_projection.py: does the rigid-body projection produce the
acceleration it claims to, asked three ways.

Reads ONLY the tables that experiment wrote. Nothing here reloads a trial, re-projects or
re-filters, so no number in a figure can disagree with the parquet beside it — the same tier
discipline plotting/joint_dof.py and plotting/build_quality.py follow, and for the same reason: a
figure that recomputes is a second implementation nobody diffs.

    python -m plotting.acceleration_projection --dataset alborno
    python -m plotting.acceleration_projection --dataset imove
    python -m plotting.acceleration_projection --dataset imove --figures agreement segment

THE THREE FAMILIES, which every figure keys on:

    marker    the projected accelerometer at the joint centre against differentiated markers.
              The only external reference, and the only one that can catch an error both segments
              make together — but its truth is a second derivative of marker positions, so above
              ~5 Hz it is mostly differentiation noise.
    joint     the parent and child sensors project to ONE point, so their projected readings are
              one vector in two frames. No differentiated markers, and this identity IS what a
              relative filter's accelerometer residual measures. Blind to a common-mode error.
    segment   two sensors on one rigid segment, projected onto each other. The truth is another
              accelerometer, so it is band-limited by the sensors rather than by the reference —
              the sharpest of the three. IMoVE only; nothing else here carries two sensors per
              segment.

AND THE TWO CHANNELS, which are different quantities and are never drawn on one axis:

    vector    both signals in one common frame, so the full 3-vector residual and its direction
              are defined.
    norm      magnitudes only, | |a_est| - |a_truth| |, which is what survives when there is no
              common frame — and the part of a disagreement no orientation estimate can absorb.

SIX FIGURES, each answering one question the console report answers in prose:

  agreement   The headline. All three families side by side, projected against unprojected, plus
              the per-joint breakdown. This is the figure that answers "how well do they agree".

  marker      The supplement for question 1: the three signals as signals, the by-joint error and
              direction, the cutoff sweep against the reference's own noise floor, error against
              the size of the correction, Bland-Altman, and the power spectra. Panel D is the one
              that makes the rest readable — where the residual approaches the shaded band, mocap
              cannot resolve the projection error at all.

  pair        Cross-segment detail: by joint, the mocap-free magnitude channel, and the
              cross-family consistency check — the pair disagreement predicted from the two marker
              residuals against the measured one.

  segment     Same-segment detail: agreement against sensor separation, the gyro rigidity floor
              underneath it, and the geometry re-solved from the IMUs alone. IMoVE only.

  lever       Every comparison in the experiment against how far the projection had to reach. The
              unprojected residual should GROW with the arm — that is the rigid-body term the
              projection removes — and the projected one staying flat while it does is the claim
              of the whole experiment in one panel.

  gyro        The four gyro-derivative schemes, scored per family. The marker rows compare them in
              the one band where they nearly agree; the agreement families have no reference-noise
              floor forcing the cutoff down, so they are where the table can separate them.

Every figure is pooled over every subject, trial and sample present on disk, except the trace
panel, which is one comparison of one trial and says which in its own footer.
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
from experiments.acceleration_projection import (CUTOFF_SWEEP_HZ, DATASETS, FAMILIES,
                                                 FAMILY_LABELS, GYRO_METHODS, LOWPASS_CUTOFF_HZ,
                                                 PRIMARY_GYRO_METHOD, dataset_dir, get_dataset,
                                                 load_trial_table, same_segment_groups,
                                                 statistics_path)

PLOTS_DIR = paths.plots_dir("acceleration_projection")

FIGURES = ('agreement', 'marker', 'pair', 'segment', 'lever', 'gyro')

# --- Colours ----------------------------------------------------------------------------
# One colour per SIGNAL, fixed here rather than taken from a palette so the same signal is the
# same colour in every panel and across datasets. The truth is grey and heaviest so it reads as
# the thing being compared against rather than as a third competitor.
SIGNALS = {
    'truth': ('Truth', '#4a4a4a', 2.6, '-', 1),
    'proj': ('Projected', '#2f6fb5', 1.9, '-', 3),
    'raw': ('Unprojected', '#d1603d', 1.5, '--', 2),
}
ESTIMATE_COLORS = {'proj': SIGNALS['proj'][1], 'raw': SIGNALS['raw'][1]}
ESTIMATE_LABELS = {'proj': 'Projected', 'raw': 'Unprojected'}

# One colour per family, for the figures that put all three on one axis. Ordered as FAMILIES is:
# external reference, then the two internal ones.
FAMILY_COLORS = {'marker': '#8a4fa8', 'joint': '#2f6fb5', 'segment': '#4c9f70'}
FAMILY_SHORT = {'marker': 'vs markers', 'joint': 'parent vs child', 'segment': 'sensor vs sensor'}

# Hatching, not a third colour, for the magnitude channel: it is the SAME family measured a
# different way, so sharing the family's colour and marking the channel is the honest encoding.
CHANNEL_HATCH = {'vector': '', 'norm': '///'}

GYRO_COLORS = {'backward': '#2f6fb5', 'polyfit': '#e8973a', 'central': '#4c9f70',
               'first_order': '#b4432f'}

# --- Layout -----------------------------------------------------------------------------
WINDOW_S = 4.0    # long enough for ~3 gait cycles, short enough that individual peaks resolve
WHISKER_PCT = (5, 95)
# Log-spaced, because the correction spans two orders of magnitude and linear bins would put
# nine tenths of the samples in the first one.
CORRECTION_BINS = np.geomspace(0.05, 60.0, 13)
SEPARATION_BINS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60])
MAX_HEX_POINTS = 400_000  # per axis; a hexbin cannot show more than this anyway
ECDF_DRAW_POINTS = 2000   # the ECDF is computed on all samples and thinned only for drawing
RNG_SEED = 0

# Anatomical ordering, proximal to distal, because every result in this repository degrades down
# the limb and a figure sorted alphabetically hides that.
LEVEL_ORDER = ('Lumbar', 'Hip', 'Knee', 'Ankle', 'Pelvis', 'Thigh', 'Shank', 'Foot')


def _level_key(group: str) -> Tuple[int, str]:
    """Sort key placing a group at its anatomical level, proximal first, unknowns last."""
    for index, level in enumerate(LEVEL_ORDER):
        if level.lower() in group.lower():
            return index, group
    return len(LEVEL_ORDER), group


def order_groups(groups: Sequence[str], spec=None) -> List[str]:
    """The groups to draw, proximal to distal, restricted to the spec's PRIMARY joints if given.

    The restriction is not cosmetic. IMoVE's joint table has 18 pairs against 6 anatomical joints,
    because each thigh and shank carries three sensors and every placement borders the same two
    joints; drawing all 18 in a third-width panel makes the tick labels overlap into a smear and
    says nothing the 6 do not. Every variant is still MEASURED and still in the parquet — the
    placement comparison is what the `segment` figure is for, and it is the only figure that can
    say anything about placement that the joint labels cannot.
    """
    present = {str(g) for g in groups if str(g) != 'all'}
    if spec is not None and getattr(spec, 'primary_joints', None):
        primary = present & set(spec.primary_joints)
        # Only narrow when it leaves something: a spec whose primary joints are named differently
        # from its group labels (the segment family's groups are SEGMENTS, not joints) must not be
        # silently emptied.
        if primary:
            present = primary
    return sorted(present, key=_level_key)

# ==============================================================================
# Loading
# ==============================================================================

# Columns each figure needs out of agreement_samples, pushed down into the parquet read. The full
# table across every trial is tens of millions of rows and thirty columns; a panel that needs two
# of them should not pay for all thirty.
LABEL_COLUMNS = ['family', 'group', 'variant', 'target_kind', 'scope', 'channel']
ERROR_COLUMNS = ['err_proj', 'err_raw', 'ang_proj', 'ang_raw', 'dnorm_proj', 'absdnorm_proj',
                 'absdnorm_raw', 'corr_norm']
BLAND_ALTMAN_COLUMNS = ['ref_x', 'ref_y', 'ref_z', 'diff_x', 'diff_y', 'diff_z']


def load_tables(dataset: str, figures: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Every table the requested figures need, and nothing else.

    Loaded lazily per figure set because agreement_samples is by far the largest artifact here and
    three of the six figures never touch it.
    """
    wanted = set(figures)
    tables: Dict[str, pd.DataFrame] = {}

    columns = list(LABEL_COLUMNS + ERROR_COLUMNS)
    if 'marker' in wanted:
        columns += BLAND_ALTMAN_COLUMNS
    if wanted & {'agreement', 'marker', 'pair', 'segment'}:
        tables['samples'] = load_trial_table(dataset, 'agreement_samples', columns=columns)
    if wanted & {'agreement', 'pair', 'segment', 'lever'}:
        tables['stats'] = load_trial_table(dataset, 'agreement_stats')
    if 'marker' in wanted:
        tables['traces'] = load_trial_table(dataset, 'traces')
        tables['spectra'] = load_trial_table(dataset, 'spectra')
        tables['sweep'] = load_trial_table(dataset, 'cutoff_sweep')
    if 'pair' in wanted:
        tables.setdefault('sweep', load_trial_table(dataset, 'cutoff_sweep'))
    if 'gyro' in wanted:
        tables['gyro'] = load_trial_table(dataset, 'gyro_method')
    return tables


def coverage(frame: pd.DataFrame) -> str:
    """One line naming what a figure is pooled over, for the epilog."""
    if frame.empty:
        return "no data"
    cells = frame[['subject', 'trial']].drop_duplicates()
    families = [f for f in FAMILIES if f in set(frame.get('family', pd.Series(dtype=object)))]
    return (f"{len(cells)} trials, {cells['subject'].nunique()} subjects"
            + (f", families: {', '.join(families)}" if families else ""))

# ==============================================================================
# Shared panel pieces
# ==============================================================================

def _box_stats(values: pd.Series) -> Dict[str, float]:
    """5-number summary in matplotlib's bxp format, with p5/p95 whiskers (see WHISKER_PCT)."""
    lo, hi = np.percentile(values, WHISKER_PCT)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {'med': median, 'q1': q1, 'q3': q3, 'whislo': lo, 'whishi': hi, 'fliers': []}


def paired_boxes(ax: plt.Axes, samples: pd.DataFrame, group_col: str, order: Sequence[str],
                 value_cols: Sequence[str] = ('err_proj', 'err_raw'), ylabel: str = '',
                 log: bool = False, hatch_by: Optional[str] = None) -> None:
    """Grouped box plot: one pair of boxes per group, projected against unprojected.

    Drawn with ax.bxp from precomputed quantiles rather than seaborn.boxplot on melted data. Two
    reasons: melting several million rows to draw fourteen boxes doubles the memory for nothing,
    and the quantiles here are then exactly the ones the summary parquet reports instead of
    whatever seaborn recomputes after its own filtering.
    """
    if not len(order):
        return
    width = 0.36
    for offset, column in zip((-width / 2, width / 2), value_cols):
        stats, positions, hatches = [], [], []
        for index, group in enumerate(order):
            rows = samples[samples[group_col].astype(str) == group]
            values = rows[column].dropna()
            if values.empty:
                continue
            stats.append({'label': group, **_box_stats(values)})
            positions.append(index + offset)
            channel = str(rows[hatch_by].iloc[0]) if hatch_by else 'vector'
            hatches.append(CHANNEL_HATCH.get(channel, ''))
        if not stats:
            continue
        color = ESTIMATE_COLORS['proj' if column.endswith('proj') else 'raw']
        artists = ax.bxp(stats, positions=positions, widths=width * 0.9, showfliers=False,
                         patch_artist=True, medianprops={'color': 'black', 'linewidth': 1.6},
                         boxprops={'facecolor': color, 'edgecolor': color, 'alpha': 0.75},
                         whiskerprops={'color': color}, capprops={'color': color})
        for patch, hatch in zip(artists['boxes'], hatches):
            if hatch:
                patch.set_hatch(hatch)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')
    ax.grid(axis='x', visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, alpha=0.75, label=ESTIMATE_LABELS[key],
                                     facecolor=ESTIMATE_COLORS[key]) for key in ('proj', 'raw')],
              loc='upper left', fontsize=11)
    ax.text(0.99, 0.02, f'boxes: IQR, whiskers: p{WHISKER_PCT[0]}–p{WHISKER_PCT[1]}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontstyle='italic',
            color='#666666')
    sns.despine(ax=ax)


def ecdf(ax: plt.Axes, series: Dict[str, Tuple[np.ndarray, str, str]], xlabel: str) -> None:
    """Empirical CDFs on one axis, as {label: (values, colour, linestyle)}.

    Computed on every sample and thinned to ECDF_DRAW_POINTS only for drawing, so the curve is
    exact wherever it is read.
    """
    for label, (values, color, style) in series.items():
        values = np.sort(np.asarray(values)[np.isfinite(values)])
        if not len(values):
            continue
        fraction = np.arange(1, len(values) + 1) / len(values)
        step = max(len(values) // ECDF_DRAW_POINTS, 1)
        ax.plot(values[::step], fraction[::step], color=color, linewidth=2.2, linestyle=style,
                label=f'{label} (median {np.median(values):.2f})')
    ax.axhline(0.5, color='#999999', linewidth=0.8, linestyle=':')
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Fraction of samples below')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=10)
    sns.despine(ax=ax)


def _label_panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.09, 1.06, letter, transform=ax.transAxes, fontsize=20, fontweight='bold',
            va='bottom', ha='right')


def _no_data(ax: plt.Axes, message: str) -> None:
    """A panel with nothing to draw says so, rather than rendering as empty axes.

    Which happens legitimately: the segment family exists only on IMoVE, and a partial run may
    have no trial with a long enough mocap window for the marker family."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha='center', va='center', fontsize=12,
            color='#777777', wrap=True)
    ax.set_axis_off()

# ==============================================================================
# Figure: agreement — the headline
# ==============================================================================

def _family_channel_order(samples: pd.DataFrame) -> List[Tuple[str, str]]:
    present = {(str(f), str(c)) for f, c in
               samples[['family', 'channel']].drop_duplicates().to_numpy()}
    return [(f, c) for f in FAMILIES for c in ('vector', 'norm') if (f, c) in present]


def panel_families(ax: plt.Axes, samples: pd.DataFrame) -> None:
    """One pair of boxes per family and channel: projected against unprojected.

    THE ONE PANEL THAT ANSWERS THE HEADLINE QUESTION, and the one place a reader is most likely to
    compare two things that are not comparable — so the magnitude channel's boxes are hatched and
    its axis label says what it is. A vector residual and a magnitude gap are different quantities;
    what IS comparable across every pair here is the reduction from the orange box to the blue one,
    because within a pair both boxes are the same quantity on the same samples.
    """
    order = _family_channel_order(samples)
    if not order:
        _no_data(ax, "no comparisons on disk")
        return
    labeled = samples.assign(
        _cell=samples['family'].astype(str) + '\n' + samples['channel'].astype(str))
    cells = [f'{family}\n{channel}' for family, channel in order]
    paired_boxes(ax, labeled, '_cell', cells, ylabel='|estimate − truth|  (m/s²)', log=True,
                 hatch_by='channel')
    ax.set_xticklabels([f'{FAMILY_SHORT[f]}\n({c})' for f, c in order], rotation=0, ha='center',
                       fontsize=11)
    ax.set_xlabel('')
    # The reduction, printed on each pair: it is the number the caption quotes and the only one
    # that is comparable across the panel.
    for index, (family, channel) in enumerate(order):
        rows = samples[(samples['family'] == family) & (samples['channel'] == channel)]
        proj, raw = rows['err_proj'].median(), rows['err_raw'].median()
        if np.isfinite(proj) and np.isfinite(raw) and raw > 0:
            ax.text(index, ax.get_ylim()[1], f'−{100 * (1 - proj / raw):.0f}%', ha='center',
                    va='top', fontsize=12, fontweight='bold', color='#333333')


def panel_family_ecdf(ax: plt.Axes, samples: pd.DataFrame) -> None:
    """The same comparison as a distribution rather than a summary, vector channel only.

    Worth a second panel because the boxes hide the tail, and the tail is where a projection fails:
    a family whose median is fine and whose p99 is ten times another's is not equally good.
    """
    series = {}
    for family in FAMILIES:
        rows = samples[(samples['family'] == family) & (samples['channel'] == 'vector')]
        if rows.empty:
            continue
        series[f'{FAMILY_SHORT[family]}, projected'] = (rows['err_proj'].to_numpy(),
                                                        FAMILY_COLORS[family], '-')
        series[f'{FAMILY_SHORT[family]}, unprojected'] = (rows['err_raw'].to_numpy(),
                                                          FAMILY_COLORS[family], '--')
    if not series:
        _no_data(ax, "no vector-channel comparisons")
        return
    ecdf(ax, series, '|estimate − truth|  (m/s²)')


def plot_agreement(dataset: str, samples: pd.DataFrame, stats: pd.DataFrame, save: bool = True,
                   show: bool = False) -> None:
    spec = get_dataset(dataset)
    figure, axes = plt.subplots(2, 2, figsize=(17, 12))
    panel_families(axes[0, 0], samples)
    panel_family_ecdf(axes[0, 1], samples)

    for ax, family in ((axes[1, 0], 'marker'), (axes[1, 1], 'joint')):
        rows = samples[(samples['family'] == family) & (samples['channel'] == 'vector')]
        if rows.empty:
            _no_data(ax, f"no '{family}' comparisons on {dataset}")
            continue
        paired_boxes(ax, rows, 'group', order_groups(rows['group'].unique(), spec),
                     ylabel='|estimate − truth|  (m/s²)', log=True)
        ax.set_title(f'{family}: {FAMILY_LABELS[family]}', fontsize=14)

    for ax, letter in zip(axes.ravel(), 'ABCD'):
        _label_panel(ax, letter)

    plot_utils.finalize_and_save_plot(
        figure, f'Acceleration projection: the three agreements — {dataset}',
        f'agreement_{dataset}.png', PLOTS_DIR, epilog=coverage(samples), save=save, show=show,
        caption=(
            "How well the rigid-body projection reproduces the acceleration it claims to, asked "
            "three ways against three different references. A: each family's projected residual "
            "against the UNPROJECTED baseline on the same truth and the same samples; the "
            "percentage is the reduction, and it is the only quantity comparable across pairs "
            "because a vector residual (unhatched) and a magnitude gap (hatched) are different "
            "measurements. B: the same as distributions, so the tail is visible — the boxes hide "
            "it and the tail is where a projection fails. C, D: per joint, proximal to distal. "
            "'vs markers' has an external reference but its truth is a second derivative of "
            "marker positions and is mostly noise above ~5 Hz, so its residual is an upper bound; "
            "'parent vs child' has no differentiated markers and measures exactly what a relative "
            "filter's accelerometer residual measures, but is blind to an error both segments make "
            "together. Neither is quoted alone. Log y throughout."))

# ==============================================================================
# Figure: marker — the supplement for question 1
# ==============================================================================

def select_dynamic_window(traces: pd.DataFrame, window_s: float = WINDOW_S
                          ) -> Tuple[float, float]:
    """Picks the `window_s` window whose truth signal moves the most, as (start, end) seconds.

    Deterministic and stated rather than hand-picked, because the choice of window decides what
    the panel appears to show: a quiet window puts all three traces on top of each other and would
    suggest the projection is unnecessary, while the busiest stretch is where the correction is
    largest and where the projection can most easily be seen to fail. Scored by the rolling
    standard deviation of the truth magnitude summed over the window — the busiest stretch, not the
    highest-amplitude one, so a single heel strike does not win it.

    The experiment already stored only its own most dynamic TRACE_MAX_S, so this is a second, finer
    selection inside that window rather than a search over the trial.
    """
    truth = np.linalg.norm(traces[['truth_x', 'truth_y', 'truth_z']].to_numpy(), axis=1)
    timestamps = traces['timestamp'].to_numpy()
    fs = 1.0 / np.median(np.diff(timestamps))
    width = max(int(round(window_s * fs)), 2)
    activity = pd.Series(truth).rolling(window=max(width // 8, 2)).std()
    score = activity.rolling(window=width).sum().to_numpy()
    if np.all(np.isnan(score)):
        return float(timestamps[0]), float(timestamps[min(width, len(timestamps) - 1)])
    end = int(np.nanargmax(score))
    return float(timestamps[max(end - width, 0)]), float(timestamps[end])


def panel_traces(axes: Sequence[plt.Axes], traces: pd.DataFrame) -> List[Line2D]:
    """The three signals' x/y/z components, one axis of the body frame per panel.

    Qualitative, and the only panel that shows the signals as signals. The unprojected trace is on
    it for the same reason it is in every other panel: without it the reader cannot tell whether
    the projection is doing work or the signals were already close.
    """
    if traces.empty:
        for ax in axes:
            _no_data(ax, "no traces for this comparison")
        return []
    trace = traces.sort_values('timestamp')
    t0, t1 = select_dynamic_window(trace)
    trace = trace[(trace['timestamp'] >= t0) & (trace['timestamp'] <= t1)]
    rel_time = trace['timestamp'].to_numpy() - t0

    handles = {}
    for index, (ax, axis_label) in enumerate(zip(axes, 'xyz')):
        for prefix, (label, color, width, style, z) in SIGNALS.items():
            column = f'{prefix}_{axis_label}'
            if column not in trace:
                continue
            line, = ax.plot(rel_time, trace[column].to_numpy(), color=color, linewidth=width,
                            linestyle=style, zorder=z, label=label, solid_capstyle='round')
            handles[label] = line
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{axis_label} axis', fontsize=14)
        ax.set_xlim(0, rel_time[-1] if len(rel_time) else 1)
        if index == 0:
            ax.set_ylabel('Specific force (m/s²)')
        sns.despine(ax=ax)
    return [handles[label] for label, *_ in SIGNALS.values() if label in handles]


def panel_cutoff_sweep(ax: plt.Axes, sweep: pd.DataFrame, family: str = 'marker') -> None:
    """Residual against low-pass cutoff, with the truth's own in-band noise beneath it.

    This panel exists because the obvious reading of every other panel — "the projected signal is
    X from the truth" — is not what the marker data supports. Its truth is a second derivative of
    marker positions and has a noise floor, and the grey band is how much residual that floor
    accounts for by itself at each cutoff. Where the blue curve approaches the band, mocap cannot
    resolve the projection error and the measured residual is an upper bound.

    The unprojected curve is the control that survives the argument: it is compared to the same
    noisy truth, so whatever the floor is, the GAP between the two curves is not made of it.
    """
    subset = sweep[sweep['family'] == family] if 'family' in sweep else sweep
    if subset.empty:
        _no_data(ax, f"no cutoff sweep for family '{family}'")
        return
    grouped = subset.groupby('cutoff_hz')
    cutoffs = np.array(sorted(subset['cutoff_hz'].unique()))

    ax.fill_between(cutoffs, 0, grouped['ref_excess_rms'].median().reindex(cutoffs),
                    color='#999999', alpha=0.3, linewidth=0,
                    label="Truth's own in-band noise")
    for column, marker in (('median_err_raw', 's'), ('median_err_proj', 'o')):
        estimate = 'raw' if column.endswith('raw') else 'proj'
        quantiles = grouped[column].quantile([0.25, 0.5, 0.75]).unstack().reindex(cutoffs)
        ax.fill_between(cutoffs, quantiles[0.25], quantiles[0.75],
                        color=ESTIMATE_COLORS[estimate], alpha=0.18, linewidth=0)
        ax.plot(cutoffs, quantiles[0.5], color=ESTIMATE_COLORS[estimate], linewidth=2.2,
                marker=marker, markersize=5, label=ESTIMATE_LABELS[estimate])

    ax.axvline(LOWPASS_CUTOFF_HZ, color='#555555', linewidth=1.2, linestyle='--')
    ax.text(LOWPASS_CUTOFF_HZ, 0.97, f' cutoff used: {LOWPASS_CUTOFF_HZ:.0f} Hz',
            transform=ax.get_xaxis_transform(), fontsize=10, va='top', ha='left', color='#555555')
    ax.set_xlabel('Low-pass cutoff (Hz)')
    ax.set_ylabel('Residual |estimate − truth| (m/s²)')
    # Headroom for the legend, which sits top-left and otherwise collides with the panel title.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.32)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.94), fontsize=10)
    sns.despine(ax=ax)


def panel_error_vs_correction(ax: plt.Axes, samples: pd.DataFrame) -> None:
    """Median error against the size of the correction, with IQR ribbons and the identity line.

    The identity line is the panel's whole argument. If the projection were exact then the
    unprojected residual would EQUAL the correction, so the orange curve would lie on the diagonal
    and the blue one on zero. How far the blue curve rises off the floor is how much of what the
    projection added it got wrong.
    """
    if samples.empty:
        _no_data(ax, "no samples")
        return
    binned = samples.assign(bin=pd.cut(samples['corr_norm'], CORRECTION_BINS))
    grouped = binned.groupby('bin', observed=True)
    centers = grouped['corr_norm'].median()

    for column in ('err_raw', 'err_proj'):
        quantiles = grouped[column].quantile([0.25, 0.5, 0.75]).unstack()
        color = ESTIMATE_COLORS['proj' if column.endswith('proj') else 'raw']
        ax.fill_between(centers, quantiles[0.25], quantiles[0.75], color=color, alpha=0.18,
                        linewidth=0)
        ax.plot(centers, quantiles[0.5], color=color, linewidth=2.2, marker='o', markersize=4,
                label=ESTIMATE_LABELS['proj' if column.endswith('proj') else 'raw'])

    limits = (CORRECTION_BINS[0], CORRECTION_BINS[-1])
    ax.plot(limits, limits, color='#555555', linewidth=1.2, linestyle='--',
            label='Exact projection\n(unprojected error = correction)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*limits)
    ax.set_xlabel('Size of the projection correction (m/s²)')
    ax.set_ylabel('|estimate − truth| (m/s²)')
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(ax=ax)


def panel_bland_altman(ax: plt.Axes, samples: pd.DataFrame, seed: int = RNG_SEED) -> None:
    """Bland-Altman of the projected estimate against its truth, all three body axes pooled.

    x is the mean of the pair and y their difference, per axis and per sample. Pooling the axes is
    what makes the bias line meaningful — a per-axis version would mostly report which axis gravity
    was on. The lobes along x are exactly that gravity offset and are expected.

    Subsampled to MAX_HEX_POINTS with a fixed seed: a hexbin resolves nothing beyond that, and the
    bias and limits below are computed from the FULL data, not the subsample.
    """
    needed = BLAND_ALTMAN_COLUMNS
    if samples.empty or not set(needed).issubset(samples.columns):
        _no_data(ax, "per-axis columns not loaded")
        return
    reference = samples[['ref_x', 'ref_y', 'ref_z']].to_numpy()
    difference = samples[['diff_x', 'diff_y', 'diff_z']].to_numpy()
    mean_of_pair = (reference + difference / 2.0).ravel()
    difference = difference.ravel()
    finite = np.isfinite(mean_of_pair) & np.isfinite(difference)
    mean_of_pair, difference = mean_of_pair[finite], difference[finite]
    if not len(difference):
        _no_data(ax, "no finite samples")
        return

    bias = float(np.mean(difference))
    limit = 1.96 * float(np.std(difference))

    if len(difference) > MAX_HEX_POINTS:
        keep = np.random.default_rng(seed).choice(len(difference), MAX_HEX_POINTS, replace=False)
        drawn_mean, drawn_difference = mean_of_pair[keep], difference[keep]
    else:
        drawn_mean, drawn_difference = mean_of_pair, difference

    # Axis limits from percentiles, not from the data range. A few dozen samples in millions reach
    # tens of g at a joint centre — mocap reconstruction glitches rather than motion — and on the
    # full range they flatten the whole distribution into a single line. They are NOT dropped: the
    # bias and limits come from every sample, and the count outside the view is stated on the panel.
    x_limit = np.percentile(np.abs(mean_of_pair), 99.9)
    y_limit = np.percentile(np.abs(difference), 99.9)
    outside = int(np.sum((np.abs(drawn_mean) > x_limit) | (np.abs(drawn_difference) > y_limit)))

    hexes = ax.hexbin(drawn_mean, drawn_difference, gridsize=70, bins='log', cmap='viridis',
                      mincnt=1, linewidths=0, extent=(-x_limit, x_limit, -y_limit, y_limit))
    ax.figure.colorbar(hexes, ax=ax, label='samples (log)', pad=0.02)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    if outside:
        ax.text(0.99, 0.02, f'axes clipped to p99.9; {outside:,} of {len(drawn_difference):,} '
                            f'drawn points outside', transform=ax.transAxes, ha='right',
                va='bottom', fontsize=9, fontstyle='italic', color='#333333',
                bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none', 'pad': 2})

    ax.axhline(bias, color='#d1603d', linewidth=1.6, label=f'bias {bias:+.3f}')
    for sign in (-1, 1):
        ax.axhline(bias + sign * limit, color='#d1603d', linewidth=1.2, linestyle='--',
                   label='95% limits of agreement' if sign == 1 else None)
    ax.set_xlabel('Mean of estimate and truth, per axis (m/s²)')
    ax.set_ylabel('Estimate − truth (m/s²)')
    ax.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', framealpha=0.9,
              edgecolor='none')
    sns.despine(ax=ax)


def panel_spectra(ax: plt.Axes, spectra: pd.DataFrame, family: str = 'marker') -> None:
    """Median Welch PSD across every trial and comparison, one curve per signal, with IQR bands.
    UNFILTERED.

    Answers the standing objection to the method — differentiating a noisy gyro to get alpha should
    inject high-frequency noise into the accelerometer — and shows why the analysis cutoff is where
    it is. For the marker family the two IMU spectra roll off above ~5 Hz while the mocap truth
    flattens onto a differentiation noise floor and stays there to Nyquist. For the agreement
    families the truth is an accelerometer, so its spectrum tracks the estimate's all the way up.

    Median across comparisons rather than mean: PSDs span orders of magnitude and one trial with a
    loose sensor would otherwise set the curve.
    """
    subset = spectra[spectra['family'] == family] if 'family' in spectra else spectra
    if subset.empty:
        _no_data(ax, f"no spectra for family '{family}'")
        return
    # Rounded before grouping: the frequency grid is identical across trials only as long as every
    # trial got the same nperseg, and a trial shorter than WELCH_NPERSEG gets its own grid.
    # Rounding merges the shared bins instead of splitting the curve into two.
    grouped = subset.assign(freq_hz=subset['freq_hz'].round(4)).groupby('freq_hz', observed=True)
    for column, signal in (('psd_truth', 'truth'), ('psd_estimate', 'proj'),
                           ('psd_estimate_raw', 'raw')):
        if column not in subset.columns:
            continue
        label, color, *_ = SIGNALS[signal]
        quantiles = grouped[column].quantile([0.25, 0.5, 0.75]).unstack()
        frequencies = quantiles.index.to_numpy()
        ax.fill_between(frequencies, quantiles[0.25], quantiles[0.75], color=color, alpha=0.18,
                        linewidth=0)
        ax.plot(frequencies, quantiles[0.5], color=color, linewidth=2.0, label=label)

    ax.axvline(LOWPASS_CUTOFF_HZ, color='#555555', linewidth=1.2, linestyle='--')
    ax.text(LOWPASS_CUTOFF_HZ, 0.02, f' analysis cutoff {LOWPASS_CUTOFF_HZ:.0f} Hz',
            transform=ax.get_xaxis_transform(), fontsize=10, va='bottom', ha='left',
            color='#555555')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD per axis ((m/s²)²/Hz)')
    ax.legend(loc='lower left', fontsize=10)
    sns.despine(ax=ax)


def plot_marker(dataset: str, samples: pd.DataFrame, traces: pd.DataFrame, spectra: pd.DataFrame,
                sweep: pd.DataFrame, save: bool = True, show: bool = False) -> None:
    marker = samples[(samples['family'] == 'marker') & (samples['channel'] == 'vector')]
    if marker.empty:
        print(f"  marker: no marker-family comparisons on {dataset}; skipping.")
        return
    marker_traces = traces[traces['family'] == 'marker'] if not traces.empty else traces
    example = ''
    if not marker_traces.empty:
        first = marker_traces.iloc[0]
        marker_traces = marker_traces[
            (marker_traces['subject'] == first['subject'])
            & (marker_traces['trial'] == first['trial'])]
        example = f"trace: {first['subject']}/{first['trial']} {first['group']} {first['variant']}"

    figure = plt.figure(figsize=(19, 21))
    grid = figure.add_gridspec(4, 3, hspace=0.42, wspace=0.28)

    trace_axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
    handles = panel_traces(trace_axes, marker_traces)
    _label_panel(trace_axes[0], 'A')
    if handles:
        trace_axes[1].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.32),
                             ncol=3, fontsize=12)

    order = order_groups(marker['group'].unique(), get_dataset(dataset))
    ax_error = figure.add_subplot(grid[1, 0])
    paired_boxes(ax_error, marker, 'group', order, ylabel='|estimate − truth| (m/s²)', log=True)
    ax_error.set_title('Error magnitude by joint', fontsize=14)
    _label_panel(ax_error, 'B')

    ax_angle = figure.add_subplot(grid[1, 1])
    paired_boxes(ax_angle, marker, 'group', order, value_cols=('ang_proj', 'ang_raw'),
                 ylabel='Angle to truth (deg)', log=True)
    ax_angle.set_title('Error direction by joint', fontsize=14)
    _label_panel(ax_angle, 'C')

    ax_sweep = figure.add_subplot(grid[1, 2])
    panel_cutoff_sweep(ax_sweep, sweep, 'marker')
    ax_sweep.set_title("Residual vs cutoff, against mocap's own noise", fontsize=14)
    _label_panel(ax_sweep, 'D')

    ax_correction = figure.add_subplot(grid[2, :2])
    panel_error_vs_correction(ax_correction, marker)
    ax_correction.set_title('Error vs the size of the correction', fontsize=14)
    _label_panel(ax_correction, 'E')

    ax_spectra = figure.add_subplot(grid[2, 2])
    panel_spectra(ax_spectra, spectra, 'marker')
    ax_spectra.set_title('Power spectra, unfiltered', fontsize=14)
    _label_panel(ax_spectra, 'F')

    ax_bland = figure.add_subplot(grid[3, :])
    panel_bland_altman(ax_bland, marker)
    ax_bland.set_title('Bland-Altman, per axis, pooled', fontsize=14)
    _label_panel(ax_bland, 'G')

    plot_utils.finalize_and_save_plot(
        figure, f'Projected acceleration against markers — {dataset}',
        f'marker_{dataset}.png', PLOTS_DIR,
        epilog=f'{coverage(marker)}{"; " + example if example else ""}', save=save, show=show,
        caption=(
            "Question 1: does the projected accelerometer at the joint centre agree with the "
            "markers? A: the three signals over the busiest window of one trial, chosen "
            "deterministically. B, C: error magnitude and direction per joint, projected against "
            "the unprojected reading it replaces — direction is separate because a filter's "
            "measurement update responds to it and not to length. D IS THE PANEL THAT MAKES THE "
            "REST READABLE: the mocap truth is a second derivative of marker positions, and the "
            "grey band is how much residual its own in-band noise accounts for at each cutoff. "
            "Where the blue curve approaches the band the residual is an upper bound, not a "
            "measurement — which is why the analysis cutoff is 6 Hz and not 15. The gap between "
            "the two curves survives that argument: both are compared to the same noisy truth, so "
            "its noise inflates both and cancels from their difference. E: the dashed diagonal is "
            "an exact projection; how far the blue curve rises off the floor is how much of what "
            "the projection added it got wrong. F: the truth flattens onto a noise floor above "
            "~5 Hz while both IMU signals roll off, so above that line mocap is measuring itself. "
            "G: constant bias against magnitude-dependent error; the lobes along x are gravity "
            "sitting on whichever body axis was up, and are expected."))

# ==============================================================================
# Figure: pair — cross-segment detail
# ==============================================================================

def panel_consistency(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """The cross-segment disagreement predicted from the two marker residuals, against measured.

    The only place the families constrain each other, and it is drawn rather than only tabulated
    because the interesting part is the SPREAD about the identity line, not the median ratio. If
    each segment's marker residual were independent of the other's, points would sit on the
    diagonal; above it means each marker residual is optimistic relative to what the two
    projections actually agree on.

    Both predictions are drawn: from each segment's OWN implied joint centre, and from the SHARED
    midpoint. The second is the test of whether the joint model explains the gap, since the marker
    family excludes the parent/child disagreement by construction and the pair comparison cannot.
    """
    marker = stats[(stats['family'] == 'marker') & (stats['channel'] == 'vector')]
    joint = stats[(stats['family'] == 'joint') & (stats['scope'] == 'mocap')
                  & (stats['channel'] == 'vector')]
    if marker.empty or joint.empty:
        _no_data(ax, "needs both the marker and cross-segment families")
        return

    keys = ['subject', 'trial', 'group']
    measured = joint.set_index(keys)['rms_before']
    drawn = False
    for source, color, label in (('rms_before', '#8a4fa8', "each segment's own centre"),
                                 ('rms_alt', '#4c9f70', 'the shared midpoint')):
        if source not in marker.columns:
            continue
        sides = marker.pivot_table(index=keys, columns='variant', values=source, observed=True)
        if not {'parent', 'child'}.issubset(sides.columns):
            continue
        predicted = np.sqrt(sides['parent'] ** 2 + sides['child'] ** 2)
        paired = pd.concat([predicted.rename('predicted'), measured.rename('measured')],
                           axis=1).dropna()
        if paired.empty:
            continue
        ratio = (paired['measured'] / paired['predicted']).median()
        ax.scatter(paired['predicted'], paired['measured'], s=18, alpha=0.55, color=color,
                   edgecolors='none', label=f'predicted from {label} (×{ratio:.2f})')
        drawn = True
    if not drawn:
        _no_data(ax, "no joint has both families on the same trial")
        return

    limits = ax.get_xlim()[0], max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot(limits, limits, color='#555555', linewidth=1.2, linestyle='--',
            label='independent residuals')
    ax.set_xlabel('Predicted pair disagreement  √(rms$_{parent}$² + rms$_{child}$²)  (m/s²)')
    ax.set_ylabel('Measured pair disagreement (m/s²)')
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(ax=ax)


def panel_alignment(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """The rms residual before and after removing one constant fitted rotation, per family.

    The misalignment floor, made visible. Both signals of a comparison are brought into one frame
    by a rotation that was FITTED, and residual misalignment turns into an apparent error of
    roughly (angle) × |a| — 0.17 m/s² per degree, since |a| is dominated by gravity. No choice of
    frame removes it, so it is measured instead: how far each point sits below the diagonal is how
    much of its residual one constant rotation explains.

    Nothing downstream is rotation-corrected. This is reported so the floor is visible, not
    subtracted.
    """
    vector = stats[stats['channel'] == 'vector']
    if vector.empty or 'rms_after' not in vector:
        _no_data(ax, "no vector-channel comparisons")
        return
    for family in FAMILIES:
        rows = vector[vector['family'] == family].dropna(subset=['rms_before', 'rms_after'])
        if rows.empty:
            continue
        ax.scatter(rows['rms_before'], rows['rms_after'], s=16, alpha=0.5,
                   color=FAMILY_COLORS[family], edgecolors='none',
                   label=f"{FAMILY_SHORT[family]} ({rows['align_angle_deg'].median():.1f}° median)")
    limits = (0, max(vector['rms_before'].quantile(0.99), vector['rms_after'].quantile(0.99)))
    ax.plot(limits, limits, color='#555555', linewidth=1.2, linestyle='--',
            label='no alignment component')
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel('rms residual as measured (m/s²)')
    ax.set_ylabel('rms after removing one constant rotation (m/s²)')
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(ax=ax)


def plot_pair(dataset: str, samples: pd.DataFrame, stats: pd.DataFrame, sweep: pd.DataFrame,
              save: bool = True, show: bool = False) -> None:
    joint = samples[samples['family'] == 'joint']
    if joint.empty:
        print(f"  pair: no cross-segment comparisons on {dataset}; skipping.")
        return
    spec = get_dataset(dataset)
    vector = joint[joint['channel'] == 'vector']
    norm = joint[joint['channel'] == 'norm']

    figure, axes = plt.subplots(2, 2, figsize=(17, 12))

    if not vector.empty:
        paired_boxes(axes[0, 0], vector, 'group', order_groups(vector['group'].unique(), spec),
                     ylabel='|a$_{parent}$ − a$_{child}$| (m/s²)', log=True)
        axes[0, 0].set_title('Vector disagreement by joint', fontsize=14)
    else:
        _no_data(axes[0, 0], "no vector channel")

    if not norm.empty:
        paired_boxes(axes[0, 1], norm, 'group', order_groups(norm['group'].unique(), spec),
                     ylabel='| |a$_{parent}$| − |a$_{child}$| | (m/s²)', log=True)
        axes[0, 1].set_title('Magnitude disagreement — no mocap orientation used', fontsize=14)
    else:
        _no_data(axes[0, 1], "no magnitude channel")

    panel_consistency(axes[1, 0], stats)
    axes[1, 0].set_title('Does the marker family predict this one?', fontsize=14)
    panel_alignment(axes[1, 1], stats)
    axes[1, 1].set_title('The misalignment floor', fontsize=14)

    for ax, letter in zip(axes.ravel(), 'ABCD'):
        _label_panel(ax, letter)

    plot_utils.finalize_and_save_plot(
        figure, f'Do the two segments agree with each other? — {dataset}',
        f'pair_{dataset}.png', PLOTS_DIR, epilog=coverage(joint), save=save, show=show,
        caption=(
            "Question 2: the parent and child sensors project to ONE physical point, so their "
            "projected readings are one vector expressed in two frames — and that identity is "
            "exactly what a relative filter's accelerometer residual measures. A: the full vector "
            "disagreement per joint, against the unprojected baseline. B: the magnitude channel, "
            "which uses the two offsets and NO mocap orientation at all, so it is the one result "
            "here a deployed system could reproduce; it is also the part of the disagreement no "
            "orientation estimate can ever absorb. C: the cross-family check — the pair "
            "disagreement predicted from the two marker residuals against the measured one. Points "
            "above the diagonal mean each marker residual is optimistic relative to what the two "
            "projections agree on; the green series tests whether the joint model explains that, "
            "since the marker family compares each segment against its OWN implied centre and so "
            "excludes the parent/child disagreement the pair comparison must carry. D: how much of "
            "each residual one constant fitted rotation explains. This family is BLIND to an error "
            "both segments make together, which is what the marker family covers, and why neither "
            "is quoted alone."))

# ==============================================================================
# Figure: segment — same-segment detail
# ==============================================================================

def panel_vs_separation(ax: plt.Axes, stats: pd.DataFrame, family: str = 'segment') -> None:
    """rms agreement against the distance between the two sensors, projected and unprojected.

    The clean statement of what the projection buys, on the family whose truth is another
    accelerometer: the unprojected residual should grow roughly linearly with the separation —
    that IS the rigid-body term — and the projected one staying flat while it does is the claim.
    """
    rows = stats[(stats['family'] == family) & (stats['channel'] == 'vector')]
    rows = rows[np.isfinite(rows['sep_m']) & (rows['sep_m'] > 0)]
    if rows.empty:
        _no_data(ax, f"no '{family}' comparisons with a measured separation")
        return
    binned = rows.assign(bin=pd.cut(rows['sep_m'], SEPARATION_BINS))
    grouped = binned.groupby('bin', observed=True)
    centers = 100 * grouped['sep_m'].median()
    for column, key in (('rms_raw', 'raw'), ('rms_before', 'proj')):
        quantiles = grouped[column].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(centers, quantiles[0.25], quantiles[0.75], color=ESTIMATE_COLORS[key],
                        alpha=0.18, linewidth=0)
        ax.plot(centers, quantiles[0.5], color=ESTIMATE_COLORS[key], linewidth=2.2, marker='o',
                markersize=5, label=ESTIMATE_LABELS[key])
    ax.set_xlabel('Separation between the two sensors (cm)')
    ax.set_ylabel('rms |estimate − truth| (m/s²)')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(ax=ax)


def panel_rigidity(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """The gyro disagreement between two sensors on one segment, against their acceleration
    disagreement.

    THE FLOOR PANEL. Two sensors on one rigid body measure the SAME angular velocity, so whatever
    |ω_A − ω_B| reads is a bound the acceleration agreement cannot beat: a dω disagreement puts
    ~|dω||ω||r| into the centripetal term alone. It costs one subtraction and it is the only thing
    here that separates "the projection is wrong" from "the segment is not the rigid body the
    projection assumes".
    """
    rows = stats[(stats['family'] == 'segment') & (stats['channel'] == 'vector')]
    rows = rows.dropna(subset=['median_gyro_mismatch']) if 'median_gyro_mismatch' in rows else rows
    if rows.empty:
        _no_data(ax, "no gyro mismatch recorded")
        return
    partner = rows[rows['target_kind'] == 'partner']
    other = rows[rows['target_kind'] != 'partner']
    for subset, color, label in ((partner, '#4c9f70', "target: the partner's own site"),
                                 (other, '#2f6fb5', 'target: a shared joint centre')):
        if subset.empty:
            continue
        ax.scatter(subset['median_gyro_mismatch'], subset['rms_before'], s=20, alpha=0.55,
                   color=color, edgecolors='none', label=label)
    ax.set_xlabel('Median |ω$_A$ − ω$_B$| in one frame (rad/s)')
    ax.set_ylabel('rms acceleration disagreement (m/s²)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(ax=ax)


def panel_inertial_geometry(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """The residual at the mocap separation against the residual at the inertially-preferred one.

    Equation (3) is linear in the separation, so it can be re-solved from the two IMUs by ordinary
    least squares. Points well below the diagonal mean the mocap geometry — two fitted sensor
    offsets, which on IMoVE scatter 8.5-21.1 mm between trials for the taped placements — is a
    substantial part of what the projection was being blamed for. Points near it mean the geometry
    was not the limiting term and the residual is the projection physics plus sensor noise.
    """
    rows = stats[(stats['family'] == 'segment') & (stats['target_kind'] == 'partner')]
    rows = (rows.dropna(subset=['rms_at_inertial', 'rms_before'])
            if 'rms_at_inertial' in rows else pd.DataFrame())
    if rows.empty:
        _no_data(ax, "no inertial geometry fit recorded")
        return
    scatter = ax.scatter(rows['rms_before'], rows['rms_at_inertial'],
                         c=rows['r_inertial_error_mm'], s=22, alpha=0.75, cmap='viridis',
                         edgecolors='none')
    ax.figure.colorbar(scatter, ax=ax, label='|r$_{inertial}$ − r$_{mocap}$| (mm)', pad=0.02)
    limits = (0, float(rows[['rms_before', 'rms_at_inertial']].to_numpy().max()) * 1.05)
    ax.plot(limits, limits, color='#555555', linewidth=1.2, linestyle='--',
            label='geometry is not the limit')
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel('rms at the MOCAP separation (m/s²)')
    ax.set_ylabel('rms at the INERTIALLY-FITTED separation (m/s²)')
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(ax=ax)


def plot_segment(dataset: str, samples: pd.DataFrame, stats: pd.DataFrame, save: bool = True,
                 show: bool = False) -> None:
    segment = samples[samples['family'] == 'segment']
    if segment.empty:
        spec = get_dataset(dataset)
        if not same_segment_groups(spec):
            print(f"  segment: {dataset} carries one sensor per segment, so there is no second "
                  f"accelerometer on the same rigid body to project onto; skipping.")
        else:
            print(f"  segment: no same-segment comparisons on disk for {dataset}; skipping.")
        return

    figure, axes = plt.subplots(2, 2, figsize=(17, 12))
    # Ordered by how far apart the two sensors actually are, not alphabetically: the panel's point
    # is that a wider pair demands a larger correction, and a reader should be able to see the
    # spacing increase left to right instead of having to look it up.
    separations = (stats[stats['family'] == 'segment']
                   .groupby('variant', observed=True)['sep_m'].median())
    variants = sorted({str(v) for v in segment['variant'].unique()},
                      key=lambda v: separations.get(v, np.inf))
    paired_boxes(axes[0, 0], segment[segment['channel'] == 'vector'], 'variant', variants,
                 ylabel='|estimate − truth| (m/s²)', log=True)
    axes[0, 0].set_title('By placement pair (proximal-to-distal spacing)', fontsize=14)
    panel_vs_separation(axes[0, 1], stats)
    axes[0, 1].set_title('Agreement vs sensor separation', fontsize=14)
    panel_rigidity(axes[1, 0], stats)
    axes[1, 0].set_title('The rigidity floor', fontsize=14)
    panel_inertial_geometry(axes[1, 1], stats)
    axes[1, 1].set_title('Is the mocap geometry the limit?', fontsize=14)

    for ax, letter in zip(axes.ravel(), 'ABCD'):
        _label_panel(ax, letter)

    plot_utils.finalize_and_save_plot(
        figure, f'Do two sensors on one segment project to each other? — {dataset}',
        f'segment_{dataset}.png', PLOTS_DIR, epilog=coverage(segment), save=save, show=show,
        caption=(
            "Question 3, and the sharpest test of the projection physics available here: the truth "
            "is another ACCELEROMETER on the same rigid body, so there is no differentiated marker "
            "trajectory anywhere in it and the comparison is band-limited by the sensors rather "
            "than by the reference. A: by placement pair; the widest pair spans the whole segment "
            "and so demands the largest correction. B: the unprojected residual grows with the "
            "separation — that IS the rigid-body term — and the projected one staying flat while "
            "it does is the claim. C: two sensors on one rigid body measure the same angular "
            "velocity, so |ω_A − ω_B| is a floor the acceleration agreement cannot beat; it is the "
            "only column that separates a wrong projection from a segment that is not the rigid "
            "body the projection assumes. D: the separation re-solved from the two IMUs alone — "
            "equation (3) is linear in it, so this is a 3-parameter least squares. Points below "
            "the diagonal mean the mocap geometry was a substantial part of the residual; the "
            "colour is how far the two answers are apart. Note that on IMoVE all placements on a "
            "segment come from ONE marker cluster, so the mocap separation is exactly the "
            "difference of two fitted sensor offsets and is only as good as those fits."))

# ==============================================================================
# Figure: lever — the unifying axis
# ==============================================================================

def plot_lever(dataset: str, stats: pd.DataFrame, save: bool = True, show: bool = False) -> None:
    vector = stats[stats['channel'] == 'vector']
    vector = vector[np.isfinite(vector['sep_m']) & (vector['sep_m'] > 0)]
    if vector.empty:
        print(f"  lever: no comparisons with a measured separation on {dataset}; skipping.")
        return

    figure, axes = plt.subplots(1, 2, figsize=(17, 7))

    for family in FAMILIES:
        rows = vector[vector['family'] == family]
        if rows.empty:
            continue
        axes[0].scatter(100 * rows['sep_m'], rows['rms_raw'], s=14, alpha=0.35,
                        color=FAMILY_COLORS[family], edgecolors='none', marker='s')
        axes[0].scatter(100 * rows['sep_m'], rows['rms_before'], s=14, alpha=0.6,
                        color=FAMILY_COLORS[family], edgecolors='none',
                        label=FAMILY_SHORT[family])
    axes[0].set_xlabel('Distance the projection reached (cm)')
    axes[0].set_ylabel('rms |estimate − truth| (m/s²)')
    axes[0].set_yscale('log')
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], marker='o', linestyle='', color='#666666', label='projected'),
                Line2D([], [], marker='s', linestyle='', color='#666666', label='unprojected')]
    axes[0].legend(handles=handles, loc='upper left', fontsize=10, ncol=2)
    sns.despine(ax=axes[0])

    # The reduction, which is the quantity that should be flat in the arm if the projection is
    # doing its job at every length rather than only where the correction is small.
    binned = vector.assign(bin=pd.cut(vector['sep_m'], SEPARATION_BINS))
    for family in FAMILIES:
        rows = binned[binned['family'] == family]
        if rows.empty:
            continue
        grouped = rows.groupby('bin', observed=True)
        reduction = 100 * (1 - grouped['rms_before'].median() / grouped['rms_raw'].median())
        axes[1].plot(100 * grouped['sep_m'].median(), reduction, color=FAMILY_COLORS[family],
                     linewidth=2.2, marker='o', markersize=5, label=FAMILY_SHORT[family])
    axes[1].axhline(0, color='#999999', linewidth=0.8, linestyle=':')
    axes[1].set_xlabel('Distance the projection reached (cm)')
    axes[1].set_ylabel('Reduction against the unprojected baseline (%)')
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='lower right', fontsize=10)
    sns.despine(ax=axes[1])

    for ax, letter in zip(axes, 'AB'):
        _label_panel(ax, letter)

    plot_utils.finalize_and_save_plot(
        figure, f'The lever arm: every comparison, by how far it reached — {dataset}',
        f'lever_{dataset}.png', PLOTS_DIR, epilog=coverage(vector), save=save, show=show,
        caption=(
            "The one axis all three families share: the distance between the two things being "
            "compared — the offset for the marker family, the two sensors' separation for the "
            "other two. A: squares are the unprojected residual, circles the projected one. The "
            "squares SHOULD climb with the arm, because that gap is the rigid-body term the "
            "projection exists to remove; the circles staying flat while they do is the claim of "
            "the whole experiment in one panel. B: the same as a percentage reduction, which is "
            "what stays comparable across families and lengths. A reduction that falls off at "
            "long arms would mean the projection works only where the correction is small — the "
            "case that would matter most for the pipeline, since its own geometry uses the long "
            "arms."))

# ==============================================================================
# Figure: gyro — the derivative schemes
# ==============================================================================

def plot_gyro(dataset: str, gyro: pd.DataFrame, save: bool = True, show: bool = False) -> None:
    if gyro.empty:
        print(f"  gyro: no gyro_method table on {dataset}; skipping.")
        return
    cells = [(family, channel) for family in FAMILIES for channel in ('vector', 'norm')
             if not gyro[(gyro['family'] == family) & (gyro['channel'] == channel)].empty]
    if not cells:
        print("  gyro: no populated family/channel cells; skipping.")
        return

    figure, axes = plt.subplots(1, len(cells), figsize=(6.0 * len(cells), 6.5), squeeze=False)
    present = [m for m in GYRO_METHODS if m in set(gyro['gyro_method'].astype(str))]
    for ax, (family, channel) in zip(axes[0], cells):
        subset = gyro[(gyro['family'] == family) & (gyro['channel'] == channel)]
        stats, positions, colors = [], [], []
        for index, method in enumerate(present):
            values = subset.loc[subset['gyro_method'].astype(str) == method,
                                'median_err'].dropna()
            if values.empty:
                continue
            stats.append({'label': method, **_box_stats(values)})
            positions.append(index)
            colors.append(GYRO_COLORS[method])
        if not stats:
            _no_data(ax, f"no methods for {family}/{channel}")
            continue
        artists = ax.bxp(stats, positions=positions, widths=0.6, showfliers=False,
                         patch_artist=True, medianprops={'color': 'black', 'linewidth': 1.6})
        for patch, color in zip(artists['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=30, ha='right')
        ax.set_yscale('log')
        ax.set_title(f'{FAMILY_SHORT[family]}\n({channel} channel)', fontsize=13)
        if ax is axes[0][0]:
            ax.set_ylabel('Median |estimate − truth| per comparison (m/s²)')
        ax.grid(axis='x', visible=False)
        sns.despine(ax=ax)

    for ax, letter in zip(axes[0], 'ABCDEF'):
        _label_panel(ax, letter)

    plot_utils.finalize_and_save_plot(
        figure, f'Is the gyro derivative the limit? — {dataset}', f'gyro_{dataset}.png',
        PLOTS_DIR, epilog=f"{coverage(gyro)}; default: {PRIMARY_GYRO_METHOD}", save=save,
        show=show,
        caption=(
            "α = dω/dt is not measured by any sensor, so the projection's tangential term has to "
            "differentiate the gyro. Same offset, same truth, same low-pass, same samples — only "
            "the derivative differs. THE MARKER PANELS COMPARE THE SCHEMES IN THE ONE BAND WHERE "
            "THEY NEARLY AGREE: they differ almost entirely above the 6 Hz cutoff, which the mocap "
            "reference's own noise floor forced down. The agreement families have no such "
            "constraint — their truth is an accelerometer with the same bandwidth as the estimate "
            "— so they are where this figure can separate them, and where a scheme that wins on "
            "mocap and loses here was winning on the reference's noise. None of it licenses a "
            "choice for the pipeline: 'polyfit' reads 90 ms of FUTURE gyro against 'central''s one "
            f"sample, so it is not solving the same problem, and '{PRIMARY_GYRO_METHOD}' stays the "
            "default because it is the only causal one."))

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
    samples = tables.get('samples', pd.DataFrame())
    stats = tables.get('stats', pd.DataFrame())
    if samples.empty and stats.empty:
        print(f"No results under {dataset_dir(args.dataset)}. Run:\n"
              f"  python -m experiments.acceleration_projection --dataset {args.dataset}")
        return 1
    print(f"  {coverage(samples if not samples.empty else stats)}")

    save = not args.no_save
    if 'agreement' in args.figures:
        plot_agreement(args.dataset, samples, stats, save=save, show=args.show)
    if 'marker' in args.figures:
        plot_marker(args.dataset, samples, tables.get('traces', pd.DataFrame()),
                    tables.get('spectra', pd.DataFrame()), tables.get('sweep', pd.DataFrame()),
                    save=save, show=args.show)
    if 'pair' in args.figures:
        plot_pair(args.dataset, samples, stats, tables.get('sweep', pd.DataFrame()),
                  save=save, show=args.show)
    if 'segment' in args.figures:
        plot_segment(args.dataset, samples, stats, save=save, show=args.show)
    if 'lever' in args.figures:
        plot_lever(args.dataset, stats, save=save, show=args.show)
    if 'gyro' in args.figures:
        plot_gyro(args.dataset, tables.get('gyro', pd.DataFrame()), save=save, show=args.show)

    print(f"\nFigures under {PLOTS_DIR}")
    print(f"Numbers: {statistics_path(args.dataset)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
