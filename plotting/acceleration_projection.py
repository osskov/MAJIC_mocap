"""
Supplementary figure for experiments/acceleration_projection.py: how well the rigid-body
projection reproduces the true acceleration at a joint center.

Everything is read back from that experiment's per-trial tables — nothing here reloads raw
data, re-projects, or re-filters, so no number in the figure can disagree with the number in
results/statistics/acceleration_projection_statistics.parquet.

Three signals throughout, always in the same sensor body frame and always in m/s^2:

    reference   mocap truth at the joint center (positions differentiated twice + gravity)
    projected   a_sensor + alpha x r + omega x (omega x r), what the filter consumes
    sensor      the raw unprojected reading, i.e. what the projection replaces

Panels, and what each one is for
-------------------------------
A  Time series, x/y/z, over the most dynamic window of one trial. Qualitative, and the only
   panel that shows the three signals as signals. The unprojected trace is on it for the
   same reason it is in every other panel: without it the reader cannot tell whether the
   projection is doing work or the signals were already close.

B  Error magnitude by joint, projected vs unprojected. The quantitative core.

C  Error direction by joint — the angle between estimated and true acceleration vectors.
   Separate from B because the EKF's measurement update responds to the direction of the acc
   vector, not its length, so a method can be good at one and bad at the other.

D  Residual against the low-pass cutoff, with the mocap reference's OWN in-band noise shaded
   underneath. The most important panel for reading the others honestly: the reference is a
   second derivative of marker positions and has a noise floor, so above ~5 Hz the residual
   stops being a measurement of the projection and becomes a measurement of mocap. Where the
   projected curve approaches the shaded band, the residual is an upper bound rather than an
   error. The gap between the two curves is what survives the argument — both are compared to
   the same reference, so its noise inflates both and cancels from their difference.
   (The pooled ECDF this replaced is still written as a standalone file.)

E  Error against the SIZE of the correction. The identity line on this panel is not
   decoration: if the projection were exact then |a_sensor - a_true| would equal
   |a_proj - a_sensor| exactly, so the unprojected curve should track the diagonal and the
   projected curve should stay flat and low. How far the projected curve rises off the floor
   is how much of the correction the projection gets wrong.

F  Bland-Altman, per axis, pooled. Separates constant bias from magnitude-dependent error,
   which the error-magnitude panels cannot. The three clusters offset along x are gravity
   sitting on whichever body axis happens to be up — expected, not an artifact.

G  Power spectra, UNFILTERED. Answers the standing objection to the method — differentiating a
   noisy gyro to get alpha should inject high-frequency noise — and shows why the analysis
   cutoff is where it is: the two IMU spectra roll off above ~5 Hz while the mocap reference
   flattens onto a noise floor and stays there to Nyquist. The analysis cutoff is drawn on it,
   since every other panel lives below that line.

Panels B-G pool every subject, activity and sample present on disk. A is one trial, chosen
by --subject/--activity, and its window is picked deterministically (see
select_dynamic_window) rather than by hand.

Each panel is also written as its own file, so a panel can be dropped into the manuscript at
full size without re-plotting; --composed-only skips those.
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
from plotting.sensor_distributions import joint_label
from experiments.experiment_utils import JOINTS
from experiments.acceleration_projection import (EXAMPLE_JOINT, LOWPASS_CUTOFF_HZ, SAMPLE_STRIDE,
                                                load_trial_table)

PLOTS_DIR = paths.plots_dir("acceleration_projection")

EXAMPLE_SUBJECT = 'Subject01'
EXAMPLE_ACTIVITY = 'walking'
WINDOW_S = 4.0  # long enough for ~3 gait cycles, short enough that individual peaks resolve

# Signal -> (display label, color, linewidth, linestyle, z-order). The reference is drawn
# thick and pale underneath with the two estimates over it, so that "does the estimate follow
# the truth" is a question about whether a thin line stays inside a thick band — much easier
# to judge than three lines of equal weight crossing each other.
SIGNALS = {
    'ref': ('Mocap truth at joint center', '#b3b3b3', 4.0, '-', 1),
    'proj': ('IMU, projected', '#1f6f8b', 1.6, '-', 3),
    'sens': ('IMU, unprojected', '#d1603d', 1.4, '--', 2),
}
# Estimate colors and labels reused by every non-time-series panel, so a color means the same
# thing in all seven. Keyed by the column's SUFFIX rather than its full name: the same two
# estimates are compared under several metrics (err_proj/err_raw, ang_proj/ang_raw), and
# keying by full column name meant every new metric had to re-register its colors.
ESTIMATE_COLORS = {'proj': SIGNALS['proj'][1], 'raw': SIGNALS['sens'][1]}
ESTIMATE_LABELS = {'proj': 'Projected', 'raw': 'Unprojected'}


def estimate_of(column: str) -> str:
    """'err_proj' / 'ang_proj' -> 'proj'; 'err_raw' / 'ang_raw' -> 'raw'."""
    return column.rsplit('_', 1)[-1]


def color_of(column: str) -> str:
    return ESTIMATE_COLORS[estimate_of(column)]


def label_of(column: str) -> str:
    return ESTIMATE_LABELS[estimate_of(column)]

# Joints grouped by anatomical level, then side, rather than in JOINTS' order (which walks
# down the right leg and then down the left). Grouping by level is what makes the panels
# readable as "error grows distally", and puts each joint next to its own mirror image so a
# left/right asymmetry is visible as a sanity check.
LEVEL_ORDER = ['Lumbar', 'Hip', 'Knee', 'Ankle']
JOINT_ORDER = [joint_label(j) for level in LEVEL_ORDER for j in JOINTS
               if joint_label(j).split()[0] == level]
JOINT_LABELS = {j: joint_label(j) for j in JOINTS}

# Whisker percentiles for the by-joint boxes. Not 1.5x IQR: every metric here is a
# non-negative magnitude with an impulsive footfall tail, so Tukey whiskers would sit far
# inside a huge flier cloud and the box would claim a range the data does not have. p5/p95 is
# stated on the axis instead of implied.
WHISKER_PCT = (5, 95)

# Correction-magnitude bins for panel E. Edges are geometric rather than uniform because
# corr_norm spans three orders of magnitude — uniform bins would put nearly every sample in
# the first one.
CORRECTION_BINS = np.geomspace(0.05, 60.0, 13)

MAX_HEX_POINTS = 400_000  # per axis, for panel F; a hexbin cannot show more than this anyway
ECDF_DRAW_POINTS = 2000   # the ECDF is computed on all samples and thinned only for drawing
RNG_SEED = 0

# ==============================================================================
# Loading
# ==============================================================================

def load_samples(columns: Sequence[str]) -> pd.DataFrame:
    """Pooled per-sample table, restricted to the columns a panel needs.

    The column list is not a micro-optimization: joint_samples across every trial is millions
    of rows and twenty columns, and a two-column panel that loaded all of it would dominate
    this script's memory."""
    return load_trial_table('joint_samples', columns=list(columns))


def describe_coverage(samples: pd.DataFrame) -> str:
    """One line naming what the pooled panels are actually pooling, for the figure footer.
    A supplementary figure that does not say how much data is behind it is not checkable."""
    trials = samples[['subject', 'activity']].drop_duplicates()
    n_subjects = trials['subject'].nunique()
    return (f"{len(samples):,} samples (every {SAMPLE_STRIDE}th) from {len(trials)} trials, "
            f"{n_subjects} subjects, {samples['joint'].nunique()} joints x 2 segments; "
            f"all signals {LOWPASS_CUTOFF_HZ:.0f} Hz zero-lag low-pass filtered")

# ==============================================================================
# Panel A: time series
# ==============================================================================

def select_dynamic_window(traces: pd.DataFrame, window_s: float = WINDOW_S
                          ) -> Tuple[float, float]:
    """Picks the `window_s` window whose reference signal moves the most, as
    (start_time, end_time).

    Deterministic and stated rather than hand-picked, because the choice of window decides
    what the panel appears to show: a quiet window puts all three traces on top of each other
    and would suggest the projection is unnecessary, while the most dynamic window is where
    the correction is largest and where the projection can most easily be seen to fail. The
    window is scored by the rolling standard deviation of the reference magnitude, summed
    over the window — i.e. the busiest stretch, not the highest-amplitude one, so a single
    heel-strike spike does not win it.
    """
    reference = np.linalg.norm(traces[['ref_x', 'ref_y', 'ref_z']].to_numpy(), axis=1)
    timestamps = traces['timestamp'].to_numpy()
    fs = 1.0 / np.median(np.diff(timestamps))
    width = max(int(round(window_s * fs)), 2)

    activity = pd.Series(reference).rolling(window=max(width // 8, 2)).std()
    score = activity.rolling(window=width).sum().to_numpy()
    if np.all(np.isnan(score)):
        return float(timestamps[0]), float(timestamps[min(width, len(timestamps) - 1)])
    end = int(np.nanargmax(score))
    return float(timestamps[max(end - width, 0)]), float(timestamps[end])


def panel_traces(axes: Sequence[plt.Axes], traces: pd.DataFrame, role: str = 'parent',
                 window: Optional[Tuple[float, float]] = None) -> List[Line2D]:
    """Draws the three signals' x/y/z components into three axes, one per axis of the body
    frame. Returns the legend handles so a composed figure can place the legend itself.

    One role (default the parent segment) rather than both: the two segments of a joint are
    two independent projections onto the same point, so drawing both doubles the panel to
    make the same qualitative point twice."""
    trace = traces[traces['role'] == role].sort_values('timestamp')
    if trace.empty:
        return []
    if window is None:
        window = select_dynamic_window(trace)
    t0, t1 = window
    trace = trace[(trace['timestamp'] >= t0) & (trace['timestamp'] <= t1)]
    rel_time = trace['timestamp'].to_numpy() - t0

    handles = {}
    for axis_index, (ax, axis_label) in enumerate(zip(axes, 'xyz')):
        # Reference first so it lies under the estimates regardless of z-order support.
        for prefix, (label, color, width, style, z) in SIGNALS.items():
            line, = ax.plot(rel_time, trace[f'{prefix}_{axis_label}'].to_numpy(), color=color,
                            linewidth=width, linestyle=style, zorder=z, label=label,
                            solid_capstyle='round')
            handles[label] = line
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{axis_label} axis', fontsize=14)
        ax.set_xlim(0, rel_time[-1] if len(rel_time) else 1)
        if axis_index == 0:
            ax.set_ylabel('Specific force (m/s²)')
        sns.despine(ax=ax)
    return [handles[label] for label, *_ in SIGNALS.values() if label in handles]

# ==============================================================================
# Panels B, C: by-joint distributions
# ==============================================================================

def _box_stats(values: pd.Series) -> Dict[str, float]:
    """5-number summary in matplotlib's bxp format, with p5/p95 whiskers (see WHISKER_PCT)."""
    lo, hi = np.percentile(values, WHISKER_PCT)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {'med': median, 'q1': q1, 'q3': q3, 'whislo': lo, 'whishi': hi, 'fliers': []}


def panel_paired_boxes(ax: plt.Axes, samples: pd.DataFrame, value_cols: Sequence[str],
                       ylabel: str, log: bool = False) -> None:
    """Grouped box plot: one pair of boxes per joint, projected vs unprojected.

    Drawn with ax.bxp from precomputed quantiles rather than seaborn.boxplot on melted data.
    Two reasons: melting several million rows to draw fourteen boxes doubles the memory for
    nothing, and the quantiles here are then exactly the ones the summary parquet reports
    instead of whatever seaborn recomputes after its own filtering.
    """
    order = [j for j in JOINT_ORDER if j in set(samples['joint_label'])]
    if not order:
        return
    width = 0.36
    for offset, column in zip((-width / 2, width / 2), value_cols):
        stats, positions = [], []
        for index, joint in enumerate(order):
            values = samples.loc[samples['joint_label'] == joint, column].dropna()
            if values.empty:
                continue
            stats.append({'label': joint, **_box_stats(values)})
            positions.append(index + offset)
        if not stats:
            continue
        color = color_of(column)
        ax.bxp(stats, positions=positions, widths=width * 0.9, showfliers=False,
               patch_artist=True, medianprops={'color': 'black', 'linewidth': 1.6},
               boxprops={'facecolor': color, 'edgecolor': color, 'alpha': 0.75},
               whiskerprops={'color': color}, capprops={'color': color})

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')
    ax.grid(axis='x', visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=color_of(c), alpha=0.75,
                                     label=label_of(c)) for c in value_cols],
              loc='upper left', fontsize=11)
    ax.text(0.99, 0.02, f'boxes: IQR, whiskers: p{WHISKER_PCT[0]}–p{WHISKER_PCT[1]}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontstyle='italic',
            color='#666666')
    sns.despine(ax=ax)

# ==============================================================================
# Panel D: ECDF
# ==============================================================================

def panel_ecdf(ax: plt.Axes, samples: pd.DataFrame,
               value_cols: Sequence[str] = ('err_proj', 'err_raw')) -> None:
    """Empirical CDF of the pooled error, one curve per signal.

    Computed on every sample and thinned to ECDF_DRAW_POINTS only for drawing, so the curve
    is exact wherever it is read. The median guide lines are the numbers the caption quotes."""
    for column in value_cols:
        values = np.sort(samples[column].dropna().to_numpy())
        if not len(values):
            continue
        fraction = np.arange(1, len(values) + 1) / len(values)
        step = max(len(values) // ECDF_DRAW_POINTS, 1)
        ax.plot(values[::step], fraction[::step], color=color_of(column), linewidth=2.2,
                label=f'{label_of(column)} (median {np.median(values):.2f})')
        ax.vlines(np.median(values), 0, 0.5, color=color_of(column), linewidth=1.0,
                  linestyle=':', alpha=0.8)
    ax.axhline(0.5, color='#999999', linewidth=0.8, linestyle=':')
    ax.set_xscale('log')
    ax.set_xlabel('|a − a$_{true}$| (m/s²)')
    ax.set_ylabel('Fraction of samples below')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=11)
    sns.despine(ax=ax)

# ==============================================================================
# Panel E: error vs the size of the correction
# ==============================================================================

def panel_cutoff_sweep(ax: plt.Axes, sweep: pd.DataFrame) -> None:
    """Residual against low-pass cutoff, with the reference's own in-band noise beneath it.

    This panel exists because the obvious reading of every other panel — "the projected signal
    is 1.x m/s^2 from the truth" — is not what the data supports. The mocap reference is a
    second derivative of marker positions and has a noise floor of its own, and the grey band
    here is how much residual that floor accounts for by itself at each cutoff. Where the blue
    curve sits close to the grey band, mocap cannot resolve the projection error at all and the
    measured residual is an upper bound.

    The unprojected curve is the control that survives the argument: it is compared to the same
    noisy reference, so whatever the floor is, the GAP between the two curves is not made of it.
    """
    if sweep.empty:
        return
    grouped = sweep.groupby('cutoff_hz')
    cutoffs = np.array(sorted(sweep['cutoff_hz'].unique()))

    ax.fill_between(cutoffs, 0, grouped['ref_excess_rms'].median().reindex(cutoffs),
                    color='#999999', alpha=0.3, linewidth=0,
                    label="Reference's own in-band noise")
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
    ax.set_ylabel('Residual |a − a$_{true}$| (m/s²)')
    # Headroom for the legend: it sits top-left, and without this it collides with the panel
    # title in the composed figure.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.32)
    # Anchored just below the axes top rather than flush with it: at 'upper left' the first
    # legend row lands on the panel title in the composed figure. The headroom above pays for
    # this nudge.
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.94), fontsize=10)
    sns.despine(ax=ax)


def panel_error_vs_correction(ax: plt.Axes, samples: pd.DataFrame) -> None:
    """Median error against |a_proj − a_sensor|, with IQR ribbons, plus the identity line.

    The identity line is the panel's whole argument. If the projection were exact then
    a_sensor − a_true would be exactly minus the correction, so the unprojected curve would
    lie ON the diagonal and the projected curve would lie on zero. What is actually drawn is
    how far each falls short of that, as a function of how much correction was needed — which
    is the closest thing here to a statement about when the projection can be trusted.
    """
    binned = samples.assign(bin=pd.cut(samples['corr_norm'], CORRECTION_BINS))
    grouped = binned.groupby('bin', observed=True)
    centers = grouped['corr_norm'].median()

    for column in ('err_raw', 'err_proj'):
        quantiles = grouped[column].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(centers, quantiles[0.25], quantiles[0.75],
                        color=color_of(column), alpha=0.18, linewidth=0)
        ax.plot(centers, quantiles[0.5], color=color_of(column), linewidth=2.2,
                marker='o', markersize=4, label=label_of(column))

    limits = (CORRECTION_BINS[0], CORRECTION_BINS[-1])
    ax.plot(limits, limits, color='#555555', linewidth=1.2, linestyle='--',
            label='Exact projection\n(unprojected error = correction)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*limits)
    ax.set_xlabel('Size of the projection correction |a$_{proj}$ − a$_{sensor}$| (m/s²)')
    ax.set_ylabel('|a − a$_{true}$| (m/s²)')
    ax.legend(loc='upper left', fontsize=10)
    sns.despine(ax=ax)

# ==============================================================================
# Panel F: Bland-Altman
# ==============================================================================

def panel_bland_altman(ax: plt.Axes, samples: pd.DataFrame, seed: int = RNG_SEED) -> None:
    """Bland-Altman of the projected estimate against mocap truth, all three body axes pooled.

    x is the mean of the pair and y their difference, per axis and per sample. Pooling the
    axes is what makes the bias line meaningful — a per-axis version would mostly report
    which axis gravity was on. The three lobes along x are exactly that gravity offset and
    are expected.

    Subsampled to MAX_HEX_POINTS per axis with a fixed seed: a hexbin resolves nothing beyond
    that, and the limits/bias lines below are computed from the FULL data, not the subsample.
    """
    reference = samples[['ref_x', 'ref_y', 'ref_z']].to_numpy()
    difference = samples[['diff_x', 'diff_y', 'diff_z']].to_numpy()
    mean_of_pair = (reference + difference / 2.0).ravel()
    difference = difference.ravel()

    bias = float(np.mean(difference))
    limit = 1.96 * float(np.std(difference))

    if len(difference) > MAX_HEX_POINTS:
        keep = np.random.default_rng(seed).choice(len(difference), MAX_HEX_POINTS, replace=False)
        mean_of_pair, drawn_difference = mean_of_pair[keep], difference[keep]
    else:
        drawn_difference = difference

    # Axis limits from percentiles, not from the data range. A few dozen samples in 5.4M reach
    # +-400 m/s^2 — 40 g at a joint center, i.e. mocap reconstruction glitches rather than
    # motion — and on the full range they flatten the entire distribution into a single line.
    # They are NOT dropped: the bias and limits above come from every sample, and the count
    # falling outside the view is stated on the panel so the clipping is visible.
    x_limit = np.percentile(np.abs(mean_of_pair), 99.9)
    y_limit = np.percentile(np.abs(difference), 99.9)
    outside = int(np.sum((np.abs(mean_of_pair) > x_limit) | (np.abs(drawn_difference) > y_limit)))

    hexes = ax.hexbin(mean_of_pair, drawn_difference, gridsize=70, bins='log',
                      cmap='viridis', mincnt=1, linewidths=0,
                      extent=(-x_limit, x_limit, -y_limit, y_limit))
    ax.figure.colorbar(hexes, ax=ax, label='samples (log)', pad=0.02)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    if outside:
        ax.text(0.99, 0.02, f'axes clipped to p99.9; {outside:,} of {len(drawn_difference):,} '
                            f'drawn points outside',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9, fontstyle='italic',
                color='#333333',
                bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none', 'pad': 2})

    ax.axhline(bias, color='#d1603d', linewidth=1.6, label=f'bias {bias:+.3f}')
    for sign in (-1, 1):
        ax.axhline(bias + sign * limit, color='#d1603d', linewidth=1.2, linestyle='--',
                   label='95% limits of agreement' if sign == 1 else None)
    ax.set_xlabel('Mean of projected and true, per axis (m/s²)')
    ax.set_ylabel('Projected − true (m/s²)')
    ax.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', framealpha=0.9,
              edgecolor='none')
    sns.despine(ax=ax)

# ==============================================================================
# Panel G: spectra
# ==============================================================================

def panel_spectra(ax: plt.Axes, spectra: pd.DataFrame) -> None:
    """Median Welch PSD across every trial, joint and segment, one curve per signal, with
    IQR bands. Unfiltered — see the module docstring.

    Median across trials rather than mean: PSDs span orders of magnitude and one trial with a
    loose sensor would otherwise set the curve.
    """
    columns = {'psd_reference': 'ref', 'psd_projected': 'proj', 'psd_sensor': 'sens'}
    # Rounded before grouping: the frequency grid is identical across trials only as long as
    # every trial got the same nperseg, and a trial shorter than WELCH_NPERSEG gets its own
    # grid. Rounding merges the shared bins instead of splitting the curve into two.
    grouped = spectra.assign(freq_hz=spectra['freq_hz'].round(4)).groupby('freq_hz', observed=True)
    for column, signal in columns.items():
        if column not in spectra.columns:
            continue
        label, color, *_ = SIGNALS[signal]
        quantiles = grouped[column].quantile([0.25, 0.5, 0.75]).unstack()
        frequencies = quantiles.index.to_numpy()
        ax.fill_between(frequencies, quantiles[0.25], quantiles[0.75], color=color,
                        alpha=0.18, linewidth=0)
        ax.plot(frequencies, quantiles[0.5], color=color, linewidth=2.0, label=label)

    ax.axvline(LOWPASS_CUTOFF_HZ, color='#555555', linewidth=1.2, linestyle='--')
    # Blended transform (data x, axes y) so the label tracks the line but not the y-limits,
    # which the log scaling below changes after this call.
    ax.text(LOWPASS_CUTOFF_HZ, 0.98, f' {LOWPASS_CUTOFF_HZ:.0f} Hz cutoff',
            transform=ax.get_xaxis_transform(), fontsize=10, va='top', ha='left',
            color='#555555', rotation=90)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD ((m/s²)²/Hz)')
    ax.legend(loc='lower left', fontsize=10)
    sns.despine(ax=ax)

# ==============================================================================
# Composed supplementary figure
# ==============================================================================

def _label_panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=20, fontweight='bold',
            va='bottom', ha='left')


def figure_supplement(samples: pd.DataFrame, traces: pd.DataFrame, spectra: pd.DataFrame,
                      sweep: pd.DataFrame, subject: str, activity: str,
                      filename: str = 'acceleration_projection.png',
                      save: bool = True, show: bool = False) -> None:
    """The whole supplementary figure: A on the top row across three axes, B-G below.

    Not routed through plot_utils.finalize_and_save_plot — that helper writes one suptitle and
    runs tight_layout over the whole figure, and this layout needs a per-row title (the top
    row is three views of ONE panel, not three panels) and a constrained layout that keeps the
    hexbin's colorbar from eating its neighbour.
    """
    fig = plt.figure(figsize=(19, 16), layout='constrained')
    rows = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.15, 1.15], hspace=0.09)

    trace_axes = rows[0].subgridspec(1, 3, wspace=0.18).subplots()
    handles = panel_traces(trace_axes, traces)
    _label_panel(trace_axes[0], 'A')
    joint = traces['joint'].iloc[0] if not traces.empty else EXAMPLE_JOINT
    trace_axes[1].annotate(
        f'{JOINT_LABELS.get(joint, joint)}, proximal segment — {subject} {activity}, '
        f'most dynamic {WINDOW_S:.0f} s window',
        xy=(0.5, 1.22), xycoords='axes fraction', ha='center', va='bottom',
        fontsize=16, fontweight='bold')
    if handles:
        trace_axes[2].legend(handles=handles, loc='upper right', fontsize=11)

    middle = rows[1].subgridspec(1, 3, wspace=0.28).subplots()
    lower = rows[2].subgridspec(1, 3, wspace=0.28).subplots()

    panels = [
        (middle[0], 'B', lambda ax: panel_paired_boxes(
            ax, samples, ('err_proj', 'err_raw'), '|a − a$_{true}$| (m/s²)')),
        (middle[1], 'C', lambda ax: panel_paired_boxes(
            ax, samples, ('ang_proj', 'ang_raw'), 'Angle to true acceleration (deg)')),
        (middle[2], 'D', lambda ax: panel_cutoff_sweep(ax, sweep)),
        (lower[0], 'E', lambda ax: panel_error_vs_correction(ax, samples)),
        (lower[1], 'F', lambda ax: panel_bland_altman(ax, samples)),
        (lower[2], 'G', lambda ax: panel_spectra(ax, spectra)),
    ]
    titles = {'B': 'Error magnitude, by joint', 'C': 'Error direction, by joint',
              'D': "Residual vs cutoff, vs the reference's noise",
              'E': 'Error vs size of the correction',
              'F': 'Agreement, per axis', 'G': 'Power spectra (unfiltered)'}
    for ax, letter, draw in panels:
        draw(ax)
        ax.set_title(titles[letter], fontsize=15)
        _label_panel(ax, letter)

    # Coverage goes in the suptitle rather than a figure footer: constrained_layout does not
    # reserve space for a stray fig.text, so the footer landed on top of panel G's x label.
    fig.suptitle('Quality of the rigid-body acceleration projection to the joint center\n'
                 + describe_coverage(samples), fontsize=22, fontweight='bold')

    if save:
        path = paths.ensure_parent(PLOTS_DIR / filename)
        fig.savefig(path, dpi=400, bbox_inches='tight')
        print(f"Saved plot to {path}")
    if show:
        plt.show()
    plt.close(fig)

# ==============================================================================
# Standalone panels
# ==============================================================================

def _standalone(draw, filename: str, title: str, figsize: Tuple[float, float],
                samples: Optional[pd.DataFrame] = None, save: bool = True,
                show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    epilog = describe_coverage(samples) if samples is not None else None
    plot_utils.finalize_and_save_plot(fig, title, filename, PLOTS_DIR, epilog=epilog,
                                     save=save, show=show)


def figures_standalone(samples: pd.DataFrame, traces: pd.DataFrame, spectra: pd.DataFrame,
                       sweep: pd.DataFrame, subject: str, activity: str,
                       save: bool = True, show: bool = False) -> None:
    """Every panel again at full size, one file each, for the manuscript and for reading the
    ones that are dense at composed size (E, F and G all are)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    handles = panel_traces(axes, traces)
    if handles:
        axes[2].legend(handles=handles, loc='upper right', fontsize=11)
    joint = traces['joint'].iloc[0] if not traces.empty else EXAMPLE_JOINT
    plot_utils.finalize_and_save_plot(
        fig, f'Projected vs true acceleration at the {JOINT_LABELS.get(joint, joint)} centre '
             f'({subject}, {activity})',
        'traces.png', PLOTS_DIR, save=save, show=show)

    _standalone(lambda ax: panel_paired_boxes(ax, samples, ('err_proj', 'err_raw'),
                                              '|a − a$_{true}$| (m/s²)'),
                'error_by_joint.png', 'Acceleration error at the joint center, by joint',
                (11, 6), samples, save, show)
    _standalone(lambda ax: panel_paired_boxes(ax, samples, ('ang_proj', 'ang_raw'),
                                              'Angle to true acceleration (deg)'),
                'angle_by_joint.png', 'Acceleration direction error, by joint',
                (11, 6), samples, save, show)
    _standalone(lambda ax: panel_ecdf(ax, samples), 'error_ecdf.png',
                'Pooled acceleration error distribution', (9, 6), samples, save, show)
    _standalone(lambda ax: panel_cutoff_sweep(ax, sweep), 'cutoff_sweep.png',
                "Residual vs low-pass cutoff, against the mocap reference's own noise",
                (10, 6.5), samples, save, show)
    _standalone(lambda ax: panel_error_vs_correction(ax, samples), 'error_vs_correction.png',
                'Projection error against the size of the correction', (10, 6.5), samples,
                save, show)
    _standalone(lambda ax: panel_bland_altman(ax, samples), 'bland_altman.png',
                'Projected vs true acceleration: agreement per axis', (10, 6.5), samples,
                save, show)
    _standalone(lambda ax: panel_spectra(ax, spectra), 'spectra.png',
                'Power spectra of the three signals (unfiltered)', (10, 6.5), None, save, show)

# ==============================================================================
# CLI
# ==============================================================================

# Columns the pooled panels need. Listed once, so the load reads exactly this set and adding
# a panel that needs a new column fails loudly here rather than silently loading everything.
POOLED_COLUMNS = ['joint', 'role', 'err_proj', 'err_raw', 'ang_proj', 'ang_raw', 'corr_norm',
                  'ref_x', 'ref_y', 'ref_z', 'diff_x', 'diff_y', 'diff_z']


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subject', default=EXAMPLE_SUBJECT,
                        help="Trial shown in the time-series panel, e.g. Subject01.")
    parser.add_argument('--activity', default=EXAMPLE_ACTIVITY)
    parser.add_argument('--joint', default=EXAMPLE_JOINT,
                        help=f"Joint for the time-series panel. Only joints stored by the "
                             f"experiment are available (default {EXAMPLE_JOINT}).")
    parser.add_argument('--composed-only', action='store_true',
                        help="Write only the composed supplementary figure, not the per-panel files.")
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    print("Loading per-trial tables...")
    samples = load_samples(POOLED_COLUMNS)
    if samples.empty:
        print("No per-sample table found. Run `python -m experiments.acceleration_projection` first.")
        return
    samples['joint_label'] = samples['joint'].map(JOINT_LABELS).astype('category')
    spectra = load_trial_table('spectra')
    sweep = load_trial_table('cutoff_sweep')
    print(describe_coverage(samples))

    subject_id = args.subject.replace('Subject', '')
    traces = load_trial_table('traces', subjects=[subject_id], activities=[args.activity])
    traces = traces[traces['joint'] == args.joint] if not traces.empty else traces
    if traces.empty:
        print(f"No traces for {args.subject} {args.activity} {args.joint}; the time-series "
              f"panel will be empty. The experiment stores traces for {EXAMPLE_JOINT} only.")

    figure_supplement(samples, traces, spectra, sweep, args.subject, args.activity, show=args.show)
    if not args.composed_only:
        figures_standalone(samples, traces, spectra, sweep, args.subject, args.activity,
                           show=args.show)
    print(f"\nFigures under {PLOTS_DIR}")


if __name__ == '__main__':
    main()
