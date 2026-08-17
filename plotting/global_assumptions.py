"""
Figures for experiments/global_assumptions.py: how far the accelerometer-reads-gravity and
magnetometer-reads-one-constant-field assumptions are from true, split by whether the sensor
was moving.

Everything here reads back the per-trial tables that experiment wrote. Nothing reloads or
re-projects a trial, which is what makes the figures cheap to re-tune.

Metrics
-------
    linacc        |a_world - g|, non-gravity acceleration              m/s^2   (needs mocap)
    magdev        |m_world - m_global|, field deviation                a.u.    (needs mocap)
    magdev_angle  the same, as a direction error                       deg     (needs mocap)
    acc_norm_dev  ||a| - |g||, the reference-free accelerometer twin   m/s^2
    mag_norm_dev  ||m| - |m_global||, the reference-free mag twin      a.u.

Figures
-------
  summary            (--summary) THE figure, and the only one here that crosses datasets: both
                     assumptions x {still, moving} on one anatomical axis, every dataset drawn
                     separately so no single lab carries the claim. Saved to the plots root
                     rather than under a dataset, and reads the pooled statistics parquets
                     rather than any per-sample table.
  headline           the whole argument in one panel pair: every segment's accelerometer and
                     magnetometer departure, static against moving, on a log axis. The
                     accelerometer's two bars are an order of magnitude apart and the
                     magnetometer's are not, which is the asymmetry the paper turns on.
  distributions      per metric x {box | joy | strip}. Box is regime-split and is the one that
                     carries a number; joy shows the SHAPE (a bimodal segment means the field
                     it sits in has two regimes, which no box plot shows); strip shows the raw
                     cloud behind the summary.
  noise floor        per-axis intrinsic noise per segment, whole-body-static against
                     per-sensor-static, against the std the filter is tuned with.
  static coverage    how much of each recording each sensor was still for, and how much of
                     that the mocap window actually covers.
  per subject        subject x segment heatmap per metric — the between-subject spread that a
                     pooled median hides.
  per trial          every trial's median as a point, by segment, so one anomalous recording
                     is visible rather than averaged away.
  time series        both assumptions through a representative whole-body-static bout, shaded.

NOT here: observability. o^J is a property of a joint PAIR and of the joint-center projection
rather than of the global sensor assumptions this experiment measures, so its figures live with
the joint-offset work instead — see scratch/observability_plots_for_joint_offset.py for the ones
lifted out of this module. The experiment still writes joint_samples and joint_stats, and report
sections 6 and 7 still print off them; only the plotting moved.
  diagnostics        (--diagnostics) per-trial static-detector traces, so a detector failure
                     is visible rather than silently shifting a distribution.

Box statistics are computed from the FULL sample tables with a groupby and drawn with `bxp`,
not handed to seaborn, so no box in this module is estimated off a subsample. The joy and
strip figures do subsample, and say so in their footers.

    python -m plotting.global_assumptions --summary
    python -m plotting.global_assumptions --dataset alborno
    python -m plotting.global_assumptions --dataset imove --sides both
    python -m plotting.global_assumptions --dataset alborno --diagnostics
"""
import argparse
import textwrap
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
from experiments.experiment_utils import pipeline_constants
from experiments.global_assumptions import (BODY_STATIC_MIN_S, DATASETS, MAG_UNIT, METRIC_UNITS,
                                            SEGMENT_METRICS, STATIC_GYRO_MAX, STATIC_WINDOW_S,
                                            dataset_dir, enumerate_trials, get_dataset,
                                            load_trial_table, statistics_path)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MAX_STRIP_POINTS = 5000     # per category; see plot_stripplot
MAX_JOY_POINTS = 200_000    # per category; a KDE over 24 M samples is minutes for no visible gain
STRIP_CLIP_QUANTILE = 0.99

# Display-only truncation, per metric: the KDE is fit to samples below `percentile` and the axis
# capped at `xlim`. Without it a single metric's long right tail (footfall impulses for linacc,
# observability spikes) compresses every ridge into the leftmost few percent of the axis. Quoted
# statistics come from the summary parquet, which is untruncated.
JOY_LIMITS = {
    'linacc': {'percentile': 0.975, 'xlim': 15},
    'acc_norm_dev': {'percentile': 0.975, 'xlim': 15},
    'magdev': {'percentile': 0.85, 'xlim': None},
    'magdev_angle': {'percentile': 0.98, 'xlim': 90},
    # The trial-reference arm shares its twin's limits deliberately: the two are the same
    # quantity against a differently-scoped constant, and the whole point of drawing them is to
    # read one against the other, which a per-metric axis would defeat.
    'magdev_trial': {'percentile': 0.85, 'xlim': None},
    'magdev_angle_trial': {'percentile': 0.98, 'xlim': 90},
    'magdev_loo': {'percentile': 0.85, 'xlim': None},
    'magdev_angle_loo': {'percentile': 0.98, 'xlim': 90},
    'magdev_clean': {'percentile': 0.85, 'xlim': None},
    'magdev_angle_clean': {'percentile': 0.98, 'xlim': 90},
    'mag_norm_dev': {'percentile': 0.90, 'xlim': None},
}
# Strip-plot x-axis clip, per metric — a display bound only: the median/IQR lines and every
# quoted statistic use all samples, and whatever falls beyond is drawn, counted and annotated at
# the axis edge. Set per metric because the tails are not comparably heavy: magdev's max is ~2x
# its p99 while observability's is ~12x, so one shared quantile either wastes most of the magdev
# axis or leaves observability unreadable.
STRIP_CLIP = {'linacc': 0.99, 'acc_norm_dev': 0.99, 'magdev': 0.999, 'magdev_angle': 0.999,
              'magdev_trial': 0.999, 'magdev_angle_trial': 0.999, 'magdev_loo': 0.999,
              'magdev_angle_loo': 0.999, 'magdev_clean': 0.999, 'magdev_angle_clean': 0.999,
              'mag_norm_dev': 0.999}

# Regime colours. Green/rose for the still/moving contrast because that pairing carries the
# figure's whole message and needs to read at a glance; both are kept out of viridis's
# purple-blue-teal-green-yellow range, which is what the per-segment colouring uses.
REGIME_COLORS = {
    'all': '#6b7280',           # slate — the undifferentiated baseline
    'static': '#2a9d5c',        # green
    'nonstatic': '#c9436b',     # rose
    'body_static': '#1f6f8b',   # teal-blue
}
REGIME_LABELS = {
    'all': 'all samples',
    'static': 'static (this sensor still)',
    'nonstatic': 'moving',
    'body_static': 'static (whole body still)',
}
# The pair the headline and the split box plots contrast. `all` is the union of the two and
# `body_static` a subset of `static`, so drawing all four would be four bars of which two are
# arithmetic consequences of the others.
CONTRAST_REGIMES = ('static', 'nonstatic')

MAJORITY_STATIC_LABEL = 'majority of sensors still'

INTERVAL_COLORS = {'sitting': '#c8890a', 'standing': '#c9436b', 'static': '#2a9d5c',
                   'body_static': '#1f6f8b', MAJORITY_STATIC_LABEL: '#1f6f8b',
                   'ambulation': '#8896a6', 'unlabeled': '#d8dde3'}

STATIC_PAD_S = 15.0   # context shown either side of the example static bout

# What the TIME-SERIES figure shades. `body_static` (every sensor still at once) is the right
# regime for the noise floor and is what the analysis reports, but it is a poor thing to draw:
# it is unanimous, so one sensor's postural sway clears the whole band, and a stretch that reads
# as obviously still leaves most of itself unshaded. Measured on alborno 07/complexTasks, the
# whole-body rule marks 62% of the time every sensor's linacc says is quiet, in 33 fragments.
#
# The figure therefore shades a MAJORITY of sensors instead. This is a DISPLAY threshold and
# nothing else reads it — no table, no statistic and no regime is computed from it, because
# loosening the analysis rule measurably contaminates the noise floor it exists to measure
# (allowing one dissenter raises the pooled acc floor 10-21%, and it climbs from there).
# Deliberately named apart from `body_static` so the two can never be confused in a caption.
MAJORITY_STATIC_FRACTION = 0.5

# Which metric pairs the headline figure contrasts, and the label for each panel. One
# mocap-referenced metric per sensor, because those are the physically meaningful ones; the
# reference-free twins appear in the per-metric figures.
HEADLINE_PANELS = (
    ('linacc', 'Accelerometer: $|a_{world} - g|$  (m/s²)',
     'Departure from "the accelerometer reads gravity"'),
    ('magdev', f'Magnetometer: $|m_{{world}} - m_{{global}}|$  ({MAG_UNIT})',
     'Departure from "the magnetometer reads one constant field"'),
)

METRIC_LABELS = {
    'linacc': 'Linear acceleration magnitude (m/s²)',
    'acc_norm_dev': 'Accelerometer magnitude error $||a|-|g||$ (m/s²)',
    'magdev': f'Magnetic field deviation ({MAG_UNIT}, 1 ≈ Earth field)',
    'magdev_angle': 'Magnetic field direction error (deg)',
    'magdev_trial': f"Magnetic field deviation vs the TRIAL's field ({MAG_UNIT})",
    'magdev_angle_trial': "Magnetic field direction error vs the TRIAL's field (deg)",
    'magdev_loo': f"Magnetic field deviation vs a LEAVE-ONE-OUT field ({MAG_UNIT})",
    'magdev_angle_loo': 'Magnetic field direction error vs a LEAVE-ONE-OUT field (deg)',
    'magdev_clean': f"Magnetic field deviation vs the CLEANEST sensor ({MAG_UNIT})",
    'magdev_angle_clean': 'Magnetic field direction error vs the CLEANEST sensor (deg)',
    'mag_norm_dev': f'Magnetometer magnitude error $||m|-|M||$ ({MAG_UNIT})',
}
METRIC_TITLES = {
    'linacc': 'Non-gravity acceleration by segment',
    'acc_norm_dev': 'Accelerometer magnitude error by segment (reference-free)',
    'magdev': "Magnetic field deviation from the subject's global field, by segment",
    'magdev_angle': 'Magnetic field DIRECTION error by segment',
    # The trial-reference arm. Same computation, a constant refit per trial instead of per
    # subject, so the difference from the two above is between-trial drift in the reference.
    'magdev_trial': "Magnetic field deviation from the TRIAL's own field, by segment",
    'magdev_angle_trial': "Magnetic field DIRECTION error vs the TRIAL's own field, by segment",
    # loo excludes the sensor itself from its own reference — the unbiased arm. clean references
    # everything against one sensor and is a DIAGNOSTIC: it zeroes that sensor's own row.
    'magdev_loo': 'Magnetic field deviation from a leave-one-out field, by segment',
    'magdev_angle_loo': 'Magnetic field DIRECTION error vs a leave-one-out field, by segment',
    'magdev_clean': "Magnetic field deviation from the cleanest sensor's field, by segment",
    'magdev_angle_clean': "Magnetic field DIRECTION error vs the cleanest sensor, by segment",
    'mag_norm_dev': 'Magnetometer magnitude error by segment (reference-free)',
}
STATIC_EPILOG = (f"static = |gyro| below {STATIC_GYRO_MAX} rad/s across a {STATIC_WINDOW_S:.2f} s "
                 f"window; gyro-only, so it is independent of both quantities under test")

# Spelled out under the time series because the shading is NOT the regime the tables report,
# and a reader who assumes it is would misread every width on the figure.
MAJORITY_STATIC_EPILOG = (
    "shading = more than half of the {n} sensors individually static; a display threshold, "
    "looser than the unanimous body_static regime the statistics are computed over")

MODALITY_UNITS = {'gyro': 'rad/s', 'acc': 'm/s²', 'mag': MAG_UNIT}
MODALITY_STD_KEY = {'gyro': 'gyro_std', 'acc': 'acc_std', 'mag': 'mag_std'}


def plots_dir(dataset: str):
    return paths.plots_dir("global_assumptions", dataset)

# ==============================================================================
# Naming / ordering / colour
# ==============================================================================

def joint_label(joint: str) -> str:
    """'R_Knee' -> 'Knee R', 'R_Knee_H' -> 'Knee High R', 'Lumbar' -> 'Lumbar'.

    Unused in this module since the observability figures moved out, and kept because it is this
    package's shared implementation: plotting/relative_vs_absolute.py and
    plotting/acceleration_projection.py both import it from here.

    Side goes last so that a sorted axis groups by joint level (every Knee together) instead of
    by side. IMoVE's placement suffix stays in the middle, between the joint and the side, for
    the same reason: it keeps the three placements of one joint adjacent."""
    parts = joint.split('_')
    side = parts[0] if parts[0] in ('R', 'L') else None
    rest = list(parts[1:] if side else parts)
    # Spell the placement out. 'L_Ankle_L' would otherwise render 'Ankle L L', where the two
    # L's mean Left and Low and nothing in the label says which is which.
    if len(rest) > 1 and rest[-1] in ('H', 'L'):
        rest[-1] = {'H': 'High', 'L': 'Low'}[rest[-1]]
    body = ' '.join(rest)
    return f"{body} {side}" if side else body


def side_filter(names: Sequence[str], sides: str) -> List[str]:
    """Restricts an ordered segment/joint list to one side, keeping midline entries.

    Default is BOTH, unlike the predecessor's right-side-only. That default was justified by
    the two sides being near-mirror images — the benchmark's own ICC puts left-vs-right of a
    joint at 0.51-0.56 — but that is a statement about FILTER OUTPUT, and it does not transfer
    to this experiment. What is measured here is how far each assumption fails as a function of
    WHERE ON THE BODY a sensor sits, and the contralateral sensor is a free replicate of
    precisely that: it is nominally the same placement in the same magnetic environment, so a
    claim that holds on one side and not the other is a finding rather than a duplicate row.
    Al Borno's magnetic environment in particular is not left-right symmetric — distortion
    scales with the sensor's height above the lab floor and with its position in the room, not
    with which leg it is on.

    IMoVE names its segments 'Thigh R Mid' rather than ending in the side, so the side token is
    matched anywhere in the name after the first word."""
    if sides == 'both':
        return list(names)
    drop = 'L' if sides == 'right' else 'R'
    return [n for n in names if drop not in n.split()[1:]]


def proximity_color(index: int, n: int, cmap_name: str = 'viridis',
                    lo: float = 0.08, hi: float = 0.92):
    """Maps an ordinal index (proximal=0 ... distal=n-1) to a perceptually uniform colormap.

    Not a hue sweep in HSL: equal-degree hue steps are perceptually uneven (steps through green
    look far smaller than the same steps through orange), so equally-spaced anatomical levels
    came out looking unevenly spaced. Viridis is built to avoid exactly that."""
    frac = index / (n - 1) if n > 1 else 0.0
    return plt.get_cmap(cmap_name)(lo + (hi - lo) * frac)


def color_map(order: Sequence[str]) -> Dict[str, tuple]:
    return {name: proximity_color(i, len(order)) for i, name in enumerate(order)}


def figure_height(n_rows: int, per_row: float = 0.9, extra: float = 0.8) -> float:
    return max(3.5, per_row * n_rows + extra)


def wrap_epilog(*parts: Optional[str], width: int = 130) -> str:
    """Joins the epilog fragments and hard-wraps them.

    Not cosmetic. `finalize_and_save_plot` writes the epilog as a single right-aligned line at
    figure coordinate (0.99, 0.01), and `bbox_inches='tight'` expands the SAVED canvas to
    contain every artist — so an epilog longer than the axes are wide silently widens the
    figure to fit it, leaving the plot squeezed into the right-hand portion of a mostly blank
    image. These figures accumulate three fragments (the box-statistics note, the o^J filter
    note and the static-detector definition), which together run to about 300 characters.
    """
    text = "; ".join(part for part in parts if part)
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""

# ==============================================================================
# Data access
# ==============================================================================

def regime_frames(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """{regime: the rows of `df` in it}. The regimes OVERLAP by design — `body_static` is a
    subset of `static`, `all` contains everything — so these are views, not a partition."""
    static = df['static'].to_numpy()
    return {'all': df, 'static': df[static], 'nonstatic': df[~static],
            'body_static': df[df['body_static'].to_numpy()]}


def load_samples(dataset: str, table: str, metrics: Sequence[str], group_col: str,
                 row_keys: Optional[Sequence[Tuple[str, str]]] = None) -> pd.DataFrame:
    """One sample table across trials, projected to the columns the figures need.

    The projection is load-bearing rather than tidy: IMoVE's segment_samples runs to ~24 M rows
    across 262 trials, and pulling the timestamps and raw signal norms in only to drop them
    costs several hundred megabytes."""
    columns = [group_col, 'static', 'body_static', *metrics]
    return load_trial_table(dataset, table, row_keys=row_keys, columns=columns)


def box_stats(df: pd.DataFrame, group_col: str, metric: str, order: Sequence[str],
              regimes: Sequence[str]) -> Tuple[Dict[Tuple[str, str], dict], List[str], List[str]]:
    """Box-plot statistics computed on the FULL data, in the dict form `Axes.bxp` consumes.

    Returns ({(regime, group): box}, the groups that have data, the regimes that have data).

    Handing millions of rows to seaborn works but re-derives these same quantiles at drawing
    time, in a code path that then also has to be told not to draw fliers. Computing them here
    means every box in this module describes every sample, no subsampling caveat is needed, and
    the figures render in seconds rather than minutes.

    Groups with nothing to draw are reported as absent rather than kept as an empty row. That
    is not tidiness: a blank row against a labelled tick reads as "measured and found to be
    nothing" rather than "this sensor has no value for this metric" — which is the case for
    every mocap-referenced metric on a trial whose static time falls outside the mocap window.

    Whiskers are p05/p95 rather than the Tukey 1.5-IQR rule. Every metric here is strongly
    right-skewed, so a 1.5-IQR whisker sits far inside the data and the mass beyond it — tens of
    percent of the samples for the distal segments — reads as "outliers" when it is the ordinary
    behaviour of a foot. Fliers are not drawn for the same reason: matplotlib scales the axis to
    the drawn artists, and the impulsive tail would squash every box to a few pixels.
    """
    boxes: Dict[Tuple[str, str], dict] = {}
    used_regimes = []
    for regime in regimes:
        frame = regime_frames(df)[regime]
        if frame.empty or frame[metric].notna().sum() == 0:
            continue
        used_regimes.append(regime)
        grouped = frame.groupby(group_col, observed=True)[metric]
        quantiles = grouped.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
        counts = grouped.count()
        for group in order:
            if group not in quantiles.index or counts.get(group, 0) == 0:
                continue
            row = quantiles.loc[group]
            if not np.isfinite(row.to_numpy()).all():
                continue
            boxes[(regime, group)] = {
                'label': group, 'whislo': row[0.05], 'q1': row[0.25], 'med': row[0.5],
                'q3': row[0.75], 'whishi': row[0.95], 'fliers': [], 'n': int(counts[group]),
            }
    present = [group for group in order if any((regime, group) in boxes for regime in used_regimes)]
    return boxes, present, used_regimes

# ==============================================================================
# Distribution figures
# ==============================================================================

def plot_split_box(df: pd.DataFrame, group_col: str, order: List[str], metric: str,
                   dataset: str, title: str, filename: str,
                   regimes: Sequence[str] = CONTRAST_REGIMES, log: bool = True,
                   epilog: Optional[str] = None, save: bool = True, show: bool = False) -> None:
    """Horizontal box plot with one box per (group, regime), grouped by group.

    THE figure of this experiment. Log x by default because the static and moving
    distributions of the accelerometer metrics are one to two orders of magnitude apart, and on
    a linear axis the static box collapses to a line at zero — which hides the very comparison
    the plot exists to make.
    """
    boxes, present, used = box_stats(df, group_col, metric, order, regimes)
    if not boxes:
        print(f"No data for {filename}, skipping.")
        return
    dropped = [group for group in order if group not in present]
    if dropped:
        print(f"  {filename}: no data for {', '.join(dropped)} — omitted from the axis.")
    if log:
        # A log axis cannot show a zero or negative lower whisker. Floor the whole figure at a
        # small fraction of the smallest positive median rather than dropping those boxes, so a
        # genuinely tiny static value still reads as tiny instead of vanishing.
        positive = [box['med'] for box in boxes.values() if box['med'] > 0]
        floor = (min(positive) / 50.0) if positive else 1e-6
        for box in boxes.values():
            for key in ('whislo', 'q1', 'med', 'q3', 'whishi'):
                box[key] = max(box[key], floor)

    n_regimes = max(len(used), 1)
    fig, ax = plt.subplots(figsize=(10, figure_height(len(present),
                                                     per_row=0.55 * n_regimes + 0.35)))
    span = 0.8
    width = span / n_regimes
    for regime_index, regime in enumerate(used):
        pairs = [(boxes[(regime, group)], i - span / 2 + width * (regime_index + 0.5))
                 for i, group in enumerate(present) if (regime, group) in boxes]
        if not pairs:
            continue
        color = REGIME_COLORS[regime]
        ax.bxp([box for box, _ in pairs], positions=[pos for _, pos in pairs], vert=False,
               widths=width * 0.82, showfliers=False, patch_artist=True,
               boxprops={'facecolor': mcolors.to_rgba(color, 0.55), 'edgecolor': color},
               medianprops={'color': 'black', 'linewidth': 1.8},
               whiskerprops={'color': color}, capprops={'color': color})

    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present)
    ax.set_ylim(-0.6, len(present) - 0.4)
    ax.invert_yaxis()
    if log:
        ax.set_xscale('log')
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.set_ylabel('')
    ax.grid(axis='y', visible=False)
    sns.despine(ax=ax)
    ax.legend(handles=[Patch(facecolor=mcolors.to_rgba(REGIME_COLORS[r], 0.55),
                             edgecolor=REGIME_COLORS[r], label=REGIME_LABELS[r]) for r in used],
              bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)

    footer = "box = IQR, whiskers = p05/p95, computed on every sample (no subsampling)"
    plot_utils.finalize_and_save_plot(fig, title, filename, plots_dir(dataset),
                                      epilog=wrap_epilog(footer, epilog), save=save, show=show)


def plot_joyplot(data: pd.DataFrame, cat_col: str, order: List[str], xlabel: str, title: str,
                 dataset: str, filename: str, colors: Dict[str, tuple], percentile: float = 0.85,
                 xlim: Optional[float] = None, epilog: Optional[str] = None,
                 save: bool = True, show: bool = False) -> None:
    """Ridgeline (joy) plot: one KDE per category, deliberately overlapping.

    Not routed through plot_utils.finalize_and_save_plot: the overlap comes from a negative
    subplot hspace, and that helper's tight_layout would immediately undo it. The rcParams
    override is applied through rc_context rather than sns.set_theme so that the transparent
    facecolor needed for the overlap does not leak into every later figure in the process.
    """
    if data.empty:
        print(f"No data for {filename}, skipping.")
        return
    # Truncate to BOTH the percentile and the x-limit, and clip the KDE's support to match. The
    # x-limit part is not cosmetic: clip_on=False (which is what lets a ridge overflow vertically
    # into its neighbour) also stops matplotlib clipping horizontally, so a curve drawn out to a
    # tail of 10^4 stays in the artist extents that bbox_inches='tight' measures — which silently
    # produced a 32-inch-wide figure with all the content squeezed into its left eighth.
    upper = data['value'].quantile(percentile)
    if xlim is not None:
        upper = min(upper, xlim)
    plot_data = data[data['value'] <= upper]
    if plot_data.empty:
        print(f"No data below the display limit for {filename}, skipping.")
        return

    with plt.rc_context({'axes.facecolor': (0, 0, 0, 0), 'axes.grid': False,
                         'axes.spines.left': False, 'axes.spines.bottom': True}):
        grid = sns.FacetGrid(plot_data, row=cat_col, hue=cat_col, row_order=order, aspect=8,
                             height=0.8, palette=colors, sharey=True)
        # The fill and its edge stroke are separate artists in seaborn's kdeplot, and the stroke
        # ignores `alpha`, so this gives each ridge a thin fully-saturated outline in its own hue
        # over a softer fill body.
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
                  f"({percentile:.0%} quantile"
                  f"{', x-limit' if xlim is not None and upper == xlim else ''})")
        grid.figure.text(0.99, -0.02, wrap_epilog(footer, epilog), ha='right',
                         va='top', fontsize=10, fontstyle='italic')

        if save:
            path = paths.ensure_parent(plots_dir(dataset) / filename)
            grid.figure.savefig(path, dpi=600, bbox_inches='tight')
            print(f"Saved plot to {path}")
        if show:
            plt.show()
        plt.close(grid.figure)


def plot_stripplot(data: pd.DataFrame, cat_col: str, order: List[str], xlabel: str, title: str,
                   dataset: str, filename: str, colors: Dict[str, tuple],
                   max_points: int = MAX_STRIP_POINTS, seed: int = 0,
                   clip_quantile: float = STRIP_CLIP_QUANTILE, epilog: Optional[str] = None,
                   save: bool = True, show: bool = False) -> None:
    """Horizontal strip plot with per-category downsampling — a category holding millions of
    samples is an unreadable smear and takes minutes to render.

    The median/IQR lines are computed from the FULL data, before downsampling, so the summary
    stays exact even though the cloud behind it is a subsample. Seeded, so the same inputs give
    the same figure.

    The x-axis is clipped at `clip_quantile` of the pooled samples, because these metrics have
    tails running one to two orders of magnitude past their own IQR and an axis sized to the
    largest sample compresses every distribution into its leftmost pixels. Nothing is dropped
    silently: the off-scale samples are drawn as carets pinned at the axis edge, each row is
    annotated with how many of its samples lie beyond, and the true maximum goes in the footer.
    """
    data = data[data['value'].notna()]
    if data.empty:
        print(f"No data for {filename}, skipping.")
        return

    grouped = data.groupby(cat_col, observed=True)['value']
    stats = grouped.quantile([0.25, 0.5, 0.75]).unstack()
    totals = grouped.size()
    limit = float(data['value'].quantile(clip_quantile))
    beyond = data[data['value'] > limit].groupby(cat_col, observed=True)['value'].agg(['size', 'max'])
    sampled = pd.concat([group.sample(n=min(len(group), max_points), random_state=seed)
                         for _, group in data.groupby(cat_col, observed=True)], ignore_index=True)
    in_range = sampled[sampled['value'] <= limit]
    off_scale = sampled[sampled['value'] > limit].assign(value=limit)

    fig, ax = plt.subplots(figsize=(9, figure_height(len(order))))
    sns.stripplot(data=in_range, x='value', y=cat_col, order=order, hue=cat_col, palette=colors,
                  dodge=False, jitter=0.35, size=2, alpha=0.25, linewidth=0, ax=ax, legend=False)
    if not off_scale.empty:
        sns.stripplot(data=off_scale, x='value', y=cat_col, order=order, hue=cat_col,
                      palette=colors, dodge=False, jitter=0.3, size=5, alpha=0.55, linewidth=0,
                      marker='>', ax=ax, legend=False)

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
              f"{clip_quantile:.1%} quantile ({limit:,.3g}), true max {data['value'].max():,.3g}")
    plot_utils.finalize_and_save_plot(fig, title, filename, plots_dir(dataset),
                                      epilog=wrap_epilog(footer, epilog), save=save, show=show)

# ==============================================================================
# Headline figure
# ==============================================================================

def plot_headline(segment_df: pd.DataFrame, segments: List[str], dataset: str, caption: str,
                  save: bool = True, show: bool = False) -> None:
    """Both assumptions, both regimes, one figure.

    Two panels side by side on log axes with a shared segment axis. The claim it is making is
    the CONTRAST BETWEEN THE PANELS, not either panel alone: the accelerometer's static and
    moving markers are an order of magnitude apart at every segment, while the magnetometer's
    sit nearly on top of each other. Gravity is recoverable by waiting; magnetic distortion is
    a property of where the sensor is and is not.
    """
    fig, axes = plt.subplots(1, len(HEADLINE_PANELS), figsize=(13, figure_height(len(segments))),
                             sharey=True)
    drew_any = False
    for ax, (metric, xlabel, panel_title) in zip(np.atleast_1d(axes), HEADLINE_PANELS):
        if metric not in segment_df.columns:
            continue
        for regime in CONTRAST_REGIMES:
            frame = regime_frames(segment_df)[regime]
            if frame.empty:
                continue
            grouped = frame.groupby('segment', observed=True)[metric]
            quantiles = grouped.quantile([0.25, 0.5, 0.75]).unstack()
            present = [s for s in segments if s in quantiles.index
                       and np.isfinite(quantiles.loc[s, 0.5])]
            if not present:
                continue
            drew_any = True
            y = [segments.index(s) for s in present]
            median = quantiles.loc[present, 0.5].to_numpy()
            lower = median - quantiles.loc[present, 0.25].to_numpy()
            upper = quantiles.loc[present, 0.75].to_numpy() - median
            ax.errorbar(median, y, xerr=np.vstack([lower, upper]), fmt='o', markersize=8,
                        color=REGIME_COLORS[regime], ecolor=REGIME_COLORS[regime], elinewidth=2.2,
                        capsize=4, label=REGIME_LABELS[regime], zorder=3)
        ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_title(panel_title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', visible=False)
        sns.despine(ax=ax)

    if not drew_any:
        print("No headline data, skipping.")
        plt.close(fig)
        return

    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(segments)))
    first.set_yticklabels(segments)
    first.set_ylim(-0.6, len(segments) - 0.4)
    first.invert_yaxis()
    # One shared legend under both panels rather than inside either. Both panels are dense at
    # every corner — the distal segments' moving IQR spans most of the accelerometer axis — so
    # any in-axes placement lands on data.
    handles = [Line2D([0], [0], marker='o', linestyle='-', linewidth=2.2, markersize=8,
                      color=REGIME_COLORS[r], label=REGIME_LABELS[r]) for r in CONTRAST_REGIMES]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=12,
               bbox_to_anchor=(0.5, -0.02))

    plot_utils.finalize_and_save_plot(
        fig, f"The two global sensor assumptions, static vs moving\n{caption}",
        "headline_assumptions.png", plots_dir(dataset),
        epilog=wrap_epilog("marker = median, bar = IQR, over every sample", STATIC_EPILOG),
        save=save, show=show)

# ==============================================================================
# Noise floor
# ==============================================================================

def plot_noise_floor(sensor_df: pd.DataFrame, segments: List[str], dataset: str, caption: str,
                     modalities: Sequence[str] = ('gyro', 'acc', 'mag'),
                     save: bool = True, show: bool = False) -> None:
    """Per-axis intrinsic noise per segment, whole-body-static against per-sensor-static.

    The two bars are two different measurements, not an estimate and its error bar. A sensor
    can be rotationally still while the body around it is not — a foot in stance carries
    footfall transients from the other leg and from the ground — so the whole-body number is
    the sensor's own floor and the per-sensor one is what a filter would actually see during a
    window it detected as quiet for itself. The dashed line is the std the filter is tuned
    with, which is the comparison that decides whether that tuning is defensible.
    """
    if sensor_df.empty or 'n_noise_samples' not in sensor_df.columns:
        print("No noise-floor data, skipping.")
        return
    constants = pipeline_constants()
    fig, axes = plt.subplots(1, len(modalities),
                             figsize=(5 * len(modalities),
                                      figure_height(len(segments), per_row=0.5)),
                             sharey=True, squeeze=False)
    axes = axes[0]
    drew_any = False
    for ax, modality in zip(axes, modalities):
        for regime_index, regime in enumerate(('body_static', 'static')):
            rows = sensor_df[(sensor_df['regime'] == regime) & (sensor_df['n_noise_samples'] > 0)]
            if rows.empty:
                continue
            values, positions = [], []
            for i, segment in enumerate(segments):
                block = rows[rows['segment'] == segment]
                if block.empty:
                    continue
                # Length-weighted across sensor-trials, matching how the console report pools
                # them: a 30 s pause is worth more than a 0.1 s stance phase.
                weights = block['n_noise_samples'].to_numpy(dtype=float)
                axis_values = np.array([block[f'{modality}_noise_{a}'].to_numpy(dtype=float)
                                        for a in 'xyz'])
                with np.errstate(invalid='ignore'):
                    pooled = (np.nansum(axis_values * weights, axis=1)
                              / np.nansum(~np.isnan(axis_values) * weights, axis=1))
                if np.isnan(pooled).all():
                    continue
                values.append(np.nanmean(pooled))
                positions.append(i - 0.2 + 0.4 * regime_index)
            if not values:
                continue
            drew_any = True
            ax.barh(positions, values, height=0.36, color=REGIME_COLORS[regime],
                    alpha=0.85, label=REGIME_LABELS[regime], zorder=3)
        tuned = constants[MODALITY_STD_KEY[modality]]
        ax.axvline(tuned, color='black', linestyle='--', linewidth=1.4, zorder=4)
        # Annotated on the line rather than in a legend entry: the bars fill every corner of
        # these panels, so an in-axes legend lands on data in at least one of the three.
        ax.text(tuned, -0.75, f" tuned {MODALITY_STD_KEY[modality]}={tuned:g}", fontsize=9,
                fontstyle='italic', ha='left', va='bottom', color='black')
        ax.set_xlabel(f"{modality} noise std ({MODALITY_UNITS[modality]})")
        ax.set_title(modality.capitalize(), fontsize=13, fontweight='bold')
        ax.grid(axis='y', visible=False)
        sns.despine(ax=ax)

    if not drew_any:
        print("No usable static intervals for the noise floor, skipping.")
        plt.close(fig)
        return
    axes[0].set_yticks(range(len(segments)))
    axes[0].set_yticklabels(segments)
    axes[0].set_ylim(-1.0, len(segments) - 0.4)
    axes[0].invert_yaxis()
    handles = [Patch(facecolor=REGIME_COLORS[r], alpha=0.85, label=REGIME_LABELS[r])
               for r in ('body_static', 'static')]
    handles.append(Line2D([0], [0], color='black', linestyle='--', label='the filter’s tuned std'))
    # Inside the Mag panel, which is reliably the empty one: the measured magnetometer floor
    # runs ~0.004 against a tuned mag_std of 0.05, so the dashed line sets the axis and the bars
    # occupy its leftmost tenth. A figure-level legend below the axes collides with the epilog
    # on the short (few-segment) version of this figure, where the two land on the same line.
    # Without a magnetometer there is no such panel, so it falls back to the last one drawn.
    axes[-1].legend(handles=handles, loc='center right', fontsize=10, framealpha=0.9,
                    frameon=True)

    plot_utils.finalize_and_save_plot(
        fig, f"Intrinsic sensor noise floor during static periods\n{caption}",
        "noise_floor_by_segment.png", plots_dir(dataset),
        epilog="mean over axes of the length-weighted per-axis std WITHIN each static interval, "
               "so between-interval pose differences do not enter",
        save=save, show=show)

# ==============================================================================
# Static coverage
# ==============================================================================

def plot_static_coverage(sensor_df: pd.DataFrame, segments: List[str], dataset: str,
                         caption: str, save: bool = True, show: bool = False) -> None:
    """How much of each recording each sensor was still for, and how much of that the mocap
    window covers.

    The second panel is the caveat the first one needs. The Al Borno walking trials open with a
    long standing pause BEFORE the cameras start, so their static regime is real and large but
    invisible to any metric needing a rotation — which is why the reference-free metrics exist
    and why every mocap-referenced static number should be read with this panel beside it.
    """
    if sensor_df.empty:
        print("No sensor stats, skipping coverage figure.")
        return
    static = sensor_df[sensor_df['regime'] == 'static']
    body = sensor_df[sensor_df['regime'] == 'body_static']
    if static.empty:
        print("No static stretches detected, skipping coverage figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, figure_height(len(segments), per_row=0.5)),
                             sharey=True)
    for regime_index, (frame, regime) in enumerate(((static, 'static'), (body, 'body_static'))):
        if frame.empty:
            continue
        coverage = frame.groupby('segment', observed=True)['coverage']
        present = [s for s in segments if s in coverage.groups]
        axes[0].barh([segments.index(s) - 0.2 + 0.4 * regime_index for s in present],
                     [100 * coverage.get_group(s).mean() for s in present],
                     xerr=[100 * coverage.get_group(s).std(ddof=0) for s in present],
                     height=0.36, color=REGIME_COLORS[regime], alpha=0.85,
                     error_kw={'elinewidth': 1.2, 'ecolor': '#444444'},
                     label=REGIME_LABELS[regime], zorder=3)
    axes[0].set_xlabel('Share of the recording (%)')
    axes[0].set_title('Time detected as still', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10, loc='lower right')

    covered = static.groupby('segment', observed=True).apply(
        lambda g: 100 * g['n_valid'].sum() / max(g['n_samples'].sum(), 1), include_groups=False)
    present = [s for s in segments if s in covered.index]
    axes[1].barh([segments.index(s) for s in present], [covered[s] for s in present],
                 height=0.6, color='#6b7280', alpha=0.85, zorder=3)
    axes[1].axvline(100, color='black', linestyle='--', linewidth=1.2)
    axes[1].set_xlabel('Static samples inside the mocap window (%)')
    axes[1].set_title('...of which mocap can be referenced', fontsize=13, fontweight='bold')

    for ax in axes:
        ax.grid(axis='y', visible=False)
        sns.despine(ax=ax)
    axes[0].set_yticks(range(len(segments)))
    axes[0].set_yticklabels(segments)
    axes[0].set_ylim(-0.6, len(segments) - 0.4)
    axes[0].invert_yaxis()

    plot_utils.finalize_and_save_plot(
        fig, f"Static coverage per sensor\n{caption}", "static_coverage.png", plots_dir(dataset),
        epilog=wrap_epilog("bars are the mean over trials, whiskers the between-trial sd",
                           STATIC_EPILOG),
        save=save, show=show)


def plot_detector_validation(sensor_df: pd.DataFrame, segments: List[str], dataset: str,
                             caption: str, save: bool = True, show: bool = False) -> None:
    """What the MOCAP says each sensor was doing over the stretches the GYRO called static.

    The static detector reads only the gyroscope, so that it cannot define the accelerometer's
    and magnetometer's answers into existence. Its blind spot is pure translation, and this is
    the independent check on it: if these speeds are not small, the static regime is not static
    and every number computed over it is wrong.
    """
    static = sensor_df[(sensor_df['regime'] == 'static')
                       & sensor_df['mocap_speed_median_mm_s'].notna()]
    if static.empty:
        print("No static samples with mocap; skipping detector-validation figure.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, figure_height(len(segments), per_row=0.5)),
                             sharey=True)
    panels = ((axes[0], 'mocap_speed_median_mm_s', 'mocap_speed_p95_mm_s',
               'Linear speed (mm/s)'),
              (axes[1], 'mocap_angular_speed_median_deg_s', 'mocap_angular_speed_p95_deg_s',
               'Angular speed (deg/s)'))
    for ax, median_col, p95_col, xlabel in panels:
        present = [s for s in segments if s in set(static['segment'])]
        grouped = static.groupby('segment', observed=True)
        ax.barh([segments.index(s) for s in present],
                [grouped.get_group(s)[median_col].median() for s in present],
                height=0.6, color=REGIME_COLORS['static'], alpha=0.8, label='median', zorder=3)
        ax.scatter([grouped.get_group(s)[p95_col].median() for s in present],
                   [segments.index(s) for s in present], marker='|', s=180, color='black',
                   linewidths=2, label='p95', zorder=4)
        ax.set_xlabel(xlabel)
        ax.grid(axis='y', visible=False)
        ax.legend(fontsize=10, loc='lower right')
        sns.despine(ax=ax)
    axes[0].set_yticks(range(len(segments)))
    axes[0].set_yticklabels(segments)
    axes[0].set_ylim(-0.6, len(segments) - 0.4)
    axes[0].invert_yaxis()

    plot_utils.finalize_and_save_plot(
        fig, f"Static-detector validation: what mocap saw during detected-static samples\n{caption}",
        "static_detector_validation.png", plots_dir(dataset),
        epilog="median over sensor-trials of each trial's own median and p95; the detector never "
               "consults mocap, so this is an independent check",
        save=save, show=show)

# ==============================================================================
# Observability
# ==============================================================================

# ==============================================================================
# Cross-dataset summary — the one figure
# ==============================================================================
# Every other figure in this module is scoped to one dataset, because every other figure is
# about a placement question that only that dataset's sensor list can answer. This one is the
# opposite: it asks the paper's headline question — how far are the two global sensor
# assumptions from true — and the only honest answer to that is one that survives a change of
# lab, protocol, subject pool and mocap system. So it pools every dataset onto one anatomical
# axis, and keeps them visually separate so that a claim resting on one of them is visible as
# such.
#
# It reads the pooled STATISTICS parquet, not the per-sample tables. That file already carries
# per-subject pooled quantiles (`trial == 'all'`, a real subject), which is exactly the unit
# this figure wants and is a few thousand rows instead of ~36 M.


def placement_class(segment: str) -> Optional[str]:
    """A dataset's own segment name mapped onto the shared anatomical axis.

    The three datasets name the same four places three different ways — Al Borno uses the
    OpenSim bone names (Femur/Tibia/Calcn), IMoVE uses limb names with a side and a height
    (Thigh R High), the biplane build uses limb names with a side — and none of that is a
    difference in where the sensor was. Mapping on the FIRST token is enough to collapse all
    three, and returning None for anything unrecognised means a new dataset shows up as an
    absent row rather than as a silently mislabeled one.

    Left and right, and IMoVE's three heights per segment, all fold into one class here. That
    is a deliberate loss: the placement figures elsewhere in this module are where those are
    resolved, and carrying them onto this axis would put 15 rows against Al Borno's 5 and turn
    a cross-dataset comparison into a within-IMoVE one."""
    return PLACEMENT_OF_TOKEN.get(segment.split()[0])


PLACEMENT_OF_TOKEN = {
    'Torso': 'Torso', 'Pelvis': 'Pelvis',
    'Femur': 'Thigh', 'Thigh': 'Thigh',
    'Tibia': 'Shank', 'Shank': 'Shank',
    'Calcn': 'Foot', 'Foot': 'Foot',
}
# Proximal to distal. Not alphabetical and not cosmetic: the finding this figure exists to show
# is that both departures grow monotonically down the limb, and that reads as a trend only on an
# anatomically ordered axis.
PLACEMENT_ORDER = ('Torso', 'Pelvis', 'Thigh', 'Shank', 'Foot')

# The datasets whose MOCAP WORLD FRAME has been checked to agree with the pipeline's
# EXPECTED_GRAVITY, which is Y-up. That check is a precondition for `linacc` meaning anything:
# it is |a_world - g| with a literal constant g, so a world frame in a different convention
# reports the frame mismatch instead of the acceleration.
#
#   alborno              median world-frame acc [ 0.016, 9.812, -0.003]   Y-up, agrees
#   imove                median world-frame acc [ 0.072, 9.810, -0.006]   Y-up, agrees
#   imove_biplane        median world-frame acc [-0.702,-0.997,  9.554]   Z-UP, DOES NOT AGREE
#   imove_biplane_vicon  median world-frame acc [-0.047,-0.050,  9.757]   Z-UP, DOES NOT AGREE
#
# The two biplane specs are therefore NOT in this list. Their linacc medians come out at
# 13.9-18.4 m/s^2, which is not a drop landing being violent — it is sqrt(2)*9.81 = 13.87, the
# length of the difference between two gravity vectors 90 degrees apart, and it is present in
# their quasi-static samples too (their reference-free ||a|-|g|| over the same frames is 0.035
# m/s^2, i.e. the accelerometer is reading a clean 1 g the whole time). Including them here
# would put a coordinate-convention artifact on the paper's headline axis.
#
# The fix belongs in the build layer — src/toolchest/building/biplane.py emits the source
# frame unrotated and nothing downstream converts it — after which this list should grow and
# this comment should shrink to a note that it was once wrong.
FRAME_VERIFIED_DATASETS = ('alborno', 'imove')

# Colour and marker both carry the dataset, so the figure survives greyscale printing and
# colour-vision deficiency. Palette is Okabe-Ito, which is designed for the latter.
DATASET_STYLE = {
    'alborno': {'label': 'Al Borno (Xsens, lab)', 'color': '#0072B2', 'marker': 'o'},
    'imove': {'label': 'IMoVE (Xsens, lab)', 'color': '#D55E00', 'marker': 's'},
    'imove_biplane': {'label': 'IMoVE biplane (BioStamp, bone pose)',
                      'color': '#009E73', 'marker': '^'},
    'imove_biplane_vicon': {'label': 'IMoVE biplane (BioStamp, marker cluster)',
                            'color': '#CC79A7', 'marker': 'D'},
}

# (metric, column heading, axis label). One mocap-referenced metric per sensor: these are the
# two quantities the assumptions are literally about, and their reference-free twins understate
# both (see the module docstring).
SUMMARY_COLUMNS = (
    ('linacc', 'ACCELEROMETER',
     'Departure from gravity   $|a_{world} - g|$   (m/s²)'),
    ('magdev', 'MAGNETOMETER',
     f'Departure from the global field   $|m_{{world}} - m_{{global}}|$   ({MAG_UNIT})'),
)
SUMMARY_ROWS = (('static', 'sensor standing still'), ('nonstatic', 'sensor moving'))

# Physical anchors drawn behind the data, per column. The tuned std is what every filter in
# this repo is told the measurement noise is; the Earth field is what the magnetic departure is
# a fraction of. Neither is derived from this figure's data — both come from `pipeline_constants`
# and from the unit convention in MAG_UNIT — so they are the fixed rulers the data is read
# against rather than another series.
def summary_reference_lines(metric: str) -> List[Tuple[float, str]]:
    constants = pipeline_constants()
    if metric == 'linacc':
        return [(constants['acc_std'], f"filter's assumed acc noise ({constants['acc_std']:g})")]
    return [(constants['mag_std'], f"filter's assumed mag noise ({constants['mag_std']:g})"),
            (1.0, 'the whole Earth field')]


SUMMARY_DODGE = 0.30    # total vertical span the datasets of one placement are spread over
SUMMARY_JITTER = 0.055  # deterministic spread of the per-subject dots around their dataset


def load_subject_summary(datasets: Sequence[str]) -> pd.DataFrame:
    """One row per (dataset, subject, regime, metric, placement): that subject's median.

    Built from the per-subject pooled rows of each dataset's statistics parquet — the ones with
    `trial == 'all'` and a real subject id. Those are quantiles over the subject's POOLED
    samples, so a subject who did more trials is described by more data rather than by more
    rows, and the two synthetic aggregate rows the experiment also writes (`subject == 'all'`,
    and the per-trial rows) are dropped so nothing is counted twice.

    THE SUBJECT IS THE UNIT. Everything above this line is sample-pooled and everything below
    it is a rank statistic over subjects, which is what makes the spread on this figure a
    between-subject spread — the one a reader deciding whether these numbers transfer to their
    own lab actually wants. Pooling samples across subjects instead would give a far tighter
    interval that says only that the datasets are large.

    A dataset whose parquet is missing is skipped with a warning rather than raising: the four
    specs are run separately and a partially-run tree is a normal state.
    """
    frames = []
    for dataset in datasets:
        # The experiment's own path function, not a second copy of its naming convention: the
        # two would drift the first time the statistics file is renamed.
        path = statistics_path(dataset)
        if not path.exists():
            print(f"  {dataset}: no statistics parquet at {path}, skipping.")
            continue
        df = pd.read_parquet(path, engine='pyarrow')
        df = df[(df['group_kind'] == 'segment') & (df['trial'] == 'all')
                & (df['subject'] != 'all')].copy()
        df['dataset'] = dataset
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out['placement'] = out['group'].map(placement_class)
    out = out[out['placement'].notna() & out['p50'].notna()]
    # Median over the placement's segments WITHIN a subject, so a subject contributes one
    # number per placement however many sensors they wore there. Without this, IMoVE's six
    # thigh sensors would outvote Al Borno's two and the "spread" of a placement would be
    # partly a count of sensors.
    return (out.groupby(['dataset', 'subject', 'regime', 'metric', 'placement'],
                        observed=True)['p50'].median().reset_index())


def summarize_across_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR of the per-subject values, per (dataset, regime, metric, placement)."""
    grouped = df.groupby(['dataset', 'regime', 'metric', 'placement'], observed=True)['p50']
    return grouped.agg(median='median', lo=lambda s: s.quantile(0.25),
                       hi=lambda s: s.quantile(0.75), n='size').reset_index()


def plot_assumption_summary(datasets: Sequence[str] = FRAME_VERIFIED_DATASETS,
                            save: bool = True, show: bool = False) -> None:
    """THE figure: both assumptions, both regimes, every dataset, one anatomical axis.

    Four panels on a 2x2 grid. Columns are the two assumptions and rows are the two regimes,
    with the x-axis SHARED DOWN EACH COLUMN — that sharing is the whole construction, because
    the claim is a comparison between the rows and a comparison between rows drawn on
    independent axes is not a comparison at all. Read it as a 2x2 of one question:

        accelerometer, still    0.09-0.75 m/s^2, within an order of magnitude of the noise floor
        accelerometer, moving   1.4-3.2 m/s^2 — 3x to 29x worse, depending on segment and dataset
        magnetometer,  still    0.05-0.67, i.e. up to two thirds of the entire Earth field
        magnetometer,  moving   0.065-0.65 — the SAME, to within 25% either way

    The asymmetry between the columns is the finding. Gravity is recoverable by waiting for the
    sensor to be still; magnetic distortion is a property of WHERE the sensor is, so standing
    still buys nothing. Every dataset here reproduces it independently, which is what the
    per-dataset colours and markers are for — a reader can check that no single lab, protocol
    or mocap system is carrying the claim.

    The two columns also differ in how they vary with PLACEMENT, and only one of them does what
    the limb ordering suggests. The magnetic departure is strictly monotonic proximal-to-distal
    in both datasets and both regimes — torso 0.073 to foot 0.671 in Al Borno, a factor of nine
    down one body. The accelerometer's is not: it is comparable at every placement and peaks at
    the SHANK rather than the foot while moving, because a foot spends half of every gait cycle
    in stance. Do not read the left column as a proximal-to-distal trend.

    Each dataset draws three things per placement: one faint dot per SUBJECT, a filled marker at
    the median over subjects, and a bar over their interquartile range.
    """
    per_subject = load_subject_summary(datasets)
    if per_subject.empty:
        print("No cross-dataset statistics found; run experiments.global_assumptions first.")
        return
    summary = summarize_across_subjects(per_subject)
    present_datasets = [d for d in datasets if d in set(summary['dataset'])]
    # Only the placements someone actually wore a sensor on. Al Borno is the only dataset with a
    # torso, so on a selection without it that row would otherwise be an empty labelled tick,
    # which reads as a measurement of nothing rather than as an absence of sensors.
    placements = [p for p in PLACEMENT_ORDER if p in set(summary['placement'])]
    # A fixed dodge per dataset across every panel, computed once from the full selection rather
    # than per panel from what that panel has. A dataset that keeps its vertical position even
    # where a neighbour is missing is traceable by eye down the column; one that slides to
    # re-centre is not.
    offsets = ({d: 0.0 for d in present_datasets} if len(present_datasets) == 1 else
               dict(zip(present_datasets,
                        np.linspace(-SUMMARY_DODGE / 2, SUMMARY_DODGE / 2, len(present_datasets)))))

    fig, axes = plt.subplots(len(SUMMARY_ROWS), len(SUMMARY_COLUMNS),
                             figsize=(13.5, 1.5 * len(placements) + 2.0),
                             sharex='col', sharey=True, squeeze=False)

    for row, (regime, regime_label) in enumerate(SUMMARY_ROWS):
        for col, (metric, heading, xlabel) in enumerate(SUMMARY_COLUMNS):
            ax = axes[row][col]
            _draw_summary_panel(ax, per_subject, summary, metric, regime, placements, offsets)
            ax.set_xscale('log')
            for value, note in summary_reference_lines(metric):
                ax.axvline(value, color='#4a4a4a', linestyle=(0, (4, 3)), linewidth=1.1,
                           zorder=1)
                # Labeled in the TOP row only. The x-axis is shared down the column, so the line
                # is at the same place in both rows and a second copy of its label is two more
                # rotated strings competing with the data for no added information. It rides in
                # the headroom `set_ylim` reserves above the first placement, in axes coordinates
                # so it stays there whatever the data does. A legend entry for a ruler would
                # compete with the dataset legend, which is the one a reader needs.
                if row == 0:
                    ax.text(value, 0.995, f" {note}", transform=ax.get_xaxis_transform(),
                            rotation=90, ha='right', va='top', fontsize=8.5, fontstyle='italic',
                            color='#4a4a4a', zorder=5)
            if row == 0:
                ax.set_title(heading, fontsize=15, fontweight='bold', pad=12,
                             color='#222222')
            if row == len(SUMMARY_ROWS) - 1:
                ax.set_xlabel(xlabel, fontsize=12.5)
            if col == len(SUMMARY_COLUMNS) - 1:
                # The regime named on the right-hand edge, in the colour this module uses for it
                # everywhere else, so the two rows are identifiable without reading a legend.
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(regime_label.upper(), fontsize=12.5, fontweight='bold',
                              color=REGIME_COLORS[regime], rotation=270, labelpad=22)
            sns.despine(ax=ax)

    axes[0][0].set_yticks(range(len(placements)))
    axes[0][0].set_yticklabels(placements, fontsize=13)
    # Inverted, so proximal is at the top and the axis reads down the body. The extra headroom
    # above the first placement is where the reference-line labels sit; without it they overlap
    # the topmost row's data, which on the magnetometer panel is exactly where the torso sits.
    axes[0][0].set_ylim(len(placements) - 0.5, -1.0)

    handles = [Line2D([0], [0], marker=DATASET_STYLE[d]['marker'], linestyle='none',
                      markersize=9, color=DATASET_STYLE[d]['color'],
                      markeredgecolor='white', markeredgewidth=0.8,
                      label=f"{DATASET_STYLE[d]['label']}  "
                            f"(n={per_subject.loc[per_subject['dataset'] == d, 'subject'].nunique()})")
               for d in present_datasets]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 3), fontsize=11.5,
               bbox_to_anchor=(0.5, -0.055), frameon=False)

    missing_mag = [DATASET_STYLE[d]['label'] for d in present_datasets
                   if not DATASETS[d].has_magnetometer]
    if missing_mag:
        axes[0][1].text(0.5, 0.5, "no magnetometer:\n" + "\n".join(missing_mag),
                        transform=axes[0][1].transAxes, ha='center', va='center',
                        fontsize=10, fontstyle='italic', color='#8896a6')

    trials = ", ".join(f"{DATASET_STYLE[d]['label'].split(' (')[0]} "
                       f"{per_subject.loc[per_subject['dataset'] == d, 'subject'].nunique()} subj"
                       for d in present_datasets)
    plot_utils.finalize_and_save_plot(
        fig,
        "How far the two global sensor assumptions are from true",
        "assumption_summary.png", paths.plots_dir("global_assumptions"),
        caption=(
            "Departure from each of the two assumptions every orientation filter in this work "
            "rests on — that the accelerometer reads gravity (left) and that the magnetometer "
            "reads one constant field everywhere on the body (right) — by where on the body the "
            "sensor sat, split by whether that sensor was standing still (top) or moving "
            "(bottom). Faint marks are individual subjects, the filled marker is the median over "
            "subjects and the bar their interquartile range; the x-axis is logarithmic and is "
            "shared down each column, so the top and bottom panels of a column are directly "
            "comparable. Dashed rulers are fixed references, not measurements: the measurement "
            "noise each filter is tuned to assume, and the magnitude of the Earth's field "
            "itself. READ THE CONTRAST BETWEEN THE COLUMNS. Standing still recovers gravity: the "
            "accelerometer's departure falls by 3x to 29x, from 1.4-3.2 m/s² moving to "
            "0.09-0.75 m/s² still. It does not recover the magnetic field, whose departure is "
            "the same in both rows to within 25% either way and reaches two thirds of the entire "
            "Earth field at the foot. Magnetic distortion is a property of where the sensor is, "
            "not of what it is doing, so a filter cannot wait it out. The two columns also vary "
            "differently with placement: the magnetic departure rises monotonically "
            "proximal-to-distal in both datasets and both regimes — a factor of nine from torso "
            "to foot — while the accelerometer's is comparable at every placement and peaks at "
            "the shank rather than the foot, so the left column should not be read as a "
            "proximal-to-distal trend. Both effects appear independently in each dataset, which "
            "is what the separate colours and markers are there to let you check."),
        epilog=wrap_epilog(f"{trials}; one dot per subject, over that subject's pooled samples",
                           STATIC_EPILOG),
        save=save, show=show)


def _draw_summary_panel(ax, per_subject: pd.DataFrame, summary: pd.DataFrame, metric: str,
                        regime: str, placements: List[str], offsets: Dict[str, float]) -> None:
    """One panel of `plot_assumption_summary`: subject dots, median markers and IQR bars."""
    for index, placement in enumerate(placements):
        # Alternating bands rather than gridlines between the rows. On a log axis the vertical
        # gridlines are already dense, and a second set of horizontal ones turns the panel into
        # graph paper; a band ties a row's four dodged series together at a glance.
        if index % 2:
            ax.axhspan(index - 0.5, index + 0.5, color='#f4f5f7', zorder=0)

    subject_rows = per_subject[(per_subject['metric'] == metric)
                               & (per_subject['regime'] == regime)]
    summary_rows = summary[(summary['metric'] == metric) & (summary['regime'] == regime)]
    for dataset, offset in offsets.items():
        style = DATASET_STYLE[dataset]
        block = subject_rows[subject_rows['dataset'] == dataset]
        for index, placement in enumerate(placements):
            values = block.loc[block['placement'] == placement, 'p50'].to_numpy()
            if not len(values):
                continue
            # Deterministic, evenly spaced jitter rather than a random one. These figures are
            # regenerated for every draft and a random jitter makes every regeneration a visual
            # diff; spreading the subjects evenly also stops two of them landing on the same
            # pixel, which random jitter does not.
            spread = (np.linspace(-1, 1, len(values)) if len(values) > 1 else np.zeros(1))
            ax.scatter(values, index + offset + SUMMARY_JITTER * spread, s=13,
                       color=style['color'], alpha=0.32, linewidths=0, zorder=2)
        bands = summary_rows[(summary_rows['dataset'] == dataset)
                             & summary_rows['placement'].isin(placements)]
        if bands.empty:
            continue
        y = np.array([placements.index(p) + offset for p in bands['placement']])
        median = bands['median'].to_numpy()
        ax.errorbar(median, y,
                    xerr=np.vstack([median - bands['lo'].to_numpy(),
                                    bands['hi'].to_numpy() - median]),
                    fmt=style['marker'], markersize=8.5, color=style['color'],
                    ecolor=style['color'], elinewidth=2.4, capsize=0, alpha=0.95,
                    markeredgecolor='white', markeredgewidth=0.9, zorder=4)
    ax.grid(axis='x', which='major', color='#e3e5e9', linewidth=0.8, zorder=0)
    ax.grid(axis='y', visible=False)
    ax.set_axisbelow(True)

# ==============================================================================
# Per-subject and per-trial splits
# ==============================================================================

def plot_subject_heatmap(segment_df: pd.DataFrame, segments: List[str], dataset: str,
                         caption: str, save: bool = True, show: bool = False) -> None:
    """Subject x segment median, one panel per metric — the between-subject spread a pooled
    median hides.

    Each panel is normalized to its own colour scale because the metrics do not share units;
    the numbers are printed in the cells so the panels stay readable as tables as well as
    heatmaps.
    """
    metrics = [m for m in SEGMENT_METRICS if m in segment_df.columns]
    if not metrics or segment_df.empty:
        print("No segment samples, skipping per-subject heatmap.")
        return
    subjects = sorted(segment_df['subject'].astype(str).unique())
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(4.2 * len(metrics), figure_height(len(subjects), 0.42, 2.0)),
                             sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        table = (segment_df.assign(subject=segment_df['subject'].astype(str))
                 .groupby(['subject', 'segment'], observed=True)[metric].median().unstack())
        table = table.reindex(index=subjects, columns=[s for s in segments if s in table.columns])
        if table.empty or table.isna().all().all():
            ax.set_visible(False)
            continue
        sns.heatmap(table, ax=ax, cmap='viridis', annot=len(subjects) * len(table.columns) <= 200,
                    fmt='.2f', annot_kws={'fontsize': 7}, cbar_kws={'shrink': 0.6},
                    linewidths=0.4, linecolor='white')
        ax.set_title(f"{metric}\n({METRIC_UNITS[metric]})", fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=90, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

    plot_utils.finalize_and_save_plot(
        fig, f"Median per subject and segment\n{caption}", "per_subject_heatmap.png",
        plots_dir(dataset), epilog="median over every sample that subject has for that segment, "
                                   "all regimes pooled",
        save=save, show=show)


def plot_trial_spread(segment_df: pd.DataFrame, segments: List[str], dataset: str, caption: str,
                      save: bool = True, show: bool = False) -> None:
    """One point per trial per segment: the trial's median, static and moving side by side.

    A pooled quantile cannot show whether a metric is consistent across recordings or driven by
    a handful of them. This can, and on IMoVE — where a single session mixes a 30 s static pose
    with a 2800 s walk — that distinction is the difference between a property of the sensor
    and a property of the task.
    """
    metrics = [m for m in ('linacc', 'magdev') if m in segment_df.columns]
    if not metrics or segment_df.empty:
        print("No segment samples, skipping per-trial spread.")
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics),
                                                       figure_height(len(segments))), sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for regime in CONTRAST_REGIMES:
            frame = regime_frames(segment_df)[regime]
            if frame.empty:
                continue
            per_trial = (frame.groupby(['subject', 'trial', 'segment'], observed=True)[metric]
                         .median().reset_index().dropna(subset=[metric]))
            if per_trial.empty:
                continue
            present = per_trial[per_trial['segment'].isin(segments)]
            jitter = np.random.default_rng(0).uniform(-0.16, 0.16, len(present))
            offset = -0.18 if regime == 'static' else 0.18
            ax.scatter(present[metric].to_numpy(),
                       [segments.index(s) for s in present['segment']] + jitter + offset,
                       s=18, alpha=0.55, color=REGIME_COLORS[regime], linewidths=0,
                       label=REGIME_LABELS[regime])
        ax.set_xscale('log')
        ax.set_xlabel(METRIC_LABELS.get(metric, metric))
        ax.grid(axis='y', visible=False)
        sns.despine(ax=ax)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(segments)))
    first.set_yticklabels(segments)
    first.set_ylim(-0.7, len(segments) - 0.3)
    first.invert_yaxis()
    first.legend(fontsize=10, loc='lower right')

    plot_utils.finalize_and_save_plot(
        fig, f"Per-trial medians\n{caption}", "per_trial_spread.png", plots_dir(dataset),
        epilog="one point per trial per segment; trials with no valid mocap sample in a regime "
               "contribute no point there",
        save=save, show=show)

# ==============================================================================
# Time series
# ==============================================================================

def add_fading_span(ax: plt.Axes, x0: float, x1: float, color: str, ylim: Tuple[float, float],
                    max_alpha: float = 0.28, fade: float = 1.0, n: int = 300) -> None:
    """Shades [x0, x1] with alpha ramping 0 -> max_alpha -> 0 across a `fade`-second margin at
    each edge, rather than as a hard-edged axvspan. The edges are what the fade is honest
    about: these boundaries come from a threshold on a rolling statistic, so they are accurate
    to about a window, and a crisp rectangle would assert a precision the detector lacks."""
    xs = np.linspace(x0 - fade, x1 + fade, n)
    ramp_in = (xs - (x0 - fade)) / fade
    ramp_out = ((x1 + fade) - xs) / fade
    rgba = np.zeros((1, n, 4))
    rgba[0, :, :3] = mcolors.to_rgb(color)
    rgba[0, :, 3] = np.clip(np.minimum(ramp_in, ramp_out), 0, 1) * max_alpha
    ax.imshow(rgba, extent=(x0 - fade, x1 + fade, ylim[0], ylim[1]), aspect='auto', zorder=0,
              interpolation='bilinear')


def shade_intervals(ax: plt.Axes, regions: Sequence[Tuple[float, float, str]]) -> List[Patch]:
    """Draws every (start, end, label) region as a fading span, holding the data limits fixed
    (imshow would otherwise expand them), and returns one legend patch per label."""
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


def majority_static_regions(intervals: pd.DataFrame, n_sensors: int,
                            fraction: float = MAJORITY_STATIC_FRACTION
                            ) -> List[Tuple[float, float]]:
    """Absolute-time spans where MORE THAN `fraction` of the trial's sensors are individually
    static, from the per-sensor `static` rows of the intervals table.

    Swept as +1/-1 events rather than rasterized onto a grid: the intervals table is already
    exact in seconds, so counting occupancy at the event boundaries avoids inventing a sample
    rate here and avoids off-by-one banding at the edges. Openings are ordered before closings
    at an identical timestamp, so two intervals that merely touch merge instead of producing a
    zero-width gap.

    `n_sensors` is passed in rather than counted from these rows, because a sensor that was
    never still contributes no rows at all and would otherwise silently shrink the denominator
    — which is exactly the sensor whose dissent the threshold is meant to tolerate.
    """
    rows = intervals[(intervals['label'] == 'static') & (intervals['sensor'].astype(str) != '')]
    if rows.empty or n_sensors <= 0:
        return []
    needed = int(np.floor(fraction * n_sensors)) + 1  # strictly more than the fraction
    events = [(float(t), +1) for t in rows['start_time']]
    events += [(float(t), -1) for t in rows['end_time']]
    events.sort(key=lambda e: (e[0], -e[1]))

    regions: List[Tuple[float, float]] = []
    count, start = 0, None
    for time, delta in events:
        was_open = count >= needed
        count += delta
        if count >= needed and not was_open:
            start = time
        elif was_open and count < needed and start is not None:
            regions.append((start, time))
            start = None
    return regions


def mocap_valid_spans(segment_df: pd.DataFrame, min_gap_s: float = 1.0
                      ) -> List[Tuple[float, float]]:
    """Contiguous absolute-time spans in which the mocap-referenced metrics exist at all.

    Both panels of the time series plot `linacc` and `magdev`, which are NaN outside `valid`
    — so a window chosen without consulting this draws two empty axes. That is not a corner
    case on the IMoVE long walks: they merge three mocap takes onto one 4537 s inertial
    record, leaving 42% validity in four blocks with ~1100 s gaps between them.
    """
    if 'valid' not in segment_df.columns or segment_df.empty:
        return []
    times = np.unique(segment_df.loc[segment_df['valid'].to_numpy(), 'timestamp'].to_numpy())
    if times.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(times) > min_gap_s)
    return list(zip(np.r_[times[0], times[breaks + 1]], np.r_[times[breaks], times[-1]]))


def select_static_window(intervals: pd.DataFrame, n_sensors: int,
                         valid_spans: Optional[List[Tuple[float, float]]] = None,
                         fraction: float = MAJORITY_STATIC_FRACTION
                         ) -> Optional[Tuple[float, float, float, List[Tuple[float, float, str]]]]:
    """Picks the median-ranked MAJORITY-static bout and returns (onset, t0, t1, regions), the
    latter three in seconds RELATIVE to that onset.

    Median rather than longest: the longest bout is the least representative one, and the
    figure is making a claim about what going still does generally.

    EVERY qualifying region inside the window is returned, not only the chosen bout. Shading
    just the one leaves neighbouring still time bare, which reads as "the detector missed
    this" when it is really "the figure only drew one of them" — and that misreading is the
    reason this function changed.

    Only bouts of at least BODY_STATIC_MIN_S are eligible to be CENTRED on, though every bout
    is still shaded. Relaxing unanimity to a majority multiplies the region count — 42 on
    alborno 07/complexTasks against a handful of whole-body bouts — and most of the extra ones
    are sub-second, so a plain median lands on a 0.8 s sliver and the figure is centred
    wherever that sliver happens to fall. Reusing the analysis's own minimum keeps this from
    being one more tunable.
    """
    regions = majority_static_regions(intervals, n_sensors, fraction)
    regions = [(s, e) for s, e in regions if e > s]
    if not regions:
        return None
    substantial = [r for r in regions if (r[1] - r[0]) >= BODY_STATIC_MIN_S] or regions

    # Prefer a bout whose whole padded window has mocap, fall back to one that merely overlaps
    # it, and only then to any bout at all. Three tiers rather than a hard filter because a
    # trial with no mocap-valid still stretch should still draw SOMETHING recognisable rather
    # than vanish -- but it should never prefer an empty window when a populated one exists.
    ranked = sorted(substantial, key=lambda r: r[1] - r[0])
    if valid_spans:
        def covered(bout, pad):
            return any(bout[0] - pad >= lo and bout[1] + pad <= hi for lo, hi in valid_spans)
        padded = [r for r in ranked if covered(r, STATIC_PAD_S)]
        overlapping = [r for r in ranked if covered(r, 0.0)]
        chosen_from = padded or overlapping or ranked
        if not padded:
            print(f"  no majority-static bout has {STATIC_PAD_S:.0f}s of mocap either side; "
                  f"falling back to {'bout-only' if overlapping else 'no'} mocap coverage")
        ranked = chosen_from
    onset, end = ranked[len(ranked) // 2]
    t0, t1 = -STATIC_PAD_S, (end - onset) + STATIC_PAD_S
    print(f"Using majority-static bout {len(ranked) // 2 + 1} of {len(ranked)} eligible "
          f"({len(regions)} shaded, >{fraction:.0%} of {n_sensors} sensors): "
          f"t={onset:.1f}-{end:.1f}s ({end - onset:.1f}s)")

    # Clip every region into the window, in coordinates relative to the chosen bout's onset.
    shaded = []
    for start, stop in sorted(regions):
        lo, hi = max(start - onset, t0), min(stop - onset, t1)
        if hi > lo:
            shaded.append((lo, hi, MAJORITY_STATIC_LABEL))
    return onset, t0, t1, shaded


def plot_static_timeseries(dataset: str, subject: str, trial: str, segments: List[str],
                           save: bool = True, show: bool = False) -> None:
    """The two assumptions through one going-still bout — the case where they decouple.

    Going still collapses the accelerometer's departure while leaving magnetic distortion where
    it was, and that is worth seeing as a time series rather than only as two boxes: the
    accelerometer trace drops off a cliff at the bout boundary and the magnetometer trace walks
    straight through it.

    Shaded on MAJORITY_STATIC_FRACTION, not on the `body_static` regime the tables report; see
    that constant for why the unanimous rule is right for the statistics and wrong for this
    picture.
    """
    row_keys = [(subject, trial)]
    segment_df = load_trial_table(dataset, 'segment_samples', row_keys=row_keys)
    intervals = load_trial_table(dataset, 'intervals', row_keys=row_keys)
    if segment_df.empty or intervals.empty:
        print(f"No per-trial tables for {dataset}/{subject}/{trial}; skipping time series.")
        return
    # The trial's own sensor count, not the dataset's: a trial that dropped a sensor should be
    # judged a majority of what it actually recorded.
    n_sensors = int(segment_df['sensor'].nunique())
    window = select_static_window(intervals, n_sensors, mocap_valid_spans(segment_df))
    if window is None:
        print(f"No majority-static bout in {dataset}/{subject}/{trial}; skipping time series.")
        return
    onset, t0, t1, regions = window

    def in_window(df: pd.DataFrame) -> pd.DataFrame:
        shifted = df.assign(timestamp=df['timestamp'] - onset)
        return shifted[(shifted['timestamp'] >= t0) & (shifted['timestamp'] <= t1)]

    seg_window = in_window(segment_df)
    # Require the metric to be POPULATED here, not merely present as a column. Both panels are
    # mocap-referenced and so are NaN outside `valid`; on the Al Borno walking trials the
    # subject stood for ~430 s before the cameras rolled, so every still stretch they have
    # falls outside the mocap window and no choice of bout can put data on these axes. Without
    # this the figure still saved -- two empty panels autoscaled to +-0.05 -- which is worse
    # than not drawing it, because a blank plot looks like a finding.
    panels = [(seg_window, 'segment', metric, segments, METRIC_LABELS[metric])
              for metric in ('linacc', 'magdev')
              if metric in seg_window.columns and seg_window[metric].notna().any()]
    if not panels:
        print(f"No mocap-referenced samples in the chosen window for {dataset}/{subject}/"
              f"{trial}; its static stretches lie outside the mocap window. Skipping the "
              f"time series (the reference-free metrics still cover this trial elsewhere).")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 3.4 * len(panels)), sharex=True)
    for ax, (frame, group_col, metric, order, ylabel) in zip(np.atleast_1d(axes), panels):
        present = [g for g in order if g in set(frame[group_col])]
        colors = color_map(order)
        # Reverse order so the most proximal (lowest-amplitude) trace ends up on top rather
        # than buried under a distal one; the legend is rebuilt in the original order.
        for name in reversed(present):
            trace = frame[frame[group_col] == name].sort_values('timestamp')
            ax.plot(trace['timestamp'].to_numpy(), trace[metric].to_numpy(),
                    color=colors[name], linewidth=1.3, alpha=0.9, label=name)
        ax.set_ylabel(ylabel, fontsize=12)
        handles = {line.get_label(): line for line in ax.get_lines()}
        for patch in shade_intervals(ax, regions):
            handles[patch.get_label()] = patch
        # De-duplicated: every shaded span now carries the same label, so the raw
        # present + regions list would repeat it once per region.
        entries = list(dict.fromkeys(present + [r[2] for r in regions]))
        ax.legend([handles[n] for n in entries if n in handles],
                  [n for n in entries if n in handles],
                  bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
        sns.despine(ax=ax)
    np.atleast_1d(axes)[-1].set_xlabel('Time relative to the start of the static bout (s)')

    plot_utils.finalize_and_save_plot(
        fig, f"Through a going-still bout — {dataset} {subject}/{trial}",
        f"timeseries_static_{subject}_{trial}.png", plots_dir(dataset),
        epilog=wrap_epilog(MAJORITY_STATIC_EPILOG.format(n=n_sensors), STATIC_EPILOG),
        save=save, show=show)

# ==============================================================================
# Diagnostics
# ==============================================================================

def plot_static_diagnostic(dataset: str, subject: str, trial: str, spec,
                           save: bool = True, show: bool = False) -> None:
    """Per-trial diagnostic for the static detector: each sensor's gyro magnitude with its
    detected static stretches shaded, plus the whole-body mask underneath.

    This is how a detector failure becomes visible rather than silently shifting a
    distribution — shading that plainly disagrees with the quiet stretch underneath it.
    """
    row_keys = [(subject, trial)]
    segment_df = load_trial_table(dataset, 'segment_samples', row_keys=row_keys,
                                  columns=['timestamp', 'segment', 'gyro_norm', 'static',
                                           'body_static', 'valid'])
    if segment_df.empty:
        print(f"No samples for {dataset}/{subject}/{trial}; skipping diagnostic.")
        return
    segments = [s for s in spec.segment_sensor if s in set(segment_df['segment'])]
    t_start = float(segment_df['timestamp'].min())

    fig, axes = plt.subplots(len(segments), 1, figsize=(13, 1.5 * len(segments) + 1.5),
                             sharex=True)
    for ax, segment in zip(np.atleast_1d(axes), segments):
        trace = segment_df[segment_df['segment'] == segment].sort_values('timestamp')
        time = trace['timestamp'].to_numpy() - t_start
        ax.plot(time, trace['gyro_norm'].to_numpy(), color='#c9721f', linewidth=0.7, alpha=0.85)
        ax.axhline(STATIC_GYRO_MAX, color='black', linestyle=':', linewidth=1.0)
        ax.set_yscale('log')
        ax.set_ylabel(segment, fontsize=9, rotation=0, ha='right', va='center')
        ax.tick_params(labelsize=8)
        # Shading from the per-sample mask rather than the intervals table, so what is drawn is
        # exactly what the regime split used.
        ylim = ax.get_ylim()
        for mask_col, color in (('static', INTERVAL_COLORS['static']),
                                ('body_static', INTERVAL_COLORS['body_static'])):
            mask = trace[mask_col].to_numpy()
            edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
            for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                ax.axvspan(time[start], time[min(end, len(time) - 1)], color=color, alpha=0.18,
                           linewidth=0, zorder=0)
        ax.set_ylim(ylim)
        sns.despine(ax=ax)
    np.atleast_1d(axes)[-1].set_xlabel('Time (s)')
    handles = [Patch(facecolor=INTERVAL_COLORS['static'], alpha=0.45, label='static (this sensor)'),
               Patch(facecolor=INTERVAL_COLORS['body_static'], alpha=0.45, label='whole body static'),
               Line2D([0], [0], color='black', linestyle=':', label=f'{STATIC_GYRO_MAX} rad/s')]
    np.atleast_1d(axes)[0].legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left',
                                  fontsize=9)

    plot_utils.finalize_and_save_plot(
        fig, f"Static detection — {dataset} {subject}/{trial}",
        f"diagnostics/static_detection_{subject}_{trial}.png", plots_dir(dataset),
        epilog=STATIC_EPILOG, save=save, show=show)

# ==============================================================================
# Figure sets
# ==============================================================================

def plot_distributions(segment_df: pd.DataFrame, segments: List[str], dataset: str,
                       plot_types: Sequence[str], caption: str,
                       strip_clip: Optional[float] = None, show: bool = False) -> None:
    segment_colors = color_map(segments)
    clip = lambda metric: strip_clip if strip_clip is not None else STRIP_CLIP[metric]

    for metric in [m for m in SEGMENT_METRICS if m in segment_df.columns]:
        title = f"{METRIC_TITLES[metric]}\n{caption}"
        if 'box' in plot_types:
            plot_split_box(segment_df, 'segment', segments, metric, dataset, title,
                           f"{metric}_by_segment_box.png", epilog=STATIC_EPILOG, show=show)
        pooled = pd.DataFrame({'segment': segment_df['segment'],
                               'value': segment_df[metric]}).dropna()
        pooled = pooled[pooled['segment'].isin(segments)]
        if 'joy' in plot_types:
            plot_joyplot(_subsample(pooled, 'segment', MAX_JOY_POINTS), 'segment', segments,
                         METRIC_LABELS[metric], title, dataset,
                         f"{metric}_by_segment_joy.png", segment_colors,
                         **JOY_LIMITS[metric], show=show)
        if 'strip' in plot_types:
            plot_stripplot(pooled, 'segment', segments, METRIC_LABELS[metric], title, dataset,
                           f"{metric}_by_segment_strip.png", segment_colors,
                           clip_quantile=clip(metric), show=show)

def _subsample(df: pd.DataFrame, group_col: str, cap: int, seed: int = 0) -> pd.DataFrame:
    """At most `cap` rows per group. A KDE over tens of millions of samples takes minutes and
    is visually identical to one over a few hundred thousand."""
    if df.empty:
        return df
    return pd.concat([group.sample(n=min(len(group), cap), random_state=seed)
                      for _, group in df.groupby(group_col, observed=True)], ignore_index=True)


# ==============================================================================
# CLI
# ==============================================================================

def pick_example_trial(dataset: str, row_keys: Sequence[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    """The trial with the most whole-body-static time, for the time-series figure.

    Chosen from the data rather than hard-coded, because the two datasets have nothing in
    common in either naming or protocol and a constant would be wrong for one of them.
    """
    best, best_duration = None, 0.0
    for subject, trial in row_keys:
        intervals = load_trial_table(dataset, 'intervals', row_keys=[(subject, trial)])
        if intervals.empty:
            continue
        duration = intervals.loc[intervals['label'] == 'body_static', 'duration_s'].sum()
        if duration > best_duration:
            best, best_duration = (subject, trial), duration
    if best is not None:
        print(f"Example trial for the time series: {best[0]}/{best[1]} "
              f"({best_duration:.0f}s whole-body static)")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='alborno', choices=sorted(DATASETS))
    parser.add_argument('--summary', action='store_true',
                        help="Draw ONLY the cross-dataset summary figure and exit. It spans "
                             "every dataset in --summary-datasets, so --dataset does not apply "
                             "to it.")
    parser.add_argument('--summary-datasets', nargs='+', default=list(FRAME_VERIFIED_DATASETS),
                        choices=sorted(DATASETS),
                        help="Datasets the summary figure draws (default: the ones whose mocap "
                             "world frame is known to agree with the pipeline's Y-up "
                             "EXPECTED_GRAVITY; see FRAME_VERIFIED_DATASETS).")
    parser.add_argument('--subjects', nargs='+', default=None,
                        help="Restrict to these subject/session ids (default: everything on disk).")
    parser.add_argument('--trials', nargs='+', default=None,
                        help="Restrict to these trial names (default: everything on disk).")
    parser.add_argument('--sides', choices=['right', 'left', 'both'], default='both',
                        help="Which side's segments/joints to draw (default: both — the "
                             "contralateral sensor is a replicate of the same placement in the "
                             "same magnetic environment, not a duplicate row; see side_filter). "
                             "Pass 'right' for a compact figure when that replication is not "
                             "the question.")
    parser.add_argument('--plot-types', nargs='+', choices=['box', 'joy', 'strip'],
                        default=['box', 'joy', 'strip'])
    parser.add_argument('--example-trial', nargs=2, metavar=('SUBJECT', 'TRIAL'), default=None,
                        help="Trial for the static-bout time series (default: whichever has the "
                             "most whole-body-static time).")
    parser.add_argument('--strip-clip', type=float, default=None, metavar='QUANTILE',
                        help="Override the strip plots' x-axis clip for every metric (defaults "
                             "are per metric, see STRIP_CLIP). Samples beyond the clip are still "
                             "drawn at the axis edge and counted, so tightening this hides "
                             "nothing.")
    parser.add_argument('--diagnostics', action='store_true',
                        help="Also draw the per-trial static-detector diagnostics. One figure "
                             "per trial, so this is opt-in.")
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    if args.summary:
        plot_assumption_summary(args.summary_datasets, show=args.show)
        return

    spec = get_dataset(args.dataset)
    row_keys = enumerate_trials(args.dataset)
    if args.subjects:
        row_keys = [(s, t) for s, t in row_keys if s in args.subjects]
    if args.trials:
        row_keys = [(s, t) for s, t in row_keys if t in args.trials]
    if not row_keys:
        print(f"No trials selected in {args.dataset}.")
        return

    segments = side_filter(list(spec.segment_sensor), args.sides)

    print(f"Loading per-sample tables from {dataset_dir(args.dataset)}...")
    # `spec.segment_metrics()` rather than SEGMENT_METRICS: a dataset with no magnetometer never
    # writes the magnetic columns, and asking a parquet reader for a column that is not there is
    # an error rather than a NaN column.
    segment_df = load_samples(args.dataset, 'segment_samples', spec.segment_metrics(), 'segment',
                              row_keys)
    # The joint tables are deliberately NOT read here. o^J is a property of a joint PAIR and of
    # the joint-center projection rather than of the global sensor assumptions this experiment
    # measures, so its figures moved out to the module taking up the joint-offset question (see
    # scratch/observability_plots_for_joint_offset.py). The tables are still written by the
    # experiment and still drive report sections 6 and 7; not loading 17 M joint-sample rows here
    # is also most of this script's runtime and memory.
    sensor_df = load_trial_table(args.dataset, 'sensor_stats', row_keys=row_keys)
    if segment_df.empty:
        print(f"No per-trial tables found under {dataset_dir(args.dataset)}. "
              f"Run `python -m experiments.global_assumptions --dataset {args.dataset}` first.")
        return

    trials = segment_df[['subject', 'trial']].drop_duplicates()
    n_subjects = trials['subject'].nunique()
    caption = (f"{args.dataset}: n={n_subjects} subject{'s' if n_subjects != 1 else ''}, "
               f"{len(trials)} trials"
               f"{'' if args.sides == 'both' else f', {args.sides} side'}")
    # Say it on the figure when the sensors were not all recorded together. Every by-segment
    # distribution here is sample-pooled, and on IMoVE the Mid sensors carry the long-walk
    # sessions (100 Hz, ~2800 s, Mid only) while the High and Low ones do not — so a
    # Mid-vs-High difference in these figures is partly a difference in recordings. The
    # observability figures pair over trials to remove it; these do not, because "what does a
    # thigh accelerometer read" legitimately wants every sample, so the caveat is stated instead.
    # Distinct (subject, trial) PAIRS, not trial names: every session reuses the same trial names
    # ('t1_walking_001' in 26 of them), so counting names alone reported 10-13 where the answer
    # is 195-243.
    per_segment_trials = (segment_df[['segment', 'subject', 'trial']].drop_duplicates()
                          .groupby('segment', observed=True).size())
    if per_segment_trials.nunique() > 1:
        caption += (f"\nnote: sensors not all recorded together "
                    f"({per_segment_trials.min()}-{per_segment_trials.max()} trials per segment); "
                    f"cross-placement differences are partly a difference in recordings")
    print(f"Loaded {len(trials)} trial(s), {len(segment_df):,} segment-samples.")

    plot_headline(segment_df, segments, args.dataset, caption, show=args.show)
    plot_distributions(segment_df, segments, args.dataset, args.plot_types, caption,
                       strip_clip=args.strip_clip, show=args.show)
    plot_noise_floor(sensor_df, segments, args.dataset, caption, spec.noise_modalities(),
                     show=args.show)
    plot_static_coverage(sensor_df, segments, args.dataset, caption, show=args.show)
    plot_detector_validation(sensor_df, segments, args.dataset, caption, show=args.show)
    plot_subject_heatmap(segment_df, segments, args.dataset, caption, show=args.show)
    plot_trial_spread(segment_df, segments, args.dataset, caption, show=args.show)

    example = tuple(args.example_trial) if args.example_trial else pick_example_trial(
        args.dataset, row_keys)
    if example is not None:
        plot_static_timeseries(args.dataset, example[0], example[1], segments, show=args.show)

    if args.diagnostics:
        for subject, trial in row_keys:
            plot_static_diagnostic(args.dataset, subject, trial, spec, show=args.show)


if __name__ == '__main__':
    main()
