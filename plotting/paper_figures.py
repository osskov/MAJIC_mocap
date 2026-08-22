"""
Generates paper figures from results/statistics/all_subject_<dataset>_statistics.parquet,
produced by experiments/benchmark_experiment.py (flat columns, one row per
dataset/subject/trial/trial_type/method/joint_name/axis, with *_rad error metrics).

ONE DATASET PER RUN, chosen with --dataset and defaulting to alborno, whose figures land in
plots/ exactly as before; any other dataset gets plots/<dataset>/ so the two cannot overwrite
each other. The method labels, the figure-1 method order and the joint order below were
written for the Al Borno arms — they are inputs to the builders rather than assumptions in
them, so a different dataset draws correct figures with whatever methods and joints its
statistics file happens to hold, but the LABELLING has not been revisited for IMoVE's
placement variants or the biplane half's two ground truths.
# Note: the 'marker' method is the mocap ground truth used to compute these
# error/correlation stats against, so it never appears as a row value — only the
# IMU-derived methods (ekf, mag_off, mag_on, mag_adapt_15, mag_adapt_100, mag_adapt_200) do.

The figure builders take the statistics name, the output directory and the method
list as arguments so a benchmark-shaped run at a different filter tuning can draw the
identical figures from its own statistics file without a second copy of this module —
see plotting/normalized_benchmark.py. main() supplies the benchmark's own values, so
`python -m plotting.paper_figures` is unchanged.

    python -m plotting.paper_figures --dataset alborno
    python -m plotting.paper_figures --dataset alborno --axes anatomical
    python -m plotting.paper_figures --dataset alborno --axes both

--axes anatomical DRAWS THE SAME ERROR, SPLIT rather than pooled: one panel per anatomical axis of
the parent segment (flexion, adduction, internal rotation) instead of one number per method. The
`axis` column has carried a per-component breakdown all along, but in the PARENT SENSOR's own
frame, which is re-strapped per subject and means a different direction for each of them;
`experiments/anatomical_frames.py` measures the rotation onto the segment's anatomical axes from
the source mocap's landmarks, and `benchmark_experiment --anatomical-axes` writes the resulting
'FE'/'AA'/'IE' rows beside the existing 'MAG'/'X'/'Y'/'Z'. Because the basis is orthonormal the
three panels are components of the magnitude figure and add up to it in quadrature, exactly.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import paths
from experiments.experiment_utils import load_statistics
from experiments.benchmark_experiment import normalized_method_names, variant_name
from experiments.global_assumptions import DATASETS
from plotting.utils import plot_metric_distribution, plot_metric_heatmap, DEFAULT_JOINT_ORDER

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# The dataset is a SUFFIX on this, not part of it: the benchmark writes
# all_subject_<dataset>_statistics.parquet, one file per dataset.
STATS_NAME = "all_subject"
DEFAULT_DATASET = "alborno"
PLOTS_DIR = paths.plots_dir()

SHOW_PLOTS = True
SAVE_PLOTS = True

# Al Borno's roster, kept for a caller that wants to pin the subject set explicitly. NOT the
# default any more — see _load_stats, where the default is whatever the file contains.
SUBJECTS_TO_PLOT = [f"Subject{i:02d}" for i in range(1, 12)]

# Display labels for the method keys written by experiments/benchmark_experiment.py.
METHOD_LABELS = {
    'ekf': 'EKF',
    'unprojected': 'Unprojected',
    'mag_off': 'Mag Off',
    'mag_off_flat': 'Mag Off (Flat)',
    'mag_off_dyn': 'Mag Off (Dynamic)',
    'mag_off_unnormalized': 'Mag Off (Unnormalized)',
    'mag_off_normalized': 'Mag Off (Normalized)',
    'mag_off_rescaled': 'Mag Off (Rescaled)',
    'mag_on': 'Mag On',
    'mag_on_normalized': 'Mag On (Normalized)',
    'mag_adapt': 'MAJIC',
    'mag_adapt_normalized': 'MAJIC (Normalized)',
    'mag_adapt_15': 'MAJIC (15)',
    'mag_adapt_100': 'MAJIC (100)',
    'mag_adapt_200': 'MAJIC (200)',
}
METHODS_TO_PLOT = ['ekf', 'mag_off', 'mag_off_flat', 'mag_off_dyn', 'mag_off_unnormalized',
                   'mag_off_normalized', 'mag_off_rescaled', 'mag_on', 'mag_on_normalized',
                   'mag_adapt_15', 'mag_adapt_100', 'mag_adapt_200', 'mag_adapt',
                   'mag_adapt_normalized', 'mag_adapt_dyn']

# Left/right joints are pooled together under a common name; list them
# individually (e.g. 'R_Hip') here instead if that's not desired.
RENAME_JOINTS = {
    'R_Hip': 'Hip', 'L_Hip': 'Hip',
    'R_Knee': 'Knee', 'L_Knee': 'Knee',
    'R_Ankle': 'Ankle', 'L_Ankle': 'Ankle',
}

# The methods compared in Figure 1 — the four the benchmark grid runs
# (experiment_utils.METHODS minus 'marker', which is the ground truth), ordered
# baseline -> ablations -> MAJIC.
FIGURE_1_METHODS = ['ekf', 'mag_off', 'mag_on', 'mag_adapt']

# One of: mean, std, rmse, mae, mad, min, q25, median, q75, max.
METRIC = 'rmse'
METRIC_UNITS = 'deg'  # 'deg' or 'rad'

# Which rotation-vector axis to summarize for the error-metric plots.
# 'MAG' (rotation magnitude) is the usual choice.
AXIS_TO_PLOT = 'MAG'

# The anatomical breakdown, drawn by --axes anatomical. These `axis` values only exist in a
# statistics file written by `benchmark_experiment --anatomical-axes`, which needs
# `experiments/anatomical_frames.py` to have measured the bases first; a file without them draws
# nothing and says so rather than falling back to MAG under an anatomical title.
#
# THE THREE PANELS ADD UP TO THE MAG PANEL, in quadrature and exactly: the basis is orthonormal,
# so RMSE_MAG^2 = RMSE_FE^2 + RMSE_AA^2 + RMSE_IE^2 in every cell. That is what makes this a
# breakdown of figure 1 rather than a second, differently-scaled measurement of the same thing.
# It is NOT a clinical per-plane angle error — see experiments/anatomical_frames.py.
ANATOMICAL_AXIS_ORDER = ['FE', 'AA', 'IE']
AXIS_LABELS = {
    'FE': 'Flexion / Extension',
    'AA': 'Adduction / Abduction',
    'IE': 'Internal / External rotation',
    'MAG': 'Total (magnitude)',
    'X': 'Sensor X', 'Y': 'Sensor Y', 'Z': 'Sensor Z',
}

# Facet the distribution plot by joint. Set to None for one pooled plot
# across all joints (the heatmap always breaks down by joint regardless).
FACET_BY = ''

# 'strip' (median/IQR whiskers + points), 'box', or 'bar' (mean + 95% CI).
PLOT_STYLE = 'strip'

# --- Data loading ---


def _load_stats(stats_name: str = STATS_NAME,
                methods_to_plot: Optional[List[str]] = None,
                subjects_to_plot: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """The statistics file as the figures want it: degrees, pooled joint names, filtered.

    `subjects_to_plot` defaults to EVERY SUBJECT IN THE FILE rather than to the Al Borno
    roster. The roster is a list of 'Subject01'-style labels, and no other dataset produces
    one — filtering IMoVE's 's13l' against it silently returns an empty frame and draws an
    empty figure, which is the worst of the available failures.
    """
    df = load_statistics(stats_name)
    if df is None:
        print(f"Error: statistics file not found at {paths.statistics_path(stats_name)}. "
              f"Run the experiment that writes it first.")
        return None
    methods_to_plot = METHODS_TO_PLOT if methods_to_plot is None else methods_to_plot
    for col in [c for c in df.columns if c.endswith('_rad')]:
        df[col.replace('_rad', '_deg')] = np.degrees(df[col])
    df['joint_name'] = df['joint_name'].replace(RENAME_JOINTS)
    keep_subjects = df['subject'].unique() if subjects_to_plot is None else subjects_to_plot
    return df[df['method'].isin(methods_to_plot) & df['subject'].isin(keep_subjects)]


# --- Figures ---


def figure_1(error_df: pd.DataFrame, metric_col: str, plot_type: str = PLOT_STYLE,
             plots_dir: Path = PLOTS_DIR, group_order: Optional[List[str]] = None,
             labels: Optional[Dict[str, str]] = None, title: str = "Joint Angle Error by Method") -> None:
    """Figure 1: joint-angle error by method — one point per subject/trial/joint,
    summarized per method by median/IQR ('strip') or mean/SD ('bar_sd').

    The style is in the filename so the variants don't overwrite each other."""
    unit_label = 'degrees' if METRIC_UNITS == 'deg' else 'radians'
    suffix = '' if plot_type == 'strip' else f"_{plot_type}"
    plot_metric_distribution(
        error_df, metric_col, group_col='method',
        group_order=FIGURE_1_METHODS if group_order is None else group_order, plots_dir=plots_dir,
        labels=METHOD_LABELS if labels is None else labels,
        plot_type=plot_type, facet_by=(FACET_BY or None), facet_order=DEFAULT_JOINT_ORDER,
        ylabel=f"{METRIC.upper()} ({unit_label})", title=title,
        filename=f"figure_1_{METRIC}_by_method{suffix}.png", save=SAVE_PLOTS, show=SHOW_PLOTS
    )


def figure_1_by_axis(error_df: pd.DataFrame, metric_col: str, plot_type: str = PLOT_STYLE,
                     plots_dir: Path = PLOTS_DIR, group_order: Optional[List[str]] = None,
                     labels: Optional[Dict[str, str]] = None,
                     title: str = "Joint Angle Error by Method and Anatomical Axis") -> None:
    """Figure 1, split into one panel per ANATOMICAL AXIS instead of pooled into a magnitude.

    `error_df` must already be narrowed to the anatomical axes and hold every joint, because the
    point of the split is what it does to a pooled number: one point per subject/trial/joint in
    each panel, exactly as the magnitude figure has.

    SHARED Y AXIS across the panels (`plot_metric_distribution` passes sharey=True), which is
    deliberate — the three panels are components of one vector, so their relative height IS the
    result. Reading them on independent scales would show three similar-looking distributions.

    Every panel belongs to ONE Holm family blocked on subject x joint, so a significance bracket
    means "differs after correcting across all three axes", not within one panel.
    """
    unit_label = 'degrees' if METRIC_UNITS == 'deg' else 'radians'
    suffix = '' if plot_type == 'strip' else f"_{plot_type}"
    present = [axis for axis in ANATOMICAL_AXIS_ORDER if axis in set(error_df['axis'])]
    panelled = error_df.assign(axis=error_df['axis'].map(lambda a: AXIS_LABELS.get(a, a)))
    plot_metric_distribution(
        panelled, metric_col, group_col='method',
        group_order=FIGURE_1_METHODS if group_order is None else group_order, plots_dir=plots_dir,
        labels=METHOD_LABELS if labels is None else labels,
        plot_type=plot_type, facet_by='axis',
        facet_order=[AXIS_LABELS.get(axis, axis) for axis in present],
        ylabel=f"{METRIC.upper()} ({unit_label})", title=title,
        filename=f"figure_1_{METRIC}_by_axis{suffix}.png", save=SAVE_PLOTS, show=SHOW_PLOTS
    )


def make_figures(stats_name: str = STATS_NAME, plots_dir: Path = PLOTS_DIR,
                 figure_methods: Optional[List[str]] = None,
                 methods_to_plot: Optional[List[str]] = None,
                 labels: Optional[Dict[str, str]] = None,
                 title: str = "Joint Angle Error by Method",
                 axes: str = 'magnitude') -> None:
    """The benchmark's figures — the by-method distribution and the joint x method
    heatmap — drawn from `stats_name` into `plots_dir`.

    Parametrized rather than reading the module constants directly so a re-tuned
    benchmark run (plotting/normalized_benchmark.py) draws the same two figures from its
    own statistics file, into its own directory, under its own method names. The
    filenames are fixed ('figure_1_*.png', 'heatmap_*.png'), so a caller passing a
    different stats_name must pass a different plots_dir or it will overwrite the
    paper's figures.

    `axes` selects which error split to draw: 'magnitude' is the historical pair and the default,
    so an existing caller is unchanged; 'anatomical' adds the three-panel flexion / adduction /
    rotation version and one heatmap per axis, under their own filenames; 'both' draws all of
    them. The anatomical figures need a statistics file written with
    `benchmark_experiment --anatomical-axes` and say so rather than falling back if it was not."""
    metric_col = f"{METRIC}_{METRIC_UNITS}"
    figure_methods = FIGURE_1_METHODS if figure_methods is None else figure_methods
    methods_to_plot = METHODS_TO_PLOT if methods_to_plot is None else methods_to_plot
    labels = METHOD_LABELS if labels is None else labels
    if axes not in ('magnitude', 'anatomical', 'both'):
        raise ValueError(f"axes must be 'magnitude', 'anatomical' or 'both', got {axes!r}")

    stats_df = _load_stats(stats_name, methods_to_plot)
    if stats_df is None:
        return
    print(stats_df.head())

    if axes in ('magnitude', 'both'):
        error_df = stats_df[stats_df['axis'] == AXIS_TO_PLOT]

        figure_1(error_df, metric_col, plots_dir=plots_dir, group_order=figure_methods,
                 labels=labels, title=title)

        plot_metric_heatmap(
            error_df, metric_col, group_col='method', group_order=methods_to_plot,
            plots_dir=plots_dir, labels=labels, save=SAVE_PLOTS, show=SHOW_PLOTS
        )

    if axes in ('anatomical', 'both'):
        anatomical_df = stats_df[stats_df['axis'].isin(ANATOMICAL_AXIS_ORDER)]
        if anatomical_df.empty:
            print(f"No anatomical axes in {stats_name}: it holds "
                  f"{sorted(stats_df['axis'].unique())}. Re-run the statistics stage with them:\n"
                  f"  python -m experiments.anatomical_frames --dataset <name>\n"
                  f"  python -m experiments.benchmark_experiment --dataset <name> --stats-only "
                  f"--anatomical-axes")
        else:
            figure_1_by_axis(anatomical_df, metric_col, plots_dir=plots_dir,
                             group_order=figure_methods, labels=labels)
            # One heatmap per axis rather than one with joint x axis rows: the rows are joints in
            # every other heatmap in the repo, and a reader comparing this against the magnitude
            # one should not have to re-learn the index.
            for axis in [a for a in ANATOMICAL_AXIS_ORDER if a in set(anatomical_df['axis'])]:
                plot_metric_heatmap(
                    anatomical_df[anatomical_df['axis'] == axis], metric_col, group_col='method',
                    group_order=methods_to_plot, plots_dir=plots_dir, labels=labels,
                    filename=f"heatmap_{METRIC}_{axis}.png",
                    title=f"Mean {metric_col} by Joint and method — "
                          f"{AXIS_LABELS.get(axis, axis)}",
                    save=SAVE_PLOTS, show=SHOW_PLOTS
                )

    print("\n--- All plotting complete ---")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default=DEFAULT_DATASET, choices=sorted(DATASETS),
                        help="Which dataset's benchmark statistics to draw.")
    parser.add_argument('--tuning', default=None,
                        help="Draw a re-tuned benchmark run instead of the shipped one — the same "
                             "value passed to `benchmark_experiment --tuning` (e.g. static_mag_x5).")
    parser.add_argument('--normalized', action='store_true',
                        help="Draw the run whose filter arms normalize their measurements, i.e. "
                             "`benchmark_experiment --normalized`.")
    parser.add_argument('--axes', default='magnitude',
                        choices=['magnitude', 'anatomical', 'both'],
                        help="Which error split to draw: the pooled rotation magnitude (default, "
                             "the paper's figures), the flexion/adduction/rotation breakdown, or "
                             "both. The anatomical split needs statistics written with "
                             "`benchmark_experiment --anatomical-axes`.")
    args = parser.parse_args()

    # The variant name comes from benchmark_experiment, not from a second spelling of the same
    # rule here: it is the string that experiment wrote its statistics under, and a figure drawn
    # from the wrong file into a directory claiming otherwise is the failure worth engineering
    # against.
    variant = variant_name(args.tuning, args.normalized)

    # alborno keeps plots/ so the paper's own figure paths are unchanged; anything else is
    # namespaced, because the filenames are identical between datasets. A variant nests OUTSIDE
    # the dataset for the same reason — the shipped figures must not be overwritten by a re-tuned
    # run that happens to draw the same four arms under the same four labels.
    root = PLOTS_DIR if variant is None else paths.plots_dir(variant)
    plots_dir = root if args.dataset == DEFAULT_DATASET else root / args.dataset

    stats_name = (f"{STATS_NAME}_{args.dataset}" if variant is None
                  else f"{STATS_NAME}_{variant}_{args.dataset}")

    # A normalized run's arms carry the '_normalized' suffix, so figure 1 has to ask for those
    # names or it finds none of them and draws an empty panel. Labels stay the SHORT paper ones:
    # every arm in such a run normalizes, so tagging each of the four with '(Normalized)' repeats
    # one fact four times and says nothing that distinguishes them. What the run was is in the
    # title instead, where it cannot be mistaken for an arm name.
    figure_methods = FIGURE_1_METHODS
    labels = METHOD_LABELS
    if args.normalized:
        figure_methods = normalized_method_names(FIGURE_1_METHODS)
        labels = {**METHOD_LABELS,
                  **{normalized: METHOD_LABELS[base]
                     for base, normalized in zip(FIGURE_1_METHODS, figure_methods)
                     if base in METHOD_LABELS}}

    title = "Joint Angle Error by Method"
    if variant is not None:
        detail = ", ".join(filter(None, [f"{args.tuning} gains" if args.tuning else None,
                                         "normalized" if args.normalized else None]))
        title = f"{title}\n({detail})"

    make_figures(stats_name=stats_name, plots_dir=plots_dir, figure_methods=figure_methods,
                 methods_to_plot=list(dict.fromkeys(METHODS_TO_PLOT + figure_methods)),
                 labels=labels, title=title, axes=args.axes)


if __name__ == "__main__":
    main()
