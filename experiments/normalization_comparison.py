"""
Three-arm comparison of how the vector measurements are scaled going into the filter,
across every base method.

Each arm is a parametrized method name (resolved by
experiment_utils.resolve_method_spec), so this reuses the same generate+stats workers as
the vanilla pipeline (benchmark_experiment.py) — just with a different method list. Every
arm is spelled out rather than letting the bare base name stand in for the control, so
each writes its own parquet and none overwrites the benchmark's outputs.

    <base>_unnormalized   raw-magnitude acc/mag
    <base>_normalized     unit-length acc/mag, stds untouched
    <base>_rescaled       unit-length acc/mag, each std divided by its nominal magnitude

Every arm names its normalization explicitly, so each is independent of whatever the base
defaults to. That matters for the EKF, whose base default is now 'normalized' (see the
METHODS note in experiment_utils): 'ekf_unnormalized' is this experiment's control arm but
is NOT what the pipeline ships, while 'ekf_normalized' is. For the mag methods the
unnormalized arm and the shipped method still coincide.

WHY THREE ARMS AND NOT TWO. Normalizing scales a sensor's residual and its Jacobian by
1/|v| while its entry in R stays put, so 'normalized' also trusts the accelerometer
~|a|^2 ~ 96x less relative to the gyro. It therefore differs from 'unnormalized' in two
ways at once — geometry and tuning — and a difference between those two arms cannot say
which one moved. 'rescaled' holds the tuning fixed and changes only the geometry, so:

    rescaled vs unnormalized  ->  the effect of discarding per-sample magnitude
    normalized vs rescaled    ->  the effect of the 96x de-weighting alone

The rescaling uses a FIXED nominal magnitude per sensor, not each sample's own |v|.
Rescaling per sample would be an exact algebraic no-op reproducing 'unnormalized' bit for
bit — see the NOMINAL_*_MAGNITUDE comment in experiment_utils.
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import argparse
from functools import partial
from typing import List

import numpy as np
import pandas as pd

from experiments.experiment_utils import (
    SUBJECTS, ACTIVITIES, load_all_joint_angles, compute_error_stats, save_statistics,
    run_tracked_grid, generate_joint_angles_worker, compute_stats_worker,
)

NORMALIZATION_BASE_METHODS = ['mag_on', 'mag_off', 'mag_adapt', 'ekf']

# Control first, then the two treatments. 'rescaled' is the controlled one; 'normalized'
# is kept because it is what a naive normalization change actually does.
NORMALIZATION_ARMS = ['unnormalized', 'normalized', 'rescaled']


def build_normalization_methods(bases: List[str]) -> List[str]:
    """Every base crossed with every normalization arm, base-major so a base's arms sit
    next to each other in the progress table and in the summary."""
    return [f"{base}_{arm}" for base in bases for arm in NORMALIZATION_ARMS]


def _print_comparison(summary_stats_df: pd.DataFrame, metric: str = 'rmse_rad') -> None:
    """Prints the arms side by side, one row per base method, pooling over
    subjects/activities/joints.

    Both deltas are against 'unnormalized', but they answer different questions and the
    column headers say which: 'rescaled' isolates the geometry, 'normalized' carries the
    geometry and the de-weighting together.

    compute_error_stats returns one row per (subject, activity, joint, axis) with the
    metrics as columns in RADIANS; degrees are a plotting-side convention, so the
    conversion happens here too rather than in the saved statistics."""
    if metric not in summary_stats_df.columns:
        available = [c for c in summary_stats_df.columns if c.endswith('_rad')]
        print(f"\nNo '{metric}' column to compare — skipping the summary table. Available: {available}")
        return

    df = summary_stats_df[summary_stats_df['axis'] == 'MAG']
    if df.empty:
        print("\nNo axis 'MAG' rows to compare — skipping the summary table.")
        return

    table = np.degrees(df.pivot_table(index='base_method', columns='normalization',
                                      values=metric, aggfunc='mean'))
    if 'unnormalized' not in table.columns:
        print("\nThe 'unnormalized' control arm produced no results — skipping the summary "
              "table, since both deltas are measured against it.")
        return
    treatments = [arm for arm in NORMALIZATION_ARMS
                  if arm != 'unnormalized' and arm in table.columns]
    if not treatments:
        print("\nOnly the control arm produced results — skipping the summary table.")
        return

    print(f"\nMean {metric.replace('_rad', '')} (axis MAG, degrees), pooled over "
          f"subjects / activities / joints:\n")
    header = f"  {'method':<12} {'unnormalized':>13}"
    for arm in treatments:
        header += f" {arm:>12} {'delta %':>9}"
    print(header)
    for base in table.index:
        un = table.loc[base, 'unnormalized']
        row = f"  {base:<12} {un:>13.3f}"
        for arm in treatments:
            value = table.loc[base, arm]
            row += f" {value:>12.3f} {100 * (value - un) / un:>+8.1f}%"
        print(row)

    print("\n  Negative delta = that arm beat the raw-magnitude control.")
    if 'rescaled' in treatments:
        print("  'rescaled' is the controlled contrast: same sensor weighting as the "
              "control, so it\n  isolates the cost of throwing away per-sample magnitude. "
              "'normalized' additionally\n  de-weights the accelerometer ~96x, so the gap "
              "between the two treatment columns is\n  that de-weighting on its own.")
    else:
        print("  Without the 'rescaled' arm this is a tuning comparison, not a verdict on "
              "\n  normalization itself — see the module docstring.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare normalized vs unnormalized vector measurements into the filter."
    )
    parser.add_argument("--subjects", nargs='+', default=SUBJECTS)
    parser.add_argument("--activities", nargs='+', default=ACTIVITIES)
    parser.add_argument("--bases", nargs='+', default=NORMALIZATION_BASE_METHODS,
                         help="Base filter methods to run under both normalization arms.")
    parser.add_argument("--metric", default='rmse_rad',
                         help="Metric column used for the printed side-by-side summary "
                              "(printed in degrees).")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    normalization_methods = build_normalization_methods(args.bases)
    methods = ['marker'] + normalization_methods
    print(f"Running normalization comparison with methods: {methods}")

    row_keys = [(subject, activity) for subject in args.subjects for activity in args.activities]

    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['load'] + methods, generate_joint_angles_worker,
                      args.workers, title="NORMALIZATION COMPARISON GENERATION")
    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['stats'],
                      partial(compute_stats_worker, methods=methods, stats_name="normalization_comparison"),
                      args.workers, title="NORMALIZATION COMPARISON STATISTICS")

    all_data_df = load_all_joint_angles(args.subjects, args.activities, methods)
    if all_data_df.empty:
        print("Error: no data was loaded for any subject.")
        return

    summary_stats_df = compute_error_stats(all_data_df)
    if summary_stats_df.empty:
        print("Warning: summary statistics are empty.")
        return

    # Split each method name back into the two factors the comparison is over, so the
    # plotting side can group on them without re-parsing strings.
    method_meta = {f"{base}_{arm}": (base, arm)
                   for base in args.bases for arm in NORMALIZATION_ARMS}
    summary_stats_df['base_method'] = summary_stats_df['method'].map(lambda m: method_meta.get(m, (None, None))[0])
    summary_stats_df['normalization'] = summary_stats_df['method'].map(lambda m: method_meta.get(m, (None, None))[1])

    path = save_statistics(summary_stats_df, "normalization_comparison")
    print(f"\nSaved normalization comparison statistics to {path}")

    _print_comparison(summary_stats_df, metric=args.metric)


if __name__ == '__main__':
    main()
