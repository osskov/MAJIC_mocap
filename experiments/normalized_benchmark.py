"""
The benchmark grid, re-run with unit-length filter inputs and a re-tuned noise model.

Same shape as experiments/benchmark_experiment.py — the same four IMU methods against the
same marker ground truth over the same subject x activity grid, aggregated into the same
statistics table — with exactly two things changed:

  1. NORMALIZATION IS ON FOR EVERY ARM. Each arm is named '<base>_normalized', so the
     method-spec grammar (experiment_utils.resolve_method_spec) sets
     normalize_measurements=True regardless of what its base defaults to. This is the
     '_normalized' arm of experiments/normalization_comparison.py, not '_rescaled': the
     stds below are taken at face value against unit-length measurements rather than being
     divided by each sensor's nominal magnitude.

  2. THE STDS ARE RE-TUNED. gyro 0.0116, acc 0.03, mag 0.05 (against the shipped defaults
     of 0.0045 / 0.018 / 0.05).

The two changes are not separable and this experiment does not try to separate them —
normalizing at a fixed std is itself a retuning (a unit-length accelerometer with
acc_std=0.018 is the unnormalized filter at acc_std~0.18, i.e. ~10x lighter), so a std
chosen for raw-magnitude inputs means something different here. The point of this script
is a single benchmark-shaped answer for one specific normalized tuning, not an ablation;
normalization_comparison is where the two factors get pulled apart.

WHY A SEPARATE OUTPUT TREE. The std triple lives outside the method name (it is a module
constant, not a suffix), so 'mag_on_normalized' at this tuning and 'mag_on_normalized' at
the default tuning would write the same parquet path. Everything here is namespaced under
the 'normalized_benchmark' variant (results/joint_angles/normalized_benchmark/...) and its
own statistics file, so this run and the benchmark's coexist and each manifest records the
stds that produced it. Nothing overwrites benchmark_experiment.py's outputs.
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import argparse
from functools import partial
from typing import List

import numpy as np
import pandas as pd

import paths
from experiments.experiment_utils import (
    SUBJECTS, ACTIVITIES,
    TRIAL_DATASET, load_all_joint_angles, load_statistics, compute_error_stats, save_statistics,
    run_tracked_grid, resolve_method_spec, resolve_stds, pipeline_constants,
    generate_joint_angles_worker, compute_stats_worker,
)

# THIS EXPERIMENT OWNS ITS OWN JOINT-ANGLE TREE:
# results/experiments/normalized_benchmark/joint_angles/. `results/joint_angles/` belongs to
# benchmark_experiment.py alone — see paths.joint_angles_write_path for the failure
# that rule exists to prevent.
EXPERIMENT_NAME = "normalized_benchmark"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

STATS_NAME = "normalized_benchmark"

# The default-tuned run this one is a delta against. Al Borno's, explicitly: this experiment
# sweeps a TUNING on one dataset, so the comparison has to come from the same dataset — and
# the benchmark's statistics file is namespaced per dataset now, so 'all_subject' alone names
# nothing and would silently resolve to a file left behind by the pre-dataset pipeline.
BENCHMARK_STATS_NAME = f"all_subject_{TRIAL_DATASET}"

# The output namespace for this tuning's joint angles, and the one thing that keeps this
# run from overwriting the default-tuned pipeline's parquets — see paths.joint_angles_path.
VARIANT = "normalized_benchmark"

# The re-tuned noise model. These are absolute stds handed to RelativeFilter unchanged,
# and the measurements they weight are UNIT LENGTH here, so they are not comparable to the
# DEFAULT_*_STD constants without accounting for that (see the module docstring).
TUNED_STDS = {'gyro_std': 0.0045, 'acc_std': 0.0036, 'mag_std': 0.02}

# The benchmark's four IMU arms plus the ground truth, each spelled '_normalized' rather
# than left to its base default. Spelling it out matters for the EKF, whose base already
# normalizes ('ekf_normalized' is the same spec as a bare 'ekf') and for the three relative
# filters, whose bases do not — one uniform suffix makes the grid uniform.
BASE_METHODS = ['ekf', 'mag_off', 'mag_on', 'mag_adapt']
NORMALIZED_METHODS = [f"{base}_normalized" for base in BASE_METHODS]
DEFAULT_METHODS = ['marker'] + NORMALIZED_METHODS

# 'marker' is the mocap ground truth: no filter runs, so the tuning and the normalization
# are both inert for it and its output is bit-identical to the benchmark's. It is still
# regenerated inside the variant tree so this experiment's statistics phase is
# self-contained and does not silently depend on a benchmark run having happened.


def base_of(method: str) -> str:
    """'mag_on_normalized' -> 'mag_on'. Used only for the printed summary."""
    return method[: -len('_normalized')] if method.endswith('_normalized') else method


# ==============================================================================
# Summary report
# ==============================================================================

def _benchmark_provenance() -> str:
    """One line describing what the benchmark statistics file on disk actually is: when it
    was written and at which stds, read from its manifest sidecar.

    Not decoration. The benchmark is overwritten in place on every re-run, so the numbers
    this script prints a delta against are whichever run happened last — and if that run
    predates a change to the DEFAULT_*_STD constants, the delta spans two retunings rather
    than one. The manifest is the only record of which, so it gets printed."""
    manifest = paths.read_manifest(paths.statistics_path(BENCHMARK_STATS_NAME))
    if manifest is None:
        return ("That file has no manifest sidecar, so its tuning and code version are "
                "unknown — treat the delta as indicative only.")
    constants = manifest.get('constants') or {}
    recorded = {key: constants[key] for key in ('gyro_std', 'acc_std', 'mag_std') if key in constants}
    current = resolve_stds()
    written = manifest.get('written_at', 'an unknown time')
    sha = (manifest.get('git_sha') or '')[:7] or 'unknown'
    line = f"It was written {written} at {sha}, tuned {recorded or 'unrecorded'}."
    if recorded and recorded != current:
        line += (f" NOTE: that is not the current default tuning ({current}), so the delta "
                 f"spans two retunings, not one.")
    return line

def _print_summary(summary_stats_df: pd.DataFrame, subjects: List[str], activities: List[str],
                    metric: str = 'rmse_rad') -> None:
    """Pooled error per arm, in degrees, next to the shipped benchmark's number for the
    same base method where that file exists.

    The benchmark column is a CROSS-RUN comparison and is labelled as one: it is whatever
    results/statistics/all_subject_alborno_statistics.parquet holds, pooled over the same subjects
    and activities requested here. It is here to answer "did this tuning help", which is
    the question this script exists to ask, but it is not a controlled contrast — two
    things moved at once.

    That file's own tuning is read out of its manifest and printed, rather than assumed to
    be the current DEFAULT_*_STD: the benchmark is overwritten in place on re-run and the
    constants have moved during this project, so the file on disk is frequently not the one
    the current constants would produce. Quoting a delta against it without saying what it
    was tuned at is how a retuning comparison ends up measuring two retunings.

    compute_error_stats reports metrics in RADIANS; degrees are a presentation
    convention, so the conversion happens here rather than in the saved table."""
    if metric not in summary_stats_df.columns:
        available = [c for c in summary_stats_df.columns if c.endswith('_rad')]
        print(f"\nNo '{metric}' column to summarize. Available: {available}")
        return
    df = summary_stats_df[summary_stats_df['axis'] == 'MAG']
    if df.empty:
        print("\nNo axis 'MAG' rows to summarize.")
        return

    tuned = np.degrees(df.groupby('method')[metric].mean())

    benchmark_df = load_statistics(BENCHMARK_STATS_NAME)
    benchmark, benchmark_provenance = None, ""
    if benchmark_df is not None and metric in benchmark_df.columns:
        subject_names = [f"Subject{s}" for s in subjects]
        rows = benchmark_df[(benchmark_df['axis'] == 'MAG')
                            & benchmark_df['subject'].isin(subject_names)
                            & benchmark_df['trial_type'].isin(activities)]
        if not rows.empty:
            benchmark = np.degrees(rows.groupby('method')[metric].mean())
            benchmark_provenance = _benchmark_provenance()

    label = metric.replace('_rad', '')
    print(f"\nMean {label} (axis MAG, degrees), pooled over subjects / activities / joints:\n")
    header = f"  {'arm':<22} {'this run':>10}"
    if benchmark is not None:
        header += f" {'benchmark':>10} {'delta %':>9}"
    print(header)
    for method in [m for m in NORMALIZED_METHODS if m in tuned.index]:
        row = f"  {method:<22} {tuned[method]:>10.3f}"
        if benchmark is not None:
            base = base_of(method)
            if base in benchmark.index:
                reference = benchmark[base]
                row += f" {reference:>10.3f} {100 * (tuned[method] - reference) / reference:>+8.1f}%"
            else:
                row += f" {'—':>10} {'—':>9}"
        print(row)

    stds = resolve_stds(TUNED_STDS)
    print(f"\n  Tuning: gyro_std={stds['gyro_std']}, acc_std={stds['acc_std']}, "
          f"mag_std={stds['mag_std']}, normalize_measurements=True.")
    if benchmark is not None:
        print(f"  'benchmark' is {paths.statistics_path(BENCHMARK_STATS_NAME).name}, pooled over the\n"
              "  same subjects and activities — unnormalized except its EKF.")
        print(f"  {benchmark_provenance}")
        print("  Negative delta = this run beat it. Two factors moved (normalization and the\n"
              "  stds), so this ranks tunings; it does not attribute the difference — see\n"
              "  experiments/normalization_comparison.py for that.")
    else:
        print(f"  No {paths.statistics_path(BENCHMARK_STATS_NAME).name} on disk, so there is no\n"
              "  benchmark column. Run experiments/benchmark_experiment.py for the comparison.")


# ==============================================================================
# CLI / ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="The benchmark grid with normalized filter inputs and a re-tuned noise model."
    )
    parser.add_argument("--subjects", nargs='+', default=SUBJECTS, help="Subject IDs to process.")
    parser.add_argument("--activities", nargs='+', default=ACTIVITIES, help="Activities/trials to process.")
    parser.add_argument("--methods", nargs='+', default=DEFAULT_METHODS,
                         help="Methods to process. Defaults to marker plus every base under "
                              "'_normalized'.")
    parser.add_argument("--gyro-std", type=float, default=TUNED_STDS['gyro_std'])
    parser.add_argument("--acc-std", type=float, default=TUNED_STDS['acc_std'])
    parser.add_argument("--mag-std", type=float, default=TUNED_STDS['mag_std'])
    parser.add_argument("--stats-only", action="store_true",
                         help="Skip joint-angle generation and only recompile statistics from "
                              "what is already under the variant tree.")
    parser.add_argument("--save-timeseries", action="store_true",
                         help="Also write the concatenated all-method joint-angle time series "
                              "(~1 GB). Off by default: only ad-hoc scripts read it, and the "
                              "statistics table is what the figures use.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    # Fail fast on a typo rather than surfacing it as a per-cell "Failed" after the pool
    # has spun up — same up-front validation as benchmark_experiment.py.
    for method in args.methods:
        try:
            resolve_method_spec(method)
        except ValueError as e:
            print(f"Error: {e}")
            return
    for activity in args.activities:
        if activity not in ACTIVITIES:
            print(f"Error: Unknown activity '{activity}'. Allowed: {ACTIVITIES}")
            return

    stds = {'gyro_std': args.gyro_std, 'acc_std': args.acc_std, 'mag_std': args.mag_std}
    print(f"Normalized benchmark: {args.methods}")
    print(f"Tuning: {resolve_stds(stds)} (normalize_measurements comes from each method name)")
    print(f"Output namespace: {paths.JOINT_ANGLES_DIR / VARIANT} and "
          f"{paths.statistics_path(STATS_NAME)}")

    row_keys = [(subject, activity) for subject in args.subjects for activity in args.activities]

    # 1. Joint-angle generation phase
    if not args.stats_only:
        print(f"\nStarting parallel generation of joint angles for {len(row_keys)} tasks "
              f"using {args.workers} workers...")
        run_tracked_grid(row_keys, ['Subject', 'Activity'], ['load'] + args.methods,
                          partial(generate_joint_angles_worker, stds=stds, variant=VARIANT,
                                  experiment=EXPERIMENT_NAME),
                          args.workers, title="NORMALIZED BENCHMARK GENERATION")
        print("Joint angles generation phase complete.\n")

    # 2. Per-subject/activity statistics phase
    print("--- Starting Statistics Aggregation Phase ---")
    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['stats'],
                      partial(compute_stats_worker, methods=args.methods, stats_name=STATS_NAME,
                              variant=VARIANT, stds=stds, experiment=EXPERIMENT_NAME),
                      args.workers, title="NORMALIZED BENCHMARK STATISTICS")

    # 3. Global aggregation & statistics phase
    print("\n--- Starting Global Aggregation & Statistics Phase ---")

    # `stds` travels with `variant` on the read side too: these arms were generated under this
    # experiment's re-tuning, so checking them against the module default would call every one
    # of them stale.
    all_data_df = load_all_joint_angles(TRIAL_DATASET, row_keys, args.methods, variant=VARIANT,
                                        experiment=EXPERIMENT_NAME, stds=stds)
    if all_data_df.empty:
        print("Error: No data was loaded for any subject. Exiting.")
        return

    if args.save_timeseries:
        joint_angles_path = paths.ensure_parent(
            paths.STATISTICS_DIR / f"{STATS_NAME}_joint_angles.parquet")
        all_data_df.to_parquet(joint_angles_path, engine='pyarrow')
        paths.write_manifest(joint_angles_path, constants=pipeline_constants(stds),
                             methods=args.methods, subjects=args.subjects,
                             activities=args.activities, variant=VARIANT, n_rows=len(all_data_df))
        print(f"Concatenated DataFrame saved to {joint_angles_path}")

    print("\n--- Generating summary statistics... ---")
    summary_stats_df = compute_error_stats(all_data_df)
    if summary_stats_df.empty:
        print("Warning: Summary statistics DataFrame is empty.")
        return

    # The two factors this run changed, as columns, so the plotting side can group on them
    # without re-parsing method names.
    summary_stats_df['base_method'] = summary_stats_df['method'].map(base_of)
    summary_stats_df['normalization'] = 'normalized'

    stats_path = save_statistics(summary_stats_df, STATS_NAME, stds=stds,
                                 activities=args.activities, variant=VARIANT)
    print(f"Summary statistics saved to {stats_path}")

    _print_summary(summary_stats_df, args.subjects, args.activities)

    print("\nPipeline finished successfully!")


if __name__ == '__main__':
    main()
