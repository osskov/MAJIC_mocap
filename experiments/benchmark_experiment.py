"""
Vanilla full-grid pipeline (the benchmark experiment): computes joint angles for
every method in METHODS across every subject/activity, then aggregates error
statistics against the marker (mocap) ground truth. See experiments/experiment_utils.py for
the shared engine this is built on, and the rest of experiments/ for method-name
sweeps, noise-tuning sweeps, and other variations on this same pipeline.
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import argparse
from functools import partial

import paths
from experiments.experiment_utils import (
    SUBJECTS, ACTIVITIES, METHODS,
    load_all_joint_angles, compute_error_stats, save_statistics, run_tracked_grid,
    resolve_method_spec, generate_joint_angles_worker, compute_stats_worker,
)

STATS_NAME = "all_subject"

# ==============================================================================
# CLI / ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Segment Orientation and Statistics Pipeline.")
    parser.add_argument("--subjects", nargs='+', default=SUBJECTS, help="Subject IDs to process.")
    parser.add_argument("--activities", nargs='+', default=ACTIVITIES, help="Activities/trials to process.")
    parser.add_argument("--methods", nargs='+', default=list(METHODS.keys()), help="Methods to process.")
    parser.add_argument("--stats-only", action="store_true", help="Skip orientation generation and only compile statistics.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    # Validate up front, so a typo fails fast instead of surfacing as a per-cell
    # "Failed" status after workers have already spun up.
    for m in args.methods:
        try:
            resolve_method_spec(m)
        except ValueError as e:
            print(f"Error: {e}")
            return
    for a in args.activities:
        if a not in ACTIVITIES:
            print(f"Error: Unknown activity '{a}'. Allowed: {ACTIVITIES}")
            return

    row_keys = [(subject, activity) for subject in args.subjects for activity in args.activities]

    # 1. Joint-angle generation phase
    if not args.stats_only:
        print(f"Starting parallel generation of joint angles for {len(row_keys)} tasks using {args.workers} workers...")
        run_tracked_grid(row_keys, ['Subject', 'Activity'], ['load'] + args.methods, generate_joint_angles_worker,
                          args.workers, title="MAJIC MOCAP GENERATION")
        print("Joint angles generation phase complete.\n")

    # 2. Per-subject/activity statistics phase
    print("--- Starting Statistics Aggregation Phase ---")
    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['stats'],
                      partial(compute_stats_worker, methods=args.methods, stats_name=STATS_NAME),
                      args.workers, title="MAJIC MOCAP STATISTICS")

    # 3. Global aggregation & statistics phase
    print("\n--- Starting Global Aggregation & Statistics Phase ---")

    all_data_df = load_all_joint_angles(args.subjects, args.activities, args.methods)
    if all_data_df.empty:
        print("Error: No data was loaded for any subject. Exiting.")
        return

    print("--- Saving concatenated all subject joint angles ---")
    joint_angles_path = paths.ensure_parent(paths.all_subject_joint_angles_path())
    all_data_df.to_parquet(joint_angles_path, engine='pyarrow')
    paths.write_manifest(joint_angles_path, methods=args.methods, subjects=args.subjects,
                         activities=args.activities, n_rows=len(all_data_df))
    print(f"Concatenated DataFrame saved to {joint_angles_path}")

    print("\n--- Generating summary statistics... ---")
    summary_stats_df = compute_error_stats(all_data_df)
    if not summary_stats_df.empty:
        stats_path = save_statistics(summary_stats_df, STATS_NAME, activities=args.activities)
        print(f"Summary statistics saved to {stats_path}")
    else:
        print("Warning: Summary statistics DataFrame is empty.")

    print("\nPipeline finished successfully!")

if __name__ == '__main__':
    main()
