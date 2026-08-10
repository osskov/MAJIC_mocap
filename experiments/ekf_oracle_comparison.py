"""
Compares the EKF baseline against its acc/mag oracle variants, plus mag_adapt as a
reference point: ekf (real acc/real mag), ekf_perfect_mag (real acc/oracle mag),
ekf_perfect_acc (oracle acc/real mag), and mag_adapt (real/real).

marker/ekf/mag_adapt joint angles are assumed already computed by
experiments/benchmark_experiment.py — only the two oracle variants are generated
here (pass --regenerate to recompute all four instead of trusting what's on disk).
This intentionally skips the per-subject-statistics stage (unlike
benchmark_experiment.py) since data/Subject*/​*/​subject_statistics.parquet already
holds the full vanilla method set and has no downstream reader — recomputing it
here would just narrow it to this comparison's 5 methods for no benefit.
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_utils import (
    SUBJECTS, ACTIVITIES, load_all_joint_angles, compute_error_stats, save_statistics,
    run_tracked_grid, generate_joint_angles_worker,
)

ALREADY_COMPUTED_METHODS = ['marker', 'ekf', 'mag_adapt']
ORACLE_METHODS = ['ekf_perfect_mag', 'ekf_perfect_acc']
ALL_METHODS = ALREADY_COMPUTED_METHODS + ORACLE_METHODS


def main():
    parser = argparse.ArgumentParser(
        description="Compare EKF real/real vs. its acc/mag oracle variants vs. mag_adapt."
    )
    parser.add_argument("--subjects", nargs='+', default=SUBJECTS)
    parser.add_argument("--activities", nargs='+', default=ACTIVITIES)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--regenerate", action="store_true",
                         help="Also regenerate marker/ekf/mag_adapt instead of trusting what's "
                              "already on disk from experiments/benchmark_experiment.py.")
    args = parser.parse_args()

    row_keys = [(subject, activity) for subject in args.subjects for activity in args.activities]
    to_generate = ALL_METHODS if args.regenerate else ORACLE_METHODS

    print(f"Generating joint angles for: {to_generate}")
    run_tracked_grid(row_keys, ['Subject', 'Activity'], ['load'] + to_generate, generate_joint_angles_worker,
                      args.workers, title="EKF ORACLE COMPARISON GENERATION")

    print("\n--- Aggregating statistics ---")
    all_data_df = load_all_joint_angles(args.subjects, args.activities, ALL_METHODS)
    if all_data_df.empty:
        print("Error: no data was loaded for any subject.")
        return

    summary_stats_df = compute_error_stats(all_data_df)
    if summary_stats_df.empty:
        print("Warning: summary statistics are empty.")
        return

    path = save_statistics(summary_stats_df, "ekf_oracle_comparison")
    print(f"\nSaved EKF oracle comparison statistics to {path}")


if __name__ == '__main__':
    main()
