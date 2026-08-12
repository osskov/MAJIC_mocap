"""
Materializes each trial's fully-loaded PlateTrials to results/trials/, so the
pipeline stops re-deriving them on every run.

What gets cached is the EXPENSIVE half of a load — marker reconstruction, IMU/mocap
cross-correlation sync, sensor-to-segment alignment — not the parsing, which is
already fast. See the "Trial cache" section of experiments/experiment_utils.py for
the reasoning and experiments/../src/toolchest/trial_io.py for the column layout.

Every artifact gets the usual provenance sidecar, plus two things specific to this
one: a `cache_key` recording the inputs and code version it was built from (checked
on load, so a stale entry is a miss rather than a wrong answer), and `diagnostics`
holding per-plate alignment residuals for triage.

    python -m experiments.cache_trials                 # build whatever is missing or stale
    python -m experiments.cache_trials --check         # report status, write nothing
    python -m experiments.cache_trials --force         # rebuild everything
    python -m experiments.cache_trials --subjects 01 02 --activities walking

Nothing else has to change to benefit: `load_raw_data` consults the cache already,
and falls back to loading from source when there is nothing valid to use.
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import argparse
import time
from functools import partial
from typing import Any, Dict, List, Tuple

import paths
from experiments.experiment_utils import (
    SUBJECTS, ACTIVITIES, TRIAL_DATASET,
    cached_trial_status, load_raw_data, run_tracked_grid, save_cached_trial,
)

# Alignment residual above which a plate is called out at the end of a run. Chosen off
# the observed spread on this dataset — Subject01/walking's eight plates sit at 8-14
# deg/s low-passed — so this flags a plate that is roughly double the worst normal one,
# not one that is merely on the high side. It is a review prompt, not a pass/fail gate:
# nothing downstream reads it.
RESIDUAL_WARN_DEG_S = 25.0


def cache_trial_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                       force: bool = False) -> Dict[str, Any]:
    """stage_labels = ['cache']. Builds one trial's cache entry if it needs building."""
    subject, activity = row_key
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()

    try:
        status, reason = cached_trial_status(subject, activity)
        if status == 'absent':
            # Not an error: SUBJECTS x ACTIVITIES is a full cross product and the
            # dataset is not (Subjects 05, 08 and 10 have walking only).
            shared_state[(row_key, stage)] = "Skipped"
            return {'subject': subject, 'activity': activity, 'action': 'absent',
                    'status': status, 'reason': reason}
        if status == 'fresh' and not force:
            shared_state[(row_key, stage)] = "Skipped"
            return {'subject': subject, 'activity': activity, 'action': 'skipped',
                    'status': status, 'reason': reason}

        # use_cache=False: this is the thing that BUILDS the cache, so reading it would
        # either be a no-op or, under --force, quietly re-serialize a stale entry
        # instead of rebuilding from source.
        plates = load_raw_data(subject, activity, use_cache=False)
        path = save_cached_trial(plates, subject, activity)

        manifest = paths.read_manifest(path) or {}
        diagnostics = manifest.get('diagnostics', {})
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success"
        return {'subject': subject, 'activity': activity, 'action': 'built',
                'status': status, 'reason': reason, 'path': path,
                'bytes': path.stat().st_size, 'diagnostics': diagnostics}

    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
        return {'subject': subject, 'activity': activity, 'action': 'failed', 'error': str(e)}


def report_check(row_keys: List[Tuple[str, str]]) -> None:
    """Prints cache status for every trial without loading or writing anything."""
    counts: Dict[str, int] = {}
    for subject, activity in row_keys:
        status, reason = cached_trial_status(subject, activity)
        counts[status] = counts.get(status, 0) + 1
        detail = f"  ({reason})" if reason else ""
        print(f"  Subject{subject}/{activity:14s} {status}{detail}")
    print("\n" + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))


def report_build(results: Dict[Any, Dict[str, Any]]) -> None:
    """Summarizes a build pass: what was written, and which plates look suspect."""
    built = [r for r in results.values() if r and r['action'] == 'built']
    skipped = [r for r in results.values() if r and r['action'] == 'skipped']
    absent = [r for r in results.values() if r and r['action'] == 'absent']
    failed = [r for r in results.values() if r and r['action'] == 'failed']

    total_bytes = sum(r['bytes'] for r in built)
    print(f"\n{len(built)} built, {len(skipped)} already fresh, {len(absent)} not in the dataset,"
          f" {len(failed)} failed"
          f"  ({total_bytes / 1e6:.0f} MB written to {paths.TRIALS_DIR.relative_to(paths.REPO_ROOT)}/{TRIAL_DATASET})")

    for r in failed:
        print(f"  FAILED Subject{r['subject']}/{r['activity']}: {r['error']}")

    suspect = [
        (r['subject'], r['activity'], name, stats['gyro_residual_lowpass_rms_deg_s'])
        for r in built
        for name, stats in r['diagnostics'].get('plates', {}).items()
        if stats.get('gyro_residual_lowpass_rms_deg_s', 0.0) > RESIDUAL_WARN_DEG_S
    ]
    if suspect:
        print(f"\nPlates with a low-passed gyro alignment residual over {RESIDUAL_WARN_DEG_S} deg/s —"
              f" worth a look before trusting their joint angles:")
        for subject, activity, name, residual in sorted(suspect, key=lambda t: -t[3]):
            print(f"  Subject{subject}/{activity}/{name}: {residual:.1f} deg/s")


def main():
    parser = argparse.ArgumentParser(description="Build the PlateTrial parquet cache.")
    parser.add_argument("--subjects", nargs='+', default=SUBJECTS, help="Subject IDs to cache.")
    parser.add_argument("--activities", nargs='+', default=ACTIVITIES, help="Activities to cache.")
    parser.add_argument("--force", action="store_true", help="Rebuild even entries that are already fresh.")
    parser.add_argument("--check", action="store_true", help="Report cache status and exit without writing.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    for a in args.activities:
        if a not in ACTIVITIES:
            print(f"Error: Unknown activity '{a}'. Allowed: {ACTIVITIES}")
            return

    row_keys = [(subject, activity) for subject in args.subjects for activity in args.activities]

    if args.check:
        print(f"Trial cache status for {len(row_keys)} trials "
              f"({paths.TRIALS_DIR.relative_to(paths.REPO_ROOT)}/{TRIAL_DATASET}):\n")
        report_check(row_keys)
        return

    print(f"Caching {len(row_keys)} trials using {args.workers} workers...")
    _, results = run_tracked_grid(
        row_keys, ['Subject', 'Activity'], ['cache'],
        partial(cache_trial_worker, force=args.force),
        args.workers, title="MAJIC MOCAP TRIAL CACHE",
    )
    report_build(results)


if __name__ == '__main__':
    main()
