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
    TRIAL_DATASET, cached_trial_status, run_tracked_grid, save_cached_trial,
)
from src.toolchest.building.sources import SOURCES, get_source

# Alignment residual above which a plate is called out at the end of a run. Chosen off
# the observed spread on this dataset — Subject01/walking's eight plates sit at 8-14
# deg/s low-passed — so this flags a plate that is roughly double the worst normal one,
# not one that is merely on the high side. It is a review prompt, not a pass/fail gate:
# nothing downstream reads it.
RESIDUAL_WARN_DEG_S = 25.0


def build_trial_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                       dataset: str = TRIAL_DATASET, force: bool = False) -> Dict[str, Any]:
    """stage_labels = ['build']. Builds one trial's parquet if it needs building."""
    subject, activity = row_key
    source = get_source(dataset)
    stage = stage_labels[0]
    shared_state[(row_key, stage)] = "Running"
    started = time.time()

    try:
        status, reason = cached_trial_status(subject, activity, dataset=dataset)
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

        # Straight to the dataset's reader. This is the thing that BUILDS the parquet, so
        # it is the one place that may touch source files at all — everything else in the
        # codebase goes through experiment_utils.load_trial, which reads only the artifact.
        plates = source.load(subject, activity, True)
        path = save_cached_trial(plates, subject, activity, dataset=dataset)

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


def report_check(row_keys: List[Tuple[str, str]], dataset: str) -> None:
    """Prints build status for every trial without loading or writing anything."""
    counts: Dict[str, int] = {}
    for subject, trial in row_keys:
        status, reason = cached_trial_status(subject, trial, dataset=dataset)
        counts[status] = counts.get(status, 0) + 1
        detail = f"  ({reason})" if reason else ""
        print(f"  {subject}/{trial:22s} {status}{detail}")
    print("\n" + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))


def report_build(results: Dict[Any, Dict[str, Any]], out: str) -> None:
    """Summarizes a build pass: what was written, and which plates look suspect."""
    built = [r for r in results.values() if r and r['action'] == 'built']
    skipped = [r for r in results.values() if r and r['action'] == 'skipped']
    absent = [r for r in results.values() if r and r['action'] == 'absent']
    failed = [r for r in results.values() if r and r['action'] == 'failed']

    total_bytes = sum(r['bytes'] for r in built)
    print(f"\n{len(built)} built, {len(skipped)} already fresh, {len(absent)} not in the dataset,"
          f" {len(failed)} failed"
          f"  ({total_bytes / 1e6:.0f} MB written to {out})")

    for r in failed:
        print(f"  FAILED {r['subject']}/{r['activity']}: {r['error']}")

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
            print(f"  {subject}/{activity}/{name}: {residual:.1f} deg/s")


def main():
    parser = argparse.ArgumentParser(description="Build the PlateTrial parquets.")
    parser.add_argument("--dataset", default=TRIAL_DATASET, choices=sorted(SOURCES),
                        help="Which registered dataset to build.")
    parser.add_argument("--subjects", nargs='+', default=None,
                        help="Restrict to these subject IDs (default: all the source lists).")
    parser.add_argument("--trials", nargs='+', default=None,
                        help="Restrict to these trial names (default: all the source lists).")
    parser.add_argument("--force", action="store_true", help="Rebuild even entries that are already fresh.")
    parser.add_argument("--check", action="store_true", help="Report status and exit without writing.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    source = get_source(args.dataset)

    # The source lists what is on disk, so a typo in --subjects is caught against reality
    # rather than against a constant, and a subject with only one of two trials needs no
    # special case.
    row_keys = source.enumerate_trials()
    if args.subjects:
        unknown = set(args.subjects) - {s for s, _ in row_keys}
        if unknown:
            print(f"Error: no such subject(s) in {args.dataset}: {sorted(unknown)}")
            return
        row_keys = [(s, t) for s, t in row_keys if s in args.subjects]
    if args.trials:
        unknown = set(args.trials) - {t for _, t in row_keys}
        if unknown:
            print(f"Error: no such trial(s) in {args.dataset}: {sorted(unknown)}")
            return
        row_keys = [(s, t) for s, t in row_keys if t in args.trials]

    if not row_keys:
        print(f"No trials selected in {args.dataset}.")
        return

    out = f"{paths.TRIALS_DIR.relative_to(paths.REPO_ROOT)}/{args.dataset}"
    if args.check:
        print(f"Build status for {len(row_keys)} trials in {args.dataset} ({out}):\n")
        report_check(row_keys, args.dataset)
        return

    print(f"Building {len(row_keys)} trials from {args.dataset} using {args.workers} workers...")
    _, results = run_tracked_grid(
        row_keys, ['Subject', 'Trial'], ['build'],
        partial(build_trial_worker, dataset=args.dataset, force=args.force),
        args.workers, title=f"BUILD TRIALS — {args.dataset}",
    )
    report_build(results, out)


if __name__ == '__main__':
    main()
