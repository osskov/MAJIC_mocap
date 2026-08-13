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
import re
import time
import sys
import textwrap
import traceback
from functools import partial
from typing import Any, Dict, List, Tuple

import paths
from rich.console import Console
from rich.table import Table
from experiments.experiment_utils import (
    TRIAL_DATASET, cached_trial_status, run_tracked_grid, save_cached_trial,
)
from experiments.experiment_utils import RESIDUAL_WARN_DEG_S
from src.toolchest.building.sources import SOURCES, get_source



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
        plates = source.load(subject, activity)
        path = save_cached_trial(plates, subject, activity, dataset=dataset)

        manifest = paths.read_manifest(path) or {}
        diagnostics = manifest.get('diagnostics', {})
        shared_state[(row_key, f"{stage}_time")] = time.time() - started
        shared_state[(row_key, stage)] = "Success"
        return {'subject': subject, 'activity': activity, 'action': 'built',
                'status': status, 'reason': reason, 'path': path,
                'bytes': path.stat().st_size, 'diagnostics': diagnostics}

    except Exception as e:
        # The traceback, not just the message. A failure deep in the reconstruction stack is
        # expensive to reproduce -- the build that surfaced it took minutes -- and str(e)
        # alone routinely does not say which of several call paths raised.
        shared_state[(row_key, stage)] = f"Failed ({e})"
        return {'subject': subject, 'activity': activity, 'action': 'failed',
                'error': str(e), 'traceback': traceback.format_exc()}


def _natural_key(name: str):
    """Split digits from text so 's2' < 's13' and 't2' < 't10'."""
    return [int(part) if part.isdigit() else part
            for part in re.split(r'(\d+)', name)]


_STATUS_MARKS = {
    'fresh':   ('[green]●[/green]', 'built and current'),
    'suspect': ('[yellow]◐[/yellow]', 'built, but a plate exceeds the residual threshold'),
    'stale':   ('[yellow]▵[/yellow]', 'built from different code or inputs'),
    'missing': ('[dim]·[/dim]', 'never built'),
    'failed':  ('[red]✗[/red]', 'the build raised'),
    'absent':  ('[dim] [/dim]', 'the source trial does not exist'),
}


def status_grid(row_keys: List[Tuple[str, str]], dataset: str,
                statuses: Dict[Tuple[str, str], str] = None) -> Table:
    """Sessions down, trials across, one mark per cell.

    262 trials is past the point where a scrolling list tells you anything -- "19 failed" is
    a number, whereas a column of red says the static poses failed and a row of it says a
    session did. `statuses` overrides what is read from the manifests, so this renders a
    build that just happened as well as a tree sitting on disk.
    """
    # Natural order, not lexicographic: 't10' sorts before 't2' as a string, which puts the
    # columns in an order nobody reading them expects.
    sessions = sorted({session for session, _ in row_keys}, key=_natural_key)
    trials = sorted({trial for _, trial in row_keys}, key=_natural_key)

    table = Table(title=f"[bold magenta]{dataset}[/bold magenta]", show_header=True,
                  header_style="bold cyan")
    table.add_column("session", style="bold")
    for trial in trials:
        # Trial names are long and highly repetitive ('t6_drop_jump_001'); the leading token
        # is what distinguishes them and is narrow enough to fit 13 across.
        table.add_column(trial.split('_')[0], justify="center")

    present = set(row_keys)
    for session in sessions:
        cells = []
        for trial in trials:
            if (session, trial) not in present:
                cells.append(_STATUS_MARKS['absent'][0])
                continue
            if statuses is not None:
                status = statuses.get((session, trial), 'missing')
            else:
                status, _ = cached_trial_status(session, trial, dataset)
                if status == 'fresh' and _has_suspect_plate(session, trial, dataset):
                    status = 'suspect'
            cells.append(_STATUS_MARKS.get(status, _STATUS_MARKS['missing'])[0])
        table.add_row(session, *cells)
    return table


def _has_suspect_plate(session: str, trial: str, dataset: str) -> bool:
    manifest = paths.read_manifest(paths.cached_trial_path(dataset, session, trial)) or {}
    return bool((manifest.get('diagnostics') or {}).get('suspect'))


def print_status_grid(row_keys: List[Tuple[str, str]], dataset: str,
                      statuses: Dict[Tuple[str, str], str] = None,
                      quiet: bool = False) -> None:
    """The grid plus its legend, or nothing at all under --quiet.

    Gated on its own flag rather than on DISABLE_TQDM, which this module sets to True at
    import so the worker pool does not each render a progress bar. Reusing it here silenced
    the grid unconditionally.
    """
    if quiet:
        return
    console = Console()
    console.print(status_grid(row_keys, dataset, statuses))
    console.print("  " + "   ".join(f"{mark} {meaning}"
                                    for mark, meaning in _STATUS_MARKS.values()),
                  highlight=False)


def report_check(row_keys: List[Tuple[str, str]], dataset: str,
                 verbose: bool = False) -> int:
    """Prints build status for every trial. Returns the number that are not usable.

    A return value so --check can gate something. It reported and exited 0 regardless, which
    makes it useless in a script: "everything is stale" and "everything is fine" were
    indistinguishable to a caller.
    """
    counts: Dict[str, int] = {}
    for subject, trial in row_keys:
        status, reason = cached_trial_status(subject, trial, dataset=dataset)
        counts[status] = counts.get(status, 0) + 1
        # The grid above already says WHICH; the per-trial reason is only interesting when
        # you are chasing one, so it is opt-in rather than 262 lines by default.
        if verbose:
            detail = f"  ({reason})" if reason else ""
            print(f"  {subject}/{trial:22s} {status}{detail}")
    print("\n" + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
    return counts.get('missing', 0) + counts.get('stale', 0)


def report_build(results: Dict[Any, Dict[str, Any]], out: str) -> int:
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
        if r.get('traceback'):
            print(textwrap.indent(r['traceback'].rstrip(), '    | '))

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
    return len(failed)


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
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the status grid, for headless and CI runs.")
    parser.add_argument("--verbose", action="store_true",
                        help="With --check, list every trial and why it is stale, not just the grid.")
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
            return 1
        row_keys = [(s, t) for s, t in row_keys if s in args.subjects]
    if args.trials:
        unknown = set(args.trials) - {t for _, t in row_keys}
        if unknown:
            print(f"Error: no such trial(s) in {args.dataset}: {sorted(unknown)}")
            return 1
        row_keys = [(s, t) for s, t in row_keys if t in args.trials]

    if not row_keys:
        print(f"No trials selected in {args.dataset}.")
        return 1

    out = f"{paths.TRIALS_DIR.relative_to(paths.REPO_ROOT)}/{args.dataset}"
    if args.check:
        print(f"Build status for {len(row_keys)} trials in {args.dataset} ({out}):\n")
        print_status_grid(row_keys, args.dataset, quiet=args.quiet)
        return 1 if report_check(row_keys, args.dataset, verbose=args.verbose) else 0

    print(f"Building {len(row_keys)} trials from {args.dataset} using {args.workers} workers...")
    _, results = run_tracked_grid(
        row_keys, ['Subject', 'Trial'], ['build'],
        partial(build_trial_worker, dataset=args.dataset, force=args.force),
        args.workers, title=f"BUILD TRIALS — {args.dataset}",
    )
    statuses = {}
    for key, result in results.items():
        if not result:
            continue
        action = result.get('action')
        statuses[key] = {'built': 'fresh', 'skipped': 'fresh', 'failed': 'failed',
                         'absent': 'absent'}.get(action, 'missing')
        if action == 'built' and result.get('diagnostics', {}).get('suspect'):
            statuses[key] = 'suspect'
    print_status_grid(row_keys, args.dataset, statuses, quiet=args.quiet)
    return 1 if report_build(results, out) else 0


if __name__ == '__main__':
    # Exit code, so a build that reports failures does not look like a success to a script.
    sys.exit(main())
