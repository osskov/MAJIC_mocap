"""Re-derive imove_mocap.RIGID_SENSOR_OFFSET_MM from the cached trials.

Run this when TestClusterOffset fails, or when the loader changes in a way that could move
the answer -- which it demonstrably can: replacing linear interpolation with band-limited
resampling moved these offsets by 70-90%.

Not a tracked experiment. It writes no artifact and nothing depends on its output at run
time; it exists so a constant in the source has a reproducible derivation rather than a
provenance story in a comment. The fit itself is PlateTrial.fit_sensor_offset, and the
pooling is imove_mocap.pool_cluster_offsets, so this file is only the loop over trials.

    python -m experiments.refit_cluster_offset
    python -m experiments.refit_cluster_offset --cutoffs 5 8 12 --limit 40
"""
import argparse
import os

os.environ.setdefault("DISABLE_TQDM", "True")

import warnings

import numpy as np

from experiments.experiment_utils import (StaleTrialCache, SuspectTrialWarning, load_trial)

# This loads every trial on purpose, so the per-trial suspect warning -- which is the right
# thing when someone loads ONE trial -- would be 151 lines of noise here.
warnings.simplefilter('ignore', SuspectTrialWarning)
from src.toolchest.building.imove_mocap import _fit_iteratively, pool_cluster_offsets
from src.toolchest.building.sources import get_source

# A static pose has no lever arm to see, so its fit is singular rather than merely noisy.
SKIP_TRIALS = {'t0_static_pose_001'}


def collect(cutoff_hz, limit=None, dataset='imove', sessions=None):
    """{sensor: (N, 3) offsets in metres} over every cached trial that will load.

    `sessions` restricts to a subset, which is how the rate comparison separates the 40 Hz
    17-sensor sessions from the 100 Hz long walks.
    """
    samples, skipped = {}, 0
    trials = [(s, t) for s, t in get_source(dataset).enumerate_trials()
              if t not in SKIP_TRIALS and (sessions is None or s in sessions)]
    for session, trial in trials[:limit]:
        try:
            plates = load_trial(session, trial, dataset=dataset)
        except StaleTrialCache:
            # Deliberately NOT swallowed. Staleness is code-wide, so catching it turns
            # "you need to rebuild" into "236 trials skipped", which reads like a data
            # problem and sent me looking in the wrong place once already.
            raise
        except (FileNotFoundError, ValueError):
            skipped += 1
            continue
        # The build already shifted each pose onto its sensor, so a fit here returns what
        # REMAINS. The absolute offset is that plus what was applied, which the manifest
        # records per plate. Skipping this step would drive the constant to zero one refit at
        # a time, each run looking like a small correction on the last.
        applied = _applied_offsets(dataset, session, trial)
        for name, plate in plates.items():
            # ITERATED, not a single step. The fit reads about 16% low -- `A` comes from
            # twice-differentiated markers, so noise in the regressor biases least squares
            # toward zero -- and one pass therefore lands short. Taking one step here is what
            # made the previous run report 77 -> 89 mm instead of the answer: it was a Newton
            # step from the current constant rather than the fixed point.
            residual = _fit_iteratively(plate, lowpass_hz=cutoff_hz)
            if residual is None:
                continue
            samples.setdefault(name, []).append(residual + applied.get(name, 0.0))
    return {name: np.array(values) for name, values in samples.items()}, skipped


def _applied_offsets(dataset, session, trial):
    """{sensor: offset in metres} that the build applied, from the manifest."""
    import paths
    manifest = paths.read_manifest(paths.cached_trial_path(dataset, session, trial)) or {}
    plates = (manifest.get('diagnostics') or {}).get('plates') or {}
    return {name: np.asarray(stats.get('sensor_offset_mm', (0.0, 0.0, 0.0))) / 1000.0
            for name, stats in plates.items()}


def report(pooled, cutoff_hz):
    print(f"\n--- {cutoff_hz} Hz " + "-" * 62)
    print(f"{'sensor':12s} {'x':>7s} {'y':>7s} {'z':>7s} {'|p|':>7s}   "
          f"{'sx':>5s} {'sy':>5s} {'sz':>5s} {'n':>5s}")
    for sensor in sorted(pooled):
        entry = pooled[sensor]
        median, spread = entry['median_mm'], entry['spread_mm']
        print(f"{sensor:12s} {median[0]:7.1f} {median[1]:7.1f} {median[2]:7.1f} "
              f"{np.linalg.norm(median):7.1f}   {spread[0]:5.1f} {spread[1]:5.1f} "
              f"{spread[2]:5.1f} {entry['n']:5d}")


def rate_agreement(samples_by_rate):
    """How far apart the 40 Hz and 100 Hz sessions land, per sensor.

    The sharpest available check on the resampling: the same physical unit measured through
    two different rate-conversion paths must give the same offset. It read 12 mm apart while
    the resampler was doing linear interpolation.
    """
    slow, fast = samples_by_rate
    print(f"\n{'sensor':12s} {'40 Hz |p|':>10s} {'100 Hz |p|':>11s} {'gap':>7s}")
    for sensor in sorted(set(slow) & set(fast)):
        a = np.linalg.norm(np.median(slow[sensor], axis=0)) * 1000
        b = np.linalg.norm(np.median(fast[sensor], axis=0)) * 1000
        print(f"{sensor:12s} {a:10.1f} {b:11.1f} {a - b:7.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', type=float, nargs='+', default=[8.0])
    parser.add_argument('--limit', type=int, default=None,
                        help="Only the first N trials, for a quick look.")
    parser.add_argument('--rates', action='store_true',
                        help="Compare the 40 Hz sessions against the 100 Hz long walks.")
    args = parser.parse_args()

    for cutoff in args.cutoffs:
        samples, skipped = collect(cutoff, args.limit)
        if not samples:
            print(f"No trials loaded at {cutoff} Hz ({skipped} skipped).")
            continue
        report(pool_cluster_offsets(samples), cutoff)
        if skipped:
            print(f"  ({skipped} trials skipped — unbuilt, stale or unsyncable)")

        if args.rates:
            # The long-walk sessions are the only 100 Hz ones, and their IMU matches the
            # mocap rate exactly, so they involve no resampling at all.
            everything = [s for s, _ in get_source('imove').enumerate_trials()]
            long_walk = {s for s in everything if s.endswith('l')}
            slow, _ = collect(cutoff, args.limit,
                              sessions=set(everything) - long_walk)
            fast, _ = collect(cutoff, args.limit, sessions=long_walk)
            rate_agreement((slow, fast))

    print("\nPaste the BOLTED medians into imove_mocap.RIGID_SENSOR_OFFSET_MM. The taped\n"
          "sensors are fitted per trial, so their medians are only fallback nominals for\n"
          "imove_mocap.NOMINAL_SENSOR_OFFSET_MM.")


if __name__ == '__main__':
    main()
