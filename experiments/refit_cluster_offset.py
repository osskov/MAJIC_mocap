"""Re-derive imove_mocap.RIGID_SENSOR_OFFSET_MM from the cached trials.

Run this when TestClusterOffset fails, or when the loader changes in a way that could move the
answer -- which it demonstrably can: replacing linear interpolation with band-limited resampling
moved these offsets by 70-90%.

DEFAULTS: the five 100 Hz long-walk sessions, omega from the GYRO, 8 Hz cutoff. All three were
measured rather than assumed, and the reasoning is recorded here because the numbers in
imove_mocap are meaningless without it.

  WHY THE GYRO. A = [alpha]x + omega omega^T - |omega|^2 I is the design matrix, so ITS noise
  biases p toward zero rather than merely scattering it. Taking omega by differencing the
  reconstructed mocap pose means alpha is a second derivative of a marker reconstruction; taking
  it from the gyroscope means alpha is a first derivative of a direct measurement. Measured over
  the 100 Hz sessions, sensitivity of the answer to the analysis cutoff -- which a rigid lever
  arm should not have at all -- falls from 77% to 26%, and on THIGH_R_M from 206% to 26%. The
  independent check is left/right agreement, since the two sides are separate hardware fitted
  separately: shanks 2.3 -> 1.1 mm, thighs 2.9 -> 0.3 mm.

  WHY 100 Hz ONLY. The 40 Hz sessions are 238 of 243 trials, and excluding them costs a lot of
  data, but they carry a bias that no cutoff removes. Decimating one 100 Hz trial to 40 Hz --
  same session, same taping, same code path, so rate is the only variable -- moves the answer by
  up to 13 mm, and moves the thighs and shanks in OPPOSITE directions. At 40 Hz the usable
  analysis band ends where the polyfit derivative gives out, well below the 20 Hz Nyquist, and
  inside that band the thigh estimates still swing 89-229% between 3 and 8 Hz.

  WHY 8 Hz. With gyro omega at 100 Hz the estimate is flat for cutoffs at or above 8 Hz (8 vs
  15 Hz: 17.0/17.5, 20.3/20.1, 21.4/19.4, 11.7/11.7). It is 3 Hz that is the outlier -- and
  3 Hz previously looked best only because it inflated BOTH rates until they met, which is why
  minimising the cross-rate gap turned out to be the wrong objective.

WHAT IS STILL NOT RIGHT. The bolted sensors ought to share one offset -- same plate, same
bracket -- and they do not: shanks ~20, pelvis ~17, thighs ~12.5 mm. |p| is rotation-invariant,
so this is not the IMUs sitting differently in their brackets. It is narrower than the 3x spread
the mocap path gave, but it is not one number, and no mechanism explains the remainder.

Not a tracked experiment. It writes no artifact and nothing depends on its output at run time;
it exists so a constant in the source has a reproducible derivation rather than a provenance
story in a comment. The fit itself is PlateTrial.fit_sensor_offset and the pooling is
imove_mocap.pool_cluster_offsets, so this file is only the loop over trials.

    python -m experiments.refit_cluster_offset
    python -m experiments.refit_cluster_offset --rates --workers 4
    python -m experiments.refit_cluster_offset --all-sessions --omega-from mocap
"""
import argparse
import os

os.environ.setdefault("DISABLE_TQDM", "True")

import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.experiment_utils import (StaleTrialCache, SuspectTrialWarning, load_trial)

# This loads every trial on purpose, so the per-trial suspect warning -- which is the right
# thing when someone loads ONE trial -- would be 151 lines of noise here.
warnings.simplefilter('ignore', SuspectTrialWarning)
from src.toolchest.building.imove_mocap import _fit_iteratively, pool_cluster_offsets
from src.toolchest.building.sources import get_source

# A static pose has no lever arm to see, so its fit is singular rather than merely noisy.
SKIP_TRIALS = {'t0_static_pose_001'}

# Progress line cadence, in trials. The fits are minutes of work with nothing to show for it
# otherwise, and the long walks alone take ~20 s each.
_REPORT_EVERY = 20


def trial_keys(dataset: str = 'imove', sessions: Optional[set] = None,
               limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """The (session, trial) pairs to fit, in enumeration order.

    `sessions` restricts to a subset, which is how the rate comparison separates the 40 Hz
    17-sensor sessions from the 100 Hz long walks. `limit` takes the first N, for a quick look.
    """
    keys = [(session, trial) for session, trial in get_source(dataset).enumerate_trials()
            if trial not in SKIP_TRIALS and (sessions is None or session in sessions)]
    return keys[:limit]


def fit_trial(task: Tuple[str, str, float, str, str]) -> Optional[Tuple[str, Dict[str, np.ndarray]]]:
    """One trial -> (session, {sensor: absolute offset in metres}), or None if it will not load.

    A separate process runs this, so it takes a plain tuple and returns plain arrays.
    """
    session, trial, cutoff_hz, dataset, omega_from = task
    try:
        plates = load_trial(session, trial, dataset=dataset)
    except StaleTrialCache:
        # Deliberately NOT swallowed. Staleness is code-wide, so catching it turns
        # "you need to rebuild" into "236 trials skipped", which reads like a data
        # problem and sent me looking in the wrong place once already. Raised from a worker it
        # still reaches the caller: the executor re-raises it in the parent process.
        raise
    except (FileNotFoundError, ValueError):
        return None

    # The build already shifted each pose onto its sensor, so a fit here returns what
    # REMAINS. The absolute offset is that plus what was applied, which the manifest
    # records per plate. Skipping this step would drive the constant to zero one refit at
    # a time, each run looking like a small correction on the last.
    applied = _applied_offsets(dataset, session, trial)
    offsets = {}
    for name, plate in plates.items():
        # ITERATED, not a single step. The fit reads about 16% low -- `A` comes from
        # twice-differentiated markers, so noise in the regressor biases least squares
        # toward zero -- and one pass therefore lands short. Taking one step here is what
        # made the previous run report 77 -> 89 mm instead of the answer: it was a Newton
        # step from the current constant rather than the fixed point.
        residual = _fit_iteratively(plate, lowpass_hz=cutoff_hz,
                                    omega_from=omega_from)
        if residual is None:
            continue
        offsets[name] = residual + applied.get(name, 0.0)
    return session, offsets


def gather(keys: List[Tuple[str, str]], cutoff_hz: float, dataset: str = 'imove',
           workers: Optional[int] = None,
           omega_from: str = 'gyro') -> Tuple[List[Tuple[str, Dict[str, np.ndarray]]], int]:
    """Fit every trial in `keys`, across processes. Returns the per-trial fits and a skip count.

    Per-trial rather than pre-pooled by sensor, because which trials belong in a pool is a
    question the caller answers more than once: `--rates` needs the same fits grouped three
    ways, and re-fitting for each grouping was three passes over the whole dataset.
    """
    tasks = [(session, trial, cutoff_hz, dataset, omega_from)
             for session, trial in keys]
    rows, skipped = [], 0
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        for done, result in enumerate(pool.map(fit_trial, tasks), start=1):
            if result is None:
                skipped += 1
            else:
                rows.append(result)
            if done % _REPORT_EVERY == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} trials fitted", flush=True)
    return rows, skipped


def pool_samples(rows: List[Tuple[str, Dict[str, np.ndarray]]],
                 sessions: Optional[set] = None) -> Dict[str, np.ndarray]:
    """{sensor: (N, 3) offsets in metres} over the given sessions, or over all of them."""
    samples: Dict[str, List[np.ndarray]] = {}
    for session, offsets in rows:
        if sessions is not None and session not in sessions:
            continue
        for name, offset in offsets.items():
            samples.setdefault(name, []).append(offset)
    return {name: np.array(values) for name, values in samples.items()}


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
    parser.add_argument('--all-sessions', action='store_true',
                        help="Include the 40 Hz sessions. Off by default -- see the module "
                             "docstring for why they are excluded.")
    parser.add_argument('--omega-from', choices=('gyro', 'mocap'), default='gyro',
                        help="Source of omega for the design matrix. 'mocap' reproduces the "
                             "pre-2026-08-13 behaviour.")
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    # The long-walk sessions are the only 100 Hz ones, and their IMU matches the mocap rate
    # exactly, so they involve no resampling at all.
    everything = {session for session, _ in get_source('imove').enumerate_trials()}
    long_walk = {session for session in everything if session.endswith('l')}

    # DEFAULT: the 100 Hz long walks only. Both halves of that choice are measured; see the
    # module docstring. Widening it is a flag rather than the default because the 40 Hz
    # sessions do not merely add noise, they add a bias that no cutoff removes.
    keys = trial_keys(sessions=None if (args.all_sessions or args.rates) else long_walk,
                      limit=args.limit)
    if args.rates:
        # Both sides of the comparison have to be represented, and under --limit the first N
        # trials need not include a single long walk. Taking N from each subset keeps a quick
        # look quick without making the comparison vacuous; with no limit this is just
        # everything, in enumeration order, once.
        ordered = trial_keys(sessions=everything - long_walk, limit=args.limit) + \
                  trial_keys(sessions=long_walk, limit=args.limit)
        keys = list(dict.fromkeys(ordered))

    for cutoff in args.cutoffs:
        print(f"Fitting {len(keys)} trials at {cutoff} Hz, omega from "
              f"{args.omega_from}, on {args.workers} workers...")
        rows, skipped = gather(keys, cutoff, workers=args.workers,
                               omega_from=args.omega_from)
        if not rows:
            print(f"No trials loaded at {cutoff} Hz ({skipped} skipped).")
            continue
        report(pool_cluster_offsets(pool_samples(rows)), cutoff)
        if skipped:
            print(f"  ({skipped} trials skipped — unbuilt, stale or unsyncable)")

        if args.rates:
            # The same fits, grouped two ways. No second pass over the data.
            rate_agreement((pool_samples(rows, everything - long_walk),
                            pool_samples(rows, long_walk)))

    print("\nPaste the BOLTED medians into imove_mocap.RIGID_SENSOR_OFFSET_MM. The taped\n"
          "sensors are fitted per trial, so their medians are only fallback nominals for\n"
          "imove_mocap.NOMINAL_SENSOR_OFFSET_MM.")


if __name__ == '__main__':
    main()
