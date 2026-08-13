"""
Shared engine for every "run the filters over subjects/activities" script in this
repo — the vanilla full-grid pipeline (experiments/benchmark_experiment.py) and every
experiment under experiments/ (noise sensitivity, threshold sensitivity, oracle
ablation, drift observability). Anything that's physics, IO, or orchestration and
would otherwise be copy-pasted or reached into via private imports lives here.

All filesystem paths come from paths.py: `data/` is read-only source data and every
generated artifact lands under `results/`, with a provenance manifest sidecar.
"""
import os
from pathlib import Path
os.environ["DISABLE_TQDM"] = "True"
import hashlib
import re
import time
import multiprocessing
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from rich.live import Live
from rich.table import Table

import paths
from paths import (DATA_DIR, RESULTS_DIR, raw_trial_dir, ensure_parent, write_manifest,
                   read_manifest)
from src.toolchest.building.sources import SOURCES, TrialSource, get_source
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from src.toolchest.gyro_utils import relative_rotvec
from src.toolchest import trial_io
from src.RelativeFilterPlus import RelativeFilter
from src import relative_filter_fast

# ==============================================================================
# CONFIGURATION
# ==============================================================================

JOINTS = {
    'Lumbar': ('pelvis_imu', 'torso_imu'),
    'R_Hip':  ('pelvis_imu', 'femur_r_imu'),
    'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
    'R_Ankle':('tibia_r_imu', 'calcn_r_imu'),
    'L_Hip':  ('pelvis_imu', 'femur_l_imu'),
    'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
    'L_Ankle':('tibia_l_imu', 'calcn_l_imu'),
}

SUBJECTS = [f'{i:02d}' for i in range(1, 12)]
ACTIVITIES = ['walking', 'complexTasks']

# Gravity as an accelerometer reads it, expressed in the mocap world frame.
#
# Two conventions are baked in here and both matter:
#   * Axis   — this dataset's world frame is Y-UP (the .trc mocap frame). Measured
#              across all 11 subjects x 151 sensor-trials, the mean world-frame
#              accelerometer vector is [+0.016, +9.812, -0.003] (sd 0.035).
#   * Sign   — POSITIVE, i.e. the specific-force convention: a stationary sensor
#              reads +9.81 along the up axis, not -9.81. Note this is the opposite
#              sign to the `acc_from_gravity=[0, 0, -9.81]` used by the synthetic
#              fixture generators in src/toolchest/PlateTrial.py, which live in
#              their own self-consistent synthetic world and are test-only.
#
# Getting either wrong is silent: it does not raise, it just corrupts the EKF's
# world reference and both acc oracles. Verified against this dataset by
# test/TestGravityConvention.py; if you point the pipeline at different data, run
# check_gravity_convention() below against it before trusting any output.
EXPECTED_GRAVITY = np.array([0.0, 9.81, 0.0])

# Default filter tuning. Hoisted out of _run_relative_filter's signature so the
# values can be recorded in every output's provenance manifest — their meaning
# depends on whether RelativeFilter normalizes its vector measurements, which is
# not visible from here.
DEFAULT_GYRO_STD = 0.0045
DEFAULT_ACC_STD = 0.018
DEFAULT_MAG_STD = 0.05

# Nominal magnitude of each vector sensor's reading, used ONLY by the '_rescaled' method
# arm to convert the absolute stds above into the units a unit-length measurement lives
# in (std / nominal). Without that conversion, normalizing silently retunes the filter —
# see RelativeFilter._normalize_vector_measurements.
#
# These are deliberately FIXED constants rather than each sample's own |v|. Rescaling by
# the actual per-sample magnitude would be an exact algebraic no-op: h, H and R would all
# scale together, leaving K @ e and K @ H untouched, so that arm would reproduce the
# unnormalized one bit for bit. Holding the nominal fixed preserves the AVERAGE weighting
# while still discarding the per-sample magnitude — which is the geometric change
# normalization actually makes, and the only thing the rescaled arm is meant to isolate.
#
# Acc is gravity's magnitude. Mag is 1.0 because these are Xsens exports, whose
# magnetometer channels are normalized at calibration so a nominal Earth field reads 1.0
# (see MAG_UNIT in experiments/sensor_distributions.py); measured |mag| medians across
# this dataset run 0.4-1.1, so this is a nominal, not a per-sensor calibration.
NOMINAL_ACC_MAGNITUDE = float(np.linalg.norm(EXPECTED_GRAVITY))
NOMINAL_MAG_MAGNITUDE = 1.0

# The o^J value above which mag_adapt stops trusting the magnetometer, in the units of
# segment_observability, i.e. (m/s^2)(m/s^3).
#
# This is NOT comparable to the 150.0 used before the missing-dt fix in
# segment_observability: that metric was ~100x too small in its difference term and
# ranked samples differently (the two correlate only r~0.55 on real trials), so no
# threshold reproduces the old gating exactly.
#
# 1000 was chosen to preserve the DUTY CYCLE the method was tuned around rather than the
# number: 150.0 on the old metric gated 18-22% of samples across trials, and 1000 on the
# corrected metric gates 18-22% (per-trial duty-matched values 956 / 1024 / 1228 for
# Subject01 walking, Subject05 walking, Subject01 complexTasks). For scale, the corrected
# metric's percentiles over a full trial are roughly p25=130, p50=300, p90=2100, and its
# static noise floor is 15-35.
#
# This is a default, not a claim of optimality — experiments/threshold_sensitivity.py
# sweeps it, and that sweep's range was rescaled alongside this constant. Re-run it
# before quoting any threshold as chosen.
DEFAULT_MAG_ADAPT_THRESHOLD = 1000.0


STD_KEYS = ('gyro_std', 'acc_std', 'mag_std')


def resolve_stds(stds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """The three filter stds in force for a run: the DEFAULT_* constants, with any
    key present in `stds` overriding its default.

    Exists so a re-tuning is passed as data (one dict, threaded down to
    _run_relative_filter and into every manifest) rather than by reassigning the
    module constants, which multiprocessing workers would not see — each worker
    re-imports this module in a fresh interpreter, so a parent-process mutation of
    DEFAULT_ACC_STD would silently run the default tuning in every child while the
    parent's manifest claimed the override.

    Unknown keys raise: 'acc_stdev' or 'mag' would otherwise be dropped on the floor
    and the run would quietly use the default for that sensor."""
    resolved = {'gyro_std': DEFAULT_GYRO_STD, 'acc_std': DEFAULT_ACC_STD, 'mag_std': DEFAULT_MAG_STD}
    if stds:
        unknown = sorted(set(stds) - set(STD_KEYS))
        if unknown:
            raise ValueError(f"Unknown filter std key(s) {unknown}. Allowed: {list(STD_KEYS)}")
        resolved.update({key: float(value) for key, value in stds.items()})
    return resolved


def pipeline_constants(stds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """The physical/tuning constants in force, for provenance manifests.

    `stds` records an override rather than the defaults — a manifest that reported
    DEFAULT_ACC_STD next to a run tuned elsewhere is worse than no manifest."""
    return {
        'expected_gravity': EXPECTED_GRAVITY.tolist(),
        **resolve_stds(stds),
        'mag_adapt_threshold': DEFAULT_MAG_ADAPT_THRESHOLD,
    }

# Per-base defaults. 'normalize_measurements' is set here rather than being uniformly
# False because the EKF needs a different accelerometer weighting than the relative
# filters do — see the note below.
METHODS = {
    'marker':      {'kind': 'marker'},
    'mag_on':      {'kind': 'filter', 'project': True,  'mag_mode': 'on'},
    'mag_off':     {'kind': 'filter', 'project': True,  'mag_mode': 'off'},
    'mag_adapt':   {'kind': 'filter', 'project': True,  'mag_mode': 'adapt'},
    'ekf':         {'kind': 'ekf', 'normalize_measurements': True},
}

# ==============================================================================
# Method name resolution
# ==============================================================================
# A method name is a base (one of METHODS) plus optional suffixes, in this fixed
# order: a measurement-normalization flag, then an observability threshold override
# (mag_adapt only), then a magnetic-distortion scale, then an accelerometer oracle flag,
# then a magnetometer oracle flag.
# Examples:
#   'mag_adapt_th50.00'                    -> base mag_adapt, threshold=50.0
#   'mag_on_dist0.50'                       -> base mag_on with the magnetometer's estimated
#                                              distortion halved (see _compute_scaled_mag).
#                                              'dist0.00' IS the mag oracle and 'dist1.00'
#                                              IS the real reading, both exactly, so the
#                                              sweep's endpoints coincide with existing arms
#   'ekf_perfect_acc_perfect_mag'           -> base ekf, both oracle
#   'mag_on_real_acc_perfect_mag'           -> base mag_on, mag oracle only
#   'mag_off_perfect_acc'                   -> base mag_off, acc oracle only (mag_off
#                                              always zeroes mag regardless, so a mag
#                                              suffix would be a no-op and is omitted)
#   'mag_off_normalized'                    -> base mag_off, unit-length acc/mag into
#                                              the measurement update
#   'mag_off_unnormalized'                  -> base mag_off, raw-magnitude acc/mag; the
#                                              same thing a bare 'mag_off' does, spelled
#                                              out so a normalization comparison can name
#                                              every arm symmetrically
#   'ekf_unnormalized'                      -> base ekf with normalization turned OFF. Not
#                                              the same as a bare 'ekf', which normalizes
#                                              by default (see the METHODS note above) —
#                                              the suffix overrides the base default
#   'mag_off_rescaled'                      -> base mag_off, unit-length acc/mag AND each
#                                              std divided by that sensor's nominal
#                                              magnitude, so the weighting matches the
#                                              unnormalized arm and only the geometry moves

_METHOD_SUFFIX_RE = re.compile(
    r'^(?P<base>.+?)'
    r'(?:_(?P<norm>unnormalized|normalized|rescaled))?'
    r'(?:_th(?P<threshold>[\d.]+))?'
    r'(?:_dist(?P<distortion>[\d.]+))?'
    r'(?:_(?P<acc_src>real|perfect)_acc)?'
    r'(?:_(?P<mag_src>real|perfect)_mag)?$'
)


def resolve_method_spec(method: str) -> Dict[str, Any]:
    """Parses a method name into a spec dict: the base METHODS entry plus
    'acc_source' / 'mag_source' ('real', 'perfect' or 'scaled', default 'real'),
    'normalize_measurements' / 'rescale_stds' (both default False), for mag_adapt an
    overridden 'mag_adapt_threshold' if a _th<value> suffix is present, and for a
    _dist<value> suffix a 'mag_distortion_scale' with mag_source set to 'scaled'.

    The '_rescaled' arm sets both normalization flags: it is '_normalized' plus the std
    conversion that keeps its sensor weighting equal to the '_unnormalized' arm's."""
    m = _METHOD_SUFFIX_RE.match(method)
    if m is None:  # only reachable for an empty name; every other string has a base
        raise ValueError(f"Unparseable method name '{method}'. Allowed bases: {list(METHODS)}")
    base = m.group('base')
    if base not in METHODS:
        raise ValueError(f"Unknown method '{method}' (base '{base}' not recognized). Allowed bases: {list(METHODS)}")
    spec = dict(METHODS[base])
    spec['acc_source'] = m.group('acc_src') or 'real'
    spec['mag_source'] = m.group('mag_src') or 'real'
    # The base may carry its own normalization default (the EKF does); an explicit suffix
    # overrides it, and its absence leaves the base default alone.
    spec.setdefault('normalize_measurements', False)
    spec.setdefault('rescale_stds', False)
    if m.group('norm') is not None:
        spec['normalize_measurements'] = m.group('norm') in ('normalized', 'rescaled')
        spec['rescale_stds'] = m.group('norm') == 'rescaled'
    if m.group('threshold') is not None:
        spec['mag_adapt_threshold'] = float(m.group('threshold'))
    if m.group('distortion') is not None:
        # Both suffixes replace the magnetometer wholesale, so a name carrying both asks for
        # two different readings at once. Raising is the only safe answer: silently letting
        # one win would run 'mag_on_dist0.50_perfect_mag' as an oracle arm under a name that
        # says it is a 50%-distortion arm, and the sweep would read a flat curve off it.
        if m.group('mag_src') == 'perfect':
            raise ValueError(
                f"Method '{method}' asks for both a distortion scale and the mag oracle. "
                f"They are the same knob: '_dist0.00' IS '_perfect_mag'. Use one."
            )
        spec['mag_source'] = 'scaled'
        spec['mag_distortion_scale'] = float(m.group('distortion'))
    return spec

# ==============================================================================
# Trial cache
# ==============================================================================
# Loading a trial from source costs ~8 s: ~2 s parsing the Xsens .txts, ~3 s parsing
# and reconstructing the .trc's marker plates, ~3 s on cross-correlation sync and
# sensor-to-segment alignment. The parsing is not the interesting part — the derived
# steps are, because they are DECISIONS (which lag, which marker was faulty, which
# blocks were half-turn-flipped) that the pipeline currently makes silently on every
# run and never records.
#
# Caching the finished PlateTrials fixes both: runs get faster, and the decisions
# land in a manifest you can inspect, diff and — once the many-to-one IMoVE trials
# arrive, where a 2.6 h IMU record has to be matched against ~27 mocap trials — correct
# by hand instead of re-deriving from scratch every time.

# Namespace for this repo's own source tree, from Al Borno et al. (2022). The IMoVE
# trees will add their own; the namespace exists because both contain a 'Subject01'.
TRIAL_DATASET = 'alborno'

# Toolchest modules whose code determines the CONTENT of a cached trial. Their bytes
# are hashed into the cache key, so editing any of them invalidates every artifact.
#
# Hashing whole files is deliberately blunt — a docstring edit invalidates the cache
# just as a threshold change does. The precise alternative (enumerate the constants
# that matter: _reconstruct_from_markers' fault threshold, MAX_RECONSTRUCTION_ANGULAR_
# SPEED_DEG_S, GLITCH_DILATION_FRAMES, MAX_FLIP_SNAP_RESIDUAL_DEG, the sync and
# resample logic) fails open: someone adds a constant, forgets to list it here, and
# every downstream result is quietly computed from stale inputs. Blunt-and-safe wins
# because a rebuild is 8 s per trial and a silently stale cache is a retracted figure.
#
# Paths are relative to src/toolchest. The whole building/ package is in here because it
# now owns every step between a file on disk and a PlateTrial — parsing, reconstruction,
# sync, alignment. It determines a cached trial's contents as directly as the physics does,
# and leaving any of it out would be exactly the fail-open hole this list exists to avoid.
_CONTENT_MODULES = ('PlateTrial.py', 'WorldTrace.py', 'IMUTrace.py',
                    'gyro_utils.py', 'finite_difference_utils.py',
                    'building/xsens.py', 'building/reconstruction.py',
                    'building/assembly.py', 'building/alborno.py')

# Cutoff for the alignment-residual diagnostic below.
_RESIDUAL_LOWPASS_HZ = 10.0


@lru_cache(maxsize=1)
def _toolchest_digest() -> str:
    """SHA-256 over the toolchest modules that produce a cached trial's contents."""
    digest = hashlib.sha256()
    toolchest = Path(__file__).resolve().parent.parent / 'src' / 'toolchest'
    for name in sorted(_CONTENT_MODULES):
        digest.update(name.encode())
        digest.update((toolchest / name).read_bytes())
    return digest.hexdigest()[:16]


# The files a trial load actually reads, as globs relative to the trial folder. These
# mirror IMUTrace.from_folder (the 'imu data' subdirectory, or the folder itself when
# that is absent) and PlateTrial.from_folder (the .trc).
#
# Deliberately narrower than "everything under the folder": the trial directories also
# hold a 66 MB .mtb (the raw Xsens binary, never parsed) and a 'madgwick (al borno)/'
# subdirectory of third-party outputs whose filenames COLLIDE with the real IMU ones.
# Keying on those would invalidate every cache entry whenever an unrelated file moved.
#
# Narrowing is safe here in a way it was not for the constants above, because the fail
# case is covered from the other side: if the loader ever starts reading a new file,
# that is a change to IMUTrace.py or PlateTrial.py, and the toolchest digest invalidates
# everything on its own.
def _source_inventory(folder: Path, globs: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Names and byte counts of the files a trial load reads, sorted.

    `globs` come from the dataset's TrialSource, because which files matter is a property
    of the dataset, not of the cache. Al Borno's are `*.trc` and `imu data/*.txt`; IMoVE's
    will be `mocap_data/*.csv` and `imu_data/*.txt`.

    Sizes rather than content hashes: `data/` is declared read-only (see paths.py), so the
    realistic failure is a file being replaced or a re-download landing a different trial,
    both of which change the size. Hashing 76 MB of .trc on every cache check would buy
    protection against an edit the repo's own rules forbid.
    """
    found = {p for glob in globs for p in folder.glob(glob)}
    return [{'name': str(p.relative_to(folder)), 'bytes': p.stat().st_size}
            for p in sorted(found) if p.is_file() and not p.name.startswith('.')]


def trial_cache_key(subject: str, trial: str, align: bool,
                    dataset: str = TRIAL_DATASET) -> Dict[str, Any]:
    """Everything that determines a cached trial's contents.

    Compared field-by-field against the stored manifest on load; any difference is a cache
    miss. `align` is in here because the sensor-to-segment alignment rewrites every rotation
    in the file, and `dataset` selects which source's layout and globs to key against.
    """
    source = get_source(dataset)
    return {
        'schema_version': trial_io.SCHEMA_VERSION,
        'toolchest_digest': _toolchest_digest(),
        'align_plate_trials': align,
        'sources': _source_inventory(source.source_dir(subject, trial), source.source_globs),
    }


def _alignment_residuals(plate: PlateTrial) -> Dict[str, float]:
    """How well a plate's measured gyro matches the one implied by its mocap rotations.

    This is the number to triage on: after `assembly.align_world_to_imu` the two
    should agree, and a plate where they do not has a bad sync lag, a bad alignment, or
    corrupt marker reconstruction underneath it.

    Reported both raw and low-passed, because the raw figure is misleading on its own.
    The mocap-derived gyro comes from finite-differencing rotations, which amplifies
    marker noise at high frequency: on Subject01/walking the raw residual is 35-78 deg/s
    against a signal of 55-161 deg/s, but 8-14 deg/s below 10 Hz. The raw number is
    differentiation noise; the low-passed one is alignment quality.
    """
    measured = np.asarray(plate.imu_trace.gyro, dtype=np.float64)
    implied = plate.world_trace.calculate_imu_trace(skip_lin_acc=True).gyro
    residual = measured - implied

    # Scored over valid frames only. An interpolated or corrupt pose has no measured
    # gyro to disagree with, so including it reports reconstruction damage as though it
    # were misalignment — which is the confusion this diagnostic exists to avoid.
    # Filtering still runs on the FULL trace: filtfilt needs the uniform grid, and
    # dropping frames first would splice unrelated motion together at the seam.
    valid = plate.valid

    def rms(v: np.ndarray) -> float:
        scored = v[valid]
        if not len(scored):
            return float('nan')
        return float(np.degrees(np.sqrt((np.linalg.norm(scored, axis=1) ** 2).mean())))

    out = {'gyro_residual_raw_rms_deg_s': rms(residual),
           'n_invalid_frames': int((~valid).sum())}

    fs = float(plate.imu_trace.get_sample_frequency())
    cutoff = min(_RESIDUAL_LOWPASS_HZ, 0.4 * fs / 2.0)
    # filtfilt's default padlen is 3 * max(len(a), len(b)); a trace shorter than that
    # raises rather than returning something approximate.
    if fs > 0 and len(plate) > 30:
        b, a = butter(4, cutoff / (fs / 2.0), btype='low')
        out['gyro_residual_lowpass_rms_deg_s'] = rms(filtfilt(b, a, residual, axis=0))
        out['residual_lowpass_hz'] = cutoff
    return out


def trial_diagnostics(plates: Dict[str, PlateTrial]) -> Dict[str, Any]:
    """Per-trial and per-plate quality numbers, recorded in the cache manifest.

    The point is triage at scale: with 22 trials you notice a bad one by eye, with the
    660-odd IMoVE adds. Having these in the manifests means a bad sync is a query over
    sidecars rather than a filter run that produces nonsense.
    """
    any_plate = next(iter(plates.values()))
    per_plate = {}
    for name in sorted(plates):
        plate = plates[name]
        per_plate[name] = {
            'n_frames': len(plate),
            'acc_norm_median': float(np.median(np.linalg.norm(plate.imu_trace.acc, axis=1))),
            'mag_norm_median': float(np.median(np.linalg.norm(plate.imu_trace.mag, axis=1))),
            **_alignment_residuals(plate),
        }
    return {
        'n_plates': len(plates),
        'n_frames': len(any_plate),
        'duration_s': float(any_plate.imu_trace.timestamps[-1] - any_plate.imu_trace.timestamps[0]),
        'sample_rate_hz': float(any_plate.imu_trace.get_sample_frequency()),
        'world_frame_gravity': measure_world_frame_gravity(plates).tolist(),
        # Frames invalid on ANY plate: a joint angle needs two plates, so one bad plate
        # takes the whole frame out of every joint it participates in.
        'n_invalid_frames_any_plate': int(sum(
            ~np.logical_and.reduce([p.valid for p in plates.values()]))),
        'plates': per_plate,
    }


def save_cached_trial(plates: Dict[str, PlateTrial], subject: str, trial: str,
                      align: bool = True, dataset: str = TRIAL_DATASET) -> Path:
    """Writes a trial's PlateTrials plus the manifest that validates them on load."""
    source = get_source(dataset)
    path = ensure_parent(paths.cached_trial_path(dataset, subject, trial))
    trial_io.plates_to_frame(plates).to_parquet(path, engine='pyarrow', index=False)
    write_manifest(
        path,
        cache_key=trial_cache_key(subject, trial, align, dataset),
        dataset=dataset, subject=subject, trial=trial,
        source=str(source.source_dir(subject, trial).relative_to(paths.REPO_ROOT)),
        diagnostics=trial_diagnostics(plates),
    )
    return path


def cached_trial_status(subject: str, trial: str, align: bool = True,
                        dataset: str = TRIAL_DATASET) -> Tuple[str, Optional[str]]:
    """(status, reason) for one trial's cache entry, without loading the parquet.

    status is one of:
      'fresh'   — usable as-is
      'missing' — nothing cached yet
      'stale'   — cached, but built from different inputs or code; reason names the field
      'absent'  — the SOURCE trial does not exist, so there is nothing to cache

    'absent' should not arise now that TrialSource.enumerate_trials lists what is on disk
    rather than crossing two constants — it existed because SUBJECTS x ACTIVITIES claimed
    trials the dataset does not have. It is kept for the case a listed trial's files vanish
    between enumeration and the build.

    Split out from `load_trial` so `build_trials.py --check` can audit the tree cheaply, and
    so a stale entry reports WHICH input moved rather than just rebuilding.
    """
    source = get_source(dataset)
    folder = source.source_dir(subject, trial)
    if not folder.is_dir() or not _source_inventory(folder, source.source_globs):
        return 'absent', f'no source trial at {folder.relative_to(paths.REPO_ROOT)}'

    path = paths.cached_trial_path(dataset, subject, trial)
    if not path.exists():
        return 'missing', None

    manifest = read_manifest(path)
    if manifest is None:
        return 'stale', 'no manifest sidecar'

    stored = manifest.get('cache_key')
    if stored is None:
        return 'stale', 'manifest predates cache_key'

    expected = trial_cache_key(subject, trial, align, dataset)
    for field, want in expected.items():
        if stored.get(field) != want:
            if field == 'sources':
                return 'stale', 'source files changed'
            return 'stale', f'{field}: cached {stored.get(field)!r} != current {want!r}'
    return 'fresh', None


class StaleTrialCache(RuntimeError):
    """A trial's parquet is missing, or was built from different inputs or code.

    Raised rather than silently falling back to parsing from source. That fallback made a
    stale cache cost time and nothing else, which sounds safe and is the problem: it also
    made it invisible, and it meant a run could mix cached and freshly-parsed trials with no
    record of which was which. The manifest's job is to say what code version produced the
    data a result rests on; a fallback means the result may not rest on it at all.
    """


def load_trial(subject: str, trial: str, align: bool = True,
               dataset: str = TRIAL_DATASET) -> Dict[str, PlateTrial]:
    """One trial's PlateTrials, read from its parquet.

    The parquet is the interface, not an optimisation: this never parses from source. Build
    it first with `python -m experiments.build_trials --dataset <name>`.

    Raises StaleTrialCache if the artifact is missing or no longer matches its inputs.
    """
    status, reason = cached_trial_status(subject, trial, align, dataset)
    if status != 'fresh':
        detail = f" ({reason})" if reason else ""
        raise StaleTrialCache(
            f"{dataset}/{subject}/{trial}: trial cache is {status}{detail}. "
            f"Run: python -m experiments.build_trials --dataset {dataset}")
    path = paths.cached_trial_path(dataset, subject, trial)
    return trial_io.plates_from_frame(pd.read_parquet(path, engine='pyarrow'))


def load_raw_data(subject: str, activity: str, align: bool = True) -> Dict[str, PlateTrial]:
    """Deprecated name for `load_trial`, kept so existing experiments keep working.

    Misleading now: it does not load raw data, it reads a built trial.
    """
    return load_trial(subject, activity, align=align)


# ==============================================================================
# Gravity convention check
# ==============================================================================

def measure_world_frame_gravity(plates: Dict[str, PlateTrial]) -> np.ndarray:
    """Mean accelerometer reading rotated into the mocap world frame, averaged over
    every sensor and sample in the trial.

    Over a full trial the subject starts and ends at rest and linear accelerations
    average out, so this converges on gravity as the accelerometers report it —
    which is exactly what EXPECTED_GRAVITY is supposed to be. Measured spread
    across this dataset is 0.035 m/s^2, so it is a sharp check, not a fuzzy one.
    """
    # Valid frames only. The rotation is what puts the reading in the world frame, so a
    # frame whose pose is padded or corrupt contributes a correctly-measured accelerometer
    # vector rotated by the wrong matrix — which is worse than no sample at all. Since
    # alignment stopped trimming, the padded stretches can outnumber the real ones.
    world_accs = []
    for plate in plates.values():
        valid = np.asarray(plate.valid)
        if not valid.any():
            continue
        rotated = np.einsum('nij,nj->ni', plate.world_trace.rotations[valid],
                            plate.imu_trace.acc[valid])
        world_accs.append(rotated.mean(axis=0))
    if not world_accs:
        raise ValueError("No plate has a valid frame; cannot measure world-frame gravity.")
    return np.mean(world_accs, axis=0)


def check_gravity_convention(plates: Dict[str, PlateTrial], tol: float = 0.5,
                             context: str = "") -> np.ndarray:
    """Verifies EXPECTED_GRAVITY against a loaded trial, raising on a mismatch.

    Not called by the pipeline — the convention is settled for this dataset and
    covered by test/TestGravityConvention.py. Kept for the tests and for the case
    where the pipeline is pointed at new data, where an axis or sign mismatch would
    otherwise pass silently and corrupt the EKF world reference and both acc oracles.
    """
    measured = measure_world_frame_gravity(plates)
    deviation = float(np.linalg.norm(measured - EXPECTED_GRAVITY))
    if deviation > tol:
        where = f" for {context}" if context else ""
        raise ValueError(
            f"Gravity convention mismatch{where}: EXPECTED_GRAVITY is "
            f"{EXPECTED_GRAVITY.tolist()} but the data's mean world-frame accelerometer "
            f"reads [{measured[0]:+.3f}, {measured[1]:+.3f}, {measured[2]:+.3f}] "
            f"(off by {deviation:.3f} m/s^2, tolerance {tol}).\n"
            f"Either this dataset uses a different world-frame axis or sign convention, "
            f"or EXPECTED_GRAVITY in experiments/experiment_utils.py is wrong. Set it to the measured "
            f"vector above if the dataset is correct."
        )
    return measured

# ==============================================================================
# Physics: oracle ("perfect" acc/mag) helpers
# ==============================================================================

def _compute_expected_mag_field(plate_trials: List[PlateTrial]) -> np.ndarray:
    """Median world-frame magnetic field across all torso-mounted IMU readings.

    Valid frames only, for the same reason as measure_world_frame_gravity: rotating a real
    magnetometer reading by a padded or corrupt pose puts it somewhere it never was.
    """
    all_global_mags = []
    for plate in plate_trials:
        if 'torso' not in plate.name:
            continue
        valid = np.asarray(plate.valid)
        if not valid.any():
            continue
        all_global_mags.append(
            (plate.world_trace.rotations[valid] @ plate.imu_trace.mag[valid][..., None])[..., 0])
    if not all_global_mags:
        raise ValueError("No torso plate has a valid frame; cannot estimate the field.")
    return np.median(np.concatenate(all_global_mags, axis=0), axis=0)


def _setup_ekf_ground_plate_(plate_trials: List[PlateTrial]) -> PlateTrial:
    """Precomputes global expected gravity and magnetic field to build a virtual parent ground plate."""
    base_plate = plate_trials[0]
    expected_mag = _compute_expected_mag_field(plate_trials)

    ground_plate = base_plate.copy()
    ground_plate.name = "ground"
    ground_plate.world_trace.rotations = np.tile(np.eye(3), (len(base_plate), 1, 1))
    ground_plate.imu_trace.gyro = np.zeros_like(base_plate.imu_trace.gyro)
    ground_plate.imu_trace.acc = np.tile(EXPECTED_GRAVITY, (len(base_plate), 1))
    ground_plate.imu_trace.mag = np.tile(expected_mag, (len(base_plate), 1))
    return ground_plate


def _compute_perfect_segment_acc(plate: PlateTrial, gravity: Optional[np.ndarray] = None) -> np.ndarray:
    """Oracle acc for a lone segment (EKF path): gravity rotated into the segment's
    ground-truth orientation, with no linear-acceleration component.

    gravity defaults to EXPECTED_GRAVITY, resolved at call time rather than bound as
    a default argument — a default would freeze the value at import and silently
    ignore any override, which makes the constant untestable."""
    gravity = EXPECTED_GRAVITY if gravity is None else gravity
    return np.einsum('nji,j->ni', plate.world_trace.rotations, gravity)


def _compute_perfect_mag(plate: PlateTrial, expected_mag: np.ndarray) -> np.ndarray:
    """Oracle mag: the expected (median) global field rotated into the plate's
    ground-truth orientation."""
    return np.einsum('nji,j->ni', plate.world_trace.rotations, expected_mag)


def _compute_scaled_mag(plate: PlateTrial, expected_mag: np.ndarray,
                        distortion_scale: float) -> np.ndarray:
    """The plate's magnetometer with its estimated distortion multiplied by
    `distortion_scale` — a dial between the mag oracle and the real reading.

    THE DEFINITION. Rotate the reading into the world frame with ground truth, where the
    undistorted field would be the constant `expected_mag` (e). Whatever is left over,
    d(t) = m_world(t) - e, is this sensor's estimated distortion at that instant. Scale
    only that, and rotate back:

        m_body(t; a) = R(t)^T [ e + a * ( R(t) m_body(t) - e ) ]

    The two ends are exact, not approximate, and that is the point of writing it this way:
      a = 0  reproduces _compute_perfect_mag bit for bit (the d term vanishes);
      a = 1  reproduces the raw reading bit for bit (R^T R = I, verified to 1e-15 on real
             trials — the world round trip is a pure rotation).
    So a sweep over a interpolates continuously between the 'perfect_mag' oracle arm and
    the ordinary real-magnetometer arm, and both endpoints are checkable against arms the
    pipeline already runs rather than being a separate code path. a > 1 extrapolates:
    the same distortion pattern, amplified, which is how the sweep finds a breaking point
    that a >= 1 alone would not reach if the real field is already tolerable.

    WHAT IS ACTUALLY BEING SCALED. Everything that makes this sensor's reading differ from
    one common rigid field — the lab's ferrous distortion (which is what dominates: it
    scales with sensor height and is localized in lab coordinates, see
    experiments/sensor_distributions.py), plus that sensor's own calibration gain error and
    noise, plus any error in e itself. It is an upper bound on the field anomaly rather
    than an isolate of it, so read the sweep's x-axis as "total inconsistency between this
    sensor and the assumed field", not as "milligauss of ferrous distortion". The
    experiment reports that x-axis in degrees of field disagreement for exactly this
    reason (experiments/distortion_tolerance.py).

    NOT A ROTATION. Scaling shortens the vector as well as turning it, so |m| moves with a
    — at a = 0 every sensor reads |e| exactly. That matters because the relative filter is
    fed raw-magnitude measurements (normalize_measurements=False), so its magnetometer
    weighting drifts slightly across the sweep along with the magnitude. The alternative,
    renormalizing to the original |m|, would hold the weighting fixed but would no longer
    reduce to either endpoint, and would keep a magnitude error the filter treats as
    signal. The endpoints are worth more than the constant weighting, so this is a plain
    linear interpolation.
    """
    world_rots = plate.world_trace.rotations
    world_mag = np.einsum('nij,nj->ni', world_rots, plate.imu_trace.mag)
    scaled_world = expected_mag + distortion_scale * (world_mag - expected_mag)
    return np.einsum('nji,nj->ni', world_rots, scaled_world)


def _mag_override(plate: PlateTrial, mag_source: str, expected_mag: Optional[np.ndarray],
                  mag_distortion_scale: Optional[float]) -> Optional[np.ndarray]:
    """The magnetometer reading to hand the filter in place of the plate's own, or None to
    leave it alone.

    Shared by the relative-filter and EKF paths so the three mag sources are resolved in
    exactly one place: both paths take the same '_perfect_mag' / '_dist<a>' method suffixes,
    and a source honoured on one path but silently ignored on the other would run the
    requested configuration under one name and the default under another."""
    if mag_source == 'real':
        return None
    if mag_source == 'perfect':
        return _compute_perfect_mag(plate, expected_mag)
    if mag_source == 'scaled':
        if mag_distortion_scale is None:
            raise ValueError(
                "mag_source='scaled' needs a mag_distortion_scale; got None. The scale is "
                "the multiplier on the estimated distortion (0 = the mag oracle, 1 = the "
                "real reading), so there is no sensible default to fall back on."
            )
        return _compute_scaled_mag(plate, expected_mag, mag_distortion_scale)
    raise ValueError(f"Unknown mag_source '{mag_source}'")


def _needs_expected_mag(mag_source: str) -> bool:
    """Whether a mag source is measured against the world field reference. Both the oracle
    and the distortion sweep are; the real reading is not."""
    return mag_source in ('perfect', 'scaled')


def _compute_perfect_joint_acc(parent_trial: PlateTrial, child_trial: PlateTrial,
                                gravity: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Oracle acc for a joint pair (relative-filter path): the true linear
    acceleration of the shared joint center (averaged from both segments' offset
    estimates) plus gravity, rotated into each segment's own ground-truth
    orientation. Unlike the EKF oracle acc, this includes the linear-acceleration
    term, since the joint center itself translates through space.

    gravity is resolved at call time (see _compute_perfect_segment_acc)."""
    gravity = EXPECTED_GRAVITY if gravity is None else gravity
    parent_offset, child_offset, _ = parent_trial.world_trace.get_joint_center(child_trial.world_trace)

    joint_pos_parent = parent_trial.world_trace.positions + np.einsum(
        'nij,j->ni', parent_trial.world_trace.rotations, parent_offset)
    joint_pos_child = child_trial.world_trace.positions + np.einsum(
        'nij,j->ni', child_trial.world_trace.rotations, child_offset)
    joint_pos = 0.5 * (joint_pos_parent + joint_pos_child)

    # Rotations are unused by finite_difference_world_frame_accelerations; reuse
    # the parent's purely as a placeholder to satisfy the WorldTrace constructor.
    joint_trace = WorldTrace(parent_trial.world_trace.timestamps, joint_pos, parent_trial.world_trace.rotations)
    global_acc = joint_trace.finite_difference_world_frame_accelerations(acc_from_gravity=gravity)

    acc_parent = np.einsum('nji,nj->ni', parent_trial.world_trace.rotations, global_acc)
    acc_child = np.einsum('nji,nj->ni', child_trial.world_trace.rotations, global_acc)
    return acc_parent, acc_child

# ==============================================================================
# Physics: relative filter
# ==============================================================================

def segment_observability(imu_trace: IMUTrace) -> np.ndarray:
    """o = |a x (d/dt a_world)|, one sensor, one value per sample, in (m/s^2)(m/s^3).

    What makes a segment's orientation observable from its accelerometer is the
    accelerometer vector CHANGING DIRECTION IN THE WORLD FRAME. Gravity alone does not:
    a sensor rotating steadily under gravity sees its acc vector sweep around the body
    frame, but that sweep is fully explained by the gyro, so it carries no independent
    orientation information. The quantity that does carry information is the world-frame
    derivative of the accelerometer vector, expressed back in the body frame:

        d/dt(a_world) |_body = a_dot + w x a

    which vanishes exactly when the only acceleration is gravity. Crossing it with a
    itself drops the component along a (a change in magnitude tells us nothing about
    direction) and leaves the part that actually rotates the measured direction.

    THE dt MATTERS. a_dot is a per-second rate, so the finite difference has to be
    divided by the sample interval. Without that division the difference term is ~100x
    too small at this dataset's 100 Hz, the `w x a` term dominates, and the metric
    silently degenerates into |a|^2|w_perp| — an angular-rate detector that scores pure
    rotation under gravity (the maximally UNOBSERVABLE case) at ~190, and whose value
    drifts with sample rate. That was the behaviour through the runs preceding this fix;
    see test/TestExperimentPhysics.py, which now pins the corrected physics.

    The difference is BACKWARD, not central, deliberately: this gates the magnetometer
    inside a causal filter, so the value at sample t must not depend on sample t+1. The
    cost is noise amplification (1/dt on a difference of two noisy samples), which was
    measured rather than assumed — at this dataset's accelerometer noise the static
    noise floor is ~15-35, against a trial median of ~300 and a gating threshold of
    1000, so no smoothing is needed. Central differencing would halve the floor and move
    the median by <5%, which does not buy back the loss of causality.

    Sample 0 is padded with 0.0: there is no difference available there, so the first
    sample always counts as unobservable.
    """
    acc, gyro = imu_trace.acc, imu_trace.gyro
    dt = np.diff(imu_trace.timestamps)[:, None]
    acc_dot_world = np.diff(acc, axis=0) / dt + np.cross(gyro[1:], acc[1:])
    return np.concatenate(([0.0], np.linalg.norm(np.cross(acc[1:], acc_dot_world), axis=1)))


def project_pair_to_joint_center(parent_trial: PlateTrial, child_trial: PlateTrial
                                 ) -> Tuple[PlateTrial, PlateTrial]:
    """Both plates with their IMU traces rigid-body projected to the shared joint center —
    the same step _run_relative_filter(project=True) performs before the EKF sees the data.

    Factored out because o^J is only comparable to the value the filter gates on if it is
    computed after the identical projection, and the analysis scripts that want o^J without
    running the EKF (experiments/drift_observability.py,
    experiments/sensor_distributions.py) would otherwise each carry their own copy of these
    four lines. Returns copies: projection replaces imu_trace, so operating in place would
    silently change every later use of the caller's plates.
    """
    parent_trial = parent_trial.copy()
    child_trial = child_trial.copy()
    parent_offset, child_offset, _ = parent_trial.world_trace.get_joint_center(child_trial.world_trace)
    parent_trial.imu_trace = parent_trial.project_imu_trace(parent_offset)
    child_trial.imu_trace = child_trial.project_imu_trace(child_offset)
    return parent_trial, child_trial


def _calculate_observability_metric_(parent_trial: PlateTrial, child_trial: PlateTrial) -> np.ndarray:
    """o^J for a joint: the worse-conditioned of its two segments.

    A joint's relative orientation is only as observable as the less informative of the
    two accelerometers, so this is a minimum and not a sum or a mean."""
    return np.minimum(segment_observability(parent_trial.imu_trace),
                      segment_observability(child_trial.imu_trace))


def _run_relative_filter(parent_trial: PlateTrial,
                         child_trial: PlateTrial,
                         project: bool,
                         mag_mode: str,
                         acc_override_parent: Optional[np.ndarray] = None,
                         acc_override_child: Optional[np.ndarray] = None,
                         mag_override_parent: Optional[np.ndarray] = None,
                         mag_override_child: Optional[np.ndarray] = None,
                         gyro_std_parent: float = DEFAULT_GYRO_STD,
                         acc_std_parent: float = DEFAULT_ACC_STD,
                         mag_std_parent: float = DEFAULT_MAG_STD,
                         gyro_std_child: float = DEFAULT_GYRO_STD,
                         acc_std_child: float = DEFAULT_ACC_STD,
                         mag_std_child: float = DEFAULT_MAG_STD,
                         mag_adapt_threshold: float = DEFAULT_MAG_ADAPT_THRESHOLD,
                         normalize_measurements: bool = False,
                         rescale_stds: bool = False,
                         init_orientation_std: Optional[float] = None,
                         return_observability: bool = False):
    """Estimates joint orientations between parent and child trials using specified filter configurations.

    acc/mag_override_* replace the trial's real IMU reading before anything else
    runs (used for oracle acc/mag ablations). Providing an acc override implies the
    override is already the correctly-projected acceleration at the point of
    interest, so `project` is forced off in that case — otherwise the rigid-body
    projection would be double-applied.

    normalize_measurements scales the acc/mag vectors to unit length inside the filter's
    measurement update while leaving the *_std arguments untouched, which changes how far
    the filter trusts them relative to the gyro prediction — read
    RelativeFilter._normalize_vector_measurements before comparing across this flag.

    rescale_stds undoes exactly that side effect, dividing each vector sensor's std by its
    NOMINAL_*_MAGNITUDE so a unit-length measurement carries the same weight the raw one
    did. Set together (the '_rescaled' method arm), the two isolate the geometric effect of
    normalizing from the retuning it otherwise smuggles in. It is meaningless on its own,
    so it raises rather than quietly running a filter nobody asked for.

    If return_observability, also returns the o^J observability metric (see
    _calculate_observability_metric_), computed post-projection regardless of
    mag_mode — used by experiments/drift_observability.py to relate drift to
    observability independent of whether the magnetometer was used."""
    if rescale_stds and not normalize_measurements:
        raise ValueError(
            "rescale_stds=True with normalize_measurements=False divides the sensor stds "
            "by their nominal magnitudes while the measurements keep their raw scale, "
            "which is not a configuration anything wants — it just over-trusts every "
            "sensor by that factor. Use both together (the '_rescaled' method arm) or "
            "neither."
        )
    if rescale_stds:
        acc_std_parent /= NOMINAL_ACC_MAGNITUDE
        acc_std_child /= NOMINAL_ACC_MAGNITUDE
        mag_std_parent /= NOMINAL_MAG_MAGNITUDE
        mag_std_child /= NOMINAL_MAG_MAGNITUDE

    parent_trial = parent_trial.copy()
    child_trial = child_trial.copy()

    if acc_override_parent is not None:
        parent_trial.imu_trace.acc = acc_override_parent
    if acc_override_child is not None:
        child_trial.imu_trace.acc = acc_override_child
    if mag_override_parent is not None:
        parent_trial.imu_trace.mag = mag_override_parent
    if mag_override_child is not None:
        child_trial.imu_trace.mag = mag_override_child

    do_project = project and acc_override_parent is None and acc_override_child is None

    # 1. IMU projection
    parent_offset, child_offset = None, None
    if do_project:
        parent_offset, child_offset, error = parent_trial.world_trace.get_joint_center(child_trial.world_trace)
        if np.mean(np.linalg.norm(error, axis=1)) > 0.05:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Warning: High joint center error ({np.mean(np.linalg.norm(error, axis=1))} m) "
                      f"between {parent_trial.name} and {child_trial.name}. Check marker placement.")
        parent_trial.imu_trace = parent_trial.project_imu_trace(parent_offset)
        child_trial.imu_trace = child_trial.project_imu_trace(child_offset)

    # 2. Magnetometer modifications
    if mag_mode == 'adapt':
        obs = _calculate_observability_metric_(parent_trial, child_trial)
        high_idx = obs > mag_adapt_threshold
        parent_trial.imu_trace.mag[high_idx] = 0.0
        child_trial.imu_trace.mag[high_idx] = 0.0
    elif mag_mode == 'off':
        parent_trial.imu_trace.mag = np.zeros_like(parent_trial.imu_trace.mag)
        child_trial.imu_trace.mag = np.zeros_like(child_trial.imu_trace.mag)
    elif mag_mode != 'on':
        raise ValueError(f"Unknown mag_mode '{mag_mode}' specified.")

    # 3. Filter execution
    #
    # R_pc[t] must be the estimate *at* timestamps[t]. The update propagates the state
    # forward by dt before correcting it, so the sample driving the step into time t is
    # gyro[t-1]: a filter-free check (marker-derived angular velocity over (t, t+1]
    # against gyro[t+shift]) minimises at shift=0, i.e. gyro[t] spans the interval
    # starting at t. The measurement correction uses acc/mag[t], which are valid at t.
    # Index 0 is the seeded state itself, so the error there is exactly zero.
    #
    # This runs on the compiled kernel in src/relative_filter_fast.py, which is ~55x
    # faster than looping RelativeFilter.update() from Python and is held to it by
    # test/TestRelativeFilterFast.py. RelativeFilter remains the reference, and is
    # still what runs if numba is unavailable.
    gyro_std_p = np.ones(3) * gyro_std_parent
    gyro_std_c = np.ones(3) * gyro_std_child
    sensor_stds_p = [np.ones(3) * acc_std_parent, np.ones(3) * mag_std_parent]
    sensor_stds_c = [np.ones(3) * acc_std_child, np.ones(3) * mag_std_child]
    init_std_kwargs = ({} if init_orientation_std is None
                       else {'init_orientation_std': init_orientation_std})

    R_wp0 = parent_trial.world_trace.rotations[0]
    R_wc0 = child_trial.world_trace.rotations[0]
    dt = np.mean(parent_trial.imu_trace.timestamps[1:] - parent_trial.imu_trace.timestamps[:-1])
    N = len(parent_trial)

    if relative_filter_fast.NUMBA_AVAILABLE:
        R_pc = relative_filter_fast.run_relative_filter(
            parent_trial.imu_trace.gyro, child_trial.imu_trace.gyro,
            np.stack([parent_trial.imu_trace.acc, parent_trial.imu_trace.mag], axis=1),
            np.stack([child_trial.imu_trace.acc, child_trial.imu_trace.mag], axis=1),
            dt,
            gyro_std_parent=gyro_std_p, gyro_std_child=gyro_std_c,
            vector_sensor_stds_parent=sensor_stds_p,
            vector_sensor_stds_child=sensor_stds_c,
            R_wp0=R_wp0, R_wc0=R_wc0,
            normalize_measurements=normalize_measurements,
            **init_std_kwargs)
    else:
        joint_filter = RelativeFilter(
            gyro_std_parent=gyro_std_p, gyro_std_child=gyro_std_c,
            vector_sensor_stds_parent=sensor_stds_p,
            vector_sensor_stds_child=sensor_stds_c,
            normalize_measurements=normalize_measurements,
            **init_std_kwargs
        )
        joint_filter.set_qs(Rotation.from_matrix(R_wp0), Rotation.from_matrix(R_wc0))
        R_pc = np.empty((N, 3, 3), dtype=np.float64)
        R_pc[0] = joint_filter.get_R_pc()
        for t in range(1, N):
            joint_filter.update(
                parent_trial.imu_trace.gyro[t - 1], child_trial.imu_trace.gyro[t - 1],
                [parent_trial.imu_trace.acc[t], parent_trial.imu_trace.mag[t]],
                [child_trial.imu_trace.acc[t], child_trial.imu_trace.mag[t]], dt
            )
            R_pc[t] = joint_filter.get_R_pc()

    if return_observability:
        return R_pc, _calculate_observability_metric_(parent_trial, child_trial)
    return R_pc

# ==============================================================================
# Joint angles per method kind
# ==============================================================================

def _joint_valid(parent_plate: PlateTrial, child_plate: PlateTrial) -> np.ndarray:
    """Per-frame validity of a JOINT: both segments have to be trustworthy.

    A joint angle is a relative rotation between two plates, so it inherits the worse of
    the two masks. One corrupt plate takes the frame out of every joint it participates in
    — Subject06's femur_l invalidates both L_Hip and L_Knee at those frames, not one.

    This is the column that carries plate-level validity into the error statistics, which
    is what makes it safe to stop trimming traces at load: an unscoreable frame becomes one
    that is present and excluded, rather than one that was silently deleted.
    """
    return np.asarray(parent_plate.valid) & np.asarray(child_plate.valid)


def _joint_angles_from_marker(plates: Dict[str, PlateTrial]) -> pd.DataFrame:
    all_joint_data = []
    any_plate = next(iter(plates.values()))
    timestamps = any_plate.imu_trace.timestamps

    for joint_name, (parent, child) in JOINTS.items():
        if parent not in plates or child not in plates:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Skipping joint {joint_name}: parent '{parent}' or child '{child}' not in loaded plates.")
            continue
        parent_plate = plates[parent]
        child_plate = plates[child]
        R_joint = np.einsum('tji,tjk->tik', parent_plate.world_trace.rotations, child_plate.world_trace.rotations)
        rotvec = Rotation.from_matrix(R_joint).as_rotvec()

        df = pd.DataFrame({
            'timestamp': timestamps,
            'joint_name': joint_name,
            'rx': rotvec[:, 0],
            'ry': rotvec[:, 1],
            'rz': rotvec[:, 2],
            'valid': _joint_valid(parent_plate, child_plate),
        })
        all_joint_data.append(df)

    return pd.concat(all_joint_data, ignore_index=True) if all_joint_data else pd.DataFrame()


def _joint_angles_from_filter(plates: Dict[str, PlateTrial], project: bool, mag_mode: str,
                               acc_source: str = 'real', mag_source: str = 'real',
                               mag_adapt_threshold: float = DEFAULT_MAG_ADAPT_THRESHOLD,
                               normalize_measurements: bool = False,
                               rescale_stds: bool = False,
                               mag_distortion_scale: Optional[float] = None,
                               stds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    all_joint_data = []
    any_plate = next(iter(plates.values()))
    timestamps = any_plate.imu_trace.timestamps
    expected_mag = (_compute_expected_mag_field(list(plates.values()))
                    if _needs_expected_mag(mag_source) else None)
    tuning = resolve_stds(stds)

    for joint_name, (parent, child) in JOINTS.items():
        if parent not in plates or child not in plates:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Skipping joint {joint_name}: parent '{parent}' or child '{child}' not in loaded plates.")
            continue
        parent_plate = plates[parent]
        child_plate = plates[child]

        acc_override_parent = acc_override_child = None
        if acc_source == 'perfect':
            acc_override_parent, acc_override_child = _compute_perfect_joint_acc(parent_plate, child_plate)
        elif acc_source != 'real':
            raise ValueError(f"Unknown acc_source '{acc_source}'")

        mag_override_parent = _mag_override(parent_plate, mag_source, expected_mag, mag_distortion_scale)
        mag_override_child = _mag_override(child_plate, mag_source, expected_mag, mag_distortion_scale)

        R_pc = _run_relative_filter(
            parent_plate, child_plate, project=project, mag_mode=mag_mode,
            acc_override_parent=acc_override_parent, acc_override_child=acc_override_child,
            mag_override_parent=mag_override_parent, mag_override_child=mag_override_child,
            mag_adapt_threshold=mag_adapt_threshold,
            normalize_measurements=normalize_measurements,
            rescale_stds=rescale_stds,
            gyro_std_parent=tuning['gyro_std'], acc_std_parent=tuning['acc_std'],
            mag_std_parent=tuning['mag_std'],
            gyro_std_child=tuning['gyro_std'], acc_std_child=tuning['acc_std'],
            mag_std_child=tuning['mag_std'],
        )
        rotvec = Rotation.from_matrix(R_pc).as_rotvec()

        df = pd.DataFrame({
            'timestamp': timestamps,
            'joint_name': joint_name,
            'rx': rotvec[:, 0],
            'ry': rotvec[:, 1],
            'rz': rotvec[:, 2],
            # The filter's own output is defined everywhere it ran; `valid` describes the
            # GROUND TRUTH it will be scored against, which is why it is the same column
            # for every method on a given trial.
            'valid': _joint_valid(parent_plate, child_plate),
        })
        all_joint_data.append(df)

    return pd.concat(all_joint_data, ignore_index=True) if all_joint_data else pd.DataFrame()


def _joint_angles_from_ekf(plates: Dict[str, PlateTrial], acc_source: str = 'real', mag_source: str = 'real',
                            normalize_measurements: bool = False,
                            rescale_stds: bool = False,
                            mag_distortion_scale: Optional[float] = None,
                            stds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    plate_trials = list(plates.values())
    ground_plate = _setup_ekf_ground_plate_(plate_trials)
    expected_mag = _compute_expected_mag_field(plate_trials) if _needs_expected_mag(mag_source) else None
    tuning = resolve_stds(stds)

    segment_orientations = {}
    for plate_name, plate in plates.items():
        acc_override = _compute_perfect_segment_acc(plate) if acc_source == 'perfect' else None
        mag_override = _mag_override(plate, mag_source, expected_mag, mag_distortion_scale)
        # The virtual ground plate is noiseless by construction, but its stds are set to the
        # same values as the real sensor's: the filter estimates a RELATIVE orientation, so
        # zeroing the parent's stds would make its perfect readings infinitely trusted and the
        # state unidentifiable. Symmetric stds are what the unswept pipeline has always run.
        segment_orientations[plate_name] = _run_relative_filter(
            ground_plate, plate, project=False, mag_mode='on',
            acc_override_child=acc_override, mag_override_child=mag_override,
            normalize_measurements=normalize_measurements,
            rescale_stds=rescale_stds,
            gyro_std_parent=tuning['gyro_std'], acc_std_parent=tuning['acc_std'],
            mag_std_parent=tuning['mag_std'],
            gyro_std_child=tuning['gyro_std'], acc_std_child=tuning['acc_std'],
            mag_std_child=tuning['mag_std'],
        )

    all_joint_data = []
    timestamps = plate_trials[0].imu_trace.timestamps

    for joint_name, (parent, child) in JOINTS.items():
        if parent not in segment_orientations or child not in segment_orientations:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Skipping joint {joint_name}: parent '{parent}' or child '{child}' not in segment orientations.")
            continue
        R_parent = np.array(segment_orientations[parent])
        R_child = np.array(segment_orientations[child])
        R_joint = np.einsum('tji,tjk->tik', R_parent, R_child)
        rotvec = Rotation.from_matrix(R_joint).as_rotvec()

        df = pd.DataFrame({
            'timestamp': timestamps,
            'joint_name': joint_name,
            'rx': rotvec[:, 0],
            'ry': rotvec[:, 1],
            'rz': rotvec[:, 2],
        })
        all_joint_data.append(df)

    return pd.concat(all_joint_data, ignore_index=True) if all_joint_data else pd.DataFrame()


def run_window(plates: Dict[str, PlateTrial]) -> slice:
    """The span a filter should actually be run over: t = 0 to the last scoreable frame.

    Since alignment stopped trimming, a trial carries the whole inertial record — 430 s of
    it before t = 0 on Subject01's walking trial. None of that is scoreable, and a filter
    run through it would pay 1.7x the runtime to produce output nobody can evaluate.

    The start is t = 0, which is the first overlap sample and therefore the exact instant
    the old code re-zeroed to; filters see an identical first sample, so the burn-in
    transient that dominates pooled error is unchanged.

    The end is the last frame valid on ANY plate, taken trial-wide rather than per plate so
    that one corrupt segment cannot shorten the run for the others. For IMoVE's long-walk
    trials this spans the first mocap take to the last, gaps included — which is right, since
    the filter has to run continuously through them to carry its state across.
    """
    any_plate = next(iter(plates.values()))
    timestamps = any_plate.imu_trace.timestamps
    start = int(np.searchsorted(timestamps, 0.0))

    scoreable = np.logical_or.reduce([np.asarray(p.valid) for p in plates.values()])
    if not scoreable.any():
        return slice(start, len(timestamps))
    return slice(start, int(np.flatnonzero(scoreable)[-1]) + 1)


def compute_joint_angles(plates: Dict[str, PlateTrial], method: str,
                          stds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Joint angles for one method. `stds` overrides the DEFAULT_*_STD tuning per
    sensor (see resolve_stds); it is inert for 'marker', which runs no filter.

    Every method is evaluated over the same `run_window`, so their outputs share a timestamp
    axis and `compute_error_stats` can merge them frame for frame.
    """
    window = run_window(plates)
    plates = {name: plate[window] for name, plate in plates.items()}
    spec = resolve_method_spec(method)
    if spec['kind'] == 'marker':
        return _joint_angles_from_marker(plates)
    if spec['kind'] == 'ekf':
        return _joint_angles_from_ekf(plates, acc_source=spec['acc_source'], mag_source=spec['mag_source'],
                                      normalize_measurements=spec['normalize_measurements'],
                                      rescale_stds=spec['rescale_stds'],
                                      mag_distortion_scale=spec.get('mag_distortion_scale'),
                                      stds=stds)
    return _joint_angles_from_filter(
        plates,
        project=spec['project'],
        mag_mode=spec['mag_mode'],
        acc_source=spec['acc_source'],
        mag_source=spec['mag_source'],
        mag_adapt_threshold=spec.get('mag_adapt_threshold', DEFAULT_MAG_ADAPT_THRESHOLD),
        normalize_measurements=spec['normalize_measurements'],
        rescale_stds=spec['rescale_stds'],
        mag_distortion_scale=spec.get('mag_distortion_scale'),
        stds=stds
    )

# ==============================================================================
# Joint-angle intermediate save/load
# ==============================================================================

joint_angles_path = paths.joint_angles_path


def save_joint_angles(df: pd.DataFrame, subject: str, activity: str, method: str,
                      variant: Optional[str] = None, stds: Optional[Dict[str, float]] = None):
    """`variant` and `stds` travel together: the first namespaces the output so a
    re-tuned run does not overwrite the default-tuned one, the second is what the
    manifest records as the tuning in force (see paths.joint_angles_path)."""
    path = ensure_parent(joint_angles_path(subject, activity, method, variant=variant))
    df.to_parquet(path, engine='pyarrow')
    spec = resolve_method_spec(method)
    write_manifest(
        path,
        constants=pipeline_constants(stds),
        subject=f"Subject{subject}", activity=activity, method=method, method_spec=spec,
        variant=variant,
        source=str(raw_trial_dir(subject, activity).relative_to(paths.REPO_ROOT)),
        n_rows=len(df),
    )


def load_joint_angles(subject: str, activity: str, method: str,
                      variant: Optional[str] = None) -> Optional[pd.DataFrame]:
    path = joint_angles_path(subject, activity, method, variant=variant)
    return pd.read_parquet(path, engine='pyarrow') if path.exists() else None


LABEL_COLUMNS = ('joint_name', 'subject', 'trial_type', 'method')


def as_categorical_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Store the label columns as categoricals rather than Python string objects.

    These four columns hold a handful of distinct values each, but as object dtype every
    cell is a separate str: measured on the pooled table (40.8M rows across 11 subjects
    x 2 activities x 5 methods) they were 9.2 of 10.5 GB, and categorical brings the whole
    frame to 1.5 GB. That is the difference between the pooled summary fitting in memory
    and not, and it compounds with the method count -- the noise sweep has 176 of them.

    It also saves time downstream: merge and groupby both factorise these columns, and
    categorical arrives pre-factorised.

    Anything grouping or pivoting on these columns afterwards must pass observed=True, or
    pandas expands to the full cartesian product of categories and invents all-NaN rows
    for combinations that were never run.
    """
    for col in LABEL_COLUMNS:
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype('category')
    return df


def load_all_joint_angles(subjects: List[str], activities: List[str], methods: List[str],
                          variant: Optional[str] = None) -> pd.DataFrame:
    frames = []
    for subject in subjects:
        for activity in activities:
            for method in methods:
                df = load_joint_angles(subject, activity, method, variant=variant)
                if df is None:
                    if os.environ.get("DISABLE_TQDM") != "True":
                        print(f"Warning: missing joint angles for Subject{subject}/{activity}/{method} — skipping")
                    continue
                frames.append(df.assign(subject=f"Subject{subject}", trial_type=activity, method=method))
    if not frames:
        return pd.DataFrame()
    return as_categorical_labels(pd.concat(frames, ignore_index=True))

# ==============================================================================
# Statistics
# ==============================================================================

def compute_error_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates summary statistics using fast wide-format C-aggregations."""
    if df.empty:
        return pd.DataFrame()

    marker_df = df[df['method'] == 'marker']
    imu_df = df[df['method'] != 'marker']

    if marker_df.empty or imu_df.empty:
        return pd.DataFrame()

    join_cols = ['subject', 'trial_type', 'joint_name', 'timestamp']
    merged_df = pd.merge(imu_df, marker_df, on=join_cols, suffixes=('_imu', '_marker'))

    if merged_df.empty:
        return pd.DataFrame()

    # Drop frames whose GROUND TRUTH is not trustworthy — interpolated marker poses,
    # unresolved reconstruction failures, and (once alignment stops trimming) the stretches
    # of IMU record the mocap never covered. Scoring against an invented pose measures the
    # invention, not the filter.
    #
    # Filtered on the marker side specifically: `valid` describes the reference, and the
    # IMU-side copy is the same values for the same trial. Absent on artifacts written
    # before the column existed, which are treated as fully valid — that was the effective
    # behaviour when they were produced, so it reproduces them rather than silently
    # reinterpreting them.
    if 'valid_marker' in merged_df.columns:
        merged_df = merged_df[merged_df['valid_marker'].to_numpy(dtype=bool)]
    elif 'valid' in merged_df.columns:
        merged_df = merged_df[merged_df['valid'].to_numpy(dtype=bool)]
    if merged_df.empty:
        return pd.DataFrame()

    rotvec_imu = merged_df[['rx_imu', 'ry_imu', 'rz_imu']].to_numpy()
    rotvec_marker = merged_df[['rx_marker', 'ry_marker', 'rz_marker']].to_numpy()

    # rotvec of (R_imu @ R_marker^-1). relative_rotvec is the scipy expression
    #     (Rotation.from_rotvec(imu) * Rotation.from_rotvec(marker).inv()).as_rotvec()
    # done in plain numpy: identical to ~1e-15 rad but ~9x faster, and this runs on every
    # sample of every method, which made it 45% of this function.
    rotvec_error = relative_rotvec(rotvec_imu, rotvec_marker)

    merged_df['X'] = rotvec_error[:, 0]
    merged_df['Y'] = rotvec_error[:, 1]
    merged_df['Z'] = rotvec_error[:, 2]
    merged_df['MAG'] = np.linalg.norm(rotvec_error, axis=1)

    merged_df = merged_df.rename(columns={'method_imu': 'method'})
    group_cols = ['trial_type', 'method', 'joint_name', 'subject']
    target_cols = ['MAG', 'X', 'Y', 'Z']

    # Pre-calculate squared & absolute columns for vectorized MAE/RMSE
    for col in target_cols:
        merged_df[f'{col}_sq'] = merged_df[col] ** 2
        merged_df[f'{col}_abs'] = merged_df[col].abs()

    # Fast built-in aggregations across wide format.
    #
    # observed=True is required, not cosmetic: load_all_joint_angles hands the label
    # columns over as categoricals (88% of this table's memory otherwise), and a
    # categorical groupby without it expands to the full cartesian product of categories,
    # inventing all-NaN rows for subject/method/joint combinations that were never run.
    grouped = merged_df.groupby(group_cols, observed=True)

    means = grouped[target_cols].mean()
    stds = grouped[target_cols].std()
    mins = grouped[target_cols].min()
    maxs = grouped[target_cols].max()
    # One pass for all three order statistics rather than three: each quantile() call
    # sorts every group independently, so asking together is ~2.5x cheaper. quantile(0.5)
    # and median() agree exactly, including for even-sized groups.
    quantiles = grouped[target_cols].quantile([0.25, 0.5, 0.75])
    q25s = quantiles.xs(0.25, level=-1)
    medians = quantiles.xs(0.50, level=-1)
    q75s = quantiles.xs(0.75, level=-1)
    maes = grouped[[f'{c}_abs' for c in target_cols]].mean().rename(columns=lambda c: c.replace('_abs', ''))
    rmses = np.sqrt(grouped[[f'{c}_sq' for c in target_cols]].mean()).rename(columns=lambda c: c.replace('_sq', ''))

    # MAD: median absolute deviation from median
    mads = grouped[target_cols].apply(lambda g: (g - g.median()).abs().median())

    # Build multi-index summary and melt at the very end
    summary_list = []
    metric_map = {
        'mean_rad': means, 'std_rad': stds, 'rmse_rad': rmses,
        'mae_rad': maes, 'mad_rad': mads, 'min_rad': mins,
        'q25_rad': q25s, 'median_rad': medians, 'q75_rad': q75s, 'max_rad': maxs
    }

    for metric_name, metric_df in metric_map.items():
        melted = metric_df.reset_index().melt(
            id_vars=group_cols, value_vars=target_cols, var_name='axis', value_name=metric_name
        )
        summary_list.append(melted.set_index(group_cols + ['axis']))

    summary_df = pd.concat(summary_list, axis=1).reset_index()
    # Hand the label columns back as plain strings whatever came in. The input may arrive
    # with them as categoricals (load_all_joint_angles does that for the memory), and
    # leaking that dtype into the statistics tables would change how every downstream
    # consumer sorts, uniques and groups them for no benefit — the output is a few hundred
    # rows, so there is nothing to save here.
    for col in group_cols:
        if isinstance(summary_df[col].dtype, pd.CategoricalDtype):
            summary_df[col] = summary_df[col].astype(str)
    return summary_df


def save_statistics(df: pd.DataFrame, name: str, stds: Optional[Dict[str, float]] = None,
                    **manifest_extra: Any) -> Path:
    """Saves a summary-statistics DataFrame to results/statistics/<name>_statistics.parquet.

    `stds` records a filter re-tuning in the manifest (see pipeline_constants)."""
    path = ensure_parent(paths.statistics_path(name))
    df.to_parquet(path, engine='pyarrow')
    write_manifest(
        path, constants=pipeline_constants(stds), experiment=name,
        methods=sorted(df['method'].unique().tolist()) if 'method' in df.columns else None,
        subjects=sorted(df['subject'].unique().tolist()) if 'subject' in df.columns else None,
        n_rows=len(df), **manifest_extra,
    )
    return path


def load_statistics(name: str) -> Optional[pd.DataFrame]:
    path = paths.statistics_path(name)
    return pd.read_parquet(path, engine='pyarrow') if path.exists() else None

# ==============================================================================
# Parallel runner with live status table
# ==============================================================================
# Every experiment shares this shape: a set of "rows" (e.g. subject/activity pairs,
# or joints) each processed in its own worker process, with a fixed set of "stage"
# columns (e.g. methods, noise combos, thresholds) whose status is tracked live.
# worker_fn runs ALL stages for one row in a single process (so it can load raw data
# once and reuse it across stages) and returns whatever payload the caller wants
# collected (typically a DataFrame, or None).

_STATUS_COLORS = {"Pending": "white", "Running": "yellow", "Success": "green", "Skipped": "yellow"}


def _render_grid_table(title: str, row_keys: List[Any], row_labels: List[str],
                        stage_labels: List[str], shared_state: Dict) -> Table:
    table = Table(title=f"[bold magenta]{title}[/bold magenta]", show_header=True,
                  header_style="bold cyan", border_style="bold blue")
    for label in row_labels:
        table.add_column(label, style="bold white", justify="center")
    for stage in stage_labels:
        table.add_column(stage, justify="center")

    for row_key in row_keys:
        row = list(row_key) if isinstance(row_key, tuple) else [row_key]
        for stage in stage_labels:
            status = shared_state.get((row_key, stage), "Pending")
            t = shared_state.get((row_key, f"{stage}_time"), None)
            suffix = f" [dim]({t:.1f}s)[/dim]" if t is not None else ""
            color = _STATUS_COLORS.get(status, "red")
            row.append(f"[bold {color}]■[/bold {color}]{suffix}")
        table.add_row(*row)
    return table


def run_tracked_grid(row_keys: List[Any], row_labels: List[str], stage_labels: List[str],
                      worker_fn: Callable[[Any, List[str], Dict], Any], workers: int,
                      title: str = "MAJIC MOCAP PIPELINE STATUS",
                      per_cell: bool = False) -> Tuple[Dict, Dict[Any, Any]]:
    """Runs `worker_fn(row_key, stage_labels, shared_state)` across a
    ProcessPoolExecutor pool, rendering a live status table with one row per
    row_key and one status column per stage. `worker_fn` should write
    shared_state[(row_key, stage)] = "Running"/"Success"/"Skipped"/"Failed (...)"
    (and optionally shared_state[(row_key, f'{stage}_time')]) for each stage it's
    given, and return whatever payload should be collected.

    By default (per_cell=False), one process handles ALL stages for a given
    row_key sequentially — worker_fn receives the full stage_labels list. This is
    the right choice when stages share expensive setup (e.g. raw data loaded once
    and reused across every method for that subject/activity). Returns
    {row_key: worker_fn's return value}.

    per_cell=True instead submits one process PER (row_key, stage) pair —
    worker_fn receives a single-element stage list each call. Use this when
    stages are independently expensive and don't share setup worth amortizing
    (e.g. noise-sensitivity combos, where reloading raw data per combo is
    negligible next to the combo's own filter pass): it exposes every cell to the
    worker pool individually instead of serializing a row's stages behind one
    process. Returns {(row_key, stage): worker_fn's return value}.
    """
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    for row_key in row_keys:
        for stage in stage_labels:
            shared_state[(row_key, stage)] = "Pending"

    with Live(_render_grid_table(title, row_keys, row_labels, stage_labels, shared_state),
              refresh_per_second=4) as live:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            if per_cell:
                futures = {executor.submit(worker_fn, row_key, [stage], shared_state): (row_key, stage)
                           for row_key in row_keys for stage in stage_labels}
            else:
                futures = {executor.submit(worker_fn, row_key, stage_labels, shared_state): row_key
                           for row_key in row_keys}

            while any(not f.done() for f in futures):
                time.sleep(0.25)
                live.update(_render_grid_table(title, row_keys, row_labels, stage_labels, shared_state))

            results = {key: future.result() for future, key in futures.items()}

            live.update(_render_grid_table(title, row_keys, row_labels, stage_labels, shared_state))

    return dict(shared_state), results

# ==============================================================================
# Method-name-driven grid workers
# ==============================================================================
# Shared by experiments/benchmark_experiment.py (the vanilla method list) and any
# experiment that sweeps method *names* rather than filter-tuning parameters
# (e.g. experiments/threshold_sensitivity.py's mag_adapt_th<value> sweep,
# experiments/oracle_ablation.py's acc/mag oracle combos). Each is meant to be
# passed straight to run_tracked_grid as the worker_fn, with stage_labels = the
# method names to run for that call.

def generate_joint_angles_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                                  stds: Optional[Dict[str, float]] = None,
                                  variant: Optional[str] = None) -> None:
    """stage_labels = ['load'] + method names. Loads raw data once, then computes
    and saves joint angles for every method, reusing the loaded data across methods.

    `stds` re-tunes the filter (see resolve_stds) and `variant` namespaces the output
    tree; bind both with functools.partial. Pass them together — a re-tuned run with
    variant=None overwrites the default-tuned pipeline's parquets in place.

    Also usable with run_tracked_grid(per_cell=True), where each call receives a single
    method and no 'load' stage: the load then happens once per method instead of once per
    trial, which costs ~3 s against a method's ~3 min of filter time and buys parallelism
    across methods. That matters when a sweep has many arms and few trials — the default
    shape serializes every arm of a trial behind one process, so a one-subject sweep would
    use two cores no matter how many are free. Pass the method names WITHOUT a leading
    'load' in that mode; the load's own status is folded into the method's cell, since a
    per-cell grid has nowhere to show a stage the trial no longer has one of."""
    subject, activity = row_key
    methods = [stage for stage in stage_labels if stage != 'load']
    tracks_load = 'load' in stage_labels

    t_start = time.time()
    if tracks_load:
        shared_state[(row_key, 'load')] = "Running"
    try:
        plates = load_raw_data(subject, activity)
        if tracks_load:
            shared_state[(row_key, 'load_time')] = time.time() - t_start
            shared_state[(row_key, 'load')] = "Success"
    except Exception as e:
        if tracks_load:
            shared_state[(row_key, 'load')] = f"Failed ({e})"
        for method in methods:
            shared_state[(row_key, method)] = f"Failed (load: {e})"
        return None

    for method in methods:
        t_method = time.time()
        shared_state[(row_key, method)] = "Running"
        try:
            df = compute_joint_angles(plates, method, stds=stds)
            if df is not None and not df.empty:
                save_joint_angles(df, subject, activity, method, variant=variant, stds=stds)
                shared_state[(row_key, f"{method}_time")] = time.time() - t_method
                shared_state[(row_key, method)] = "Success"
            else:
                shared_state[(row_key, method)] = "Skipped"
        except Exception as e:
            shared_state[(row_key, method)] = f"Failed ({e})"
    return None


def compute_stats_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                          methods: List[str], stats_name: str,
                          variant: Optional[str] = None,
                          stds: Optional[Dict[str, float]] = None) -> None:
    """Single-stage worker (stage_labels should be a single name, e.g. ['stats']).
    Reads back whatever per-method parquets generate_joint_angles_worker managed to
    save for this row and computes error stats against 'marker'. Naturally
    fails/no-ops if the load or every method failed there, since no parquet files
    exist to read.

    `stats_name` namespaces the output under results/statistics/per_subject/<stats_name>/.
    It is required rather than defaulted: benchmark, oracle-ablation and
    threshold-sweep runs each produce per-subject stats over a different method
    set, and a shared default filename meant whichever ran last silently won.

    `variant` must match the one the generation phase wrote under, and `stds` is
    recorded in the manifest — both default to the untuned pipeline's."""
    subject, activity = row_key
    stage = stage_labels[0]

    t_start = time.time()
    shared_state[(row_key, stage)] = "Running"
    try:
        frames = []
        for method in methods:
            df = load_joint_angles(subject, activity, method, variant=variant)
            if df is not None:
                frames.append(df.assign(subject=f"Subject{subject}", trial_type=activity, method=method))

        if not frames:
            shared_state[(row_key, stage)] = "Failed"
            return None

        all_df = as_categorical_labels(pd.concat(frames, ignore_index=True))
        stats_df = compute_error_stats(all_df)
        if not stats_df.empty:
            stats_path = ensure_parent(paths.per_subject_statistics_path(stats_name, subject, activity))
            stats_df.to_parquet(stats_path, engine='pyarrow')
            write_manifest(stats_path, constants=pipeline_constants(stds), experiment=stats_name,
                           subject=f"Subject{subject}", activity=activity, methods=methods,
                           variant=variant)

        shared_state[(row_key, f"{stage}_time")] = time.time() - t_start
        shared_state[(row_key, stage)] = "Success"
        return None
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
        return None
