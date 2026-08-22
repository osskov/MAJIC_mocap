"""
Single source of truth for every filesystem path in this repo.

The one rule
------------
`data/` is READ-ONLY source data, exactly as downloaded from SimTK / Google Drive
(see README). Nothing in this repo ever writes under `data/`. Every generated
artifact goes under `results/` (tabular data) or `plots/` (figures).

That rule is enforced mechanically, not by convention: every write helper here
calls `_reject_if_under_data()`, so an accidental write into the source tree
raises instead of silently polluting it.

Layout
------
    data/                                       <- inputs, never written
      alborno/Subject01/walking/imu data/*.txt
      alborno/Subject01/walking/walking.trc
      alborno/Subject01/walking/madgwick (al borno)/
      IMoveLab_Raw_Data/mocap_ref/data/s13/...

    results/
      joint_angles/alborno/01/walking/mag_on.parquet
      joint_angles/imove/s13/t1_walking_001/mag_on.parquet
      statistics/all_subject_alborno_statistics.parquet
      statistics/oracle_ablation_statistics.parquet
      statistics/per_subject/<stats_name>/alborno/01/walking.parquet
      experiments/filter_gains/...
      experiments/drift_observability/...

    plots/

Provenance
----------
Every written artifact gets a `<stem>.manifest.json` sidecar recording the git
SHA, whether the tree was dirty, the timestamp, the invoking command line, and
the physical constants in force. Outputs are overwritten in place on re-run, so
the sidecar is what tells you which code version produced a given file — see
`write_manifest`.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

# ==============================================================================
# Roots
# ==============================================================================
# Anchored to this file, not the working directory, so every script resolves the
# same paths regardless of where it was invoked from.

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "plots"

# One directory per dataset under data/, so the source tree says which dataset a
# subject belongs to. Al Borno's eleven subject folders used to sit loose at the top of
# data/ — from when there was only one dataset — beside IMoveLab_Raw_Data, which meant
# a bare `Subject01` was ambiguous the moment a second dataset arrived (IMoVE's mocap
# tree has its own `Subject01` under the biplane half). Everything that reaches for an
# Al Borno file goes through here or `raw_trial_dir`.
ALBORNO_DIR = DATA_DIR / "alborno"
IMOVE_DIR = DATA_DIR / "IMoveLab_Raw_Data"

JOINT_ANGLES_DIR = RESULTS_DIR / "joint_angles"
STATISTICS_DIR = RESULTS_DIR / "statistics"
PER_SUBJECT_STATS_DIR = STATISTICS_DIR / "per_subject"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
TRIALS_DIR = RESULTS_DIR / "trials"


def _reject_if_under_data(path: Path) -> Path:
    """Guards the read-only-inputs rule. Raises if `path` would land under data/."""
    resolved = Path(path).resolve()
    if resolved == DATA_DIR or DATA_DIR in resolved.parents:
        raise ValueError(
            f"Refusing to write inside the read-only source tree: {resolved}\n"
            f"data/ holds inputs only; generated artifacts belong under {RESULTS_DIR} or {PLOTS_DIR}."
        )
    return resolved


def ensure_parent(path: Path) -> Path:
    """Creates the parent directory for an output path, enforcing the data/ guard."""
    _reject_if_under_data(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

# ==============================================================================
# Inputs (read-only)
# ==============================================================================

def alborno_subject_dir(subject: str) -> Path:
    """One Al Borno subject's folder: `data/alborno/Subject01`."""
    return ALBORNO_DIR / f"Subject{subject}"


def raw_trial_dir(subject: str, activity: str) -> Path:
    """Al Borno source data folder for one subject/activity: IMU .txt files, the .trc, etc.

    Al Borno-specific despite the generic name — the other datasets do not lay their
    trials out as `Subject<NN>/<activity>` and reach their files through their own
    helpers in `toolchest.building.sources`.
    """
    return alborno_subject_dir(subject) / activity

# ==============================================================================
# Outputs: cached trials
# ==============================================================================

def cached_trial_path(dataset: str, subject: str, trial: str) -> Path:
    """One trial's fully-loaded PlateTrials, cached as a flat table.

    Materializes the expensive half of loading: marker reconstruction, IMU/mocap
    cross-correlation sync and sensor-to-segment alignment. Parsing the source files
    is the cheap part (a 184 MB CSV reads in 0.7 s); it is these derived steps that
    cost seconds per trial, and — more importantly — they are *decisions* that are
    otherwise recomputed silently on every run and never written down. The sidecar
    manifest is what makes them auditable.

    `subject` and `trial` are taken verbatim rather than templated, because the
    datasets disagree on how a subject is named: this repo's own tree uses
    'Subject01', IMoVE's motion-capture half uses 's13l'. `dataset` is the namespace
    that keeps them from colliding — the source trees both contain a 'Subject01'.
    """
    return TRIALS_DIR / dataset / subject / f"{trial}.parquet"

# ==============================================================================
# Outputs: joint angles
# ==============================================================================

BENCHMARK_EXPERIMENT = 'benchmark'


def joint_angles_dir(experiment: Optional[str] = None) -> Path:
    """Root of a joint-angle tree. None is the CANONICAL one, which only the benchmark writes.

    Every other experiment gets `results/experiments/<experiment>/joint_angles/`, beside that
    experiment's other intermediate artifacts — see `joint_angles_write_path`, which is what
    enforces it, and `experiment_dir`, whose layout this matches.
    """
    # BENCHMARK_EXPERIMENT and None are the SAME TREE, and that equivalence belongs here rather
    # than only in `joint_angles_write_path`. It was only there at first, which meant writes with
    # the sentinel landed in results/joint_angles/ while READS with the same sentinel resolved to
    # results/experiments/benchmark/joint_angles/ — a directory nothing creates. The benchmark's
    # own statistics phase therefore found none of its arms. One rule, both directions.
    if experiment is None or experiment == BENCHMARK_EXPERIMENT:
        return JOINT_ANGLES_DIR
    return experiment_dir(experiment) / "joint_angles"


def joint_angles_path(dataset: str, subject: str, trial: str, method: str,
                      variant: Optional[str] = None,
                      experiment: Optional[str] = None) -> Path:
    """Per-method joint angles for one trial. `variant` namespaces an alternative
    filter TUNING under its own subdirectory.

    A method NAME already encodes everything the method-spec grammar can express —
    mag mode, oracles, normalization, threshold, distortion scale — so those need no
    namespacing: they produce distinct filenames. The gyro/acc/mag standard
    deviations live outside the name (they are module constants, see
    experiment_utils.DEFAULT_*_STD), so two runs at different stds would otherwise
    write the same filename and the second would silently overwrite the first under a
    name that says nothing about which tuning produced it. Anything that overrides
    those stds passes a variant; the default-tuned pipeline passes None.

    `dataset` IS REQUIRED AND HAS NO DEFAULT, and that is the point. The layout was
    `Subject{n}/{activity}/{method}.parquet` for as long as there was one dataset, and both
    halves of that break on the others: IMoVE's subjects are 's13l' rather than a zero-padded
    number, and 'Subject' + 's13l' is not a name anything recognises. Worse, the trial names
    collide — every dataset has trials this pipeline would write to the same file — so a
    default would let an IMoVE run silently overwrite Al Borno's parquets under a path that
    claims to be Al Borno's. Making it required moves that from a wrong answer to an
    unrunnable call.

    `subject` and `trial` are taken VERBATIM, matching `cached_trial_path` — the build tree
    this reads its inputs from is already keyed that way, so the two trees line up
    trial-for-trial and a trial key containing slashes (the biplane half's
    'Test1/A/RSDrop1') lands at the same depth in both.
    """
    root = joint_angles_dir(experiment)
    root = root if variant is None else root / variant
    return root / dataset / subject / trial / f"{method}.parquet"


def joint_angles_write_path(experiment: str, dataset: str, subject: str, trial: str,
                            method: str, variant: Optional[str] = None) -> Path:
    """Where `experiment` is ALLOWED to write a joint-angle parquet.

    ONE TREE, ONE WRITER. `results/joint_angles/` belongs to the benchmark and to nothing
    else. Every other experiment writes under its own
    `results/experiments/<experiment>/joint_angles/`, and this function is what makes that
    a property of the code rather than a convention someone remembers.

    The reason is the failure it prevents, which has happened repeatedly: a method name is
    keyed by mag mode and oracles and nothing else, so `threshold_sensitivity`,
    `oracle_ablation` and the benchmark all wanted to write the same `mag_on.parquet`.
    While they compute the same thing that is merely wasteful; the moment a tuning or a
    constant differs between two runs, one experiment's arms silently become a mix of two
    estimators, every file still loads, every figure still renders, and the only symptom is
    a number that should be impossible. Two full sweeps were thrown away to that before it
    was diagnosed.

    Reading is NOT restricted, and deliberately so — an experiment comparing itself against
    the benchmark's arms should read them, which is what `joint_angles_path(...)` with no
    `experiment` is for. Only writing is owned.

    Raises unless `experiment` is given. A default would recreate exactly the hole this
    closes: the script that forgets to declare itself is the one that overwrites the
    benchmark, and it would do so silently.
    """
    if not experiment:
        raise ValueError(
            "joint_angles_write_path needs an experiment name. Pass "
            f"paths.BENCHMARK_EXPERIMENT ({BENCHMARK_EXPERIMENT!r}) if this IS the benchmark, "
            f"otherwise the experiment's own name — its parquets then land under "
            f"{EXPERIMENTS_DIR / '<experiment>' / 'joint_angles'} instead of in the "
            f"benchmark's tree.")
    # No special case for the benchmark here: `joint_angles_dir` maps its sentinel onto the
    # canonical tree, so read and write agree by construction.
    return joint_angles_path(dataset, subject, trial, method, variant=variant,
                             experiment=experiment)

# ==============================================================================
# Outputs: statistics
# ==============================================================================

def statistics_path(name: str) -> Path:
    """Summary statistics for a named experiment, e.g. 'all_subject', 'oracle_ablation'."""
    return STATISTICS_DIR / f"{name}_statistics.parquet"


def per_subject_statistics_path(stats_name: str, dataset: str, subject: str,
                                trial: str) -> Path:
    """Per-trial statistics, namespaced by experiment and dataset.

    The namespacing matters: benchmark, oracle-ablation and threshold-sweep runs
    all produce per-subject stats over different method sets, and previously all
    three wrote the same filename inside the trial folder, so whichever ran last
    silently won. `dataset` is the same guard one level out — see `joint_angles_path`,
    where the trial names of two datasets collide.

    `subject` and `trial` are verbatim, as in `joint_angles_path` and `cached_trial_path`.
    """
    return PER_SUBJECT_STATS_DIR / stats_name / dataset / subject / f"{trial}.parquet"


def filter_config_path(name: str, dataset: str) -> Path:
    """The FILTER CONFIGURATION a benchmark run actually used, one row per method.

    Separate from the statistics manifest beside it, and not redundant with it. A manifest
    records what the run BELIEVED — `compute_stats_worker` stamps it with the tuning the command
    line asked for — while this table is built by reading the joint-angle sidecars back off disk,
    so it records what the arms on disk were REALLY computed with. When those two disagree the
    statistics are from a different filter than their own manifest claims, which has happened and
    cost a long wrong investigation: a `--stats-only` run re-aggregated angles carrying
    acc_std=0.018 / mag_std=0.05 and stamped them 0.09695 / 0.009695, a 28x difference in how far
    the magnetometer was trusted, and the resulting "mag_on ~= mag_off at every joint" was an
    artifact of the mismatch.

    Queryable rather than prose: one row per (method, arm) with its resolved spec, the stds the
    run intended, the stds the parquets carry, and whether they match.
    """
    return STATISTICS_DIR / f"{name}_{dataset}_filters.parquet"


def all_subject_joint_angles_path(dataset: str) -> Path:
    """The concatenated every-trial/method joint-angle table for one dataset.

    Large — ~1 GB for Al Borno's 19 trials and roughly ten times that for IMoVE's 261 —
    which is why the benchmark writes it only on request. The summary statistics do not
    depend on it: `compute_error_stats` already groups per trial, so the dataset summary is
    the concatenation of the per-trial tables.
    """
    return STATISTICS_DIR / f"all_subject_{dataset}_joint_angles.parquet"

# ==============================================================================
# Outputs: per-experiment scratch trees
# ==============================================================================

def experiment_dir(name: str) -> Path:
    """Root for an experiment's own intermediate artifacts, e.g. 'filter_gains'."""
    return EXPERIMENTS_DIR / name

# ==============================================================================
# Outputs: figures
# ==============================================================================

def plots_dir(*parts: str) -> Path:
    """Figure output directory, optionally namespaced, e.g. plots_dir('oracle_ablation')."""
    return PLOTS_DIR.joinpath(*parts)

# ==============================================================================
# Provenance
# ==============================================================================

@lru_cache(maxsize=1)
def git_provenance() -> Dict[str, Any]:
    """Current git SHA/branch and whether the tree is dirty.

    Cached: the pipeline writes hundreds of artifacts per run and this would
    otherwise fork git once per file. Values are best-effort — a missing git or a
    non-repo checkout yields nulls rather than failing the run.
    """
    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                 text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    status = _git("status", "--porcelain")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


def manifest_path(output_path: Path) -> Path:
    """Sidecar path for an artifact: foo.parquet -> foo.manifest.json"""
    return Path(output_path).with_suffix(".manifest.json")


def write_manifest(output_path: Path, constants: Optional[Dict[str, Any]] = None, **extra: Any) -> Path:
    """Writes the provenance sidecar for `output_path`.

    Records git state, UTC timestamp, the invoking command line, the physical
    constants in force, and whatever caller-specific keys are passed as `extra`
    (method name, subject, activity, swept parameter values, ...).

    Outputs are overwritten in place on re-run, so this sidecar is the only record
    of which code version and which constants produced the file sitting next to it.
    """
    path = Path(output_path)
    _reject_if_under_data(path)

    manifest = {
        **git_provenance(),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "written_by": Path(sys.argv[0]).name if sys.argv and sys.argv[0] else None,
        "argv": sys.argv[1:],
        "artifact": path.name,
    }
    if constants:
        manifest["constants"] = constants
    manifest.update(extra)

    target = manifest_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return target


def read_manifest(output_path: Path) -> Optional[Dict[str, Any]]:
    """Reads the provenance sidecar for an artifact, or None if it has none."""
    target = manifest_path(output_path)
    if not target.exists():
        return None
    return json.loads(target.read_text())
