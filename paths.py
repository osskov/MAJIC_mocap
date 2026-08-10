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
      Subject01/walking/imu data/*.txt
      Subject01/walking/walking.trc
      Subject01/walking/madgwick (al borno)/

    results/
      joint_angles/Subject01/walking/mag_on.parquet
      statistics/all_subject_statistics.parquet
      statistics/oracle_ablation_statistics.parquet
      statistics/per_subject/<stats_name>/Subject01/walking.parquet
      experiments/noise_sensitivity/...
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

JOINT_ANGLES_DIR = RESULTS_DIR / "joint_angles"
STATISTICS_DIR = RESULTS_DIR / "statistics"
PER_SUBJECT_STATS_DIR = STATISTICS_DIR / "per_subject"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"


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

def raw_trial_dir(subject: str, activity: str) -> Path:
    """Source data folder for one subject/activity: IMU .txt files, the .trc, etc."""
    return DATA_DIR / f"Subject{subject}" / activity

# ==============================================================================
# Outputs: joint angles
# ==============================================================================

def joint_angles_path(subject: str, activity: str, method: str) -> Path:
    return JOINT_ANGLES_DIR / f"Subject{subject}" / activity / f"{method}.parquet"

# ==============================================================================
# Outputs: statistics
# ==============================================================================

def statistics_path(name: str) -> Path:
    """Summary statistics for a named experiment, e.g. 'all_subject', 'oracle_ablation'."""
    return STATISTICS_DIR / f"{name}_statistics.parquet"


def per_subject_statistics_path(stats_name: str, subject: str, activity: str) -> Path:
    """Per-subject/activity statistics, namespaced by experiment.

    The namespacing matters: benchmark, oracle-ablation and threshold-sweep runs
    all produce per-subject stats over different method sets, and previously all
    three wrote the same filename inside the trial folder, so whichever ran last
    silently won.
    """
    return PER_SUBJECT_STATS_DIR / stats_name / f"Subject{subject}" / f"{activity}.parquet"


def all_subject_joint_angles_path() -> Path:
    """The concatenated every-subject/method joint-angle table (large: ~1 GB)."""
    return STATISTICS_DIR / "all_subject_joint_angles.parquet"

# ==============================================================================
# Outputs: per-experiment scratch trees
# ==============================================================================

def experiment_dir(name: str) -> Path:
    """Root for an experiment's own intermediate artifacts, e.g. 'noise_sensitivity'."""
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
