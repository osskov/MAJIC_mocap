"""
The registry of datasets the build script knows how to ingest.

One `TrialSource` per dataset, supplying the four things the cache layer would otherwise
have to assume: what trials exist, where each one's files are, which of those files affect
its contents, and how to turn them into PlateTrials.

Before this, all four were hardcoded to the Al Borno layout — `paths.raw_trial_dir` templated
`Subject{n}/{activity}`, `SUBJECTS x ACTIVITIES` was a cross product of two constants, and the
cache key hashed a fixed tuple of globs. IMoVE matches none of them: its sessions are `s2`
through `s25l`, its trials are `t1_walking_001`, and its files are `mocap_data/*.csv` beside
`imu_data/*.txt`.

Adding a dataset means adding an entry here and a reader beside it. Nothing else changes.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import paths

from ..PlateTrial import PlateTrial
from . import alborno, imove_mocap


@dataclass(frozen=True)
class TrialSource:
    """How to find, key and load one dataset's trials.

    `enumerate_trials` lists what is actually on disk rather than a cross product of subjects
    and trial names. The distinction is not cosmetic: Subjects 05, 08 and 10 have no
    complexTasks trial, and the old cross product reported those as failures until an
    'absent' status was added to paper over it. A source that enumerates cannot produce the
    problem.

    `source_globs` are relative to `source_dir` and decide the cache key. They must cover
    everything the loader reads and nothing it does not — an over-broad glob invalidates the
    cache whenever an unrelated file moves, and a missing one lets a real change slip
    through. Loader changes are caught separately, by the toolchest digest.
    """
    name: str
    enumerate_trials: Callable[[], List[Tuple[str, str]]]
    source_dir: Callable[[str, str], Path]
    source_globs: Tuple[str, ...]
    load: Callable[[str, str], Dict[str, PlateTrial]]


# ==============================================================================
# Al Borno et al. (2022) — this repo's original dataset
# ==============================================================================

def _alborno_dir(subject: str, trial: str) -> Path:
    return paths.DATA_DIR / f"Subject{subject}" / trial


def _alborno_trials() -> List[Tuple[str, str]]:
    """Every Subject<NN>/<trial>/ that actually holds a .trc, in sorted order."""
    found = []
    for subject_dir in sorted(paths.DATA_DIR.glob("Subject*")):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name.replace("Subject", "")
        for trial_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            if any(trial_dir.glob("*.trc")):
                found.append((subject, trial_dir.name))
    return found


ALBORNO = TrialSource(
    name='alborno',
    enumerate_trials=_alborno_trials,
    source_dir=_alborno_dir,
    # Deliberately narrower than "everything under the folder": the trial directories also
    # hold a 66 MB .mtb (the raw Xsens binary, never parsed) and a 'madgwick (al borno)/'
    # subdirectory of third-party outputs whose filenames COLLIDE with the real IMU ones.
    # No bare '*.txt': every Al Borno trial keeps its IMU files in 'imu data/', verified
    # across all 19, so the loose glob only ever meant "any stray note dropped in the folder
    # invalidates this trial".
    source_globs=('*.trc', 'imu data/*.txt'),
    load=lambda subject, trial: alborno.load_trial(_alborno_dir(subject, trial)),
)


# ==============================================================================
# IMoVE / CMU-MBL — OptiTrack markers beside Xsens MTw2
# ==============================================================================

IMOVE_ROOT = paths.DATA_DIR / 'IMoveLab_Raw_Data' / 'mocap_ref' / 'data'


def _imove_dir(session: str, trial: str) -> Path:
    return IMOVE_ROOT / session


def _imove_trials() -> List[Tuple[str, str]]:
    """Every (session, inertial record) that has both IMU files and a mocap take.

    Enumerated over the INERTIAL records, not the mocap ones, because that is the unit a
    trial is: the long-walk sessions hold three Motive takes against one continuous Xsens
    file, and `imove_mocap.mocap_takes_for` gathers them under that one record. Enumerating
    mocap instead would claim three trials where there is one recording.

    Skips inertial records with no mocap at all -- the long-walk sessions also carry
    t2_treadmill_walking and t7_cmjdl IMU files that were never mocapped.
    """
    found = []
    for session_dir in sorted(p for p in IMOVE_ROOT.glob('s*') if p.is_dir()):
        imu_dir, mocap_dir = session_dir / 'imu_data', session_dir / 'mocap_data'
        if not imu_dir.is_dir() or not mocap_dir.is_dir():
            continue
        records = sorted({p.name.split('-000_')[0] for p in imu_dir.glob('*-000_*.txt')})
        for trial in records:
            if imove_mocap.mocap_takes_for(session_dir, trial):
                found.append((session_dir.name, trial))
    return found


IMOVE = TrialSource(
    name='imove',
    enumerate_trials=_imove_trials,
    source_dir=_imove_dir,
    # Whole directories: unlike Al Borno there is nothing here that is not input, and a
    # long-walk trial legitimately reads three mocap files whose names it cannot predict
    # from the trial name alone.
    source_globs=('mocap_data/*.csv', 'imu_data/*.txt'),
    load=lambda session, trial: imove_mocap.load_trial(_imove_dir(session, trial), trial),
)


SOURCES: Dict[str, TrialSource] = {
    ALBORNO.name: ALBORNO,
    IMOVE.name: IMOVE,
}


def get_source(dataset: str) -> TrialSource:
    """The registered source for `dataset`, or a ValueError naming what is available."""
    try:
        return SOURCES[dataset]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Registered: {sorted(SOURCES)}.") from None
