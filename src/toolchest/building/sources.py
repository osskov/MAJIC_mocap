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
from typing import Callable, Dict, List, Optional, Set, Tuple

import paths

from ..PlateTrial import PlateTrial
from . import alborno, biplane, imove_mocap


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

    `extra_inputs` COVERS WHAT source_dir CANNOT REACH, and exists because one dataset does not
    keep a trial's inputs in one directory. A biplane trial reads four separate trees --
    Kinematics for the bone poses, Vicon for the markers, IMUs for the BioStamps, and a
    trigger workbook at the root -- and `source_dir` can name only one of them. Without this
    the key hashed the Kinematics CSVs alone, so editing the markers, the inertial export or
    the trigger times left every artifact validating as fresh. Returns absolute paths; a
    source whose inputs really do live in one place leaves it None.
    """
    name: str
    enumerate_trials: Callable[[], List[Tuple[str, str]]]
    source_dir: Callable[[str, str], Path]
    source_globs: Tuple[str, ...]
    load: Callable[[str, str], Dict[str, PlateTrial]]
    extra_inputs: Optional[Callable[[str, str], List[Path]]] = None

    # WHAT THIS TRIAL COULD HAVE PRODUCED, when the dataset-wide roster is the wrong answer.
    #
    # Coverage is otherwise scored against the union of every plate name the dataset produced
    # anywhere, which is right when every trial instruments the same body -- Al Borno's eight
    # segments, IMoVE's fifteen. It is wrong when the roster varies by trial: a biplane trial
    # images ONE knee, so it can produce four plates out of the eight names that exist, and
    # scoring it against all eight reported 1552 sensor-trials absent from a complete dataset.
    #
    # Returns None to mean "use the union", so a source that does not need this says nothing.
    expected_plates: Optional[Callable[[str, str], Optional[Set[str]]]] = None


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
    load=lambda subject, trial, report=None: alborno.load_trial(
        _alborno_dir(subject, trial), report=report),
)


# ==============================================================================
# IMoVE / CMU-MBL — OptiTrack markers beside Xsens MTw2
# ==============================================================================

IMOVE_ROOT = paths.DATA_DIR / 'IMoveLab_Raw_Data' / 'mocap_ref' / 'data'


def _imove_dir(session: str, trial: str) -> Path:
    return IMOVE_ROOT / session


# Inertial records this dataset has but this pipeline does not carry, with the reason.
#
# THE STATIC POSES CANNOT BE SYNCED, and that is not a bug to work around. Every trial's
# IMU-to-mocap lag is found by cross-correlating the measured gyroscope against the angular
# velocity implied by the markers, and a held pose has neither: all 19 fail with per-plate lag
# estimates scattered over 1.2-4.1 s, because there is no motion for a correlation peak to
# form on. Their alignment is unverifiable for the same reason -- the fitted sensor-to-segment
# rotation comes out with a residual of 1.006 times the signal, meaning it explains none of it.
#
# So they are excluded HERE rather than left to fail 19 times per build. A trial that is not
# enumerated is not built, not loaded, not counted in coverage and not scored, which is the
# honest state for data whose ground-truth correspondence cannot be established.
#
# This is a pipeline decision, not a claim that the recordings are worthless: a static pose is
# still the natural place to measure a sensor noise floor or a bias, and anything doing that
# reads the raw files directly and needs no mocap correspondence at all.
#
# NOTHING IS DELETED. The source recordings are untouched, and the 18 parquets an earlier build
# left in results/trials/imove stay where they are -- excluded from enumeration, so no build
# writes them, no table counts them and no loader reaches them, but still on disk if the
# exclusion is ever revisited. They are stale against the current digest as well, so even a
# direct load_trial by name raises rather than returning them.
UNSYNCABLE_TRIALS = ('t0_static_pose_001',)


def _imove_trials() -> List[Tuple[str, str]]:
    """Every (session, inertial record) that has both IMU files and a mocap take.

    Enumerated over the INERTIAL records, not the mocap ones, because that is the unit a
    trial is: the long-walk sessions hold three Motive takes against one continuous Xsens
    file, and `imove_mocap.mocap_takes_for` gathers them under that one record. Enumerating
    mocap instead would claim three trials where there is one recording.

    Skips inertial records with no mocap at all -- the long-walk sessions also carry
    t2_treadmill_walking and t7_cmjdl IMU files that were never mocapped -- and the static
    poses, which have no motion to sync on. See UNSYNCABLE_TRIALS.
    """
    found = []
    for session_dir in sorted(p for p in IMOVE_ROOT.glob('s*') if p.is_dir()):
        imu_dir, mocap_dir = session_dir / 'imu_data', session_dir / 'mocap_data'
        if not imu_dir.is_dir() or not mocap_dir.is_dir():
            continue
        records = sorted({p.name.split('-000_')[0] for p in imu_dir.glob('*-000_*.txt')})
        for trial in records:
            if trial in UNSYNCABLE_TRIALS:
                continue
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
    load=lambda session, trial, report=None: imove_mocap.load_trial(
        _imove_dir(session, trial), trial, report=report),
)


# ==============================================================================
# IMoVE / CMU-MBL — the biplane half: MC10 IMUs against fluoroscopy bone poses
# ==============================================================================

def _biplane_dir(subject: str, key: str) -> Path:
    """The Kinematics directory for one capture.

    A SOURCE-INVENTORY GAP LIVES HERE, and it is worth stating rather than discovering. A
    biplane trial reads from three trees -- Kinematics, Vicon and IMUs -- but `source_dir`
    names one directory and `source_globs` is a static tuple, so only the kinematics files
    reach the cache key. Editing a .c3d or a BioStamp export will NOT invalidate the artifact
    built from it.

    That is fail-open, which this codebase rejects elsewhere, and it is a deliberate stopgap:
    the alternative under the current TrialSource shape is to glob the whole 19 GB study for
    every trial. The toolchest digest still catches reader changes, so what escapes is
    specifically an edit to the raw data, which for a published dataset is rare. Fixing it
    properly means letting a source report the paths it actually opened.
    """
    session, block, trial = key.split('/')
    return biplane.BIPLANE_ROOT / 'Kinematics' / biplane.STUDY / subject / session / block / trial


# The biplane half's equivalent of UNSYNCABLE_TRIALS, for the same reason and matched on the
# task rather than the whole name -- these are called Lstatic1, Rstatic2 and so on.
#
# Measured on the one static trial already built, the fitted sensor-to-segment rotation comes
# out with a residual of 2.87 times the signal on the shank and 2.00 on the thigh. Above 1.0
# means the rotation is worse than doing nothing, so there is no alignment to speak of: a held
# pose gives the gyro correlation nothing to lock onto, exactly as on the mocap half.
BIPLANE_UNSYNCABLE_TASKS = ('static',)


def _biplane_trials() -> List[Tuple[str, str]]:
    """(subject, 'session/block/trial') for every capture that has a trigger time.

    Gated on the trigger, because without one there is no way onto the IMU clock and the
    trial cannot be assembled however complete its files are. Static holds are skipped as
    well -- see BIPLANE_UNSYNCABLE_TASKS.
    """
    triggers = biplane.trigger_times()
    found = []
    for subject in sorted({s for s in triggers.subject.unique()}):
        named = set(triggers[triggers.subject == subject].trial)
        for session, block, trial in biplane.biplane_trials(subject):
            if biplane.trial_task(trial) in BIPLANE_UNSYNCABLE_TASKS:
                continue
            if trial in named:
                found.append((subject, f'{session}/{block}/{trial}'))
    return found


def _biplane_inputs(subject: str, key: str) -> List[Path]:
    """Everything outside the Kinematics directory that a biplane trial reads.

    Four trees, and `source_dir` names only the first. The Vicon markers set the trial's
    clock, the BioStamp exports are the measurement, and the trigger workbook is what puts
    them on the same timeline -- so a change to any of them changes the artifact, and until
    this existed none of them was in the cache key.

    Narrowed to what this trial reads, matching load_trial: the two sites on the imaged knee,
    and within each only accel.csv and gyro.csv. A whole-directory glob would pull in
    `accel-errors.csv`, which nothing parses, and all four sites, so a change to the right leg
    would invalidate every left-leg trial.
    """
    session, block, trial = key.split('/')
    found = [biplane.vicon_path(subject, session, trial),
             biplane.BIPLANE_ROOT / 'TriggerTimes.xlsx']

    side = biplane.trial_side(trial)
    if side is not None:
        imu_root = biplane.BIPLANE_ROOT / 'IMUs' / biplane.STUDY / subject
        for site in biplane.BONE_TO_SITE.values():
            found += [imu_root / f'{site}_{side}' / name
                      for name in ('accel.csv', 'gyro.csv')]
    return [p for p in found if p.exists()]


def _biplane_expected(subject: str, key: str) -> Optional[Set[str]]:
    """The four plates a biplane trial can produce: its imaged side, two sites, two references.

    A trial's leading letter picks the knee the fluoroscopy imaged, so `RSDrop1` can only ever
    yield the right thigh and right shank -- against Vicon and against biplane. The other four
    names in the dataset belong to the other leg and are not missing from this trial in any
    sense worth reporting.
    """
    side = biplane.trial_side(key.split('/')[-1])
    if side is None:
        return None
    return {f'{site}_{side}__{reference}'
            for site in biplane.BONE_TO_SITE.values()
            for reference in ('vicon', 'biplane')}


IMOVE_BIPLANE = TrialSource(
    name='imove_biplane',
    enumerate_trials=_biplane_trials,
    source_dir=_biplane_dir,
    source_globs=('HomoTransMatrices_*.csv',),
    load=lambda subject, key, report=None: biplane.load_trial(subject, key, report=report),
    expected_plates=_biplane_expected,
    extra_inputs=_biplane_inputs,
)


SOURCES: Dict[str, TrialSource] = {
    ALBORNO.name: ALBORNO,
    IMOVE.name: IMOVE,
    IMOVE_BIPLANE.name: IMOVE_BIPLANE,
}


def get_source(dataset: str) -> TrialSource:
    """The registered source for `dataset`, or a ValueError naming what is available."""
    try:
        return SOURCES[dataset]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Registered: {sorted(SOURCES)}.") from None
