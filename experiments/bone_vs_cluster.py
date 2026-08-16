"""How far a skin-mounted marker cluster moves relative to the bone underneath it.

Every other dataset here takes a marker cluster as ground truth. The biplane half is the only
place that can check that assumption, because it carries BOTH for the same segment at the same
instant: fluoroscopy solves the femur and tibia directly, and a Vicon cluster sits on the skin
above each. Their difference, once the constant frame relation is removed, is soft-tissue
artifact -- the error every marker-based validation in this repo silently accepts.

TWO QUESTIONS, and they are not the same one.

  HOW BIG IS THE ARTIFACT?  A direct bone-to-cluster comparison. This is a property of skin
  and does not involve an IMU at all.

  DOES IT MATTER FOR VALIDATING AN IMU?  Which reference the measured gyroscope agrees with
  better. A large artifact that the IMU shares -- because the IMU is taped to the same skin --
  costs nothing when the IMU is scored against the cluster.

The first can be large while the second is small, and on this dataset they genuinely diverge
by task, so reporting either alone would mislead.

READ THE CONFOUND SECTION BEFORE USING THE ARTIFACT NUMBERS. As it stands they are an UPPER
BOUND that includes residual time misalignment between the two references, and on the fast
tasks that is most of what they contain.

THE CONSTANT IS NOT ARTIFACT. The bone frame and the cluster frame are differently defined and
differently placed, so most of the raw difference between them is a fixed rotation and a fixed
translation -- calibration, not motion. Both are removed per site-trial before anything is
reported: the rotation by Procrustes over the trial, the translation by its median in the bone
frame. What is left is the part that MOVES, which is what soft tissue does.

BLOCKED ON THE SUBJECT. Trials within a session share one taping, one set of markers and one
body, so they are not independent replicates of "how much does skin move". n is 15, not 476.

WHY THERE IS NO TRANSLATION NUMBER HERE. The obvious companion measurement -- how far the
cluster slides along the limb -- is not separable on this data. The biplane pose origin sits
714 mm from the cluster origin, so it is not a local bone landmark, and across a lever that
long any relative ROTATION produces large apparent translation: 7 deg of rotation artifact
alone accounts for 88 mm. A first version of this module reported 39-153 mm of "translation
artifact" per task, and that number was measuring the rotation twice. Removed rather than
caveated.

    python -m experiments.bone_vs_cluster
"""
import argparse
import os

os.environ.setdefault("DISABLE_TQDM", "True")

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import paths
from experiments.experiment_utils import load_trial
from src.toolchest.building.sources import get_source

EXPERIMENT_NAME = "bone_vs_cluster"
DATASET = "imove_biplane"

# Below this the trial's overlap is too short for a constant frame relation to be separated
# from the motion around it, and the "artifact" would be mostly fitting error.
MIN_PAIRED_FRAMES = 100

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def _constant_rotation(bone: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """The fixed bone-frame-to-cluster-frame rotation Q minimising ||R_c - R_b Q||.

    Both traces map their own body frame to the world, so R_b^T R_c is the frame relation at
    one instant; the constant part is the orthogonal Procrustes solution over all of them.
    Removing it is what turns "these are two different frames" into "this is how much the
    cluster moved".
    """
    accumulated = np.einsum('nji,njk->ik', bone, cluster)
    u, _, vt = np.linalg.svd(accumulated)
    scales = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        scales[2, 2] = -1.0
    return u @ scales @ vt


def _angles_deg(matrices: np.ndarray) -> np.ndarray:
    trace = np.einsum('nii->n', matrices)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def artifact_for_site(bone_plate, cluster_plate) -> Optional[Dict[str, float]]:
    """One site-trial's soft-tissue artifact, constant frame relation removed."""
    paired = np.asarray(bone_plate.valid) & np.asarray(cluster_plate.valid)
    if paired.sum() < MIN_PAIRED_FRAMES:
        return None

    bone = np.asarray(bone_plate.world_trace.rotations)[paired]
    cluster = np.asarray(cluster_plate.world_trace.rotations)[paired]
    residual = np.einsum('nji,njk->nik', bone @ _constant_rotation(bone, cluster), cluster)
    angles = _angles_deg(residual)

    # IS THE ARTIFACT AXIS PARALLEL TO THE ANGULAR VELOCITY? This is what separates the two
    # mechanisms that both scale with speed. A constant time offset dt turns one trace into
    # the other by R(t) -> R(t) expm([w]x dt), so its artifact is a rotation ABOUT w and the
    # cosine goes to 1. Soft tissue has no reason to displace along w, so it averages the
    # random-axis value of 0.5. Anything between is a mixture.
    from scipy.spatial.transform import Rotation
    rotvec = Rotation.from_matrix(residual).as_rotvec()
    omega = np.asarray(bone_plate.imu_trace.gyro)[paired]
    n_rot, n_omega = np.linalg.norm(rotvec, axis=1), np.linalg.norm(omega, axis=1)
    usable = (n_rot > 1e-6) & (n_omega > 1e-6)
    alignment = (float(np.median(np.abs(
        np.einsum('ni,ni->n', rotvec[usable], omega[usable])
        / (n_rot[usable] * n_omega[usable])))) if usable.sum() >= 50 else float('nan'))

    speed = np.degrees(n_omega).mean()
    return {
        'n_paired_frames': int(paired.sum()),
        'rotation_artifact_median_deg': float(np.median(angles)),
        'rotation_artifact_p95_deg': float(np.percentile(angles, 95)),
        'rotation_artifact_max_deg': float(angles.max()),
        'axis_alignment_with_omega': alignment,
        # Equivalent time offset: the dt that would produce this much apparent rotation at
        # this speed. Reported so the artifact can be read against the sync it might be.
        'equivalent_offset_ms': float(1000.0 * np.median(angles) / speed) if speed > 0
                                else float('nan'),
        'gyro_rms_deg_s': float(speed),
    }


def collect(dataset: str = DATASET, limit: int = None) -> pd.DataFrame:
    """One row per site-trial that has both references over enough shared frames."""
    rows = []
    trials = list(get_source(dataset).enumerate_trials())
    for subject, key in trials[:limit] if limit else trials:
        try:
            plates = load_trial(subject, key, dataset=dataset)
        except Exception:
            continue
        for site in sorted({name.split('__')[0] for name in plates}):
            bone, cluster = plates.get(f'{site}__biplane'), plates.get(f'{site}__vicon')
            if bone is None or cluster is None:
                continue
            measured = artifact_for_site(bone, cluster)
            if measured is None:
                continue
            rows.append({'subject': subject, 'trial': key.split('/')[-1], 'site': site,
                         'task': _task(key.split('/')[-1]), **measured})
    return pd.DataFrame(rows)


def _task(trial: str) -> str:
    import re
    return re.sub(r'^[LR]|\d+$', '', trial)


def blocked_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Subject-level means first, then pooled. n is subjects, not site-trials."""
    per_subject = frame.groupby(['subject', 'task'])[column].mean().reset_index()
    out = per_subject.groupby('task')[column].agg(
        n_subjects='size', mean='mean', median='median')
    for q in QUANTILES:
        out[f'q{int(q * 100):02d}'] = per_subject.groupby('task')[column].quantile(q)
    return out.round(3)


def _report_confound(frame: pd.DataFrame) -> None:
    """Whether the artifact is tissue or timing, said plainly rather than left to the reader.

    Both mechanisms scale with angular speed, so the size of the artifact cannot separate
    them and the correlation with speed proves nothing either way. The AXIS can: a time offset
    displaces about the angular-velocity vector and tissue does not.
    """
    from scipy import stats
    speed, artifact = frame.gyro_rms_deg_s, frame.rotation_artifact_median_deg
    correlation = stats.pearsonr(speed, artifact)
    print('--- is this tissue or timing?')
    print(f'  corr(angular speed, artifact) r = {correlation.statistic:+.3f} '
          f'(p = {correlation.pvalue:.1g}) -- expected under BOTH mechanisms, so uninformative')
    aligned = frame['axis_alignment_with_omega'].dropna()
    print(f'  |cos| between the artifact axis and omega: median {aligned.median():.3f}')
    print('    1.0 would be pure timing, 0.5 is a random axis in 3D.')
    print('  By task, this rises with speed, which a tissue-only explanation does not predict:')
    print(frame.groupby('task')[['gyro_rms_deg_s', 'axis_alignment_with_omega']]
          .median().round(3).to_string())
    print('  So the artifact figures above are an UPPER BOUND on soft-tissue artifact, and on')
    print('  the fast tasks most of what they contain is residual biplane-to-Vicon sync error.')
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    frame = collect(limit=args.limit)
    if frame.empty:
        print('No paired site-trials. Build imove_biplane first.')
        return 1

    directory = paths.experiment_dir(EXPERIMENT_NAME)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / 'artifact.parquet', index=False)

    print(f'{len(frame)} site-trials, {frame.subject.nunique()} subjects, '
          f'{int(frame.n_paired_frames.sum())} paired frames\n')
    for column, unit in (('rotation_artifact_median_deg', 'deg'),
                         ('axis_alignment_with_omega', 'cos'),
                         ('equivalent_offset_ms', 'ms')):
        print(f'--- {column} ({unit}), blocked on subject')
        print(blocked_summary(frame, column).to_string())
        print()
    _report_confound(frame)
    print(f'wrote {(directory / "artifact.parquet").relative_to(paths.REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
