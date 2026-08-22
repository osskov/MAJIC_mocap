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
import ast
import hashlib
import json
import sys
import warnings
import re
import time
import multiprocessing
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from rich.live import Live
from rich.table import Table

import paths
from paths import (DATA_DIR, RESULTS_DIR, ensure_parent, write_manifest, read_manifest)
from src.toolchest.building import biplane
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

# Namespace for this repo's own source tree, from Al Borno et al. (2022), and the default
# dataset for everything here that predates the second one. The other trees register their own
# in src/toolchest/building/sources.py; the namespace exists because they all contain a
# 'Subject01' and, before it, a trial name was not a unique key. See the "Trial cache" section
# below for what it keys, and paths.joint_angles_path for why it is not optional there.
TRIAL_DATASET = 'alborno'

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

# Default filter tuning, PER DATASET. Hoisted out of _run_relative_filter's signature so the
# values can be recorded in every output's provenance manifest — their meaning depends on
# whether RelativeFilter normalizes its vector measurements, which is not visible from here.
#
# ONLY THE RATIOS ARE REAL. The kernel adds process noise as dt^2 * Q against a measurement
# covariance built from R, so multiplying all three stds by a common factor leaves the
# steady-state Kalman gain exactly unchanged; the only thing it can touch is how fast P
# leaves its seed P0 = init_orientation_std^2 * I. That is not an argument, it is measured:
# `filter_gains.py --stage scale` scales a whole row by 1e-4 to 1e2 and the pooled RMSE moves
# by 0.001 deg on Al Borno's 19 trials across all six decades. The absolute magnitude of
# these numbers therefore carries no information about the filter's behaviour, and the rows
# below are written at SENSOR-NOISE SCALE — gyro_std pinned at 0.0045, the value the pipeline
# shipped with — purely so they read as sensor stds and stay comparable to what they replace.
# That number has no known provenance (see filter_gains.LEGACY_STDS) and the choice carries no
# information, because the scale is inert. Any common rescaling of a row is the same filter.
#
# WHAT CHANGED IS THE TWO RATIOS, and they were badly wrong. R does not model sensor noise;
# it models everything the measurement equation gets wrong, and that equation claims the
# accelerometer reads gravity and the magnetometer reads a constant field. Neither is true
# while someone is walking. Measured directly — substitute the mocap rotations into
# e = R_wp v_p - R_wc v_c and take what is left — the accelerometer's innovation is 2.5 m/s^2
# per sensor on Al Borno against the old constant of 0.018, and the magnetometer's is 0.094
# against 0.05. The two channels are wrong by very different factors, which is precisely why
# their RATIO was off:
#
#     acc_std / gyro_std     4.0  ->  21.5      (accelerometer distrusted 5.4x more)
#     mag_std / acc_std      2.8  ->   0.1      (magnetometer trusted 28x more)
#
# In the currency the filter uses — information about a direction error goes as |v|^2/sigma^2
# at the nominal magnitudes below — the old tuning gave the accelerometer 743x the
# magnetometer's weight. It is now 0.96:1, i.e. the two sensors carry the same weight. On Al
# Borno that swap is worth 40% of mag_on's pooled RMSE (15.9 deg -> 9.5 deg).
#
# The magnetometer is deliberately NOT set at its measured residual, which would put it 7x
# AHEAD of the accelerometer. Its residual has a 0.6-1.1 s autocorrelation time and survives a
# 2 Hz low-pass essentially unchanged — it is a slowly varying distortion bias, not the white
# noise a Kalman filter assumes, and a filter handed the true variance of a correlated error
# tracks the error instead of averaging it out. The sweep prices that in; the residual
# measurement on its own cannot.
#
# PER DATASET, BUT LESS SO THAN EXPECTED. Different sensors in different rooms ought to want
# different gains, and the measured innovations do differ — but almost entirely in the scale,
# which is inert. Al Borno and IMoVE come out on IDENTICAL ratios (their accelerometer
# residuals are 2.49 and 2.20, a 13% difference that the filter cannot see), so those two rows
# are the same filter written twice. The biplane rows differ for real, because those IMUs have
# no magnetometer and their trials dead-reckon for two minutes before anything is scoreable.
# See experiments/filter_gains.py for how each row was fixed.
#
# UNITS ARE PHYSICAL — rad/s, m/s^2, and the magnetometer's own export scale (Xsens
# normalizes at calibration so a nominal Earth field reads 1.0). The 'ekf' arm normalizes its
# vector measurements to unit length and therefore cannot consume these directly; it takes
# them through `rescale_stds`, which divides by NOMINAL_*_MAGNITUDE. That is a unit
# conversion, not a second tuning, and METHODS['ekf'] sets both flags together for exactly
# that reason.
DATASET_STDS: Dict[str, Dict[str, float]] = {}   # filled in below, after NOMINAL_* are defined

# Backwards-compatible names for the default dataset's tuning. Kept because a dozen call
# sites and `scratch/` read them directly, and because TRIAL_DATASET is what an unqualified
# "the default tuning" has always meant here.
DEFAULT_GYRO_STD = 0.0045
DEFAULT_ACC_STD = 0.09695
DEFAULT_MAG_STD = 0.009695

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
# (see MAG_UNIT in experiments/global_assumptions.py); measured |mag| medians across
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
#
# AND IT IS DUE A RE-RUN. The threshold itself is independent of the filter tuning — o^J is
# computed from the projected traces, not from the filter state — but which threshold is BEST
# is not: the gate decides when to stop trusting the magnetometer, and DATASET_STDS now trusts
# the magnetometer ~740x more relative to the accelerometer than it did when this sweep last
# ran. The previously measured result (every joint's optimum an endpoint, never an interior
# threshold) was obtained at a tuning where the magnetometer was carrying almost no weight,
# so it does not carry over.
DEFAULT_MAG_ADAPT_THRESHOLD = 1000.0


STD_KEYS = ('gyro_std', 'acc_std', 'mag_std')

# The tuning each dataset's filters run at. Every row is `experiments/filter_gains.py`'s
# output for that dataset: the acc/gyro and mag/gyro ratios from the sweep, the scale from
# the measured accelerometer innovation. See the DATASET_STDS comment above the constants for
# what the numbers mean and why they are 20-180x the values that preceded them.
#
# Filled in here rather than at the declaration above because the biplane rows are written
# relative to the marker datasets' and read better next to each other.
# The scale every row is written at. Conventional, not measured — the measurement says the
# scale is free (see above) — and set to the gyro_std the pipeline already shipped with, so
# the numbers below read as sensor stds and stay directly comparable to the triple they
# replace. It is NOT a published noise figure; the measured static gyro floor is 0.0077.
SCALE_ANCHOR_GYRO_STD = 0.0045

DATASET_STDS.update({
    # 19 trials, Xsens MTw, Y-up marker world frame. acc/gyro 21.5, mag/gyro 2.15; pooled
    # mag_on optimum 9.47 deg against 15.9 deg at the tuning this replaced.
    'alborno': {'gyro_std': 0.0045, 'acc_std': 0.09695, 'mag_std': 0.009695},

    # 236 trials, 40 and 100 Hz. IDENTICAL TO AL BORNO, and that is a result rather than a
    # copied row. IMoVE's own argmin is acc/gyro 46.4, but its surface scores 21.5 at 10.03
    # deg against 9.89 — 1.1% apart, so the tie-break in filter_gains.shipped_row takes the
    # value nearer the measured residual, and both datasets land on 21.5. What genuinely
    # differs between them is the scale (accelerometer innovation 2.20 against Al Borno's
    # 2.49), and the scale does not reach the filter. Two Xsens datasets in two labs want the
    # same gains, which is worth knowing: the innovation is dominated by soft-tissue motion
    # and joint-centre error, not by the IMU.
    'imove': {'gyro_std': 0.0045, 'acc_std': 0.09695, 'mag_std': 0.009695},

    # MC10 BioStamps, no magnetometer, Z-up fluoroscopic world frame, 2 knees. Two specs read
    # the same build tree against two ground truths and they do NOT agree: the bone-referenced
    # arm's accelerometer innovation is 11.4 m/s^2 against the Vicon cluster's 7.7, and its
    # best achievable RMSE is 71 deg against 37. That gap is the sync between the fluoroscopic
    # pose and the IMU, not the tuning.
    #
    # THE SWEEP FOUND ALMOST NOTHING TO WIN HERE and both rows should be read in that light.
    # These trials carry 95-190 fluoroscopic frames about 114 s into a 240 s BioStamp record,
    # so the filter dead-reckons for two minutes with no magnetometer before anything is
    # scoreable; every point on either surface is above 36 deg and the old default already sat
    # within 2% of the best. The accelerometer innovation is also measurable ONLY inside that
    # window — i.e. only during the drop landing or run stance — so it is an upper bound on
    # what the filter sees for the other 99.7% of the trial. The acc/gyro ratios (4.64 and 10)
    # are lower than the marker datasets' 21.5 because with no magnetometer and two minutes of
    # dead reckoning the filter has nothing but the accelerometer to lean on.
    #
    # mag_std is finite and inert. The channel is all zeros, which zeroes its rows of H, so its
    # value cannot change the estimate — but its variance still enters S, and a NaN there would
    # propagate through the Cholesky solve into every state. It is set from the mag/acc ratio
    # both marker datasets land on (see filter_gains.MAG_OVER_ACC_WHEN_ABSENT).
    'imove_biplane': {'gyro_std': 0.0045, 'acc_std': 0.02089, 'mag_std': 0.002089},
    'imove_biplane_vicon': {'gyro_std': 0.0045, 'acc_std': 0.045, 'mag_std': 0.0045},
})


def resolve_stds(stds: Optional[Dict[str, float]] = None,
                 dataset: str = TRIAL_DATASET) -> Dict[str, float]:
    """The three filter stds in force for a run: `dataset`'s row of DATASET_STDS, with any
    key present in `stds` overriding it.

    Exists so a re-tuning is passed as data (one dict, threaded down to
    _run_relative_filter and into every manifest) rather than by reassigning the
    module constants, which multiprocessing workers would not see — each worker
    re-imports this module in a fresh interpreter, so a parent-process mutation of
    DEFAULT_ACC_STD would silently run the default tuning in every child while the
    parent's manifest claimed the override.

    `dataset` is the OUTPUT dataset name (TrackingSpec.dataset), not the build tree:
    'imove_biplane' and 'imove_biplane_vicon' read the same trials but are separate specs,
    and a tuning keyed on the build tree could not tell them apart if they ever diverged.

    An unknown dataset raises rather than falling back to Al Borno. A silent fallback is the
    exact failure this table exists to prevent — a new dataset would run at another lab's
    magnetometer weighting and nothing in the manifest would say so.

    Unknown keys raise too: 'acc_stdev' or 'mag' would otherwise be dropped on the floor
    and the run would quietly use the default for that sensor."""
    if dataset not in DATASET_STDS:
        raise ValueError(f"No filter tuning for dataset '{dataset}'. Known: "
                         f"{sorted(DATASET_STDS)}. Measure one with "
                         f"`python -m experiments.filter_gains --dataset {dataset}`.")
    resolved = dict(DATASET_STDS[dataset])
    if stds:
        unknown = sorted(set(stds) - set(STD_KEYS))
        if unknown:
            raise ValueError(f"Unknown filter std key(s) {unknown}. Allowed: {list(STD_KEYS)}")
        resolved.update({key: float(value) for key, value in stds.items()})
    return resolved


def pipeline_constants(stds: Optional[Dict[str, float]] = None,
                       dataset: str = TRIAL_DATASET) -> Dict[str, Any]:
    """The physical/tuning constants in force, for provenance manifests.

    `stds` records an override rather than the defaults — a manifest that reported
    DEFAULT_ACC_STD next to a run tuned elsewhere is worse than no manifest. `dataset` is
    recorded alongside for the same reason: now that the tuning is per dataset, three stds
    with no note of which row they came from cannot be checked against anything."""
    return {
        'expected_gravity': EXPECTED_GRAVITY.tolist(),
        **resolve_stds(stds, dataset=dataset),
        'stds_dataset': dataset,
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
# What a dataset gives the tracking pipeline
# ==============================================================================
# The four things above this line that were Al Borno constants and are not properties of the
# PHYSICS: which sensor pairs form a joint, which way is up, which sensor the magnetometer
# oracle is referenced against, and whether there is a magnetometer at all.
#
# Bundled into one frozen record rather than passed as four arguments because it travels
# through a multiprocessing boundary: `generate_joint_angles_worker` is bound with
# functools.partial in the parent and unpickled in each child, and one object that either
# arrives whole or does not is much harder to half-thread than four keyword arguments, three
# of which have plausible-looking defaults.
#
# It deliberately does NOT hold the filter tuning. That travels separately as `stds` (see
# resolve_stds) because a re-tuning is a property of the RUN and namespaces its own output
# tree, while this is a property of the data.


@dataclass(frozen=True)
class TrackingSpec:
    """One dataset's answer to "what can be tracked here, and against what".

    Built from `global_assumptions.DatasetSpec` by `global_assumptions.tracking_spec`, which is
    where the dataset registry lives; this module only defines the shape and the Al Borno
    default, so that it stays importable without pulling the registry in (global_assumptions
    imports this module, so the dependency cannot run the other way).

    `joints` is the same {name: (parent_sensor, child_sensor)} table `JOINTS` has always been,
    but with the dataset's own plate names in it — 'pelvis_imu' on Al Borno, 'PELVIS_M' on
    IMoVE, 'lateral_thigh_left__biplane' on the biplane half.

    `gravity` is gravity AS AN ACCELEROMETER READS IT, in this dataset's mocap world frame.
    Y-up for the two marker-referenced datasets; the biplane half's fluoroscopy world frame is
    Z-UP, measured at [0.79, -1.58, 9.90] on its built trials, and getting it wrong is silent —
    it does not raise, it just corrupts the EKF's world reference and both acc oracles. The
    three relative-filter arms never consult it (they are seeded from mocap and estimate a
    relative rotation), which is why a dataset can be usefully benchmarked before this value is
    pinned down, but the EKF arm cannot.

    `mag_reference` names the ONE sensor whose world-frame field defines the magnetometer
    oracle and the EKF's virtual ground plate. One clean sensor rather than a pooled average is
    the long-standing choice here and is the opposite of what `global_assumptions` wants for
    its own measurements — see the note on `DatasetSpec.clean_sensor`.

    `has_magnetometer` is False where the IMUs have none (the biplane half's MC10 BioStamps),
    and it is a REFUSAL rather than a flag: see `check_method`.

    `dataset` NAMES THE OUTPUTS and `build_dataset` NAMES THE INPUTS, and they differ where two
    specs analyse one build tree. The biplane half is built once and carries every IMU twice —
    against the fluoroscopic bone pose and against the Vicon marker cluster — so 'imove_biplane'
    and 'imove_biplane_vicon' read the same trials and must not write the same files: the joint
    names and the method names are identical between them, so a shared output namespace would
    have the second run overwrite the first and the difference between two ground truths, which
    is the whole point of running both, would be unmeasurable.
    """
    dataset: str
    joints: Dict[str, Tuple[str, str]]
    gravity: np.ndarray
    mag_reference: Optional[str]
    has_magnetometer: bool = True
    subject_label: str = '{}'
    trials_dataset: Optional[str] = None

    @property
    def build_dataset(self) -> str:
        """The build tree under results/trials/ this spec's trials are read from."""
        return self.trials_dataset or self.dataset

    @property
    def sensors(self) -> Tuple[str, ...]:
        """Every sensor named by a joint, deduplicated, in joint-table order.

        What `narrow_plates` keeps, and it is narrower than "the plates in the trial" on two
        datasets: IMoVE carries three sensors per thigh and shank where only the Mid one spans
        a benchmarked joint, and every biplane trial carries each IMU TWICE — once against the
        fluoroscopic bone pose and once against the Vicon cluster. Handing the EKF path all of
        them would estimate orientations nothing scores, and on the biplane half it would let
        two specs over one build silently mix references.
        """
        seen = {}
        for parent, child in self.joints.values():
            seen.setdefault(parent, None)
            seen.setdefault(child, None)
        return tuple(seen)

    def label_subject(self, subject: str) -> str:
        """A subject id as the statistics tables should print it: 'Subject01', 's13l'.

        Separate from the id used in paths, which is verbatim (see paths.joint_angles_path).
        The label keeps Al Borno's 'Subject01' in the `subject` column that every existing
        figure and table groups on, without putting a 'Subject' prefix on 's13l'.
        """
        return self.subject_label.format(subject)

    def narrow_plates(self, plates: Dict[str, PlateTrial]) -> Dict[str, PlateTrial]:
        """`plates`, keeping only the sensors this spec's joints name.

        Missing sensors are simply absent from the result rather than an error: a biplane trial
        images ONE knee, so half this spec's sensors are legitimately not in it, and an Al Borno
        subject can be missing a segment to a reconstruction failure. The per-joint loops
        already skip a joint whose parent or child is absent, so a partial trial yields partial
        results instead of none.
        """
        return {name: plates[name] for name in self.sensors if name in plates}

    def check_method(self, method: str) -> None:
        """Raises if `method` asserts a magnetometer this dataset does not have.

        THE FAILURE THIS PREVENTS IS A MISLABELLED RESULT, not a wrong number. The biplane
        reader fills `mag` with exact zeros, the filter normalizes each vector measurement and
        passes a zero vector through untouched, so the sensor's residual and its Jacobian block
        both drop out of the update — which is precisely the documented mag_off path. Running
        'mag_on' there therefore succeeds and produces mag_off's answer under mag_on's name, and
        a magnetometer-comparison figure built from it would show the two methods agreeing
        perfectly on a dataset that never measured a field.

        'ekf' is NOT refused, though its magnetometer channel is equally inert there. The name
        makes no claim about the magnetometer — it names an absolute filter, and an acc-only
        absolute filter is the only absolute baseline this dataset can support, so it is a real
        arm rather than a mislabelled one. The distinction is exactly what the name asserts.
        """
        if self.has_magnetometer:
            return
        spec = resolve_method_spec(method)
        if spec.get('mag_mode') in ('on', 'adapt'):
            raise ValueError(
                f"Method '{method}' asks for the magnetometer, but dataset "
                f"'{self.dataset}' has none — its IMUs measure acceleration and rotation only. "
                f"The filter would silently produce mag_off's answer under this name. Use "
                f"'mag_off' (or 'ekf' for an acc-only absolute baseline).")
        if spec.get('mag_source') != 'real':
            raise ValueError(
                f"Method '{method}' asks for a magnetometer oracle or a scaled magnetic "
                f"distortion, but dataset '{self.dataset}' has no magnetometer to reference: "
                f"its field would be the zero vector and every distortion scale would be the "
                f"same run.")


# Al Borno et al. (2022), and the default everywhere a spec is optional — every experiment in
# this repo predating the second dataset ran on it, so this keeps their behaviour bit-identical.
ALBORNO_TRACKING = TrackingSpec(
    dataset=TRIAL_DATASET,
    joints=JOINTS,
    gravity=EXPECTED_GRAVITY,
    mag_reference='torso_imu',
    has_magnetometer=True,
    subject_label='Subject{}',
)

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

# The dataset namespace a cached trial is keyed under is TRIAL_DATASET, hoisted to the
# configuration section above so the tracking specs can name it.
#
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
# Every module whose contents can change what lands in a cached trial. A module missing from
# here is a SILENT stale cache: the parquet keeps being served after the code that produced
# it changed. `resampling.py` was missing until 2026-08-12, which is the worst possible
# omission -- it is the module whose rewrite moved the measured cluster-to-IMU offsets by
# 70-90%, and it only failed loudly because PlateTrial.py happened to be edited alongside it.
# SHARED core: everything that shapes a cached trial regardless of which dataset it came
# from. `trial_io.py` is in here because it WRITES the parquet -- change the float32 cast,
# the plate sort order or the rot_ij convention and every artifact is different, yet its only
# guard was a hand-maintained SCHEMA_VERSION. That is exactly the "someone forgets to bump
# the constant" failure this digest exists to make impossible. Split from the per-reader modules below so that editing one dataset's parser does not
# invalidate the other dataset's artifacts -- alborno.py was invalidating all 262 IMoVE
# trials, which is fail-closed but needlessly so.
_CORE_MODULES = ('PlateTrial.py', 'WorldTrace.py', 'IMUTrace.py', 'gyro_utils.py',
                 'finite_difference_utils.py', 'resampling.py', 'trial_io.py',
                 'building/assembly.py', 'building/reconstruction.py',
                 'building/sources.py')

# Per-dataset readers, hashed only into their own dataset's key.
_READER_MODULES = {
    'alborno': ('building/alborno.py', 'building/xsens.py'),
    'imove': ('building/imove_mocap.py', 'building/xsens.py'),
    # No xsens here: the biplane half's IMUs are MC10 BioStamps with their own CSV export,
    # and its ground truth is fluoroscopy rather than markers.
    'imove_biplane': ('building/biplane.py',),
}

# Kept as the union so anything still asking for "every module that matters" gets the honest
# answer, and so a new reader cannot be added without appearing here.
_CONTENT_MODULES = tuple(sorted(set(_CORE_MODULES).union(
    *(mods for mods in _READER_MODULES.values()))))

# Cutoff for the alignment-residual diagnostic below.
_RESIDUAL_LOWPASS_HZ = 10.0

# Low-passed alignment residual above which a plate is called SUSPECT, AS A FRACTION OF THAT
# PLATE'S OWN LOW-PASSED GYRO SIGNAL.
#
# It used to be an absolute 25 deg/s, and that threshold was measuring SPEED rather than
# quality. Across 3303 plates the absolute residual correlates with the gyro RMS at r = +0.60,
# and the fraction of plates it flags orders almost perfectly by how fast the activity is:
#
#     treadmill running   94.0% flagged        lat_step            13.0% flagged
#     treadmill walking   83.9%                squat               26.0%
#     walking             74.9%                static pose          0.0%
#
# Nobody believes 94% of running plates are misaligned and no static pose is. Normalizing by
# the plate's own signal breaks that dependence (r = -0.31) and the ordering disappears.
#
# It is wrong in the other direction too. A static pose flags at 0% on the absolute rule while
# its NORMALIZED residual is 1.006 -- the fitted rotation explains none of the measured motion,
# because there is no motion to fit it on. That is the plate whose alignment is least
# trustworthy in the dataset and the old rule called it clean.
#
# 0.5 is set off the same distribution: the normalized residual has q50 0.33 and q95 0.74
# across every non-static plate, so this flags roughly the worst 10-15% within each activity
# rather than whichever activity was fastest.
RESIDUAL_WARN_FRACTION = 0.5

# The absolute threshold, kept only so the old figure and the tests that pin it still resolve.
# Not used for the suspect flag any more. See RESIDUAL_WARN_FRACTION for why.
RESIDUAL_WARN_DEG_S = 25.0

# Below this the sensor is holding still and gravity is the whole accelerometer signal, so its
# magnitude is a scale check. Generous: a limb "at rest" in a standing trial still sways, and
# a tighter gate finds no frames at all on half the dataset.
STATIC_GYRO_LIMIT_DEG_S = 5.0
# Fewer than this and the median is dominated by whichever handful of frames qualified.
STATIC_MIN_FRAMES = 50
GRAVITY_MS2 = 9.80665


def _semantic_source(path: Path) -> bytes:
    """A module's CODE, with comments and docstrings removed.

    Hashing raw bytes made every prose edit a full rebuild, and this codebase is deliberately
    comment-dense -- most invalidations were re-explaining something, not changing behaviour.
    Parsing to an AST and dumping it drops comments (the tokenizer never keeps them) and this
    strips docstrings explicitly, so the digest moves on semantic edits and only those.

    Still FAIL-CLOSED, which is the property that matters: it is derived from the whole file
    rather than from an enumerated list of the constants someone remembered to include, so a
    new behaviour cannot slip in unhashed. Falls back to raw bytes if a file will not parse,
    because refusing to hash is worse than hashing too much.

    `ast.unparse`, NOT `ast.dump`. dump emits internal field names, which change between
    CPython releases -- 3.12 and 3.13 produce different digests for identical source, so two
    interpreters on one machine could not share a cache and each invalidated the other's
    281 artifacts on every switch. unparse emits canonical source and was verified identical
    across both. A digest that moves when the interpreter does is fail-closed but useless.
    """
    source = path.read_bytes()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree).encode()


@lru_cache(maxsize=None)
def _toolchest_digest(dataset: str = None) -> str:
    """SHA-256 over the code that produces a cached trial's contents.

    Scoped: the shared core plus `dataset`'s own reader. Passing None hashes everything,
    which is what a caller wanting "did any of it change" should ask for.
    """
    modules = list(_CORE_MODULES)
    if dataset is None:
        modules = list(_CONTENT_MODULES)
    else:
        modules += list(_READER_MODULES.get(dataset, ()))

    digest = hashlib.sha256()
    toolchest = Path(__file__).resolve().parent.parent / 'src' / 'toolchest'
    for name in sorted(set(modules)):
        digest.update(name.encode())
        digest.update(_semantic_source(toolchest / name))
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
def _source_inventory(folder: Path, globs: Tuple[str, ...],
                      extra: Sequence[Path] = ()) -> List[Dict[str, Any]]:
    """Names and byte counts of the files a trial load reads, sorted.

    `globs` come from the dataset's TrialSource, because which files matter is a property
    of the dataset, not of the cache. Al Borno's are `*.trc` and `imu data/*.txt`; IMoVE's
    will be `mocap_data/*.csv` and `imu_data/*.txt`.

    Sizes rather than content hashes: `data/` is declared read-only (see paths.py), so the
    realistic failure is a file being replaced or a re-download landing a different trial,
    both of which change the size. Hashing 76 MB of .trc on every cache check would buy
    protection against an edit the repo's own rules forbid.

    `extra` carries inputs that do not live under `folder` at all -- see
    TrialSource.extra_inputs. Named relative to the data root rather than to `folder`, since
    the whole point is that they are somewhere else, and relative_to would raise.
    """
    found = {p for glob in globs for p in folder.glob(glob)}
    rows = [{'name': str(p.relative_to(folder)), 'bytes': p.stat().st_size}
            for p in sorted(found) if p.is_file() and not p.name.startswith('.')]
    for path in sorted(set(extra)):
        if not path.is_file() or path.name.startswith('.'):
            continue
        try:
            name = str(path.relative_to(paths.DATA_DIR))
        except ValueError:
            name = str(path)
        rows.append({'name': name, 'bytes': path.stat().st_size})
    return rows


def _content_key(frame, plates: Dict[str, PlateTrial]) -> Dict[str, int]:
    """What the artifact should contain, so a SHORT READ is a miss rather than a wrong answer.

    cached_trial_status never opened the parquet -- it compared manifest fields and returned
    'fresh'. A truncated file therefore passed. Both numbers are already computed for the
    diagnostics; they just were not in the part that gets checked.
    """
    return {'n_rows': int(len(frame)), 'n_plates': int(len(plates))}


def trial_cache_key(subject: str, trial: str, dataset: str = TRIAL_DATASET,
                    content: Dict[str, int] = None) -> Dict[str, Any]:
    """Everything that determines a cached trial's contents.

    Compared field-by-field against the stored manifest on load; any difference is a cache
    miss. `align` is in here because the sensor-to-segment alignment rewrites every rotation
    in the file, and `dataset` selects which source's layout and globs to key against.
    """
    source = get_source(dataset)
    return {
        'schema_version': trial_io.SCHEMA_VERSION,
        'toolchest_digest': _toolchest_digest(dataset),
        'sources': _source_inventory(
            source.source_dir(subject, trial), source.source_globs,
            extra=source.extra_inputs(subject, trial) if source.extra_inputs else ()),
        **(content or {}),
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
           'gyro_signal_rms_deg_s': rms(measured),
           'n_invalid_frames': int((~valid).sum())}

    fs = float(plate.imu_trace.get_sample_frequency())
    cutoff = min(_RESIDUAL_LOWPASS_HZ, 0.4 * fs / 2.0)
    # filtfilt's default padlen is 3 * max(len(a), len(b)); a trace shorter than that
    # raises rather than returning something approximate.
    if fs > 0 and len(plate) > 30:
        b, a = butter(4, cutoff / (fs / 2.0), btype='low')
        low_residual = rms(filtfilt(b, a, residual, axis=0))
        # AGAINST THE SIGNAL IN THE SAME BAND. Dividing a low-passed residual by a broadband
        # signal would flatter every fast trial, since the denominator picks up energy the
        # numerator has had removed -- reintroducing by the back door exactly the speed
        # dependence the fraction exists to remove.
        low_signal = rms(filtfilt(b, a, measured, axis=0))
        out['gyro_residual_lowpass_rms_deg_s'] = low_residual
        out['gyro_signal_lowpass_rms_deg_s'] = low_signal
        out['residual_fraction'] = (low_residual / low_signal if low_signal > 0
                                    else float('nan'))
        out['residual_lowpass_hz'] = cutoff
    return out


def _static_calibration(plate: PlateTrial) -> Dict[str, float]:
    """The accelerometer scale check, taken where it actually means something.

    `acc_norm_median` -- the median of |acc| over the WHOLE trial -- was being read against
    9.81 as a calibration check, and on a dynamic trial that reading is simply wrong. All 159
    plates more than 1 m/s^2 from gravity are treadmill running, topping out at 16.97, and a
    running limb genuinely spends more than half its time above 1 g. Nothing is miscalibrated;
    the statistic was answering a different question from the one being asked of it.

    A scale error only shows up where the sensor is NEARLY STILL, because there gravity is the
    entire signal. So the frames are selected first, on the gyroscope -- which is independent
    of the accelerometer being checked, and would not be if the selection used |acc| itself:
    picking frames where |acc| is near 9.81 and then reporting that |acc| is near 9.81 is
    circular and would hide the very error it is looking for.

    Returns nothing when the plate never holds still. That is honest -- a treadmill-running
    plate carries no evidence about its own scale -- and it is why this is separate from
    `acc_norm_median` rather than replacing it.
    """
    gyro = np.degrees(np.linalg.norm(np.asarray(plate.imu_trace.gyro), axis=1))
    still = (gyro < STATIC_GYRO_LIMIT_DEG_S) & np.asarray(plate.valid)
    if still.sum() < STATIC_MIN_FRAMES:
        return {'n_static_frames': int(still.sum())}
    magnitudes = np.linalg.norm(np.asarray(plate.imu_trace.acc)[still], axis=1)
    return {'n_static_frames': int(still.sum()),
            'acc_norm_static_median': float(np.median(magnitudes)),
            'acc_scale_error': float(np.median(magnitudes) / GRAVITY_MS2 - 1.0)}


def trial_diagnostics(plates: Dict[str, PlateTrial], report=None) -> Dict[str, Any]:
    """Per-trial and per-plate quality numbers, recorded in the cache manifest.

    The point is triage at scale: with 22 trials you notice a bad one by eye, with the
    660-odd IMoVE adds. Having these in the manifests means a bad sync is a query over
    sidecars rather than a filter run that produces nonsense.

    `report` supplies the one thing the plates cannot say about themselves: whether their
    cluster-to-IMU offset was FITTED or DEFAULTED. A plate carries `sensor_offset` either way,
    so `sensor_offset_mm` alone cannot be told apart from a nominal constant, and on 10.4% of
    IMoVE's taped sensors it is one. That does not touch a joint angle -- `shift_world_origin`
    moves positions only -- but it costs a lever-arm acceleration error with a median of
    0.04 m/s^2 and a p95 of 0.30, so anything comparing measured against mocap-derived
    ACCELERATION wants to know. It goes in the manifest rather than being left in the sidecar
    because the manifest is what `load_trial` surfaces.
    """
    fallbacks = {}
    if report is not None:
        rows = report.to_frame()
        lever = rows[(rows.step == 'S8_lever_arm')
                     & (rows.metric.isin(('used_fallback', 'distance_from_nominal_mm')))]
        for entity, group in lever.groupby('entity'):
            fallbacks[entity] = {row.metric: float(row.value_num)
                                 for row in group.itertuples()
                                 if row.value_num is not None}
    any_plate = next(iter(plates.values()))
    per_plate = {}
    for name in sorted(plates):
        plate = plates[name]
        per_plate[name] = {
            'n_frames': len(plate),
            'acc_norm_median': float(np.median(np.linalg.norm(plate.imu_trace.acc, axis=1))),
            **_static_calibration(plate),
            # What shift_world_origin moved this plate's pose by, so the shift is auditable
            # and a re-derivation can add it back. Without it a refit measures the RESIDUAL
            # and pasting that in as the new constant walks it to zero one run at a time.
            'sensor_offset_mm': [float(v) for v in np.asarray(plate.sensor_offset) * 1000.0],
            # 1.0 means the offset is the nominal constant, not this trial's own fit.
            **{f'sensor_offset_{key}': value
               for key, value in fallbacks.get(name, {}).items()},
            **_alignment_residuals(plate),
        }
    return {
        'n_plates': len(plates),
        'n_frames': len(any_plate),
        'duration_s': float(any_plate.imu_trace.timestamps[-1] - any_plate.imu_trace.timestamps[0]),
        'sample_rate_hz': float(any_plate.imu_trace.get_sample_frequency()),
        'world_frame_gravity': measure_world_frame_gravity(plates).tolist(),
        # WHETHER THERE IS A MAGNETOMETER AT ALL. An MC10 BioStamp has an accelerometer and a
        # gyroscope and nothing else, so the biplane reader fills `mag` with zeros.
        #
        # This is recorded rather than enforced, and the reason is worth stating: a zero
        # magnetometer does NOT produce a fabricated heading. The filter normalizes each vector
        # measurement to unit length and passes a zero vector through untouched, which is the
        # documented mag_off path -- so the sensor's residual and its Jacobian block both drop
        # out of the update. Running mag_on against these trials silently yields mag_off
        # behaviour instead. The risk is therefore MISLABELLING a result, not computing a wrong
        # one, and the fix for that is a queryable fact rather than a refusal layer.
        'has_magnetometer': bool(np.any(any_plate.imu_trace.mag)),
        # Frames invalid on ANY plate: a joint angle needs two plates, so one bad plate
        # takes the whole frame out of every joint it participates in.
        'n_invalid_frames_any_plate': int(sum(
            ~np.logical_and.reduce([p.valid for p in plates.values()]))),
        'plates': per_plate,
        # Derived at write time so load_trial can warn without recomputing anything.
        # Keyed on the NORMALIZED residual: the absolute one flagged 94% of treadmill-running
        # plates and 0% of static poses, which is a speed detector wearing a quality label.
        # A plate with no `residual_fraction` at all (too short to filter) is not called
        # suspect -- absence of evidence is not evidence, and .get's 0.0 default says clean.
        'suspect': sorted(name for name, stats in per_plate.items()
                          if stats.get('residual_fraction', 0.0)
                          > RESIDUAL_WARN_FRACTION),
    }


def build_report_path(dataset: str, subject: str, trial: str) -> Path:
    """Where a trial's BuildReport sidecar lives, beside its parquet."""
    path = paths.cached_trial_path(dataset, subject, trial)
    return path.with_suffix('.build.parquet')


def save_build_report(report, subject: str, trial: str,
                      dataset: str = TRIAL_DATASET) -> Optional[Path]:
    """Writes one trial's BuildReport sidecar. Returns the path, or None if there was none.

    Split out of `save_cached_trial` so THE FAILURE PATH CAN CALL IT TOO. The report used to
    be written only on the way past a successful save, which meant a trial that raised
    produced no diagnostics at all -- and a build that raises is precisely the one whose
    diagnostics you want. Whatever the readers measured before the exception is still valid
    and is now kept.

    Written atomically but NOT part of the cache key: it describes the artifact rather than
    determining it, so adding a metric must not invalidate 281 trials. A missing or outdated
    sidecar therefore means "tier-1 unavailable for this trial", which build_quality reports
    rather than treating as staleness.
    """
    if report is None:
        return None
    frame = report.to_frame()
    if frame.empty:
        return None
    report_path = ensure_parent(build_report_path(dataset, subject, trial))
    staging = report_path.with_suffix(report_path.suffix + '.tmp')
    try:
        frame.to_parquet(staging, engine='pyarrow', index=False)
        os.replace(staging, report_path)
    finally:
        staging.unlink(missing_ok=True)
    return report_path


def save_cached_trial(plates: Dict[str, PlateTrial], subject: str, trial: str,
                      dataset: str = TRIAL_DATASET, report=None) -> Path:
    """Writes a trial's PlateTrials plus the manifest that validates them on load.

    ATOMIC. The parquet goes to a temporary name and is renamed into place, because the two
    writes below are not one operation: interrupt a --force run between them and the PREVIOUS
    manifest is still on disk. Its cache key is unchanged -- that is precisely what --force
    means -- so a half-written parquet validates as fresh and stays that way forever.
    os.replace is atomic within a filesystem, so a reader sees the old file or the new one.
    """
    source = get_source(dataset)
    path = ensure_parent(paths.cached_trial_path(dataset, subject, trial))
    frame = trial_io.plates_to_frame(plates)

    staging = path.with_suffix(path.suffix + '.tmp')
    try:
        frame.to_parquet(staging, engine='pyarrow', index=False)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)

    save_build_report(report, subject, trial, dataset=dataset)

    write_manifest(
        path,
        cache_key=trial_cache_key(subject, trial, dataset,
                                  content=_content_key(frame, plates)),
        dataset=dataset, subject=subject, trial=trial,
        source=str(source.source_dir(subject, trial).relative_to(paths.REPO_ROOT)),
        diagnostics=trial_diagnostics(plates, report=report),
    )
    return path


def cached_trial_status(subject: str, trial: str,
                        dataset: str = TRIAL_DATASET) -> Tuple[str, Optional[str]]:
    """(status, reason). See `_cached_trial_state`, which also hands back the manifest."""
    status, reason, _ = _cached_trial_state(subject, trial, dataset)
    return status, reason


def _cached_trial_state(subject: str, trial: str, dataset: str = TRIAL_DATASET
                        ) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
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
        return 'absent', f'no source trial at {folder.relative_to(paths.REPO_ROOT)}', None

    path = paths.cached_trial_path(dataset, subject, trial)
    if not path.exists():
        return 'missing', None, None

    manifest = read_manifest(path)
    if manifest is None:
        return 'stale', 'no manifest sidecar', None

    stored = manifest.get('cache_key')
    if stored is None:
        return 'stale', 'manifest predates cache_key', manifest

    expected = trial_cache_key(subject, trial, dataset)
    for field, want in expected.items():
        if stored.get(field) != want:
            if field == 'sources':
                return 'stale', 'source files changed', manifest
            return 'stale', f'{field}: cached {stored.get(field)!r} != current {want!r}', manifest

    # The only field checked against the ARTIFACT rather than against the inputs. Everything
    # above compares manifest to code and source files, which a truncated parquet passes
    # unchanged -- and a --force run interrupted between the two writes leaves exactly that.
    # Reading the row count costs a footer read, not a load.
    claimed_rows = stored.get('n_rows')
    if claimed_rows is None:
        return 'stale', 'manifest predates the content check', manifest
    actual_rows = _parquet_row_count(path)
    if actual_rows != claimed_rows:
        return 'stale', (f'n_rows: file holds {actual_rows}, manifest claims {claimed_rows} '
                         f'— the artifact is truncated'), manifest
    return 'fresh', None, manifest


def _parquet_row_count(path: Path) -> Optional[int]:
    """Rows in a parquet, from its footer. None if the file cannot be opened at all."""
    import pyarrow.parquet as pq
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


class SuspectTrialWarning(UserWarning):
    """A loaded trial has plates whose alignment residual is above the warn threshold."""


class SuspectTrial(RuntimeError):
    """Raised instead of the warning when `load_trial(..., strict=True)`."""


class StaleArtifact(RuntimeError):
    """A cached artifact no longer matches the code, configuration or inputs behind it.

    One base class for all three cache layers -- trials, joint angles, per-trial statistics --
    so a caller that genuinely wants to survive any of them catches one name, and so the rule
    reads the same at every layer: an artifact whose provenance cannot be verified is refused,
    not served with a footnote.
    """


class StaleTrialCache(StaleArtifact):
    """A trial's parquet is missing, or was built from different inputs or code.

    Raised rather than silently falling back to parsing from source. That fallback made a
    stale cache cost time and nothing else, which sounds safe and is the problem: it also
    made it invisible, and it meant a run could mix cached and freshly-parsed trials with no
    record of which was which. The manifest's job is to say what code version produced the
    data a result rests on; a fallback means the result may not rest on it at all.
    """


class StaleJointAngles(StaleArtifact):
    """Joint angles on disk were produced by different code, a different tuning, or a
    different build of the trial underneath them."""


class StaleStatistics(StaleArtifact):
    """A per-trial statistics table was computed from joint angles this run would not accept,
    or under a configuration this run did not ask for."""


class StaleArtifactWarning(UserWarning):
    """A stale artifact was served anyway, because the run opted out. See MAJIC_ALLOW_STALE."""


class MissingJointAnglesWarning(UserWarning):
    """A pooled table is short an arm because its joint angles were never written.

    A warning rather than a refusal: a partial run is a legitimate state, and which arms a
    caller requires is the caller's business. It is loud because the silent version of this
    produced summary figures whose panels were drawn from different method sets.
    """


ALLOW_STALE_ENV = 'MAJIC_ALLOW_STALE'


def stale_artifacts_allowed() -> bool:
    """True when this run has explicitly opted out of the freshness checks.

    An ENVIRONMENT VARIABLE rather than a module-level flag because the checks run inside
    worker PROCESSES. `run_tracked_grid` starts them with the default start method, which is
    spawn on macOS, so a global set in the parent never arrives in the child -- the opt-out
    has to survive that boundary or it silently applies on one side of it only.

    Deliberately awkward to reach. There is no default-on fallback anywhere in this pipeline
    (see StaleTrialCache), and this exists for the one legitimate case: looking at what an old
    run produced, on purpose, knowing the summary will not describe current code.
    """
    return os.environ.get(ALLOW_STALE_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on')


def load_trial(subject: str, trial: str, strict: bool = False,
               dataset: str = TRIAL_DATASET) -> Dict[str, PlateTrial]:
    """One trial's PlateTrials, read from its parquet.

    The parquet is the interface, not an optimisation: this never parses from source. Build
    it first with `python -m experiments.build_trials --dataset <name>`.

    Raises StaleTrialCache if the artifact is missing or no longer matches its inputs.
    """
    status, reason, manifest = _cached_trial_state(subject, trial, dataset)
    if status != 'fresh':
        detail = f" ({reason})" if reason else ""
        raise StaleTrialCache(
            f"{dataset}/{subject}/{trial}: trial cache is {status}{detail}. "
            f"Run: python -m experiments.build_trials --dataset {dataset}")
    path = paths.cached_trial_path(dataset, subject, trial)

    # Alignment quality reaches the caller, not just the build log. `strict` turns it into a
    # refusal for callers that would rather not compute a joint angle at all than compute one
    # from a plate whose mocap and IMU disagree by twice the worst normal amount.
    suspect = ((manifest or {}).get('diagnostics') or {}).get('suspect') or []
    if suspect:
        message = (f"{dataset}/{subject}/{trial}: {len(suspect)} plate(s) have an alignment "
                   f"residual over {RESIDUAL_WARN_FRACTION:.0%} of their own gyro signal and "
                   f"may have a bad sync, a bad sensor-to-segment rotation, or corrupt "
                   f"markers underneath: "
                   f"{', '.join(suspect)}")
        if strict:
            raise SuspectTrial(message)
        warnings.warn(message, SuspectTrialWarning, stacklevel=2)

    return trial_io.plates_from_frame(pd.read_parquet(path, engine='pyarrow'))


def load_raw_data(subject: str, activity: str) -> Dict[str, PlateTrial]:
    """Deprecated name for `load_trial`, kept so existing experiments keep working.

    Misleading now: it does not load raw data, it reads a built trial.
    """
    return load_trial(subject, activity)


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

def _compute_expected_mag_field(plate_trials: List[PlateTrial],
                                spec: TrackingSpec = None) -> np.ndarray:
    """Median world-frame magnetic field read by the dataset's reference sensor.

    Valid frames only, for the same reason as measure_world_frame_gravity: rotating a real
    magnetometer reading by a padded or corrupt pose puts it somewhere it never was.

    The reference is `spec.mag_reference` — the torso on Al Borno, the pelvis on IMoVE — matched
    on the plate name EXACTLY. It was a substring test against 'torso' while there was one
    dataset, which is the same set of plates there and no longer a well-defined question once a
    dataset carries three sensors per segment.

    Returns the ZERO VECTOR on a dataset with no magnetometer, which is the one place in this
    module where zeros are the right answer rather than a silent failure: the reader has already
    filled every magnetometer channel with zeros, so this makes the EKF's virtual ground plate
    agree with the sensors it is being compared against. A nonzero reference against zero
    readings would be a constant residual the filter would chase. `TrackingSpec.check_method`
    is what stops a method NAME from claiming this is a magnetometer measurement.
    """
    spec = ALBORNO_TRACKING if spec is None else spec
    if not spec.has_magnetometer:
        return np.zeros(3)
    if spec.mag_reference is None:
        raise ValueError(
            f"Dataset '{spec.dataset}' has a magnetometer but no mag_reference sensor, so there "
            f"is nothing to take the world field from. Set clean_sensor on its DatasetSpec.")
    all_global_mags = []
    for plate in plate_trials:
        if plate.name != spec.mag_reference:
            continue
        valid = np.asarray(plate.valid)
        if not valid.any():
            continue
        all_global_mags.append(
            (plate.world_trace.rotations[valid] @ plate.imu_trace.mag[valid][..., None])[..., 0])
    if not all_global_mags:
        raise ValueError(
            f"No '{spec.mag_reference}' plate has a valid frame; cannot estimate the field.")
    return np.median(np.concatenate(all_global_mags, axis=0), axis=0)


def _setup_ekf_ground_plate_(plate_trials: List[PlateTrial],
                             spec: TrackingSpec = None) -> PlateTrial:
    """Precomputes global expected gravity and magnetic field to build a virtual parent ground plate.

    Both references come from `spec`: gravity in this dataset's world frame (Y-up on the
    marker-referenced datasets, Z-up on the biplane half) and the field its reference sensor
    reads. This is where a wrong world frame enters the EKF, and it enters silently."""
    spec = ALBORNO_TRACKING if spec is None else spec
    base_plate = plate_trials[0]
    expected_mag = _compute_expected_mag_field(plate_trials, spec=spec)

    ground_plate = base_plate.copy()
    ground_plate.name = "ground"
    ground_plate.world_trace.rotations = np.tile(np.eye(3), (len(base_plate), 1, 1))
    ground_plate.imu_trace.gyro = np.zeros_like(base_plate.imu_trace.gyro)
    ground_plate.imu_trace.acc = np.tile(spec.gravity, (len(base_plate), 1))
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
    experiments/global_assumptions.py), plus that sensor's own calibration gain error and
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
    experiments/global_assumptions.py) would otherwise each carry their own copy of these
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
                         heading_only_mag: bool = False,
                         heading_only_pure_mag: bool = False,
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

    heading_only_pure_mag additionally re-attributes that single row to the joint-centre
    acceleration direction itself (see RelativeFilter.heading_only_pure); it needs
    heading_only_mag.

    heading_only_mag restricts the magnetometer to the one relative DOF the two
    accelerometers cannot see -- rotation about the joint-centre acceleration -- instead of
    letting it vote on all three. See RelativeFilter.heading_only_sensor. It is a geometric
    restriction, not a temporal one, so it composes with any mag_mode: 'off' zeroes the
    sensor and the projection is then a no-op on an already-dead sensor.

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
            heading_only_sensor=1 if heading_only_mag else None,
            heading_only_pure=heading_only_pure_mag,
            **init_std_kwargs)
    else:
        joint_filter = RelativeFilter(
            gyro_std_parent=gyro_std_p, gyro_std_child=gyro_std_c,
            vector_sensor_stds_parent=sensor_stds_p,
            vector_sensor_stds_child=sensor_stds_c,
            normalize_measurements=normalize_measurements,
            heading_only_sensor=1 if heading_only_mag else None,
            heading_only_pure=heading_only_pure_mag,
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


def _joint_angles_from_marker(plates: Dict[str, PlateTrial],
                              spec: TrackingSpec = None) -> pd.DataFrame:
    spec = ALBORNO_TRACKING if spec is None else spec
    all_joint_data = []
    any_plate = next(iter(plates.values()))
    timestamps = any_plate.imu_trace.timestamps

    for joint_name, (parent, child) in spec.joints.items():
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
                               stds: Optional[Dict[str, float]] = None,
                               spec: TrackingSpec = None) -> pd.DataFrame:
    spec = ALBORNO_TRACKING if spec is None else spec
    all_joint_data = []
    any_plate = next(iter(plates.values()))
    timestamps = any_plate.imu_trace.timestamps
    expected_mag = (_compute_expected_mag_field(list(plates.values()), spec=spec)
                    if _needs_expected_mag(mag_source) else None)
    tuning = resolve_stds(stds, dataset=spec.dataset)

    for joint_name, (parent, child) in spec.joints.items():
        if parent not in plates or child not in plates:
            if os.environ.get("DISABLE_TQDM") != "True":
                print(f"Skipping joint {joint_name}: parent '{parent}' or child '{child}' not in loaded plates.")
            continue
        parent_plate = plates[parent]
        child_plate = plates[child]

        acc_override_parent = acc_override_child = None
        if acc_source == 'perfect':
            acc_override_parent, acc_override_child = _compute_perfect_joint_acc(
                parent_plate, child_plate, gravity=spec.gravity)
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
                            stds: Optional[Dict[str, float]] = None,
                            spec: TrackingSpec = None) -> pd.DataFrame:
    spec = ALBORNO_TRACKING if spec is None else spec
    plate_trials = list(plates.values())
    ground_plate = _setup_ekf_ground_plate_(plate_trials, spec=spec)
    expected_mag = (_compute_expected_mag_field(plate_trials, spec=spec)
                    if _needs_expected_mag(mag_source) else None)
    tuning = resolve_stds(stds, dataset=spec.dataset)

    segment_orientations = {}
    for plate_name, plate in plates.items():
        acc_override = (_compute_perfect_segment_acc(plate, gravity=spec.gravity)
                        if acc_source == 'perfect' else None)
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

    for joint_name, (parent, child) in spec.joints.items():
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
    """The span a filter is run over: t = 0 to the END OF THE INERTIAL RECORD.

    RUN EVERYWHERE, SCORE WHERE THERE IS GROUND TRUTH. These are two different questions and
    this function answers only the first. Which frames are scoreable is carried per sample by
    the `valid` column and applied once, in `compute_error_stats`; the filter itself has no
    business knowing where the cameras were looking.

    The start is t = 0, which is the first overlap sample and therefore the exact instant the
    pre-cache code re-zeroed to. Filters see an identical first sample, so the burn-in transient
    that dominates pooled error is unchanged. The record before it — 430 s of standing on
    Subject01's walking trial — is still excluded, because a filter seeded from mocap at t = 0
    cannot be initialised earlier than the pose that seeds it.

    The end WAS the last frame valid on any plate, on the reasoning that a filter run past it
    pays runtime to produce output nobody can evaluate. That is true of the output and false of
    the state: drift is cumulative and causal, so how far an estimate has walked by the last
    mocap frame is a fact about the whole record leading up to it, and a run that stops at the
    last scoreable sample cannot be extended afterwards without re-running from t = 0. The
    biplane half makes the asymmetry stark — 240 s of BioStamp recording against 0.48 s of
    fluoroscopy — and IMoVE's long walks put their mocap takes in the middle of a 2.6 h record.

    Nothing that is scored moves as a result, and that is checkable rather than hoped for: the
    added frames are invalid by construction, `compute_error_stats` drops them, the joint-centre
    fit and the world-field reference are both already valid-only
    (`WorldTrace.get_joint_center`, `_compute_expected_mag_field`), and the filters are causal so
    a later sample cannot reach an earlier estimate. What it costs is runtime and disk, both
    roughly in proportion to how much unscored record a dataset carries.
    """
    any_plate = next(iter(plates.values()))
    timestamps = any_plate.imu_trace.timestamps
    return slice(int(np.searchsorted(timestamps, 0.0)), len(timestamps))


def compute_joint_angles(plates: Dict[str, PlateTrial], method: str,
                          stds: Optional[Dict[str, float]] = None,
                          tracking: TrackingSpec = None) -> pd.DataFrame:
    """Joint angles for one method. `stds` overrides the DEFAULT_*_STD tuning per
    sensor (see resolve_stds); it is inert for 'marker', which runs no filter.

    `tracking` is the dataset's TrackingSpec: its joint table, world-frame gravity and
    magnetometer reference. It defaults to Al Borno's, which is what every caller predating the
    second dataset gets and is bit-identical to the constants this used to read directly.

    Every method is evaluated over the same `run_window`, so their outputs share a timestamp
    axis and `compute_error_stats` can merge them frame for frame.

    The plates are NARROWED to the spec's sensors first. That is not just a saving on the two
    datasets that carry sensors outside the joint table — on the biplane half it is correctness,
    since every trial carries each IMU twice, once against each ground-truth reference, and the
    EKF path would otherwise estimate orientations for both and the marker path would emit
    joints from whichever the table did not name.
    """
    tracking = ALBORNO_TRACKING if tracking is None else tracking
    tracking.check_method(method)
    plates = tracking.narrow_plates(plates)
    if not plates:
        raise ValueError(
            f"None of dataset '{tracking.dataset}''s sensors {tracking.sensors} are in this "
            f"trial. Either the trial is from another dataset or its build produced no plates.")
    window = run_window(plates)
    plates = {name: plate[window] for name, plate in plates.items()}
    spec = resolve_method_spec(method)
    if spec['kind'] == 'marker':
        return _joint_angles_from_marker(plates, spec=tracking)
    if spec['kind'] == 'ekf':
        return _joint_angles_from_ekf(plates, acc_source=spec['acc_source'], mag_source=spec['mag_source'],
                                      normalize_measurements=spec['normalize_measurements'],
                                      rescale_stds=spec['rescale_stds'],
                                      mag_distortion_scale=spec.get('mag_distortion_scale'),
                                      stds=stds, spec=tracking)
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
        stds=stds,
        spec=tracking,
    )

# ==============================================================================
# Joint-angle intermediate save/load
# ==============================================================================

joint_angles_path = paths.joint_angles_path


def save_joint_angles(df: pd.DataFrame, dataset: str, subject: str, trial: str, method: str,
                      *, experiment: str,
                      variant: Optional[str] = None, stds: Optional[Dict[str, float]] = None,
                      tracking: Optional[TrackingSpec] = None):
    """`variant` and `stds` travel together: the first namespaces the output so a
    re-tuned run does not overwrite the default-tuned one, the second is what the
    manifest records as the tuning in force (see paths.joint_angles_path).

    `experiment` says WHOSE TREE this parquet belongs in, and is keyword-only and required.
    `results/joint_angles/` is the benchmark's; every other experiment writes under
    `results/experiments/<experiment>/joint_angles/`. See `paths.joint_angles_write_path` for
    why one shared tree with method-keyed filenames kept turning into two estimators under one
    name. Required rather than defaulted because the script that forgets to say who it is is
    exactly the one that would overwrite the benchmark, silently.

    `source` points at the CACHED TRIAL, not at the raw folder, because that parquet is what was
    read -- `load_trial` refuses to parse from source -- and it carries its own manifest naming
    the code and inputs it was built from. Pointing at the raw directory named a file this run
    never opened and skipped the one link that makes the provenance chain complete.

    The fields `joint_angles_cache_key` names are what `joint_angles_status` reads back, so
    everything the reader needs to decide whether this file still describes current code is
    written here, in one place, from one definition.
    """
    tracking = ALBORNO_TRACKING if tracking is None else tracking
    path = ensure_parent(paths.joint_angles_write_path(experiment, dataset, subject, trial,
                                                       method, variant=variant))
    df.to_parquet(path, engine='pyarrow')
    key = joint_angles_cache_key(dataset, subject, trial, method, stds=stds, tracking=tracking)
    # `constants` is not passed separately: it is one of the fields `joint_angles_cache_key`
    # defines and arrives through **key, which is the point of having one definition -- the
    # tuning recorded and the tuning checked cannot disagree if only one line produces both.
    write_manifest(
        path,
        dataset=dataset, subject=subject, trial=trial,
        subject_label=tracking.label_subject(subject), trial_type=trial_task(dataset, trial),
        method=method, variant=variant, experiment=experiment,
        source=_repo_relative(paths.cached_trial_path(tracking.build_dataset, subject, trial)),
        n_rows=len(df),
        artifact_key=artifact_key(key),
        **key,
    )


# ==============================================================================
# Joint-angle freshness
# ==============================================================================
# THE TRIAL CACHE'S RULE, ONE LAYER DOWN. `load_trial` has refused a stale parquet for as long
# as there has been one to refuse; the joint angles COMPUTED from it had no such check, so
# every experiment that reads them -- and `--stats-only`, which reads them and nothing else --
# was free to build a summary out of angles produced by code or a tuning that no longer exists.
#
# That is not a hypothetical. Al Borno's cached angles carried acc_std=0.018 / mag_std=0.05
# while the summary manifest claimed 0.09695 / 0.009695 -- a 28x difference in magnetometer
# trust -- and the "mag_on and mag_off agree at every joint" that came out of it was pure
# artifact. The sidecars had recorded the constants the whole time. Nothing compared them.
#
# Two scripts grew their own partial version of this check afterwards
# (`benchmark_experiment.filter_configuration`, `threshold_sensitivity.constants_disagreements`),
# each covering the three stds, each in one experiment. This is that check moved to where every
# reader necessarily passes through it, and widened from the tuning to the whole identity of the
# artifact.

# Modules that implement the ESTIMATORS. Hashed semantically (see `_semantic_source`), so a
# comment does not invalidate 7000 parquets but a gain, a state layout or an update rule does.
#
# The trial digest cannot cover these and the joint-angle layer has to: `_toolchest_digest`
# hashes what turns files on disk into a PlateTrial, which is upstream of every filter and
# blind to all of them. Editing the relative filter's update step changed every joint angle in
# the tree and moved no digest at all.
#
# WHAT THIS DOES NOT COVER, stated rather than discovered: the glue in THIS module --
# `compute_joint_angles`, `_run_relative_filter`, `_joint_angles_from_*`, `run_window`. Hashing
# experiment_utils.py would be fail-closed and is what the trial layer does for the toolchest,
# but this file is 2600 lines that every experiment shares, and invalidating the entire
# joint-angle tree on an edit to an unrelated plotting helper is the kind of check that gets
# switched off. The method spec, the constants and the estimator modules are hashed; a change
# to how those are assembled needs a manual regeneration.
_ESTIMATOR_MODULES = ('RelativeFilterPlus.py', 'relative_filter_fast.py')


@lru_cache(maxsize=1)
def _estimator_digest() -> str:
    """SHA-256 over the code that turns a PlateTrial pair into a relative orientation."""
    digest = hashlib.sha256()
    src = Path(__file__).resolve().parent.parent / 'src'
    for name in sorted(_ESTIMATOR_MODULES):
        digest.update(name.encode())
        digest.update(_semantic_source(src / name))
    return digest.hexdigest()[:16]


def _repo_relative(path: Path) -> str:
    """A path for a manifest: repo-relative when it can be, absolute when it cannot.

    `relative_to` RAISES for a path outside the repo, which turns a provenance field into a
    failed write whenever the results tree is pointed elsewhere -- a test fixture, a scratch
    run. Same fallback `_source_inventory` already makes for inputs it cannot reach from a
    trial folder.
    """
    try:
        return str(Path(path).relative_to(paths.REPO_ROOT))
    except ValueError:
        return str(path)


def artifact_key(fields: Dict[str, Any]) -> str:
    """A short stable identity for a dict of provenance fields.

    Used where a DOWNSTREAM artifact needs to record which version of an upstream one it read
    without copying the whole key into its own sidecar -- the per-trial statistics do this for
    the joint angles they pool. Field-by-field comparison is still what produces the error
    messages; this is only for "is it the same file I read last time".
    """
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:16]


@lru_cache(maxsize=None)
def _read_side_trial_state(dataset: str, subject: str, trial: str
                           ) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """`_cached_trial_state`, memoized for the READ side only.

    A benchmark run asks this once per arm and again per arm at the statistics layer -- five
    methods over 261 IMoVE trials is 2610 calls, each of which globs and stats every source file
    of a trial. Memoizing is sound HERE because `data/` is declared read-only for the life of a
    process and the read side never writes a trial. `build_trials` deliberately calls the
    uncached function: it checks status, writes, and checks again.
    """
    return _cached_trial_state(subject, trial, dataset)


def _source_trial_key(dataset: str, subject: str, trial: str) -> Optional[str]:
    """The identity of the cached trial a joint angle was computed from, or None if it has no
    manifest to be identified by.

    This is what makes a REBUILT trial invalidate the angles built from the previous build. The
    trial's own status cannot: a rebuild leaves it fresh again, under a new content, while the
    angles beside it still describe the old plates.
    """
    manifest = read_manifest(paths.cached_trial_path(dataset, subject, trial)) or {}
    key = manifest.get('cache_key')
    return artifact_key(key) if key is not None else None


def joint_angles_cache_key(dataset: str, subject: str, trial: str, method: str, *,
                           stds: Optional[Dict[str, float]] = None,
                           tracking: Optional[TrackingSpec] = None) -> Dict[str, Any]:
    """Everything that determines a joint-angle parquet's contents.

    Written into the manifest by `save_joint_angles` and compared field-by-field by
    `joint_angles_status`, so the two cannot drift apart: a field added here is recorded by the
    next run and checked by every read after it.

    `method_spec` rather than the method NAME: the name is already in the path, and the grammar
    behind it is what decides which estimator ran. Changing what 'mag_adapt' means in `METHODS`
    leaves every filename identical and every parquet wrong.
    """
    tracking = ALBORNO_TRACKING if tracking is None else tracking
    return {
        'source_key': _source_trial_key(tracking.build_dataset, subject, trial),
        'estimator_digest': _estimator_digest(),
        'method_spec': resolve_method_spec(method),
        'constants': pipeline_constants(stds, dataset=tracking.dataset),
        'joints': sorted(tracking.joints),
        'world_frame_gravity': tracking.gravity.tolist(),
    }


def _values_match(got: Any, want: Any, rtol: float = 1e-9) -> bool:
    """Manifest field against its current value, tolerant of a JSON float round trip.

    Structural for dicts and lists so a spec that gained a key is a mismatch rather than a
    silent pass, and numeric for floats because a tuning read back out of JSON is not always
    bit-identical to the one recomputed from a ladder rung.
    """
    if isinstance(want, dict):
        return (isinstance(got, dict) and set(got) == set(want)
                and all(_values_match(got[key], want[key], rtol) for key in want))
    if isinstance(want, (list, tuple)):
        return (isinstance(got, (list, tuple)) and len(got) == len(want)
                and all(_values_match(g, w, rtol) for g, w in zip(got, want)))
    if isinstance(want, bool) or isinstance(got, bool):
        return got is want
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return abs(float(got) - float(want)) <= rtol * max(1.0, abs(float(want)))
    return got == want


def joint_angles_status(dataset: str, subject: str, trial: str, method: str,
                        variant: Optional[str] = None, experiment: Optional[str] = None,
                        stds: Optional[Dict[str, float]] = None,
                        tracking: Optional[TrackingSpec] = None) -> Tuple[str, Optional[str]]:
    """(status, reason) for one joint-angle artifact, without loading it.

    status is one of:
      'fresh'   -- describes the code, tuning and trial this run is configured for
      'missing' -- never written; a partial run is a legitimate state
      'stale'   -- on disk, but not that; reason names what differs

    Ordered so the reason points at the FIRST thing worth fixing. The trial underneath is
    checked before anything about the angles themselves, because a stale trial makes every arm
    over it stale and rebuilding it is the one action that clears them all.
    """
    tracking = ALBORNO_TRACKING if tracking is None else tracking
    path = joint_angles_path(dataset, subject, trial, method, variant=variant,
                             experiment=experiment)
    if not path.exists():
        return 'missing', None

    manifest = read_manifest(path)
    if manifest is None:
        return 'stale', 'no manifest sidecar, so nothing about it can be verified'

    trial_status, trial_reason, _ = _read_side_trial_state(tracking.build_dataset, subject, trial)
    if trial_status != 'fresh':
        detail = f' ({trial_reason})' if trial_reason else ''
        return 'stale', (f'the cached trial it was computed from is {trial_status}{detail} '
                         f'-- rebuild the trial first')

    # Reported as one thing rather than as whichever field the loop reaches first: an artifact
    # written before this check existed is not missing a field, it is unverifiable, and the two
    # read very differently to someone deciding whether to trust a figure.
    if manifest.get('artifact_key') is None:
        return 'stale', ('written before the joint-angle freshness check existed, so neither the '
                         'filter code nor the build of the trial behind it can be verified')

    expected = joint_angles_cache_key(dataset, subject, trial, method, stds=stds,
                                      tracking=tracking)
    for field, want in expected.items():
        got = manifest.get(field)
        if got is None and want is not None:
            return 'stale', (f'manifest predates {field}, so it cannot be checked against '
                             f'current {field} {want!r}')
        if not _values_match(got, want):
            if field == 'source_key':
                return 'stale', ('the cached trial has been rebuilt since these angles were '
                                 'written, so they describe the previous build of it')
            if field == 'estimator_digest':
                return 'stale', (f'the filter code has changed since these angles were written '
                                 f'({got} -> {want})')
            return 'stale', f'{field}: on disk {got!r} != current {want!r}'

    # Checked against the ARTIFACT rather than the inputs, exactly as at the trial layer: a
    # truncated parquet passes every comparison above unchanged. A footer read, not a load.
    claimed_rows = manifest.get('n_rows')
    if claimed_rows is None:
        return 'stale', 'manifest predates the content check'
    actual_rows = _parquet_row_count(path)
    if actual_rows != claimed_rows:
        return 'stale', (f'n_rows: file holds {actual_rows}, manifest claims {claimed_rows} '
                         f'-- the artifact is truncated')
    return 'fresh', None


def _refuse_stale(error: type, header: str, entries: Sequence[Tuple[str, Optional[str]]],
                  fix: str, allow_stale: Optional[bool] = None) -> None:
    """Raises `error` naming every stale artifact, or warns loudly if the run opted out.

    Every one of them, not the first: a run whose tuning moved has every arm of every trial
    stale, and an error that names one of 7000 invites fixing them one at a time.
    """
    shown = entries[:10]
    lines = [f'  {label}: {reason}' for label, reason in shown]
    if len(entries) > len(shown):
        lines.append(f'  ... and {len(entries) - len(shown)} more')
    message = '\n'.join([f'{header} ({len(entries)} artifact(s))', *lines, f'Fix: {fix}'])
    # An explicit `allow_stale` outranks the environment in both directions, so a script that
    # must not run on stale data cannot be talked into it by an exported variable.
    allowed = stale_artifacts_allowed() if allow_stale is None else allow_stale
    if allowed:
        why = 'the caller passed allow_stale' if allow_stale else f'{ALLOW_STALE_ENV} is set'
        warnings.warn(f'{why}, so these are being used ANYWAY. Any number computed from them '
                      f'describes code that is no longer here.\n{message}',
                      StaleArtifactWarning, stacklevel=3)
        print(f'\n*** STALE ARTIFACTS IN USE ***\n{message}\n', file=sys.stderr, flush=True)
        return
    raise error(f'{message}\nOr set {ALLOW_STALE_ENV}=1 to use them anyway, knowing the '
                f'results will not describe current code.')


def _regenerate_hint(dataset: str, experiment: Optional[str], variant: Optional[str]) -> str:
    """The command that rewrites a joint-angle tree, as best it can be named from here.

    The benchmark owns the canonical tree and has a stable command line; every other
    experiment's is its own module, which is a good enough pointer to the script whose name is
    in the path.
    """
    if experiment in (None, paths.BENCHMARK_EXPERIMENT):
        base = f'python -m experiments.benchmark_experiment --dataset {dataset}'
    else:
        base = f'python -m experiments.{experiment} --dataset {dataset}'
    return base + (f'   (variant {variant!r})' if variant else '')


def load_joint_angles(dataset: str, subject: str, trial: str, method: str,
                      variant: Optional[str] = None,
                      experiment: Optional[str] = None,
                      stds: Optional[Dict[str, float]] = None,
                      tracking: Optional[TrackingSpec] = None,
                      allow_stale: Optional[bool] = None) -> Optional[pd.DataFrame]:
    """One method's joint angles, or None if they were never written.

    `experiment` selects WHICH tree to read: None is the benchmark's canonical one, a name is
    that experiment's own. Reading is open -- an experiment comparing itself against the
    benchmark's arms should read them -- so unlike the write path this defaults, and defaults to
    the tree the benchmark owns.

    RAISES StaleJointAngles if what is on disk was produced by different code, a different
    tuning, or a different build of the trial. `stds` and `tracking` are what it is checked
    against, and they must be the ones the run intends: a caller that passes `variant` because
    it wrote under a re-tuning and then omits the matching `stds` is asking whether a re-tuned
    parquet matches the default tuning, and the answer is no.

    Missing is still None. A partial run is a legitimate state and the callers report it; an
    artifact that contradicts its own manifest is not, and no caller was checking.
    """
    status, reason = joint_angles_status(dataset, subject, trial, method, variant=variant,
                                        experiment=experiment, stds=stds, tracking=tracking)
    if status == 'missing':
        return None
    label = f'{dataset}/{subject}/{trial}/{method}' + (f' [{variant}]' if variant else '')
    if status != 'fresh':
        _refuse_stale(StaleJointAngles, 'Joint angles are stale', [(label, reason)],
                      _regenerate_hint(dataset, experiment, variant), allow_stale)
    path = joint_angles_path(dataset, subject, trial, method, variant=variant,
                             experiment=experiment)
    return pd.read_parquet(path, engine='pyarrow')


# ==============================================================================
# Trial naming
# ==============================================================================

def trial_task(dataset: str, trial: str) -> str:
    """The ACTIVITY a trial name denotes: 'walking', 'lat_step', 'SDrop'.

    Al Borno's trial names already are the activity, which is why `trial_type` and the trial key
    were one column for as long as there was one dataset. The others encode it: IMoVE numbers
    and suffixes its trials ('t4_lat_step_001'), and the biplane half prefixes a side and
    suffixes a repeat ('LSDrop1' -> 'SDrop', via `biplane.trial_task`, which is also what the
    build layer excludes static holds on).

    This is what the statistics tables carry as `trial_type`, so pooling by activity works
    across datasets, while `trial` keeps the verbatim key. On Al Borno the two are equal and
    every existing figure that groups on `trial_type` is unaffected.

    Falls back to the verbatim trial name for an unregistered dataset or a name that does not
    match the expected shape. That is honest — an unparsed name pools with nothing but itself —
    and it is why this returns rather than raising.
    """
    if dataset == 'alborno':
        return trial
    if dataset.startswith('imove_biplane'):
        return biplane.trial_task(trial.split('/')[-1]) or trial
    if dataset == 'imove':
        # 't4_lat_step_001' -> 'lat_step': strip the leading task index and the trailing take
        # number. Both are positional rather than semantic, and the middle is what two sessions
        # have in common.
        match = re.match(r'^t\d+_(?P<task>.+?)(?:_\d+)?$', trial)
        return match.group('task') if match else trial
    return trial


LABEL_COLUMNS = ('joint_name', 'subject', 'trial', 'trial_type', 'method', 'dataset')

# The three ANATOMICAL axes an error can be split along, and the columns a basis is stored in.
# Defined here rather than in experiments/anatomical_frames.py, which measures the bases: that
# module imports this one, so the vocabulary has to live on this side of the dependency.
# Flexion/Extension, Adduction/Abduction, Internal/External rotation, of the PARENT segment.
ANATOMICAL_AXES = ('FE', 'AA', 'IE')

# a<row><col> of a 3x3 basis whose COLUMNS are ANATOMICAL_AXES expressed in the parent sensor's
# own frame, flattened row-major.
BASIS_COLUMNS = tuple(f'a{row}{col}' for row in range(1, 4) for col in range(1, 4))


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


def load_all_joint_angles(dataset: str, row_keys: Sequence[Tuple[str, str]],
                          methods: List[str], variant: Optional[str] = None,
                          tracking: Optional[TrackingSpec] = None,
                          experiment: Optional[str] = None,
                          stds: Optional[Dict[str, float]] = None,
                          allow_stale: Optional[bool] = None) -> pd.DataFrame:
    """Every (trial, method) joint-angle table concatenated, with its labels attached.

    `row_keys` is a list of (subject, trial) pairs — what was actually run — rather than the
    subjects x activities cross product it used to be. The cross product claimed trials the
    datasets do not have (three Al Borno subjects have no complexTasks) and cannot express the
    others at all: IMoVE's trial names differ per session, and a biplane key carries its session
    and block ('Test1/A/RSDrop1').

    LARGE. ~1 GB for Al Borno and roughly ten times that for IMoVE, which is why the benchmark
    builds its summary from the per-trial statistics instead and calls this only on request.

    RAISES StaleJointAngles, once, naming every arm that does not match this run -- see
    `joint_angles_status`. `stds` must be the tuning the arms were generated under, and travels
    with `variant` for the same reason it does on the write side.
    """
    tracking = ALBORNO_TRACKING if tracking is None else tracking

    # STATUS FIRST, ACROSS THE WHOLE GRID, then load. Two passes rather than one because the
    # answer to "is any of this stale" should not cost the gigabyte of reading that comes after
    # it, and because a run whose tuning moved has EVERY arm stale -- reporting the first one
    # and dying invites fixing them one at a time.
    fresh, stale, missing = [], [], []
    for subject, trial in row_keys:
        for method in methods:
            label = f'{dataset}/{subject}/{trial}/{method}' + (f' [{variant}]' if variant else '')
            status, reason = joint_angles_status(dataset, subject, trial, method, variant=variant,
                                                experiment=experiment, stds=stds,
                                                tracking=tracking)
            if status == 'fresh':
                fresh.append((subject, trial, method))
            elif status == 'missing':
                missing.append(label)
            else:
                stale.append((label, reason))
    if stale:
        _refuse_stale(StaleJointAngles, 'Joint angles are stale', stale,
                      _regenerate_hint(dataset, experiment, variant), allow_stale)

    # Missing is reported through `warnings`, not a bare print gated on DISABLE_TQDM. That gate
    # was set to "True" at import by this very module, so the message it guarded could only ever
    # appear in a process that had overridden it afterwards -- which is to say a pooled summary
    # could silently be short an entire arm.
    if missing:
        shown = ', '.join(missing[:10]) + (f' ... (+{len(missing) - 10})' if len(missing) > 10 else '')
        warnings.warn(f'{len(missing)} joint-angle artifact(s) were never written and are absent '
                      f'from this table: {shown}', MissingJointAnglesWarning, stacklevel=2)

    frames = []
    for subject, trial, method in fresh:
        df = pd.read_parquet(joint_angles_path(dataset, subject, trial, method, variant=variant,
                                               experiment=experiment), engine='pyarrow')
        frames.append(df.assign(subject=tracking.label_subject(subject), trial=trial,
                                trial_type=trial_task(dataset, trial), method=method,
                                dataset=dataset))
    if not frames:
        return pd.DataFrame()
    return as_categorical_labels(pd.concat(frames, ignore_index=True))

# ==============================================================================
# Statistics
# ==============================================================================

def _project_onto_basis(merged_df: pd.DataFrame, rotvec_error: np.ndarray,
                        basis: pd.DataFrame) -> np.ndarray:
    """(N, 3) error components along each row's anatomical axes; NaN where there is no basis.

    Grouped rather than merged. A merge would attach nine float columns to a table that reaches
    tens of millions of rows on IMoVE — 700 MB of basis for 20 MB of distinct values — whereas the
    key has a handful of levels, so this walks them and writes into a preallocated array instead.

    NaN, not identity, for a joint with no basis. An identity would not fail: it would relabel the
    parent sensor's own axes as anatomical ones and put them in the same column as the real thing.
    """
    keys = [column for column in ('dataset', 'subject', 'trial', 'joint_name')
            if column in basis.columns and column in merged_df.columns]
    if not keys:
        raise ValueError(
            f"`basis` shares no key column with the joint angles; it carries "
            f"{sorted(set(basis.columns) - set(BASIS_COLUMNS))} and needs at least 'joint_name'.")
    missing = [column for column in BASIS_COLUMNS if column not in basis.columns]
    if missing:
        raise ValueError(f"`basis` is missing basis columns {missing}; expected all of "
                         f"{list(BASIS_COLUMNS)}.")

    lookup = {(tuple(row[key] for key in keys) if len(keys) > 1 else row[keys[0]]):
              np.asarray([row[column] for column in BASIS_COLUMNS], dtype=float).reshape(3, 3)
              for _, row in basis.iterrows()}
    out = np.full_like(rotvec_error, np.nan)
    # `.indices` gives POSITIONS, not index labels, which is what makes this safe: `merged_df`
    # arrives boolean-filtered by `valid_marker`, so its labels have gaps while `rotvec_error` is
    # dense. Indexing the dense array by labels would silently pair each row's error with another
    # row's basis (or raise, once a label passed the end).
    for key, positions in merged_df.groupby(keys, observed=True).indices.items():
        matrix = lookup.get(key)
        if matrix is not None:
            # rows of rotvec_error are error vectors, so `e @ A` is A^T e per row: the error's
            # components along the columns of A, which are the anatomical axes.
            out[positions] = rotvec_error[positions] @ matrix
    return out


def compute_error_stats(df: pd.DataFrame, basis: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Calculates summary statistics using fast wide-format C-aggregations.

    One row per (dataset, subject, trial, trial_type, method, joint, axis) present in the input,
    for whichever of those label columns the input carries — `trial` and `dataset` are optional
    so a frame written before they existed still aggregates, and so a single-trial caller need
    not invent them.

    `trial` IS PART OF THE KEY when present, on both sides of the merge and in the grouping.
    Without it, two trials of one activity — IMoVE sessions hold several walking takes — would
    cross-join on (subject, trial_type, joint, timestamp) and produce every pairing of one
    trial's estimate against the other's ground truth.

    `basis` ADDS THE ANATOMICAL AXES and changes nothing else. It is the table
    `anatomical_frames.basis_frame` produces: `joint_name` plus `BASIS_COLUMNS`, optionally keyed
    by `subject` and `trial` as well, holding the rotation from the parent sensor's frame to its
    segment's anatomical frame. Given one, the same error vector is ALSO reported along
    `ANATOMICAL_AXES` — flexion, adduction, internal rotation — as extra `axis` rows beside the
    existing 'MAG', 'X', 'Y', 'Z'. Nothing is replaced: every caller that reads `axis == 'MAG'`
    is unaffected, and the sensor-frame components stay available for regression checks.

    Because a basis is orthonormal, the split is exact in the strong sense that
    RMSE_MAG^2 = RMSE_FE^2 + RMSE_AA^2 + RMSE_IE^2 in every group, so the anatomical panels of a
    figure add up to its magnitude panel. THIS IS NOT A CARDAN DECOMPOSITION and is not the
    difference of two clinical angle sequences; see the header of experiments/anatomical_frames.py.

    KEYED ON WHAT BOTH SIDES CARRY. A basis is per (subject, sensor) at heart — plates are
    re-strapped per subject and the alignment rotation is fitted per trial — so a `basis` holding
    only `joint_name` applied to a POOLED multi-subject frame would put one subject's mounting on
    every other subject's error. Pass `subject` (and `trial`) columns whenever the input has more
    than one, which is what `basis_frame` does by default.
    """
    if df.empty:
        return pd.DataFrame()

    marker_df = df[df['method'] == 'marker']
    imu_df = df[df['method'] != 'marker']

    if marker_df.empty or imu_df.empty:
        return pd.DataFrame()

    optional = [col for col in ('dataset', 'trial') if col in df.columns]
    join_cols = ['subject', 'trial_type', 'joint_name', 'timestamp'] + optional
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
    group_cols = ['trial_type', 'method', 'joint_name', 'subject'] + optional
    target_cols = ['MAG', 'X', 'Y', 'Z']

    if basis is not None and not basis.empty:
        anatomical = _project_onto_basis(merged_df, rotvec_error, basis)
        for index, axis in enumerate(ANATOMICAL_AXES):
            merged_df[axis] = anatomical[:, index]
        target_cols = target_cols + list(ANATOMICAL_AXES)

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
    # An anatomical axis is NaN for every metric where that joint had no basis (see
    # _project_onto_basis). Those rows say nothing and would be counted by anything that reports
    # how many rows an axis has, so they go rather than being carried as blanks.
    metric_columns = list(metric_map)
    all_nan = summary_df[metric_columns].isna().all(axis=1)
    if all_nan.any():
        summary_df = summary_df[~all_nan].reset_index(drop=True)
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
                    dataset: str = TRIAL_DATASET, **manifest_extra: Any) -> Path:
    """Saves a summary-statistics DataFrame to results/statistics/<name>_statistics.parquet.

    `stds` records a filter re-tuning in the manifest (see pipeline_constants), and `dataset`
    selects which row of DATASET_STDS the un-overridden tuning came from. It is an explicit
    parameter rather than one more key in `manifest_extra` because it now CHANGES the
    constants recorded, not just the label on them — a caller that passed it as extra
    metadata (benchmark_experiment always has) would otherwise have written IMoVE results
    with Al Borno's stds in the sidecar."""
    path = ensure_parent(paths.statistics_path(name))
    df.to_parquet(path, engine='pyarrow')
    write_manifest(
        path, constants=pipeline_constants(stds, dataset=dataset), experiment=name,
        dataset=dataset,
        methods=sorted(df['method'].unique().tolist()) if 'method' in df.columns else None,
        subjects=sorted(df['subject'].unique().tolist()) if 'subject' in df.columns else None,
        n_rows=len(df), estimator_digest=_estimator_digest(), **manifest_extra,
    )
    return path


def load_statistics(name: str) -> Optional[pd.DataFrame]:
    """A named summary table, or None if it was never written.

    WARNS rather than raises when the summary predates the current filter code. This is the last
    layer and it is read almost entirely by `plotting/`, whose job is to draw what a run
    produced; refusing here would mean a figure cannot be redrawn from a finished run without
    also regenerating it, and the layers below already refuse the inputs. A warning is enough to
    stop someone reading a stale number as a current one, which is the actual failure.
    """
    path = paths.statistics_path(name)
    if not path.exists():
        return None
    recorded = (read_manifest(path) or {}).get('estimator_digest')
    if recorded is not None and recorded != _estimator_digest():
        warnings.warn(
            f"'{name}' statistics were computed with filter code that has changed since "
            f"({recorded} -> {_estimator_digest()}). Every number in this table describes the "
            f"old estimator. Regenerate it before quoting or plotting it.",
            StaleArtifactWarning, stacklevel=2)
    return pd.read_parquet(path, engine='pyarrow')

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
                                  variant: Optional[str] = None,
                                  tracking: Optional[TrackingSpec] = None,
                                  experiment: Optional[str] = None) -> None:
    """stage_labels = ['load'] + method names. Loads the trial once, then computes
    and saves joint angles for every method, reusing the loaded data across methods.

    `tracking` says WHICH DATA and WHAT CAN BE TRACKED IN IT — which build tree the trial comes
    from, which namespace the results go under, and the joint table, world-frame gravity and
    magnetometer reference to run it with. Bind it with functools.partial; it defaults to Al
    Borno's, which is what every caller predating the second dataset gets. The input and output
    namespaces travel inside it rather than as separate arguments precisely because a mismatch
    between them is silent in the direction that matters: the right sensors written under the
    wrong dataset name is an overwrite, not an error.

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
    subject, trial = row_key
    tracking = ALBORNO_TRACKING if tracking is None else tracking
    methods = [stage for stage in stage_labels if stage != 'load']
    tracks_load = 'load' in stage_labels

    # Checked HERE rather than left to the save at the end of a three-minute filter run.
    # `paths.joint_angles_write_path` would raise either way, but a worker exception surfaces as
    # one red cell with the message buried in it, and the run would have burned every trial's
    # compute before reporting what is really a bad call.
    if not experiment:
        raise ValueError(
            "generate_joint_angles_worker needs experiment=<name>: it decides which joint-angle "
            "tree the parquets land in. Pass paths.BENCHMARK_EXPERIMENT for the benchmark, or "
            "this experiment's own name to write under "
            "results/experiments/<name>/joint_angles/.")

    t_start = time.time()
    if tracks_load:
        shared_state[(row_key, 'load')] = "Running"
    try:
        plates = load_trial(subject, trial, dataset=tracking.build_dataset)
        if tracks_load:
            shared_state[(row_key, 'load_time')] = time.time() - t_start
            shared_state[(row_key, 'load')] = "Success"
    except Exception as e:
        if tracks_load:
            shared_state[(row_key, 'load')] = f"Failed ({e})"
        for method in methods:
            shared_state[(row_key, method)] = f"Failed (load: {e})"
        return None

    # A trial that holds NONE of this spec's sensors is skipped, not failed. It is the same
    # distinction `build_trial_worker` draws with its 'absent' status: two of the biplane half's
    # 379 trials were built with their Vicon plates only, so the bone-pose spec has no reference
    # in them at all, and no re-run can change that. Reported as a Skip because a permanently red
    # cell with no command that clears it is indistinguishable from a bug.
    if not tracking.narrow_plates(plates):
        reason = f"none of {tracking.dataset}'s sensors are in this trial"
        for method in methods:
            shared_state[(row_key, method)] = "Skipped"
        if tracks_load:
            shared_state[(row_key, 'load')] = "Skipped"
        return {'skipped': reason}

    for method in methods:
        t_method = time.time()
        shared_state[(row_key, method)] = "Running"
        try:
            df = compute_joint_angles(plates, method, stds=stds, tracking=tracking)
            if df is not None and not df.empty:
                save_joint_angles(df, tracking.dataset, subject, trial, method,
                                  experiment=experiment, variant=variant,
                                  stds=stds, tracking=tracking)
                shared_state[(row_key, f"{method}_time")] = time.time() - t_method
                shared_state[(row_key, method)] = "Success"
            else:
                shared_state[(row_key, method)] = "Skipped"
        except Exception as e:
            shared_state[(row_key, method)] = f"Failed ({e})"
    return None


def _trial_basis(dataset: str, subject: str, trial: str,
                 tracking: TrackingSpec) -> pd.DataFrame:
    """One trial's anatomical bases as `compute_error_stats` wants them, or a raising error.

    Imported lazily because experiments/anatomical_frames.py imports THIS module — it needs
    `TrackingSpec`, `load_trial` and `build_report_path` — so the dependency can only run one way
    at module scope.
    """
    from experiments.anatomical_frames import basis_frame, bases_path
    frame = basis_frame(dataset, subject=subject, trial=trial, spec=tracking)
    if frame.empty:
        raise FileNotFoundError(
            f"No anatomical basis for {dataset}/{subject}/{trial} in {bases_path(dataset)}. "
            f"Run: python -m experiments.anatomical_frames --dataset {dataset}")
    return frame


def compute_stats_worker(row_key: Tuple[str, str], stage_labels: List[str], shared_state: Dict,
                          methods: List[str], stats_name: str,
                          variant: Optional[str] = None,
                          stds: Optional[Dict[str, float]] = None,
                          tracking: Optional[TrackingSpec] = None,
                          anatomical_axes: bool = False,
                          experiment: Optional[str] = None) -> None:
    """Single-stage worker (stage_labels should be a single name, e.g. ['stats']).
    Reads back whatever per-method parquets generate_joint_angles_worker managed to
    save for this row and computes error stats against 'marker'. Naturally
    fails/no-ops if the load or every method failed there, since no parquet files
    exist to read.

    `stats_name` namespaces the output under results/statistics/per_subject/<stats_name>/.
    It is required rather than defaulted: benchmark, oracle-ablation and
    threshold-sweep runs each produce per-subject stats over a different method
    set, and a shared default filename meant whichever ran last silently won. `dataset`
    namespaces one level in from that, for the same reason it does in the joint-angle tree.

    THIS TABLE IS THE SUMMARY, not an intermediate on the way to one. `compute_error_stats`
    groups per trial, so concatenating these across trials is exactly what pooling every trial's
    samples and aggregating once produces — which is what lets the benchmark skip materializing
    a ten-gigabyte joint-angle table it would only group back down again.

    `variant` must match the one the generation phase wrote under, and `stds` is
    recorded in the manifest — both default to the untuned pipeline's.

    `anatomical_axes` adds the FE/AA/IE breakdown, which needs
    `python -m experiments.anatomical_frames --dataset <name>` to have run. It RAISES for this
    trial rather than silently writing magnitude-only statistics if the artifact has no basis for
    it: a table missing an axis for some trials and not others pools into a figure whose panels
    are drawn from different subject sets."""
    subject, trial = row_key
    tracking = ALBORNO_TRACKING if tracking is None else tracking
    dataset = tracking.dataset
    stage = stage_labels[0]

    t_start = time.time()
    shared_state[(row_key, stage)] = "Running"
    try:
        frames, pooled = [], []
        for method in methods:
            df = load_joint_angles(dataset, subject, trial, method, variant=variant,
                                   experiment=experiment, stds=stds, tracking=tracking)
            if df is not None:
                pooled.append(method)
                frames.append(df.assign(subject=tracking.label_subject(subject), trial=trial,
                                        trial_type=trial_task(dataset, trial), method=method,
                                        dataset=dataset))

        if not frames:
            shared_state[(row_key, stage)] = "Failed"
            return None

        all_df = as_categorical_labels(pd.concat(frames, ignore_index=True))
        stats_df = compute_error_stats(all_df, basis=_trial_basis(dataset, subject, trial, tracking)
                                       if anatomical_axes else None)
        if not stats_df.empty:
            stats_path = ensure_parent(
                paths.per_subject_statistics_path(stats_name, dataset, subject, trial))
            stats_df.to_parquet(stats_path, engine='pyarrow')
            # `methods` is what was POOLED, not what was requested. Recording the request meant
            # a table built from three arms of a five-arm run claimed all five, which is the one
            # thing the reader most needs to be told and the manifest was actively hiding.
            write_manifest(stats_path, constants=pipeline_constants(stds, dataset=tracking.dataset),
                           experiment=stats_name,
                           dataset=dataset, subject=subject, trial=trial,
                           subject_label=tracking.label_subject(subject),
                           trial_type=trial_task(dataset, trial), methods=sorted(pooled),
                           requested_methods=sorted(methods),
                           variant=variant, anatomical_axes=anatomical_axes,
                           n_rows=len(stats_df),
                           angles_key=pooled_angles_key(dataset, subject, trial, pooled,
                                                        variant=variant, experiment=experiment))

        shared_state[(row_key, f"{stage}_time")] = time.time() - t_start
        shared_state[(row_key, stage)] = "Success"
        return None
    except Exception as e:
        shared_state[(row_key, stage)] = f"Failed ({e})"
        return None


def pooled_angles_key(dataset: str, subject: str, trial: str, methods: Sequence[str],
                      variant: Optional[str] = None,
                      experiment: Optional[str] = None) -> Optional[str]:
    """The identity of the joint-angle artifacts a statistics table pooled.

    Built from each arm's `artifact_key` AND the fact of its last write, so regenerating any arm
    invalidates the table that pooled it. Without this, the statistics layer's only defence was
    its recorded constants, and a change that moves the angles without moving the tuning -- an
    edit to the filter itself -- left a stale table looking perfectly current.

    `artifact_key` alone is not enough and it is worth saying why: it is the identity of a
    CONFIGURATION, so two runs of the same configuration share it, which is exactly what makes
    it useful for deciding whether an arm matches this run and useless for deciding whether the
    file changed. `written_at` and `n_rows` are what answer the second question. The cost is
    that a re-run producing byte-identical angles still invalidates the tables over them, which
    is a few hundred rows of statistics recomputed against a false negative that would put the
    previous filter's numbers in the summary.

    None if any arm has no `artifact_key` to read, which is the same "cannot be verified" the
    other layers report rather than papering over.
    """
    parts = {}
    for method in sorted(methods):
        manifest = read_manifest(joint_angles_path(dataset, subject, trial, method,
                                                   variant=variant, experiment=experiment)) or {}
        key = manifest.get('artifact_key')
        if key is None:
            return None
        parts[method] = [key, manifest.get('n_rows'), manifest.get('written_at')]
    return artifact_key(parts)


def per_trial_statistics_status(stats_name: str, dataset: str, subject: str, trial: str,
                                methods: Optional[Sequence[str]] = None,
                                variant: Optional[str] = None,
                                experiment: Optional[str] = None,
                                stds: Optional[Dict[str, float]] = None,
                                tracking: Optional[TrackingSpec] = None,
                                anatomical_axes: Optional[bool] = None
                                ) -> Tuple[str, Optional[str]]:
    """(status, reason) for one per-trial statistics table.

    The hazard this exists for is narrow and real: the aggregation phase reads whatever tables
    are on disk, and a trial whose statistics worker FAILED this run still has last run's table
    sitting there. It gets pooled into the summary, and the summary's own manifest -- stamped
    from the command line -- says it describes this run.

    `methods` is what the caller means to pool. A table that pooled a different set is stale for
    this purpose even though it is internally consistent: half the arms of a comparison is not a
    comparison. Passing None skips that check and accepts whatever the table holds.
    """
    tracking = ALBORNO_TRACKING if tracking is None else tracking
    path = paths.per_subject_statistics_path(stats_name, dataset, subject, trial)
    if not path.exists():
        return 'missing', None

    manifest = read_manifest(path)
    if manifest is None:
        return 'stale', 'no manifest sidecar, so nothing about it can be verified'

    want_constants = pipeline_constants(stds, dataset=tracking.dataset)
    if not _values_match(manifest.get('constants'), want_constants):
        return 'stale', (f"constants: table carries {manifest.get('constants')!r} != current "
                         f"{want_constants!r}")
    if (manifest.get('variant') or None) != (variant or None):
        return 'stale', (f"variant: table carries {manifest.get('variant')!r}, this run wants "
                         f"{variant!r}")
    if anatomical_axes and not manifest.get('anatomical_axes'):
        return 'stale', ('table has no anatomical-axis rows and this run wants them -- pooling it '
                         'would give a figure whose panels come from different trial sets')

    pooled = manifest.get('methods')
    if methods is not None and sorted(methods) != sorted(pooled or []):
        return 'stale', (f'methods: table pooled {sorted(pooled or [])}, this run wants '
                         f'{sorted(methods)}')

    if pooled:
        recorded = manifest.get('angles_key')
        if recorded is None:
            return 'stale', ('manifest predates angles_key, so which joint angles it pooled '
                             'cannot be verified')
        current = pooled_angles_key(dataset, subject, trial, pooled, variant=variant,
                                    experiment=experiment)
        if current != recorded:
            return 'stale', ('the joint angles it pooled have been regenerated since, so this '
                             'table describes the previous ones')

    claimed_rows = manifest.get('n_rows')
    if claimed_rows is None:
        return 'stale', 'manifest predates the content check'
    actual_rows = _parquet_row_count(path)
    if actual_rows != claimed_rows:
        return 'stale', (f'n_rows: file holds {actual_rows}, manifest claims {claimed_rows} '
                         f'-- the artifact is truncated')
    return 'fresh', None


def load_per_trial_statistics(stats_name: str, dataset: str,
                              row_keys: Sequence[Tuple[str, str]],
                              methods: Optional[Sequence[str]] = None,
                              variant: Optional[str] = None,
                              experiment: Optional[str] = None,
                              stds: Optional[Dict[str, float]] = None,
                              tracking: Optional[TrackingSpec] = None,
                              anatomical_axes: Optional[bool] = None,
                              allow_stale: Optional[bool] = None) -> pd.DataFrame:
    """Every per-trial statistics table for a run, concatenated — the dataset summary.

    Cheap: a few hundred rows per trial against the millions of samples they were computed from,
    so this is what the benchmark's global aggregation phase reads instead of reloading angles.
    Trials with no table are skipped silently, since a partial run is a legitimate state and the
    caller reports what it found.

    RAISES StaleStatistics for a table this run would not have written -- see
    `per_trial_statistics_status`. The check arguments describe what the caller is pooling FOR;
    omitting them checks less, and omitting all of them checks only that each table can account
    for itself.
    """
    frames, stale = [], []
    for subject, trial in row_keys:
        status, reason = per_trial_statistics_status(
            stats_name, dataset, subject, trial, methods=methods, variant=variant,
            experiment=experiment, stds=stds, tracking=tracking, anatomical_axes=anatomical_axes)
        if status == 'missing':
            continue
        if status != 'fresh':
            stale.append((f'{dataset}/{subject}/{trial}', reason))
            continue
        frames.append(pd.read_parquet(
            paths.per_subject_statistics_path(stats_name, dataset, subject, trial),
            engine='pyarrow'))
    if stale:
        _refuse_stale(StaleStatistics,
                      f"Per-trial statistics for '{stats_name}' are stale", stale,
                      _regenerate_hint(dataset, experiment, variant), allow_stale)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
