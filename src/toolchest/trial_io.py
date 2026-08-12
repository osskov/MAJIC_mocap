"""
Flat-table serialization for a trial's worth of PlateTrials.

This module knows the SHAPE of the cached artifact and nothing else — no paths, no
manifests, no provenance. Those live in paths.py and experiments/experiment_utils.py,
so toolchest stays free of repo layout and can be tested on its own.

Layout
------
One row per (plate, sample), one file per trial. All plates share a file because
`PlateTrial.from_traces` trims every plate in a trial to a common minimum length —
that is a trial-scoped decision, and splitting plates across files would silently
drop it on reassembly.

    plate      category   the PlateTrial name, e.g. 'femur_r_imu'
    timestamp  float64    seconds from the start of the synced trial
    acc_{xyz}  float32    accelerometer, sensor frame, m/s^2
    gyro_{xyz} float32    gyroscope, sensor frame, rad/s
    mag_{xyz}  float32    magnetometer, sensor frame, Earth-field units
    pos_{xyz}  float32    plate origin in the mocap world frame, m
    rot_{ij}   float32    world-from-body rotation matrix, row i column j (9 columns)
    valid      bool       is this frame trustworthy ground truth? (see WorldTrace.valid)

Two choices worth stating outright:

*Rotations are stored as nine matrix entries, not as a quaternion.* WorldTrace holds
(N, 3, 3) natively and every consumer uses it as a matrix, so this avoids both a
conversion on each load and the quaternion double-cover problem, where a sign flip
between adjacent samples is numerically invisible but breaks any code that
interpolates or differentiates the sequence. Nine float32s per sample is 36 bytes
against a quaternion's 16; parquet compresses the difference away.

*Timestamps are float64, everything else float32.* Not cosmetic:
`PlateTrial.__init__` asserts the IMU and world timestamps agree to within 1e-8, and
float32 has ~1e-7 relative precision, so a float32 timestamp column would fail that
assert on load for any trial longer than a second. float32 elsewhere costs ~1e-7
relative on the sensor channels and ~1e-5 degrees on the rotations, both far below
the measurement noise.

A consequence of the single `timestamp` column: the IMU and world traces come back
with bit-identical timestamps. In a live load they differ by ~7e-13, from
independently re-zeroing two float64 vectors. Collapsing them is well inside the
tolerance the class already enforces, and it makes the assert exact rather than
merely satisfied.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from .IMUTrace import IMUTrace
from .PlateTrial import PlateTrial
from .WorldTrace import WorldTrace

_AXES = ('x', 'y', 'z')

ACC_COLUMNS: List[str] = [f'acc_{a}' for a in _AXES]
GYRO_COLUMNS: List[str] = [f'gyro_{a}' for a in _AXES]
MAG_COLUMNS: List[str] = [f'mag_{a}' for a in _AXES]
POS_COLUMNS: List[str] = [f'pos_{a}' for a in _AXES]
ROT_COLUMNS: List[str] = [f'rot_{i}{j}' for i in range(3) for j in range(3)]

VALUE_COLUMNS: List[str] = ACC_COLUMNS + GYRO_COLUMNS + MAG_COLUMNS + POS_COLUMNS + ROT_COLUMNS
COLUMNS: List[str] = ['plate', 'timestamp'] + VALUE_COLUMNS + ['valid']

# Bumped whenever the column set or its meaning changes. Recorded in the manifest and
# checked on load, so an artifact written by an older schema is a cache miss rather
# than a confusing KeyError halfway through a run.
#
#   1 -> 2  added the `valid` column. A v1 artifact has no mask, and silently defaulting
#           it to all-True would claim interpolated and corrupt frames are measured
#           ground truth — exactly the assertion the column exists to stop making.
SCHEMA_VERSION = 2


def plates_to_frame(plates: Dict[str, PlateTrial]) -> pd.DataFrame:
    """Flattens a trial's PlateTrials into the long-form table described above.

    Plates are emitted in sorted name order so the artifact is byte-reproducible.
    The live loader's order comes from `Path.glob`, which is filesystem-dependent;
    nothing downstream looks plates up positionally, so imposing an order here costs
    nothing and removes a source of spurious diffs between machines.
    """
    if not plates:
        raise ValueError("Refusing to serialize an empty trial: no plates to write.")

    blocks = []
    for name in sorted(plates):
        plate = plates[name]
        imu, world = plate.imu_trace, plate.world_trace
        n = len(plate)

        block = pd.DataFrame({'plate': name, 'timestamp': np.asarray(imu.timestamps, dtype=np.float64)})
        for columns, values in ((ACC_COLUMNS, imu.acc), (GYRO_COLUMNS, imu.gyro),
                                (MAG_COLUMNS, imu.mag), (POS_COLUMNS, world.positions)):
            block[columns] = np.asarray(values, dtype=np.float32)
        block[ROT_COLUMNS] = np.asarray(world.rotations, dtype=np.float32).reshape(n, 9)
        block['valid'] = np.asarray(world.valid, dtype=bool)
        blocks.append(block)

    frame = pd.concat(blocks, ignore_index=True)
    frame['plate'] = frame['plate'].astype('category')
    return frame[COLUMNS]


def plates_from_frame(frame: pd.DataFrame) -> Dict[str, PlateTrial]:
    """Rebuilds the PlateTrials from a table written by `plates_to_frame`.

    Rotations are handed back as float32 rather than being re-orthonormalized. The
    round trip perturbs orthonormality by ~1e-7, which is six orders of magnitude
    below the reconstruction error already present in marker-derived frames, and
    re-orthonormalizing would mean an SVD per sample per plate for no measurable gain.
    """
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Cached trial is missing columns {missing}; expected schema {COLUMNS}.")

    plates: Dict[str, PlateTrial] = {}
    # observed=True: 'plate' is categorical, and without it pandas emits a group for
    # every category in the dtype, including ones filtered out of this frame.
    for name, block in frame.groupby('plate', sort=True, observed=True):
        timestamps = block['timestamp'].to_numpy(dtype=np.float64)
        imu = IMUTrace(
            timestamps=timestamps,
            gyro=block[GYRO_COLUMNS].to_numpy(dtype=np.float32),
            acc=block[ACC_COLUMNS].to_numpy(dtype=np.float32),
            mag=block[MAG_COLUMNS].to_numpy(dtype=np.float32),
        )
        world = WorldTrace(
            timestamps=timestamps,
            positions=block[POS_COLUMNS].to_numpy(dtype=np.float32),
            rotations=block[ROT_COLUMNS].to_numpy(dtype=np.float32).reshape(len(block), 3, 3),
            valid=block['valid'].to_numpy(dtype=bool),
        )
        plates[str(name)] = PlateTrial(str(name), imu, world)
    return plates
