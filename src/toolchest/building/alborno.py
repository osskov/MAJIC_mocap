"""
Reader for this repo's original dataset, adapted from Al Borno et al. (2022).

Layout it expects, one directory per subject/activity:

    data/Subject01/walking/
        walking.trc                  mocap markers, 4-marker plate per segment
        imu data/femur_r_imu.txt     Xsens export, one file per segment

Two conventions are load-bearing and neither is discoverable from the files themselves:

  * IMU traces are named by FILENAME STEM, and those stems have to match the marker names
    in the .trc for `assemble_plate_trials` to pair them up.
  * Marker plates are found by the `_o` suffix, with `_d`, `_x`, `_y` completing each
    quadruple, and `o`-`d` / `x`-`y` are the plate's DIAGONALS.

This is dataset knowledge, which is why it lives in the build package rather than on the
trace classes — keeping it there is what stopped any of it being reusable for IMoVE, whose
clusters are neither named this way nor rectangular.
"""
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from ..IMUTrace import IMUTrace
from ..PlateTrial import PlateTrial
from ..WorldTrace import WorldTrace
from .assembly import assemble_plate_trials
from .reconstruction import (_reconstruct_from_markers, reconstruct_plate,
                             repair_reconstruction_glitches)
from .xsens import read_xsens_txt

# Fault score above which _reconstruct_from_markers treats a marker as displaced. Was a
# default argument at the call site; hoisted so it is visible as a reader-level choice.
MARKER_FAULT_THRESHOLD = 2.0

# The .trc header declares its units, but this dataset's files are inconsistent about it,
# so the magnitude is used instead: no lab coordinate legitimately exceeds 1000 m.
MM_DETECTION_THRESHOLD = 1000.0


def load_world_traces(trc_path: Union[str, Path],
                      method: str = 'merged') -> Dict[str, WorldTrace]:
    """Every marker plate in a .trc, reconstructed, repaired and keyed by segment name.

    `method` selects the reconstruction:

      'merged' (default) — `reconstruct_plate`: template Kabsch fit for the pose and a
          residual in millimetres, leave-one-out fault isolation, then the angular-speed
          repair with un-flip corrections derived from the plate's own geometry. Detects the
          union of what the two older detectors saw: shape faults the speed detector is
          blind to, and relabelings the residual is blind to.
      'legacy' — `_reconstruct_from_markers` + `repair_reconstruction_glitches`, the
          rectangle-specific path. Kept so the two can be diffed on the same data; see
          scratch/compare_plate_reconstructions.py.
    """
    if method not in ('merged', 'legacy'):
        raise ValueError(f"Unknown reconstruction method {method!r}; use 'merged' or 'legacy'.")
    trc_path = Path(trc_path)
    if not trc_path.is_file():
        raise FileNotFoundError(f"No valid TRC file found at: {trc_path}")

    with open(trc_path, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(6)]

    headers = lines[3].strip().split('\t')
    # ENDSWITH, not 'in'. Each plate contributes four marker columns named <plate>_o/_d/_x/_y
    # and the origin is the one that names the plate. A substring test would also match a
    # marker called e.g. 'shank_offset_d', quietly treating it as a plate origin.
    imu_headers = [h for h in headers if h.lower().endswith('_o')]

    df = pd.read_csv(trc_path, delimiter='\t', skiprows=6, header=None, engine='c')
    timestamps = df.iloc[:, 1].to_numpy(dtype=np.float64)

    world_traces = {}
    for imu_o_name in imu_headers:
        o_idx = headers.index(imu_o_name)
        x_idx = headers.index(imu_o_name.replace('_o', '_x'))
        y_idx = headers.index(imu_o_name.replace('_o', '_y'))
        d_idx = headers.index(imu_o_name.replace('_o', '_d'))

        o_loc = df.iloc[:, o_idx: o_idx + 3].to_numpy(dtype=np.float64)
        x_loc = df.iloc[:, x_idx: x_idx + 3].to_numpy(dtype=np.float64)
        y_loc = df.iloc[:, y_idx: y_idx + 3].to_numpy(dtype=np.float64)
        d_loc = df.iloc[:, d_idx: d_idx + 3].to_numpy(dtype=np.float64)

        if np.max(np.abs(o_loc)) > MM_DETECTION_THRESHOLD:
            o_loc /= 1000.0
            x_loc /= 1000.0
            y_loc /= 1000.0
            d_loc /= 1000.0

        clean_name = imu_o_name.lower().replace('_o', '')
        if method == 'merged':
            positions, rotations, valid, report = reconstruct_plate(
                np.stack([o_loc, d_loc, x_loc, y_loc], axis=1), timestamps, name=clean_name)
            if report['flipped'] or report['interpolated'] or report['unresolved']:
                print(f"Warning: {trc_path.name}/{clean_name}: marker reconstruction repaired "
                      f"({report['flipped']} frame(s) un-flipped from a relabeling, "
                      f"{report['interpolated']} transition frame(s) interpolated, "
                      f"{report['unresolved']} unexplained discontinuity(ies), "
                      f"{report['n_repaired_by_dropping_a_marker']} frame(s) fixed by dropping "
                      f"a marker) — {int((~valid).sum())} frame(s) marked invalid; "
                      f"fit residual median {report['residual_median_mm']:.2f} mm.")
        else:
            positions, rotations = _reconstruct_from_markers(
                o_loc, d_loc, x_loc, y_loc, threshold=MARKER_FAULT_THRESHOLD
            )
            positions, rotations, valid, report = repair_reconstruction_glitches(
                positions, rotations, timestamps, name=clean_name)
            if any(report.values()):
                print(f"Warning: {trc_path.name}/{clean_name}: marker reconstruction repaired "
                      f"({report['flipped']} frame(s) un-flipped from swapped marker labels, "
                      f"{report['interpolated']} transition frame(s) interpolated, "
                      f"{report['unresolved']} discontinuity(ies) not attributable to a swap) "
                      f"— {int((~valid).sum())} frame(s) marked invalid; "
                      f"see repair_reconstruction_glitches.")
        world_traces[clean_name] = WorldTrace(timestamps, positions, rotations, valid=valid)
    return world_traces


def load_imu_traces(folder_path: Union[str, Path]) -> Dict[str, IMUTrace]:
    """Every Xsens .txt in a trial folder, keyed by filename stem.

    The stem IS the segment name, and has to match the .trc marker names — that pairing is
    the whole reason `assemble_plate_trials` can match by exact key.
    """
    folder = Path(folder_path)
    imu_dir = folder / 'imu data' if (folder / 'imu data').is_dir() else folder

    imu_files = list(imu_dir.glob("*.txt"))
    imu_traces = {f.name.replace('.txt', ''): read_xsens_txt(f) for f in imu_files}

    if not imu_traces:
        raise FileNotFoundError(f"No IMU .txt files found in: {imu_dir}")

    return imu_traces


def load_trial(folder_path: Union[str, Path],
               align_plate_trials: bool = True) -> Dict[str, PlateTrial]:
    """One subject/activity folder -> its synchronized PlateTrials."""
    folder = Path(folder_path).resolve()
    imu_traces = load_imu_traces(folder)

    # SORTED, because glob order is filesystem order. Every trial folder holds exactly one
    # .trc today, so this has never mattered -- but "whichever the filesystem returned first"
    # is not something a cached artifact should depend on.
    trc_files = sorted(folder.glob("*.trc"))
    if not trc_files:
        raise FileNotFoundError(f"No .trc file found in {folder}")
    if len(trc_files) > 1:
        raise ValueError(f"{folder.name} holds {len(trc_files)} .trc files "
                         f"({[f.name for f in trc_files]}); which one is the trial is not "
                         f"something this should guess at.")
    world_traces = load_world_traces(trc_files[0])

    return assemble_plate_trials(imu_traces=imu_traces, world_traces=world_traces,
                                 align_plate_trials=align_plate_trials)
