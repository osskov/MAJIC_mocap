"""
Xsens MT Manager .txt exports in, IMUTrace out.

Named for the sensor rather than the extension because it is not the only .txt in play:
IMoVE's biplane half ships MC10 BioStamp data as separate accel.csv and gyro.csv per sensor
directory, which needs its own reader beside this one.
"""
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from ..IMUTrace import IMUTrace


# MTw2 datasheet full-scale ranges. A sample beyond these did not happen: it is a corrupt
# export.
#
# Measured across both datasets: 36 samples in 18 IMoVE files, all ACCELEROMETER -- no
# gyroscope sample anywhere exceeds full scale, and Al Borno is clean throughout. The
# signature is the same every time: two ADJACENT rows holding near-exact negatives of each
# other. s16/t2_treadmill_walking reads
#
#     row 3937  [   9.86,      0.66,      1.15]
#     row 3938  [ 109094.73, -29145.45, -66244.38]
#     row 3939  [-109218.09,  29214.16,  66102.56]
#     row 3940  [   9.38,      2.24,      1.54]
#
# which is a sign flip in the export, not a measurement -- an eleven-thousand-g impulse
# would have destroyed the sensor. Reconstructing those two rows recovers 9.88 and 9.54.
ACC_RANGE_MS2 = 160.0
GYRO_RANGE_DEG_S = 2000.0

# The counter is 16-bit and wraps.
_COUNTER_MODULUS = 65536


def read_xsens_txt(file_path: Union[str, Path],
                   sample_rate_hz: float = None,
                   report=None) -> 'IMUTrace':
    """Parses one Xsens MT Manager .txt export into one IMUTrace.

    Named for the sensor, not the extension, because it is not the only .txt format in
    play: the biplane half of IMoVE ships MC10 BioStamp data as separate accel.csv and
    gyro.csv per sensor directory, which needs its own constructor at this same level.

    The comment block is SCANNED rather than assumed to be a fixed length. Exports in
    hand run to 4 lines (IMoVE's 40 Hz sessions), 5 (this repo's Al Borno data) and 12
    (IMoVE's long-walk sessions); the previous hardcoded `skiprows=5` silently consumed
    the column header on the first and third of those.

    `sample_rate_hz` overrides the header's `// Update Rate`. Required when the header
    has none — the long-walk exports omit it, and the old code's silent fall back to
    100 Hz would have mislabelled the 40 Hz sessions by a factor of 2.5 with no warning.
    """
    file_path = Path(file_path)

    header_rate = None
    n_comment_lines = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("//"):
                break
            if "Update Rate" in line:
                try:
                    header_rate = float(line.split(":")[1].split("Hz")[0].strip())
                except (IndexError, ValueError):
                    pass
            n_comment_lines += 1

    freq = sample_rate_hz if sample_rate_hz is not None else header_rate
    if freq is None:
        raise ValueError(
            f"{file_path.name} has no '// Update Rate' header line, so its sample rate "
            f"cannot be determined from the file. Pass sample_rate_hz explicitly.")

    channels = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z',
                'Mag_X', 'Mag_Y', 'Mag_Z']
    # PacketCounter is read WHEN PRESENT rather than required. Every export in both datasets
    # has one, but demanding it would make an otherwise-readable file unreadable for the sake
    # of a diagnostic -- and it is a diagnostic that improves the timeline, not one the data
    # is meaningless without.
    header = pd.read_csv(file_path, delimiter='\t', skiprows=n_comment_lines, nrows=0)
    has_counter = 'PacketCounter' in header.columns
    df = pd.read_csv(
        file_path,
        delimiter='\t',
        skiprows=n_comment_lines,
        engine='c',
        usecols=(['PacketCounter'] if has_counter else []) + channels,
    )

    acc = df[['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy(dtype=np.float64)
    gyro = df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy(dtype=np.float64)
    mag = df[['Mag_X', 'Mag_Y', 'Mag_Z']].to_numpy(dtype=np.float64)

    # TIME COMES FROM THE PACKET COUNTER, not from the row index.
    #
    # `arange(n) / freq` assumes the radio never dropped a packet, and on a 17-sensor 40 Hz
    # link it drops plenty: 442 of 3935 IMoVE files have counter gaps, and in all 92 affected
    # inertial records the sensors within one record disagree about how many they lost. Each
    # dropped packet COMPRESSES that sensor's timeline by one sample from then on, so plates
    # of the same trial drift apart mid-recording -- 194 packets, 4.85 s, on
    # s24/t1_walking_001. `_shared_lag`'s median cannot absorb that because it is not a
    # constant offset, and `_assert_one_clock` cannot see it because the synthesised
    # timestamps agree perfectly while the physical content does not.
    #
    # Reading the counter puts every sample at the time it was actually measured. The trace is
    # then resampled back onto a uniform grid, because resampling, filtering and every
    # finite-difference in this repo assume one -- so a gap becomes interpolated data at the
    # right INSTANT rather than correct data at the wrong instant.
    if has_counter:
        elapsed, gaps, missing, largest = _elapsed_from_counter(df['PacketCounter'], freq)
    else:
        elapsed = np.arange(len(df), dtype=np.float64) / freq
        gaps, missing, largest = 0, 0, 0
    timestamps = elapsed

    # A row carrying a non-finite value cannot anchor an interpolant, and some exports have
    # them: s4l's long walk has rows of NaN that made the spline refuse outright. Dropped from
    # the fit rather than allowed to abort the correction, since the alternative is keeping a
    # timeline that is known to be wrong.
    finite = np.isfinite(acc).all(axis=1) & np.isfinite(gyro).all(axis=1) \
        & np.isfinite(mag).all(axis=1)

    # A SAMPLE BEYOND THE SENSOR'S FULL SCALE IS NOT A MEASUREMENT, so it is dropped from the
    # fit alongside the NaNs and reconstructed from its neighbours rather than kept.
    #
    # Counting these was not enough. An impulse of 1.15e5 m/s^2 rings through a Butterworth
    # for hundreds of samples, and `_lag_seconds` correlates on gyro MAGNITUDE, where one
    # such sample dominates the entire trace. A diagnostic that records the spike and then
    # hands it downstream anyway buys nothing.
    #
    # Whole ROWS are dropped, not individual channels, because the corruption signature is a
    # burst rather than a channel fault, and a spline fitted per channel on different support
    # would leave the axes mutually inconsistent for that instant.
    in_range = ((np.abs(np.nan_to_num(acc)) <= ACC_RANGE_MS2).all(axis=1)
                & (np.abs(np.nan_to_num(gyro)) <= GYRO_RANGE_DEG_S).all(axis=1))
    n_out_of_range_acc = int((np.abs(acc) > ACC_RANGE_MS2).any(axis=1).sum())
    n_out_of_range_gyro = int((np.abs(gyro) > GYRO_RANGE_DEG_S).any(axis=1).sum())
    raw_acc_abs_max = float(np.abs(acc[finite]).max()) if finite.any() else np.nan
    raw_gyro_abs_max = float(np.abs(gyro[finite]).max()) if finite.any() else np.nan

    usable = finite & in_range
    if (missing or not usable.all()) and usable.sum() > 3:
        uniform = np.arange(elapsed[-1] * freq + 1, dtype=np.float64) / freq
        # extrapolate=False rather than the default. A corrupt or non-finite row at either
        # END of the record leaves the grid running past the fit's support, and a cubic
        # extrapolated even a few samples is unbounded -- the same failure that produced
        # 1.3e10 m positions when the resampler extrapolated Al Borno's padding. The edges
        # are held at the nearest real sample instead, which is wrong by at most one
        # sample's motion and cannot diverge.
        acc, gyro, mag = (_spline_onto(elapsed[usable], values[usable], uniform)
                          for values in (acc, gyro, mag))
        timestamps = uniform
    else:
        # Too few usable rows to fit anything. The counter's own elapsed time is not uniform
        # when packets were lost, and every resampler and finite-difference downstream
        # assumes it is, so the index grid is the only safe thing to hand back.
        timestamps = np.arange(len(acc), dtype=np.float64) / freq

    if report is not None:
        report.add('S1_parse', 'file', file_path.name,
                   n_samples=len(timestamps), rate_hz=freq,
                   rate_source='header' if sample_rate_hz is None else 'caller',
                   n_comment_lines=n_comment_lines,
                   has_packet_counter=has_counter,
                   packet_counter_gaps=gaps, missing_samples=missing,
                   largest_gap_samples=largest,
                   timeline_error_s=missing / freq,
                   n_non_finite_rows=int((~finite).sum()),
                   n_out_of_range_acc=n_out_of_range_acc,
                   n_out_of_range_gyro=n_out_of_range_gyro,
                   n_rows_dropped=int((~usable).sum()),
                   # Post-repair, so this is what actually reached the parquet; the raw_*
                   # pair beside it is what the export claimed.
                   acc_abs_max=float(np.abs(acc).max()),
                   gyro_abs_max=float(np.abs(gyro).max()),
                   raw_acc_abs_max=raw_acc_abs_max,
                   raw_gyro_abs_max=raw_gyro_abs_max)

    return IMUTrace(timestamps=timestamps, gyro=gyro, acc=acc, mag=mag)


def _spline_onto(source_times: np.ndarray, values: np.ndarray,
                 target_times: np.ndarray) -> np.ndarray:
    """Cubic spline through `values`, evaluated on `target_times`, edges held constant."""
    resampled = CubicSpline(source_times, values, axis=0,
                            extrapolate=False)(target_times)
    before = target_times < source_times[0]
    after = target_times > source_times[-1]
    resampled[before] = values[0]
    resampled[after] = values[-1]
    return resampled


def _elapsed_from_counter(counter, freq: float):
    """(elapsed seconds per row, n gaps, n missing samples, largest gap).

    The counter is 16-bit, so a negative step is a wrap rather than time running backwards.
    """
    values = counter.to_numpy(dtype=np.int64)
    step = np.diff(values)
    step = np.where(step < 0, step + _COUNTER_MODULUS, step)
    skipped = step - 1
    elapsed = np.concatenate([[0.0], np.cumsum(step)]) / freq
    return (elapsed, int((skipped > 0).sum()), int(skipped[skipped > 0].sum()),
            int(skipped.max()) if len(skipped) else 0)

