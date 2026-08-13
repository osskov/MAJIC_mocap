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

from ..IMUTrace import IMUTrace


def read_xsens_txt(file_path: Union[str, Path],
                   sample_rate_hz: float = None) -> 'IMUTrace':
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

    df = pd.read_csv(
        file_path,
        delimiter='\t',
        skiprows=n_comment_lines,
        engine='c',
        usecols=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']
    )

    timestamps = np.arange(len(df), dtype=np.float64) / freq
    acc = df[['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy(dtype=np.float64)
    gyro = df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy(dtype=np.float64)
    mag = df[['Mag_X', 'Mag_Y', 'Mag_Z']].to_numpy(dtype=np.float64)

    return IMUTrace(timestamps=timestamps, gyro=gyro, acc=acc, mag=mag)

