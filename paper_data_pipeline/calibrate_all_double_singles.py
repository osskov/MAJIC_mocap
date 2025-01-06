import json
import os
from typing import List, Dict, Tuple

import numpy as np

from src.toolchest.MagnetometerCalibration import MagnetometerCalibration

root_path = '../../test_data/S3_IMU_Data/Calibration'
mapping_path = '../../test_data/S3_IMU_Data/mapping.json'
output_path = '../../test_data/S3_IMU_Data/magnetometer_calibration.json'

# Load the mapping
with open(mapping_path, 'r') as f:
    mapping = json.load(f)

calibrated_imus: Dict[str, List[MagnetometerCalibration]] = {}

data_files = os.listdir(root_path)
data_files = [f for f in data_files if f.endswith('_mag.csv')]
calibrations_and_average_magnitudes: List[Tuple[MagnetometerCalibration, float]] = []
for f in data_files:
    name = ''
    for key in mapping:
        if key in f:
            name = mapping[key]
    print(f'Processing {f} for {name}')
    # Expected headers
    # unix_timestamp_microsec, time_s, mx_microT, my_microT, mz_microT
    data = np.loadtxt(os.path.join(root_path, f), delimiter=',', skiprows=1)
    time = data[:, 1]
    mag = data[200:-200, 2:5]

    mag_calibration = MagnetometerCalibration()
    mag_calibration.fit_ellipsoid(mag)

    if name not in calibrated_imus:
        calibrated_imus[name] = []
    calibrated_imus[name].append(mag_calibration)

    processed_mags = mag_calibration.process(mag)
    processed_mag_norms = np.linalg.norm(processed_mags, axis=1)
    calibrations_and_average_magnitudes.append((mag_calibration, np.mean(processed_mag_norms)))
overall_average_magnitude = np.mean([m[1] for m in calibrations_and_average_magnitudes])
for i in range(len(calibrations_and_average_magnitudes)):
    mag_calibration = calibrations_and_average_magnitudes[i][0]
    mag_calibration.overall_scale = overall_average_magnitude / calibrations_and_average_magnitudes[i][1]

    print(f'File: {f}')
    print(f'Estimated center: {mag_calibration.center}')
    print(f'Estimated radius: {mag_calibration.radius}')
    print(f'Estimated scaling: {mag_calibration.scaling}')
    print(f'Estimated axis: {mag_calibration.axis}')
    print(f'Estimated overall scale: {mag_calibration.overall_scale}')

output_dict = {}
for name in calibrated_imus:
    calibrations = calibrated_imus[name]
    if len(calibrations) == 2:
        # Average the center point
        calibrations[0].center = (calibrations[0].center + calibrations[1].center) / 2
    output_dict[name] = calibrations[0].to_dict()

with open(output_path, 'w') as f:
    json.dump(output_dict, f, indent=4)
