import os
import re

import nimblephysics as nimble
import numpy as np

trial_directory = os.path.abspath('../../test_data/Pilot_IMU_Data/Random')
model_path = os.path.abspath('../../test_data/Pilot_IMU_Data/scaled_with_imus.osim')
# Open skeleton
model_skeleton = nimble.biomechanics.OpenSimParser.parseOsim(model_path).skeleton

# get files that contain "segment" and ".mot" in the name
mot_files = [f for f in os.listdir(trial_directory) if 'segment' in f and '.mot' in f]

# Organize the files by segment number
# Function to extract segment number from file name
def extract_segment_number(filename):
    match = re.search(r'_segment_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return -1  # Default return if segment not found

# Sort files based on segment number
mot_files = sorted(mot_files, key=extract_segment_number)


# Load all of them
kinematics = [nimble.biomechanics.OpenSimParser.loadMot(model_skeleton, trial_directory + '/' + f) for f in mot_files]

# Stitch them together
stitched_kinematics = np.concatenate([k.poses for k in kinematics], axis=1)
stitched_timestamps = np.concatenate([k.timestamps for k in kinematics])
start_time = stitched_timestamps[0]
num_stamps = len(stitched_timestamps)
dt = round(stitched_timestamps[1] - stitched_timestamps[0], 3)
end_time = start_time + num_stamps * dt

stitched_timestamps = np.linspace(start=start_time, num=num_stamps, stop=end_time, endpoint=False)

# Save the stitched file
nimble.biomechanics.OpenSimParser.saveMot(model_skeleton, trial_directory + '/OMC_ik.mot', stitched_timestamps, stitched_kinematics)