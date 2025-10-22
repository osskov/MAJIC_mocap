import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

def load_trc_to_dataframe(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract header metadata
    metadata = lines[:5]  # PathFileType, DataRate line, and column header line
    marker_names = lines[3].strip().split('\t')[2:] # Skip the first column which is the frame number
    marker_names = [name for name in marker_names if name]  # Remove empty names
    column_names = lines[4].strip().split('\t')

    marker_to_column_name_map = {marker_name: column_name[1:] for column_name, marker_name in zip(column_names[::3], marker_names)}
    column_names = ['Frame#', 'Time'] + column_names

    # Load numerical data starting from line 5
    data = pd.read_csv(file_path, sep='\t', skiprows=5, header=None, names=column_names, index_col=False)
    data.columns = column_names

    return data, metadata, marker_to_column_name_map

def write_dataframe_to_trc(df, metadata, output_path):
    with open(output_path, 'w') as f:
        # Write metadata lines
        for line in metadata:
            f.write(line if line.endswith('\n') else line + '\n')

        # Write data
        df.to_csv(f, sep='\t', index=False, header=False, float_format='%.6f')

def swap_markers(df, marker1: str, marker2: str, index1: int, index2: int):
    """
    Swaps the X/Y/Z position data of two markers over a range of frames
    from index1 to index2 (inclusive).

    Args:
        df (pd.DataFrame): The TRC-format DataFrame.
        marker1 (str): Name of the first marker (e.g., 'R.Heel').
        marker2 (str): Name of the second marker (e.g., 'L.Heel').
        index1 (int): Start frame index (inclusive).
        index2 (int): End frame index (inclusive).

    Returns:
        None (modifies df in-place)
    """
    if index1 > index2:
        index1, index2 = index2, index1  # Ensure proper slice order

    for axis in ['X', 'Y', 'Z']:
        col1 = f"{axis}{marker1}"
        col2 = f"{axis}{marker2}"
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"Missing expected columns: {col1} or {col2}")

        # Swap values over the full slice
        temp = df.loc[index1:index2, col1].copy()
        df.loc[index1:index2, col1] = df.loc[index1:index2, col2].values
        df.loc[index1:index2, col2] = temp.values

# c1_trc_file = f"../data/DO_NOT_MODIFY_AlBorno/Subject01/complexTasks/Mocap/complexTasks.trc"
# df, metadata, transfer_dict = load_trc_to_dataframe(c1_trc_file)
# swap_markers(df, transfer_dict['L.Foot_IMU_D'], transfer_dict['L.Foot_IMU_O'], 45022, len(df) - 1)
# write_dataframe_to_trc(df, metadata, c1_trc_file.replace('DO_NOT_MODIFY_AlBorno', 'ODay_Data'))
#
# w6_trc_file = f"../data/DO_NOT_MODIFY_AlBorno/Subject06/walking/Mocap/walking.trc"
# df, metadata, transfer_dict = load_trc_to_dataframe(w6_trc_file)
# swap_markers(df, transfer_dict['L.Femur_IMU_D'], transfer_dict['L.Femur_IMU_Y'], 42087, len(df) - 1)
# for axis in ['X', 'Y', 'Z']:
#     # Swap values over the full slice
#     df.loc[:, axis+transfer_dict["L.Femur_IMU_D"]] = df.loc[:, axis+transfer_dict["L.Femur_IMU_X"]].values + df.loc[:, axis+transfer_dict["L.Femur_IMU_Y"]].values - df.loc[:, axis+transfer_dict["L.Femur_IMU_O"]].values
#     df.loc[:, axis + transfer_dict["L.Femur_IMU_5"]] = np.zeros(len(df))  # Set L.Femur_IMU_5 to zero
# write_dataframe_to_trc(df, metadata, w6_trc_file.replace('DO_NOT_MODIFY_AlBorno', 'ODay_Data'))
#
# c6_trc_file = f"../data/DO_NOT_MODIFY_AlBorno/Subject06/complexTasks/Mocap/complexTasks.trc"
# df, metadata, transfer_dict = load_trc_to_dataframe(c6_trc_file)
# for axis in ['X', 'Y', 'Z']:
#     # Swap values over the full slice
#     df.loc[:, axis+transfer_dict["L.Femur_IMU_D"]] = df.loc[:, axis+transfer_dict["L.Femur_IMU_X"]].values + df.loc[:, axis+transfer_dict["L.Femur_IMU_Y"]].values - df.loc[:, axis+transfer_dict["L.Femur_IMU_O"]].values
#     df.loc[:, axis+transfer_dict["L.Femur_IMU_5"]] = np.zeros(len(df))  # Set L.Femur_IMU_5 to zero
# write_dataframe_to_trc(df, metadata, c6_trc_file.replace('DO_NOT_MODIFY_AlBorno', 'ODay_Data'))
#
# c9_trc_file = f"../data/DO_NOT_MODIFY_AlBorno/Subject09/complexTasks/Mocap/complexTasks.trc"
# df, metadata, transfer_dict = load_trc_to_dataframe(c9_trc_file)
# swap_markers(df, transfer_dict['R.Femur_IMU_D'], transfer_dict['R.Femur_IMU_X'], 1453, 1658)
# swap_markers(df, transfer_dict['L.Tibia_IMU_D'], transfer_dict['L.Tibia_IMU_Y'], 1453, 1658)
# write_dataframe_to_trc(df, metadata, c9_trc_file.replace('DO_NOT_MODIFY_AlBorno', 'ODay_Data'))
#
# w10_trc_file = f"../data/DO_NOT_MODIFY_AlBorno/Subject10/walking/Mocap/walking.trc"
# df, metadata, transfer_dict = load_trc_to_dataframe(w10_trc_file)
# swap_markers(df, transfer_dict['L.Tibia_IMU_D'], transfer_dict['L.Tibia_IMU_Y'], 51451, 51501)
# swap_markers(df, transfer_dict['L.Tibia_IMU_O'], transfer_dict['L.Tibia_IMU_X'], 53863, 53933)
# swap_markers(df, transfer_dict['L.Tibia_IMU_D'], transfer_dict['L.Tibia_IMU_Y'], 56466, 56536)
# write_dataframe_to_trc(df, metadata, w10_trc_file.replace('DO_NOT_MODIFY_AlBorno', 'ODay_Data'))

# w9_trc_file = f"data/DO_NOT_MODIFY_AlBorno/Subject09/walking/Mocap/walking.trc"
# df, metadata, transfer_dict = load_trc_to_dataframe(w9_trc_file)
# swap_markers(df, transfer_dict['R.Foot_IMU_D'], transfer_dict['L.Foot_IMU_D'], 0, len(df) - 1)
# swap_markers(df, transfer_dict['R.Foot_IMU_O'], transfer_dict['L.Foot_IMU_O'], 0, len(df) - 1)
# swap_markers(df, transfer_dict['R.Foot_IMU_X'], transfer_dict['L.Foot_IMU_X'], 0, len(df) - 1)
# swap_markers(df, transfer_dict['R.Foot_IMU_Y'], transfer_dict['L.Foot_IMU_Y'], 0, len(df) - 1)
# write_dataframe_to_trc(df, metadata, w9_trc_file.replace('DO_NOT_MODIFY_AlBorno', 'ODay_Data'))

# w5_trc_file = f"data/DO_NOT_MODIFY_AlBorno/Subject05/walking/Mocap/walking.trc"
# df, metadata, transfer_dict = load_trc_to_dataframe(w5_trc_file)
# swap_markers(df, transfer_dict['R.Foot_IMU_D'], transfer_dict['L.Foot_IMU_D'], 0, len(df) - 1)
# swap_markers(df, transfer_dict['R.Foot_IMU_O'], transfer_dict['L.Foot_IMU_O'], 0, len(df) - 1)
# swap_markers(df, transfer_dict['R.Foot_IMU_X'], transfer_dict['L.Foot_IMU_X'], 0, len(df) - 1)
# swap_markers(df, transfer_dict['R.Foot_IMU_Y'], transfer_dict['L.Foot_IMU_Y'], 0, len(df) - 1)
# write_dataframe_to_trc(df, metadata, w5_trc_file.replace('DO_NOT_MODIFY_AlBorno', 'ODay_Data'))

for subject_num in range(0, 12):  # Subjects 01 to 11
    # Format subject number with leading zero
    subject = f"Subject{subject_num:02d}"
    for activity in ['walking', 'complexTasks']:

        trc_file = f"data/ODay_Data/{subject}/{activity}/Mocap/{activity}.trc"
        try:
            with open(trc_file, 'r') as file:
                lines = file.readlines()

            headers = lines[3].strip().split('\t')
            imu_headers = [header.split('_')[0] for header in headers if ('_O' in header or '_3' in header)]
            data = [line.strip().split('\t') for line in lines[6:]]  # Skip empty line and read the data
            data = np.array(data, dtype=float)
            timestamps = data[:, 1]

            # For each IMU header, isolate any data that has a matchign name
            for header in imu_headers:
                plt.figure(figsize=(12, 6))
                indices = [i for i, h in enumerate(headers) if h.startswith(header)]
                if indices:
                    for index in indices:
                        marker_j = data[:, index:index+3]
                        for second_index in indices:
                            if second_index <= index:
                                continue
                            if "_5" in headers[index] or "_5" in headers[second_index]:
                                continue
                            marker_k = data[:, second_index:second_index+3]
                            # Calculate the distance between the two markers
                            distances = np.linalg.norm(marker_j - marker_k, axis=1)
                            plt.plot(distances, label=f"{headers[index]} vs {headers[second_index]}")
                plt.xlabel('Sample')
                plt.ylabel('Distance (m)')
                plt.title(f"Distances for {header} IMU data {subject} - {activity}")
                plt.legend()
                plt.show()
        except:
            print(f"Could not read TRC file for {subject} - {activity}. File may not exist or be corrupted.")
            continue