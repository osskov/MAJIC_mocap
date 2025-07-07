import numpy as np
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt


def _read_mot_file_(file_path) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    Reads a .mot file and parses its header and data.

    Args:
        file_path (str): Path to the .mot file.

    Returns:
        Tuple[Dict[str, Any], Dict[str, List[float]]]: Returns the header information and data as a dictionary.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

        # Parse header
        header_info = {}
        i = 0
        while lines[i].strip() != "endheader":
            if "=" in lines[i]:
                key, value = lines[i].strip().split("=")
                header_info[key] = value
            i += 1

        # Skip the "endheader" line
        i += 1

        # Parse column headers
        headers = lines[i].strip().split()
        i += 1

        # Parse data rows
        data = []
        for line in lines[i:]:
            values = list(map(float, line.strip().split()))
            data.append(values)

        # Create a dictionary where each header maps to its corresponding data column
        data_dict = {headers[j]: [row[j] for row in data] for j in range(len(headers))}

    return header_info, data_dict

mot_files = ["data/ODay_Data/Subject03/walking/Mocap/ikResults/walking_IK.mot",
             "data/ODay_Data/Subject03/walking/Mocap/ikResults/IKWithErrorsUniformWeights/walking_IK.mot",
             "data/ODay_Data/Subject03/walking/IMU/xsens/IKResults/IKWithErrorsUniformWeights/walking_IK.mot",
             "data/ODay_Data/Subject03/walking/IMU/mahony/IKResults/IKWithErrorsUniformWeights/walking_IK.mot",
             "data/ODay_Data/Subject03/walking/IMU/madgwick/IKResults/IKWithErrorsUniformWeights/walking_IK.mot",
             "data/ODay_Data/Subject03/walking/IMU/majic/IKResults/IKWithErrorsUniformWeights/walking_IK.mot"]

# Add "_orientationErrors.sto" to the end of each file name
error_files = [mot_file.replace(".mot", ".mot_orientationErrors.sto") for mot_file in mot_files]

datasets = []
for mot_file in mot_files:
    header, data = _read_mot_file_(mot_file)
    datasets.append(data)

def compare_kinematics(datasets: List[Dict[str, List[float]]]) -> None:
    ground_truth = datasets[1]
    marker_ik = datasets[0]
    xsens_ik = datasets[2]
    mahony_ik = datasets[3]
    madgwick_ik = datasets[4]
    majic_ik = datasets[5]  # Assuming this is the Majic IK results

    errors = {
        "marker": {},
        "xsens": {},
        "mahony": {},
        "madgwick": {},
        "majic": {}
    }

    for key in ground_truth.keys():
        if key == "time" or "pelvis" in key or "arm" in key or "elbow" in key or "pro" in key or "beta" in key:
            continue
        print(f"Comparing {key}:")
        gt_values = ground_truth[key]
        marker_values = marker_ik[key]
        xsens_values = xsens_ik[key]
        mahony_values = mahony_ik[key]
        madgwick_values = madgwick_ik[key]
        majic_values = majic_ik[key]
        marker_error = [abs(gt - mk) for gt, mk in zip(gt_values, marker_values)]
        xsens_error = [abs(gt - xs) for gt, xs in zip(gt_values, xsens_values)]
        mahony_error = [abs(gt - mh) for gt, mh in zip(gt_values, mahony_values)]
        madgwick_error = [abs(gt - md) for gt, md in zip(gt_values, madgwick_values)]
        majic_error = [abs(gt - mj) for gt, mj in zip(gt_values, majic_values)]
        errors["majic"][key] = majic_error
        errors["marker"][key] = marker_error
        errors["xsens"][key] = xsens_error
        errors["mahony"][key] = mahony_error
        errors["madgwick"][key] = madgwick_error

        plt.Figure()
        plt.plot(ground_truth["time"], marker_error, label='Marker IK Error', color='blue')
        plt.plot(ground_truth["time"], xsens_error, label='Xsens IK Error', color='orange')
        plt.plot(ground_truth["time"], mahony_error, label='Mahony IK Error', color='green')
        plt.plot(ground_truth["time"], madgwick_error, label='Madgwick IK Error', color='red')
        plt.plot(ground_truth["time"][:-1], majic_error, label='Majic IK Error', color='purple')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (deg)')
        plt.title(f'IK Error Comparison for {key}')
        plt.legend()
        plt.grid()
        plt.show()

        # Make bar chart too for mean and whiskers for std
        plt.Figure()
        plt.bar(['Marker', 'Xsens', 'Mahony', 'Madgwick', 'Majic'],
                [sum(marker_error)/len(marker_error), sum(xsens_error)/len(xsens_error),
                 sum(mahony_error)/len(mahony_error), sum(madgwick_error)/len(madgwick_error), sum(majic_error)/len(majic_error)],
                yerr=[np.std(marker_error), np.std(xsens_error), np.std(mahony_error), np.std(madgwick_error), np.std(majic_error)],
                capsize=5, color=['blue', 'orange', 'green', 'red', 'purple'])
        plt.ylabel('Mean Error (deg)')
        plt.title(f'Mean IK Error Comparison for {key}')
        plt.grid()
        plt.show()

def plot_errors(error_files):
    for error_file in error_files:
        if "ik" in error_file:
            continue
        header, data = _read_mot_file_(error_file)
        time = data['time']
        errors = {key: data[key] for key in data.keys() if key != 'time'}

        plt.figure(figsize=(10, 6))
        for key, values in errors.items():
            values = np.array(values)
            plt.plot(time, values * 180/np.pi, label=key)
        plt.xlabel('Time (s)')
        plt.ylabel('Error (deg)')
        plt.title(f'Orientation Errors from {error_file}')
        plt.legend()
        plt.grid()
        plt.show()

# plot_errors(error_files)
compare_kinematics(datasets)


