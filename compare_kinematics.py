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

mot_files = {
    "markers": "data/ODay_Data/Subject03/walking/Mocap/ikResults/IKWithErrorsUniformWeights/walking_IK.mot",
    "ground truth": "data/ODay_Data/Subject03/walking/Mocap/ikResults/walking_IK.mot",
    "original madgwick": "data/DO_NOT_MODIFY_AlBorno/Subject03/walking/IMU/madgwick/IKResults/IKWithErrorsUniformWeights/walking_IK.mot",
    # "original xsens": "data/DO_NOT_MODIFY_AlBorno/Subject03/walking/IMU/xsens/IKResults/IKWithErrorsUniformWeights/walking_IK.mot",
    # "original orien, original model": "data/ODay_Data/Subject03/walking/IMU/madgwick/IKResults/IKWithErrorsUniformWeights/walking_IK.mot",
    # "original orien, modified model": "data/ODay_Data/Subject03/walking/IMU/madgwick/IKResults/IKWithErrorsUniformWeights/walking_IK_markermodel.mot",
    # "madgwick": "data/ODay_Data/Subject03/walking/IMU/madgwick/IKResults/IKWithErrorsUniformWeights/walking_IK_customOrien.mot",
    "mag free": "data/ODay_Data/Subject03/walking/IMU/Mag Free/IKResults/IKWithErrorsUniformWeights/walking_IK.mot",
    # "unprojected": "data/ODay_Data/Subject03/walking/IMU/Unprojected/IKResults/IKWithErrorsUniformWeights/walking_IK.mot",
    "never project": "data/ODay_Data/Subject03/walking/IMU/Never Project/IKResults/IKWithErrorsUniformWeights/walking_IK.mot",
    # "new xsens": "data/ODay_Data/Subject03/walking/IMU/xsens/IKResults/IKWithErrorsUniformWeights/walking_IK.mot",
    # "new mahony": "data/ODay_Data/Subject03/walking/IMU/mahony/IKResults/IKWithErrorsUniformWeights/walking_IK.mot",
}

# mot_files = {
#     "markers": "data/ODay_Data/Subject03/complexTasks/Mocap/ikResults/IKWithErrorsUniformWeights/complexTasks_IK.mot",
#     "ground truth": "data/DO_NOT_MODIFY_AlBorno/Subject03/complexTasks/Mocap/ikResults/complexTasks_IK.mot",
#     "madgwick": "data/ODay_Data/Subject03/complexTasks/IMU/madgwick/IKResults/IKWithErrorsUniformWeights/complexTasks_IK.mot",
#     "xsens": "data/ODay_Data/Subject03/complexTasks/IMU/xsens/IKResults/IKWithErrorsUniformWeights/complexTasks_IK.mot",
#     # "mag free": "data/ODay_Data/Subject03/complexTasks/IMU/Mag Free/IKResults/IKWithErrorsUniformWeights/complexTasks_IK.mot",
#     "unprojected": "data/ODay_Data/Subject03/complexTasks/IMU/Unprojected/IKResults/IKWithErrorsUniformWeights/complexTasks_IK.mot",
#     "never project": "data/ODay_Data/Subject03/complexTasks/IMU/Never Project/IKResults/IKWithErrorsUniformWeights/complexTasks_IK.mot",
# }

# Add "_orientationErrors.sto" to the end of each file name
error_files = {name: mot_file.replace(".mot", ".mot_orientationErrors.sto") for name, mot_file in mot_files.items()}

datasets = {}
for name, mot_file in mot_files.items():
    header, data = _read_mot_file_(mot_file)
    datasets[name] = data


def compare_kinematics(datasets: Dict[str, Dict[str, List[float]]]) -> None:
    ground_truth = datasets['ground truth']
    num_rows = 3
    num_cols = 5
    errors = {}

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 12), sharey=True)
    fig.suptitle("Kinematic Error Comparison", fontsize=16)
    ax_idx = 0

    ignore_keywords = ["time", "pelvis", "arm", "elbow", "pro", "beta"]

    for segment in ground_truth:
        if any(key in segment for key in ignore_keywords):
            continue

        print(f"\nComparing {segment}:")
        ground_truth_values = ground_truth[segment]

        for method in datasets:
            if method == 'ground truth':
                continue
            if method not in errors:
                errors[method] = {}
            values = datasets[method].get(segment, [])
            num_samples = min(len(ground_truth_values), len(values))
            error = [abs(gt - val) for gt, val in zip(ground_truth_values[:num_samples], values[:num_samples])]
            errors[method][segment] = error

        # Plotting per-segment bar chart
        row, col = divmod(ax_idx, num_cols)
        ax = axes[row][col]
        ax_idx += 1

        for i, (method, seg_errors) in enumerate(errors.items()):
            if segment not in seg_errors:
                continue
            err = np.array(seg_errors[segment])
            mean = np.mean(err ** 2) ** 0.5  # RMS error
            std = np.std(err)
            print(f"{method}: Mean = {mean:.2f} deg, Std = {std:.2f} deg")
            ax.bar(i, mean, yerr=std, capsize=5, label=method)

        ax.set_title(segment)
        ax.set_ylabel("Error (deg)")
        ax.set_xticks(range(len(errors)))
        ax.set_xticklabels([m for m in errors if segment in errors[m]], rotation=45, ha='right')

    # Hide unused subplots
    for i in range(ax_idx, num_rows * num_cols):
        row, col = divmod(i, num_cols)
        fig.delaxes(axes[row][col])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_errors(error_files):
    error_by_segment = {}
    for method, error_file in error_files.items():
        if "ground truth" in method or "trad markers" in method:
            continue
        header, data = _read_mot_file_(error_file)
        for segment, values in data.items():
            if segment == 'time':
                continue
            if segment not in error_by_segment:
                error_by_segment[segment] = {}
            error_by_segment[segment][method] = values


    # Make a plot with axes for each segment
    num_segments = len([key for key in error_by_segment.keys()])
    counter = 0
    fig, ax = plt.subplots(1, num_segments, sharey=True)
    for key, errors_by_method in error_by_segment.items():
        for method, values in errors_by_method.items():
            error_mean = np.mean(values)
            error_std = np.std(values)
            ax[counter].bar(method, error_mean, yerr=error_std, label=method)
        ax[counter].set_title(key)
        ax[counter].set_xlabel('Method')
        ax[counter].set_ylabel('Error (deg)')
        counter += 1
    plt.show()


# plot_errors(error_files)
compare_kinematics(datasets)


