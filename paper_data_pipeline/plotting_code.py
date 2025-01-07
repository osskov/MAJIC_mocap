import os.path
from typing import List, Dict

import numpy as np

from paper_data_pipeline.generate_kinematics_from_precomputed_plates import generate_kinematics_report
import nimblephysics as nimble
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 20,  # Increase font size for all text
    'font.family': 'serif',  # Use a serif font family
    'font.serif': ['Times New Roman'],  # Set Times New Roman as the default font
    'axes.titlesize': 22,  # Larger title font size
    'axes.labelsize': 18,  # Larger axis label font size
    'xtick.labelsize': 18,  # Larger x-axis tick font size
    'ytick.labelsize': 18,  # Larger y-axis tick font size
    'legend.fontsize': 16,  # Larger legend font size
})

kinematics: List[Dict] = []

# Collect kinematics
for subject_number in range(1, 4):
    kinematic = generate_kinematics_report(
        b3d_path=f"../data/S{subject_number}_IMU_Data/S{subject_number}.b3d",
        model_path=f"../data/S{subject_number}_IMU_Data/SimplifiedIMUModel.osim",
        cached_history_path=f"../data/S{subject_number}_IMU_Data/Walking/cached_skeleton_history.npz",
        output_path=f"../data/S{subject_number}_IMU_Data/Walking/",
        output_csv=f"../data/Walking_Results/S{subject_number}_Walking_joint_angles.csv"
    )

    kinematics.append(kinematic)

dof_labels = []
def split_median_by_activity():
    activity_order = ['Walking', 'Running', 'Stairs and Side Stepping', 'Standing and Sitting',
                      'Stairs and Side Stepping']
    activity_timestamps = [
        [0, 2300, 3400, 5000, 8100, 10100, 11900, 13000, 14400, 18000, 20200, 22400, 23600, 25000, 28800, 30700, 32800, 34000, 35200, 38500, 41000, 43500, 44900, 46300, 48000],
        [0, 2400, 3500, 4600, 7000, 9300, 10800, 11800, 14000, 15300, 17000, 18700, 19800, 21500, 22400, 24100, 25500, 27600, 30200, 32100, 33800, 36000, 37600, 38900, 40800, 42900, 44800, 46000, 47400, 48000],
        [0, 2400, 4000, 5700, 8800, 12400, 13800, 15000, 16300, 18800, 21000, 22700, 23700, 25500, 30000, 32000, 33700, 35100, 36500, 40700, 43600, 45800, 46900, 48000]]

    walking_splits = {method: [] for method in kinematics[0].keys()}
    running_splits = {method: [] for method in kinematics[0].keys()}
    stairs_splits = {method: [] for method in kinematics[0].keys()}
    standing_splits = {method: [] for method in kinematics[0].keys()}

    for subj_num in range(len(kinematics)):
        for method, subj_kinem in kinematics[subj_num].items():
            for joint_angles in subj_kinem:
                if joint_angles[0] == 0. and joint_angles[100] == 0.:
                    print('Skipping unfilled joint')
                    continue
                for activity_index in range(len(activity_timestamps[subj_num]) - 1):
                    activity = activity_order[activity_index % 5]
                    activity_start = activity_timestamps[subj_num][activity_index]
                    activity_end = activity_timestamps[subj_num][activity_index + 1]

                    if activity_end <= 5000:
                        continue
                    if activity_start < 5000:
                        activity_start = 5000

                    if activity == 'Walking':
                        walking_splits[method].extend(joint_angles[activity_start:activity_end])
                    elif activity == 'Running':
                        running_splits[method].extend(joint_angles[activity_start:activity_end])
                    elif activity == 'Stairs and Side Stepping':
                        stairs_splits[method].extend(joint_angles[activity_start:activity_end])
                    elif activity == 'Standing and Sitting':
                        standing_splits[method].extend(joint_angles[activity_start:activity_end])

    print("data split")
    return walking_splits, running_splits, stairs_splits, standing_splits

def plot_median_by_activity():
    activity_splits = split_median_by_activity()
    activity_splits = [activity_splits[3], activity_splits[2], activity_splits[0], activity_splits[1]]
    activities_display_names = ['Standing and Sitting', 'Stairs and Side Stepping', 'Walking', 'Running']

    # Prepare the data for plotting
    boxplot_data = []
    positions = []
    labels = []
    current_pos = 1  # Starting position for the first boxplot
    width = 0.8  # Total width allocated for each group of boxplots (per joint)
    spacing = 0.5  # Spacing between groups
    title = 'Median Joint Angle Error by Activity per Method'
    for i, activity_split in enumerate(activity_splits):
        method_data_list = []

        # Get the methods present for this joint
        methods_in_activity = list(activity_split.keys())
        num_methods_in_activity = len(methods_in_activity)
        # Calculate offsets for methods within the joint group
        offsets = np.linspace(-width / 2, width / 2, num=num_methods_in_activity + 2)[1:-1]

        for offset, method in zip(offsets, methods_in_activity):
            if 'Markers' in method:
                continue
            marker_data = np.array(activity_split['Markers']) * 180 / np.pi
            method_data = np.array(activity_split[method]) * 180 / np.pi

            error = np.abs(marker_data - method_data)
            error_slope = np.diff(error)

            plot_quantity = error_slope
            # Extract statistics required for the boxplot
            stats = {
                'med': np.median(plot_quantity),
                'q1': np.percentile(plot_quantity, 30),
                'q3': np.percentile(plot_quantity, 70),
                'whislo': np.percentile(plot_quantity, 10),
                'whishi': np.percentile(plot_quantity, 90),
                'mean': np.mean(plot_quantity),
                'fliers': [],  # Empty list since we're not displaying outliers
                'label': method
            }
            boxplot_data.append(stats)
            # Calculate position for this boxplot
            pos = current_pos + offset
            positions.append(pos)
            method_data_list.append(stats)

        labels.append(activities_display_names[i])
        current_pos += width + spacing  # Move to the next group position

    plot_boxes(boxplot_data, positions, labels, width, spacing, title)

def plot_median_by_joint_axis():
    joint_splits = [{method: [] for method in kinematics[0].keys()} for _ in range(3)]
    for subj_num in range(len(kinematics)):
        for method, subj_kinem in kinematics[subj_num].items():
            for dof_num, joint_angles in enumerate(subj_kinem):
                joint_splits[dof_num % 3][method].extend(joint_angles)

    # Prepare the data for plotting
    boxplot_data = []
    positions = []
    labels = []
    current_pos = 1  # Starting position for the first boxplot
    width = 0.8  # Total width allocated for each group of boxplots (per joint)
    spacing = 0.5  # Spacing between groups
    title = 'Median Joint Angle Error by Joint Axis per Method'

    for i, joint_split in enumerate(joint_splits):
        method_data_list = []

        # Get the methods present for this joint
        methods_in_joint = list(joint_split.keys())
        num_methods_in_joint = len(methods_in_joint)
        # Calculate offsets for methods within the joint group
        offsets = np.linspace(-width / 2, width / 2, num=num_methods_in_joint + 2)[1:-1]

        for offset, method in zip(offsets, methods_in_joint):
            if 'Markers' in method:
                continue
            marker_data = np.array(joint_split['Markers']) * 180 / np.pi
            method_data = np.array(joint_split[method]) * 180 / np.pi

            # Extract statistics required for the boxplot
            stats = {
                'med': np.median(np.abs(marker_data - method_data)),
                'q1': np.percentile(np.abs(marker_data - method_data), 30),
                'q3': np.percentile(np.abs(marker_data - method_data), 70),
                'whislo': np.percentile(np.abs(marker_data - method_data), 10),
                'whishi': np.percentile(np.abs(marker_data - method_data), 90),
                'mean': np.mean(np.abs(marker_data - method_data)),
                'fliers': [],  # Empty list since we're not displaying outliers
                'label': method
            }
            boxplot_data.append(stats)
            # Calculate position for this boxplot
            pos = current_pos + offset
            positions.append(pos)
            method_data_list.append(stats)


        current_pos += width + spacing  # Move to the next group position
    labels = ['Flexion', 'Adduction', 'Rotation']
    plot_boxes(boxplot_data, positions, labels, width, spacing, title)

# def plot_all_dofs():
#     joint_splits = split_median_by_joint()
#
#     # Prepare the data for plotting
#     boxplot_data = []
#     positions = []
#     labels = []
#     current_pos = 1  # Starting position for the first boxplot
#     width = 0.8  # Total width allocated for each group of boxplots (per joint)
#     spacing = 0.5  # Spacing between groups
#
#     for i, joint_split in enumerate(joint_splits):
#         # Get the name of this joint
#         if i < skeleton.getNumDofs():
#             dof_name = skeleton.getDofByIndex(i).getName()
#         else:
#             dof_name = 'All Joints'
#
#         method_data_list = []
#
#         # Get the methods present for this joint
#         methods_in_joint = list(joint_split.keys())
#         num_methods_in_joint = len(methods_in_joint)
#
#         # Calculate offsets for methods within the joint group
#         offsets = np.linspace(-width / 2, width / 2, num=num_methods_in_joint + 2)[1:-1]
#
#         for offset, method in zip(offsets, methods_in_joint):
#             if 'Markers' in method:
#                 continue
#             marker_data = np.array(joint_split['Markers']) * 180 / np.pi
#             method_data = np.array(joint_split[method]) * 180 / np.pi
#
#             # Extract statistics required for the boxplot
#             stats = {
#                 'med': np.median(np.abs(marker_data - method_data)),
#                 'q1': np.percentile(np.abs(marker_data - method_data), 30),
#                 'q3': np.percentile(np.abs(marker_data - method_data), 70),
#                 'whislo': np.percentile(np.abs(marker_data - method_data), 10),
#                 'whishi': np.percentile(np.abs(marker_data - method_data), 90),
#                 'mean': np.mean(np.abs(marker_data - method_data)),
#                 'fliers': [],  # Empty list since we're not displaying outliers
#                 'label': method
#             }
#             boxplot_data.append(stats)
#             # Calculate position for this boxplot
#             pos = current_pos + offset
#             positions.append(pos)
#             method_data_list.append(stats)
#
#         labels.append(
#             dof_name.replace('_', ' ').replace(' hip', '').replace(' knee', '').replace(' lumbar', '').replace(' arm',
#                                                                                                                '').replace(
#                 ' elbow', '').replace(' ankle', ''))
#
#         current_pos += width + spacing  # Move to the next group position

def plot_boxes(boxplot_data, positions, labels, width, spacing, title):

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot the boxplots
    meanprops = dict(linestyle='--', linewidth=1.5, color='grey')
    medianprops = dict(linestyle='-', linewidth=2.5, color='black')

    # create a bar chart for each method in each activity
    method_display_names = [
        'Magnetometer Free',
        'MAJIC Zeroth Order',
        'MAJIC First Order',
        'MAJIC Adaptive'
    ]

    methods = list(kinematics[0].keys())

    # Assign colors to each method
    method_colors = {
        method: color for method, color in zip(methods, plt.cm.tab10.colors)
    }

    # Since we have custom positions, we need to use bxp
    for idx, stats in enumerate(boxplot_data):
        bxp_stats = [stats]
        method = stats['label']
        ax.bxp(bxp_stats, positions=[positions[idx]], widths=width / len(methods) * 0.9,
               showfliers=False, showmeans=True, meanline=True,
               boxprops=dict(facecolor=method_colors[method], color=method_colors[method]),
               medianprops=medianprops,
               meanprops=meanprops,
               whiskerprops=dict(color=method_colors[method]),
               capprops=dict(color=method_colors[method]),
               flierprops=dict(markeredgecolor=method_colors[method]),
               patch_artist=True  # This line enables 'facecolor' in boxprops
               )

    # Set x-ticks and labels
    group_positions = []
    current_pos = 1
    for _ in labels:
        group_positions.append(current_pos)
        current_pos += width + spacing

    ax.set_xticks(group_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Create a legend for the methods
    handles = [plt.Line2D([0], [0], color=method_colors[method], lw=10) for method in methods]
    ax.legend(handles, method_display_names, title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title(title)
    ax.set_ylabel('Error (degrees)')
    ax.set_ylim([0, 40])
    plt.tight_layout()
    plt.show()


plot_median_by_joint_axis()
# plot_median_by_activity()
# plot_all_dofs()
# plot_ankle_elbow()




