import os.path
from typing import List, Dict

import numpy as np
from scipy.stats import wilcoxon
from src.IMUSkeleton import IMUSkeleton, RawReading
from src.toolchest.PlateTrial import PlateTrial
from paper_data_pipeline.generate_kinematics_from_precomputed_plates import generate_kinematics_report
import nimblephysics as nimble
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.size': 20,  # Increase font size for all text
    'font.family': 'sans-serif',  # Use a sans-serif font family
    'font.sans-serif': ['Arial'],  # Set Arial as the default sans-serif font
    'axes.titlesize': 22,  # Larger title font size
    'axes.labelsize': 22,  # Larger axis label font size
    'xtick.labelsize': 22,  # Larger x-axis tick font size
    'ytick.labelsize': 22,  # Larger y-axis tick font size
    'legend.fontsize': 16,  # Larger legend font size
    'axes.labelweight': 'bold',  # Make axis labels bold
    'axes.titleweight': 'bold',  # Make titles bold
})
method_display_names = {
    'Mag Free': 'Magnetometer Free',
    'Never Project': 'MAJIC 0th Order',
    'Always Project': 'MAJIC 1st Order',
    'Cascade': 'MAJIC Adaptive'
}
method_colors = {
    method: color for method, color in zip(list(method_display_names.keys()), plt.cm.tab10.colors)
}


def plot_boxes(boxplot_data, positions, labels, width, spacing, title, y_lim=[0, 40], keep_legend=True):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(4, 8))

    # Plot the boxplots
    medianprops = dict(linestyle='-', linewidth=2.5, color='black')

    methods = [stats['label'] for stats in boxplot_data]
    methods = list(dict.fromkeys(methods))  # Remove duplicates

    # Since we have custom positions, we need to use bxp
    for idx, stats in enumerate(boxplot_data):
        bxp_stats = [stats]
        method = stats['label']
        ax.bxp(bxp_stats, positions=[positions[idx]], widths=width,
               showfliers=False,
               boxprops=dict(facecolor=method_colors[method], color=method_colors[method]),
               medianprops=medianprops,
               whiskerprops=dict(color=method_colors[method]),
               capprops=dict(color=method_colors[method]),
               flierprops=dict(markeredgecolor=method_colors[method]),
               patch_artist=True  # This line enables 'facecolor' in boxprops
               )

    # Set x-ticks and labels
    group_positions = []
    current_pos = positions[0]
    for _ in labels:
        group_positions.append(current_pos)
        current_pos += width + spacing

    # ax.set_xticks(group_positions)
    # ax.set_xticklabels(labels, rotation=45, ha='right')

    if keep_legend:
        # Create a legend for the methods
        handles = [plt.Line2D([0], [0], color=method_colors[method], lw=10) for method in methods]
        ax.legend(handles, [method_display_names[method] for method in methods], title='Filter',
                   loc='upper right')

    ax.set_title(title, pad=20)
    ax.set_ylabel('Error (degrees)')
    ax.set_ylim(y_lim)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# Load the precomputed joint angle errors for each subject
subjects_errors: List[Dict] = []

# Collect kinematics
for subject_number in range(1, 4):
    subject_error = generate_kinematics_report(
        b3d_path=f"../data/S{subject_number}_IMU_Data/S{subject_number}.b3d",
        model_path=f"../data/S{subject_number}_IMU_Data/SimplifiedIMUModel.osim",
        cached_history_path=f"../data/S{subject_number}_IMU_Data/Walking/cached_skeleton_history.npz",
        output_path=f"../data/S{subject_number}_IMU_Data/Walking/",
        output_csv=f"../data/Walking_Results/S{subject_number}_Walking_joint_angles.csv"
    )
    subject_error = {joint_name: {method: method_data[5000:48000] for method, method_data in joint_data.items()} for
                     joint_name, joint_data in subject_error.items() if '_2' not in joint_name}

    subjects_errors.append(subject_error)

def generate_all_joints_box_plot():
    # From just this data, we can create our first figure: a boxplot of the errors for each method
    # First we need to group the errors by method
    method_errors = {method: np.array([]) for method in ['Mag Free', 'Never Project', 'Always Project', 'Cascade']}
    for subject_kinematics in subjects_errors:
        for joint_name, methods_data in subject_kinematics.items():
            for method, angle_errors in methods_data.items():
                angle_errors = np.array(angle_errors).flatten()
                method_errors[method] = np.concatenate([method_errors[method], angle_errors])

    print('Errors collected by method.')

    # Prepare the data for plotting
    boxplot_data = []
    positions = []
    labels = []
    current_pos = 1  # Starting position for the first boxplot
    width = 1  # Total width allocated for each group of boxplots
    spacing = 0  # Spacing between groups
    title = 'Distribution of All Joint Angle Errors for Each Sensor Fusion Filter'
    for method_name, method_error in method_errors.items():
        method_error = method_error * 180 / np.pi  # Convert to degrees

        plot_quantity = method_error

        # Remove outliers based on 1.5 * IQR rule
        q1 = np.percentile(plot_quantity, 25)
        q3 = np.percentile(plot_quantity, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        plot_quantity = plot_quantity[(plot_quantity >= lower_bound) & (plot_quantity <= upper_bound)]
        outliers = plot_quantity[(plot_quantity < lower_bound) | (plot_quantity > upper_bound)]

        # Print number of data points considered outliers
        print(f'Number of outliers for {method_name}: {len(method_error) - len(plot_quantity)}')

        stats = {
            'med': np.median(plot_quantity),
            'q1': np.percentile(plot_quantity, 30),
            'q3': np.percentile(plot_quantity, 70),
            'whislo': np.min([lower_bound, np.min(plot_quantity)]),  # Ensure that the whiskers don't go below 0
            'whishi': np.max([upper_bound, np.max(plot_quantity)]),
            'fliers': outliers,
            'label': method_name
        }
        boxplot_data.append(stats)
        # Calculate position for this boxplot
        pos = current_pos
        positions.append(pos)

        labels.append(
            method_display_names[method_name] + '\n(Median Error: {:.2f} degrees)'.format(np.median(plot_quantity)))
        current_pos += width + spacing  # Move to the next group position

        print(f'Errors calculated for {method_name}.')
        print(f'Median: {stats["med"]}, Q1: {stats["q1"]}, Q3: {stats["q3"]}')
        print(f'RMSE: {np.sqrt(np.mean(plot_quantity ** 2))}')

        # Perform Wilcoxon signed-rank test against the Mag Free method
        if method_name == 'Mag Free':
            continue
        _, p_value = wilcoxon(method_error, method_errors['Mag Free'])
        print(f'Wilcoxon signed-rank test p-value for {method_name} vs. Mag Free: {p_value}')
        if p_value < 0.05:
            print(f'{method_name} is significantly different from Mag Free.')
        else:
            print(f'{method_name} is not significantly different from Mag Free.')

    plot_boxes(boxplot_data, positions, labels, width, spacing, title, y_lim=[-20, 20], keep_legend=False)

def generate_abs_val_all_joints_box_plot():
    # From just this data, we can create our first figure: a boxplot of the errors for each method
    # First we need to group the errors by method
    method_errors = {method: np.array([]) for method in ['Mag Free', 'Never Project', 'Always Project']}
    for subject_kinematics in subjects_errors:
        for joint_name, methods_data in subject_kinematics.items():
            if joint_name == 'All Joints':
                continue
            for method, angle_errors in methods_data.items():
                if method == 'Cascade':
                    continue
                angle_errors = np.array(np.abs(angle_errors)).flatten()
                method_errors[method] = np.concatenate([method_errors[method], angle_errors])

    print('Errors collected by method.')

    # Prepare the data for plotting
    boxplot_data = []
    positions = []
    labels = []
    current_pos = 0.5  # Starting position for the first boxplot
    width = 0.8  # Total width allocated for each group of boxplots
    spacing = 0.4  # Spacing between groups
    title = 'Angle Error Distribution \nPer Sensor Fusion Filter'
    for method_name, method_error in method_errors.items():
        method_error = method_error * 180 / np.pi  # Convert to degrees

        plot_quantity = np.abs(method_error)

        # Remove outliers based on 1.5 * IQR rule
        q1 = np.percentile(plot_quantity, 25)
        q3 = np.percentile(plot_quantity, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr if q1 - 1.5 * iqr > 0 else min(plot_quantity)
        upper_bound = q3 + 1.5 * iqr

        plot_quantity = plot_quantity[(plot_quantity >= lower_bound) & (plot_quantity <= upper_bound)]
        outliers = plot_quantity[(plot_quantity < lower_bound) | (plot_quantity > upper_bound)]

        # Print number of data points considered outliers
        print(f'Number of outliers for {method_name}: {len(method_error) - len(plot_quantity)}')

        stats = {
            'med': np.median(plot_quantity),
            'q1': np.percentile(plot_quantity, 30),
            'q3': np.percentile(plot_quantity, 70),
            'whislo': np.max([lower_bound, np.min(plot_quantity)]),  # Ensure that the whiskers don't go below 0
            'whishi': np.max([upper_bound, np.max(plot_quantity)]),
            'fliers': outliers,
            'label': method_name
        }

        boxplot_data.append(stats)
        # Calculate position for this boxplot
        pos = current_pos
        positions.append(pos)

        labels.append(
            method_display_names[method_name]) #+ '\n(Median Error: {:.2f} degrees)'.format(np.median(method_error)))
        current_pos += width + spacing  # Move to the next group position

        print(f'Errors calculated for {method_name}.')
        print(f'Median: {stats["med"]}, Q1: {stats["q1"]}, Q3: {stats["q3"]}')
        print(f'RMSE: {np.sqrt(np.mean(plot_quantity ** 2))}')

        # Perform Wilcoxon signed-rank test against the Mag Free method
        if method_name == 'Mag Free':
            continue
        _, p_value = wilcoxon(method_error, method_errors['Mag Free'])
        print(f'Wilcoxon signed-rank test p-value for {method_name} vs. Mag Free: {p_value}')
        if p_value < 0.05:
            print(f'{method_name} is significantly different from Mag Free.')
        else:
            print(f'{method_name} is not significantly different from Mag Free.')

    plot_boxes(boxplot_data, positions, labels, width, spacing, title, y_lim=[0, 22], keep_legend=True)


def generate_per_axis_box_plot():
    # First we need to split our data by axis (flexion, abduction, and rotation)
    axis_errors = [{method: [] for method in ['Mag Free', 'Never Project', 'Always Project', 'Cascade']} for _ in
                   range(3)]
    axis_display_names = ['Flexion', 'Adduction', 'Rotation']
    # method_positions = {'Mag Free': 1, 'Never Project': 2, 'Always Project': 3, 'Cascade': 4}

    for subject_kinematics in subjects_errors:
        for joint_name, methods_data in subject_kinematics.items():
            if joint_name == 'All Joints':
                continue
            for method, angle_errors in methods_data.items():
                for axis_idx in range(3):
                    axis_error = [error[axis_idx] for error in angle_errors]
                    axis_errors[axis_idx][method].extend(axis_error)

    print('Errors collected by axis.')

    # Prepare the data for plotting
    boxplot_data = []
    positions = []
    width = 1  # Total width allocated for each group of boxplots (per joint)
    spacing = -0.1  # Spacing between groups
    title = 'Joint Angle Error Distribution per Axis'
    y_lim_max = 0
    current_pos = 0.2  # Starting position for the first boxplot
    for i, axis_error_dict in enumerate(axis_errors):
        method_data_list = []

        # Get the methods present for this joint
        num_axis_in_method = 3
        # Calculate offsets for methods within the joint group
        offsets = np.linspace(-width / 2, width / 2, num=num_axis_in_method + 2)[1:-1]
        for offset, method in zip(offsets, ['Mag Free', 'Never Project',  'Cascade']):
            plot_quantity = np.abs(np.array(axis_error_dict[method])) * 180 / np.pi  # Convert to degrees

            # Remove outliers based on 1.5 * IQR rule
            q1 = np.percentile(plot_quantity, 25)
            q3 = np.percentile(plot_quantity, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr if q1 - 1.5 * iqr > 0 else min(plot_quantity)
            upper_bound = q3 + 1.5 * iqr if q3 + 1.5 * iqr < 180 else max(plot_quantity)
            if upper_bound > y_lim_max:
                y_lim_max = upper_bound + 5
            plot_quantity = plot_quantity[(plot_quantity >= lower_bound) & (plot_quantity <= upper_bound)]
            outliers = plot_quantity[(plot_quantity < lower_bound) | (plot_quantity > upper_bound)]

            # Print number of data points considered outliers
            print(
                f'Number of outliers for {method} in {axis_display_names[i]}: {len(axis_error_dict[method]) - len(plot_quantity)}')

            # Extract statistics required for the boxplot
            stats = {
                'med': np.median(plot_quantity),
                'q1': q1,
                'q3': q3,
                'whislo': np.min([lower_bound, np.min(plot_quantity)]),
                'whishi': np.max([upper_bound, np.max(plot_quantity)]),
                'mean': np.mean(plot_quantity),
                'fliers': outliers,
                'label': method
            }
            boxplot_data.append(stats)
            # Calculate position for this boxplot
            pos = current_pos + offset
            positions.append(pos)
            method_data_list.append(stats)

            print(f'Errors calculated for {axis_display_names[i]} {method}.')
            print(f'Median: {stats["med"]}, Q1: {stats["q1"]}, Q3: {stats["q3"]}')
            print(f'RMSE: {np.sqrt(np.mean(plot_quantity ** 2))}')
        current_pos += width + spacing  # Move to the next group position
    # labels = ['Mag Free', 'Never Project',  'Cascade']
    plot_boxes(boxplot_data, positions, axis_display_names, width, spacing, title, y_lim=[0, y_lim_max])


def generate_per_dof_box_plot(title, joint_filter=None, axis_filter=None, method_filter=None):
    # First we need to split our data by dof
    dof_errors = {}
    axis_display_names = ['Flexion', 'Adduction', 'Rotation']

    for subject_kinematics in subjects_errors:
        for joint_name, methods_data in subject_kinematics.items():
            if joint_filter and joint_name not in joint_filter:
                continue
            for method, angle_errors in methods_data.items():
                if method_filter and method not in method_filter:
                    continue
                for axis_idx in range(3):
                    if axis_filter and axis_display_names[axis_idx] not in axis_filter:
                        continue
                    dof_name = joint_name.replace('_', ' ') + ' ' + axis_display_names[axis_idx]
                    if dof_name not in dof_errors.keys():
                        dof_errors[dof_name] = {method: [] for method in
                                                ['Mag Free', 'Never Project', 'Always Project', 'Cascade']}
                    axis_error = [error[axis_idx] for error in angle_errors]
                    dof_errors[dof_name][method].extend(axis_error)

    print('Errors collected by axis.')

    # Prepare the data for plotting
    boxplot_data = []
    positions = []
    labels = []
    current_pos = 1  # Starting position for the first boxplot
    width = 1  # Total width allocated for each group of boxplots (per joint)
    spacing = 0  # Spacing between groups
    y_lim_max = 0
    last_dof_name = ''
    for dof_name, dof_error_dict in dof_errors.items():
        method_data_list = []

        if last_dof_name and (last_dof_name.split(' ')[0] != dof_name.split(' ')[0] or
                              ('Lumbar' in dof_name and last_dof_name.split(' ')[1] != dof_name.split(' ')[1]) or
                                ('Shoulder' in dof_name and last_dof_name.split(' ')[1] != dof_name.split(' ')[1])):
            current_pos += 1
            labels.append('')  # Add a blank label to separate the groups

        # Get the methods present for this joint
        num_methods_in_dof = len(method_filter) if method_filter else 4

        # Calculate offsets for methods within the joint group
        offsets = np.linspace(-width / 2, width / 2, num=num_methods_in_dof + 2)[1:-1]

        for offset, method in zip(offsets,
                                  method_filter if method_filter else ['Mag Free', 'Never Project', 'Always Project',
                                                                       'Cascade']):
            plot_quantity = np.abs(np.array(dof_error_dict[method])) * 180 / np.pi  # Convert to degrees

            # Remove outliers based on 1.5 * IQR rule
            q1 = np.percentile(plot_quantity, 25)
            q3 = np.percentile(plot_quantity, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr if q1 - 1.5 * iqr > 0 else min(plot_quantity)
            upper_bound = q3 + 1.5 * iqr
            if upper_bound > y_lim_max:
                y_lim_max = upper_bound + 5
            plot_quantity = plot_quantity[(plot_quantity >= lower_bound) & (plot_quantity <= upper_bound)]
            outliers = plot_quantity[(plot_quantity < lower_bound) | (plot_quantity > upper_bound)]

            # Print number of data points considered outliers
            print(
                f'Number of outliers for {method} in {dof_name}: {len(dof_error_dict[method]) - len(plot_quantity)}')

            # Extract statistics required for the boxplot
            stats = {
                'med': np.median(plot_quantity),
                'q1': q1,
                'q3': q3,
                'whislo': np.min([lower_bound, np.min(plot_quantity)]),
                'whishi': np.max([upper_bound, np.max(plot_quantity)]),
                'mean': np.mean(plot_quantity),
                'fliers': outliers,
                'label': method
            }
            boxplot_data.append(stats)
            # Calculate position for this boxplot
            pos = current_pos + offset
            positions.append(pos)
            method_data_list.append(stats)

            print(f'Errors calculated for {method}.')
            print(f'Median: {stats["med"]}, Q1: {stats["q1"]}, Q3: {stats["q3"]}')
            print(f'RMSE: {np.sqrt(np.mean(plot_quantity ** 2))}')
        labels.append(dof_name)
        current_pos += width + spacing  # Move to the next group position
        last_dof_name = dof_name
    plot_boxes(boxplot_data, positions, labels, width, spacing, title, y_lim=[0, y_lim_max])


def generate_all_dof_box_plot():
    generate_per_dof_box_plot('Joint Angle Error Distribution per Degree of Freedom')


def generate_ankle_lumbar_box_plot():
    generate_per_dof_box_plot('Ankle and Lumbar Joint Angle Error Distribution per Degree of Freedom',
                              ['Ankle', 'Lumbar_1', 'Lumbar_2'], method_filter=['Mag Free', 'Cascade'])


def generate_ankle_shoulder_box_plot():
    generate_per_dof_box_plot('Ankle and Shoulder Joint Angle Error Distribution per Degree of Freedom',
                              ['Ankle', 'Shoulder_1', 'Shoulder_2'], method_filter=['Never Project', 'Always Project', 'Cascade'])


def generate_lower_body_box_plot():
    generate_per_dof_box_plot('Lower Body Joint Angle Error Distribution per Degree of Freedom',
                              joint_filter=['Ankle', 'Knee', 'Hip'])


def generate_lumbar_box_plot():
    generate_per_dof_box_plot('Lumbar Joint Angle Error Distribution per Degree of Freedom',
                              joint_filter=['Lumbar_1', 'Lumbar_2'])


def generate_upper_body_box_plot():
    generate_per_dof_box_plot('Upper Body Joint Angle Error Distribution per Degree of Freedom',
                              joint_filter=['Shoulder_1', 'Shoulder_2', 'Elbow'])


def generate_per_activity_box_plot():
    activity_order = ['Walking', 'Running', 'Stairs and Side Stepping', 'Standing and Sitting',
                      'Stairs and Side Stepping']
    activity_timestamps = [
        [0, 2300, 3400, 5000, 8100, 10100, 11900, 13000, 14400, 18000, 20200, 22400, 23600, 25000, 28800, 30700, 32800,
         34000, 35200, 38500, 41000, 43500, 44900, 46300, 48000],
        [0, 2400, 3500, 4600, 7000, 9300, 10800, 11800, 14000, 15300, 17000, 18700, 19800, 21500, 22400, 24100, 25500,
         27600, 30200, 32100, 33800, 36000, 37600, 38900, 40800, 42900, 44800, 46000, 47400, 48000],
        [0, 2400, 4000, 5700, 8800, 12400, 13800, 15000, 16300, 18800, 21000, 22700, 23700, 25500, 30000, 32000, 33700,
         35100, 36500, 40700, 43600, 45800, 46900, 48000]]

    walking_splits = {method: [] for method in subjects_errors[0]['Hip'].keys()}
    running_splits = {method: [] for method in subjects_errors[0]['Hip'].keys()}
    stairs_splits = {method: [] for method in subjects_errors[0]['Hip'].keys()}
    standing_splits = {method: [] for method in subjects_errors[0]['Hip'].keys()}

    for subj_num, subject_errors in enumerate(subjects_errors):
        for joint_name, methods_errors in subject_error.items():
            for method, joint_angles in methods_errors.items():
                joint_angles = np.array(joint_angles)[:, 2]
                joint_angles = np.array(joint_angles).flatten() * 180 / np.pi  # Convert to degrees
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

        activity_splits = [standing_splits, stairs_splits, walking_splits, running_splits]
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

                plot_quantity = activity_split[method]
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


def generate_joint_error_vs_time_plot_activity_split_plot():
    activity_order = ['Walking', 'Running', 'Stairs and Side Stepping', 'Standing and Sitting',
                      'Stairs and Side Stepping']
    activity_timestamps = [
        [0, 2300, 3400, 5000, 8100, 10100, 11900, 13000, 14400, 18000, 20200, 22400, 23600, 25000, 28800, 30700,
         32800, 34000, 35200, 38500, 41000, 43500, 44900, 46300, 48000],
        [0, 2400, 3500, 4600, 7000, 9300, 10800, 11800, 14000, 15300, 17000, 18700, 19800, 21500, 22400, 24100,
         25500, 27600, 30200, 32100, 33800, 36000, 37600, 38900, 40800, 42900, 44800, 46000, 47400, 48000],
        [0, 2400, 4000, 5700, 8800, 12400, 13800, 15000, 16300, 18800, 21000, 22700, 23700, 25500, 30000, 32000,
         33700, 35100, 36500, 40700, 43600, 45800, 46900, 48000]]

    for subject_number, subject_error in enumerate(subjects_errors):
        if subject_number != 0:
            continue
        for joint_name, methods_data in subject_error.items():
            if joint_name != 'Hip':
                continue
            for dof, dof_name in enumerate(['Flexion', 'Adduction', 'Rotation']):
                timestamps = [time * 0.005 for time in range(len(np.array(methods_data['Mag Free'])[:, 0]))]
                subject_activity_timestamps = activity_timestamps[subject_number]
                legend_patches = []
                for index in range(len(subject_activity_timestamps) - 1):
                    start = subject_activity_timestamps[index]
                    end = subject_activity_timestamps[index + 1]
                    activity = activity_order[index % 5]

                    if activity == 'Walking':
                        color = 'black'
                        alpha = 0.25
                    elif activity == 'Running':
                        color = 'black'
                        alpha = 0.4
                    elif activity == 'Stairs and Side Stepping':
                        color = 'black'
                        alpha = 0.15
                    elif activity == 'Standing and Sitting':
                        color = 'green'
                        alpha = 0.05

                    if end <= 5000:
                        continue
                    if start < 5000:
                        start = 5000
                    plt.axvspan((start - 5000) * 0.005, (end - 5000) * 0.005, color=color, alpha=alpha, linewidth=0)
                    # Add a patch for this activity to the legend if not already added
                    if activity not in [patch.get_label() for patch in legend_patches]:
                        legend_patches.append(mpatches.Patch(color=color, alpha=alpha, label=activity))

                for method in methods_data.keys():
                    plt.plot(timestamps, np.array(methods_data[method])[:, dof] * 180.0 / np.pi,
                             label=method)
                    plt.title(f"Subject {subject_number + 1} {joint_name} {dof_name} Joint Error over Time per Filter")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Angle Error (degrees)")

                # Add all the patches to the legend
                plt.legend(handles=legend_patches + plt.gca().get_legend_handles_labels()[0], loc='upper left', bbox_to_anchor=(1, 1))
                plt.show()


if True:
    # FIGURE 1
    print('Generating Figure 1')
    generate_abs_val_all_joints_box_plot()

    # # FIGURE 2
    # print('Generating Figure 2')
    # generate_per_axis_box_plot()
    #
    # # FIGURE 3
    # print('Generating Figure 3')
    # generate_ankle_lumbar_box_plot()
    #
    # # FIGURE 4
    # print('Generating Figure 4')
    # generate_ankle_shoulder_box_plot()
    #
    # # FIGURE 5
    # print('Generating Figure 5')
    # generate_joint_error_vs_time_plot_activity_split_plot()
    #
    # print('Generating Supplementary Figures')
    # # SUPP 1: Error Symmetry
    # generate_all_joints_box_plot()
    #
    # # SUPP 2: Per Joint Break Down
    # generate_upper_body_box_plot()
    # generate_lumbar_box_plot()
    # generate_lower_body_box_plot()

print('Done')
