import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Read the data from CSV
df = pd.read_csv('../data/Walking_Results/averaged_results.csv')

# Get the list of unique Joint_DOFs and Methods
joints = df['Joint_DOF'].unique()
methods = df['Method'].unique()
num_methods = len(methods)

joints_to_plot = ['Shoulder', 'Elbow']

method_display_names = [
    'Magnetometer Free',
    'MAJIC Zeroth Order',
    'MAJIC First Order',
    'MAJIC Adaptive'
]

# Assign colors to each method
method_colors = {
    method: color for method, color in zip(methods, plt.cm.tab10.colors)
}

# Prepare the data for plotting
boxplot_data = []
positions = []
labels = []
current_pos = 1  # Starting position for the first boxplot
width = 0.8  # Total width allocated for each group of boxplots (per joint)
spacing = 0.5  # Spacing between groups

method_avg_median = [0.0 for _ in methods]
method_mean = [0.0 for _ in methods]

for joint in joints:
    if any(joint_to_plot in joint for joint_to_plot in joints_to_plot):
        joint_data = df[df['Joint_DOF'] == joint]
        method_data_list = []
        method_positions = []

        # Get the methods present for this joint
        methods_in_joint = joint_data['Method'].unique()
        num_methods_in_joint = len(methods_in_joint)
        # Calculate offsets for methods within the joint group
        offsets = np.linspace(-width / 2, width / 2, num=num_methods_in_joint + 2)[1:-1]

        for offset, method in zip(offsets, methods_in_joint):
            method_data = joint_data[joint_data['Method'] == method]
            if method_data.empty:
                continue  # Skip if there's no data for this method and joint

            median = method_data['Median Error (degrees)'].values[0]
            method_index = np.where(methods == method)[0][0]
            method_avg_median[method_index] += median
            # Extract statistics required for the boxplot
            stats = {
                'med': method_data['Median Error (degrees)'].values[0],
                'q1': method_data['30th Percentile Error (degrees)'].values[0],
                'q3': method_data['70th Percentile Error (degrees)'].values[0],
                'whislo': method_data['10th Percentile Error (degrees)'].values[0],
                'whishi': method_data['90th Percentile Error (degrees)'].values[0],
                'mean': method_data['Mean Error (degrees)'].values[0],
                'fliers': [],  # Empty list since we're not displaying outliers
                'label': method
            }
            boxplot_data.append(stats)
            # Calculate position for this boxplot
            pos = current_pos + offset
            positions.append(pos)
            method_data_list.append(stats)

        labels.append(joint.replace('_', ' ').replace(' hip', '').replace(' knee', '').replace(' lumbar', '').replace(' arm', '').replace(' elbow', '').replace(' ankle', ''))
        current_pos += width + spacing  # Move to the next group position

# Create the figure and axis
fig, ax = plt.subplots(figsize=(15, 8))

# Plot the boxplots
meanprops = dict(linestyle='--', linewidth=1.5, color='grey')
medianprops = dict(linestyle='-', linewidth=2.5, color='black')

# Since we have custom positions, we need to use bxp
for idx, stats in enumerate(boxplot_data):
    bxp_stats = [stats]
    method = stats['label']
    ax.bxp(bxp_stats, positions=[positions[idx]], widths=width / num_methods * 0.9,
           showfliers=False, showmeans=True, meanline=True,
           boxprops=dict(facecolor=method_colors[method], color=method_colors[method]),
           medianprops=medianprops,
           meanprops=meanprops,
           whiskerprops=dict(color=method_colors[method]),
           capprops=dict(color=method_colors[method]),
           flierprops=dict(markeredgecolor=method_colors[method]),
           patch_artist=True  # This line enables 'facecolor' in boxprops
           )

# Add a horizontal red dashed line at 5 degrees
# ax.axhline(y=5, color='grey', linestyle='--', linewidth=1.5, label='5-degree Threshold')

method_avg_median = [val / len(joints) for val in method_avg_median]

print(method_avg_median)

# Add a horizontal line for the average median error for each method
for method, avg_median in zip(methods, method_avg_median):
    method_index = np.where(methods == method)[0][0]
    ax.axhline(y=avg_median, color=method_colors[method], linestyle=':', linewidth=1.5, label=f'{method_display_names[method_index]} Average Median')

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

ax.set_title('Error Distribution for All Degrees of Freedom')
ax.set_ylabel('Error (degrees)')
ax.set_ylim([0, 40])
plt.tight_layout()
plt.show()
