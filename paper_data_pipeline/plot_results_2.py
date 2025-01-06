import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple, Dict

# Read the data from CSV
df = pd.read_csv('../data/Walking_Results/averaged_results.csv')

# Get the list of unique Joint_DOFs and Methods
joints_dofs = df['Joint_DOF'].unique()
methods = df['Method'].unique()
num_methods = len(methods)

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

mean_errors: np.ndarray = np.zeros((len(joints_dofs), len(methods)))
std_errors: np.ndarray = np.zeros((len(joints_dofs), len(methods)))
for row in df.itertuples():
    joint_idx = np.where(joints_dofs == row.Joint_DOF)[0][0]
    method_idx = np.where(methods == row.Method)[0][0]
    print(row)
    mean_errors[joint_idx, method_idx] = row[3]
    std_errors[joint_idx, method_idx] = row[5]

for j, joint_name in enumerate(joints_dofs):
    for m, method_name in enumerate(methods):
        print(f"Joint: {joint_name}, Method: {method_name}, Mean Errors: {mean_errors[j, m]}, Std Error: {std_errors[j, m]}")

# Create the bar plot
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(joints_dofs))  # Group positions for joints
width = 0.2  # Width of each bar

# Plot each method as a cluster
for m, method_name in enumerate(methods):
    ax.bar(
        x + m * width, mean_errors[:, m], width, yerr=std_errors[:, m],
        capsize=5, label=method_name
    )

display_names = [name.replace('_', ' ').replace(' hip', '').replace(' knee', '').replace(' lumbar', '').replace(' arm', '') for name in joints_dofs]

# Customize the plot
ax.set_ylabel("Mean Error (degrees)")
ax.set_xlabel("Joints")
ax.set_title("Mean Error and Standard Deviation for All Degrees of Freedom")
ax.set_xticks(x + width * (len(methods) - 1) / 2)
ax.set_xticklabels(display_names, rotation=45, ha='right')
ax.set_ylim([0, 40])
ax.legend()
plt.tight_layout()
plt.show()
