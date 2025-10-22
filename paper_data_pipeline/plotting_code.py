import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
df = pd.read_csv("/Users/six/projects/work/MAJIC_mocap/data/ODay_Data/all_trial_statistics.csv")

# Filter by subject number
df = df[df["subject"].isin([f"Subject{i:02d}" for i in range(1, 6)])]

# Filter by activity
df = df[df["activity"].str.contains("walking")]

# Filter by method
df = df[df["method"].isin([
    "madgwick", "mag free", "never project"
])]

# Filter by joint
df = df[df["joint"].isin(["hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
                         "knee_angle_l", "hip_flexion_r",
                         "hip_adduction_r", "hip_rotation_r", "knee_angle_r"])]

# Group by method and joint
def combine_squared(series):
    return np.sqrt(np.mean(series ** 2))


grouped = df.groupby(["joint", "method"]).agg({
    "rmse": combine_squared,
    "std": combine_squared
}).rename(columns={"rmse": "combined_rmse", "std": "combined_std"}).reset_index()

# Plot overall summary bar chart
summary_stats = df.groupby("method").agg({
    "rmse": combine_squared,
    "std": combine_squared
}).rename(columns={"rmse": "combined_rmse", "std": "combined_std"})

fig1, ax1 = plt.subplots(figsize=(8, 6))
bars = ax1.bar(
    summary_stats.index,
    summary_stats["combined_rmse"],
    yerr=summary_stats["combined_std"],
    capsize=5
)

for i, bar in enumerate(bars):
    bar.set_color(plt.cm.tab10(i % 10))

ax1.set_ylabel("Combined RMSE")
ax1.set_title("Combined RMSE per Method (±1 Combined STD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot subplot per joint
unique_joints = grouped["joint"].unique()
n_joints = len(unique_joints)
n_cols = 3
n_rows = int(np.ceil(n_joints / n_cols))

fig2, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axs = axs.flatten()

for idx, joint in enumerate(unique_joints):
    ax = axs[idx]
    joint_data = grouped[grouped["joint"] == joint]

    bars = ax.bar(
        joint_data["method"],
        joint_data["combined_rmse"],
        yerr=joint_data["combined_std"],
        capsize=5
    )

    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.tab10(i % 10))

    ax.set_title(joint)
    ax.set_ylabel("Combined RMSE")
    ax.set_xticks(range(len(joint_data["method"])))
    ax.set_xticklabels(joint_data["method"], rotation=45)

# Hide any unused subplots
for j in range(idx + 1, len(axs)):
    fig2.delaxes(axs[j])

fig2.suptitle("Combined RMSE per Method by Joint (±1 Combined STD)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()