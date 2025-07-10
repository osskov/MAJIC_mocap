import pandas as pd

# Generate dummy data
data = []
for i in range(1, 12):
    subject = f'Subject{i:02d}'
    for activity in ['walking', 'complexTasks']:
        for method in ['madgwick', 'markers', 'mag free', 'unprojected', 'never project']:
            for joint in ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']:
                rmse = 0.0  # Fake RMSE value
                std = 0.0 # Fake std
                data.append({
                    'subject': subject,
                    'activity': activity,
                    'joint': joint,
                    'method': method,
                    'rmse': rmse,
                    'std': std
                })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('../data/ODay_Data/all_trial_statistics.csv', index=False)

new_df = pd.read_csv('../data/ODay_Data/all_trial_statistics.csv')

print(new_df.head())

# Show the DataFrame
print(df.head())