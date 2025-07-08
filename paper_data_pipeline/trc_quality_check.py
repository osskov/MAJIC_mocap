import numpy as np
import matplotlib.pyplot as plt

for subject_num in range(11, 12):
    # Format subject number with leading zero
    subject = f"Subject{subject_num:02d}"
    for activity in ['walking', 'complexTasks']:

        trc_file = f"../data/ODay_Data/{subject}/{activity}/MoCap/{activity}.trc"

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
                        marker_k = data[:, second_index:second_index+3]
                        # Calculate the distance between the two markers
                        distances = np.linalg.norm(marker_j - marker_k, axis=1)
                        plt.plot(distances, label=f"{headers[index]} vs {headers[second_index]}")
            plt.xlabel('Sample')
            plt.ylabel('Distance (m)')
            plt.title(f"Distances for {header} IMU data {subject} - {activity}")
            plt.legend()
            plt.show()