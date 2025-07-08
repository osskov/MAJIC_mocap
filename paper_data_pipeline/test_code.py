import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
from io import StringIO

# Load TRC file
subject = 'Subject06'
activity = 'walking'
file_path = f"../data/ODay_Data/{subject}/{activity}/MoCap/{activity}.trc"

with open(file_path, "r") as f:
    lines = f.readlines()

# Parse header
data_start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("Frame#"))
data_lines = lines[data_start_idx:]
column_names = data_lines[0].strip().split('\t')
cleaned_columns = [col if col else f'Unnamed_{i}' for i, col in enumerate(column_names)]
data_str = "".join(data_lines[1:])
data_df = pd.read_csv(StringIO(data_str), sep='\t', names=cleaned_columns)

# Extract markers into a dict of Nx3 arrays
def extract_marker_positions(df):
    markers = {}
    base_names = set(col.rsplit('_', 1)[0] for col in df.columns if any(c in col for c in ['_X', '_Y', '_Z']))
    for base in base_names:
        try:
            markers[base] = df[[f"{base}_X", f"{base}_Y", f"{base}_Z"]].to_numpy(dtype=np.float32)
        except KeyError:
            continue
    return markers

# Plot inter-marker distances for markers with the same base name
def plot_marker_distances_by_name(markers):
    grouped = defaultdict(list)
    for name in markers:
        base = ''.join(filter(str.isalpha, name))
        grouped[base].append(name)

    for base, names in grouped.items():
        if len(names) < 2:
            continue
        plt.figure(figsize=(10, 4))
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                dists = np.linalg.norm(markers[names[i]] - markers[names[j]], axis=1)
                plt.plot(dists, label=f'{names[i]} <-> {names[j]}')
        plt.title(f"Marker Distances for '{base}'")
        plt.xlabel("Frame")
        plt.ylabel("Distance (mm)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Automatically correct marker swaps based on sharp distance jumps
def undo_marker_swaps(markers, threshold=50.0):
    corrected = deepcopy(markers)
    swapped_frames = {}

    grouped = defaultdict(list)
    for name in markers:
        base = ''.join(filter(str.isalpha, name))
        grouped[base].append(name)

    for base, names in grouped.items():
        if len(names) < 2:
            continue
        name1, name2 = names[:2]
        dist = np.linalg.norm(markers[name1] - markers[name2], axis=1)
        jumps = np.where(np.abs(np.diff(dist)) > threshold)[0]
        for j in jumps:
            corrected[name1][j + 1], corrected[name2][j + 1] = (
                corrected[name2][j + 1].copy(),
                corrected[name1][j + 1].copy(),
            )
        if jumps.any():
            swapped_frames[(name1, name2)] = jumps.tolist()
    return corrected, swapped_frames

# Run the pipeline
markers = extract_marker_positions(data_df)
plot_marker_distances_by_name(markers)
corrected, swaps = undo_marker_swaps(markers)
print("Corrected markers:")
for name, positions in corrected.items():
    print(f"{name}: {positions.shape[0]} frames")

print("Swapped frames:")
for pair, frames in swaps.items():
    print(f"{pair}: {frames}")