import xml.etree.ElementTree as ET

import nimblephysics as nimble
import numpy as np
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def plot_imu_orientation_3d(brick_name, all_data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"3D Orientation of {brick_name}")

    origin = np.array([0, 0, 0])
    axis_length = 1.0

    colors = ['r', 'g', 'b']  # X, Y, Z axes
    axis_labels = ['X', 'Y', 'Z']

    for i, (label, bricks) in enumerate(all_data.items()):
        if brick_name not in bricks:
            continue

        rot_vec = bricks[brick_name]['rotation']
        if np.allclose(rot_vec, 0):
            continue

        # Rotation is assumed to be XYZ Euler angles in radians
        R_matrix = R.from_euler('xyz', rot_vec).as_matrix()

        # Shifted origin for visibility
        offset = np.array([i * 2.0, 0, 0])

        for j in range(3):
            vec = R_matrix[:, j] * axis_length
            ax.quiver(*offset, *vec, color=colors[j], arrow_length_ratio=0.2)

        # Label each set
        ax.text(*offset, f"{label}", size=10, zorder=1)

    ax.set_xlim([-1, len(all_data)*2])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=30)
    # Make sure the axes are equal
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def parse_osim_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0][1:]
    else:
        namespace = ""

    def ns(tag): return f"{{{namespace}}}{tag}" if namespace else tag

    bricks = {}
    for frame in root.findall(f".//{ns('PhysicalOffsetFrame')}"):
        name = frame.get("name")
        if "imu" not in name.lower():
            continue

        # Look for <socket_parent> and <orientation>
        parent_elem = frame.find(ns("socket_parent"))
        parent = parent_elem.text.strip() if parent_elem is not None else "UNKNOWN"

        orient_elem = frame.find(ns("orientation"))
        orient_text = orient_elem.text.strip() if orient_elem is not None else ""
        rotation = np.fromstring(orient_text, sep=' ') if orient_text else np.zeros(3)

        bricks[name] = {
            "parent": parent,
            "rotation": rotation
        }
    return bricks

def compare_brick_rotations(file_paths):
    all_data = {}
    for path in file_paths:
        # Get a more descriptive name: e.g., 'xsens_model_Rajagopal2015_calibrated.osim'
        parent_dir = os.path.basename(os.path.dirname(path))
        name = path
        all_data[name] = parse_osim_file(path)

    # Collect all brick names
    brick_names = set()
    for bricks in all_data.values():
        brick_names.update(bricks.keys())

    print(f"{'Brick':30} {'Parent':20} " + "  ".join([f"{name[:15]:>25}" for name in all_data.keys()]))

    for brick in sorted(brick_names):
        line = f"{brick:30}"
        parents = []
        rotations = []
        for file in all_data.values():
            if brick in file:
                parents.append(file[brick]['parent'])
                rot = file[brick]['rotation']
                rotations.append(rot)
            else:
                parents.append("N/A")
                rotations.append(None)

        parent_str = parents[0] if all(p == parents[0] for p in parents) else "VARIES"
        line += f"{parent_str:20}"

        for rot in rotations:
            if rot is None:
                line += f"{'MISSING':>25}"
            else:
                rot_str = np.array2string(np.round(rot, 2), separator=",", precision=2)
                line += f"{rot_str:>25}"

        print(line)
    return all_data, brick_names

# Example usage
file_paths = [
    # "data/ODay_Data/Subject03/walking/IMU/xsens/model_Rajagopal2015_calibrated.osim",
    # "data/ODay_Data/Subject03/walking/IMU/mahony/model_Rajagopal2015_calibrated.osim",
    "data/ODay_Data/Subject03/model_Rajagopal2015_calibrated.osim",
    "data/ODay_Data/Subject03/model_Rajagopal2015_calibrated_walking.osim",
    # "data/DO_NOT_MODIFY_AlBorno/Subject03/walking/IMU/madgwick/model_Rajagopal2015_calibrated.osim",
    # "data/ODay_Data/Subject03/model_Rajagopal2015_calibrated.osim",
    # "data/ODay_Data/Subject03/walking/IMU/xsens/model_Rajagopal2015_calibrated.osim",
    # "data/ODay_Data/Subject03/walking/IMU/mahony/model_Rajagopal2015_calibrated_new.osim",
    # "data/ODay_Data/Subject03/walking/IMU/mahony/model_Rajagopal2015_calibrated.osim",
    # "data/ODay_Data/Subject03/walking/IMU/madgwick/model_Rajagopal2015_calibrated_new.osim",
    # "data/ODay_Data/Subject03/walking/IMU/madgwick/model_Rajagopal2015_calibrated.osim",
]

def print_orientation_differences(all_data, brick_names):
    methods = ['xsens', 'mahony', 'madgwick']
    method_files = {method: name for name in all_data for method in methods if method in name and 'new' not in name}

    def get_rotation(method, brick):
        file = method_files.get(method)
        if file is None:
            return None
        return all_data[file].get(brick, {}).get("rotation", None)

    print("\n")
    for brick in sorted(brick_names):
        if not any(m in brick for m in ['imu']):  # focus only on IMUs
            continue

        print(f"Sensor: {brick}")
        xsens = get_rotation('xsens', brick)
        mahony = get_rotation('mahony', brick)
        madgwick = get_rotation('madgwick', brick)

        # Skip if fewer than 2 present
        if xsens is None or mahony is None or madgwick is None:
            print("  [Warning] Missing data for one or more methods.\n")
            continue

        print(f"  Method: xsens to madgwick, Difference: {np.round(nimble.math.matrixToEulerXYZ(nimble.math.eulerXYZToMatrix(xsens).T @ nimble.math.eulerXYZToMatrix(madgwick)), 8)}")
        print(f"  Method: xsens to mahony, Difference: {np.round(nimble.math.matrixToEulerXYZ(nimble.math.eulerXYZToMatrix(xsens).T @ nimble.math.eulerXYZToMatrix(mahony)), 8)}")
        print(f"  Method: mahony to madgwick, Difference: {np.round(nimble.math.matrixToEulerXYZ(nimble.math.eulerXYZToMatrix(mahony).T @ nimble.math.eulerXYZToMatrix(madgwick)), 8)}")

all_data, brick_names = compare_brick_rotations(file_paths)
# print_orientation_differences(all_data, brick_names)
# import matplotlib.pyplot as plt

def plot_brick_rotations(brick_name, all_data):
    # Extract rotation values for the given brick across all files
    labels = []
    rotations = []

    for file_label, brick_dict in all_data.items():
        if brick_name in brick_dict:
            rot = brick_dict[brick_name]['rotation']
            labels.append(file_label)
            rotations.append(rot)

    if not rotations:
        print(f"No data for {brick_name}")
        return

    rotations = np.array(rotations)  # Shape: (n_files, 3)

    x = np.arange(len(labels))  # x axis positions

    plt.figure(figsize=(10, 5))
    plt.plot(x, rotations[:, 0], marker='o', label='X rotation')
    plt.plot(x, rotations[:, 1], marker='s', label='Y rotation')
    plt.plot(x, rotations[:, 2], marker='^', label='Z rotation')

    plt.xticks(x, labels, rotation=45, ha='right')
    plt.title(f"Orientation offset for {brick_name}")
    plt.ylabel("Rotation (rad)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# for brick in sorted(brick_names):
#     plot_imu_orientation_3d(brick, all_data)