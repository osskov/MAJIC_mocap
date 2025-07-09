import numpy as np
from typing import Dict, List, Any, Tuple

class KinematicsTrace:
    """
    A class to handle kinematic traces from .mot files.
    """
    joint_angles: Dict[str, List[np.array]]

    def __init__(self, joint_angles: Dict[str, List[np.array]]):
        """
        Initializes the KinematicsTrace with segment orientations.

        Args:
            joint_angles (Dict[str, List[np.array]]): A dictionary mapping segment names to their orientations.
        """
        self.joint_angles = joint_angles

    def __len__(self) -> int:
        """
        Returns the number of segments in the trace.

        Returns:
            int: The number of segments.
        """
        return len(self.joint_angles['time'])

    def __getitem__(self, key) -> 'KinematicsTrace':
        """
        Returns a KinematicsTrace of orientations depending on the key type.
        Args:
            key (str): The key to access the segment orientations.
        Returns:
            KinematicsTrace: A new KinematicsTrace instance with the specified segment orientations.
        """
        if isinstance(key, str):
            return KinematicsTrace({key: self.joint_angles[key], 'time': self.joint_angles['time']})
        elif isinstance(key, int) or isinstance(key, slice):
            return KinematicsTrace({k: v[key] for k, v in self.joint_angles.items()})
        else:
            raise TypeError("Key must be a string, an integer or a slice.")

    def __sub__(self, other: 'KinematicsTrace'):
        """
        Subtracts the segment orientations of another KinematicsTrace from this one.

        Args:
            other (KinematicsTrace): The other KinematicsTrace to subtract.

        Returns:
            KinematicsTrace: A new KinematicsTrace instance with the result of the subtraction.
        """
        if not isinstance(other, KinematicsTrace):
            raise TypeError("Subtraction is only supported between two KinematicsTrace instances.")
        assert len(self) == len(other), "KinematicsTrace instances must have the same length for subtraction."
        assert set(self.joint_angles.keys()) == set(other.joint_angles.keys()), \
            "KinematicsTrace instances must have the same segment names for subtraction."

        result = {}
        for key in self.joint_angles.keys():
            if key == 'time':
                result[key] = self.joint_angles[key]
            else:
                result[key] = [a - b for a, b in zip(self.joint_angles[key], other.joint_angles[key])]

        return KinematicsTrace(result)

    def __str__(self):
        """
        Returns a string representation of the KinematicsTrace.

        Returns:
            str: A string representation of the KinematicsTrace.
        """
        return f"KinematicsTrace with {len(self)} timestamps and segments: {list(self.joint_angles.keys())}"

    def re_zero_timestamps(self) -> 'KinematicsTrace':
        """
        Re-zeroes the timestamps of the KinematicsTrace.

        Returns:
            KinematicsTrace: A new KinematicsTrace instance with re-zeroed timestamps.
        """
        new_segment_orientations = self.joint_angles.copy()
        new_segment_orientations['time'] = [t - self.joint_angles['time'][0] for t in self.joint_angles['time']]
        return KinematicsTrace(new_segment_orientations)

    @staticmethod
    def load_kinematics_from_mot_file(file_path: str) -> 'KinematicsTrace':
        """
        Loads a .mot file and returns a KinematicsTrace object.

        Args:
            file_path (str): Path to the .mot file.

        Returns:
            KinematicsTrace: An instance of KinematicsTrace containing segment orientations.
        """
        header_info, data_dict = KinematicsTrace._read_mot_file_(file_path)
        return KinematicsTrace(data_dict)

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

    def write_kinematics_to_mot(self, output_path: str):
        """
        Write a KinematicsTrace object to a .mot file in OpenSim format.

        Parameters:
            kinematics (KinematicsTrace): The joint angle data.
            output_path (str): File path to write the .mot file.
        """
        header = [
            "inDegrees=yes",
            f"name={output_path.split('/')[-1]}",
            "DataType=double",
            "version=3",
            "OpenSimVersion=4.5.2-2025-06-25-4987e40",
            "endheader"
        ]

        # Extract time vector
        time_vector = self.joint_angles['time']
        num_frames = len(time_vector)

        # All other joint names (excluding 'time')
        joint_names = [key for key in self.joint_angles.keys() if key != 'time']

        with open(output_path, 'w') as f:
            # Write header
            for line in header:
                f.write(f"{line}\n")

            # Write column names
            f.write("time\t" + "\t".join(joint_names) + "\n")

            # Write data row by row
            for i in range(num_frames):
                row = [f"{time_vector[i]:.17g}"]
                for joint in joint_names:
                    value = self.joint_angles[joint][i]
                    if isinstance(value, (np.ndarray, list)):
                        value = value[0]  # assume it's a single-valued array/list
                    row.append(f"{value:.17g}")
                f.write("\t".join(row) + "\n")

if __name__ == "__main__":
    # Example usage
    kinematics_trace = KinematicsTrace.load_kinematics_from_mot_file("/Users/six/projects/work/MAJIC_mocap/data/ODay_Data/Subject03/walking/Mocap/ikResults/walking_IK.mot")
    print(kinematics_trace.joint_angles.keys())
