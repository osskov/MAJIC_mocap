import os
from .IMUTrace import IMUTrace
from typing import List, Tuple
import numpy as np
from .finite_difference_utils import central_difference
from .gyro_utils import finite_difference_rotations
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp
from typing import Dict
import xml.etree.ElementTree as ET
import pandas as pd

class WorldTrace:
    """
    This class contains a trace of a world frame over time. Optionally, this can attach an IMUTrace and manipulate it.
    Or, it can generate a synthetic trace by finite differencing the world frames over time.
    """

    def __init__(self, timestamps: np.ndarray, positions: List[np.ndarray], rotations: List[np.ndarray]):
        self.timestamps = timestamps
        self.positions = positions
        self.rotations = rotations

    def __len__(self):
        """
        Returns the number of samples in the WorldTrace. This allows us to call len(trace) on a WorldTrace instance.
        """
        return len(self.timestamps)

    def __sub__(self, other: 'WorldTrace') -> 'WorldTrace':
        """
        Allows us to subtract two WorldTrace instances. This will subtract the positions and rotations of the two traces.
        """
        if len(self) != len(other):
            raise ValueError(f"WorldTraces must have the same length to subtract them. Got self {len(self)} and other {len(other)}.")
        assert np.array_equal(self.timestamps[0],
                              other.timestamps[0]), "WorldTraces must have the same start time to subtract them."
        return WorldTrace(self.timestamps, [np.array(pos1) - np.array(pos2) for pos1, pos2 in zip(self.positions, other.positions)],
                          [rot1 @ rot2.T for rot1, rot2 in zip(self.rotations, other.rotations)])

    def __getitem__(self, key) -> 'WorldTrace':
        """
        Allows us to use the square bracket notation to access the WorldTrace instance. This allows us to slice the
        WorldTrace instance and access ranges of items with `sub_trace = trace[1:4]`. If we pass an integer, we can
        return the corresponding item as a length 1 trace with `sub_trace = trace[2]` and `len(sub_trace) == 1`.
        """
        if isinstance(key, slice):
            # If key is a slice object, return a new WorldTrace instance with the sliced items
            return WorldTrace(self.timestamps[key], self.positions[key], self.rotations[key])
        else:
            # If key is an integer, return the corresponding item as a solo list
            return WorldTrace(np.array([self.timestamps[key]]), [self.positions[key]], [self.rotations[key]])

    def __eq__(self, other):
        """
        Allows us to compare two WorldTrace instances for equality. This will return True if the timestamps, positions, and
        rotations are all _exactly_ equal.
        """
        if not isinstance(other, WorldTrace):
            return False
        if len(self) != len(other):
            return False
        if self.positions[0].shape != other.positions[0].shape or self.rotations[0].shape != other.rotations[0].shape:
            return False
        return ((self.timestamps == other.timestamps).all() and
                all(np.all(self.positions[i] == other.positions[i]) for i in range(len(self.positions))) and
                all(np.all(self.rotations[i] == other.rotations[i]) for i in range(len(self.rotations))))

    def transform(self, rotate: np.ndarray = np.eye(3), translate: np.ndarray = np.zeros(3)) -> 'WorldTrace':
        """
        This function transforms the WorldTrace by rotating and translating the positions and rotations.
        """
        return WorldTrace(self.timestamps,
                          [rotate @ pos + translate for pos in self.positions],
                          [rotate @ rot for rot in self.rotations])

    def allclose(self, other, atol=1e-6):
        """
        Allows us to compare two WorldTrace instances for approximate equality. This will return True if the timestamps,
        positions, and rotations are all approximately equal within the specified tolerance.
        """
        if not isinstance(other, WorldTrace):
            return False
        if len(self) != len(other):
            return False
        if self.positions[0].shape != other.positions[0].shape or self.rotations[0].shape != other.rotations[0].shape:
            return False
        return (np.allclose(self.timestamps, other.timestamps, atol=atol) and
                all(np.allclose(self.positions[i], other.positions[i], atol=atol) for i in
                    range(len(self.positions))) and
                all(np.allclose(self.rotations[i], other.rotations[i], atol=atol) for i in range(len(self.rotations))))
    
    def copy(self) -> 'WorldTrace':
        """
        Returns a deep copy of the WorldTrace object.
        """
        return WorldTrace(
            self.timestamps.copy(),
            [pos.copy() for pos in self.positions],
            [rot.copy() for rot in self.rotations]
        )
    
    def resample(self, new_frequency: float) -> 'WorldTrace':
        """
        This function resamples the WorldTrace to a new, specified frequency using linear interpolation for
        positions and spherical linear interpolation (slerp) for rotations.

        Args:
            new_frequency: The desired sampling frequency in Hz.

        Returns:
            A new WorldTrace instance resampled to the new frequency.
        """
        if new_frequency <= 0:
            raise ValueError("New frequency must be a positive number.")

        original_timestamps = self.timestamps
        if len(original_timestamps) < 2:
            return self.copy() # Cannot resample a trace with 0 or 1 samples

        # 1. Determine the new timestamps, prioritizing a perfect time step (dt)
        start_time = original_timestamps[0]
        last_original_time = original_timestamps[-1]
        new_dt = 1.0 / new_frequency
        
        # Define a small tolerance (epsilon) based on the new time step. 
        # This is used to make the 'stop' value in np.arange inclusive.
        # We use a fraction of the new_dt to ensure precision.
        epsilon = new_dt * 1e-6 
        
        # Calculate the theoretical end of the uniform grid. 
        # This is the last point generated by the perfect steps.
        time_duration = last_original_time - start_time
        num_intervals = np.round(time_duration / new_dt)
        theoretical_last_step = start_time + num_intervals * new_dt
        
        # np.arange creates a perfectly uniform time series
        new_timestamps = np.arange(start=start_time, 
                                   stop=theoretical_last_step, 
                                   step=new_dt)

        # 2. Interpolate Positions (Linear Interpolation)
        # Convert list of 3-vectors to a (N, 3) NumPy array for interpolation
        original_positions_np = np.array(self.positions)

        # Create interpolation functions for each dimension (x, y, z)
        interp_x = np.interp(new_timestamps, original_timestamps, original_positions_np[:, 0])
        interp_y = np.interp(new_timestamps, original_timestamps, original_positions_np[:, 1])
        interp_z = np.interp(new_timestamps, original_timestamps, original_positions_np[:, 2])

        # Combine interpolated axes back into a list of 3-vectors
        new_positions = [np.array([x, y, z]) for x, y, z in zip(interp_x, interp_y, interp_z)]

        # 3. Interpolate Rotations (SLERP via Quaternions)
        # Convert 3x3 rotation matrices to quaternions (x, y, z, w)
        original_quats = Rotation.from_matrix(self.rotations).as_quat()

        # Create a Rotation object for interpolation
        # from_quat creates a set of rotations
        original_rotations = Rotation.from_quat(original_quats)

        # Slerp the rotations to the new timestamps
        # The 'Slerp' method internally uses the original timestamps for interpolation
        slerp = Slerp(original_timestamps, original_rotations)
        new_rotations_obj = slerp(new_timestamps)

        # Convert interpolated Rotation objects back to a list of 3x3 matrices
        new_rotations = [rot.copy() for rot in new_rotations_obj.as_matrix()]

        # 4. Return the new WorldTrace
        return WorldTrace(new_timestamps, new_positions, new_rotations)

    def finite_difference_world_frame_accelerations(self, acc_from_gravity: np.ndarray = np.zeros(3)) -> List[
        np.ndarray]:
        """
        This function computes the acceleration of the world frame by finite differencing the positions.
        """
        acc_axis = []
        for axis in range(3):
            vel_axis = central_difference(np.array([pos[axis] for pos in self.positions]), self.timestamps)
            acc_axis.append(central_difference(np.array(vel_axis), self.timestamps))
        # Convert back to a list of 3-vectors
        return [np.array([acc_axis[0][i], acc_axis[1][i], acc_axis[2][i]]) + acc_from_gravity for i in
                range(len(acc_axis[0]))]

    def calculate_imu_trace(self,
                            acc_from_gravity: np.ndarray = np.zeros(3),
                            magnetic_field: np.ndarray = np.zeros(3),
                            skip_lin_acc=False) -> IMUTrace:
        """
        This function computes the IMU trace from the world trace by finite differencing the positions and rotations.
        """
        if not skip_lin_acc:
            world_acc = self.finite_difference_world_frame_accelerations(acc_from_gravity)
            local_acc = [rot.T @ acc for rot, acc in zip(self.rotations, world_acc)]
        else:
            local_acc = [rot.T @ acc_from_gravity for rot in self.rotations]
        assert isinstance(magnetic_field, np.ndarray)
        local_mag = [rot.T @ magnetic_field for rot in self.rotations]
        local_gyros = finite_difference_rotations(self.rotations, self.timestamps)
        return IMUTrace(self.timestamps, local_gyros, local_acc, local_mag)

    def re_zero_timestamps(self) -> 'WorldTrace':
        """
        Start timestamps at 0
        """
        return WorldTrace(self.timestamps - self.timestamps[0], self.positions, self.rotations)

    @staticmethod
    def load_from_trc_file(trc_file: str, max_trc_timestamp=-1.0) -> dict[str, 'WorldTrace']:
        """
        This function loads a list of WorldTrace instances from a folder. Each file in the folder should contain a
        WorldTrace instance saved with numpy.savez.
        """
        if trc_file is None or os.path.isfile(trc_file) is False or not trc_file.endswith('.trc'):
            raise FileNotFoundError("No TRC file found.")

        with open(trc_file, 'r') as file:
            lines = file.readlines()

        headers = lines[3].strip().split('\t')
        imu_headers = [header for header in headers if ('_O' in header or '_3' in header)]
        data = [line.strip().split('\t') for line in lines[6:]]  # Skip empty line and read the data
        data = np.array(data, dtype=float)
        timestamps = data[:, 1]

        if max_trc_timestamp > 0.0:
            first_exceeding_timestamp = np.argmax(timestamps > max_trc_timestamp)
            if first_exceeding_timestamp > 0:
                print(f"Trimming TRC data to {first_exceeding_timestamp} samples in order to stay below the cutoff timestamp of {max_trc_timestamp}")
                data = data[:first_exceeding_timestamp]
                timestamps = timestamps[:first_exceeding_timestamp]

        # assert each timestamp is unique
        assert len(np.unique(timestamps)) == len(timestamps), "Timestamps must be unique."

        world_traces = {}

        for i, imu_o_name in enumerate(imu_headers):
            if '_O' in imu_o_name:
                imu_o_idx = headers.index(imu_o_name)
                imu_x_idx = headers.index(imu_o_name.replace('_O', '_X'))
                imu_y_idx = headers.index(imu_o_name.replace('_O', '_Y'))
                imu_d_idx = headers.index(imu_o_name.replace('_O', '_D'))
            else:
                assert '_3' in imu_o_name
                imu_o_idx = headers.index(imu_o_name)
                imu_x_idx = headers.index(imu_o_name.replace('_3', '_2'))
                imu_y_idx = headers.index(imu_o_name.replace('_3', '_4'))
                imu_d_idx = headers.index(imu_o_name.replace('_3', '_1'))

            # Extracting marker locations
            imu_o_loc = data[:, imu_o_idx: imu_o_idx + 3]
            imu_x_loc = data[:, imu_x_idx: imu_x_idx + 3]
            imu_y_loc = data[:, imu_y_idx: imu_y_idx + 3]
            imu_d_loc = data[:, imu_d_idx: imu_d_idx + 3]

            if "Foot" in imu_o_name:
                imu_o_copy = imu_o_loc.copy()
                imu_o_loc = imu_d_loc
                imu_d_loc = imu_o_copy

                if "R" in imu_o_name:
                    imu_x_copy = imu_x_loc.copy()
                    imu_x_loc = imu_d_loc
                    imu_d_loc = imu_x_copy

                    imu_y_copy = imu_y_loc.copy()
                    imu_y_loc = imu_o_loc
                    imu_o_loc = imu_y_copy

            if "L" in imu_o_name:
                imu_y_copy = imu_y_loc.copy()
                imu_y_loc = imu_d_loc
                imu_d_loc = imu_y_copy

                imu_x_copy = imu_x_loc.copy()
                imu_x_loc = imu_o_loc
                imu_o_loc = imu_x_copy

            # Attempt to auto-detect units
            if np.max(np.abs(imu_o_loc)) > 1000:
                imu_o_loc /= 1000
                imu_x_loc /= 1000
                imu_y_loc /= 1000
                imu_d_loc /= 1000

            world_traces[imu_o_name.replace('_O', '').replace('_3', '')] = WorldTrace.construct_from_markers(timestamps,
                                                                                                             imu_o_loc,
                                                                                                             imu_d_loc,
                                                                                                             imu_x_loc,
                                                                                                             imu_y_loc)
        return world_traces

    @staticmethod
    def construct_from_markers(timestamps: np.ndarray, marker_o: np.ndarray, marker_d: np.ndarray, marker_x: np.ndarray,
                               marker_y: np.ndarray):
        """
        This function constructs a WorldTrace from three markers. This is useful for generating synthetic data.
        """

        assert not np.isnan(marker_o).any(), "NaN in marker_o"
        assert not np.isnan(marker_d).any(), "NaN in marker_d"
        assert not np.isnan(marker_x).any(), "NaN in marker_x"
        assert not np.isnan(marker_y).any(), "NaN in marker_y"

        # Constructing axis and orientation components
        x_axis_1 = marker_x - marker_d
        x_axis_1 = x_axis_1 / np.linalg.norm(x_axis_1, axis=1)[:, None]
        assert not np.isnan(x_axis_1).any(), "NaN in x_axis_1"
        x_axis_2 = marker_o - marker_y
        x_axis_2 = x_axis_2 / np.linalg.norm(x_axis_2, axis=1)[:, None]
        if np.isnan(x_axis_2).any():
            x_axis = x_axis_1
        else:
            x_axis = (x_axis_1 + x_axis_2) / 2
        x_axis = x_axis / np.linalg.norm(x_axis, axis=1)[:, None]
        assert not np.isnan(x_axis).any(), "NaN in x_axis"

        y_axis_temp_1 = marker_o - marker_x
        y_axis_temp_1 = y_axis_temp_1 / np.linalg.norm(y_axis_temp_1, axis=1)[:, None]
        assert not np.isnan(y_axis_temp_1).any(), "NaN in y_axis_temp_1"
        y_axis_temp_2 = marker_y - marker_d
        y_axis_temp_2 = y_axis_temp_2 / np.linalg.norm(y_axis_temp_2, axis=1)[:, None]
        assert not np.isnan(y_axis_temp_2).any(), "NaN in y_axis_temp_2"
        y_axis_temp = (y_axis_temp_1 + y_axis_temp_2) / 2
        y_axis_temp = y_axis_temp / np.linalg.norm(y_axis_temp, axis=1)[:, None]
        assert not np.isnan(y_axis_temp).any(), "NaN in y_axis_temp"

        z_axis = np.cross(x_axis, y_axis_temp)
        assert not np.isnan(z_axis).any(), "NaN in z_axis"
        z_axis = z_axis / np.linalg.norm(z_axis, axis=1)[:, None]
        assert not np.isnan(z_axis).any(), "NaN in z_axis"
        y_axis = np.cross(z_axis, x_axis)
        assert not np.isnan(y_axis).any(), "NaN in y_axis"

        error_y = np.linalg.norm(y_axis - y_axis_temp_1, axis=1)
        angle_error = np.arccos(np.clip(np.sum(y_axis * y_axis_temp_1, axis=1), -1, 1)) * 180 / np.pi
        if np.mean(error_y) > 0.015 or np.mean(angle_error) > 1.0:
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 5)
            # ax[0].plot(angle_error)
            # ax[0].set_title("Angle Error (deg)")
            # ax[1].plot(np.linalg.norm(marker_o - marker_x, axis=1), label='o-x')
            # ax[1].plot(np.linalg.norm(marker_o - marker_y, axis=1), label='o-y')
            # ax[1].plot(np.linalg.norm(marker_o - marker_d, axis=1), label='o-d')
            # ax[2].plot(np.linalg.norm(marker_x - marker_y, axis=1), label='x-y')
            # ax[2].plot(np.linalg.norm(marker_x - marker_d, axis=1), label='x-d')
            # ax[3].plot(np.linalg.norm(marker_y - marker_d, axis=1), label='y-d')
            # ax[1].set_title("Marker Distances from O")
            # ax[1].legend()
            # ax[2].set_title("Marker Distances from X")
            # ax[2].legend()
            # ax[3].set_title("Marker Distances from Y")
            # ax[3].legend()
            # ax[4].plot(timestamps, error_y, label='y-y_temp')
            # plt.show()

            print(f"Mean angle error: {np.mean(angle_error)}")
            print(f"Mean norm of y-y_temp: {np.mean(error_y)}")

        # Saving the location of the marker
        loc = (marker_o + marker_d + marker_x + marker_y) / 4
        loc_list = loc.tolist()
        R_list = [np.array([x, y, z]).T for x, y, z in zip(x_axis, y_axis, z_axis)]
        return WorldTrace(timestamps, loc_list, R_list)

    def get_sample_frequency(self):
        """
        This function returns the sample frequency of the WorldTrace.
        """
        return 1 / np.mean(np.diff(self.timestamps))

    def lowpass_filter(self, cutoff_freq: float, order: int):
        """
        This function applies a lowpass filter to the WorldTrace.
        """
        sample_freq = self.get_sample_frequency()
        nyquist_freq = 0.5 * sample_freq
        cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, cutoff, btype='low')
        positions = filtfilt(b, a, self.positions, axis=0).tolist()
        angle_axis = Rotation.from_matrix(self.rotations).as_rotvec()
        angle_axis = filtfilt(b, a, angle_axis, axis=0)
        rotations = Rotation.from_rotvec(angle_axis).as_matrix().tolist()
        return WorldTrace(self.timestamps, positions, rotations)

    def get_rotation_errors_deg(self, other_trace: 'WorldTrace') -> np.ndarray:
        """
        This function returns a time series list of the rotation errors in degrees between two WorldTrace instances.
        """
        assert len(self) == len(other_trace), "WorldTraces must have the same length to compare them."
        
        errors = Rotation.from_matrix([rot1.T @ rot2 for rot1, rot2 in zip(self.rotations, other_trace.rotations)])
        angle_axis = errors.as_rotvec()
        angle_deg = np.linalg.norm(angle_axis, axis=1) * 180.0 / np.pi
        return angle_deg

    def get_joint_center(self, other_world_trace: 'WorldTrace') -> Tuple[np.array, np.array, np.ndarray]:
        """ Given two world traces, solve for the best fit constant offset from a joint center.
        This is done by minimizing the sum of the squared differences between the two traces. """
        # Parent is other, child is self
        assert isinstance(other_world_trace, WorldTrace), "Must pass a WorldTrace instance to compare."
        assert len(self) == len(other_world_trace), "WorldTraces must have the same length to compare them."

        parent_loc = np.array(self.positions)
        child_loc = np.array(other_world_trace.positions)

        r_c_p = parent_loc - child_loc
        r_c_p = r_c_p.flatten()

        R_w_parent = np.concatenate(self.rotations, axis=0)
        R_w_child = np.concatenate(other_world_trace.rotations, axis=0)
        R_w = np.hstack((-R_w_parent, R_w_child))

        offsets, res, rank, S = np.linalg.lstsq(R_w, r_c_p, rcond=None)

        parent_offset = offsets[:3]
        child_offset = offsets[3:]
        error = R_w_parent @ parent_offset - R_w_child @ child_offset + r_c_p
        error = error.reshape(-1, 3)
        return parent_offset, child_offset, error

    @staticmethod
    def load_from_sto_file(sto_file: str) -> 'Dict[str, WorldTrace]':
        """
        This function loads a dictionary of WorldTrace objects from an OpenSim .sto file.

        The file is expected to contain quaternion orientation data for multiple bodies.
        Each body's trace will have its position set to the origin [0,0,0] at all times,
        as this information is not present in the .sto orientation file.
        
        Args:
            sto_file: The full path to the .sto file.

        Returns:
            A dictionary where keys are the body names (e.g., 'torso', 'pelvis') and
            values are the corresponding WorldTrace objects.
        """
        # --- 1. Validate file existence and extension ---
        if not os.path.isfile(sto_file) or not sto_file.lower().endswith('.sto'):
            raise FileNotFoundError(f"No valid .sto file found at path: {sto_file}")

        # --- 2. Read file and locate the end of the header ---
        with open(sto_file, 'r') as f:
            lines = f.readlines()

        try:
            # Find the line number where the header definition ends
            header_end_index = next(i for i, line in enumerate(lines) if 'endheader' in line)
        except StopIteration:
            raise ValueError("'.sto' file is missing the 'endheader' line.")

        # --- 3. Parse column headers ---
        # The line immediately following 'endheader' contains the column names
        column_headers = lines[header_end_index + 1].strip().split('\t')
        try:
            time_col_idx = column_headers.index('time')
        except ValueError:
            raise ValueError("'.sto' file is missing the 'time' column.")
        
        imu_headers = [h for h in column_headers if h != 'time']
        if not imu_headers:
            raise ValueError("No data columns found in the .sto file besides 'time'.")

        # --- 4. Parse the data section ---
        data_lines = lines[header_end_index + 2:]
        
        timestamps = []
        # Initialize a dictionary to hold lists of quaternions for each IMU sensor
        imu_quat_data = {header: [] for header in imu_headers}

        for line in data_lines:
            # Skip any blank lines, which can appear at the end of the file
            if not line.strip():
                continue
            
            parts = line.strip().split('\t')
            timestamps.append(float(parts[time_col_idx]))
            
            for header in imu_headers:
                header_idx = column_headers.index(header)
                quat_str = parts[header_idx]
                
                # The .sto file format for quaternions is typically (w, x, y, z)
                w, x, y, z = [float(v) for v in quat_str.split(',')]
                
                # scipy.spatial.transform.Rotation.from_quat expects the format (x, y, z, w)
                imu_quat_data[header].append([x, y, z, w])

        # --- 5. Construct WorldTrace objects for each IMU ---
        world_traces: Dict[str, 'WorldTrace'] = {}
        timestamps_np = np.array(timestamps)
        num_frames = len(timestamps_np)
        
        # Since .sto orientation files do not contain position data, we use a zero vector for all frames.
        positions = [np.zeros(3) for _ in range(num_frames)]

        for header, quats in imu_quat_data.items():
            # Convert the list of quaternions into a list of 3x3 rotation matrices
            rotation_matrices = Rotation.from_quat(quats).as_matrix()
            
            world_traces[header] = WorldTrace(timestamps=timestamps_np, 
                                              positions=positions, 
                                              rotations=[rot for rot in rotation_matrices])
        return world_traces
    
    @staticmethod
    def load_WorldTraces_from_folder(imu_folder_path: str) -> Dict[str, 'WorldTrace']:
        world_traces = {}

        # Find the mapping xml file
        mapping_file = next((f for f in os.listdir(imu_folder_path) if f.endswith('.xml')), None)
        if mapping_file is None:
            raise FileNotFoundError("No mapping file found in IMU folder")

        # Parse the XML file
        tree = ET.parse(os.path.join(imu_folder_path, mapping_file))
        root = tree.getroot()
        trial_prefix = root.find('.//trial_prefix').text

        # Iterate over each ExperimentalSensor element and load its IMUTrace
        for sensor in root.findall('.//ExperimentalSensor'):
            sensor_name = sensor.get('name').strip()
            name_in_model = sensor.find('name_in_model').text.strip()

            file_name = f"{trial_prefix}{sensor_name}.txt"
            file_path = os.path.join(imu_folder_path, 'madgwick', 'LowerExtremity', file_name)
            try:
                # Extract update rate
                with open(file_path, "r") as f:
                    for line in f:
                        if line.startswith("// Update Rate"):
                            freq = float(line.split(":")[1].split("Hz")[0])
                            if "Subject06" in imu_folder_path or "Subject10" in imu_folder_path:
                                freq = 40.0  # Fix for known incorrect frequency in these subjects madgwick files specifically
                            break

                # Read the file into a DataFrame
                df = pd.read_csv(file_path, delimiter='\t', skiprows=5)
                df = df.apply(pd.to_numeric)
                # Extract data
                timestamps = 1 / freq * np.arange(len(df))
                if df['Mat[3][3]'].isna().any():
                    rotations = [np.array(row).reshape(3, 3) for row in df[['Mag_Z','Mat[3][1]', 'Mat[3][2]', 'Mat[1][1]', 'Mat[1][2]', 'Mat[1][3]', 'Mat[2][1]', 'Mat[2][2]', 'Mat[2][3]']].values]
                else:
                    rotations = [np.array(row).reshape(3, 3) for row in df[['Mat[1][1]', 'Mat[1][2]', 'Mat[1][3]', 'Mat[2][1]', 'Mat[2][2]', 'Mat[2][3]', 'Mat[3][1]', 'Mat[3][2]', 'Mat[3][3]']].values]

                # Create WorldTrace objects
                world_traces[name_in_model] = WorldTrace(timestamps=timestamps, positions=[np.zeros(3) for _ in range(len(timestamps))], rotations=rotations)
                if freq < 100.0:
                    world_traces[name_in_model] = world_traces[name_in_model].resample(100.0)
            except FileNotFoundError:
                print(f"File {file_path} not found. Skipping sensor {sensor_name}.")
        return world_traces