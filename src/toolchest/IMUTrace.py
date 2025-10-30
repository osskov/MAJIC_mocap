import os
import numpy as np
from typing import List, Dict, Union
import xml.etree.ElementTree as ET
import pandas as pd
from scipy.interpolate import interp1d

from .finite_difference_utils import central_difference, forward_difference, polynomial_fit_derivative
from .gyro_utils import calculate_best_fit_rotation


class IMUTrace:
    """
    A container for a time-series trace of IMU data.

    This class stores synchronized gyroscope, accelerometer, and magnetometer
    data along with their corresponding timestamps. It provides utility methods
    for common operations like slicing, resampling, coordinate projection,
    and comparison.

    Attributes:
        timestamps (np.ndarray): A 1D array of timestamps, typically in seconds.
        gyro (List[np.ndarray]): A list of 3-element numpy arrays representing
            angular velocity (e.g., in rad/s) at each timestamp.
        acc (List[np.ndarray]): A list of 3-element numpy arrays representing
            linear acceleration (e.g., in m/s^2) at each timestamp.
        mag (List[np.ndarray]): A list of 3-element numpy arrays representing
            magnetic field data (e.g., in arbitrary units or Gauss) at each timestamp.
    """
    timestamps: np.ndarray
    gyro: List[np.ndarray]
    acc: List[np.ndarray]
    mag: List[np.ndarray]

    def __init__(self, timestamps: np.ndarray, gyro: List[np.ndarray], acc: List[np.ndarray], mag: List[np.ndarray]):
        """
        Initializes the IMUTrace object.

        Args:
            timestamps (np.ndarray): 1D array of timestamps.
            gyro (List[np.ndarray]): List of 3-element gyro data arrays.
            acc (List[np.ndarray]): List of 3-element accelerometer data arrays.
            mag (List[np.ndarray]): List of 3-element magnetometer data arrays.

        Raises:
            AssertionError: If the lengths of timestamps, gyro, acc, and mag
            lists do not all match.
        """
        assert (len(timestamps) == len(gyro) == len(acc) == len(mag)), \
            "All data streams (timestamps, gyro, acc, mag) must have the same length."
        self.timestamps = timestamps
        self.gyro = gyro
        self.acc = acc
        self.mag = mag

    def __len__(self):
        """
        Returns the number of samples in the IMUTrace.

        This allows us to call len(trace) on an IMUTrace instance.
        """
        return len(self.timestamps)

    def __getitem__(self, key) -> 'IMUTrace':
        """
        Allows indexing or slicing of the IMUTrace.

        This allows us to slice the IMUTrace instance and access ranges of
        items with `sub_trace = trace[1:4]`. If an integer is passed, it
        returns a new IMUTrace of length 1, e.g., `sub_trace = trace[2]`.

        Args:
            key (slice or int): The index or slice to apply.

        Returns:
            IMUTrace: A new IMUTrace object containing the selected data.
        """
        if isinstance(key, slice):
            # If key is a slice object, return a new IMUTrace instance with the sliced items
            return IMUTrace(self.timestamps[key], self.gyro[key], self.acc[key], self.mag[key])
        else:
            # If key is an integer, return the corresponding item as a length 1 trace
            return IMUTrace(np.array([self.timestamps[key]]), [self.gyro[key]], [self.acc[key]], [self.mag[key]])

    def __eq__(self, other):
        """
        Checks for exact equality between two IMUTrace instances.

        This will return True if the timestamps, gyro, acc, and mag data
        are all *exactly* equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if all corresponding data points are identical, False otherwise.
        """
        if not isinstance(other, IMUTrace):
            return False
        if len(self) != len(other):
            return False
        return ((self.timestamps == other.timestamps).all() and
                all(np.all(self.gyro[i] == other.gyro[i]) for i in range(len(self.gyro))) and
                all(np.all(self.acc[i] == other.acc[i]) for i in range(len(self.acc))) and
                all(np.all(self.mag[i] == other.mag[i]) for i in range(len(self.mag))))

    def __sub__(self, other):
        """
        Subtracts the data of another IMUTrace from this one.

        This will return a new IMUTrace instance with the gyro, acc, and mag
        data subtracted element-wise. The timestamps must be almost equal.

        Args:
            other (IMUTrace): The IMUTrace to subtract.

        Returns:
            IMUTrace: A new IMUTrace containing the difference.

        Raises:
            AssertionError: If `other` is not an IMUTrace, if lengths differ,
                or if timestamps are not sufficiently close.
        """
        assert isinstance(other, IMUTrace), "Subtraction can only be done between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to subtract."
        np.testing.assert_array_almost_equal(self.timestamps, other.timestamps, decimal=6,
                                             err_msg="IMUTraces must have the same timestamps to subtract.")

        gyro = [gyro1 - gyro2 for gyro1, gyro2 in zip(self.gyro, other.gyro)]
        acc = [acc1 - acc2 for acc1, acc2 in zip(self.acc, other.acc)]
        mag = [mag1 - mag2 for mag1, mag2 in zip(self.mag, other.mag)]
        return IMUTrace(self.timestamps, gyro, acc, mag)

    def allclose(self, other, atol=1e-6):
        """
        Checks for approximate equality between two IMUTrace instances.

        This will return True if the timestamps, gyro, acc, and mag data
        are all approximately equal within the specified absolute tolerance.

        Args:
            other (object): The object to compare with.
            atol (float, optional): The absolute tolerance. Defaults to 1e-6.

        Returns:
            bool: True if all corresponding data points are close, False otherwise.
        """
        if not isinstance(other, IMUTrace):
            return False
        if len(self) != len(other):
            return False
        if len(self.gyro) != len(other.gyro) or len(self.acc) != len(other.acc) or len(self.mag) != len(other.mag):
            return False
        # Check for empty traces
        if len(self) == 0:
            return True
        if self.gyro[0].shape != other.gyro[0].shape or self.acc[0].shape != other.acc[0].shape or self.mag[0].shape != \
                other.mag[0].shape:
            return False
        return (np.allclose(self.timestamps, other.timestamps, atol=atol) and
                all(np.allclose(self.gyro[i], other.gyro[i], atol=atol) for i in range(len(self.gyro))) and
                all(np.allclose(self.acc[i], other.acc[i], atol=atol) for i in range(len(self.acc))) and
                all(np.allclose(self.mag[i], other.mag[i], atol=atol) for i in range(len(self.mag))))

    def copy(self) -> 'IMUTrace':
        """
        Returns a deep copy of the IMUTrace object.

        This ensures all underlying numpy arrays (timestamps, gyro, acc, mag)
        and their containing lists are duplicated to prevent unintended
        modification of the original object.

        Returns:
            IMUTrace: A new, independent copy of the IMUTrace.
        """
        return IMUTrace(
            timestamps=self.timestamps.copy(),
            # Deep copy the lists of numpy arrays
            gyro=[g.copy() for g in self.gyro],
            acc=[a.copy() for a in self.acc],
            mag=[m.copy() for m in self.mag]
        )

    def _finite_difference_gyros(self, method='polyfit') -> List[np.ndarray]:
        """
        This is a private method to compute the finite difference of the gyros
        in the IMUTrace. This can be calculated a number of different ways.

        method='central' computes the finite difference using a central difference.
        method='first_order' computes the finite difference using a first order approximation.
        method='polyfit' computes the derivatives using a polynomial fit.
        """
        derivates: List[np.ndarray] = []
        if method == 'central':
            derivates = [central_difference(np.array([gyro[axis] for gyro in self.gyro]), self.timestamps) for axis in
                         range(3)]
        elif method == 'first_order':
            derivates = [forward_difference(np.array([gyro[axis] for gyro in self.gyro]), self.timestamps) for axis in
                         range(3)]
        elif method == 'polyfit':
            derivates = [
                polynomial_fit_derivative(np.array([gyro[axis] for gyro in self.gyro]), self.timestamps, order=2) for
                axis in range(3)]
        # Now we need to flip the axes back to the original shape
        return [np.array([derivate[i] for derivate in derivates]) for i in range(len(self.timestamps))]

    def project_acc(self, local_offset: Union[np.ndarray, List[np.ndarray]],
                    finite_difference_gyro_method='polyfit') -> 'IMUTrace':
        """
        Projects acceleration to a new point on the same rigid body.

        This method calculates the linear acceleration at a point defined by
        `local_offset` relative to the IMU's measurement frame. It uses the
        full rigid body acceleration equation:
        $a_p = a_o + \dot{\omega} \times r + \omega \times (\omega \times r)$

        where:
        - $a_p$ is the projected acceleration (the result)
        - $a_o$ is the original measured acceleration (self.acc)
        - $\omega$ is the angular velocity (self.gyro)
        - $\dot{\omega}$ is the angular acceleration (computed from gyro)
        - $r$ is the local_offset vector

        Args:
            local_offset (Union[np.ndarray, List[np.ndarray]]): The 3D position
                vector (or list of vectors) from the IMU measurement frame to the
                new point of interest, expressed in the IMU's local frame.
            finite_difference_gyro_method (str, optional): The method used to
                calculate the angular acceleration ($\dot{\omega}$).
                Passed to `_finite_difference_gyros`. Defaults to 'polyfit'.

        Returns:
            IMUTrace: A new IMUTrace object with the same timestamps, gyro, and
            mag data, but with the `acc` data projected to the new location.
        """
        if isinstance(local_offset, np.ndarray):
            local_offset = [local_offset] * len(self)
        gyro_derivative = self._finite_difference_gyros(finite_difference_gyro_method)
        acc = [acc + np.cross(d_gyro, offset) + np.cross(gyro, np.cross(gyro, offset)) for acc, gyro, d_gyro, offset in
               zip(self.acc, self.gyro, gyro_derivative, local_offset)]
        return IMUTrace(self.timestamps, self.gyro, acc, self.mag)

    def get_sample_frequency(self) -> float:
        """
        Calculates and returns the average sample frequency of the trace.

        The frequency is computed as the inverse of the mean difference
        between consecutive timestamps.

        Returns:
            float: The average sample frequency in Hz.
        """
        return 1 / np.mean(np.diff(self.timestamps))

    def calculate_gyro_angle_error(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculates the pointwise angle error between two gyroscope traces.

        This method computes the angle (in radians) between the angular velocity
        vectors from `self` and `other` at each corresponding timestamp.

        Args:
            other (IMUTrace): Another IMUTrace object to compare against.

        Returns:
            np.ndarray: A 1D array containing the angle error (in radians)
            at each sample. Returns 0 if both vectors are near-zero,
            and np.nan if only one is near-zero.

        Raises:
            AssertionError: If `other` is not an IMUTrace or if the
            traces have different lengths.
        """
        assert isinstance(other, IMUTrace), "Angle error can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate angle error."

        angle_error = np.zeros(len(self))
        for i in range(len(self)):
            self_gyro = self.gyro[i]
            other_gyro = other.gyro[i]
            if np.linalg.norm(self_gyro) > 0 and np.linalg.norm(other_gyro) > 0:
                dot_product = np.dot(self_gyro, other_gyro)
                dot_product /= np.linalg.norm(self_gyro)
                dot_product /= np.linalg.norm(other_gyro)
                dot_product = np.clip(dot_product, -1, 1)
                angle_error[i] = (np.arccos(dot_product))
            elif np.linalg.norm(self_gyro) < 1e-8 and np.linalg.norm(other_gyro) < 1e-8:
                angle_error[i] = 0
            else:
                angle_error[i] = np.nan
        return angle_error

    def calculate_rotation_offset_from_gyros(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculates the best-fit static rotation offset between two gyro traces.

        This method finds a single 3x3 rotation matrix (R_self_other) that
        best aligns the gyroscope data from `other` to `self` over the entire
        trace. This is typically used to find the relative orientation
        between two IMUs that are rigidly attached.

        Assumes: `self.gyro[i] \approx R_self_other @ other.gyro[i]`

        Args:
            other (IMUTrace): The other IMUTrace object (representing the
                'other' coordinate frame).

        Returns:
            np.ndarray: A 3x3 rotation matrix (R_self_other) that rotates
            vectors from the 'other' frame to the 'self' frame.

        Raises:
            AssertionError: If `other` is not an IMUTrace or if the
            traces have different lengths.
        """
        assert isinstance(other, IMUTrace), "Rotation offset can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate rotation offset."

        R_so = calculate_best_fit_rotation(self.gyro, other.gyro)
        return R_so

    def re_zero_timestamps(self) -> 'IMUTrace':
        """
        Creates a new IMUTrace with timestamps starting from zero.

        This shifts the entire timestamp vector by subtracting the
        first timestamp value (`self.timestamps[0]`).

        Returns:
            IMUTrace: A new IMUTrace object with re-zeroed timestamps.
        """
        return IMUTrace(self.timestamps - self.timestamps[0], self.gyro, self.acc, self.mag)

    def resample(self, new_frequency: float):
        """
        Resamples the IMU trace to a new, constant frequency.

        This method uses linear interpolation (`scipy.interpolate.interp1d`)
        to resample the gyro, accelerometer, and magnetometer data to a
        new set of timestamps defined by the `new_frequency`.

        The new time vector starts at the original `self.timestamps[0]` and
        ends at or slightly after `self.timestamps[-1]`.

        Args:
            new_frequency (float): The target sample frequency in Hz.

        Returns:
            IMUTrace: A new, resampled IMUTrace object.
        """
        new_dt = 1 / new_frequency
        old_dt = np.mean(np.diff(self.timestamps))
        if np.isclose(new_dt, old_dt):
            # No resampling needed if the frequency is the same
            return self.copy()

        new_timestamps = np.arange(start=self.timestamps[0], stop=self.timestamps[-1] + min(old_dt, new_dt),
                                   step=new_dt)
        
        # Stack data for efficient interpolation
        gyro_data = np.vstack(self.gyro)
        acc_data = np.vstack(self.acc)
        mag_data = np.vstack(self.mag)

        gyro_interpolator = interp1d(self.timestamps, gyro_data, axis=0, kind='linear',
                                     fill_value='extrapolate')
        acc_interpolator = interp1d(self.timestamps, acc_data, axis=0, kind='linear',
                                    fill_value='extrapolate')
        mag_interpolator = interp1d(self.timestamps, mag_data, axis=0, kind='linear',
                                    fill_value='extrapolate')

        # Get new data as monolithic arrays
        new_gyro_data = gyro_interpolator(new_timestamps)
        new_acc_data = acc_interpolator(new_timestamps)
        new_mag_data = mag_interpolator(new_timestamps)

        # Convert back to list of arrays
        new_gyro = [row for row in new_gyro_data]
        new_acc = [row for row in new_acc_data]
        new_mag = [row for row in new_mag_data]

        return IMUTrace(new_timestamps, new_gyro, new_acc, new_mag)

    @staticmethod
    def _parse_imu_txt_file(file_path: str) -> 'IMUTrace':
        """
        Parses a single Xsens-formatted IMU .txt file into an IMUTrace object.
        
        Raises:
            FileNotFoundError: If the file_path does not exist.
        """
        # Extract update rate
        freq = 100.0  # Default fallback
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("// Update Rate"):
                    freq = float(line.split(":")[1].split("Hz")[0])
                    break

        # Read the file into a DataFrame
        df = pd.read_csv(file_path, delimiter='\t', skiprows=5)
        df = df.apply(pd.to_numeric)

        # Extract data
        timestamps = 1 / freq * np.arange(len(df))
        acc = [np.array(row) for row in df[['Acc_X', 'Acc_Y', 'Acc_Z']].values]
        gyro = [np.array(row) for row in df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values]
        mag = [np.array(row) for row in df[['Mag_X', 'Mag_Y', 'Mag_Z']].values]

        return IMUTrace(timestamps=timestamps, acc=acc, gyro=gyro, mag=mag)

    @staticmethod
    def _load_imu_traces_from_structure(
        imu_folder_path: str, 
        data_subdirectory_parts: List[str]
    ) -> Dict[str, 'IMUTrace']:
        """
        Generic helper to load IMU traces based on a common XML mapping
        and a specific data subdirectory.

        Args:
            imu_folder_path (str): The path to the root folder of a specific trial.
            data_subdirectory_parts (List[str]): A list of path components
                that form the subdirectory from the root to the data files
                (e.g., ['xsens', 'LowerExtremity'] or ['imu data']).

        Returns:
            Dict[str, 'IMUTrace']: A dictionary mapping 'name_in_model' to its IMUTrace.
        """
        imu_traces = {}

        # Find the mapping xml file
        mapping_file = next((f for f in os.listdir(imu_folder_path) if f.endswith('.xml')), None)
        if mapping_file is None:
            raise FileNotFoundError(f"No mapping file (.xml) found in IMU folder: {imu_folder_path}")

        # Parse the XML file
        tree = ET.parse(os.path.join(imu_folder_path, mapping_file))
        root = tree.getroot()
        trial_prefix_element = root.find('.//trial_prefix')
        
        # Handle cases where trial_prefix might be missing or empty
        trial_prefix = trial_prefix_element.text if trial_prefix_element is not None else ""

        # Iterate over each ExperimentalSensor element and load its IMUTrace
        for sensor in root.findall('.//ExperimentalSensor'):
            sensor_name = sensor.get('name').strip()
            name_in_model = sensor.find('name_in_model').text.strip()

            file_name = f"{trial_prefix}{sensor_name}.txt"
            
            # Use the * operator to unpack the list of subdirectory parts
            file_path = os.path.join(imu_folder_path, *data_subdirectory_parts, file_name)
            
            try:
                # Call the dedicated parser
                imu_traces[name_in_model] = IMUTrace._parse_imu_txt_file(file_path)
            except FileNotFoundError:
                print(f"File {file_path} not found. Skipping sensor {sensor_name}.")
        
        return imu_traces
    
    @staticmethod
    def load_IMUTraces_from_Al_Borno_folder(imu_folder_path: str) -> Dict[str, 'IMUTrace']:
        """
        Loads IMU traces from a folder structured like the Al Borno et al (2022) dataset.
        ... (rest of docstring) ...
        """
        # Define the specific subdirectory path for this dataset
        subdir_parts = ['xsens', 'LowerExtremity']
        return IMUTrace._load_imu_traces_from_structure(imu_folder_path, subdir_parts)

    @staticmethod
    def load_IMUTraces_from_Skov_folder(imu_folder_path: str) -> Dict[str, 'IMUTrace']:
        """
        Loads IMU traces from a folder structured for the dataset presented in Skov et al (2025).
        ... (rest of docstring) ...
        """
        # Note: I renamed this from 'load_IMUTraces_from_folder' to
        # 'load_IMUTraces_from_Skov_folder' to be more specific,
        # matching its docstring and the Al_Borno function.
        
        # Define the specific subdirectory path for this dataset
        subdir_parts = ['imu data']
        return IMUTrace._load_imu_traces_from_structure(imu_folder_path, subdir_parts)
    
    def find_spheroidal_joint_offset(self,
                                     other: 'IMUTrace',
                                     initial_offset_self: np.ndarray = None,
                                     initial_offset_other: np.ndarray = None,
                                     max_iterations: int = 20,
                                     tolerance: float = 1e-6,
                                     subsample_rate: int = 5) -> Dict[str, Union[np.ndarray, bool]]:
        """
        Estimates the position of a spheroidal joint center relative to two IMUs.

        This method implements the algorithm described in Section II-B of the paper
        "Joint axis and position estimation from inertial measurement data by
        exploiting kinematic constraints" by Seel et al. (2012). It uses a
        Gauss-Newton optimization to find the offset vectors from each IMU's
        origin to the common joint center.

        The core constraint is that the magnitude of the joint center's acceleration,
        when calculated from either IMU's frame of reference, must be equal.

        Args:
            other (IMUTrace): The second IMUTrace object, rigidly attached to the
                other body segment. Data is assumed to be time-synchronized.
            initial_offset_self (np.ndarray, optional): A 3-element initial guess
                for the offset vector from this IMU's origin to the joint center.
                If None, a small random guess is used.
            initial_offset_other (np.ndarray, optional): A 3-element initial guess
                for the offset vector from the other IMU's origin to the joint
                center. If None, a small random guess is used.
            max_iterations (int, optional): The maximum number of iterations for
                the optimization. Defaults to 20.
            tolerance (float, optional): The convergence tolerance. The optimization
                stops when the norm of the update step is less than this value.
                Defaults to 1e-6.
            subsample_rate (int, optional): The rate at which to subsample the data
                to speed up computation (e.g., a value of 5 uses every 5th sample).
                Defaults to 5.

        Returns:
            Dict[str, Union[np.ndarray, bool]]: A dictionary containing:
                - 'offset_self': The estimated 3D offset vector in self's frame.
                - 'offset_other': The estimated 3D offset vector in other's frame.
                - 'converged': A boolean indicating if the algorithm converged.
        """
        # Ensure data streams are synchronized
        np.testing.assert_array_almost_equal(
            self.timestamps, other.timestamps, decimal=5,
            err_msg="IMU traces must be time-synchronized. Use the resample() method first."
        )

        # Pre-compute gyro derivatives
        g1_dot = self._finite_difference_gyros()
        g2_dot = other._finite_difference_gyros()

        # Subsample data to speed up computation
        indices = np.arange(0, len(self), subsample_rate)
        g1, g2 = [self.gyro[i] for i in indices], [other.gyro[i] for i in indices]
        a1, a2 = [self.acc[i] for i in indices], [other.acc[i] for i in indices]
        g1_dot, g2_dot = [g1_dot[i] for i in indices], [g2_dot[i] for i in indices]
        num_samples = len(indices)

        # Initialize the state vector x = [o1_x, o1_y, o1_z, o2_x, o2_y, o2_z]
        o1 = np.random.rand(3) * 0.1 if initial_offset_self is None else initial_offset_self.copy()
        o2 = np.random.rand(3) * 0.1 if initial_offset_other is None else initial_offset_other.copy()
        x = np.concatenate([o1, o2])

        # Gauss-Newton optimization loop
        for _ in range(max_iterations):
            o1, o2 = x[:3], x[3:]

            e = np.zeros(num_samples)
            J = np.zeros((num_samples, 6))

            for k in range(num_samples):
                g1_k, g2_k = g1[k], g2[k]
                a1_k, a2_k = a1[k], a2[k]
                g1_dot_k, g2_dot_k = g1_dot[k], g2_dot[k]

                # Eq(3): Γ_g(o) = g x (g x o) + ġ x o
                gamma1 = np.cross(g1_k, np.cross(g1_k, o1)) + np.cross(g1_dot_k, o1)
                gamma2 = np.cross(g2_k, np.cross(g2_k, o2)) + np.cross(g2_dot_k, o2)

                joint_acc1 = a1_k - gamma1
                joint_acc2 = a2_k - gamma2

                norm1 = np.linalg.norm(joint_acc1)
                norm2 = np.linalg.norm(joint_acc2)

                if norm1 > 1e-9 and norm2 > 1e-9:
                    e[k] = norm1 - norm2

                    # Eq(4) Jacobian: d(||a-Γ(o)||)/do = -Γ_T(a-Γ(o)) / ||a-Γ(o)||
                    # where Γ_T(v) = (v x g) x g + v x ġ
                    v1 = joint_acc1
                    gamma_T1_v1 = np.cross(np.cross(v1, g1_k), g1_k) + np.cross(v1, g1_dot_k)
                    J[k, :3] = -gamma_T1_v1 / norm1

                    v2 = joint_acc2
                    gamma_T2_v2 = np.cross(np.cross(v2, g2_k), g2_k) + np.cross(v2, g2_dot_k)
                    # de/do2 = -d(norm2)/do2 leads to a double negative
                    J[k, 3:] = gamma_T2_v2 / norm2

            try:
                # Update step: x_new = x - pinv(J) * e
                delta_x = -np.linalg.pinv(J) @ e
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix in pseudoinverse calculation. Stopping iteration.")
                return {'offset_self': x[:3], 'offset_other': x[3:], 'converged': False}

            x += delta_x

            if np.linalg.norm(delta_x) < tolerance:
                return {'offset_self': x[:3], 'offset_other': x[3:], 'converged': True}

        return {'offset_self': x[:3], 'offset_other': x[3:], 'converged': False}


    def find_hinge_joint_axis(self,
                              other: 'IMUTrace',
                              initial_axis_self: np.ndarray = None,
                              initial_axis_other: np.ndarray = None,
                              max_iterations: int = 20,
                              tolerance: float = 1e-6,
                              subsample_rate: int = 5) -> Dict[str, Union[np.ndarray, bool]]:
        """
        Estimates the axis of a hinge joint relative to two IMUs.

        This method implements the algorithm described in Section II-A of the paper
        "Joint axis and position estimation from inertial measurement data by
        exploiting kinematic constraints" by Seel et al. (2012). It uses a
        Gauss-Newton optimization to find the joint axis vectors in each IMU's
        local coordinate system.

        The core constraint is that the magnitude of the angular velocity projected
        onto the plane normal to the joint axis must be equal for both bodies.

        Args:
            other (IMUTrace): The second IMUTrace object, rigidly attached to the
                other body segment. Data is assumed to be time-synchronized.
            initial_axis_self (np.ndarray, optional): A 3-element initial guess
                for the unit vector of the hinge axis in this IMU's frame.
                If None, a random guess is used.
            initial_axis_other (np.ndarray, optional): A 3-element initial guess
                for the unit vector of the hinge axis in the other IMU's frame.
                If None, a random guess is used.
            max_iterations (int, optional): The maximum number of iterations for
                the optimization. Defaults to 20.
            tolerance (float, optional): The convergence tolerance. The optimization
                stops when the norm of the update step is less than this value.
                Defaults to 1e-6.
            subsample_rate (int, optional): The rate at which to subsample the data
                to speed up computation (e.g., a value of 5 uses every 5th sample).
                Defaults to 5.

        Returns:
            Dict[str, Union[np.ndarray, bool]]: A dictionary containing:
                - 'axis_self': The estimated 3D unit axis vector in self's frame.
                - 'axis_other': The estimated 3D unit axis vector in other's frame.
                - 'converged': A boolean indicating if the algorithm converged.
        """
        # Ensure data streams are synchronized
        np.testing.assert_array_almost_equal(
            self.timestamps, other.timestamps, decimal=5,
            err_msg="IMU traces must be time-synchronized. Use the resample() method first."
        )

        # Subsample data
        indices = np.arange(0, len(self), subsample_rate)
        g1, g2 = [self.gyro[i] for i in indices], [other.gyro[i] for i in indices]
        num_samples = len(indices)

        # Helper functions for Eq(5) spherical coordinate parametrization
        def spherical_to_cartesian(phi, theta):
            return np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)])

        def cartesian_derivatives(phi, theta):
            dj_dphi = np.array([-np.sin(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi)])
            dj_dtheta = np.array([-np.cos(phi) * np.sin(theta), np.cos(phi) * np.cos(theta), 0])
            return dj_dphi, dj_dtheta

        # Initialize state vector x = [phi1, theta1, phi2, theta2]
        if initial_axis_self is None:
            phi1, theta1 = np.random.uniform(-np.pi / 2, np.pi / 2), np.random.uniform(-np.pi, np.pi)
        else:
            j1_init = initial_axis_self / np.linalg.norm(initial_axis_self)
            phi1 = np.arcsin(j1_init[2])
            theta1 = np.arctan2(j1_init[1], j1_init[0])

        if initial_axis_other is None:
            phi2, theta2 = np.random.uniform(-np.pi / 2, np.pi / 2), np.random.uniform(-np.pi, np.pi)
        else:
            j2_init = initial_axis_other / np.linalg.norm(initial_axis_other)
            phi2 = np.arcsin(j2_init[2])
            theta2 = np.arctan2(j2_init[1], j2_init[0])

        x = np.array([phi1, theta1, phi2, theta2])

        # Gauss-Newton optimization loop
        for _ in range(max_iterations):
            phi1, theta1, phi2, theta2 = x
            j1, j2 = spherical_to_cartesian(phi1, theta1), spherical_to_cartesian(phi2, theta2)

            e = np.zeros(num_samples)
            J = np.zeros((num_samples, 4))
            dj1_dphi1, dj1_dtheta1 = cartesian_derivatives(phi1, theta1)
            dj2_dphi2, dj2_dtheta2 = cartesian_derivatives(phi2, theta2)

            for k in range(num_samples):
                g1_k, g2_k = g1[k], g2[k]

                v1, v2 = np.cross(g1_k, j1), np.cross(g2_k, j2)
                norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)

                if norm1 > 1e-9 and norm2 > 1e-9:
                    # Eq(1) Error: e = ||g1 x j1|| - ||g2 x j2||
                    e[k] = norm1 - norm2

                    # Eq(2) Gradient: d(||g x j||)/dj = ((g x j) x g) / ||g x j||
                    d_norm1_dj1 = np.cross(v1, g1_k) / norm1
                    d_norm2_dj2 = np.cross(v2, g2_k) / norm2

                    # Chain rule for Jacobian columns
                    J[k, 0] = np.dot(d_norm1_dj1, dj1_dphi1)
                    J[k, 1] = np.dot(d_norm1_dj1, dj1_dtheta1)
                    J[k, 2] = -np.dot(d_norm2_dj2, dj2_dphi2) # de/dphi2 = -d(norm2)/dphi2
                    J[k, 3] = -np.dot(d_norm2_dj2, dj2_dtheta2) # de/dtheta2 = -d(norm2)/dtheta2

            try:
                delta_x = -np.linalg.pinv(J) @ e
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix in pseudoinverse calculation. Stopping iteration.")
                j1, j2 = spherical_to_cartesian(x[0], x[1]), spherical_to_cartesian(x[2], x[3])
                return {'axis_self': j1, 'axis_other': j2, 'converged': False}

            x += delta_x

            if np.linalg.norm(delta_x) < tolerance:
                j1, j2 = spherical_to_cartesian(x[0], x[1]), spherical_to_cartesian(x[2], x[3])
                return {'axis_self': j1, 'axis_other': j2, 'converged': True}

        j1, j2 = spherical_to_cartesian(x[0], x[1]), spherical_to_cartesian(x[2], x[3])
        return {'axis_self': j1, 'axis_other': j2, 'converged': False}
    
    