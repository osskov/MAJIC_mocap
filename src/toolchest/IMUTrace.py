import os

import numpy as np
from typing import List, Dict, Tuple, Union
import xml.etree.ElementTree as ET
import pandas as pd
import scipy.stats
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from .finite_difference_utils import central_difference, forward_difference, polynomial_fit_derivative
from scipy.signal import butter, filtfilt
from .gyro_utils import integrate_rotations, finite_difference_rotations, calculate_best_fit_rotation, angular_velocity_to_rotation_matrix


class IMUTrace:
    timestamps: np.ndarray
    gyro: List[np.ndarray]
    acc: List[np.ndarray]
    mag: List[np.ndarray]

    def __init__(self, timestamps: np.ndarray, gyro: List[np.ndarray], acc: List[np.ndarray], mag: List[np.ndarray]):
        assert (len(timestamps) == len(gyro) == len(acc))
        self.timestamps = timestamps
        self.gyro = gyro
        self.acc = acc
        self.mag = mag

    def __len__(self):
        """
        Returns the number of samples in the IMUTrace. This allows us to call len(trace) on an IMUTrace instance.
        """
        return len(self.timestamps)

    def __getitem__(self, key) -> 'IMUTrace':
        """
        Allows us to use the square bracket notation to access the IMUTrace instance. This allows us to slice the
        IMUTrace instance and access ranges of items with `sub_trace = trace[1:4]`. If we pass an integer, we can
        return the corresponding item as a length 1 trace with `sub_trace = trace[2]` and `len(sub_trace) == 1`.
        """
        if isinstance(key, slice):
            # If key is a slice object, return a new IMUTrace instance with the sliced items
            return IMUTrace(self.timestamps[key], self.gyro[key], self.acc[key], self.mag[key])
        else:
            # If key is an integer, return the corresponding item as a solo list
            return IMUTrace(np.array([self.timestamps[key]]), [self.gyro[key]], [self.acc[key]], [self.mag[key]])

    def __eq__(self, other):
        """
        Allows us to compare two IMUTrace instances for equality. This will return True if the timestamps, gyro, and
        acc are all _exactly_ equal.
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
        Allows us to subtract two IMUTrace instances. This will return a new IMUTrace instance with the gyro and acc
        data subtracted.
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
        Allows us to compare two IMUTrace instances for approximate equality. This will return True if the timestamps,
        gyro, and acc are all approximately equal within the specified tolerance.
        """
        if not isinstance(other, IMUTrace):
            return False
        if len(self) != len(other):
            return False
        if len(self.gyro) != len(other.gyro) or len(self.acc) != len(other.acc) or len(self.mag) != len(other.mag):
            return False
        if self.gyro[0].shape != other.gyro[0].shape or self.acc[0].shape != other.acc[0].shape or self.mag[0].shape != \
                other.mag[0].shape:
            return False
        return (np.allclose(self.timestamps, other.timestamps, atol=atol) and
                all(np.allclose(self.gyro[i], other.gyro[i], atol=atol) for i in range(len(self.gyro))) and
                all(np.allclose(self.acc[i], other.acc[i], atol=atol) for i in range(len(self.acc))) and
                all(np.allclose(self.mag[i], other.mag[i], atol=atol) for i in range(len(self.mag))))

    def _finite_difference_gyros(self, method='polyfit') -> List[np.ndarray]:
        """
        This is a private method to compute the finite difference of the gyros in the IMUTrace. This can be calculated
        a number of different ways.

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
        This function projects the gyro and acc data in the IMUTrace by a local offset within the frame.
        """
        if isinstance(local_offset, np.ndarray):
            local_offset = [local_offset] * len(self)
        gyro_derivative = self._finite_difference_gyros(finite_difference_gyro_method)
        acc = [acc + np.cross(d_gyro, offset) + np.cross(gyro, np.cross(gyro, offset)) for acc, gyro, d_gyro, offset in
               zip(self.acc, self.gyro, gyro_derivative, local_offset)]
        return IMUTrace(self.timestamps, self.gyro, acc, self.mag)

    def project_mag(self, other_imu: 'IMUTrace', local_offset_to_other_imu: np.array, local_offset_to_projection_point) -> 'IMUTrace':
        """
        This function projects the gyro and mag data in the IMUTrace by a local offset within the frame.
        """
        sensor_len = np.linalg.norm(local_offset_to_other_imu)
        if sensor_len == 0:
            return self
        proj_distance = np.dot(local_offset_to_projection_point, local_offset_to_other_imu) / sensor_len
        delta_mag = [(m2 - m1) * proj_distance / sensor_len for m1, m2 in zip(self.mag, other_imu.mag)]

        projected_mag = [m1 + delta for m1, delta in zip(self.mag, delta_mag)]
        return IMUTrace(self.timestamps, self.gyro, self.acc, projected_mag)




    def left_rotate(self, R: Union[np.ndarray, List[np.ndarray]]):
        """
        Apply a left rotation to an IMUTrace. R can be thought of as R_newworld_oldworld.
        """
        if isinstance(R, np.ndarray):
            R = [R] * len(self)
        R_world_old = integrate_rotations(self.gyro, self.timestamps)
        R_world_new = [R_rotate @ R_w_o for R_rotate, R_w_o in zip(R, R_world_old)]
        gyro = finite_difference_rotations(R_world_new, self.timestamps)
        a_world = [R_w_o @ a_o for R_w_o, a_o in zip(R_world_old, self.acc)]
        a_imu = [R_w_n.T @ a_w for R_w_n, a_w in zip(R_world_new, a_world)]
        m_world = [R_w_o @ m_o for R_w_o, m_o in zip(R_world_old, self.mag)]
        m_imu = [R_w_n.T @ m_w for R_w_n, m_w in zip(R_world_new, m_world)]
        return IMUTrace(self.timestamps, gyro, a_imu, m_imu)

    def right_rotate(self, R: Union[np.ndarray, List[np.ndarray]]):
        """
        Rotate the gyro and acc data in the IMUTrace by 90 degrees to the right. R_rotate can be thought of as R_oldimu_newimu.
        """
        if isinstance(R, np.ndarray):
            R = [R] * len(self)
            gyro = [R_rotate.T @ gyro for R_rotate, gyro in zip(R, self.gyro)]
        else:
            R_world_old = integrate_rotations(self.gyro, self.timestamps)
            R_world_new = [R_w_o @ R_rotate for R_w_o, R_rotate in zip(R_world_old, R)]
            gyro = finite_difference_rotations(R_world_new, self.timestamps)
        a_new = [R_rotate.T @ a_oi for R_rotate, a_oi in zip(R, self.acc)]
        m_new = [R_rotate.T @ m_oi for R_rotate, m_oi in zip(R, self.mag)]
        return IMUTrace(self.timestamps, gyro, a_new, m_new)


    def lowpass_filter_gyro(self, cutoff_frequency: float, order: int) -> 'IMUTrace':
        """
        Apply a Butterworth lowpass filter to the gyro data.
        :param cutoff_frequency: The cutoff frequency of the filter.
        :param order: The order of the filter.
        :return: A new IMUTrace instance with the filtered gyro data.
        """
        # Calculate the sample rate from the timestamps
        sample_rate = 1 / np.mean(np.diff(self.timestamps))

        # Normalize the cutoff frequency with respect to the Nyquist frequency
        nyquist_frequency = sample_rate / 2
        normalized_cutoff = cutoff_frequency / nyquist_frequency

        # Design the Butterworth filter
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)

        # Apply the filter to each component of the gyro data
        filtered_gyro = [filtfilt(b, a, np.vstack(self.gyro)[:, i]) for i in range(3)]
        filtered_gyro = np.column_stack(filtered_gyro)
        filtered_gyro_list = [filtered_gyro[i] for i in range(filtered_gyro.shape[0])]

        return IMUTrace(self.timestamps, filtered_gyro_list, self.acc, self.mag)

    def lowpass_filter_acc(self, cutoff_frequency: float, order: int) -> 'IMUTrace':
        """
        Apply a Butterworth lowpass filter to the acc data.
        :param cutoff_frequency: The cutoff frequency of the filter.
        :param order: The order of the filter.
        :return: A new IMUTrace instance with the filtered acc data.
        """
        # Calculate the sample rate from the timestamps
        sample_rate = 1 / np.mean(np.diff(self.timestamps))

        # Normalize the cutoff frequency with respect to the Nyquist frequency
        nyquist_frequency = sample_rate / 2
        normalized_cutoff = cutoff_frequency / nyquist_frequency

        # Design the Butterworth filter
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)

        # Apply the filter to each component of the acc data
        filtered_acc = [filtfilt(b, a, np.vstack(self.acc)[:, i]) for i in range(3)]
        filtered_acc = np.column_stack(filtered_acc)
        filtered_acc_list = [filtered_acc[i] for i in range(filtered_acc.shape[0])]

        return IMUTrace(self.timestamps, self.gyro, filtered_acc_list, self.mag)

    def lowpass_filter_mag(self, cutoff_frequency: float, order: int) -> 'IMUTrace':
        """
        Apply a Butterworth lowpass filter to the acc data.
        :param cutoff_frequency: The cutoff frequency of the filter.
        :param order: The order of the filter.
        :return: A new IMUTrace instance with the filtered acc data.
        """
        # Calculate the sample rate from the timestamps
        sample_rate = 1 / np.mean(np.diff(self.timestamps))

        # Normalize the cutoff frequency with respect to the Nyquist frequency
        nyquist_frequency = sample_rate / 2
        normalized_cutoff = cutoff_frequency / nyquist_frequency

        # Design the Butterworth filter
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)

        # Apply the filter to each component of the acc data
        filtered_mag = [filtfilt(b, a, np.vstack(self.mag)[:, i]) for i in range(3)]
        filtered_mag = np.column_stack(filtered_mag)
        filtered_mag_list = [filtered_mag[i] for i in range(filtered_mag.shape[0])]

        return IMUTrace(self.timestamps, self.gyro, self.acc, filtered_mag_list)

    def geometric_filter_mag_rotated(self, new_data_weight: float = 0.1) -> List[np.ndarray]:
        """
        This function applies a geometric filter to the mag data in the IMUTrace, using the gyroscope to rotate the
        mag data.
        """
        running = self.mag[0]
        filtered_mag_list = [running]
        for i in range(1, len(self.mag)):
            dt = self.timestamps[i] - self.timestamps[i - 1]
            rot = angular_velocity_to_rotation_matrix(self.gyro[i], dt)
            running = rot.T @ running
            running = running * (1 - new_data_weight) + self.mag[i] * new_data_weight
            filtered_mag_list.append(running)
        return filtered_mag_list

    def get_gyro_axis(self, axis: int) -> np.ndarray:
        """
        Returns the gyro data for a specific axis.
        """
        return np.array([gyro[axis] for gyro in self.gyro])

    def get_acc_axis(self, axis: int) -> np.ndarray:
        """
        Returns the acc data for a specific axis.
        """
        return np.array([acc[axis] for acc in self.acc])

    def get_mag_axis(self, axis: int) -> np.ndarray:
        """
        Returns the mag data for a specific axis.
        """
        return np.array([mag[axis] for mag in self.mag])

    def get_timestamps(self) -> np.ndarray:
        """
        Returns the timestamps.
        """
        return self.timestamps

    def get_sample_frequency(self) -> float:
        """
        Returns the sample frequency of the IMUTrace.
        """
        return 1 / np.mean(np.diff(self.timestamps))

    def calculate_gyro_RMSD(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the root mean square deviation of the gyro data compared to another gyro data.
        """
        assert isinstance(other, IMUTrace), "RMSD can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate RMSD."

        gyro = np.vstack(self.gyro)
        gyro_other = np.vstack(other.gyro)
        return np.sqrt(np.mean((gyro - gyro_other) ** 2, axis=0))

    def calculate_gyro_angle_error(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the angle error between two gyroscopes.
        """
        assert isinstance(other, IMUTrace), "RMSD can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate RMSD."

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

    def calculate_acc_RMSD(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the root mean square deviation of the acc data compared to another acc data.
        """
        assert isinstance(other, IMUTrace), "RMSD can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate RMSD."

        acc = np.vstack(self.acc)
        acc_other = np.vstack(other.acc)
        return np.sqrt(np.mean((acc - acc_other) ** 2, axis=0))

    def calculate_mag_RMSD(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the root mean square deviation of the mag data compared to another mad data.
        """
        assert isinstance(other, IMUTrace), "RMSD can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate RMSD."

        mag = np.vstack(self.mag)
        mag_other = np.vstack(other.mag)
        return np.sqrt(np.mean((mag - mag_other) ** 2, axis=0))

    def calculate_gyro_pearson_correlation(self, other: 'IMUTrace') -> Tuple[float, float, float, float]:
        """
        Calculate the Pearson correlation coefficient of the gyro data.
        """
        assert isinstance(other, IMUTrace), "Pearson correlation can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate Pearson correlation."

        gyro_self = np.vstack(self.gyro)
        gyro_other = np.vstack(other.gyro)
        p_x = scipy.stats.pearsonr(gyro_self[:, 0], gyro_other[:, 0])[0]
        p_y = scipy.stats.pearsonr(gyro_self[:, 1], gyro_other[:, 1])[0]
        p_z = scipy.stats.pearsonr(gyro_self[:, 2], gyro_other[:, 2])[0]
        p_norm = scipy.stats.pearsonr(np.linalg.norm(gyro_self, axis=1), np.linalg.norm(gyro_other, axis=1))[0]
        return p_x, p_y, p_z, p_norm

    def calculate_acc_pearson_correlation(self, other: 'IMUTrace') -> Tuple[float, float, float, float]:
        """
        Calculate the Pearson correlation coefficient of the acc data.
        """
        assert isinstance(other, IMUTrace), "Pearson correlation can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate Pearson correlation."

        acc_self = np.vstack(self.acc)
        acc_other = np.vstack(other.acc)
        p_x = scipy.stats.pearsonr(acc_self[:, 0], acc_other[:, 0])[0]
        p_y = scipy.stats.pearsonr(acc_self[:, 1], acc_other[:, 1])[0]
        p_z = scipy.stats.pearsonr(acc_self[:, 2], acc_other[:, 2])[0]
        p_norm = scipy.stats.pearsonr(np.linalg.norm(acc_self, axis=1), np.linalg.norm(acc_other, axis=1))[0]
        return p_x, p_y, p_z, p_norm

    def calculate_mag_pearson_correlation(self, other: 'IMUTrace') -> Tuple[float, float, float, float]:
        """
        Calculate the Pearson correlation coefficient of the mag data.
        """
        assert isinstance(other, IMUTrace), "Pearson correlation can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate Pearson correlation."

        mag_self = np.vstack(self.mag)
        mag_other = np.vstack(other.mag)
        p_x = scipy.stats.pearsonr(mag_self[:, 0], mag_other[:, 0])[0]
        p_y = scipy.stats.pearsonr(mag_self[:, 1], mag_other[:, 1])[0]
        p_z = scipy.stats.pearsonr(mag_self[:, 2], mag_other[:, 2])[0]
        p_norm = scipy.stats.pearsonr(np.linalg.norm(mag_self, axis=1), np.linalg.norm(mag_other, axis=1))[0]
        return p_x, p_y, p_z, p_norm

    def calculate_rotation_offset_from_gyros(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the rotation offset between two IMUTraces. Returns rotation matrix R_self_other.
        """
        assert isinstance(other, IMUTrace), "Rotation offset can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate rotation offset."

        R_so = calculate_best_fit_rotation(self.gyro, other.gyro)
        return R_so

    def calculate_rotation_offset_from_mags(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the rotation offset between two IMUTraces.
        """
        assert isinstance(other, IMUTrace), "Rotation offset can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have    the same length to calculate rotation offset."
        R_so = calculate_best_fit_rotation(self.mag, other.mag)
        return R_so

    def calculate_rotation_offset_from_gyros_and_mags(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the rotation offset between two IMUTraces.
        """
        assert isinstance(other, IMUTrace), "Rotation offset can only be calculated between two IMUTraces."
        assert len(self) == len(other), "IMUTraces must have the same length to calculate rotation offset."

        R_so = calculate_best_fit_rotation(self.gyro + self.mag, other.gyro + other.mag)
        return R_so

    def calculate_gyro_bias(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the gyro bias.
        """
        gyro = np.vstack(self.gyro)
        other_gyro = np.vstack(other.gyro)
        return np.mean(gyro - other_gyro, axis=0)

    def calculate_acc_bias(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the acc bias.
        """
        acc = np.vstack(self.acc)
        other_acc = np.vstack(other.acc)
        return np.mean(acc - other_acc, axis=0)

    def calculate_mag_bias(self, other: 'IMUTrace') -> np.ndarray:
        """
        Calculate the mag bias.
        """
        mag = np.vstack(self.mag)
        other_mag = np.vstack(other.mag)
        return np.mean(mag - other_mag, axis=0)

    def add_offset_to_gyro(self, offset: np.ndarray) -> 'IMUTrace':
        """
        Add an offset to the gyro data.
        """
        gyro = [gyro + offset for gyro in self.gyro]
        return IMUTrace(self.timestamps, gyro, self.acc, self.mag)

    def add_offset_to_acc(self, offset: np.ndarray) -> 'IMUTrace':
        """
        Add an offset to the acc data.
        """
        acc = [acc + offset for acc in self.acc]
        return IMUTrace(self.timestamps, self.gyro, acc, self.mag)

    def scale_mags(self, sensitivity: np.ndarray) -> 'IMUTrace':
        """
        Add an offset to the mag data.
        """
        mag = [mag * sensitivity for mag in self.mag]
        return IMUTrace(self.timestamps, self.gyro, self.acc, mag)

    def add_offset_to_mag(self, offset: np.ndarray) -> 'IMUTrace':
        """
        Add an offset to the mag data.
        """
        mag = [mag + offset for mag in self.mag]
        return IMUTrace(self.timestamps, self.gyro, self.acc, mag)

    def re_zero_timestamps(self) -> 'IMUTrace':
        """
        Start timestamps at 0
        """
        return IMUTrace(self.timestamps - self.timestamps[0], self.gyro, self.acc, self.mag)

    def resample(self, new_frequency: float):
        new_dt = 1 / new_frequency
        old_dt = np.mean(np.diff(self.timestamps))
        if np.isclose(new_dt, old_dt):
            # No resampling needed if the frequency is the same
            return IMUTrace(self.timestamps, self.gyro, self.acc, self.mag)

        new_timestamps = np.arange(start=self.timestamps[0], stop=self.timestamps[-1] + min(old_dt, new_dt), step=new_dt)
        gyro_interpolator = interp1d(self.timestamps, np.vstack(self.gyro), axis=0, kind='linear', fill_value='extrapolate')
        acc_interpolator = interp1d(self.timestamps, np.vstack(self.acc), axis=0, kind='linear', fill_value='extrapolate')
        mag_interpolator = interp1d(self.timestamps, np.vstack(self.mag), axis=0, kind='linear', fill_value='extrapolate')

        new_gyro = gyro_interpolator(new_timestamps)
        new_acc = acc_interpolator(new_timestamps)
        new_mag = mag_interpolator(new_timestamps)

        return IMUTrace(new_timestamps, new_gyro, new_acc, new_mag)

    @staticmethod
    def load_IMUTraces_from_folder(imu_folder_path: str) -> Dict[str, 'IMUTrace']:
        imu_traces = {}

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
            file_path = os.path.join(imu_folder_path, 'xsens', 'LowerExtremity', file_name)

            # Extract update rate
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

            # Create IMUTrace objects
            imu_traces[name_in_model] = IMUTrace(timestamps=timestamps, acc=acc, gyro=gyro, mag=mag)
        return imu_traces

    def lowpass_filter(self, cutoff: float, order: int) -> 'IMUTrace':
        return IMUTrace(self.timestamps, self.lowpass_filter_gyro(cutoff, order).gyro,
                        self.lowpass_filter_acc(cutoff, order).acc, self.lowpass_filter_mag(cutoff, order).mag)

    def add_noise_to_gyro(self, noise_std: float) -> 'IMUTrace':
        gyro = [gyro + np.random.normal(scale=noise_std, size=gyro.shape) for gyro in self.gyro]
        return IMUTrace(self.timestamps, gyro, self.acc, self.mag)

    def add_noise_to_acc(self, noise_std: float) -> 'IMUTrace':
        acc = [acc + np.random.normal(scale=noise_std, size=acc.shape) for acc in self.acc]
        return IMUTrace(self.timestamps, self.gyro, acc, self.mag)

    def add_noise_to_mag(self, noise_std: float) -> 'IMUTrace':
        mag = [mag + np.random.normal(scale=noise_std, size=mag.shape) for mag in self.mag]
        return IMUTrace(self.timestamps, self.gyro, self.acc, mag)

    def sphere_fit_mag(self, override_radius=None) -> Tuple[np.ndarray, float]:
        """
        Intended to help de-bias a magnetometer using the sphere fitting method.

        Parameters:
        override_radius (float): If provided, this radius will be forced to be used instead of the estimated radius.
                                 This makes it possible to share radii across multiple IMUs.

        Returns:
        bias (np.ndarray): The estimated bias [bx, by, bz].
        radius (float): The estimated radius of the sphere.
        """
        mag_data = np.array(self.mag)

        # Objective function to minimize
        def objective(x):
            cx, cy, cz, r = x
            if override_radius is not None:
                r = override_radius
            distance_errors = np.sqrt((mag_data[:, 0] - cx) ** 2 + (mag_data[:, 1] - cy) ** 2 + (mag_data[:, 2] - cz) ** 2) - r
            loss = np.sum(distance_errors ** 2)
            return loss

        # Initial guess for the parameters [cx, cy, cz, r]
        x0 = np.mean(mag_data, axis=0)
        r0 = np.mean(np.linalg.norm(mag_data - x0, axis=1))
        initial_guess = np.append(x0, r0)

        # Perform the least-squares optimization
        result = minimize(objective, initial_guess.astype(np.float64), tol=1e-12)

        # Extract the parameters
        cx, cy, cz, r = result.x

        if override_radius is not None:
            r = override_radius

        # Calculate the bias
        bias = np.array([cx, cy, cz])

        return bias, r

