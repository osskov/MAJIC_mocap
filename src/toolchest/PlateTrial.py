import os
import numpy as np
import scipy.signal as signal
from .IMUTrace import IMUTrace
from .WorldTrace import WorldTrace
from .KinematicsTrace import KinematicsTrace
from typing import Tuple, List

IMU_TO_TRC_NAME_MAP = {
    'pelvis_imu': 'Pelvis_IMU', 'femur_r_imu': 'R.Femur_IMU', 'femur_l_imu': 'L.Femur_IMU',
    'tibia_r_imu': 'R.Tibia_IMU', 'tibia_l_imu': 'L.Tibia_IMU', 'calcn_r_imu': 'R.Foot_IMU',
    'calcn_l_imu': 'L.Foot_IMU', 'torso_imu': 'Back_IMU'
}

class PlateTrial:
    """
    This class contains a trace of a plate frame over time.
    """

    name: str
    imu_trace: IMUTrace
    world_trace: WorldTrace

    def __init__(self, name: str, imu_trace: IMUTrace, world_trace: WorldTrace):
        assert len(imu_trace) == len(world_trace), "IMU and World traces must have the same length"
        if max(np.abs(imu_trace.timestamps - world_trace.timestamps)) > 1e-8:
            print(f"IMU and World traces must have the same timestamps. Max difference: {max(np.abs(imu_trace.timestamps - world_trace.timestamps))}")
        assert max(np.abs(imu_trace.timestamps - world_trace.timestamps)) < 1e-8, "Timestamps must match"
        assert isinstance(imu_trace, IMUTrace)
        assert isinstance(world_trace, WorldTrace)

        self.name = name
        self.imu_trace = imu_trace
        self.world_trace = world_trace


    def __len__(self):
        return len(self.imu_trace)

    def __getitem__(self, key):
        return PlateTrial(self.name, self.imu_trace[key], self.world_trace[key])
    
    def copy(self) -> 'PlateTrial':
        """
        Returns a deep copy of the PlateTrial object.

        Note: This relies on IMUTrace and WorldTrace having a functioning .copy() method
                to ensure the underlying data arrays are also duplicated.
        """
        return PlateTrial(
            self.name,
            self.imu_trace.copy(),
            self.world_trace.copy()
        )

    def _align_world_trace_to_imu_trace(self) -> 'PlateTrial':
        synthetic_imu_trace = self.world_trace.calculate_imu_trace()
        R_wt_it = synthetic_imu_trace.calculate_rotation_offset_from_gyros(self.imu_trace)
        new_world_rotations = [R_wt_it.T @ rot for rot in self.world_trace.rotations]
        new_world_trace = WorldTrace(self.world_trace.timestamps, self.world_trace.positions, new_world_rotations)
        return PlateTrial(self.name, self.imu_trace, new_world_trace)

    def project_imu_trace(self, local_offset: np.ndarray) -> IMUTrace:
        """
        This function estimates the values for an IMUTrace at a different location relative to the plate.
        """
        return self.imu_trace.project_acc(local_offset)

    @staticmethod
    def load_trial_from_folder(folder_path: str, align_plate_trials=True) -> List['PlateTrial']:
        imu_traces = IMUTrace.load_IMUTraces_from_folder(os.path.join(folder_path, 'IMU'))

        trc_file_path = [os.path.join(folder_path, 'Mocap', file)
            for file in os.listdir(os.path.join(folder_path, 'Mocap'))
            if '.trc' in file][0]

        world_traces = WorldTrace.load_from_trc_file(trc_file_path)

        kinematics_file_path = [
            os.path.join(folder_path, 'Mocap', 'ikResults', file)
            for file in os.listdir(os.path.join(folder_path, 'Mocap', 'ikResults'))
            if '.mot' in file][0]

        kinematics_trace = KinematicsTrace.load_kinematics_from_mot_file(kinematics_file_path)

        plate_trials = []
        imu_slice, world_slice = slice(0, 0), slice(0, 0)
        for imu_name, imu_trace in imu_traces.items():
            try:
                world_trace = world_traces[IMU_TO_TRC_NAME_MAP[imu_name]]
            except KeyError:
                print(f"IMU {imu_name} not found in TRC file")
                continue

            if abs(imu_trace.get_sample_frequency() - world_trace.get_sample_frequency()) > 0.2:
                # print(f"Sample frequency mismatch for {imu_name}: IMU {imu_trace.get_sample_frequency()} Hz, World {world_trace.get_sample_frequency()} Hz")
                imu_trace = imu_trace.resample(float(world_trace.get_sample_frequency()))

            if imu_slice == slice(0, 0) and world_slice == slice(0, 0):
                imu_slice, world_slice = PlateTrial._sync_traces(imu_trace, world_trace)
            synced_imu_trace = imu_trace[imu_slice].re_zero_timestamps()
            synced_world_trace = world_trace[world_slice].re_zero_timestamps()
            new_plate_trial = PlateTrial(imu_name, synced_imu_trace, synced_world_trace)
            if align_plate_trials:
                new_plate_trial = new_plate_trial._align_world_trace_to_imu_trace()
            if plate_trials and len(plate_trials[0]) != len(new_plate_trial):
                raise ValueError("All PlateTrials must have the same length")
            plate_trials.append(new_plate_trial)

        synced_kinematics_file_path = kinematics_file_path.replace('.mot', '_trimmed.mot')
        synced_kinematics_trace = kinematics_trace[world_slice].re_zero_timestamps()
        synced_kinematics_trace.write_kinematics_to_mot(synced_kinematics_file_path)
        return plate_trials

    @staticmethod
    def _sync_traces(imu_trace: IMUTrace, world_trace: WorldTrace) -> Tuple[slice, slice]:
        if not np.isclose(imu_trace.get_sample_frequency(), world_trace.get_sample_frequency(), rtol=0.2):
            imu_trace = imu_trace.resample(float(world_trace.get_sample_frequency()))

        synthetic_imu_trace = world_trace.calculate_imu_trace()
        imu_slice, world_slice = PlateTrial._sync_arrays(
            np.linalg.norm(imu_trace.gyro, axis=1),
            np.linalg.norm(synthetic_imu_trace.gyro, axis=1)
        )
        return imu_slice, world_slice

    @staticmethod
    def _sync_arrays(array1: np.ndarray, array2: np.ndarray) -> Tuple[slice, slice]:
        assert array1.ndim == array2.ndim == 1
        max_len = max(len(array1), len(array2))
        a1 = np.pad(array1, (0, max_len - len(array1)), mode='constant')
        a2 = np.pad(array2, (0, max_len - len(array2)), mode='constant')
        lag = np.argmax(signal.correlate(a1, a2, mode='full')) - (max_len - 1)
        i1, i2 = max(0, lag), max(0, -lag)
        new_len = min(len(array1) - i1, len(array2) - i2)
        return slice(i1, i1 + new_len), slice(i2, i2 + new_len)