import unittest
import os

import numpy as np
import nimblephysics as nimble

from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.WorldTrace import WorldTrace

from typing import List, Tuple


class TestPlateTrial(unittest.TestCase):
    def test_load_trace(self):
        test_directory = os.getcwd()
        test_osim_file = os.path.join(test_directory, '..', 'test_data', 'oday_data', 'Subject02', 'complexTasks')
        plates = PlateTrial.load_trial_from_folder(test_osim_file)
        self.assertIsNotNone(plates)

    def test_load_cheeseburger_trace(self):
        test_directory = os.getcwd()
        test_folder = os.path.join(test_directory, '..', 'test_data', 'pilot_test')
        plates = PlateTrial.load_cheeseburger_trial_from_folder(test_folder, 'Clean_Segment_0.trc')
        self.assertIsNotNone(plates)

    def test_identical_arrays(self):
        array1 = [1, 2, 3, 4, 5]
        array2 = [1, 2, 3, 4, 5]
        expected_slice1 = slice(0, 5)
        expected_slice2 = slice(0, 5)
        slice1, slice2 = PlateTrial._sync_arrays(np.array(array1), np.array(array2))
        self.assertEqual(expected_slice1, slice1)
        self.assertEqual(expected_slice2, slice2)

    def test_offset_arrays(self):
        array1 = [1., 2, 3, 4, 5, 6, 7, -4]
        array2 = [0.1, 1, 2, 3, 4, 5, 6, 7]
        expected_slice1 = slice(0, 7)
        expected_slice2 = slice(1, 8)
        slice1, slice2 = PlateTrial._sync_arrays(np.array(array1), np.array(array2))
        self.assertEqual(expected_slice1, slice1)
        self.assertEqual(expected_slice2, slice2)

        # Try swapping order of arrays
        slice1, slice2 = PlateTrial._sync_arrays(np.array(array2), np.array(array1))
        self.assertEqual(expected_slice2, slice1)
        self.assertEqual(expected_slice1, slice2)

    def test_no_overlap(self):
        array1 = [1, 2, 3, 4, 5]
        array2 = [6, 7, 8, 9, 10]
        expected_slice1 = slice(0, 5)
        expected_slice2 = slice(0, 5)
        slice1, slice2 = PlateTrial._sync_arrays(np.array(array1), np.array(array2))
        self.assertEqual(expected_slice1, slice1)
        self.assertEqual(expected_slice2, slice2)

    def test_sync_arrays_long_function(self):
        len = 250
        start = 100
        t = np.linspace(0, 100, len + start)
        base_function = np.sin(t) + np.sin(17*t) - np.cos(3*t) - np.cos(60*t)
        expected_slice1 = slice(start, start + len)
        expected_slice2 = slice(0, len)
        array1 = base_function
        array2 = base_function[expected_slice1]

        slice1, slice2 = PlateTrial._sync_arrays(np.array(array1), np.array(array2))
        self.assertEqual(expected_slice1, slice1)
        self.assertEqual(expected_slice2, slice2)

    def test_estimate_world_magnetic_field(self):
        random_rotations = [nimble.math.expMapRot(np.random.rand(3)) for _ in range(100)]
        zeros = [np.zeros(3) for _ in range(100)]
        world_magnetic_field = np.array([0, 0, 1])
        local_magnetic_fields = [rotation.T @ world_magnetic_field for rotation in random_rotations]

        imu_trace = IMUTrace(np.linspace(0, 1, 100), zeros, zeros, local_magnetic_fields)
        world_trace = WorldTrace(np.linspace(0, 1, 100), zeros, random_rotations)
        plate_trial = PlateTrial('test', imu_trace, world_trace)

        recovered_magnetic_field = plate_trial.estimate_world_magnetic_field()

        np.testing.assert_allclose(recovered_magnetic_field, world_magnetic_field, atol=1e-6)

    def test_project(self):
        timestamps = np.array([0, 1, 2, 3, 4])
        gyro = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3]), np.array([4, 4, 4])]
        acc = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3]), np.array([4, 4, 4])]
        mag_1 = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3]), np.array([4, 4, 4])]
        mag_2 = [mag + np.array([0.5, 0.5, 0.5]) for mag in mag_1]
        imu_trace_1 = IMUTrace(timestamps, gyro, acc, mag_1)
        imu_trace_2 = IMUTrace(timestamps, gyro, acc, mag_2)
        world_trace = WorldTrace(timestamps, [np.zeros(3) for _ in range(len(timestamps))], [np.eye(3) for _ in range(len(timestamps))])
        plate_trial = PlateTrial('test', imu_trace_1, world_trace, imu_trace_2)
        plate_trial.imu_offset = np.array([0, -0.5, 0])
        plate_trial.second_imu_offset = np.array([0, 0.5, 0])

        recovered_mag_1 = plate_trial.project_imu_trace(plate_trial.imu_offset, skip_acc=True).mag
        for mag, recovered_mag in zip(mag_1, recovered_mag_1):
            np.testing.assert_allclose(mag, recovered_mag, atol=1e-6)
        recovered_mag_2 = plate_trial.project_imu_trace(plate_trial.second_imu_offset, skip_acc=True).mag
        for mag, recovered_mag in zip(mag_2, recovered_mag_2):
            np.testing.assert_allclose(mag, recovered_mag, atol=1e-6)

        mag_3 = [mag + np.array([1.0, 1.0, 1.0]) for mag in mag_1]
        recovered_mag_3 = plate_trial.project_imu_trace(np.array([0, 1.5, 0]), skip_acc=True).mag
        for mag, recovered_mag in zip(mag_3, recovered_mag_3):
            np.testing.assert_allclose(mag, recovered_mag, atol=1e-6)

    def test_calibrate_group_plate_mags(self):
        test_directory = os.getcwd()
        test_folder = os.path.join(test_directory, '..', 'test_data', 'pilot_test')
        plates = PlateTrial.load_cheeseburger_trial_from_folder(test_folder, 'Clean_Segment_0.trc')
        joint_segment_dict = {'hip': ('Pelvis', 'Thigh'),
                              'knee': ('Thigh', 'Shank'),
                              'ankle': ('Shank', 'Foot')}
        joint_index_pairs: List[Tuple[int, int]] = []
        for joint_name, segment_tuple in joint_segment_dict.items():
            # Find the two relevant plate trials
            parent_name, child_name = segment_tuple
            parent_trial = None
            child_trial = None
            for plate_trial in plates:
                if parent_trial is not None and child_trial is not None:
                    break
                if plate_trial.name == parent_name and parent_trial is None:
                    parent_trial = plate_trial
                elif plate_trial.name == child_name and child_trial is None:
                    child_trial = plate_trial
            if parent_trial is None or child_trial is None:
                continue
            joint_index_pairs.append((plates.index(parent_trial), plates.index(child_trial)))

        new_plates = PlateTrial.calibrate_group_plate_mags(plates, joint_index_pairs)

    def test_calibrate_group_plate_mags_with_mocap(self):
        test_directory = os.getcwd()
        test_folder = os.path.join(test_directory, '..', 'test_data', 'pilot_test')
        plates = PlateTrial.load_cheeseburger_trial_from_folder(test_folder, 'Clean_Segment_0.trc')
        joint_segment_dict = {'hip': ('Pelvis', 'Thigh'),
                              'knee': ('Thigh', 'Shank'),
                              'ankle': ('Shank', 'Foot')}
        joint_index_pairs: List[Tuple[int, int]] = []
        for joint_name, segment_tuple in joint_segment_dict.items():
            # Find the two relevant plate trials
            parent_name, child_name = segment_tuple
            parent_trial = None
            child_trial = None
            for plate_trial in plates:
                if parent_trial is not None and child_trial is not None:
                    break
                if plate_trial.name == parent_name and parent_trial is None:
                    parent_trial = plate_trial
                elif plate_trial.name == child_name and child_trial is None:
                    child_trial = plate_trial
            if parent_trial is None or child_trial is None:
                continue
            joint_index_pairs.append((plates.index(parent_trial), plates.index(child_trial)))

        new_plates = PlateTrial.calibrate_group_plate_mags_with_mocap(plates, joint_index_pairs)

    def test_align_world_trace_to_imu_trace(self):
        pass

    def test_align_imu_trace_to_world_trace(self):
        pass

if __name__ == '__main__':
    unittest.main()
