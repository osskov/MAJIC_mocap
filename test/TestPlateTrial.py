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
        test_osim_file = os.path.join('..', 'data', 'ODay_Data', 'Subject02', 'complexTasks')
        plates = PlateTrial.load_trial_from_folder(test_osim_file)
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

if __name__ == '__main__':
    unittest.main()
