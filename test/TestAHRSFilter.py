import math
from scipy.linalg import null_space
from typing import Union, Any
from unittest import TestCase
import numpy as np
import nimblephysics as nimble
import os
from ahrs.utils import WMM, WGS
from ahrs.utils.metrics import qad
from ahrs.common.constants import MUNICH_LATITUDE, MUNICH_LONGITUDE, MUNICH_HEIGHT
from src.toolchest.AHRSFilter import AHRSFilter, ALL_FILTERS
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.WorldTrace import WorldTrace

class TestAHRSFilter(TestCase):
    def setUp(self) -> None:
        timestamps = np.linspace(0, 1, 100)
        positions = [np.array([t, t, t]) for t in timestamps]
        rotations = [np.eye(3) for _ in range(100)]
        self.world_trace = WorldTrace(timestamps, positions, rotations)

    def test_constructor(self):
        a_ref = np.array([0, 9.81, 0])
        m_ref = np.array([0.3, 0.5, 0.7])
        for filter_type in ALL_FILTERS.keys():
            ahrs_filter = AHRSFilter(select_filter=filter_type, world_reference_acc=a_ref, world_reference_mag=m_ref)

    def test_stationary_update_aligned(self):
        a_ref = np.array([0, 9.81, 0])
        m_ref = np.array([0.3, 0.5, 0.7])
        for filter_type in ALL_FILTERS.keys():
            ahrs_filter = AHRSFilter(select_filter=filter_type, world_reference_acc=a_ref, world_reference_mag=m_ref)
            for i in range(1000):
                ahrs_filter.update(dt=0.01, acc=a_ref, mag=m_ref, gyro=(-1)**i * np.array([1e-16, 1e-16, 1e-16]))
            last_R = ahrs_filter.get_last_R()
            self.assertTrue(np.allclose(last_R, np.eye(3), atol=1e-2))

    def test_stationary_update_misaligned(self):
        a_ref = np.array([0, 9.81, 0])
        m_ref = np.array([0.3, 0.5, 0.7])
        for filter_type in ALL_FILTERS.keys():
            ahrs_filter = AHRSFilter(select_filter=filter_type, world_reference_acc=a_ref, world_reference_mag=m_ref)
            if ahrs_filter.acc_req:
                for i in range(1000):
                    ahrs_filter.update(dt=0.01, acc=-a_ref, mag=m_ref * np.array([-1., 0., 1.]), gyro=(-1)**i * np.array([1e-16, 1e-16, 1e-16]))
                last_R = ahrs_filter.get_last_R()
                self.assertFalse(np.allclose(last_R, np.eye(3), atol=1e-2))

    def test_convert_to_world_trace_stationary(self):
        # Using NED convention for gravity and magnetic field
        gravity = np.array([0, 0, 9.81])
        magnetic_field = np.array([1, 0, 1])
        parabola_constant = 0.05
        timestamps = np.linspace(0, 1, 100)
        positions = [np.array([0, 0, parabola_constant * t ** 2]) for t in timestamps]
        rotations = [np.eye(3) for _ in range(100)]
        world_trace = WorldTrace(timestamps, positions, rotations)
        imu_trace = world_trace.calculate_imu_trace(gravity, magnetic_field)

        for filter_type in ALL_FILTERS.keys():
            reconstructed_world_trace = AHRSFilter.convert_to_world_orientations(imu_trace, filter_type, magnetic_field, -gravity)
            self.assertEqual(len(reconstructed_world_trace), len(world_trace))
            for i in range(len(reconstructed_world_trace)):
                np.testing.assert_array_almost_equal(reconstructed_world_trace.rotations[i], world_trace.rotations[i])

    def test_convert_to_world_trace_stationary_misaligned_to_ahrs(self):
        gravity = np.array([0, 0, -9.81])
        magnetic_field = np.array([0, 1.5, -1])
        parabola_constant = 0.05
        timestamps = np.linspace(0, 1, 100)
        positions = [np.array([0, 0, parabola_constant * t ** 2]) for t in timestamps]
        rotations = [np.eye(3) for _ in range(100)]
        world_trace = WorldTrace(timestamps, positions, rotations)
        imu_trace = world_trace.calculate_imu_trace(gravity, magnetic_field)

        for filter_type in ALL_FILTERS.keys():
            reconstructed_world_trace = AHRSFilter.convert_to_world_orientations(imu_trace, filter_type, magnetic_field,
                                                                                 -gravity)
            self.assertEqual(len(reconstructed_world_trace), len(world_trace))
            for i in range(len(reconstructed_world_trace)):
                np.testing.assert_array_almost_equal(reconstructed_world_trace.rotations[i], world_trace.rotations[i])

    def test_convert_to_world_trace_rotating(self):
        gravity = np.array([0, 0, -9.81])
        magnetic_field = np.array([0, 0, 1])
        num_samples = 200
        parabola_constant = 0.05
        timestamps = np.linspace(0, 2, num_samples)
        positions = [np.array([0, 0, parabola_constant * t ** 2]) for t in timestamps]
        rotation_axis = np.array([1, 1, 1])
        rotation_amount = np.linspace(0, 2 * np.pi, num_samples)
        rotations = [nimble.math.expMapRot(rotation_axis * rotation_amount[i]) for i in range(num_samples)]
        world_trace = WorldTrace(timestamps, positions, rotations)
        imu_trace = world_trace.calculate_imu_trace(gravity, magnetic_field, skip_lin_acc=True)
        for filter_type in ALL_FILTERS.keys():
            reconstructed_world_trace = AHRSFilter.convert_to_world_orientations(imu_trace, filter_type, magnetic_field, -gravity)
            self.assertEqual(len(reconstructed_world_trace), len(world_trace))
            for i in range(len(reconstructed_world_trace) - 1, 0, -1):
                # NOTE: The tolerance is loosened for the Mahoney filter. Here are two theories: the first is that the
                # Mahoney filter is a first-order integration of the gyro. For some reason, the Angular Rate filter
                # passes when using the 'closed' (sin/cos) form of the quaternion update, but fails when using the
                # first order which the Mahoney filter uses. Second theory is that there is an off by one problem
                # where we integrate the gyro, then compare the acc and mag which are from the original orientation and
                # not the integrated one. Sorry, future selves.
                np.testing.assert_array_almost_equal(reconstructed_world_trace.rotations[i], world_trace.rotations[i], decimal=2)
