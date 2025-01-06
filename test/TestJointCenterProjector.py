import numpy as np

from src.cascade_filter.JointCenterProjector import JointCenterProjector
from src.cascade_filter.Board import Board, ProjectedReading
import unittest

class TestJointCenterProjector(unittest.TestCase):
    def setUp(self):
        self.offset_1 = [np.random.rand(3)] * 3
        self.offset_2 = [np.random.rand(3)] * 3
        self.parent_board = Board(self.offset_1, self.offset_2)
        self.offset_3 = [np.random.rand(3)] * 3
        self.offset_4 = [np.random.rand(3)] * 3
        self.child_board = Board(self.offset_3, self.offset_4)
        self.observability_threshold = 0.5
        self.joint_center_projector = JointCenterProjector(self.parent_board, self.child_board, self.observability_threshold)

    def test_to_from_dict(self):
        joint_center_projector_dict = self.joint_center_projector.to_dict()
        new_joint_center_projector = JointCenterProjector.from_dict(joint_center_projector_dict)
        self.assertEqual(self.joint_center_projector, new_joint_center_projector)

    def test_choose_mag(self):
        gyro = np.zeros(3)
        acc = np.array([1., 0., 0.])
        mag = np.array([1., 0., 0.])
        alpha = np.zeros(3)
        acc_change = np.array([0., 1., 0.])
        timestamp = 0

        parent_reading = ProjectedReading(gyro=gyro, acc=acc, mag=mag, timestamp=timestamp, alpha=alpha, acc_change=acc_change)
        child_reading = ProjectedReading(gyro=gyro, acc=acc, mag=mag, timestamp=timestamp, alpha=alpha, acc_change=acc_change)
        unprojected_mag = np.array([1., 0., 0.])

        # Acc change is orthogonal to acc, so the motion is observed
        chosen_mag_parent, chosen_mag_child = self.joint_center_projector._choose_mag_(parent_reading, child_reading, unprojected_mag, unprojected_mag)
        self.assertTrue(np.array_equal(chosen_mag_parent, np.zeros(3)), msg="When the observability threshold is met by both readings, the chosen mag should be zero.")
        self.assertTrue(np.array_equal(chosen_mag_child, np.zeros(3)), msg="When the observability threshold is met by both readings, the chosen mag should be zero.")

        # Acc change is parallel to acc for one sensor, so the motion is not observed by one sensor
        parent_reading.acc_change = np.array([1., 0., 0.])
        chosen_mag_parent, chosen_mag_child = self.joint_center_projector._choose_mag_(parent_reading, child_reading, unprojected_mag, unprojected_mag)
        self.assertTrue(np.array_equal(chosen_mag_parent, unprojected_mag), msg="When the observability threshold is not met by any reading, the chosen mag should nonzero.")
        self.assertTrue(np.array_equal(chosen_mag_child, unprojected_mag), msg="When the observability threshold is not met by any reading, the chosen mag should nonzero.")

        # Acc change is parallel to acc for both sensors, so the motion is not observed by both sensors
        child_reading.acc_change = np.array([1., 0., 0.])
        chosen_mag_parent, chosen_mag_child = self.joint_center_projector._choose_mag_(parent_reading, child_reading, unprojected_mag, unprojected_mag)
        self.assertTrue(np.array_equal(chosen_mag_parent, unprojected_mag), msg="When the observability threshold is not met by either reading, the chosen mag should nonzero.")
        self.assertTrue(np.array_equal(chosen_mag_child, unprojected_mag), msg="When the observability threshold is not met by either reading, the chosen mag should nonzero.")

        # Acc change is parallel to acc for both sensors, so the motion is not observed by both sensors
        # The unprojected mag norm difference is less than the projected mag norm difference
        child_reading.mag = np.array([2., 0., 0.])
        chosen_mag_parent, chosen_mag_child = self.joint_center_projector._choose_mag_(parent_reading, child_reading, unprojected_mag, unprojected_mag)
        self.assertTrue(np.array_equal(chosen_mag_parent, unprojected_mag), msg="When the unprojected mag norm difference is less than the projected mag norm difference, the chosen mag should be the unprojected mag.")
        self.assertTrue(np.array_equal(chosen_mag_child, unprojected_mag), msg="When the unprojected mag norm difference is less than the projected mag norm difference, the chosen mag should be the unprojected mag.")

        # The projected mag norm difference is less than the unprojected mag norm difference
        parent_reading.mag = np.array([2., 0., 0.])
        chosen_mag_parent, chosen_mag_child = self.joint_center_projector._choose_mag_(parent_reading, child_reading, unprojected_mag * 8, unprojected_mag)
        self.assertTrue(np.array_equal(chosen_mag_parent, parent_reading.mag), msg="When the projected mag norm difference is less than the unprojected mag norm difference, the chosen mag should be the projected mag.")
        self.assertTrue(np.array_equal(chosen_mag_child, child_reading.mag), msg="When the projected mag norm difference is less than the unprojected mag norm difference, the chosen mag should be the projected mag.")


