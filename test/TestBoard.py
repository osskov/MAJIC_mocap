import unittest
import numpy as np
from src.Board import RawReading, ProjectedReading, Board

class TestBoard(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.gyros = [np.array([0.1, 0.2, 0.3])] * 2
        self.acc_offsets = [np.array([0.01, 0.02, 0.03]), np.array([0.04, 0.05, 0.06])]
        self.accs = [np.array([0.0, 9.8, 0.0])] * len(self.acc_offsets)
        self.mag_offsets = [np.array([0.001, 0.002, 0.003]), np.array([0.004, 0.005, 0.006])]
        self.mags = [np.array([1.0, 0.0, 0.0])] * len(self.mag_offsets)

        # Creating Board object
        self.board = Board(self.acc_offsets, self.mag_offsets)
        self.board.acc_history = [self.accs[0]] * self.board.history_length
        self.board.gyro_history = [self.gyros[0]] * self.board.history_length
        self.board.timestamps = [i for i in range(self.board.history_length)]

        self.raw_reading = RawReading(self.gyros, self.accs, self.mags, self.board.timestamps[-1] + 1.0)

    def test_estimate_alpha(self):
        """ Test that the alpha estimate is of correct format and shape."""
        alpha = self.board._estimate_alpha_()
        assert isinstance(alpha, np.ndarray)
        assert alpha.shape == (3,)

    def test_estimate_alpha_no_rotation(self):
        """ Test that the alpha estimate is zero when no rotation is applied. """
        self.board.gyro_history = [np.zeros(3)] * self.board.history_length
        alpha = self.board._estimate_alpha_()
        np.testing.assert_allclose(alpha, np.zeros(3))

    def test_project_reading(self):
        """ Test that the projected reading is of the correct format and shape. """
        projected_reading = self.board.project_reading_to_joint_center(self.raw_reading)
        assert isinstance(projected_reading, ProjectedReading)

    def test_to_and_from_dict(self):
        """ Test that the Board can be converted to and from a dictionary. """
        board_dict = self.board.to_dict()
        new_board = Board.from_dict(board_dict)
        assert isinstance(new_board, Board)
        assert new_board == self.board

    def test_project_reading_skip_acc(self):
        """ Test that the projected reading is the same as the mean of the raw reading when skipping acceleration."""
        board = Board(self.acc_offsets, self.mag_offsets)
        acc = [np.array([0.0, 9.81, 0.0]), np.array([0.1, 9.8, 0.1])]
        time = 0.0
        raw_reading = RawReading([np.zeros(3)], acc, [np.zeros(3)], time)

        projected_reading = board.project_reading_to_joint_center(raw_reading, skip_acc=True, skip_mag=True)
        expected_acc = np.array([0.05, 9.805, 0.05])
        np.testing.assert_allclose(projected_reading.acc, expected_acc)

    def test_project_reading_skip_mag(self):
        """ Test that the projected reading is the same as the mean of the raw reading when skipping magnetometer."""
        board = Board(self.acc_offsets, self.mag_offsets)
        mag = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        time = 0.0
        raw_reading = RawReading([np.zeros(3)], [np.zeros(3)], mag, time)

        projected_reading = board.project_reading_to_joint_center(raw_reading, skip_acc=True, skip_mag=True)
        expected_mag = np.array([0.5, 0.5, 0.0])
        np.testing.assert_allclose(projected_reading.mag, expected_mag)

    def test_project_reading_with_acc(self):
        gyro = [np.array([1, 0, 0])] * 5
        acc = [np.array([0, 0, 0])]
        offsets = [np.array([0, 1, 0])] * len(acc)
        board = Board(acc_to_joint_center_offsets=offsets, mag_to_joint_center_offsets=[np.zeros(3)])

        for i in range(board.history_length + 2):
            raw_reading = RawReading(gyro, acc, [np.zeros(3)], i)
            projected_reading = board.project_reading_to_joint_center(raw_reading, skip_mag=True)
            # We expect to always accelerate back towards the center of rotation, which means in the negative direction of the
            # offset.
            expected_acc = np.array([0, -1, 0])
            np.testing.assert_allclose(projected_reading.acc, expected_acc)





class TestRawReading(unittest.TestCase):
        def setUp(self):
            # Sample data for testing
            self.gyros = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
            self.accs = [np.array([0.0, 9.81, 0.0]), np.array([0.1, 9.8, 0.1])]
            self.mags = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
            self.timestamp = 123456789.0

        def test_initialization(self):
            """Test that RawReading is correctly initialized"""
            raw_reading = RawReading(self.gyros, self.accs, self.mags, self.timestamp)
            self.assertEqual(raw_reading.gyros, self.gyros)
            self.assertEqual(raw_reading.accs, self.accs)
            self.assertEqual(raw_reading.mags, self.mags)
            self.assertEqual(raw_reading.timestamp, self.timestamp)

        def test_project_accs_offset_mismatch(self):
            """ Test that the correct error is raised when the number of acc offsets and accs do not match. """
            raw_reading = RawReading(self.gyros, self.accs, self.mags, self.timestamp)
            alpha = np.zeros(3)
            offsets = [np.array([0.01, 0.02, 0.03])] * (len(self.accs) + 1)
            with self.assertRaises(AssertionError):
                raw_reading.project_accs(alpha=alpha, offsets=offsets)
            offsets = [np.array([0.01, 0.02])] * (len(self.accs) - 1)
            with self.assertRaises(AssertionError):
                raw_reading.project_accs(alpha=alpha, offsets=offsets)

        def test_project_accs_no_rotation(self):
            """ Test acceleration is the same when no rotation is applied. """
            acc = np.array([0.0, 9.81, 0.0])
            accs = [acc]
            gyro = [np.zeros(3)]
            raw_reading = RawReading(gyro, accs, [], 0.0)

            alpha = np.zeros(3)
            offsets = [np.array([0.01, 0.02, 0.03])] * len(accs)

            projected_acc = raw_reading.project_accs(alpha=alpha, offsets=offsets)

            np.testing.assert_allclose(acc, projected_acc)

        def test_project_invertible(self):
            acc = np.array([0.0, 9.81, 0.0])
            accs = [acc]
            gyro = np.array([0.1, 0.2, 0.3])
            gyros = [gyro] * 3
            raw_reading = RawReading(gyros=gyros, accs=accs, mags=[], timestamp=0.0)

            offset = np.array([0.01, 0.02, 0.03])
            acc_offsets = [offset] * len(accs)
            alpha = np.array([0.1, 0.2, 0.3])

            projected_acc = raw_reading.project_accs(alpha=alpha, offsets=acc_offsets)
            projected_reading = RawReading(gyros=gyros, accs=[projected_acc], mags=[], timestamp=0.0)
            returned_acc = projected_reading.project_accs(alpha=alpha, offsets=[-offset])
            np.testing.assert_allclose(acc, returned_acc)

        def test_projection_orthogonal_effect(self):
            gyro = [np.array([1, 0, 0])] * 5
            acc = [np.array([0, 0, 0])]
            raw_reading = RawReading(gyro, acc, [], 0.0)
            offsets = [np.array([0, 1, 0])] * len(acc)
            projected_acc = raw_reading.project_accs(alpha=np.zeros(3), offsets=offsets)
            # We expect to accelerate back towards the center of rotation, which means in the negative direction of the
            # offset.
            expected_acc = np.array([0, -1, 0])
            np.testing.assert_allclose(projected_acc, expected_acc)

        def test_project_mags_wrong_inputs(self):
            mag = [np.array([0, 0, 1])]

            raw_reading = RawReading([], [], mag, 0.0)
            offsets = [np.array([0, 0, 0])]

            with self.assertRaises(AssertionError):
                # Wrong number of mags
                raw_reading.project_mags(offsets=offsets)

            mag = [np.array([0, 0, 1]), np.array([0, 0, 2])]
            raw_reading = RawReading([], [], mag, 0.0)
            offsets = [np.array([0, 0, 0])]

            with self.assertRaises(AssertionError):
                # Wrong number of offsets
                raw_reading.project_mags(offsets=offsets)

        def test_project_mag_possible_offsets(self):
            mag = [np.array([0, 0, 1]), np.array([0, 0, 2])]

            raw_reading = RawReading([], [], mag, 0.0)

            offsets = [np.array([0, 0, 0]), np.array([0, 0, 0])]
            with self.assertRaises(AssertionError):
                # Two mags should not be in the exact same location
                raw_reading.project_mags(offsets=offsets)

            # Projecting to one of the mag locations should just return the mag
            offsets = [np.array([0, 0, 0]), np.array([1., 0., 0.])]
            projected_mag = raw_reading.project_mags(offsets=offsets)
            np.testing.assert_allclose(projected_mag, mag[0], err_msg="Projecting to the same mag location should not change the mag")
            offsets = [np.array([1., 0, 0]), np.array([0., 0., 0.])]
            projected_mag = raw_reading.project_mags(offsets=offsets)
            np.testing.assert_allclose(projected_mag, mag[1], err_msg="Projecting to the same mag location should not change the mag")

            # Projecting to the middle should be the average of the two mags
            offsets = [np.array([0.5, 0, 0]), np.array([-0.5, 0., 0.])]
            projected_mag = raw_reading.project_mags(offsets=offsets)
            expected_mag = np.array([0, 0, 1.5])
            np.testing.assert_allclose(projected_mag, expected_mag, err_msg="Projecting to the middle should be the average of the two mags")

            # Projecting orthogonal to offset between the mags should not change the mag
            offsets = [np.array([0.5, 1., 0]), np.array([-0.5, 1., 0.])]
            projected_mag = raw_reading.project_mags(offsets=offsets)
            np.testing.assert_allclose(projected_mag, expected_mag, err_msg="Projecting orthogonal to offset between the mags should not change the mag")

            # Projecting outside the mags should perform linear interpolation
            offsets = [np.array([2., 0, 0]), np.array([1., 0., 0.])]
            projected_mag = raw_reading.project_mags(offsets=offsets)
            expected_mag = np.array([0, 0, 3])
            np.testing.assert_allclose(projected_mag, expected_mag, err_msg="Projecting outside the mags should perform linear interpolation")

if __name__ == "__main__":
    unittest.main()