import unittest
import numpy as np
from src.toolchest.MagnetometerCalibration import MagnetometerCalibration


class TestMagnetometerCalibration(unittest.TestCase):

    def setUp(self):
        self.calibration = MagnetometerCalibration()
        self.observations = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])

    def test_initialize(self):
        self.calibration.initialize(self.observations)
        np.testing.assert_array_almost_equal(self.calibration.center, np.zeros(3))
        self.assertAlmostEqual(self.calibration.radius, 1.0)
        self.assertEqual(self.calibration.scaling, [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(self.calibration.axis, [np.array([1.0, 0.0, 0.0]),
                                                                     np.array([0.0, 1.0, 0.0]),
                                                                     np.array([0.0, 0.0, 1.0])])

    def test_process(self):
        self.calibration.initialize(self.observations)
        transformed = self.calibration.process(self.observations)
        np.testing.assert_array_almost_equal(transformed, self.observations)

    def test_get_error(self):
        self.calibration.initialize(self.observations)
        error = self.calibration.get_error(self.observations)
        np.testing.assert_array_almost_equal(error, np.zeros(len(self.observations)))

    def test_get_matrix(self):
        expected_matrix = np.eye(3)
        np.testing.assert_array_almost_equal(self.calibration.get_matrix(), expected_matrix)

    def test_get_constraints(self):
        constraints = self.calibration.get_constraints()
        expected_constraints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        np.testing.assert_array_almost_equal(constraints, expected_constraints)

    def test_pack_array(self):
        self.calibration.initialize(self.observations)
        packed_array = self.calibration.pack_array()
        expected_array = np.concatenate([self.calibration.center, [self.calibration.radius],
                                         self.calibration.scaling, self.calibration.axis[0],
                                         self.calibration.axis[1], self.calibration.axis[2]])
        np.testing.assert_array_almost_equal(packed_array, expected_array)

    def test_unpack_array(self):
        self.calibration.initialize(self.observations)
        packed_array = self.calibration.pack_array()
        new_calibration = MagnetometerCalibration()
        new_calibration.unpack_array(packed_array)
        np.testing.assert_array_almost_equal(new_calibration.center, self.calibration.center)
        self.assertAlmostEqual(new_calibration.radius, self.calibration.radius)
        self.assertEqual(new_calibration.scaling, self.calibration.scaling)
        np.testing.assert_array_almost_equal(new_calibration.axis, self.calibration.axis)


if __name__ == '__main__':
    unittest.main()