"""
Guards the gravity convention in experiment_utils.EXPECTED_GRAVITY against the real
dataset.

This exists because the constant was wrong ([0, 0, 9.8], a Z-up axis on a Y-up
dataset) through a full set of runs without anything failing. An axis or sign error
in this constant is silent: the filters still run, still converge, and still emit
plausible-looking joint angles. Only a check against the data catches it.

Skips cleanly if data/ has not been populated, so it is safe in a bare checkout.
"""
import os
import unittest

from test.fixtures import require_cache, require_data

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np

import paths
from experiments import experiment_utils
from experiments.experiment_utils import (EXPECTED_GRAVITY, ACTIVITIES, SUBJECTS,
                                          StaleTrialCache, check_gravity_convention,
                                          load_raw_data, measure_world_frame_gravity)
from src.toolchest.building import alborno

# The measured spread across this dataset is 0.035 m/s^2, so this is a sharp bound.
TOLERANCE = 0.25


def _first_available_trial():
    for subject in SUBJECTS:
        for activity in ACTIVITIES:
            if paths.raw_trial_dir(subject, activity).is_dir():
                return subject, activity
    return None, None


class TestGravityConvention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subject, cls.activity = _first_available_trial()
        if cls.subject is None:
            require_data(False, f"no source data under {paths.DATA_DIR}")
        try:
            cls.plates = load_raw_data(cls.subject, cls.activity)
        except StaleTrialCache as stale:
            # A stale cache means the code moved, not that this cannot be verified.
            require_cache(False, str(stale))

    def test_expected_gravity_matches_the_data(self):
        """The constant must agree with the accelerometers, in axis, sign and magnitude."""
        measured = measure_world_frame_gravity(self.plates)
        np.testing.assert_allclose(
            measured, EXPECTED_GRAVITY, atol=TOLERANCE,
            err_msg=(f"Subject{self.subject}/{self.activity}: mean world-frame accelerometer "
                     f"is {np.round(measured, 3).tolist()} but EXPECTED_GRAVITY is "
                     f"{EXPECTED_GRAVITY.tolist()}")
        )

    def test_magnitude_is_physical(self):
        self.assertAlmostEqual(float(np.linalg.norm(EXPECTED_GRAVITY)), 9.81, places=2)

    def test_check_passes_on_the_real_convention(self):
        measured = check_gravity_convention(self.plates)
        self.assertLess(float(np.linalg.norm(measured - EXPECTED_GRAVITY)), TOLERANCE)

    def test_check_rejects_a_swapped_axis(self):
        """A Z-up constant on this Y-up dataset must raise, not pass quietly."""
        self._with_patched_gravity(np.array([0.0, 0.0, 9.81]))
        with self.assertRaises(ValueError):
            check_gravity_convention(self.plates)

    def test_check_rejects_a_flipped_sign(self):
        self._with_patched_gravity(np.array([0.0, -9.81, 0.0]))
        with self.assertRaises(ValueError):
            check_gravity_convention(self.plates)

    def _with_patched_gravity(self, gravity: np.ndarray):
        """Swaps in a wrong constant for the duration of one test.

        check_gravity_convention reads the module global, so it has to be patched on the
        module rather than on the name imported here.
        """
        original = experiment_utils.EXPECTED_GRAVITY
        experiment_utils.EXPECTED_GRAVITY = gravity
        self.addCleanup(setattr, experiment_utils, 'EXPECTED_GRAVITY', original)


if __name__ == '__main__':
    unittest.main()
