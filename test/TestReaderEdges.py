"""The reader entry points and constants that nothing else exercises.

Each of these fails QUIETLY rather than loudly, which is what makes them worth pinning:

  * an unknown IMU device id used to be dropped with no warning and no count, so a session
    recorded with a replacement sensor built successfully with 14 plates instead of 15;
  * `git_provenance` swallows every error and returns nulls, so if it started returning
    nothing every manifest would be provenance-free and the suite would stay green -- in a
    repo where the manifest is the stated mechanism for knowing which code made a figure;
  * the angular-speed threshold gates the entire repair path and was pinned only indirectly.
"""
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

import paths
from src.toolchest.building import alborno, imove_mocap
from src.toolchest.building.assembly import assemble_plate_trials
from src.toolchest.building.reconstruction import (
    MAX_RECONSTRUCTION_ANGULAR_SPEED_DEG_S, repair_reconstruction_glitches)
from src.toolchest.building.sources import IMOVE_ROOT
from test.fixtures import require_data

XSENS_FILE = """// Update Rate: 40.0Hz
PacketCounter\tSampleTimeFine\tAcc_X\tAcc_Y\tAcc_Z\tGyr_X\tGyr_Y\tGyr_Z\tMag_X\tMag_Y\tMag_Z
0\t\t9.81\t0.0\t0.0\t0.0\t0.0\t0.0\t-0.6\t-0.2\t0.5
1\t\t9.81\t0.0\t0.0\t0.0\t0.0\t0.0\t-0.6\t-0.2\t0.5
"""


class TestUnknownImoveDevices(unittest.TestCase):
    def test_an_unrecognised_device_warns_rather_than_vanishing(self):
        with tempfile.TemporaryDirectory() as folder:
            session = Path(folder) / 's99'
            (session / 'imu_data').mkdir(parents=True)
            known = next(iter(imove_mocap.DEVICE_TO_SENSOR))
            for device in (known, 'DEADBEEF'):
                (session / 'imu_data' / f't1_walking_001-000_{device}.txt').write_text(
                    XSENS_FILE)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                traces = imove_mocap.load_imu_traces(session, 't1_walking_001')

        self.assertEqual(len(traces), 1)
        unknown = [w for w in caught
                   if issubclass(w.category, imove_mocap.UnknownDeviceWarning)]
        self.assertEqual(len(unknown), 1)
        self.assertIn('DEADBEEF', str(unknown[0].message))

    def test_a_fully_known_session_is_silent(self):
        require_data((IMOVE_ROOT / 's2').is_dir(), "no IMoVE data")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            traces = imove_mocap.load_imu_traces(IMOVE_ROOT / 's2', 't6_drop_jump_001')

        self.assertEqual(len(traces), 15)
        self.assertEqual([w for w in caught
                          if issubclass(w.category,
                                        imove_mocap.UnknownDeviceWarning)], [])


class TestAlbornoImuFolder(unittest.TestCase):
    """The 'imu data/'-or-folder fallback, the last unexercised reader entry point."""

    def test_the_imu_data_subdirectory_is_preferred(self):
        with tempfile.TemporaryDirectory() as folder:
            trial = Path(folder)
            (trial / 'imu data').mkdir()
            (trial / 'imu data' / 'femur_r_imu.txt').write_text(XSENS_FILE)
            # A decoy at the top level must be ignored while the subdirectory exists.
            (trial / 'decoy.txt').write_text(XSENS_FILE)

            traces = alborno.load_imu_traces(trial)

        self.assertEqual(set(traces), {'femur_r_imu'})

    def test_the_folder_itself_is_used_when_there_is_no_subdirectory(self):
        with tempfile.TemporaryDirectory() as folder:
            trial = Path(folder)
            (trial / 'tibia_l_imu.txt').write_text(XSENS_FILE)

            self.assertEqual(set(alborno.load_imu_traces(trial)), {'tibia_l_imu'})

    def test_an_empty_folder_raises_rather_than_returning_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                alborno.load_imu_traces(Path(folder))


class TestGitProvenance(unittest.TestCase):
    """Every manifest carries this, and _git swallows all errors into nulls."""

    def test_it_reports_a_real_sha_in_this_repo(self):
        provenance = paths.git_provenance()
        self.assertIsNotNone(provenance.get('git_sha'),
                             "manifests would be provenance-free and nothing would notice")
        self.assertRegex(provenance['git_sha'], r'^[0-9a-f]{7,40}$')
        self.assertIsNotNone(provenance.get('git_branch'))

    def test_it_reports_dirtiness_as_a_bool(self):
        self.assertIn(paths.git_provenance().get('git_dirty'), (True, False))

    def test_it_degrades_to_nulls_rather_than_raising(self):
        """A non-repo checkout must not fail the run -- but the values must then be null
        rather than stale, so the manifest says 'unknown' instead of lying."""
        paths.git_provenance.cache_clear()
        try:
            with mock.patch('subprocess.run', side_effect=OSError('no git')):
                provenance = paths.git_provenance()
            self.assertIsNone(provenance.get('git_sha'))
            self.assertIsNone(provenance.get('git_dirty'))
        finally:
            paths.git_provenance.cache_clear()


class TestAngularSpeedThreshold(unittest.TestCase):
    """The constant that gates the whole repair path, pinned directly.

    Everything built on it is thoroughly tested; the number itself was only pinned by the
    legacy path's 600 deg/s "clean data untouched" case, which is five times under it.
    """

    @staticmethod
    def _trace(step_deg, rate=100.0, n=200):
        timestamps = np.arange(n) / rate
        rotations = Rotation.from_rotvec(
            np.linspace(0.0, 0.4, n)[:, None] * np.array([0.0, 0.0, 1.0])).as_matrix()
        rotations[n // 2:] = rotations[n // 2:] @ Rotation.from_euler(
            'y', step_deg, degrees=True).as_matrix()
        return np.tile(np.array([0.0, 1.0, 0.0]), (n, 1)), rotations, timestamps

    def _speed_to_step(self, speed_deg_s, rate=100.0):
        """A one-sample jump of this size registers as `speed_deg_s`."""
        return speed_deg_s / rate

    def test_just_under_the_threshold_is_left_alone(self):
        positions, rotations, timestamps = self._trace(
            self._speed_to_step(0.95 * MAX_RECONSTRUCTION_ANGULAR_SPEED_DEG_S))
        _, _, _, report = repair_reconstruction_glitches(positions, rotations, timestamps)
        self.assertEqual(report['unresolved'], 0)

    def test_just_over_the_threshold_is_caught(self):
        positions, rotations, timestamps = self._trace(
            self._speed_to_step(1.10 * MAX_RECONSTRUCTION_ANGULAR_SPEED_DEG_S))
        _, _, _, report = repair_reconstruction_glitches(positions, rotations, timestamps)
        self.assertGreaterEqual(report['unresolved'] + report['flipped'], 1)


class TestAssembleWithNoImus(unittest.TestCase):
    def test_it_returns_an_empty_mapping_not_a_list(self):
        """Just fixed from [], which was an AttributeError for any caller doing .items()."""
        result = assemble_plate_trials({}, {}, align_plate_trials=False)
        self.assertEqual(result, {})
        self.assertEqual(list(result.items()), [])

    def test_no_overlap_between_imus_and_world_traces_is_also_empty(self):
        timestamps = np.arange(10) / 100.0
        from src.toolchest.IMUTrace import IMUTrace
        from src.toolchest.WorldTrace import WorldTrace
        imu = {'a': IMUTrace(timestamps, np.zeros((10, 3)), np.zeros((10, 3)),
                             np.zeros((10, 3)))}
        world = {'b': WorldTrace(timestamps, np.zeros((10, 3)),
                                 np.tile(np.eye(3), (10, 1, 1)))}

        self.assertEqual(assemble_plate_trials(imu, world, align_plate_trials=False), {})


if __name__ == '__main__':
    unittest.main()
