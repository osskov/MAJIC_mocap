"""The IMoVE reader: src/toolchest/building/imove_mocap.py.

Four things about this dataset are not guessable, cost real time to discover, and would each
fail SILENTLY rather than loudly if they regressed. That is what these protect:

  * marker labels match on the suffix after ':', never the group -- s4/s5/s6 flatten every
    marker into one 'modified_rizzoli' group, so matching on the group finds either nothing
    or everything;
  * the pelvis cluster's correspondence is a cyclic shift, and getting it wrong silently
    rotates the pelvis frame instead of raising;
  * `Total Exported Frames` in the long-walk headers overstates the real row count by 30000;
  * one inertial record can span several mocap takes, and only the first shares its filename.

Split into a fast half that touches no trial data and a slower half that reads real sessions.
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import paths
from src.toolchest.WorldTrace import WorldTrace
from src.toolchest.building import imove_mocap
from src.toolchest.building.sources import IMOVE_ROOT, get_source


def _rotating(timestamps, rate_hz=0.3):
    return Rotation.from_rotvec(
        (2 * np.pi * rate_hz * timestamps)[:, None] * np.array([0.1, 0.2, 0.97])).as_matrix()


class TestTakeGrouping(unittest.TestCase):
    """Which mocap files belong to one inertial record. Filesystem only, no parsing."""

    def test_an_ordinary_trial_claims_only_its_own_take(self):
        takes = imove_mocap.mocap_takes_for(IMOVE_ROOT / 's2', 't6_drop_jump_001')
        self.assertEqual([p.name for p in takes], ['t6_drop_jump_001.csv'])

    def test_the_long_walk_claims_all_three_takes(self):
        """One 4538 s Xsens file covers three ~950 s Motive takes sitting at roughly 2 s,
        1756 s and 3564 s into it. Only _001 shares the IMU's filename."""
        takes = imove_mocap.mocap_takes_for(IMOVE_ROOT / 's13l', 't12_longwalk_001')
        self.assertEqual([p.name for p in takes],
                         ['t12_longwalk_001.csv', 't12_longwalk_002.csv',
                          't12_longwalk_003.csv'])

    def test_a_take_with_its_own_imu_is_not_absorbed(self):
        """Only takes that NOTHING else claims get folded in. If _002 had its own inertial
        file it would be a separate trial, and swallowing it would double-count the data."""
        session = IMOVE_ROOT / 's13l'
        claimed = {p.name.split('-000_')[0] for p in (session / 'imu_data').glob('*.txt')}
        for take in imove_mocap.mocap_takes_for(session, 't12_longwalk_001')[1:]:
            self.assertNotIn(take.stem, claimed)

    def test_a_record_with_no_mocap_claims_nothing(self):
        """The long-walk sessions carry treadmill and cmjdl IMU files that were never
        mocapped; enumeration has to drop them rather than build a trial with no truth."""
        self.assertEqual(imove_mocap.mocap_takes_for(IMOVE_ROOT / 's13l',
                                                     't7_cmjdl_001'), [])


class TestSegmentMapping(unittest.TestCase):
    def test_every_device_maps_to_a_known_segment(self):
        segments = {name.rsplit('_', 1)[0] for name in imove_mocap.DEVICE_TO_SENSOR.values()}
        self.assertEqual(segments, {'THIGH_L', 'THIGH_R', 'SHANK_L', 'SHANK_R',
                                    'PELVIS', 'FOOT_L', 'FOOT_R'})

    def test_limbs_carry_three_placements_and_the_rest_one(self):
        by_segment = {}
        for sensor in imove_mocap.DEVICE_TO_SENSOR.values():
            segment, placement = sensor.rsplit('_', 1)
            by_segment.setdefault(segment, set()).add(placement)
        for segment in ('THIGH_L', 'THIGH_R', 'SHANK_L', 'SHANK_R'):
            self.assertEqual(by_segment[segment], {'H', 'M', 'L'})
        for segment in ('PELVIS', 'FOOT_L', 'FOOT_R'):
            self.assertEqual(by_segment[segment], {'M'})

    def test_the_pelvis_cluster_order_is_the_cyclic_shift(self):
        """(RPS1, RPS2, LPS2, LPS1) matches the thigh cluster at 1.03 mm; starting from LPS1
        does not appear in the top three. A wrong order does not raise -- it silently rotates
        the pelvis frame -- so it has to be pinned here."""
        self.assertEqual(imove_mocap.CLUSTER_MARKERS['PELVIS'],
                         ['RPS1', 'RPS2', 'LPS2', 'LPS1'])

    def test_one_pose_is_shared_by_all_of_a_segment_s_sensors(self):
        timestamps = np.arange(100) / 100.0
        trace = WorldTrace(timestamps, np.zeros((100, 3)), _rotating(timestamps))
        imu = {name: None for name in
               ('THIGH_L_H', 'THIGH_L_M', 'THIGH_L_L', 'PELVIS_M')}

        paired = imove_mocap._pair_world_traces({'THIGH_L': trace, 'PELVIS': trace}, imu)

        self.assertEqual(set(paired), set(imu))
        for name in ('THIGH_L_H', 'THIGH_L_M', 'THIGH_L_L'):
            self.assertIs(paired[name], trace, "the pose should be shared, not copied")

    def test_a_segment_with_no_sensor_contributes_no_plate(self):
        timestamps = np.arange(100) / 100.0
        trace = WorldTrace(timestamps, np.zeros((100, 3)), _rotating(timestamps))
        paired = imove_mocap._pair_world_traces({'THIGH_L': trace}, {'PELVIS_M': None})
        self.assertEqual(paired, {})


class TestMergeTakes(unittest.TestCase):
    """Several mocap takes onto one inertial timeline. Synthetic, so it runs fast."""

    def setUp(self):
        self.rate = 100.0
        self.timestamps = np.arange(3000) / self.rate           # 30 s of "IMU"
        self.lags = [1.0, 12.0, 22.0]
        self.takes = []
        for _ in self.lags:
            t = np.arange(500) / self.rate                      # 5 s takes
            self.takes.append({'THIGH_L': WorldTrace(
                t, np.tile([1.0, 2.0, 3.0], (len(t), 1)), _rotating(t))})

    def test_each_take_becomes_its_own_valid_window(self):
        merged = imove_mocap._merge_takes(self.takes, self.lags, self.timestamps, self.rate)
        valid = merged['THIGH_L'].valid

        edges = np.flatnonzero(np.diff(valid.astype(int)))
        windows = len(edges) // 2 + (1 if valid[0] or valid[-1] else 0)
        self.assertEqual(windows, 3)

        for lag in self.lags:
            middle = int((lag + 2.5) * self.rate)
            self.assertTrue(valid[middle], f"take at {lag}s should be valid at its centre")
        self.assertFalse(valid[int(8.0 * self.rate)], "the gap between takes is not truth")

    def test_the_merged_trace_spans_the_whole_inertial_record(self):
        merged = imove_mocap._merge_takes(self.takes, self.lags, self.timestamps, self.rate)
        np.testing.assert_allclose(merged['THIGH_L'].timestamps, self.timestamps)

    def test_a_later_take_cannot_overwrite_an_earlier_one(self):
        """Each take is extrapolated across the whole timeline before being masked, so
        without the ~valid guard take three would stamp its held pose over takes one and two.
        """
        distinct = []
        for index, _ in enumerate(self.lags):
            t = np.arange(500) / self.rate
            distinct.append({'THIGH_L': WorldTrace(
                t, np.tile([float(index), 0.0, 0.0], (len(t), 1)), _rotating(t))})

        merged = imove_mocap._merge_takes(distinct, self.lags, self.timestamps, self.rate)

        for index, lag in enumerate(self.lags):
            middle = int((lag + 2.5) * self.rate)
            self.assertAlmostEqual(merged['THIGH_L'].positions[middle, 0], float(index),
                                   places=3)


class TestEnumeration(unittest.TestCase):
    """What gets built at all. Neither source's enumerate_trials was tested."""

    def test_imove_enumerates_inertial_records_not_mocap_takes(self):
        """The long walks hold three Motive takes against one Xsens file. Enumerating mocap
        would claim three trials where there is one recording, and build the same inertial
        data three times."""
        trials = get_source('imove').enumerate_trials()
        long_walk = [t for s, t in trials if s == 's13l']

        self.assertIn('t12_longwalk_001', long_walk)
        self.assertNotIn('t12_longwalk_002', long_walk)
        self.assertNotIn('t12_longwalk_003', long_walk)

    def test_imove_skips_records_with_no_mocap(self):
        """The long-walk sessions also carry treadmill and cmjdl IMU files that were never
        mocapped. A trial with no ground truth is not a trial."""
        trials = set(get_source('imove').enumerate_trials())
        self.assertNotIn(('s13l', 't7_cmjdl_001'), trials)
        self.assertIn(('s13', 't7_cmjdl_001'), trials)

    def test_alborno_enumerates_only_folders_holding_a_trc(self):
        trials = get_source('alborno').enumerate_trials()
        self.assertTrue(trials)
        for subject, trial in trials:
            self.assertTrue((paths.DATA_DIR / f'Subject{subject}' / trial).is_dir())
            self.assertTrue(any((paths.DATA_DIR / f'Subject{subject}' / trial).glob('*.trc')))

    def test_alborno_reflects_that_some_subjects_have_walking_only(self):
        """Subjects 05, 08 and 10 have no complexTasks. Enumerating rather than crossing two
        constants is what removed the phantom 'absent' entries."""
        trials = dict()
        for subject, trial in get_source('alborno').enumerate_trials():
            trials.setdefault(subject, set()).add(trial)
        self.assertEqual(trials['05'], {'walking'})
        self.assertIn('complexTasks', trials['01'])

    def test_an_unknown_dataset_names_what_is_available(self):
        from src.toolchest.building.sources import SOURCES
        with self.assertRaises(ValueError) as caught:
            get_source('nope')
        for name in SOURCES:
            self.assertIn(name, str(caught.exception))


class TestFootMarkers(unittest.TestCase):
    """Four naming schemes across sessions, reconciled by suffix token.

    A foot that resolves to fewer than three markers is dropped SILENTLY, which removes both
    ankle joints from that trial with no error anywhere -- so the matching has to be right.
    """

    def test_each_session_style_resolves_at_least_three_markers(self):
        for session in ('s2', 's4', 's13', 's13l'):
            csv = next((IMOVE_ROOT / session / 'mocap_data').glob('t*.csv'))
            for side in ('L', 'R'):
                labels = imove_mocap.foot_marker_labels(csv, side)
                self.assertGreaterEqual(len(labels), 3,
                                        f"{session} foot {side} resolved {labels}")
                self.assertTrue(all(l.startswith(side) for l in labels), labels)

    def test_the_two_feet_do_not_share_markers(self):
        csv = next((IMOVE_ROOT / 's2' / 'mocap_data').glob('t*.csv'))
        left = set(imove_mocap.foot_marker_labels(csv, 'L'))
        right = set(imove_mocap.foot_marker_labels(csv, 'R'))
        self.assertTrue(left and right)
        self.assertFalse(left & right, f"shared between feet: {left & right}")

    def test_foot_labels_do_not_capture_other_segments(self):
        """'MT' and 'CAL' are matched inside the label, so a thigh or shank marker with an
        unlucky name would be swept into the foot and corrupt its pose."""
        csv = next((IMOVE_ROOT / 's2' / 'mocap_data').glob('t*.csv'))
        cluster = {m for markers in imove_mocap.CLUSTER_MARKERS.values() for m in markers}
        for side in ('L', 'R'):
            self.assertFalse(set(imove_mocap.foot_marker_labels(csv, side)) & cluster)

    def test_segment_labels_includes_feet_only_when_resolvable(self):
        csv = next((IMOVE_ROOT / 's2' / 'mocap_data').glob('t*.csv'))
        labels = imove_mocap.segment_labels(csv)
        self.assertIn('FOOT_L', labels)
        self.assertEqual(set(labels) - {'FOOT_L', 'FOOT_R'},
                         set(imove_mocap.CLUSTER_MARKERS))


class TestTakeLag(unittest.TestCase):
    """_take_lag derives what TestMergeTakes is handed, so it never ran under test."""

    @staticmethod
    def _world(offset_s, rate=100.0, seconds=20.0):
        t = np.arange(int(seconds * rate)) / rate
        return WorldTrace(t, np.zeros((len(t), 3)), _rotating(t - offset_s))

    def _imu_for(self, world):
        synthetic = world.calculate_imu_trace(skip_lin_acc=True)
        from src.toolchest.IMUTrace import IMUTrace
        return IMUTrace(world.timestamps, synthetic.gyro, synthetic.acc, synthetic.mag)

    def test_it_medians_over_the_segments_that_have_an_m_sensor(self):
        world = {'THIGH_L': self._world(0.0), 'SHANK_L': self._world(0.0)}
        imu = {f'{segment}_M': self._imu_for(trace) for segment, trace in world.items()}

        self.assertAlmostEqual(imove_mocap._take_lag(world, imu), 0.0, places=2)

    def test_it_ignores_segments_with_no_m_sensor(self):
        """H and L exist on the limbs but the lag is derived from M, so a segment carrying
        only H must not contribute and must not crash."""
        world = {'THIGH_L': self._world(0.0), 'SHANK_L': self._world(0.0)}
        imu = {'THIGH_L_M': self._imu_for(world['THIGH_L']),
               'SHANK_L_H': self._imu_for(world['SHANK_L'])}

        self.assertAlmostEqual(imove_mocap._take_lag(world, imu), 0.0, places=2)

    def test_it_raises_when_nothing_can_be_synced_on(self):
        world = {'FOOT_L': self._world(0.0)}
        with self.assertRaises(ValueError) as caught:
            imove_mocap._take_lag(world, {'THIGH_L_M': self._imu_for(world['FOOT_L'])})
        self.assertIn('M sensor', str(caught.exception))


class TestAgainstRealSessions(unittest.TestCase):
    """Reads actual IMoVE files. Slower, and the only place the parsing is exercised."""

    def test_the_header_reports_metres_and_a_rate(self):
        header = imove_mocap.read_motive_header(
            IMOVE_ROOT / 's2' / 'mocap_data' / 't6_drop_jump_001.csv')
        self.assertEqual(header['Length Units'], 'Meters')
        self.assertAlmostEqual(float(header['Capture Frame Rate']), 100.0)

    def test_the_long_walk_header_frame_count_is_wrong(self):
        """`Total Exported Frames` overstates by exactly 30000 in every long-walk export, so
        the reader counts rows instead. Pinned because trusting the field would silently
        mislabel the take's duration by five minutes."""
        path = IMOVE_ROOT / 's13l' / 'mocap_data' / 't12_longwalk_001.csv'
        claimed = int(imove_mocap.read_motive_header(path)['Total Exported Frames'])

        _, timestamps = imove_mocap.read_motive_csv(path, ['RPS1'])

        self.assertEqual(claimed - len(timestamps), 30000)
        # What the reader uses must match the timestamps actually in the file.
        self.assertAlmostEqual(len(timestamps) / timestamps[-1], 100.0, places=1)

    def test_markers_are_found_in_a_flat_group_session(self):
        """s4 puts every marker in one 'modified_rizzoli' group, so a reader matching on the
        group name rather than the label suffix finds nothing here."""
        path = IMOVE_ROOT / 's4' / 'mocap_data' / 't6_drop_jump_001.csv'
        markers, _ = imove_mocap.read_motive_csv(path,
                                                 imove_mocap.CLUSTER_MARKERS['THIGH_L'])
        self.assertEqual(markers.shape[1], 4)
        self.assertTrue(np.isfinite(markers).any())

    def test_an_absent_label_returns_none_rather_than_raising(self):
        path = IMOVE_ROOT / 's2' / 'mocap_data' / 't6_drop_jump_001.csv'
        self.assertIsNone(imove_mocap.read_motive_csv(path, ['NOT_A_MARKER']))

    def test_an_untracked_segment_is_skipped_not_fatal(self):
        """s10's treadmill takes drop all four LTH and LSH markers for the whole recording.
        That is 'untracked here', and the other segments must still load."""
        traces = imove_mocap.load_world_traces(
            IMOVE_ROOT / 's10' / 'mocap_data' / 't2_treadmill_walking_001.csv')

        self.assertNotIn('THIGH_L', traces)
        self.assertIn('THIGH_R', traces)
        self.assertIn('PELVIS', traces)

    def test_a_trial_loads_one_plate_per_sensor(self):
        plates = get_source('imove').load('s2', 't6_drop_jump_001')

        self.assertEqual(len(plates), 15)
        self.assertEqual(set(plates), set(imove_mocap.DEVICE_TO_SENSOR.values()))
        # 40 Hz IMU against 100 Hz mocap, so the trial runs at the slower of the two.
        for plate in plates.values():
            self.assertAlmostEqual(plate.sample_rate, 40.0, places=3)

    def test_the_three_sensors_on_a_segment_get_independent_alignments(self):
        """They share a pose but not a mounting, so aligning must not collapse them onto one
        orientation -- if it did, H and L would inherit M's sensor frame."""
        plates = get_source('imove').load('s2', 't6_drop_jump_001')
        rotations = [plates[f'THIGH_L_{p}'].world_trace.rotations[0] for p in ('H', 'M', 'L')]

        for first, second in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(np.allclose(rotations[first], rotations[second], atol=1e-6))


if __name__ == '__main__':
    unittest.main()
