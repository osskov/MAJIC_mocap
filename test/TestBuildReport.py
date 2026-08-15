"""The build instrumentation: BuildReport and the sites threaded through it.

Two properties matter more than any individual metric.

  COLLECTING MUST NOT CHANGE THE BUILD. The report is threaded through the readers and
  assembly, so a bug there could alter what gets written while looking like it only observes.
  The first test builds the same trial twice, once with a report and once without, and demands
  the plates come out identical.

  A MISSING REPORT IS NOT STALENESS. The sidecar is deliberately outside the cache key -- so
  that adding a metric does not invalidate 281 artifacts -- which means a trial built before
  the instrumentation existed is still perfectly valid and simply has no tier-1 data. If that
  ever started reading as 'stale', every such trial would silently rebuild.
"""
import unittest

import numpy as np

from src.toolchest.building.report import COLUMNS, BuildReport


class TestBuildReportSchema(unittest.TestCase):
    def test_an_empty_report_still_has_the_columns(self):
        """A build that collected nothing must still produce a readable table, or every
        consumer needs an empty-case branch."""
        frame = BuildReport().to_frame()
        self.assertEqual(list(frame.columns), list(COLUMNS))
        self.assertTrue(frame.empty)

    def test_a_scalar_lands_in_the_numeric_column(self):
        report = BuildReport()
        report.add('S2_reconstruction', 'segment', 'thigh_r', residual_median_mm=0.44)
        row = report.to_frame().iloc[0]
        self.assertEqual(row.metric, 'residual_median_mm')
        self.assertAlmostEqual(row.value_num, 0.44)
        self.assertIsNone(row.value_str)

    def test_a_string_lands_in_the_string_column(self):
        """Kept apart from value_num so the numeric column stays a float column rather than
        an object column every consumer has to coerce."""
        report = BuildReport()
        report.add('S0_discovery', 'file', 'a.txt', rate_source='header')
        row = report.to_frame().iloc[0]
        self.assertIsNone(row.value_num)
        self.assertEqual(row.value_str, 'header')

    def test_a_bool_is_not_silently_a_count(self):
        """bool is an int in Python, so an unguarded numeric branch turns a flag into 1.0 and
        loses the distinction from a count of one."""
        report = BuildReport()
        report.add('S4_sync', 'trial', 'trial', sync_failed=True)
        row = report.to_frame().iloc[0]
        self.assertEqual(row.value_num, 1.0)
        self.assertIsNone(row.value_str)

    def test_a_short_sequence_is_expanded_by_index(self):
        report = BuildReport()
        report.add('S2_reconstruction', 'segment', 'x', fault_counts=[3, 0, 1, 7])
        metrics = dict(zip(report.to_frame().metric, report.to_frame().value_num))
        self.assertEqual(metrics['fault_counts_0'], 3.0)
        self.assertEqual(metrics['fault_counts_3'], 7.0)

    def test_a_long_sequence_is_summarised_rather_than_expanded(self):
        """Unresolved glitch indices can run to thousands. One row each would swamp the table
        with data that belongs in the per-frame tier."""
        report = BuildReport()
        report.add('S2_reconstruction', 'segment', 'x', unresolved_frames=list(range(5000)))
        frame = report.to_frame()
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].metric, 'unresolved_frames_count')
        self.assertEqual(frame.iloc[0].value_num, 5000.0)

    def test_none_is_dropped_rather_than_stored(self):
        report = BuildReport()
        report.add('S1_parse', 'file', 'a.txt', header_rate_hz=None, n_samples=10)
        self.assertEqual(list(report.to_frame().metric), ['n_samples'])

    def test_a_dict_is_flattened_with_prefixed_keys(self):
        report = BuildReport()
        report.add('S2_reconstruction', 'segment', 'x', counts={'flipped': 2, 'interp': 5})
        metrics = dict(zip(report.to_frame().metric, report.to_frame().value_num))
        self.assertEqual(metrics['counts_flipped'], 2.0)
        self.assertEqual(metrics['counts_interp'], 5.0)

    def test_extend_absorbs_a_sub_report_and_tolerates_none(self):
        """A reader building one sub-report per mocap take needs both."""
        parent, child = BuildReport(), BuildReport()
        child.add('S2_reconstruction', 'segment', 'x', residual_median_mm=1.0)
        parent.extend(child)
        parent.extend(None)
        self.assertEqual(len(parent.to_frame()), 1)

    def test_numpy_scalars_survive(self):
        """Every metric here comes out of numpy, so np.float64 must not fall through to str."""
        report = BuildReport()
        report.add('S7_alignment', 'plate', 'x', angle=np.float64(12.5),
                   count=np.int64(7))
        metrics = dict(zip(report.to_frame().metric, report.to_frame().value_num))
        self.assertAlmostEqual(metrics['angle'], 12.5)
        self.assertEqual(metrics['count'], 7.0)


class TestInstrumentedSitesFire(unittest.TestCase):
    """Each threaded site emits its rows, driven with synthetic traces rather than real data."""

    @staticmethod
    def _pair(seconds=12.0, rate=100.0, lag=0.4):
        from scipy.spatial.transform import Rotation
        from src.toolchest.IMUTrace import IMUTrace
        from src.toolchest.WorldTrace import WorldTrace

        timestamps = np.arange(int(seconds * rate)) / rate
        rotations = Rotation.from_euler('zyx', np.column_stack([
            0.8 * np.sin(2 * np.pi * 0.9 * timestamps),
            0.5 * np.sin(2 * np.pi * 1.4 * timestamps),
            0.3 * np.sin(2 * np.pi * 2.1 * timestamps)])).as_matrix()
        world = WorldTrace(timestamps, np.zeros((len(timestamps), 3)), rotations)
        synthetic = world.calculate_imu_trace(skip_lin_acc=True)
        imu = IMUTrace(timestamps + lag, synthetic.gyro, synthetic.acc, synthetic.mag)
        return {'plate_a': imu, 'plate_b': imu}, {'plate_a': world, 'plate_b': world}

    def test_pairing_sync_timeline_and_alignment_all_report(self):
        from src.toolchest.building.assembly import assemble_plate_trials

        imu_traces, world_traces = self._pair()
        report = BuildReport()
        plates = assemble_plate_trials(imu_traces, world_traces, True, report=report)
        self.assertEqual(len(plates), 2)

        frame = report.to_frame()
        for step in ('S3_pairing', 'S4_sync', 'S6_timeline', 'S7_alignment'):
            self.assertIn(step, set(frame.step), f'{step} emitted nothing')

        # Per plate, not just per trial -- that distinction is the point of the sync rows.
        sync = frame[(frame.step == 'S4_sync') & (frame.entity_kind == 'plate')]
        self.assertEqual(set(sync.entity), {'plate_a', 'plate_b'})

    def test_unmatched_names_are_recorded_not_only_printed(self):
        from src.toolchest.building.assembly import assemble_plate_trials

        imu_traces, world_traces = self._pair()
        imu_traces['orphan'] = imu_traces['plate_a']
        report = BuildReport()
        assemble_plate_trials(imu_traces, world_traces, True, report=report)

        frame = report.to_frame()
        unmatched = frame[frame.metric.str.startswith('unmatched_imu')]
        self.assertIn('orphan', set(unmatched.value_str))

    def test_collecting_does_not_change_the_result(self):
        """The instrumentation observes; it must not participate."""
        from src.toolchest.building.assembly import assemble_plate_trials

        without = assemble_plate_trials(*self._pair(), True, report=None)
        with_report = assemble_plate_trials(*self._pair(), True, report=BuildReport())

        self.assertEqual(set(without), set(with_report))
        for name, plate in without.items():
            other = with_report[name]
            np.testing.assert_array_equal(plate.imu_trace.timestamps,
                                          other.imu_trace.timestamps)
            np.testing.assert_array_equal(plate.world_trace.rotations,
                                          other.world_trace.rotations)
            np.testing.assert_array_equal(np.asarray(plate.valid), np.asarray(other.valid))


if __name__ == '__main__':
    unittest.main()


class TestPacketCounterTiming(unittest.TestCase):
    """Time comes from the packet counter, not the row index.

    This is the fix for the worst defect the build had: a dropped radio packet used to
    COMPRESS that sensor's timeline by one sample from then on, so plates of the same trial
    drifted apart mid-recording. Measured at 194 packets -- 4.85 s -- between plates of
    s24/t1_walking_001, invisible to both `_assert_one_clock` (the synthesised timestamps
    agree perfectly) and `_shared_lag` (its median absorbs a constant offset, not one that
    appears mid-trial).
    """

    @staticmethod
    def _elapsed(counter, freq=40.0):
        import pandas as pd
        from src.toolchest.building.xsens import _elapsed_from_counter
        return _elapsed_from_counter(pd.Series(counter), freq)

    def test_a_clean_counter_gives_uniform_time_and_no_gaps(self):
        elapsed, gaps, missing, largest = self._elapsed(list(range(100)))
        self.assertEqual((gaps, missing, largest), (0, 0, 0))
        np.testing.assert_allclose(np.diff(elapsed), 1 / 40.0)

    def test_a_dropped_packet_leaves_a_hole_rather_than_shifting_time(self):
        """The whole point: sample after the gap must land at its TRUE instant."""
        elapsed, gaps, missing, largest = self._elapsed([0, 1, 2, 5, 6])
        self.assertEqual((gaps, missing, largest), (1, 2, 2))
        self.assertAlmostEqual(elapsed[-1], 6 / 40.0)
        # Naive arange would have put the last sample at 4/40 -- 50 ms early.
        self.assertNotAlmostEqual(elapsed[-1], 4 / 40.0)

    def test_the_sixteen_bit_counter_wrap_is_not_a_gap(self):
        """A negative step is the counter wrapping, not time running backwards. Reading it as
        a gap would invent 65 000 missing samples."""
        elapsed, gaps, missing, largest = self._elapsed([65534, 65535, 0, 1])
        self.assertEqual((gaps, missing), (0, 0))
        np.testing.assert_allclose(np.diff(elapsed), 1 / 40.0)

    def test_several_gaps_accumulate(self):
        _, gaps, missing, largest = self._elapsed([0, 2, 4, 5])
        self.assertEqual((gaps, missing, largest), (2, 2, 1))


class TestOutOfRangeDetection(unittest.TestCase):
    def test_the_datasheet_ranges_are_the_ones_the_hardware_has(self):
        """36 samples in this dataset exceed them, peaking at 1.15e5 m/s^2 between neighbours
        reading 6.2 and 12.9 -- a corrupt export, not a measurement. A spike that size rings
        through a Butterworth for hundreds of samples and dominates any correlation taken on
        gyro magnitude."""
        from src.toolchest.building.xsens import ACC_RANGE_MS2, GYRO_RANGE_DEG_S
        self.assertAlmostEqual(ACC_RANGE_MS2, 160.0)
        self.assertAlmostEqual(GYRO_RANGE_DEG_S, 2000.0)
        # Comfortably above gravity and a fast limb, so a real sample never trips it.
        self.assertGreater(ACC_RANGE_MS2, 10 * 9.81)
        self.assertGreater(GYRO_RANGE_DEG_S, 1500.0)
