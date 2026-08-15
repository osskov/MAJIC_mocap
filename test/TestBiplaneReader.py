"""IMoVE's biplane half: unit conversions, clock arithmetic, and the chained sync.

Most of what can go wrong here is silent. A transposed rotation still looks like a rotation; a
clock offset applied to the wrong stream still produces a plausible trial; a correlation that
fails still returns a number. So the tests below check the things that would otherwise be
discovered as a wrong answer rather than as an error -- and several of them exist because
exactly that happened while the reader was being written.
"""
import unittest
import warnings

import numpy as np

from src.toolchest.building import biplane as bp

require_data = unittest.skipUnless(bp.BIPLANE_ROOT.is_dir(),
                                   "biplane source data not present")


class TestConventions(unittest.TestCase):
    """The constants that encode a decision, so a later edit has to argue with a test."""

    def test_the_search_bracket_exceeds_the_known_trigger_error(self):
        """The trigger is not merely quantised to whole seconds, it is wrong by seconds: the
        measured lag on 12/LDDrop3 is -11.53 s. A bracket sized for the quantisation alone --
        which is what 6 s was -- cannot reach the answer, and the correlation then returns the
        best wrong peak inside it with no indication anything is amiss."""
        self.assertGreater(bp.TRIGGER_BRACKET_S, 11.53)

    def test_the_imu_margin_covers_the_bracket_and_the_capture(self):
        """Otherwise the correlation runs out of record before it runs out of bracket."""
        vicon_capture_s = 10.0
        self.assertGreater(bp.IMU_MARGIN_S, bp.TRIGGER_BRACKET_S + vicon_capture_s)

    def test_the_biplane_pretrigger_is_about_three_seconds(self):
        """Measured at 2.98 +/- 0.06 s over 17 of 18 trials across three subjects. Pinned
        because assuming the Time column starts at the trigger -- the obvious reading -- puts
        the window 3 s out."""
        self.assertAlmostEqual(bp.BIPLANE_PRETRIGGER_S, 2.98, places=2)

    def test_the_magnetometer_is_declared_absent(self):
        """A BioStamp has no magnetometer, and `mag` is therefore zeros. Silently-zero field
        data would sail through a filter and produce a heading."""
        self.assertFalse(bp.HAS_MAGNETOMETER)

    def test_every_cluster_has_four_markers_on_both_sides(self):
        for (side, site), labels in bp.VICON_CLUSTERS.items():
            self.assertEqual(len(labels), 4, f'{side}/{site}')
            self.assertTrue(all(l.startswith(side[0].upper()) for l in labels),
                            f'{side}/{site}: labels do not match the side')

    def test_trial_names_decode_to_a_side(self):
        self.assertEqual(bp.trial_side('LSDrop2'), 'left')
        self.assertEqual(bp.trial_side('RrunStance1'), 'right')
        self.assertEqual(bp.trial_side('Rstatic1'), 'right')
        self.assertIsNone(bp.trial_side('Sync'))
        self.assertIsNone(bp.trial_side('nonsense'))


class TestClockArithmetic(unittest.TestCase):
    def test_eastern_offset_follows_daylight_saving(self):
        """The study spans September to November and crosses the changeover, so a hardcoded
        offset is wrong for one half of the subjects."""
        import pandas as pd
        self.assertEqual(bp._utc_offset_hours(pd.Timestamp('2019-09-11 09:00')), 4)
        self.assertEqual(bp._utc_offset_hours(pd.Timestamp('2019-11-19 09:00')), 5)


@require_data
class TestAgainstTheFiles(unittest.TestCase):
    """Reads the real study. Skipped when the source tree is absent."""

    SUBJECT, SESSION, BLOCK, TRIAL = '12', 'Test1', 'B', 'LDDrop3'

    def test_the_transform_convention_is_row_vector(self):
        """The decisive check, and the one that catches a silent inverse: the shipped forward
        and inverse transforms compose to the identity AS STORED, which is only true in the
        row-vector convention. Reading the top-left block as the rotation would transpose it
        and leave every angle plausible and wrong."""
        import pandas as pd
        base = (bp.BIPLANE_ROOT / 'Kinematics' / bp.STUDY / self.SUBJECT / self.SESSION /
                self.BLOCK / self.TRIAL)
        columns = [f'[{i}][{j}]' for i in range(4) for j in range(4)]
        forward = pd.read_csv(base / 'HomoTransMatrices_Tibia-to-Lab.csv')[columns] \
            .to_numpy().reshape(-1, 4, 4)
        inverse = pd.read_csv(base / 'HomoTransMatrices_Lab-to-Tibia.csv')[columns] \
            .to_numpy().reshape(-1, 4, 4)
        np.testing.assert_allclose(forward[0] @ inverse[0], np.eye(4), atol=1e-3)

    def test_biplane_pose_is_a_rotation_in_metres(self):
        pose = bp.read_biplane_pose(
            bp.BIPLANE_ROOT / 'Kinematics' / bp.STUDY / self.SUBJECT / self.SESSION /
            self.BLOCK / self.TRIAL / 'HomoTransMatrices_Tibia-to-Lab.csv')
        self.assertIsNotNone(pose)
        times, positions, rotations = pose
        np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1),
                                   np.tile(np.eye(3), (len(rotations), 1, 1)), atol=1e-4)
        np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-4)
        # Metres, not the millimetres the file stores: a knee is not 140 m from the origin.
        self.assertLess(np.linalg.norm(positions, axis=1).max(), 5.0)

    def test_mc10_windowing_returns_si_units(self):
        trigger = self._trigger()
        imu = bp.read_mc10_imu(
            bp.BIPLANE_ROOT / 'IMUs' / bp.STUDY / self.SUBJECT / 'lateral_shank_left',
            window=(trigger - 5, trigger + 5))
        self.assertIsNotNone(imu)
        self.assertLess(len(imu), 5000, "the window should slice, not read two hours")
        # g -> m/s^2: a resting BioStamp reads about 9.8, not about 1.
        self.assertGreater(np.median(np.linalg.norm(imu.acc, axis=1)), 5.0)
        # deg/s -> rad/s: a limb does not spin at 200 rad/s.
        self.assertLess(np.abs(imu.gyro).max(), 50.0)
        np.testing.assert_array_equal(imu.mag, 0.0)

    def test_the_gyro_resampling_is_band_limited_not_linear(self):
        """The two exports are independent streams on their own timestamps, so the gyro must
        be moved onto the accelerometer's grid -- and doing that with np.interp is the mistake
        this repo already paid for: linear interpolation is sinc^2, and the band mismatch it
        caused biased the cluster-to-IMU offset by 7-9 mm on the shank.

        Checked on a tone rather than on the reader's output, because the difference is a
        frequency-response property and real data has no known answer to compare against.
        """
        from src.toolchest.resampling import resample_values

        rate = 250.0
        source = np.arange(4000) / rate
        target = source[:3900] + 0.4 / rate      # the fractional offset real exports have
        tone = np.sin(2 * np.pi * 8.0 * source)
        truth = np.sin(2 * np.pi * 8.0 * target)
        interior = slice(200, -200)

        banded = resample_values(np.column_stack([tone] * 3), source, target,
                                 source_rate=rate, target_rate=rate)[:, 0]
        linear = np.interp(target, source, tone)

        banded_error = np.sqrt(((banded[interior] - truth[interior]) ** 2).mean())
        linear_error = np.sqrt(((linear[interior] - truth[interior]) ** 2).mean())
        self.assertLess(banded_error, linear_error / 100,
                        f"band-limited {banded_error:.2e} vs linear {linear_error:.2e}")

    def test_the_gyro_is_interpolated_onto_the_accelerometer_grid(self):
        """accel.csv and gyro.csv are independent exports with their own timestamps."""
        trigger = self._trigger()
        imu = bp.read_mc10_imu(
            bp.BIPLANE_ROOT / 'IMUs' / bp.STUDY / self.SUBJECT / 'lateral_shank_left',
            window=(trigger - 5, trigger + 5))
        self.assertEqual(len(imu.gyro), len(imu.acc))
        self.assertEqual(len(imu.timestamps), len(imu.acc))

    def test_vicon_reads_a_cluster_in_metres(self):
        markers = bp.read_vicon_c3d(
            bp.vicon_path(self.SUBJECT, self.SESSION, self.TRIAL),
            bp.VICON_CLUSTERS[('left', 'lateral_shank')])
        self.assertIsNotNone(markers)
        positions, timestamps = markers
        self.assertEqual(positions.shape[1], 4)
        self.assertLess(np.nanmax(np.abs(positions)), 10.0, "should be metres, not mm")
        self.assertAlmostEqual(1.0 / np.median(np.diff(timestamps)), 150.0, places=0)

    def test_the_biplane_kinematics_carry_no_dropout_sentinel(self):
        """The upstream pipeline treats zeros in its biplane stream as invalid and keeps only
        the largest contiguous valid run. That handling does not apply to the export we read:
        swept across all 15 subjects, 826 files and 61128 rows, there are no all-zero rows, no
        non-rotation blocks and no non-numeric entries. Pinned on one trial so that if a future
        export does carry a sentinel, this stops being silently true."""
        import pandas as pd

        columns = [f'[{i}][{j}]' for i in range(4) for j in range(4)]
        matrices = pd.read_csv(
            bp.BIPLANE_ROOT / 'Kinematics' / bp.STUDY / self.SUBJECT / self.SESSION /
            self.BLOCK / self.TRIAL / 'HomoTransMatrices_Tibia-to-Lab.csv'
        )[columns].apply(pd.to_numeric, errors='coerce').to_numpy(float).reshape(-1, 4, 4)

        self.assertFalse(np.isnan(matrices).any(), "non-numeric entries present")
        self.assertEqual(int((np.abs(matrices).sum(axis=(1, 2)) == 0).sum()), 0,
                         "all-zero rows would be a dropout sentinel we are ingesting as pose")
        determinants = np.linalg.det(matrices[:, :3, :3])
        self.assertTrue((np.abs(determinants) > 0.5).all(), "non-rotation blocks present")

    def test_an_absent_marker_label_returns_none_rather_than_raising(self):
        self.assertIsNone(bp.read_vicon_c3d(
            bp.vicon_path(self.SUBJECT, self.SESSION, self.TRIAL), ['NOSUCHMARKER']))

    def test_the_camera_offset_is_per_subject(self):
        """Stable within a session but ranging over an hour between them, so one global
        constant would be wrong by up to that much."""
        offsets = {s: bp.camera_clock_offset(s) for s in ('01', '12', '17')}
        self.assertTrue(all(v is not None for v in offsets.values()))
        self.assertGreater(max(offsets.values()) - min(offsets.values()), 60.0)

    def test_a_loaded_trial_has_both_references_on_one_clock(self):
        plates = bp.load_trial(self.SUBJECT, f'{self.SESSION}/{self.BLOCK}/{self.TRIAL}')
        self.assertTrue(plates)
        self.assertTrue(any(n.endswith('__vicon') for n in plates))
        self.assertTrue(any(n.endswith('__biplane') for n in plates))

        # One trial, one clock: every plate must share t=0 or a joint angle differences two
        # plates across an offset. This is the invariant that broke on imove/s16.
        starts = {round(float(p.imu_trace.timestamps[0]), 9) for p in plates.values()}
        self.assertEqual(len(starts), 1, f'plates disagree about t=0: {starts}')

    def test_biplane_coverage_is_short_and_vicon_is_longer(self):
        """The 0.48 s of gold-standard truth is not a defect to work around -- it is what the
        validity mask exists for."""
        plates = bp.load_trial(self.SUBJECT, f'{self.SESSION}/{self.BLOCK}/{self.TRIAL}')
        biplane = [p for n, p in plates.items() if n.endswith('__biplane')][0]
        vicon = [p for n, p in plates.items() if n.endswith('__vicon')][0]
        self.assertLess(np.asarray(biplane.valid).mean(), 0.05)
        self.assertGreater(np.asarray(vicon.valid).mean(),
                           np.asarray(biplane.valid).mean())

    def test_the_alignment_actually_lines_the_signals_up(self):
        """The end-to-end check, on gyro MAGNITUDE because that is frame-free -- a vector
        residual conflates bad timing with a missing sensor-to-segment rotation, which is how
        a working sync first read as a 236% residual."""
        plates = bp.load_trial(self.SUBJECT, f'{self.SESSION}/{self.BLOCK}/{self.TRIAL}')
        for name, plate in plates.items():
            valid = np.asarray(plate.valid)
            if valid.sum() < 50:
                continue
            synthetic = plate.world_trace.calculate_imu_trace(skip_lin_acc=True)
            reference = np.linalg.norm(synthetic.gyro[valid], axis=1)
            measured = np.linalg.norm(plate.imu_trace.gyro[valid], axis=1)
            correlation = float(np.corrcoef(reference, measured)[0, 1])
            self.assertGreater(correlation, 0.4,
                               f'{name}: |omega| correlation {correlation:.2f} — the streams '
                               f'are not aligned')

    def _trigger(self):
        rows = bp.trigger_times()
        row = rows[(rows.subject == self.SUBJECT) & (rows.trial == self.TRIAL)]
        return float(row.iloc[0].vicon_utc.timestamp())


@require_data
class TestRegistry(unittest.TestCase):
    def test_the_source_enumerates_only_syncable_trials(self):
        from src.toolchest.building.sources import get_source
        trials = get_source('imove_biplane').enumerate_trials()
        self.assertGreater(len(trials), 100)
        # Every enumerated trial must have a trigger, or it cannot reach the IMU clock.
        named = bp.trigger_times()
        for subject, key in trials[:25]:
            trial = key.split('/')[-1]
            self.assertFalse(named[(named.subject == subject) &
                                   (named.trial == trial)].empty, f'{subject}/{key}')


if __name__ == '__main__':
    unittest.main()


class TestWindowLandedCheck(unittest.TestCase):
    """Did the reference window land where the IMU was doing the same thing?

    THE PEAK-TO-SIDELOBE RATIO DOES NOT ANSWER THIS. It scores how sharp the correlation peak
    was -- that the estimator was confident, not that it was right -- and a confident wrong
    answer is exactly what slipped through. On 12/Test1/A/LSHop3 the window landed where the
    IMU reads 9.4 deg/s while the reference over the same frames reads 222.1, and nothing
    said so.
    """

    @staticmethod
    def _plate(name, scale=1.0, seconds=4.0, rate=250.0, valid_from=0):
        from scipy.spatial.transform import Rotation
        from src.toolchest.IMUTrace import IMUTrace
        from src.toolchest.PlateTrial import PlateTrial
        from src.toolchest.WorldTrace import WorldTrace

        time = np.arange(int(seconds * rate)) / rate
        angles = np.column_stack([0.7 * np.sin(2 * np.pi * 1.3 * time),
                                  0.4 * np.sin(2 * np.pi * 0.9 * time),
                                  0.2 * np.sin(2 * np.pi * 2.3 * time)])
        world = WorldTrace(time, np.zeros((len(time), 3)),
                           Rotation.from_euler('zyx', angles).as_matrix())
        synthetic = world.calculate_imu_trace(skip_lin_acc=True)
        # `scale` stands in for a misplaced window: the same shape at the wrong amplitude is
        # what "the limb was moving this much / no it wasn't" looks like in one number.
        imu = IMUTrace(time, synthetic.gyro * scale, synthetic.acc, synthetic.mag)
        valid = np.zeros(len(time), dtype=bool)
        valid[valid_from:] = True
        return {name: PlateTrial(name, imu, WorldTrace(time, world.positions,
                                                       world.rotations, valid=valid))}

    def _ratios(self, plates):
        from src.toolchest.building.biplane import _check_windows_landed
        from src.toolchest.building.report import BuildReport
        report = BuildReport()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _check_windows_landed(plates, report)
        frame = report.to_frame()
        return frame, [w for w in caught if w.category is bp.SyncWindowWarning]

    def test_a_matched_window_passes_and_is_recorded(self):
        frame, fired = self._ratios(self._plate('ok'))
        self.assertEqual(fired, [])
        metrics = dict(zip(frame.metric, frame.value_num))
        self.assertAlmostEqual(metrics['motion_ratio'], 1.0, places=6)
        self.assertEqual(metrics['window_landed'], 1.0)

    def test_a_misplaced_window_is_caught(self):
        """The LSHop3 signature: the reference says the limb moved twenty times more than
        the IMU did over the frames it claims."""
        _, fired = self._ratios(self._plate('bad', scale=1 / 20.0))
        self.assertEqual(len(fired), 1)
        self.assertIn('wrong place', str(fired[0].message))

    def test_it_is_symmetric(self):
        """A window can be wrong in either direction — reference quieter than the IMU is the
        same failure as reference louder, so the ratio is taken max-over-min."""
        _, loud = self._ratios(self._plate('loud', scale=20.0))
        _, quiet = self._ratios(self._plate('quiet', scale=1 / 20.0))
        self.assertEqual(len(loud), len(quiet), 1)

    def test_a_short_window_is_skipped_rather_than_judged(self):
        """Below MIN_SYNC_CHECK_FRAMES the RMS is dominated by whichever few frames survived
        and the ratio is noise, so a verdict there would be a coin flip presented as a fact."""
        plates = self._plate('short', scale=1 / 20.0, valid_from=1000 - 5)
        frame, fired = self._ratios(plates)
        self.assertEqual(fired, [])
        self.assertNotIn('motion_ratio', set(frame.metric))

    def test_the_threshold_sits_in_the_measured_gap(self):
        """Set from data, not picked: correctly-synced plates measured 1.00-2.75 across 104
        plates and the broken trial's four read 5.88-23.70."""
        self.assertGreater(bp.MAX_SYNC_MOTION_RATIO, 2.75)
        self.assertLess(bp.MAX_SYNC_MOTION_RATIO, 5.88)
