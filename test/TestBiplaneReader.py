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
        positions, timestamps, used = markers
        self.assertEqual(positions.shape[1], 4)
        self.assertEqual(len(used), 4)
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

    def test_too_few_labels_returns_none_rather_than_raising(self):
        self.assertIsNone(bp.read_vicon_c3d(
            bp.vicon_path(self.SUBJECT, self.SESSION, self.TRIAL), ['NOSUCHMARKER']))

    def test_three_of_four_labels_still_reconstructs(self):
        """One absent label used to cost the whole cluster, and it need not: three
        non-collinear markers determine a rigid pose exactly. Subject 02's export has no RTIP
        at all, which alone cost every one of its 13 right-side trials both right-thigh
        plates -- 26 of the dataset's 36 absences, from a marker that was never needed."""
        full = bp.VICON_CLUSTERS[('left', 'lateral_shank')]
        markers = bp.read_vicon_c3d(
            bp.vicon_path(self.SUBJECT, self.SESSION, self.TRIAL),
            list(full) + ['NOSUCHMARKER'])
        self.assertIsNotNone(markers)
        positions, _, used = markers
        self.assertEqual(used, list(full))
        self.assertEqual(positions.shape[1], 4)

    def test_two_labels_is_refused_because_the_pose_is_underdetermined(self):
        """Two points leave a free rotation about the line through them. Returning a pose
        there would be inventing one axis of it."""
        full = bp.VICON_CLUSTERS[('left', 'lateral_shank')]
        self.assertIsNone(bp.read_vicon_c3d(
            bp.vicon_path(self.SUBJECT, self.SESSION, self.TRIAL), list(full[:2])))
        self.assertIsNotNone(bp.read_vicon_c3d(
            bp.vicon_path(self.SUBJECT, self.SESSION, self.TRIAL), list(full[:3])))

    def test_the_three_marker_fit_records_that_it_used_three(self):
        """A three-marker residual is exact by construction and means less than a four-marker
        one -- 0.238 mm against 0.465 on the same plate -- so the count travels with it."""
        from src.toolchest.building.report import BuildReport
        report = BuildReport()
        bp.load_trial('02', 'Test1/A/RSDrop1', report=report)
        frame = report.to_frame()
        rows = frame[(frame.step == 'S2_reconstruction')
                     & (frame.metric == 'n_markers_used')]
        counts = dict(zip(rows.entity, rows.value_num))
        self.assertEqual(counts['RSDrop1/lateral_thigh'], 3.0)
        self.assertEqual(counts['RSDrop1/lateral_shank'], 4.0)

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

    def test_the_threshold_clears_the_bulk_of_honest_plates(self):
        """Set from data, not picked. Across all 1462 plates the ratio runs q50 1.27 and
        q95 2.46, with 88 plates in (2, 3] that are drop landings where soft tissue alone
        moves the ratio -- so the threshold has to clear 3 or it sweeps those in. The
        subject-12-only reading that put a clean empty gap at 2.75-5.88 was too confident:
        cohort-wide there are 8 plates in (3, 4] and 7 in (4, 6]."""
        self.assertGreater(bp.MAX_SYNC_MOTION_RATIO, 3.0)
        # And loose enough to leave the 29 plates above 10 as the unambiguous catch.
        self.assertLess(bp.MAX_SYNC_MOTION_RATIO, 10.0)


class TestLagUsesMeasuredFramesOnly(unittest.TestCase):
    """The lag correlation runs on the valid span, not on interpolated pose.

    The Vicon clusters lose markers heavily -- across 72 clusters only 65% of frames see all
    four, and 24% see fewer than the three a pose needs -- and `fit_plate_to_template` fills
    the rest by interpolation. Differentiating an interpolated pose gives an angular velocity
    that was never measured, and correlating on it made the two sensors of a trial disagree
    about the lag by up to 40 s. Restricting to the valid span cut the trials whose sensors
    disagree by more than a second from 25 to 7.

    Trimming rather than masking is safe here because every one of the 52 lost runs sits at a
    trial EDGE -- median 529 frames, none in the interior -- so the valid part is contiguous.
    """

    @staticmethod
    def _trace(n=800, invalid_head=200, invalid_tail=150, rate=250.0):
        from scipy.spatial.transform import Rotation
        from src.toolchest.WorldTrace import WorldTrace
        time = np.arange(n) / rate
        angles = np.column_stack([0.6 * np.sin(2 * np.pi * 1.1 * time),
                                  0.3 * np.sin(2 * np.pi * 0.7 * time),
                                  np.zeros(n)])
        valid = np.ones(n, dtype=bool)
        valid[:invalid_head] = False
        valid[n - invalid_tail:] = False
        return WorldTrace(time, np.zeros((n, 3)),
                          Rotation.from_euler('zyx', angles).as_matrix(), valid=valid)

    def test_it_trims_to_the_valid_span(self):
        trace = self._trace()
        trimmed = bp._valid_span(trace)
        self.assertEqual(len(trimmed), 800 - 200 - 150)
        self.assertTrue(np.asarray(trimmed.valid).all())
        self.assertAlmostEqual(trimmed.timestamps[0], 200 / 250.0)

    def test_a_fully_valid_trace_is_untouched(self):
        trace = self._trace(invalid_head=0, invalid_tail=0)
        self.assertIs(bp._valid_span(trace), trace)

    def test_a_fully_invalid_trace_is_returned_whole(self):
        """So the caller gets the same 'no peak' answer it would have got anyway, rather than
        an empty array to reason about."""
        trace = self._trace(invalid_head=800, invalid_tail=0)
        self.assertIs(bp._valid_span(trace), trace)

    def test_the_lag_keeps_its_meaning_after_trimming(self):
        """L = s - w0 is invariant to where w0 is taken, so trimming must not shift the
        answer. If it did, every biplane window would move by the length of the head gap."""
        from src.toolchest.IMUTrace import IMUTrace
        trace = self._trace()
        synthetic = trace.calculate_imu_trace(skip_lin_acc=True)
        shift = 0.4
        imu = IMUTrace(trace.timestamps + shift, synthetic.gyro, synthetic.acc, synthetic.mag)

        lag, _ = bp.bracketed_lag(imu, trace, expected_lag_s=0.0, bracket_s=2.0)
        self.assertAlmostEqual(lag, shift, places=2)


@require_data
class TestViconReconstructionIsRecorded(unittest.TestCase):
    """The Vicon clusters go through the same template fit as every other dataset's markers,
    and the reader used to throw the fit report away -- so the biplane half was the one
    dataset with no S2_reconstruction rows, and its marker dropout was invisible."""

    def test_a_loaded_trial_emits_reconstruction_rows(self):
        from src.toolchest.building.report import BuildReport
        report = BuildReport()
        bp.load_trial('01', 'Test1/A/LSDrop1', report=report)
        frame = report.to_frame()
        rows = frame[frame.step == 'S2_reconstruction']
        self.assertFalse(rows.empty, 'S2_reconstruction emitted nothing')
        # One entity per SITE, not per plate: the two references share one marker fit.
        self.assertEqual(len(set(rows.entity)), 2)

    def test_it_records_what_the_dropout_investigation_needs(self):
        from src.toolchest.building.report import BuildReport
        report = BuildReport()
        bp.load_trial('01', 'Test1/A/LSDrop1', report=report)
        frame = report.to_frame()
        metrics = set(frame[frame.step == 'S2_reconstruction'].metric)
        for name in ('residual_median_mm', 'valid_fraction',
                     'min_marker_presence_fraction', 'n_frames_missing_a_marker'):
            self.assertIn(name, metrics)

    def test_the_biplane_poses_emit_no_reconstruction_rows(self):
        """They are already poses -- the fluoroscopy solved them, there are no markers to fit.
        A row here would claim a fit that never happened."""
        from src.toolchest.building.report import BuildReport
        report = BuildReport()
        bp.load_trial('01', 'Test1/A/LSDrop1', report=report)
        frame = report.to_frame()
        entities = set(frame[frame.step == 'S2_reconstruction'].entity)
        self.assertFalse(any('biplane' in e for e in entities), entities)


@require_data
class TestDeadMarkerChannels(unittest.TestCase):
    """A marker can be present as a LABEL and absent as DATA, and that is not the same check.

    Subject 18's RrunStance1 exports RTSA and then fills it in 0.1% of frames. All four labels
    are in the file, so the label check passed, but no frame has all four present -- and the
    template `fit_plate_to_template` estimates needs one, so it raised, `_vicon_world` returned
    None, and the whole sensor went with it, taking the biplane plate too. Two sensor-trials
    lost to a channel that was never populated.
    """

    def test_a_dead_channel_is_dropped_and_the_cluster_survives(self):
        markers = bp.read_vicon_c3d(bp.vicon_path('18', 'Test1', 'RrunStance1'),
                                    bp.VICON_CLUSTERS[('right', 'lateral_thigh')])
        self.assertIsNotNone(markers)
        _, _, used = markers
        self.assertEqual(len(used), 3, 'the 0.1%-present marker should have been dropped')
        self.assertNotIn('RTSA', used)

    def test_the_trial_now_produces_all_four_plates(self):
        plates = bp.load_trial('18', 'Test1/C/RrunStance1')
        self.assertEqual(len(plates), 4)
        self.assertIn('lateral_thigh_right__vicon', plates)
        # The biplane plate went with the Vicon one, so recovering one recovers both.
        self.assertIn('lateral_thigh_right__biplane', plates)

    def test_an_intermittent_marker_is_kept(self):
        """The threshold catches a DEAD channel, not a patchy one -- a marker present a
        quarter of the time still constrains the frames it appears in."""
        self.assertLessEqual(bp.MIN_MARKER_PRESENCE, 0.25)
        self.assertGreater(bp.MIN_MARKER_PRESENCE, 0.0)

    def test_a_healthy_cluster_keeps_all_four(self):
        """The control: the filter must not quietly thin a cluster that is fine."""
        markers = bp.read_vicon_c3d(bp.vicon_path('01', 'Test1', 'LSDrop1'),
                                    bp.VICON_CLUSTERS[('left', 'lateral_shank')])
        _, _, used = markers
        self.assertEqual(len(used), 4)


class TestPretriggerBracket(unittest.TestCase):
    """The biplane pre-trigger search, and the check on what it comes back with.

    THE BRACKET HAS TO REACH ZERO. Over 378 trials the fitted offset is tightly unimodal --
    299 inside 2.9-3.0 -- which makes a narrow bracket look safe. 06/RSHop2 is why it is not:
    both of its bones independently put the offset at -0.03 and -0.06, meaning that trial's
    biplane pose is already on the Vicon time base with no pre-trigger at all. A bracket of
    [2.48, 3.48] cannot reach zero, so the search pinned at 2.81 and displaced a 0.78 s window
    by 2.8 s -- the whole of that trial's factor-of-75 motion mismatch.
    """

    def test_the_fallback_bracket_reaches_zero(self):
        """The specific failure. A bracket that cannot express 'no pre-trigger' silently
        returns the nearest value it can, which is interior and looks fine."""
        self.assertLess(
            bp.BIPLANE_PRETRIGGER_S - bp.BIPLANE_PRETRIGGER_WIDE_BRACKET_S, 0.0)

    def test_the_default_bracket_stays_narrow(self):
        """Widening unconditionally is not free. On a periodic task the wide bracket reaches
        the NEXT STRIDE's correlation peak: searching every trial at +/-3.5 s aliased 82 of 89
        running-stance trials onto one, against 6 of the other 289. The narrow default is what
        protects them, and it is why the wide search is a fallback rather than the rule."""
        self.assertLess(bp.BIPLANE_PRETRIGGER_BRACKET_S,
                        bp.BIPLANE_PRETRIGGER_WIDE_BRACKET_S)
        self.assertGreater(bp.BIPLANE_PRETRIGGER_S - bp.BIPLANE_PRETRIGGER_BRACKET_S, 0.0)

    def test_the_anomaly_check_is_against_the_constant_not_the_bracket(self):
        """An edge check would not have caught 06/RSHop2: 2.81 is comfortably interior and
        still wrong by the width of the window. Only comparing against the hardware value
        catches it, so the threshold must be far tighter than the bracket."""
        self.assertLess(bp.PRETRIGGER_ANOMALY_S, bp.BIPLANE_PRETRIGGER_WIDE_BRACKET_S)

    def test_the_threshold_keeps_the_measured_bulk_quiet(self):
        """Measured over 378 trials the offset spans 2.4-3.5, so a threshold under ~0.52 would
        flag ordinary trials and stop meaning anything."""
        self.assertGreaterEqual(bp.PRETRIGGER_ANOMALY_S, 0.5)


@require_data
class TestPretriggerAnomalyOnRealTrials(unittest.TestCase):
    def _load(self, subject, key):
        from src.toolchest.building.report import BuildReport
        report = BuildReport()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            bp.load_trial(subject, key, report=report)
        frame = report.to_frame()
        return frame, [w for w in caught
                       if w.category is bp.PretriggerAnomalyWarning]

    @staticmethod
    def _metric(frame, name, how='median'):
        return getattr(frame[frame.metric == name].value_num, how)()

    def test_a_periodic_trial_does_not_alias_onto_the_next_stride(self):
        """The control on the fallback. 01/RrunStance2 fits 2.980 with the narrow default and
        5.600 when the wide search runs unconditionally -- one stride out, and its window
        still lands well enough that the motion check cannot tell. Only not searching wide
        keeps it right."""
        frame, _ = self._load('01', 'Test1/C/RrunStance2')
        self.assertAlmostEqual(self._metric(frame, 'biplane_lag_s'),
                               bp.BIPLANE_PRETRIGGER_S, delta=0.1)
        self.assertEqual(self._metric(frame, 'pretrigger_widened'), 0.0)

    def test_the_zero_pretrigger_trial_is_found_and_flagged(self):
        frame, fired = self._load('06', 'Test1/A/RSHop2')
        self.assertEqual(self._metric(frame, 'pretrigger_widened'), 1.0)
        self.assertAlmostEqual(self._metric(frame, 'biplane_lag_s'), 0.0, delta=0.2)
        self.assertEqual(len(fired), 1)

    def test_finding_it_is_what_lands_the_window(self):
        """The point of widening the bracket. Before it, this trial's biplane plates read a
        motion ratio of 73 and 75; the Vicon plates beside them were always fine."""
        frame, _ = self._load('06', 'Test1/A/RSHop2')
        self.assertLess(self._metric(frame, 'motion_ratio', 'max'),
                        bp.MAX_SYNC_MOTION_RATIO)

    def test_an_ordinary_trial_is_unaffected_and_silent(self):
        """Widening a bracket normally buys spurious matches. The joint estimate over both
        bones is what stops it, and this is the control that says so."""
        frame, fired = self._load('01', 'Test1/A/LSDrop1')
        self.assertAlmostEqual(self._metric(frame, 'biplane_lag_s'),
                               bp.BIPLANE_PRETRIGGER_S, delta=0.1)
        self.assertEqual(fired, [])
