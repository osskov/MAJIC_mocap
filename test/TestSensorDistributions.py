"""
Covers the detectors and reductions in experiments/sensor_distributions.py — the parts of
that module whose output is a decision or an aggregate rather than a plotted trace.

The failures worth testing here are all silent ones. An interval detector that clips a bout
short, mislabels sitting as standing, or accepts a stretch where only one foot was still
still produces intervals of a plausible shape and length; nothing raises, and the figure
looks fine. Same for the length-weighted noise reduction: weight it wrong and you get a
number in the right ballpark that quietly favours short intervals. So these build synthetic
trials whose correct answer is known by construction and check the values.

Synthetic throughout — no dataset needed.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd

from experiments.experiment_utils import EXPECTED_GRAVITY, JOINTS
from experiments.sensor_distributions import (FIELD_REFERENCE, HEIGHT_AXIS, MIN_SITTING_S,
                                              MIN_STANDING_S, SEGMENT_SENSOR, SIT_HEIGHT_DROP_M,
                                              STATIONARY_MARGIN_S, TRIAL_TABLES, _pair_correlation,
                                              _var_reduction,
                                              find_foot_stationary_intervals, joint_mag_consistency,
                                              label_activity_intervals, load_trial_table,
                                              mask_to_intervals, segment_joint_roles,
                                              sensor_stats, smooth_observability,
                                              summarize_distributions)
# Reusing the physics suite's synthetic plate rather than rebuilding one: it is the canonical
# "plate carrying a known world-frame reading in its body frame" fixture, and a local copy could
# drift from it, leaving these tests asserting against a differently-built world than the tests
# that pin the physics itself.
from test.TestExperimentPhysics import make_plate
from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace

FS = 100.0
DT = 1.0 / FS
UPRIGHT_HEIGHT = 1.0
SEATED_HEIGHT = UPRIGHT_HEIGHT - 2 * SIT_HEIGHT_DROP_M  # unambiguously below the sitting cutoff
# Boundaries are set by a threshold on a 1s centered rolling std, so they are only accurate to
# about half a window. Every timing assertion allows that much slack rather than pinning
# sample-exact edges the detector never claimed to produce.
EDGE_TOLERANCE_S = 1.0


def make_pelvis_plate(segments, rng_seed=0):
    """A pelvis PlateTrial assembled from (duration_s, linacc_std, height) blocks.

    Identity rotations throughout, so the world-frame accelerometer reading the labeler
    computes is just the body-frame one: this test is about the detector's thresholds and
    durations, not about frame conversion (TestExperimentPhysics covers that).
    """
    rng = np.random.default_rng(rng_seed)
    acc_blocks, height_blocks = [], []
    for duration_s, linacc_std, height in segments:
        n = int(round(duration_s * FS))
        linear = rng.normal(0.0, linacc_std, size=(n, 3)) if linacc_std > 0 else np.zeros((n, 3))
        acc_blocks.append(EXPECTED_GRAVITY + linear)
        height_blocks.append(np.full(n, height))

    acc = np.concatenate(acc_blocks, axis=0)
    n = len(acc)
    timestamps = np.arange(n) * DT
    positions = np.zeros((n, 3))
    positions[:, HEIGHT_AXIS] = np.concatenate(height_blocks)
    rotations = np.tile(np.eye(3), (n, 1, 1))

    return PlateTrial('pelvis_imu',
                      IMUTrace(timestamps, np.zeros((n, 3)), acc, np.tile([1.0, 0.0, 0.0], (n, 1))),
                      WorldTrace(timestamps, positions, rotations))


def make_foot_trace(quiet_windows, duration_s=120.0, gyro_std=0.5, noise=None, seed=0):
    """A raw foot IMUTrace that is noisy everywhere except inside `quiet_windows`.

    `noise` sets the per-modality std inside the quiet windows, i.e. the sensor's own noise
    floor, which is what the noise reduction is supposed to recover.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * FS))
    noise = noise or {'gyro': 0.001, 'acc': 0.01, 'mag': 0.002}

    quiet = np.zeros(n, dtype=bool)
    for start_s, end_s in quiet_windows:
        quiet[int(start_s * FS):int(end_s * FS)] = True

    gyro = rng.normal(0.0, gyro_std, size=(n, 3))
    gyro[quiet] = rng.normal(0.0, noise['gyro'], size=(quiet.sum(), 3))
    acc = np.tile(EXPECTED_GRAVITY, (n, 1)) + rng.normal(0.0, noise['acc'], size=(n, 3))
    mag = np.tile([1.0, 0.0, 0.0], (n, 1)) + rng.normal(0.0, noise['mag'], size=(n, 3))
    return IMUTrace(np.arange(n) * DT, gyro, acc, mag)


class TestMaskToIntervals(unittest.TestCase):
    def test_runs_are_half_open_and_complete(self):
        mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
        self.assertEqual(mask_to_intervals(mask), [(1, 3), (5, 6)])

    def test_runs_touching_both_ends(self):
        """A run flush against sample 0 or the final sample is still a run. The prepend/append
        padding exists for exactly this case; without it the leading run is lost and the
        trailing one never closes."""
        self.assertEqual(mask_to_intervals(np.array([1, 1, 0, 1, 1], dtype=bool)), [(0, 2), (3, 5)])
        self.assertEqual(mask_to_intervals(np.ones(4, dtype=bool)), [(0, 4)])

    def test_no_runs(self):
        self.assertEqual(mask_to_intervals(np.zeros(5, dtype=bool)), [])


class TestLabelActivityIntervals(unittest.TestCase):
    def setUp(self):
        # 30s ambulation, 25s seated, a 1s stand-up transient, 5s standing upright, 30s
        # ambulation. Ambulation dominates, so the median height is the upright one, as in a
        # real trial. The transient matters: a quiet run is labeled from its MEAN height, so
        # sitting and standing are only told apart when the movement between them breaks the
        # run — which standing up from a chair does (see label_activity_intervals).
        self.plate = make_pelvis_plate([
            (30.0, 3.0, UPRIGHT_HEIGHT),
            (25.0, 0.0, SEATED_HEIGHT),
            (1.0, 2.0, SEATED_HEIGHT),
            (5.0, 0.0, UPRIGHT_HEIGHT),
            (30.0, 3.0, UPRIGHT_HEIGHT),
        ])
        self.labels = label_activity_intervals({'pelvis_imu': self.plate}, FS)

    def test_finds_the_seated_bout_at_the_right_time(self):
        self.assertEqual(len(self.labels['sitting']), 1)
        start, end = self.labels['sitting'][0]
        self.assertAlmostEqual(start / FS, 30.0, delta=EDGE_TOLERANCE_S)
        self.assertAlmostEqual(end / FS, 55.0, delta=EDGE_TOLERANCE_S)

    def test_quiet_at_upright_height_is_standing_not_sitting(self):
        """The whole sitting/standing split is one height comparison, and getting it backwards
        would relabel every seated bout as standing while still producing intervals in
        plausible places."""
        self.assertEqual(len(self.labels['standing']), 1)
        start, end = self.labels['standing'][0]
        self.assertAlmostEqual(start / FS, 56.0, delta=EDGE_TOLERANCE_S)
        self.assertAlmostEqual(end / FS, 61.0, delta=EDGE_TOLERANCE_S)

    def test_labels_partition_the_trial(self):
        """Every sample lands in exactly one label. Ambulation is defined as the complement of
        the quiet intervals, so a gap or an overlap here means a stretch of trial is either
        counted twice in the pooled distributions or dropped from them."""
        n = len(self.plate)
        covered = np.zeros(n, dtype=int)
        for intervals in self.labels.values():
            for start, end in intervals:
                covered[start:end] += 1
        np.testing.assert_array_equal(covered, np.ones(n, dtype=int))

    def test_short_quiet_bouts_are_not_labeled(self):
        """A brief pause mid-gait is not a standing bout, and a seated bout shorter than
        MIN_SITTING_S is not enough to fit anything over."""
        plate = make_pelvis_plate([
            (20.0, 3.0, UPRIGHT_HEIGHT),
            (MIN_STANDING_S / 2, 0.0, UPRIGHT_HEIGHT),   # too brief to count as standing
            (20.0, 3.0, UPRIGHT_HEIGHT),
            (MIN_SITTING_S / 2, 0.0, SEATED_HEIGHT),     # too brief to count as sitting
            (20.0, 3.0, UPRIGHT_HEIGHT),
        ])
        labels = label_activity_intervals({'pelvis_imu': plate}, FS)
        self.assertEqual(labels['sitting'], [])
        self.assertEqual(labels['standing'], [])

    def test_a_trial_without_sitting_yields_no_sitting(self):
        """What a walking trial should produce: no seated bouts, and no crash from the empty
        case (the pooled figures depend on this path for every walking trial)."""
        plate = make_pelvis_plate([(60.0, 3.0, UPRIGHT_HEIGHT)])
        labels = label_activity_intervals({'pelvis_imu': plate}, FS)
        self.assertEqual(labels['sitting'], [])
        self.assertTrue(len(labels['ambulation']) >= 1)


class TestFootStationaryIntervals(unittest.TestCase):
    def test_requires_both_feet_still(self):
        """One foot planted while the other shifts is not a stationary period. Accepting it
        would measure that foot's real motion as the sensor's noise floor."""
        both = {'calcn_l_imu': make_foot_trace([(20.0, 60.0)], seed=1),
                'calcn_r_imu': make_foot_trace([(20.0, 60.0)], seed=2)}
        one = {'calcn_l_imu': make_foot_trace([(20.0, 60.0)], seed=1),
               'calcn_r_imu': make_foot_trace([], seed=2)}

        intervals, sensors = find_foot_stationary_intervals(both)
        self.assertEqual(sensors, ['calcn_l_imu', 'calcn_r_imu'])
        self.assertEqual(len(intervals), 1)
        self.assertEqual(find_foot_stationary_intervals(one)[0], [])

    def test_margins_are_trimmed_from_each_end(self):
        """The reported interval is strictly inside the quiet stretch: the rolling-std window
        straddles motion at each edge, so the untrimmed edges are transitions, not noise."""
        traces = {'calcn_l_imu': make_foot_trace([(20.0, 60.0)], seed=1),
                  'calcn_r_imu': make_foot_trace([(20.0, 60.0)], seed=2)}
        (start, end), = find_foot_stationary_intervals(traces)[0]
        self.assertAlmostEqual(start / FS, 20.0 + STATIONARY_MARGIN_S, delta=EDGE_TOLERANCE_S)
        self.assertAlmostEqual(end / FS, 60.0 - STATIONARY_MARGIN_S, delta=EDGE_TOLERANCE_S)

    def test_short_pauses_are_rejected(self):
        traces = {'calcn_l_imu': make_foot_trace([(20.0, 25.0)], seed=1),
                  'calcn_r_imu': make_foot_trace([(20.0, 25.0)], seed=2)}
        self.assertEqual(find_foot_stationary_intervals(traces)[0], [])

    def test_no_foot_sensors(self):
        self.assertEqual(find_foot_stationary_intervals({'pelvis_imu': make_foot_trace([])}), ([], []))


class TestSensorStats(unittest.TestCase):
    """The noise reduction: a length-weighted mean over intervals, feet only."""

    def setUp(self):
        n = 3000
        timestamps = np.arange(n) * DT
        acc = np.tile(EXPECTED_GRAVITY, (n, 1))
        mag = np.tile([1.0, 0.0, 0.0], (n, 1))
        trace = IMUTrace(timestamps, np.zeros((n, 3)), acc, mag)
        world = WorldTrace(timestamps, np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1)))
        self.plates = {'calcn_r_imu': PlateTrial('calcn_r_imu', trace, world),
                       'pelvis_imu': PlateTrial('pelvis_imu', trace.copy(), world.copy())}

    def _plates_with_known_stds(self, stds_and_lengths, extra_sensors=()):
        """Plates whose foot gyro std inside each interval is exactly `std`, built from a
        two-level square wave (the std of +-a alternating is exactly a).

        The noise floor is read off the PLATES now, not a separately-loaded raw trace:
        alignment no longer discards the inertial record, so the anchored-foot pauses are
        present in the plate itself.
        """
        total = sum(length for _, length in stds_and_lengths)
        gyro = np.zeros((total, 3))
        cursor = 0
        for std, length in stds_and_lengths:
            block = np.empty(length)
            block[0::2], block[1::2] = std, -std
            gyro[cursor:cursor + length, :] = block[:, None]
            cursor += length
        timestamps = np.arange(total) * DT
        trace = IMUTrace(timestamps, gyro, np.tile(EXPECTED_GRAVITY, (total, 1)),
                         np.tile([1.0, 0.0, 0.0], (total, 1)))
        world = WorldTrace(timestamps, np.zeros((total, 3)), np.tile(np.eye(3), (total, 1, 1)))
        plates = {'calcn_r_imu': PlateTrial('calcn_r_imu', trace, world)}
        for name in extra_sensors:
            plates[name] = PlateTrial(name, trace.copy(), world.copy())
        return plates

    def test_noise_is_length_weighted_across_intervals(self):
        """A long quiet interval must count for more than a short one. An unweighted mean of
        the two per-interval stds would give 0.055 here instead of 0.019 — same order of
        magnitude, silently wrong, and biased toward whatever the shortest interval saw."""
        plates = self._plates_with_known_stds([(0.01, 900), (0.1, 100)])
        stats = sensor_stats(plates, [(0, 900), (900, 1000)])
        row = stats[stats['sensor'] == 'calcn_r_imu'].iloc[0]
        expected = (0.01 * 900 + 0.1 * 100) / 1000
        self.assertAlmostEqual(row['gyro_noise_x'], expected, places=6)
        self.assertEqual(row['n_stationary_samples'], 1000)

    def test_non_foot_sensors_get_no_noise_estimate(self):
        """Only the feet are ground-anchored. Filling these columns for the pelvis would look
        like more data and be a measurement of postural sway."""
        plates = self._plates_with_known_stds([(0.01, 1000)], extra_sensors=('pelvis_imu',))
        stats = sensor_stats(plates, [(0, 1000)])
        pelvis = stats[stats['sensor'] == 'pelvis_imu'].iloc[0]
        self.assertTrue(np.isnan(pelvis['gyro_noise_x']))
        self.assertEqual(pelvis['n_stationary_samples'], 0)

    def test_no_stationary_intervals_leaves_noise_unset(self):
        stats = sensor_stats(self._plates_with_known_stds([(0.01, 1000)]), [])
        self.assertTrue(stats['gyro_noise_x'].isna().all())
        self.assertTrue((stats['n_stationary_samples'] == 0).all())
        # The distortion columns do not depend on stationarity and must still be filled.
        self.assertTrue(stats['mag_norm_std'].notna().all())


class TestSmoothObservability(unittest.TestCase):
    def test_output_is_non_negative(self):
        """o^J is a norm. filtfilt rings after an impulse and would otherwise return negative
        observability, which is not a value the metric can take."""
        spiky = np.zeros(1000)
        spiky[500] = 5000.0
        self.assertGreaterEqual(smooth_observability(spiky, FS).min(), 0.0)

    def test_constant_signal_survives(self):
        constant = np.full(1000, 42.0)
        np.testing.assert_allclose(smooth_observability(constant, FS), constant, rtol=1e-6)

    def test_spikes_are_capped_not_kept(self):
        """Winsorizing happens before filtering, so the tallest spike must not survive at full
        height — that is the whole reason the figures are legible."""
        rng = np.random.default_rng(0)
        signal = np.abs(rng.normal(100.0, 20.0, size=2000))
        signal[1000] = 100000.0
        self.assertLess(smooth_observability(signal, FS).max(), signal.max() / 10)


class TestSegmentJointRoles(unittest.TestCase):
    def test_roles_match_the_pipeline_joint_definitions(self):
        """The table is derived from JOINTS rather than written out; this pins that it stays
        derived. Every joint must appear exactly twice — once as a parent, once as a child."""
        roles = segment_joint_roles()
        pairs = [(joint, role) for entries in roles.values() for joint, role in entries]
        for joint in JOINTS:
            self.assertEqual(sorted(role for j, role in pairs if j == joint), ['child', 'parent'])
        self.assertEqual(set(roles), set(SEGMENT_SENSOR))

    def test_pelvis_borders_three_joints(self):
        self.assertEqual(sorted(joint for joint, _ in segment_joint_roles()['Pelvis']),
                         ['L_Hip', 'Lumbar', 'R_Hip'])


class TestSummarizeDistributions(unittest.TestCase):
    def setUp(self):
        # Two "trials" of the same segment with deliberately different lengths and levels, so
        # a mean-of-medians reduction cannot coincide with the pooled median.
        self.segment_df = pd.concat([
            pd.DataFrame({'subject': 'Subject01', 'activity': 'walking', 'segment': 'Pelvis',
                          'linacc': np.arange(100, dtype=float), 'magdev': np.zeros(100)}),
            pd.DataFrame({'subject': 'Subject02', 'activity': 'walking', 'segment': 'Pelvis',
                          'linacc': np.full(900, 1000.0), 'magdev': np.zeros(900)}),
        ], ignore_index=True)
        self.summary = summarize_distributions(self.segment_df, pd.DataFrame())

    def _row(self, subject, activity, metric='linacc'):
        match = self.summary[(self.summary['subject'] == subject)
                             & (self.summary['activity'] == activity)
                             & (self.summary['metric'] == metric)]
        self.assertEqual(len(match), 1)
        return match.iloc[0]

    def test_margins_re_aggregate_the_samples(self):
        """The pooled row is computed over the pooled samples, not averaged from the per-trial
        rows. Averaging medians here would give 549.5 rather than the true pooled median of
        1000, and would silently weight a 100-sample trial like a 900-sample one."""
        pooled = self._row('all', 'all')
        self.assertEqual(pooled['n_samples'], 1000)
        self.assertAlmostEqual(pooled['p50'], 1000.0)

    def test_per_trial_rows_are_present_and_correct(self):
        row = self._row('Subject01', 'walking')
        self.assertEqual(row['n_samples'], 100)
        self.assertAlmostEqual(row['p50'], np.median(np.arange(100)))

    def test_units_are_recorded(self):
        self.assertEqual(self._row('all', 'all', 'magdev')['unit'], 'a.u.')
        self.assertEqual(self._row('all', 'all', 'linacc')['unit'], 'm/s^2')

    def test_empty_input(self):
        self.assertTrue(summarize_distributions(pd.DataFrame(), pd.DataFrame()).empty)


class TestJointMagConsistency(unittest.TestCase):
    """The child-vs-parent field comparison, and the reference direction it uses.

    The direction matters because var_reduction is asymmetric: which sensor references which
    decides the sign whenever the two differ in variance. FIELD_REFERENCE is supposed to be a
    fixed anatomical rule, so these tests pin that it is applied mechanically and that both
    directions stay recorded — a rule that quietly became "pick whichever came out positive"
    would still produce plausible numbers.
    """

    @staticmethod
    def _pair_plates(parent_gain=2.1, shared_sd=1.0, noise_sd=0.1, n=4000, seed=0):
        """A lumbar pair sharing one fluctuating field, with the PARENT seeing it amplified by
        `parent_gain`, plus a little independent noise on each. Identity rotations, so the world
        frame is the body frame.

        The gain is what makes this a faithful fixture rather than a convenient one. The real
        lumbar pair has corr 0.86 alongside Var(parent)/Var(child) = 4.55, which a shared-signal-
        plus-independent-noise model cannot produce at all (it caps corr at 1/sqrt(k) = 0.47).
        What fits is a common field variation that the pelvis sees at larger amplitude, being the
        sensor nearer whatever is distorting it — so unit-gain substitution of the parent
        over-corrects, and reversing the direction under-corrects harmlessly instead.
        """
        rng = np.random.default_rng(seed)
        timestamps = np.arange(n) * DT
        world = WorldTrace(timestamps, np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1)))
        shared = rng.normal(0.0, shared_sd, size=(n, 3))
        base = np.array([1.0, 0.0, 0.0])

        plates = {}
        for name, gain in (('pelvis_imu', parent_gain), ('torso_imu', 1.0)):
            mag = base + gain * shared + rng.normal(0.0, noise_sd, size=(n, 3))
            trace = IMUTrace(timestamps, np.zeros((n, 3)), np.tile(EXPECTED_GRAVITY, (n, 1)), mag)
            plates[name] = PlateTrial(name, trace, world.copy())
        return plates

    def test_lumbar_reference_is_the_torso(self):
        """The one declared exception: the lumbar references the torso, not the kinematic
        parent, because the torso is the cleaner sensor of that pair."""
        self.assertEqual(FIELD_REFERENCE.get('Lumbar'), 'child')
        plates = self._pair_plates()
        row = joint_mag_consistency(plates, np.array([1.0, 0.0, 0.0])).iloc[0]
        self.assertEqual(row['joint'], 'Lumbar')
        self.assertEqual(row['reference_sensor'], 'torso_imu')
        self.assertEqual(row['target_sensor'], 'pelvis_imu')

    def test_every_other_joint_references_its_kinematic_parent(self):
        self.assertEqual([j for j in JOINTS if FIELD_REFERENCE.get(j, 'parent') != 'parent'],
                         ['Lumbar'])

    def test_headline_value_follows_the_declared_direction(self):
        """var_reduction must be the configured direction's value, and BOTH directions must be
        stored so the choice can be undone from the saved table."""
        plates = self._pair_plates()
        row = joint_mag_consistency(plates, np.array([1.0, 0.0, 0.0])).iloc[0]
        self.assertAlmostEqual(row['var_reduction'], row['var_reduction_child_ref'], places=9)
        self.assertNotAlmostEqual(row['var_reduction'], row['var_reduction_parent_ref'], places=3)

    def test_the_flip_is_what_rescues_a_noisy_parent(self):
        """The substantive claim behind FIELD_REFERENCE: with a noisy parent and a clean child,
        referencing the parent loses and referencing the child wins — off identical samples."""
        row = joint_mag_consistency(self._pair_plates(),
                                    np.array([1.0, 0.0, 0.0])).iloc[0]
        self.assertLess(row['var_reduction_parent_ref'], 0.0)
        self.assertGreater(row['var_reduction_child_ref'], 0.5)

    def test_correlation_is_direction_independent(self):
        """The evidence for the premise — that the two sensors' fields move together — is
        symmetric, unlike var_reduction. This is why it is reported alongside."""
        plates = self._pair_plates()
        row = joint_mag_consistency(plates, np.array([1.0, 0.0, 0.0])).iloc[0]
        flipped = joint_mag_consistency(
            {'pelvis_imu': plates['torso_imu'], 'torso_imu': plates['pelvis_imu']},
            np.array([1.0, 0.0, 0.0])).iloc[0]
        self.assertAlmostEqual(row['corr_target_reference'], flipped['corr_target_reference'],
                               places=9)

    def test_acceleration_departure_columns_separate_gravity_from_motion(self):
        """A plate reading gravity alone must score zero non-gravity acceleration, and one with a
        known linear acceleration must score exactly that. This is the pair of columns behind the
        report's acc_std comparison, so a sign or frame error here would quietly rescale it."""
        gravity_only = make_plate(name='torso_imu')
        row = sensor_stats({'torso_imu': gravity_only}, []).iloc[0]
        self.assertAlmostEqual(row['linacc_rms'], 0.0, places=6)
        self.assertAlmostEqual(row['acc_norm_std'], 0.0, places=6)
        self.assertAlmostEqual(row['acc_norm_median'], np.linalg.norm(EXPECTED_GRAVITY), places=6)

        # A constant world-frame linear acceleration on top of gravity: |a_world - g| is that
        # vector's norm at every sample, whatever the plate's orientation is doing.
        world_acc = np.array([0.0, 0.0, 3.0])
        moving = make_plate(name='torso_imu', world_acc=world_acc)
        row = sensor_stats({'torso_imu': moving}, []).iloc[0]
        self.assertAlmostEqual(row['linacc_rms'], np.linalg.norm(world_acc), places=6)
        self.assertAlmostEqual(row['linacc_median'], np.linalg.norm(world_acc), places=6)

    def test_acc_norm_std_understates_perpendicular_acceleration(self):
        """The documented reason both columns are reported: |acc| adds in quadrature, so motion
        perpendicular to gravity barely moves it while linacc sees the whole thing. If these two
        ever agreed, one of them would be computed wrong."""
        perpendicular = make_plate(name='torso_imu', world_acc=np.array([3.0, 0.0, 0.0]))
        row = sensor_stats({'torso_imu': perpendicular}, []).iloc[0]
        self.assertAlmostEqual(row['linacc_rms'], 3.0, places=6)
        # |[3, 9.81, 0]| = 10.26 at every sample — constant, so its std is 0 despite 3 m/s^2 of motion
        self.assertAlmostEqual(row['acc_norm_std'], 0.0, places=6)
        self.assertGreater(row['linacc_rms'], 100 * row['acc_norm_std'])

    def test_global_field_is_the_zero_of_the_reduction_scale(self):
        """The reported baseline is definitional, not measured: referencing any CONSTANT field
        removes exactly none of the target's variance, whatever that constant is. This is what
        lets rho^2 and var_reduction share a scale, given that a correlation against a constant
        is undefined (0/0)."""
        rng = np.random.default_rng(7)
        target = rng.normal(0, 1.0, size=(2000, 3))
        for constant in ([0.0, 0.0, 0.0], [1.0, -0.5, 40.0]):
            reference = np.tile(np.array(constant), (len(target), 1))
            self.assertAlmostEqual(_var_reduction(target, reference), 0.0, places=12)
        row = joint_mag_consistency(self._pair_plates(), np.array([1.0, 0.0, 0.0])).iloc[0]
        self.assertEqual(row['var_reduction_global'], 0.0)

    def test_best_possible_reduction_is_rho_squared_and_bounds_the_actual(self):
        """rho^2 is the ceiling: forcing unit gain can never beat the optimal gain, and equals it
        only when the two sensors have matched variance. A `best` below `actual` would mean the
        pair of formulas disagree."""
        row = joint_mag_consistency(self._pair_plates(), np.array([1.0, 0.0, 0.0])).iloc[0]
        self.assertAlmostEqual(row['var_reduction_best'], row['corr_target_reference'] ** 2, places=12)
        self.assertGreaterEqual(row['var_reduction_best'], row['var_reduction'])
        self.assertGreaterEqual(row['var_reduction_best'], row['var_reduction_parent_ref'])

    def test_best_possible_reduction_is_direction_independent(self):
        """Unlike var_reduction, rho^2 cannot be moved by the FIELD_REFERENCE choice — which is
        the property that makes it quotable without also quoting the direction."""
        plates = self._pair_plates()
        row = joint_mag_consistency(plates, np.array([1.0, 0.0, 0.0])).iloc[0]
        flipped = joint_mag_consistency(
            {'pelvis_imu': plates['torso_imu'], 'torso_imu': plates['pelvis_imu']},
            np.array([1.0, 0.0, 0.0])).iloc[0]
        self.assertAlmostEqual(row['var_reduction_best'], flipped['var_reduction_best'], places=12)

    def test_var_reduction_matches_its_closed_form(self):
        """var_reduction == 2*rho*sqrt(k) - k for k = Var(reference)/Var(target). The identity is
        the whole basis for reading the sign, so it is checked against the implementation."""
        rng = np.random.default_rng(3)
        target = rng.normal(0, 1.0, size=(5000, 3))
        reference = 0.6 * target + rng.normal(0, 0.4, size=(5000, 3))
        k = np.var(reference, axis=0).sum() / np.var(target, axis=0).sum()
        # rho comes from the module's own estimator, so this pins that the two agree exactly.
        # They only do because both use the population (ddof=0) convention; np.cov's ddof=1
        # default breaks the identity in the fourth decimal.
        rho = _pair_correlation(target, reference)
        self.assertAlmostEqual(_var_reduction(target, reference), 2 * rho * np.sqrt(k) - k, places=9)


class TestLoadTrialTable(unittest.TestCase):
    def test_unknown_table_raises(self):
        """A typo'd table name must not look like "the experiment has not been run"."""
        with self.assertRaises(ValueError):
            load_trial_table('segment_sample')
        self.assertIn('segment_samples', TRIAL_TABLES)


if __name__ == '__main__':
    unittest.main()
