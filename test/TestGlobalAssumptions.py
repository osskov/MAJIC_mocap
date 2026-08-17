"""
Covers the detectors and reductions in experiments/global_assumptions.py — the parts of that
module whose output is a decision or an aggregate rather than a plotted trace.

The failures worth testing here are all silent ones. A static detector that clips a bout short,
accepts a slowly-turning sensor, or drops the record edges still produces intervals of a
plausible shape and length; nothing raises, and the figure looks fine. Same for the noise
reduction: take the std over the pooled mask instead of within each interval and you get a
number in the right ballpark that is really measuring the difference between two poses. So
these build synthetic trials whose correct answer is known by construction and check the values.

Synthetic throughout — no dataset needed.
"""
import os
import unittest
import warnings

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd

from experiments.experiment_utils import EXPECTED_GRAVITY, JOINTS
from experiments.global_assumptions import (ALBORNO, BODY_STATIC_MARGIN_S, BODY_STATIC_MIN_S,
                                            DATASETS, IMOVE, MIN_SITTING_S, MIN_STANDING_S,
                                            MIN_STATIC_S, REGIMES, SIT_HEIGHT_DROP_M,
                                            MAGDEV_ARMS, METRIC_UNITS,
                                            RECOMPUTABLE_TABLES, resolve_reference,
                                            subject_reference_table,
                                            SEGMENT_METRICS, trial_field_vector,
                                            STATIC_GYRO_MAX, TRIAL_TABLES, HEIGHT_AXIS,
                                            _pair_correlation, _var_reduction, body_static_mask,
                                            angle_between_deg, built_trials,
                                            canonical_joint, compute_trial,
                                            dataset_dir, enumerate_trials, orphaned_trials,
                                            placement_of,
                                            drop_short_runs, field_consistency, interval_noise,
                                            joint_stats, label_posture_intervals, load_trial_table,
                                            mask_to_intervals, mocap_motion_check,
                                            pooled_world_field, regime_masks, segment_joint_roles,
                                            segment_samples, sensor_stats, smooth_observability,
                                            static_mask, subject_field_from_trials, summarize,
                                            trial_field_table)
from src.toolchest.building.sources import SOURCES, UNSYNCABLE_TRIALS
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
WORLD_FIELD = np.array([1.0, 0.0, 0.0])
# The posture detector's boundaries come from a threshold on a 1 s centered rolling std, so they
# are only accurate to about half a window. Timing assertions allow that much slack rather than
# pinning sample-exact edges the detector never claimed to produce.
EDGE_TOLERANCE_S = 1.0
# The static detector's window is far shorter, so its edges are correspondingly tighter.
STATIC_EDGE_TOLERANCE_S = 0.3


def make_trace(gyro, acc=None, mag=None, fs=FS):
    """An IMUTrace of the given gyro, defaulting to a sensor reading gravity and WORLD_FIELD."""
    n = len(gyro)
    timestamps = np.arange(n) / fs
    acc = np.tile(EXPECTED_GRAVITY, (n, 1)) if acc is None else acc
    mag = np.tile(WORLD_FIELD, (n, 1)) if mag is None else mag
    return IMUTrace(timestamps, np.asarray(gyro, dtype=float), acc, mag)


def make_identity_plate(name, gyro, acc=None, mag=None, positions=None, valid=None, fs=FS):
    """A PlateTrial with identity rotations, so the world frame IS the body frame.

    That is deliberate for these tests: they are about thresholds, durations and reductions,
    not about frame conversion (TestExperimentPhysics covers that), and an identity rotation
    means the expected value of every world-frame quantity is readable off the input.
    """
    trace = make_trace(gyro, acc, mag, fs)
    n = len(trace)
    positions = np.zeros((n, 3)) if positions is None else positions
    world = WorldTrace(trace.timestamps, positions, np.tile(np.eye(3), (n, 1, 1)), valid=valid)
    return PlateTrial(name, trace, world)


def still_gyro(n, level=0.0):
    return np.full((n, 3), level / np.sqrt(3))


def moving_gyro(n, level=1.0, seed=0):
    """Gyro whose magnitude is `level` at every sample — not noise around it, so a rolling
    maximum sees exactly `level` and the threshold comparison is exact."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, 3))
    return level * direction / np.linalg.norm(direction, axis=1, keepdims=True)


def gyro_blocks(blocks, seed=0):
    """Concatenates (duration_s, |gyro|) blocks into one gyro array at FS."""
    parts = []
    for i, (duration_s, level) in enumerate(blocks):
        n = int(round(duration_s * FS))
        parts.append(still_gyro(n, level) if level < 1e-9 else moving_gyro(n, level, seed + i))
    return np.concatenate(parts, axis=0)


def make_pelvis_plate(segments, rng_seed=0):
    """A pelvis PlateTrial assembled from (duration_s, linacc_std, height) blocks, for the
    posture labeler."""
    rng = np.random.default_rng(rng_seed)
    acc_blocks, height_blocks = [], []
    for duration_s, linacc_std, height in segments:
        n = int(round(duration_s * FS))
        linear = rng.normal(0.0, linacc_std, size=(n, 3)) if linacc_std > 0 else np.zeros((n, 3))
        acc_blocks.append(EXPECTED_GRAVITY + linear)
        height_blocks.append(np.full(n, height))

    acc = np.concatenate(acc_blocks, axis=0)
    n = len(acc)
    positions = np.zeros((n, 3))
    positions[:, HEIGHT_AXIS] = np.concatenate(height_blocks)
    return make_identity_plate('pelvis_imu', np.zeros((n, 3)), acc=acc, positions=positions)


# ==============================================================================
# Interval plumbing
# ==============================================================================

class TestMaskToIntervals(unittest.TestCase):
    def test_runs_are_half_open_and_complete(self):
        mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
        self.assertEqual(mask_to_intervals(mask), [(1, 3), (5, 6)])

    def test_runs_touching_both_ends(self):
        """A run flush against sample 0 or the final sample is still a run. The prepend/append
        padding exists for exactly this case; without it the leading run is lost and the
        trailing one never closes — and in this experiment the leading and trailing runs are the
        pre- and post-trial pauses, which carry most of the static time."""
        self.assertEqual(mask_to_intervals(np.array([1, 1, 0, 1, 1], dtype=bool)), [(0, 2), (3, 5)])
        self.assertEqual(mask_to_intervals(np.ones(4, dtype=bool)), [(0, 4)])

    def test_no_runs(self):
        self.assertEqual(mask_to_intervals(np.zeros(5, dtype=bool)), [])

    def test_drop_short_runs_keeps_only_long_enough_ones(self):
        mask = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
        np.testing.assert_array_equal(drop_short_runs(mask, 3),
                                      np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=bool))

    def test_drop_short_runs_is_a_noop_below_two(self):
        mask = np.array([1, 0, 1], dtype=bool)
        np.testing.assert_array_equal(drop_short_runs(mask, 1), mask)


# ==============================================================================
# Static detection
# ==============================================================================

class TestStaticMask(unittest.TestCase):
    def test_finds_a_still_stretch_at_the_right_time(self):
        gyro = gyro_blocks([(5.0, 1.0), (10.0, 0.0), (5.0, 1.0)])
        mask = static_mask(make_trace(gyro))
        intervals = mask_to_intervals(mask)
        self.assertEqual(len(intervals), 1)
        start, end = intervals[0]
        self.assertAlmostEqual(start / FS, 5.0, delta=STATIC_EDGE_TOLERANCE_S)
        self.assertAlmostEqual(end / FS, 15.0, delta=STATIC_EDGE_TOLERANCE_S)

    def test_slow_steady_rotation_is_moving_not_static(self):
        """A rolling MAXIMUM, not a rolling standard deviation. A sensor turning steadily at a
        constant rate has near-zero gyro VARIANCE, so a std-based detector calls it still — and
        a steadily rotating sensor is precisely the case where the gravity assumption is being
        swept through the body frame."""
        steady = np.full((int(20 * FS), 3), 3 * STATIC_GYRO_MAX / np.sqrt(3))
        self.assertFalse(static_mask(make_trace(steady)).any())

    def test_the_threshold_is_where_it_says_it_is(self):
        below = gyro_blocks([(10.0, 0.9 * STATIC_GYRO_MAX)])
        above = gyro_blocks([(10.0, 1.1 * STATIC_GYRO_MAX)])
        self.assertTrue(static_mask(make_trace(below)).all())
        self.assertFalse(static_mask(make_trace(above)).any())

    def test_brief_pauses_are_rejected(self):
        """A momentary lull mid-motion is not a static stretch, and counting it as one would
        put transition samples into the noise floor."""
        gyro = gyro_blocks([(5.0, 1.0), (MIN_STATIC_S / 3, 0.0), (5.0, 1.0)])
        self.assertFalse(static_mask(make_trace(gyro)).any())

    def test_a_pause_at_the_very_start_of_the_record_is_found(self):
        """min_periods=1 at the window edges, not NaN. The longest genuinely still stretches in
        both datasets are the pauses before the first task and after the last, and those sit at
        the very start and end of the record — requiring a full window there would discard
        exactly the samples the noise floor most wants."""
        gyro = gyro_blocks([(10.0, 0.0), (5.0, 1.0), (10.0, 0.0)])
        mask = static_mask(make_trace(gyro))
        self.assertTrue(mask[0])
        self.assertTrue(mask[-1])
        self.assertEqual(len(mask_to_intervals(mask)), 2)

    def test_is_blind_to_the_accelerometer_and_magnetometer(self):
        """The detector must not read either quantity under test — otherwise "the accelerometer
        reads gravity during the stretches we selected for reading gravity" is a tautology
        rather than a measurement. Same gyro, wildly different acc and mag, same mask."""
        gyro = gyro_blocks([(5.0, 1.0), (10.0, 0.0), (5.0, 1.0)])
        n = len(gyro)
        rng = np.random.default_rng(0)
        plain = make_trace(gyro)
        corrupted = make_trace(gyro,
                               acc=rng.normal(0.0, 50.0, size=(n, 3)),
                               mag=rng.normal(0.0, 10.0, size=(n, 3)))
        np.testing.assert_array_equal(static_mask(plain), static_mask(corrupted))


class TestBodyStaticMask(unittest.TestCase):
    def _masks(self, *specs):
        return {f"s{i}": static_mask(make_trace(gyro_blocks(spec, seed=i)))
                for i, spec in enumerate(specs)}

    def test_requires_every_sensor_to_be_still_at_once(self):
        """One foot planted while the subject shifts weight on the other is not a whole-body
        pause. Accepting it would measure that other sensor's real motion as the noise floor."""
        together = self._masks([(5.0, 1.0), (20.0, 0.0), (5.0, 1.0)],
                               [(5.0, 1.0), (20.0, 0.0), (5.0, 1.0)])
        apart = self._masks([(5.0, 1.0), (20.0, 0.0), (5.0, 1.0)],
                            [(20.0, 0.0), (5.0, 1.0), (5.0, 0.0)])
        self.assertEqual(len(mask_to_intervals(body_static_mask(together, FS))), 1)
        # The overlap of the two is only the final 5 s block on one side and nothing on the
        # other; whatever survives must be shorter than the naive per-sensor answer.
        self.assertLess(body_static_mask(apart, FS).sum(), body_static_mask(together, FS).sum())

    def test_margins_are_trimmed_from_each_end(self):
        """The reported interval is strictly inside the still stretch: the rolling window
        straddles motion at each edge, so the untrimmed edges are transitions, not noise."""
        masks = self._masks([(5.0, 1.0), (20.0, 0.0), (5.0, 1.0)],
                            [(5.0, 1.0), (20.0, 0.0), (5.0, 1.0)])
        (start, end), = mask_to_intervals(body_static_mask(masks, FS))
        self.assertAlmostEqual(start / FS, 5.0 + BODY_STATIC_MARGIN_S,
                               delta=STATIC_EDGE_TOLERANCE_S)
        self.assertAlmostEqual(end / FS, 25.0 - BODY_STATIC_MARGIN_S,
                               delta=STATIC_EDGE_TOLERANCE_S)

    def test_short_pauses_are_rejected(self):
        brief = BODY_STATIC_MIN_S / 3
        masks = self._masks([(5.0, 1.0), (brief, 0.0), (5.0, 1.0)],
                            [(5.0, 1.0), (brief, 0.0), (5.0, 1.0)])
        self.assertFalse(body_static_mask(masks, FS).any())

    def test_no_sensors(self):
        self.assertEqual(len(body_static_mask({}, FS)), 0)


class TestRegimeMasks(unittest.TestCase):
    def test_static_and_nonstatic_partition_the_trial(self):
        static = np.array([1, 1, 0, 0, 1], dtype=bool)
        masks = regime_masks(static, np.zeros(5, dtype=bool), 5)
        np.testing.assert_array_equal(masks['static'] | masks['nonstatic'], np.ones(5, dtype=bool))
        self.assertFalse((masks['static'] & masks['nonstatic']).any())
        self.assertTrue(masks['all'].all())

    def test_all_regimes_are_present_and_the_right_length(self):
        masks = regime_masks(np.ones(4, dtype=bool), np.ones(4, dtype=bool), 4)
        self.assertEqual(set(masks), set(REGIMES))
        self.assertTrue(all(len(m) == 4 for m in masks.values()))

    def test_a_short_mask_is_padded_false_rather_than_raising(self):
        """Traces in one trial can differ by a sample after resampling. Padding keeps a
        one-sample mismatch from failing the whole trial, and False is the safe direction:
        it excludes the sample rather than claiming an unmeasured one was still."""
        masks = regime_masks(np.ones(3, dtype=bool), np.ones(2, dtype=bool), 5)
        self.assertEqual(masks['static'].sum(), 0)   # too short, so treated as absent
        self.assertEqual(len(masks['body_static']), 5)


# ==============================================================================
# Posture labeling
# ==============================================================================

class TestLabelPostureIntervals(unittest.TestCase):
    def setUp(self):
        # 30s ambulation, 25s seated, a 1s stand-up transient, 5s standing upright, 30s
        # ambulation. Ambulation dominates, so the median height is the upright one, as in a
        # real trial. The transient matters: a quiet run is labeled from its MEAN height, so
        # sitting and standing are only told apart when the movement between them breaks the
        # run — which standing up from a chair does.
        self.plate = make_pelvis_plate([
            (30.0, 3.0, UPRIGHT_HEIGHT),
            (25.0, 0.0, SEATED_HEIGHT),
            (1.0, 2.0, SEATED_HEIGHT),
            (5.0, 0.0, UPRIGHT_HEIGHT),
            (30.0, 3.0, UPRIGHT_HEIGHT),
        ])
        self.labels = label_posture_intervals({'pelvis_imu': self.plate}, ALBORNO, FS)

    def test_finds_the_seated_bout_at_the_right_time(self):
        self.assertEqual(len(self.labels['sitting']), 1)
        start, end = self.labels['sitting'][0]
        self.assertAlmostEqual(start / FS, 30.0, delta=EDGE_TOLERANCE_S)
        self.assertAlmostEqual(end / FS, 55.0, delta=EDGE_TOLERANCE_S)

    def test_quiet_at_upright_height_is_standing_not_sitting(self):
        """The whole sitting/standing split is one height comparison, and getting it backwards
        would relabel every seated bout as standing while still producing intervals in plausible
        places."""
        self.assertEqual(len(self.labels['standing']), 1)
        start, end = self.labels['standing'][0]
        self.assertAlmostEqual(start / FS, 56.0, delta=EDGE_TOLERANCE_S)
        self.assertAlmostEqual(end / FS, 61.0, delta=EDGE_TOLERANCE_S)

    def test_labels_partition_the_trial(self):
        """Every sample lands in exactly one label. A gap or an overlap means a stretch of trial
        is either counted twice in the pooled distributions or dropped from them."""
        n = len(self.plate)
        covered = np.zeros(n, dtype=int)
        for intervals in self.labels.values():
            for start, end in intervals:
                covered[start:end] += 1
        np.testing.assert_array_equal(covered, np.ones(n, dtype=int))

    def test_short_quiet_bouts_are_not_labeled(self):
        plate = make_pelvis_plate([
            (20.0, 3.0, UPRIGHT_HEIGHT),
            (MIN_STANDING_S / 2, 0.0, UPRIGHT_HEIGHT),   # too brief to count as standing
            (20.0, 3.0, UPRIGHT_HEIGHT),
            (MIN_SITTING_S / 2, 0.0, SEATED_HEIGHT),     # too brief to count as sitting
            (20.0, 3.0, UPRIGHT_HEIGHT),
        ])
        labels = label_posture_intervals({'pelvis_imu': plate}, ALBORNO, FS)
        self.assertEqual(labels['sitting'], [])
        self.assertEqual(labels['standing'], [])

    def test_frames_without_mocap_are_unlabeled_not_ambulation(self):
        """A stretch with no mocap has no height and no world-frame acceleration, so it cannot
        be called sitting, standing OR ambulation. Folding it into ambulation is not harmless:
        the Al Borno walking trials open with a ~430 s standing pause before the cameras start,
        which the predecessor labelled as ambulation and shaded as such under every time series.
        """
        plate = make_pelvis_plate([(40.0, 3.0, UPRIGHT_HEIGHT)])
        n = len(plate)
        valid = np.ones(n, dtype=bool)
        valid[:int(15 * FS)] = False
        plate.world_trace.valid = valid

        labels = label_posture_intervals({'pelvis_imu': plate}, ALBORNO, FS)
        unlabeled = np.zeros(n, dtype=bool)
        for start, end in labels['unlabeled']:
            unlabeled[start:end] = True
        self.assertTrue(unlabeled[: int(14 * FS)].all())
        for start, end in labels['ambulation']:
            self.assertTrue(valid[start:end].all())

    def test_a_dataset_without_that_pelvis_sensor_returns_empty_labels(self):
        """IMoVE names its pelvis PELVIS_M. A spec mismatch must yield no labels rather than a
        KeyError halfway through a 262-trial run."""
        labels = label_posture_intervals({'pelvis_imu': self.plate}, IMOVE, FS)
        self.assertEqual(labels, {'sitting': [], 'standing': [], 'ambulation': []})


# ==============================================================================
# Noise floor
# ==============================================================================

class TestIntervalNoise(unittest.TestCase):
    """The estimator: a length-weighted mean of the per-axis std WITHIN each interval."""

    @staticmethod
    def _trace_with_known_stds(stds_and_lengths, offsets=None):
        """A trace whose gyro std inside each block is exactly `std`, built from a two-level
        square wave (the std of +-a alternating is exactly a), optionally shifted by a
        per-block constant offset."""
        total = sum(length for _, length in stds_and_lengths)
        gyro = np.zeros((total, 3))
        offsets = offsets or [0.0] * len(stds_and_lengths)
        cursor = 0
        for (std, length), offset in zip(stds_and_lengths, offsets):
            block = np.empty(length)
            block[0::2], block[1::2] = std, -std
            gyro[cursor:cursor + length, :] = block[:, None] + offset
            cursor += length
        return make_trace(gyro), total

    def test_noise_is_length_weighted_across_intervals(self):
        """A long quiet interval must count for more than a short one. An unweighted mean of the
        two per-interval stds would give 0.055 here instead of 0.019 — same order of magnitude,
        silently wrong, and biased toward whatever the shortest interval saw."""
        trace, total = self._trace_with_known_stds([(0.01, 900), (0.1, 100)])
        mask = np.ones(total, dtype=bool)
        mask[900 - 1] = False  # split the two blocks into separate intervals
        result = interval_noise(trace, mask)
        expected = (0.01 * 899 + 0.1 * 100) / 999
        self.assertAlmostEqual(result['gyro_noise_x'], expected, places=6)
        self.assertEqual(result['n_noise_samples'], 999)
        self.assertEqual(result['n_noise_intervals'], 2)

    def test_between_interval_offsets_do_not_enter(self):
        """THE estimator's whole point. A sensor sitting still reads a constant gravity and
        field vector whose value depends on the orientation it happens to be in, so two pauses
        in two different poses have genuinely different means. Pooling them first would measure
        the difference between those poses — metres per second squared — and call it noise."""
        trace, total = self._trace_with_known_stds([(0.01, 500), (0.01, 500)],
                                                   offsets=[0.0, 7.0])
        mask = np.ones(total, dtype=bool)
        mask[499] = False
        within = interval_noise(trace, mask)['gyro_noise_x']
        pooled = float(np.std(trace.gyro[mask, 0]))
        self.assertAlmostEqual(within, 0.01, places=6)
        self.assertGreater(pooled, 100 * within)

    def test_intervals_shorter_than_two_samples_contribute_nothing(self):
        trace, total = self._trace_with_known_stds([(0.01, 100)])
        mask = np.zeros(total, dtype=bool)
        mask[10] = True
        result = interval_noise(trace, mask)
        self.assertEqual(result['n_noise_samples'], 0)
        self.assertTrue(np.isnan(result['gyro_noise_x']))

    def test_an_empty_mask_gives_the_full_nan_schema(self):
        """Every regime writes the same columns whether or not a noise floor is meaningful over
        it, so the per-trial table stays rectangular."""
        trace, total = self._trace_with_known_stds([(0.01, 100)])
        result = interval_noise(trace, np.zeros(total, dtype=bool))
        for modality in ('gyro', 'acc', 'mag'):
            for axis in 'xyz':
                self.assertTrue(np.isnan(result[f'{modality}_noise_{axis}']))
        self.assertEqual(result['n_noise_samples'], 0)


class TestMocapMotionCheck(unittest.TestCase):
    def test_a_stationary_plate_reads_zero_speed(self):
        plate = make_identity_plate('calcn_r_imu', still_gyro(500))
        result = mocap_motion_check(plate, np.ones(500, dtype=bool))
        self.assertAlmostEqual(result['mocap_speed_median_mm_s'], 0.0, places=6)
        self.assertAlmostEqual(result['mocap_angular_speed_median_deg_s'], 0.0, places=6)

    def test_a_translating_plate_reports_its_speed(self):
        """The check that catches the gyro detector's one blind spot: pure translation, which no
        gyroscope can see. If this returned zero for a moving plate it would validate exactly the
        cases it exists to catch."""
        n = 500
        positions = np.zeros((n, 3))
        positions[:, 0] = np.arange(n) * DT * 0.5    # 0.5 m/s along world x
        plate = make_identity_plate('calcn_r_imu', still_gyro(n), positions=positions)
        result = mocap_motion_check(plate, np.ones(n, dtype=bool))
        self.assertAlmostEqual(result['mocap_speed_median_mm_s'], 500.0, places=3)

    def test_invalid_frames_are_excluded(self):
        """It is the mocap being consulted, so a padded pose must not contribute. A regime
        entirely outside the mocap window reports NaN rather than a number derived from a
        constant fake pose."""
        n = 400
        plate = make_identity_plate('calcn_r_imu', still_gyro(n),
                                    valid=np.zeros(n, dtype=bool))
        result = mocap_motion_check(plate, np.ones(n, dtype=bool))
        self.assertEqual(result['n_mocap_check'], 0)
        self.assertTrue(np.isnan(result['mocap_speed_median_mm_s']))


# ==============================================================================
# Per-sensor scalars
# ==============================================================================

class TestSensorStats(unittest.TestCase):
    def _stats(self, plate, spec=ALBORNO, static=None, body=None):
        n = len(plate)
        static_by_sensor = {plate.name: np.zeros(n, dtype=bool) if static is None else static}
        body = np.zeros(n, dtype=bool) if body is None else body
        return sensor_stats({plate.name: plate}, spec, WORLD_FIELD, static_by_sensor, body, FS)

    def test_one_row_per_regime_present(self):
        n = 1000
        static = np.zeros(n, dtype=bool)
        static[200:600] = True
        stats = self._stats(make_identity_plate('torso_imu', still_gyro(n)), static=static)
        self.assertEqual(sorted(stats['regime']), sorted(['all', 'static', 'nonstatic']))
        self.assertEqual(int(stats.loc[stats['regime'] == 'static', 'n_samples'].iloc[0]), 400)
        self.assertAlmostEqual(stats.loc[stats['regime'] == 'static', 'coverage'].iloc[0], 0.4)

    def test_acceleration_departure_columns_separate_gravity_from_motion(self):
        """A plate reading gravity alone must score zero non-gravity acceleration, and one with
        a known linear acceleration must score exactly that. This is the pair of columns behind
        the report's acc_std comparison, so a sign or frame error here would quietly rescale it."""
        row = self._stats(make_plate(name='torso_imu')).iloc[0]
        self.assertAlmostEqual(row['linacc_rms'], 0.0, places=5)
        self.assertAlmostEqual(row['acc_norm_std'], 0.0, places=5)
        self.assertAlmostEqual(row['acc_norm_median'], np.linalg.norm(EXPECTED_GRAVITY), places=5)

        world_acc = np.array([0.0, 0.0, 3.0])
        row = self._stats(make_plate(name='torso_imu', world_acc=world_acc)).iloc[0]
        self.assertAlmostEqual(row['linacc_rms'], np.linalg.norm(world_acc), places=5)
        self.assertAlmostEqual(row['linacc_median'], np.linalg.norm(world_acc), places=5)

    def test_acc_norm_dev_understates_perpendicular_acceleration(self):
        """The documented reason both columns are reported: |acc| adds in quadrature, so motion
        perpendicular to gravity barely moves it while linacc sees the whole thing. If these two
        ever agreed, one of them would be computed wrong — and the reference-free column is the
        only one available on the trials whose static time falls outside the mocap window."""
        row = self._stats(make_plate(name='torso_imu',
                                     world_acc=np.array([3.0, 0.0, 0.0]))).iloc[0]
        self.assertAlmostEqual(row['linacc_rms'], 3.0, places=5)
        # |[3, 9.81, 0]| - 9.81 = 0.448 at every sample: real, but 6.7x smaller than the truth.
        self.assertLess(row['acc_norm_dev_rms'], row['linacc_rms'] / 5)
        self.assertGreater(row['acc_norm_dev_rms'], 0.0)

    def test_mocap_referenced_columns_are_nan_without_valid_frames(self):
        """Outside `valid` the world trace holds a constant padded pose, and rotating a real
        reading by it puts the vector somewhere it never was. The predecessor computed linacc
        and magdev there anyway and folded the result into every distribution."""
        n = 500
        plate = make_identity_plate('torso_imu', still_gyro(n), valid=np.zeros(n, dtype=bool))
        row = self._stats(plate).iloc[0]
        self.assertTrue(np.isnan(row['linacc_rms']))
        self.assertTrue(np.isnan(row['magdev_rms']))
        self.assertEqual(row['n_valid'], 0)
        # The reference-free columns need no rotation and must still be populated — they are
        # what carries the static/moving comparison on those trials.
        self.assertFalse(np.isnan(row['acc_norm_dev_rms']))
        self.assertFalse(np.isnan(row['mag_norm_std']))
        self.assertGreater(row['n_samples'], 0)

    def test_noise_columns_are_only_filled_for_the_static_regimes(self):
        """Measuring the "noise" of a moving sensor would report its motion. The columns exist
        in every row so the table stays rectangular, but carry NaN where they are meaningless."""
        n = 1000
        static = np.zeros(n, dtype=bool)
        static[100:900] = True
        rng = np.random.default_rng(0)
        gyro = rng.normal(0.0, 0.002, size=(n, 3))
        stats = self._stats(make_identity_plate('torso_imu', gyro), static=static)
        by_regime = stats.set_index('regime')
        self.assertGreater(by_regime.loc['static', 'n_noise_samples'], 0)
        self.assertFalse(np.isnan(by_regime.loc['static', 'gyro_noise_x']))
        self.assertEqual(by_regime.loc['all', 'n_noise_samples'], 0)
        self.assertTrue(np.isnan(by_regime.loc['all', 'gyro_noise_x']))
        self.assertTrue(np.isnan(by_regime.loc['nonstatic', 'gyro_noise_x']))

    def test_magdev_is_measured_against_the_supplied_global_field(self):
        n = 300
        offset = np.array([0.0, 0.3, 0.0])
        plate = make_identity_plate('torso_imu', still_gyro(n),
                                    mag=np.tile(WORLD_FIELD + offset, (n, 1)))
        row = self._stats(plate).iloc[0]
        self.assertAlmostEqual(row['magdev_median'], np.linalg.norm(offset), places=5)


# ==============================================================================
# Subject field
# ==============================================================================

class TestSubjectField(unittest.TestCase):
    def test_trial_field_uses_valid_frames_only(self):
        """A padded pose rotates a real reading into a direction it was never in, so the field
        estimate must not see it."""
        n = 400
        mag = np.tile(WORLD_FIELD, (n, 1))
        mag[:200] = np.array([0.0, 0.0, 50.0])      # nonsense, only on the invalid half
        valid = np.zeros(n, dtype=bool)
        valid[200:] = True
        plate = make_identity_plate('torso_imu', still_gyro(n), mag=mag, valid=valid)
        row = trial_field_table({'torso_imu': plate}, ALBORNO).iloc[0]
        self.assertAlmostEqual(row['world_mag_x'], WORLD_FIELD[0], places=5)
        self.assertAlmostEqual(row['world_mag_z'], WORLD_FIELD[2], places=5)
        self.assertEqual(row['n_valid'], 200)

    def test_each_trial_and_sensor_gets_one_vote(self):
        """Two levels of median, so a session's longest recording cannot outvote its others.
        Pooling samples instead would let an IMoVE 2800 s walk set the field for the 30 s static
        pose beside it."""
        trial_fields = pd.DataFrame({
            'trial': ['long', 'short', 'short'],
            'sensor': ['a', 'a', 'b'],
            'world_mag_x': [10.0, 1.0, 1.0],
            'world_mag_y': [0.0, 0.0, 0.0],
            'world_mag_z': [0.0, 0.0, 0.0],
        })
        field = subject_field_from_trials(trial_fields)
        self.assertAlmostEqual(field[0], 1.0)   # the two short-trial votes win, 2 to 1

    def test_pooled_world_field_keeps_the_older_pooled_semantics(self):
        """relative_vs_absolute's `subject_median` reference arm is built on this definition and
        its published numbers are measured against it, so it must stay a pooled-sample median."""
        short = make_identity_plate('a', still_gyro(10), mag=np.tile([1.0, 0, 0], (10, 1)))
        long = make_identity_plate('b', still_gyro(990), mag=np.tile([9.0, 0, 0], (990, 1)))
        field = pooled_world_field({'walking': {'a': short, 'b': long}})
        self.assertAlmostEqual(field[0], 9.0)   # sample-weighted, so the long plate wins


# ==============================================================================
# Observability
# ==============================================================================

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

    def test_a_cutoff_at_or_above_nyquist_passes_the_signal_through(self):
        """IMoVE's 40 Hz sessions have a 20 Hz Nyquist against a 25 Hz cutoff. `butter` raises on
        a normalized cutoff above 1, so without this guard every 40 Hz trial fails outright."""
        rng = np.random.default_rng(1)
        signal = np.abs(rng.normal(100.0, 20.0, size=2000))
        np.testing.assert_allclose(smooth_observability(signal, 40.0, cutoff=25.0), signal)

    def test_a_trace_too_short_to_filter_is_returned_clamped(self):
        short = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(smooth_observability(short, FS), [1.0, 0.0, 3.0])


class TestJointStats(unittest.TestCase):
    def test_observability_collapses_when_a_joint_stops_moving(self):
        """The finding the gating argument rests on: a joint that is not moving has almost no
        observability, which is correct — a stationary accelerometer carries no information
        about orientation beyond gravity."""
        from test.TestExperimentPhysics import make_shaking_plate
        parent = make_shaking_plate('femur_r_imu', n=600)
        child = make_shaking_plate('tibia_r_imu', n=600)
        n = len(parent)
        static = np.zeros(n, dtype=bool)

        spec = ALBORNO
        moving = joint_stats({'femur_r_imu': parent, 'tibia_r_imu': child}, spec, WORLD_FIELD,
                             {'femur_r_imu': static, 'tibia_r_imu': static},
                             np.zeros(n, dtype=bool), FS)
        row = moving[(moving['joint'] == 'R_Knee') & (moving['regime'] == 'all')].iloc[0]
        self.assertGreater(row['obs_min_median'], 0.0)

        still_parent = make_identity_plate('femur_r_imu', still_gyro(n))
        still_child = make_identity_plate('tibia_r_imu', still_gyro(n))
        stats = joint_stats({'femur_r_imu': still_parent, 'tibia_r_imu': still_child}, spec,
                            WORLD_FIELD, {'femur_r_imu': static, 'tibia_r_imu': static},
                            np.zeros(n, dtype=bool), FS)
        still_row = stats[(stats['joint'] == 'R_Knee') & (stats['regime'] == 'all')].iloc[0]
        self.assertAlmostEqual(still_row['obs_min_median'], 0.0, places=6)
        self.assertAlmostEqual(still_row['frac_below_gate'], 1.0)

    def test_a_joint_is_static_only_when_both_its_sensors_are(self):
        """The same minimum argument as o^J itself, applied to the regime: a knee with a planted
        shank and a swinging thigh is not a static knee."""
        n = 400
        parent = make_identity_plate('femur_r_imu', still_gyro(n))
        child = make_identity_plate('tibia_r_imu', still_gyro(n))
        parent_static = np.zeros(n, dtype=bool)
        parent_static[:300] = True
        child_static = np.zeros(n, dtype=bool)
        child_static[200:] = True

        stats = joint_stats({'femur_r_imu': parent, 'tibia_r_imu': child}, ALBORNO, WORLD_FIELD,
                            {'femur_r_imu': parent_static, 'tibia_r_imu': child_static},
                            np.zeros(n, dtype=bool), FS)
        row = stats[(stats['joint'] == 'R_Knee') & (stats['regime'] == 'static')].iloc[0]
        self.assertEqual(int(row['n_samples']), 100)   # the 200-300 overlap only


# ==============================================================================
# Field consistency
# ==============================================================================

class TestFieldConsistency(unittest.TestCase):
    """The child-vs-parent field comparison, and the reference direction it uses.

    The direction matters because var_reduction is asymmetric: which sensor references which
    decides the sign whenever the two differ in variance. The dataset's `field_reference` is
    supposed to be a fixed anatomical rule, so these tests pin that it is applied mechanically
    and that both directions stay recorded — a rule that quietly became "pick whichever came out
    positive" would still produce plausible numbers.
    """

    @staticmethod
    def _pair(parent_gain=2.1, shared_sd=1.0, noise_sd=0.1, n=4000, seed=0):
        """A lumbar pair sharing one fluctuating field, with the PARENT seeing it amplified by
        `parent_gain`, plus a little independent noise on each.

        The gain is what makes this a faithful fixture rather than a convenient one. The real
        lumbar pair has corr 0.86 alongside Var(parent)/Var(child) = 4.55, which a
        shared-signal-plus-independent-noise model cannot produce at all (it caps corr at
        1/sqrt(k) = 0.47). What fits is a common field variation that the pelvis sees at larger
        amplitude, being the sensor nearer whatever is distorting it.
        """
        rng = np.random.default_rng(seed)
        shared = rng.normal(0.0, shared_sd, size=(n, 3))
        fields = {}
        for name, gain in (('pelvis_imu', parent_gain), ('torso_imu', 1.0)):
            fields[name] = WORLD_FIELD + gain * shared + rng.normal(0.0, noise_sd, size=(n, 3))
        return fields, np.ones(n, dtype=bool)

    def _row(self, spec=ALBORNO, joint='Lumbar', **kwargs):
        fields, mask = self._pair(**kwargs)
        return field_consistency(joint, spec, 'pelvis_imu', 'torso_imu',
                                 fields['pelvis_imu'], fields['torso_imu'], WORLD_FIELD, mask)

    def test_lumbar_reference_is_the_torso(self):
        """The one declared exception: the lumbar references the torso, not the kinematic
        parent, because the torso is the cleaner sensor of that pair."""
        self.assertEqual(ALBORNO.field_reference.get('Lumbar'), 'child')
        row = self._row()
        self.assertEqual(row['reference_sensor'], 'torso_imu')
        self.assertEqual(row['target_sensor'], 'pelvis_imu')

    def test_every_other_joint_references_its_kinematic_parent(self):
        self.assertEqual([j for j in ALBORNO.joints
                          if ALBORNO.field_reference.get(j, 'parent') != 'parent'], ['Lumbar'])
        self.assertEqual(IMOVE.field_reference, {})

    def test_headline_value_follows_the_declared_direction(self):
        """var_reduction must be the configured direction's value, and BOTH directions must be
        stored so the choice can be undone from the saved table."""
        row = self._row()
        self.assertAlmostEqual(row['var_reduction'], row['var_reduction_child_ref'], places=9)
        self.assertNotAlmostEqual(row['var_reduction'], row['var_reduction_parent_ref'], places=3)

    def test_the_flip_is_what_rescues_a_noisy_parent(self):
        """The substantive claim behind the reference rule: with a noisy parent and a clean
        child, referencing the parent loses and referencing the child wins — off identical
        samples."""
        row = self._row()
        self.assertLess(row['var_reduction_parent_ref'], 0.0)
        self.assertGreater(row['var_reduction_child_ref'], 0.5)

    def test_correlation_is_direction_independent(self):
        """The evidence for the premise — that the two sensors' fields move together — is
        symmetric, unlike var_reduction. This is why it is reported alongside."""
        fields, mask = self._pair()
        forward = field_consistency('Lumbar', ALBORNO, 'pelvis_imu', 'torso_imu',
                                    fields['pelvis_imu'], fields['torso_imu'], WORLD_FIELD, mask)
        flipped = field_consistency('Lumbar', ALBORNO, 'pelvis_imu', 'torso_imu',
                                    fields['torso_imu'], fields['pelvis_imu'], WORLD_FIELD, mask)
        self.assertAlmostEqual(forward['corr_target_reference'],
                               flipped['corr_target_reference'], places=9)

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
        self.assertEqual(self._row()['var_reduction_global'], 0.0)

    def test_best_possible_reduction_is_rho_squared_and_bounds_the_actual(self):
        """rho^2 is the ceiling: forcing unit gain can never beat the optimal gain, and equals it
        only when the two sensors have matched variance."""
        row = self._row()
        self.assertAlmostEqual(row['var_reduction_best'], row['corr_target_reference'] ** 2,
                               places=12)
        self.assertGreaterEqual(row['var_reduction_best'], row['var_reduction'])
        self.assertGreaterEqual(row['var_reduction_best'], row['var_reduction_parent_ref'])

    def test_var_reduction_matches_its_closed_form(self):
        """var_reduction == 2*rho*sqrt(k) - k for k = Var(reference)/Var(target). The identity is
        the whole basis for reading the sign, so it is checked against the implementation. The
        two only agree because both use the population (ddof=0) convention; np.cov's ddof=1
        default breaks the identity in the fourth decimal."""
        rng = np.random.default_rng(3)
        target = rng.normal(0, 1.0, size=(5000, 3))
        reference = 0.6 * target + rng.normal(0, 0.4, size=(5000, 3))
        k = np.var(reference, axis=0).sum() / np.var(target, axis=0).sum()
        rho = _pair_correlation(target, reference)
        self.assertAlmostEqual(_var_reduction(target, reference), 2 * rho * np.sqrt(k) - k,
                               places=9)

    def test_the_global_arm_is_the_subject_field_not_a_refit_per_trial_median(self):
        """`cos_sim_target_global` is the column the report labels "vs global", and it has to be
        measured against the SUBJECT's field — the same constant section 4's magdev uses.

        An earlier version silently used the target's own median over the evaluation window,
        which is a different and strictly stronger baseline: refit per trial, per sensor and per
        regime, in-sample. That flattered the global arm by absorbing exactly the between-trial
        drift the global field cannot follow. Both are reported now, and this pins which is
        which: a subject field pointing somewhere the target's field does not can only lower the
        subject column, never the trial-median one."""
        # A quiet pair, so the target's own median IS very nearly its every sample and the two
        # baselines are separable by construction. The default fixture's shared_sd swamps the
        # mean field, which is right for the variance-ratio tests and useless for a direction one.
        quiet = dict(shared_sd=0.02, noise_sd=0.005)
        fields, mask = self._pair(**quiet)
        elsewhere = np.array([0.0, 1.0, 0.0])   # orthogonal to WORLD_FIELD
        row = field_consistency('Lumbar', ALBORNO, 'pelvis_imu', 'torso_imu',
                                fields['pelvis_imu'], fields['torso_imu'], elsewhere, mask)
        self.assertLess(row['cos_sim_target_global'], 0.1)
        self.assertGreater(row['cos_sim_target_trial_median'], 0.99)
        # And with the true field it recovers, so the column tracks the argument it is given.
        self.assertGreater(self._row(**quiet)['cos_sim_target_global'], 0.99)

    def test_var_reduction_is_untouched_by_which_constant_the_global_arm_uses(self):
        """The scale's zero is definitional — Var(target - c) = Var(target) for any constant c —
        so fixing the cos-sim column must not have moved any variance-reduction number."""
        fields, mask = self._pair()
        rows = [field_consistency('Lumbar', ALBORNO, 'pelvis_imu', 'torso_imu',
                                  fields['pelvis_imu'], fields['torso_imu'], field, mask)
                for field in (WORLD_FIELD, np.array([0.0, 1.0, 0.0]), np.array([9.0, -3.0, 2.0]))]
        for column in ('var_reduction', 'var_reduction_best', 'var_reduction_parent_ref',
                       'var_reduction_child_ref', 'corr_target_reference'):
            self.assertAlmostEqual(rows[0][column], rows[1][column], places=12, msg=column)
            self.assertAlmostEqual(rows[0][column], rows[2][column], places=12, msg=column)

    def test_too_few_samples_yields_no_columns_rather_than_nonsense(self):
        fields, _ = self._pair(n=2)
        self.assertEqual(field_consistency('Lumbar', ALBORNO, 'pelvis_imu', 'torso_imu',
                                           fields['pelvis_imu'], fields['torso_imu'],
                                           WORLD_FIELD, np.ones(2, dtype=bool)), {})


# ==============================================================================
# Dataset specs
# ==============================================================================

class TestDatasetSpecs(unittest.TestCase):
    def test_every_joint_names_sensors_the_spec_declares(self):
        """A joint referring to a sensor no segment maps to would silently produce no rows for
        that joint on every trial, which looks exactly like "that joint has no data"."""
        for name, spec in DATASETS.items():
            sensors = set(spec.segment_sensor.values())
            for joint, (parent, child) in spec.joints.items():
                self.assertIn(parent, sensors, f"{name}/{joint} parent")
                self.assertIn(child, sensors, f"{name}/{joint} child")
            self.assertIn(spec.pelvis_sensor, sensors, name)
            for foot in spec.foot_sensors:
                self.assertIn(foot, sensors, name)

    def test_field_reference_only_names_real_joints(self):
        for name, spec in DATASETS.items():
            self.assertTrue(set(spec.field_reference).issubset(set(spec.joints)), name)

    def test_alborno_joints_match_the_pipeline_definitions(self):
        """The Al Borno spec must stay the same joint table the rest of the pipeline uses;
        a divergence would make this experiment's o^J describe different pairs than the
        filter's."""
        self.assertEqual(ALBORNO.joints, dict(JOINTS))

    def test_segment_joint_roles_stays_derived_from_the_spec(self):
        """The table is derived rather than written out; this pins that it stays derived. Every
        joint must appear exactly twice — once as a parent, once as a child."""
        for name, spec in DATASETS.items():
            roles = segment_joint_roles(spec)
            pairs = [(joint, role) for entries in roles.values() for joint, role in entries]
            for joint in spec.joints:
                self.assertEqual(sorted(role for j, role in pairs if j == joint),
                                 ['child', 'parent'], f"{name}/{joint}")
            self.assertEqual(set(roles), set(spec.segment_sensor), name)

    def test_pelvis_borders_three_joints_in_alborno(self):
        self.assertEqual(sorted(joint for joint, _ in segment_joint_roles(ALBORNO)['Pelvis']),
                         ['L_Hip', 'Lumbar', 'R_Hip'])

    def test_every_segment_borders_at_least_one_joint(self):
        """IMoVE's High and Low placements sit on the SAME segment as the Mid one, against one
        marker cluster, so they border the same joints it does. An earlier version listed the Mid
        sensors only, which left eight of fifteen segments with no observability at all — and a
        segment with no o^J is invisible to every per-segment observability figure."""
        for name, spec in DATASETS.items():
            roles = segment_joint_roles(spec)
            bare = sorted(segment for segment, entries in roles.items() if not entries)
            self.assertEqual(bare, [], f"{name}: segments bordering no joint")

    def test_placement_variants_are_matched_and_named_after_their_joint(self):
        """H pairs with H and L with L, and each variant's canonical joint is recoverable from
        its name — the report and the figures both group by that."""
        variants = [j for j in IMOVE.joints if j not in IMOVE.primary_joints]
        self.assertEqual(len(variants), 12)          # 6 joints x {High, Low}
        for joint in variants:
            self.assertIn(canonical_joint(joint, IMOVE), IMOVE.primary_joints)
            self.assertIn(placement_of(joint, IMOVE), ('High', 'Low'))
            parent, child = IMOVE.joints[joint]
            placements = {s.rsplit('_', 1)[-1] for s in (parent, child)}
            # Either both sides carry this placement, or the side that cannot (the pelvis and the
            # feet have one sensor each) falls back to its Mid.
            self.assertTrue(placements <= {joint.rsplit('_', 1)[-1], 'M'}, joint)

    def test_a_placement_variant_never_duplicates_its_canonical_pair(self):
        """A variant naming the same two sensors as the Mid pair would be the same measurement
        under a second name, silently double-weighting it in anything that pools joints."""
        for spec in DATASETS.values():
            pairs = list(spec.joints.values())
            self.assertEqual(len(pairs), len(set(pairs)))

    def test_primary_joints_are_real_joints_and_all_mid(self):
        for name, spec in DATASETS.items():
            self.assertTrue(set(spec.primary_joints) <= set(spec.joints), name)
            for joint in spec.primary_joints:
                self.assertEqual(placement_of(joint, spec), 'Mid', f"{name}/{joint}")

    def test_enumeration_drops_what_the_source_no_longer_builds(self):
        """A parquet left behind by a superseded build is not analysable — `build_trials` will
        never refresh it, so it fails the staleness check on every run with no command that can
        clear it. `enumerate_trials` must exclude exactly the orphans and nothing else."""
        for name in DATASETS:
            built, usable = set(built_trials(name)), set(enumerate_trials(name))
            orphans = set(orphaned_trials(name))
            self.assertEqual(usable, built - orphans, name)
            self.assertTrue(orphans <= built, name)

    def test_orphans_are_exactly_the_unsyncable_trials(self):
        """Pins WHY the IMoVE orphans exist, so this stops silently absorbing a real build
        regression: every one of them must be a trial the source deliberately excludes, not one
        that simply failed to build."""
        orphans = orphaned_trials('imove')
        if orphans:   # empty on a checkout that never built them
            self.assertTrue(all(trial in UNSYNCABLE_TRIALS for _, trial in orphans), orphans)

    def test_every_spec_reads_from_a_real_build_tree(self):
        """`trials_dataset` has to name a dataset the build layer actually registers, or the
        spec enumerates zero trials and reports the dataset as unbuilt."""
        for name, spec in DATASETS.items():
            self.assertIn(spec.build_name, SOURCES, name)

    def test_specs_sharing_a_build_tree_write_to_different_directories(self):
        """The two biplane references analyse ONE build. If they shared an output directory the
        second run would overwrite the first, which is the failure mode `name` vs
        `trials_dataset` exists to prevent."""
        by_build = {}
        for spec in DATASETS.values():
            by_build.setdefault(spec.build_name, []).append(spec)
        for build, specs in by_build.items():
            directories = {dataset_dir(spec.name) for spec in specs}
            self.assertEqual(len(directories), len(specs), build)


# ==============================================================================
# magdev's two reference arms
# ==============================================================================

class TestMagdevReferenceArms(unittest.TestCase):
    """`magdev` is scored against the SUBJECT's field and `magdev_trial` against the TRIAL's own.

    The pair exists because a subject's per-trial fields disagree by ~10 deg of heading at the
    median in both datasets, which is more than the cleanest sensors are then reported to deviate
    from the subject constant. These pin that the two arms differ ONLY in the constant, that the
    trial arm is genuinely the smaller of the two, and that the drift between the references is
    recorded rather than left to be inferred from the gap.
    """

    def setUp(self):
        self.spec = ALBORNO
        n = 300
        # Two plates reading a field offset from WORLD_FIELD by a known rotation, so the trial's
        # own field is NOT the subject field and the two arms must separate.
        self.trial_field = np.array([np.cos(np.radians(20.0)), np.sin(np.radians(20.0)), 0.0])
        self.plates, self.static = {}, {}
        for sensor in ('torso_imu', 'pelvis_imu'):
            self.plates[sensor] = make_identity_plate(
                sensor, np.zeros((n, 3)), mag=np.tile(self.trial_field, (n, 1)))
            self.static[sensor] = np.ones(n, dtype=bool)
        self.body = np.ones(n, dtype=bool)

    def _samples(self, **references):
        return segment_samples(self.plates, self.spec, WORLD_FIELD, self.static, self.body,
                               references=references or None)

    def test_every_arm_is_written(self):
        df = self._samples(trial=self.trial_field)
        for dev, angle, _ in MAGDEV_ARMS:
            self.assertIn(dev, df.columns)
            self.assertIn(angle, df.columns)

    def test_the_arms_differ_only_in_the_constant_subtracted(self):
        """Identity rotations, so the world frame is the body frame and every expected value is
        readable off the input: the subject arm must see the full 20 deg offset and the trial
        arm, scored against the field the sensors actually read, must see none of it."""
        df = self._samples(trial=self.trial_field)
        self.assertAlmostEqual(df['magdev_angle'].median(), 20.0, places=3)
        self.assertAlmostEqual(df['magdev_angle_trial'].median(), 0.0, places=6)

    def test_the_trial_field_is_recovered_from_the_plates(self):
        """`trial_field_vector` is what `compute_trial` passes in, so it has to reproduce the
        field the sensors are actually reading rather than needing to be supplied."""
        np.testing.assert_allclose(trial_field_vector(self.plates, self.spec),
                                   self.trial_field, atol=1e-9)

    def test_an_unsupplied_arm_collapses_onto_the_subject_arm(self):
        """The degenerate default: a caller with no alternative reference gets equal arms rather
        than a silently absent column or a NaN one. A missing arm is a configuration gap, and
        identical numbers are the readable way to say so."""
        df = self._samples()
        for dev, angle, _ in MAGDEV_ARMS:
            np.testing.assert_allclose(df['magdev'].to_numpy(), df[dev].to_numpy())
            np.testing.assert_allclose(df['magdev_angle'].to_numpy(), df[angle].to_numpy())

    def test_sensor_stats_carries_the_arms_and_the_drift_between_them(self):
        stats = sensor_stats(self.plates, self.spec, WORLD_FIELD, self.static, self.body, FS,
                             {'trial': self.trial_field})
        row = stats[stats.regime == 'all'].iloc[0]
        self.assertAlmostEqual(row['magdev_angle_median'], 20.0, places=3)
        self.assertAlmostEqual(row['magdev_angle_trial_median'], 0.0, places=6)
        # The drift is the angle between an arm's constant and the subject one, not between arms.
        self.assertAlmostEqual(row['reference_drift_trial_deg'], 20.0, places=6)
        self.assertAlmostEqual(row['reference_drift_subject_deg'], 0.0, places=9)

    def test_the_sensor_world_field_is_emitted_once_not_per_arm(self):
        """`world_mag_*` is the sensor's own median field and does not depend on the reference,
        so a second arm must not produce a second, differently-named copy of it."""
        stats = sensor_stats(self.plates, self.spec, WORLD_FIELD, self.static, self.body, FS,
                             {'trial': self.trial_field})
        self.assertIn('world_mag_x', stats.columns)
        self.assertFalse([c for c in stats.columns if c.startswith('world_mag')
                          and c.endswith('_trial')])

    def _trial_fields(self, offsets):
        """A per-(trial, sensor) field table with a declared world-frame field per sensor."""
        rows = []
        for sensor, vector in offsets.items():
            for trial in ('t1', 't2'):
                rows.append({'sensor': sensor, 'segment': sensor, 'trial': trial,
                             'world_mag_x': vector[0], 'world_mag_y': vector[1],
                             'world_mag_z': vector[2]})
        return pd.DataFrame(rows)

    def test_leave_one_out_excludes_only_the_sensor_it_is_for(self):
        """The point of the arm: sensor i's reference must be built from every sensor EXCEPT i,
        so an outlying sensor cannot pull its own reference toward itself. Three sensors, one of
        them displaced, and its loo reference must be the other two's field exactly."""
        offsets = {'torso_imu': np.array([1.0, 0.0, 0.0]),
                   'pelvis_imu': np.array([1.0, 0.0, 0.0]),
                   'calcn_l_imu': np.array([0.0, 1.0, 0.0])}    # the outlier
        table = subject_reference_table('01', self._trial_fields(offsets), ALBORNO)
        loo = table[table.arm == 'loo'].set_index('sensor')
        columns = ['world_mag_x', 'world_mag_y', 'world_mag_z']
        # The outlier's own reference is the two clean sensors and carries none of its own field.
        np.testing.assert_allclose(loo.loc['calcn_l_imu', columns].to_numpy(dtype=float),
                                   [1.0, 0.0, 0.0], atol=1e-12)
        # A clean sensor's reference still contains the outlier, so it is NOT the clean field.
        self.assertGreater(np.linalg.norm(
            loo.loc['torso_imu', columns].to_numpy(dtype=float) - np.array([1.0, 0.0, 0.0])), 0.1)

    def test_leave_one_out_raises_a_self_referencing_sensor_never_lowers_it(self):
        """The direction is the whole claim: removing a sensor from its own reference can only
        move that reference AWAY from it, so the loo arm is >= the subject arm for every sensor.
        This is the bias the arm exists to expose, and its sign is not an empirical question."""
        offsets = {'torso_imu': np.array([1.0, 0.0, 0.0]),
                   'pelvis_imu': np.array([1.0, 0.1, 0.0]),
                   'calcn_l_imu': np.array([0.6, 0.8, 0.0])}
        table = subject_reference_table('01', self._trial_fields(offsets), ALBORNO)
        columns = ['world_mag_x', 'world_mag_y', 'world_mag_z']
        subject = table[table.arm == 'subject'][columns].to_numpy(dtype=float)[0]
        loo = table[table.arm == 'loo'].set_index('sensor')
        for sensor, own in offsets.items():
            here = loo.loc[sensor, columns].to_numpy(dtype=float)
            to_subject = angle_between_deg(own[None, :], subject[None, :])[0]
            to_loo = angle_between_deg(own[None, :], here[None, :])[0]
            self.assertGreaterEqual(to_loo + 1e-9, to_subject, sensor)

    def test_the_clean_arm_is_the_declared_sensor_not_the_measured_argmin(self):
        """Declared up front, so a subject whose cleanest sensor happens to be something else
        still references the spec's choice. Selecting per subject would be selecting on a
        statistic correlated with what is being measured."""
        offsets = {'torso_imu': np.array([1.0, 0.0, 0.0]),
                   'pelvis_imu': np.array([0.0, 1.0, 0.0])}
        table = subject_reference_table('01', self._trial_fields(offsets), ALBORNO)
        clean = table[table.arm == 'clean']
        self.assertEqual(clean['sensor'].iloc[0], ALBORNO.clean_sensor)
        np.testing.assert_allclose(
            clean[['world_mag_x', 'world_mag_y', 'world_mag_z']].to_numpy(dtype=float)[0],
            offsets['torso_imu'], atol=1e-12)

    def test_the_clean_arm_zeroes_its_own_reference_segment(self):
        """The reason it is a diagnostic and never a gradient. Scored against itself, the clean
        sensor's departure collapses to nothing, which is the top of any proximal-to-distal ratio
        being defined rather than measured."""
        clean_field = np.array([1.0, 0.0, 0.0])
        df = segment_samples(self.plates, self.spec, WORLD_FIELD, self.static, self.body,
                             references={'clean': self.trial_field})
        torso = df[df.sensor == 'torso_imu']
        self.assertAlmostEqual(torso['magdev_angle_clean'].median(), 0.0, places=6)
        self.assertAlmostEqual(torso['magdev_angle'].median(), 20.0, places=3)
        del clean_field

    def test_resolve_reference_falls_back_rather_than_inventing_a_number(self):
        fallback = np.array([9.0, 9.0, 9.0])
        np.testing.assert_array_equal(resolve_reference({}, 'loo', 'torso_imu', fallback),
                                      fallback)
        # A per-sensor arm with no entry for THIS sensor falls back too, not to another's.
        entry = {'loo': {'pelvis_imu': np.array([1.0, 0.0, 0.0])}}
        np.testing.assert_array_equal(resolve_reference(entry, 'loo', 'torso_imu', fallback),
                                      fallback)
        np.testing.assert_array_equal(resolve_reference(entry, 'loo', 'pelvis_imu', fallback),
                                      np.array([1.0, 0.0, 0.0]))

    def test_a_spec_without_a_clean_sensor_writes_no_clean_rows(self):
        offsets = {'lateral_thigh_left__biplane': np.array([1.0, 0.0, 0.0])}
        table = subject_reference_table('01', self._trial_fields(offsets),
                                        DATASETS['imove_biplane'])
        self.assertNotIn('clean', set(table['arm']))

    def test_the_arms_are_summarized_and_carry_units(self):
        for metric in ('magdev_trial', 'magdev_angle_trial', 'magdev_loo', 'magdev_angle_loo',
                       'magdev_clean', 'magdev_angle_clean'):
            self.assertIn(metric, SEGMENT_METRICS)
            self.assertIn(metric, METRIC_UNITS)
        # Both need a magnetometer, so a mag-free dataset drops them with the rest.
        self.assertNotIn('magdev_trial', DATASETS['imove_biplane'].segment_metrics())


# ==============================================================================
# A dataset with no magnetometer
# ==============================================================================

class TestMagnetometerFreeDataset(unittest.TestCase):
    """`imove_biplane`'s BioStamps have no magnetometer and the reader fills `mag` with exact
    zeros. Zeros are not a missing value: pooled into a subject field they give the zero vector,
    every magdev against it is exactly 0, and the report would then claim the MAG=CONSTANT
    assumption holds perfectly on a dataset that never measured a field. Every test here pins the
    columns being ABSENT rather than zero or NaN, because absent is the only one of the three a
    downstream slice cannot mistake for a measurement."""

    def setUp(self):
        self.spec = DATASETS['imove_biplane']
        n = 400
        self.plates, self.static = {}, {}
        for i, sensor in enumerate(self.spec.segment_sensor.values()):
            # `world_mag=0` is exactly what the biplane reader writes, which is the point: these
            # tests assert the analysis refuses to treat that zero as a measured field. Distinct
            # positions so the pair has a real lever arm to a joint center to project onto.
            positions = np.tile(np.array([0.1 * i, 0.0, 0.0]), (n, 1))
            self.plates[sensor] = make_plate(sensor, n=n, positions=positions,
                                             world_mag=np.zeros(3))
            self.static[sensor] = np.ones(n, dtype=bool)
        self.body = np.ones(n, dtype=bool)
        self.nan_field = np.full(3, np.nan)

    def test_the_spec_declares_itself_magnetometer_free(self):
        self.assertFalse(self.spec.has_magnetometer)
        self.assertTrue(DATASETS['imove_biplane_vicon'].has_magnetometer is False)
        for name in ('alborno', 'imove'):
            self.assertTrue(DATASETS[name].has_magnetometer, name)

    def test_stage_one_writes_nothing_rather_than_a_zero_field(self):
        """A trial_field table of zeros would roll up into a zero 'global field'."""
        self.assertTrue(trial_field_table(self.plates, self.spec).empty)

    def test_segment_samples_omits_every_magnetic_column(self):
        df = segment_samples(self.plates, self.spec, self.nan_field, self.static, self.body)
        self.assertFalse(df.empty)
        for column in ('magdev', 'magdev_angle', 'mag_norm', 'mag_norm_dev'):
            self.assertNotIn(column, df.columns)
        # The accelerometer half is untouched and still real.
        self.assertIn('linacc', df.columns)
        self.assertTrue(np.isfinite(df['acc_norm_dev']).all())

    def test_sensor_stats_omits_the_magnetic_scalars_and_the_mag_noise_floor(self):
        df = sensor_stats(self.plates, self.spec, self.nan_field, self.static, self.body, FS)
        self.assertFalse(df.empty)
        for column in ('magdev_rms', 'magdev_angle_median', 'mag_norm_std', 'mag_norm_dev_rms',
                       'world_mag_x', 'mag_noise_x'):
            self.assertNotIn(column, df.columns)
        for column in ('linacc_rms', 'acc_norm_dev_rms', 'gyro_noise_x', 'acc_noise_x'):
            self.assertIn(column, df.columns)

    def test_joint_stats_keeps_observability_and_drops_field_consistency(self):
        df = joint_stats(self.plates, self.spec, self.nan_field, self.static, self.body, FS)
        self.assertFalse(df.empty)
        for column in ('var_reduction', 'cos_sim_target_global', 'mag_residual_rms'):
            self.assertNotIn(column, df.columns)
        for column in ('obs_min_median', 'acc_residual_rms'):
            self.assertIn(column, df.columns)

    def test_the_summarized_metric_list_drops_the_magnetic_ones(self):
        self.assertEqual(self.spec.segment_metrics(), ('linacc', 'acc_norm_dev'))
        self.assertEqual(DATASETS['alborno'].segment_metrics(), SEGMENT_METRICS)
        self.assertEqual(self.spec.noise_modalities(), ('gyro', 'acc'))

    def test_an_unsolvable_joint_center_costs_the_joint_not_the_trial(self):
        """`project_pair_to_joint_center` refuses below 50 frames where BOTH plates are valid,
        which is correct at its own level and catastrophic if it propagates: it used to abort
        `compute_trial`, so 125 of imove_biplane's 379 trials wrote nothing at all — including
        the segment tables, which never touch a joint center. Here the ground-truth window is
        deliberately 10 frames, and the segment half must still come out whole."""
        n = len(next(iter(self.plates.values())))
        valid = np.zeros(n, dtype=bool)
        valid[:10] = True
        for sensor, plate in self.plates.items():
            world = plate.world_trace
            self.plates[sensor] = PlateTrial(
                sensor, plate.imu_trace,
                WorldTrace(world.timestamps, world.positions, world.rotations, valid=valid))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            tables = compute_trial(self.plates, self.spec, self.nan_field)
        self.assertTrue(any('joint center unsolvable' in str(w.message) for w in caught))

        self.assertFalse(tables['segment_samples'].empty)
        self.assertFalse(tables['sensor_stats'].empty)
        self.assertTrue(tables['joint_samples'].empty)
        self.assertTrue(tables['joint_stats'].empty)

    def test_the_whole_trial_runs_without_a_subject_field(self):
        """The end-to-end guard: `compute_trial` is what the worker calls, and on a mag-free
        dataset it is handed a NaN field because stage one never ran."""
        tables = compute_trial(self.plates, self.spec, self.nan_field)
        self.assertEqual(set(tables), set(RECOMPUTABLE_TABLES))
        for name, table in tables.items():
            self.assertFalse(any('mag' in c for c in table.columns), f"{name}: {list(table)}")


# ==============================================================================
# Summary
# ==============================================================================

class TestSummarize(unittest.TestCase):
    def setUp(self):
        # Two "trials" of the same segment with deliberately different lengths and levels, so a
        # mean-of-medians reduction cannot coincide with the pooled median. Half of each trial
        # is static, so the regime split has something to separate.
        frames = []
        for subject, trial, values in (('Subject01', 'walking', np.arange(100, dtype=float)),
                                       ('Subject02', 'walking', np.full(900, 1000.0))):
            static = np.zeros(len(values), dtype=bool)
            static[: len(values) // 2] = True
            frames.append(pd.DataFrame({
                'subject': subject, 'trial': trial, 'segment': 'Pelvis',
                'static': static, 'body_static': np.zeros(len(values), dtype=bool),
                'linacc': values, 'magdev': np.zeros(len(values)),
            }))
        self.segment_df = pd.concat(frames, ignore_index=True)
        self.summary = summarize('alborno', self.segment_df, pd.DataFrame())

    def _row(self, subject, trial, metric='linacc', regime='all'):
        match = self.summary[(self.summary['subject'] == subject)
                             & (self.summary['trial'] == trial)
                             & (self.summary['regime'] == regime)
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

    def test_all_three_splits_are_produced(self):
        """Per trial, per subject and pooled. The report and the figures each want a different
        one, and a missing margin looks like a missing trial."""
        keys = set(map(tuple, self.summary[['subject', 'trial']].drop_duplicates().to_numpy()))
        self.assertIn(('Subject01', 'walking'), keys)   # per trial
        self.assertIn(('Subject01', 'all'), keys)       # per subject
        self.assertIn(('all', 'all'), keys)             # pooled

    def test_regimes_are_a_dimension_and_they_overlap(self):
        """`all` is the union of static and nonstatic, so its count is their sum. They are
        selectors rather than a partition — body_static is a subset of static — so nothing here
        should assume they sum to the total in general."""
        static = self._row('Subject01', 'walking', regime='static')
        nonstatic = self._row('Subject01', 'walking', regime='nonstatic')
        self.assertEqual(static['n_samples'] + nonstatic['n_samples'],
                         self._row('Subject01', 'walking')['n_samples'])
        self.assertAlmostEqual(static['p50'], np.median(np.arange(50)))

    def test_a_regime_with_no_rows_is_absent_rather_than_zero(self):
        """body_static is empty in this fixture. An all-NaN row would read as "measured and
        found to be nothing"."""
        self.assertTrue(self.summary[self.summary['regime'] == 'body_static'].empty)

    def test_n_samples_counts_non_nan_values(self):
        """The mocap-referenced metrics are NaN outside the mocap window, so n_samples has to be
        the valid count and not the row count — otherwise every quantile is quoted against a
        denominator that includes samples it was not computed from."""
        frame = self.segment_df.copy()
        frame.loc[:49, 'linacc'] = np.nan
        summary = summarize('alborno', frame, pd.DataFrame())
        pooled = summary[(summary['subject'] == 'all') & (summary['regime'] == 'all')
                         & (summary['metric'] == 'linacc')].iloc[0]
        self.assertEqual(pooled['n_samples'], 950)

    def test_units_are_recorded(self):
        self.assertEqual(self._row('all', 'all', 'magdev')['unit'], 'a.u.')
        self.assertEqual(self._row('all', 'all', 'linacc')['unit'], 'm/s^2')

    def test_empty_input(self):
        self.assertTrue(summarize('alborno', pd.DataFrame(), pd.DataFrame()).empty)


class TestLoadTrialTable(unittest.TestCase):
    def test_unknown_table_raises(self):
        """A typo'd table name must not look like "the experiment has not been run"."""
        with self.assertRaises(ValueError):
            load_trial_table('alborno', 'segment_sample')
        self.assertIn('segment_samples', TRIAL_TABLES)

    def test_no_matching_trials_gives_an_empty_frame(self):
        self.assertTrue(load_trial_table('alborno', 'segment_samples',
                                         row_keys=[('nope', 'nope')]).empty)


if __name__ == '__main__':
    unittest.main()
