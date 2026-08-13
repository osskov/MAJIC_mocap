"""The cluster-to-IMU offset: PlateTrial.fit_sensor_offset and the constant it produces.

`imove_mocap.CLUSTER_TO_IMU_OFFSET_MM` is a measured constant, and unlike the device map
beside it, it depends on the PIPELINE that measured it -- replacing linear interpolation with
band-limited resampling moved it by 70-90%. A constant with that property needs a guard, or
it silently goes stale the next time the loader changes.

Two guards, and the first needs no constant at all:

  RATE AGREEMENT. The same physical sensor measured through two different rate-conversion
  paths must give the same offset. IMoVE's 17-sensor sessions run at 40 Hz against 100 Hz
  mocap, so they are resampled; the 7-sensor long walks run at 100 Hz against 100 Hz and are
  not touched. While the resampler was doing linear interpolation these disagreed by 12 mm.
  This is the sharpest statement available that the resampling is right, and it would have
  caught that bug on the day it was written.

  ZERO RESIDUAL. The loader now SHIFTS each world trace onto its IMU using the constant, so
  a built trial should have no offset left to find. That is a closed loop rather than a
  comparison against a stored copy of itself: fit_sensor_offset measures what REMAINS, so if
  the constant is wrong by delta the shift is wrong by delta and the fit returns delta.

Both read the parquet cache, so they are real-data tests and are skipped rather than failed
when it has not been built.
"""
import unittest

from test.fixtures import require_data

import numpy as np
from scipy.spatial.transform import Rotation

from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.PlateTrial import PlateTrial
from src.toolchest.WorldTrace import WorldTrace
from src.toolchest.building.imove_mocap import (CLUSTER_TO_IMU_OFFSET_MM,
                                                pool_cluster_offsets)

# Enough trials to average the per-trial scatter down, few enough to stay quick. Spread over
# sessions rather than taken consecutively, so one bad session cannot dominate.
SAMPLE_TRIALS = [('s2', 't6_drop_jump_001'), ('s3', 't1_walking_001'),
                 ('s7', 't7_cmjdl_001'), ('s17', 't6_drop_jump_001'),
                 ('s20', 't1_walking_001'), ('s25', 't7_cmjdl_001')]
LONG_WALK_TRIALS = [('s4l', 't12_longwalk_001'), ('s5l', 't12_longwalk_001'),
                    ('s6l', 't12_longwalk_001'), ('s13l', 't12_longwalk_001'),
                    ('s23l', 't12_longwalk_001')]

# Walking on BOTH sides of the rate comparison. The 100 Hz sessions only ever recorded a long
# walk, so pooling the 40 Hz side over all eleven activities would confound rate with
# activity -- and the offsets do vary by activity.
WALKING_TRIALS = [('s2', 't1_walking_001'), ('s3', 't1_walking_001'),
                  ('s7', 't1_walking_001'), ('s10', 't1_walking_001'),
                  ('s17', 't1_walking_001'), ('s20', 't1_walking_001'),
                  ('s25', 't1_walking_001')]

# The pooled MAD runs 2-4 mm per axis, so a median over several trials should land well
# inside this. Loose enough not to fire on ordinary noise, tight enough that the 12 mm
# rate disagreement the old resampler produced would trip it.
AGREEMENT_TOLERANCE_MM = 6.0

# The residual after a correct shift is small but NOT zero, and the floor is numerical rather
# than a mistake: mocap acceleration comes from double-differencing positions, which attenuates
# the lever-arm term by a few percent, so the shift under-corrects by that much. Measured on
# synthetic data it scales with motion speed and inversely with sample rate -- 2.6 mm on an
# aggressive 100 Hz case, 0.4 mm when the motion is slowed fourfold.
RESIDUAL_TOLERANCE_MM = 6.0


def _load(trials):
    """{sensor: [offset, ...]} over whatever trials are cached, or None if none are.

    Raises SkipTest on a stale cache rather than quietly returning nothing: a silent skip
    there would let these guards pass without ever running, which defeats the point of
    having them.
    """
    from experiments.experiment_utils import StaleTrialCache, load_trial

    samples = {}
    for session, trial in trials:
        try:
            plates = load_trial(session, trial, dataset='imove')
        except StaleTrialCache as stale:
            raise unittest.SkipTest(f"trial cache is stale: {stale}")
        except (FileNotFoundError, ValueError):
            continue
        for name, plate in plates.items():
            try:
                samples.setdefault(name, []).append(plate.fit_sensor_offset())
            except (ValueError, np.linalg.LinAlgError):
                continue
    return {name: np.array(values) for name, values in samples.items()} or None


class TestFitOnSyntheticData(unittest.TestCase):
    """Ground truth the real data cannot supply: a known offset, recovered."""

    OFFSET = np.array([0.021, -0.014, 0.033])

    def _plate(self, seconds=40.0, rate=100.0):
        timestamps = np.arange(int(seconds * rate)) / rate
        # Rich rotation, so the lever-arm matrix is well conditioned in every direction.
        rotation = Rotation.from_euler('zyx', np.column_stack([
            1.1 * np.sin(2 * np.pi * 1.3 * timestamps),
            0.9 * np.sin(2 * np.pi * 2.1 * timestamps),
            0.7 * np.sin(2 * np.pi * 3.7 * timestamps)]))
        positions = np.column_stack([0.3 * np.sin(2 * np.pi * 0.4 * timestamps),
                                     1.0 + 0.1 * np.sin(2 * np.pi * 0.6 * timestamps),
                                     0.2 * np.cos(2 * np.pi * 0.5 * timestamps)])
        world = WorldTrace(timestamps, positions, rotation.as_matrix())

        gravity = np.array([0.0, 9.81, 0.0])
        origin = world.calculate_imu_trace(acc_from_gravity=gravity)
        # The sensor sits at OFFSET on the same rigid body, so it reads the origin's specific
        # force plus the lever-arm term.
        sensor = origin.project_acc(self.OFFSET, finite_difference_gyro_method='polyfit')
        return PlateTrial('synthetic', sensor, world)

    def test_a_known_offset_comes_back(self):
        recovered = self._plate().fit_sensor_offset()
        error = np.linalg.norm(recovered - self.OFFSET) * 1000
        self.assertLess(error, 5.0, f"recovered {recovered * 1000} mm, "
                                    f"expected {self.OFFSET * 1000}")

    def test_gravity_is_recovered_rather_than_assumed(self):
        """The fit solves for the gravity contribution alongside the offset, so it needs no
        gravity input. Recovering the right one is the check that the joint solve is set up
        correctly -- a sign or frame error there would land in the offset instead."""
        plate = self._plate()
        plate.fit_sensor_offset()

        np.testing.assert_allclose(plate._fitted_gravity, [0.0, 9.81, 0.0], atol=1e-3)

    def test_a_sensor_at_the_origin_reads_zero(self):
        """The claim the plate figure made and the data refuted, pinned as the null case."""
        plate = self._plate()
        at_origin = PlateTrial('synthetic', plate.world_trace.calculate_imu_trace(
            acc_from_gravity=np.array([0.0, 9.81, 0.0])), plate.world_trace)

        self.assertLess(np.linalg.norm(at_origin.fit_sensor_offset()) * 1000, 5.0)


class TestShiftingTheWorldOrigin(unittest.TestCase):
    """`shift_world_origin` moves the mocap origin onto the IMU, closing the loop."""

    def setUp(self):
        self.plate = TestFitOnSyntheticData()._plate()
        self.true = TestFitOnSyntheticData.OFFSET

    def test_shifting_by_the_true_offset_leaves_almost_nothing(self):
        from src.toolchest.building.assembly import shift_world_origin

        remaining = shift_world_origin(self.plate, self.true).fit_sensor_offset()

        self.assertLess(np.linalg.norm(remaining) * 1000, RESIDUAL_TOLERANCE_MM)

    def test_a_wrong_shift_shows_up_as_exactly_its_error(self):
        """What makes the closed loop a real check rather than a tautology: the fit measures
        what REMAINS, so a constant wrong by delta leaves delta behind."""
        from src.toolchest.building.assembly import shift_world_origin

        correct = shift_world_origin(self.plate, self.true).fit_sensor_offset()
        wrong = shift_world_origin(
            self.plate, self.true + np.array([0.005, 0.0, 0.0])).fit_sensor_offset()

        np.testing.assert_allclose((wrong - correct)[0] * 1000, -5.0, atol=0.5)
        # The other axes must not move: the error is in x alone.
        np.testing.assert_allclose((wrong - correct)[1:] * 1000, 0.0, atol=0.5)

    def test_fitting_does_not_modify_the_plate(self):
        """fit_sensor_offset filters in place, and the gravity block starts as a VIEW onto
        the world trace's rotations. Without a copy it corrupts the plate it measures -- and
        since a segment's three sensors share one WorldTrace, fitting H would poison M and L.
        """
        before = self.plate.world_trace.rotations.copy()
        positions = self.plate.world_trace.positions.copy()

        self.plate.fit_sensor_offset()

        np.testing.assert_array_equal(self.plate.world_trace.rotations, before)
        np.testing.assert_array_equal(self.plate.world_trace.positions, positions)

    def test_fitting_twice_gives_the_same_answer(self):
        """The direct symptom of the above, and the cheaper thing to notice."""
        first = self.plate.fit_sensor_offset()
        np.testing.assert_allclose(self.plate.fit_sensor_offset(), first, atol=1e-12)

    def test_rotations_are_untouched(self):
        """Translation only, so joint angles cannot move and no orientation result changes."""
        from src.toolchest.building.assembly import shift_world_origin

        shifted = shift_world_origin(self.plate, self.true)

        np.testing.assert_array_equal(shifted.world_trace.rotations,
                                      self.plate.world_trace.rotations)
        np.testing.assert_array_equal(shifted.imu_trace.gyro, self.plate.imu_trace.gyro)

    def test_a_zero_offset_is_a_no_op(self):
        """Sensors with no measured constant -- the feet, and all of Al Borno -- must pass
        through untouched rather than being nudged by rounding."""
        from src.toolchest.building.assembly import shift_world_origin

        self.assertIs(shift_world_origin(self.plate, np.zeros(3)), self.plate)


class TestPooling(unittest.TestCase):
    def test_a_diverging_fit_does_not_move_the_median(self):
        """One IMoVE trial returned a magnitude sd of 205 mm. A mean would follow it."""
        good = np.tile(np.array([0.010, -0.020, 0.030]), (9, 1))
        samples = {'THIGH_L_M': np.vstack([good, np.array([[5.0, -3.0, 8.0]])])}

        pooled = pool_cluster_offsets(samples)['THIGH_L_M']

        np.testing.assert_allclose(pooled['median_mm'], [10.0, -20.0, 30.0], atol=1e-9)
        self.assertEqual(pooled['n'], 10)

    def test_spread_is_reported_per_axis(self):
        rng = np.random.default_rng(0)
        samples = {'S': np.array([0.01, 0.02, 0.03]) + rng.normal(0, 0.002, (200, 3))}
        pooled = pool_cluster_offsets(samples)['S']
        np.testing.assert_allclose(pooled['spread_mm'], 2.0, atol=0.6)


class TestAgainstCachedTrials(unittest.TestCase):
    """Reads the parquet cache. Skipped, not failed, when it has not been built."""

    @classmethod
    def setUpClass(cls):
        cls.sampled = _load(SAMPLE_TRIALS)
        if cls.sampled is None:
            require_data(False, "no IMoVE trials cached")

    def test_the_offset_is_not_zero(self):
        """The whole reason this constant exists. Pinning a lower bound rather than a value
        keeps this test meaningful even if the estimate is later refined."""
        pooled = pool_cluster_offsets(self.sampled)
        for sensor, entry in pooled.items():
            if sensor.startswith('FOOT'):
                continue          # no cluster; the markers are anatomical and deform
            self.assertGreater(np.linalg.norm(entry['median_mm']), 3.0,
                               f"{sensor} came back at the origin")

    def test_the_two_sample_rates_agree(self):
        """The sharpest check on the resampling that the data can provide.

        The 40 Hz sessions are resampled onto a common grid; the 100 Hz long walks are not
        touched at all. Same hardware, so the same answer -- unless the rate conversion is
        introducing something, which is exactly what linear interpolation was doing when
        these disagreed by 12 mm.
        """
        long_walk = _load(LONG_WALK_TRIALS)
        if long_walk is None:
            require_data(False, "no long-walk trials cached")

        walking = _load(WALKING_TRIALS)
        if walking is None:
            self.skipTest('no walking trials cached')
        slow = pool_cluster_offsets(walking)
        fast = pool_cluster_offsets(long_walk)

        # The MEAN gap, not each sensor's. A rate-conversion bias is systematic -- the old
        # linear-interpolation resampler pushed every sensor the same way, -12 mm across the
        # board -- whereas per-sensor scatter cancels. That distinction is what makes this
        # assertable at all: the 100 Hz side is 5 sessions with a per-fit MAD of 4-9 mm, so
        # any single sensor's median carries several mm of its own uncertainty and the gaps
        # legitimately run -13 to +11 with mixed signs.
        gaps = [np.linalg.norm(slow[sensor]['median_mm'])
                - np.linalg.norm(fast[sensor]['median_mm'])
                for sensor in sorted(set(slow) & set(fast))
                if not sensor.startswith('FOOT')]
        self.assertGreater(len(gaps), 2, "too few sensors comparable across both rates")

        mean_gap = float(np.mean(gaps))
        self.assertLess(abs(mean_gap), AGREEMENT_TOLERANCE_MM,
                        f"40 Hz and 100 Hz sessions disagree by {mean_gap:+.1f} mm on average "
                        f"({[round(g, 1) for g in gaps]}). A consistent offset points at the "
                        f"rate conversion rather than at the hardware.")

    def test_a_built_trial_has_no_offset_left_to_find(self):
        """The closed loop. The loader shifted each world trace onto its IMU using the
        constant, so re-measuring should find nothing.

        This is what the resampler bug slipped past: the constant was derived through one
        pipeline and went on being applied after the pipeline changed under it. Here that
        shows up directly, because a constant wrong by delta leaves exactly delta behind.
        """
        if not CLUSTER_TO_IMU_OFFSET_MM:
            self.skipTest("CLUSTER_TO_IMU_OFFSET_MM not populated yet; "
                          "run experiments/refit_cluster_offset.py")

        pooled = pool_cluster_offsets(self.sampled)
        for sensor, entry in sorted(pooled.items()):
            if sensor.startswith('FOOT') or sensor not in CLUSTER_TO_IMU_OFFSET_MM:
                continue
            remaining = np.linalg.norm(entry['median_mm'])
            self.assertLess(remaining, RESIDUAL_TOLERANCE_MM,
                            f"{sensor}: {entry['median_mm'].round(1)} mm still unaccounted "
                            f"for after the shift — the stored constant has drifted from "
                            f"what this pipeline measures. Re-run "
                            f"experiments/refit_cluster_offset.py")


if __name__ == '__main__':
    unittest.main()
