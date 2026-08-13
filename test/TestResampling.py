"""Rate conversion and time alignment: toolchest.resampling, and assembly's use of it.

Collected here from TestIMUTrace, TestWorldTrace and TestPlateTrial, because these are all
one subject and the guarantees only make sense together. What is being protected:

  * resampling is BAND-LIMITED. It used to be linear interpolation on the IMU only, which
    is sinc^2 -- it drooped the passband, fabricated images above the old Nyquist, and left
    the mocap in a different band than the IMU it was being differenced against. That put
    7-9 mm of bias into the IMoVE cluster-to-IMU offset.
  * downsampling ANTI-ALIASES, so it reduces noise instead of folding it back in.
  * the output grid is arbitrary, so alignment can be FRACTIONAL. At 40 Hz, whole-sample
    sync leaves up to 12.5 ms of error.
  * rotations go through quaternions. Matrices are worse, absolute axis-angle wraps, and
    filtering incremental rotations drifts without bound.
  * a trial runs at the SLOWEST rate any of its streams was recorded at.
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.toolchest.IMUTrace import IMUTrace
from src.toolchest.WorldTrace import WorldTrace
from src.toolchest.building import assembly
from src.toolchest.finite_difference_utils import (DEFAULT_WINDOW_SECONDS,
                                                   polynomial_fit_derivative,
                                                   window_samples)
from src.toolchest.resampling import contamination_width, resample_rotations


def tone(rate, duration, frequencies):
    """(timestamps, (N, 3)) with the given tones on every channel."""
    timestamps = np.arange(int(duration * rate)) / rate
    signal = sum(np.sin(2 * np.pi * f * timestamps) for f in frequencies)
    return timestamps, np.column_stack([signal, 0.5 * signal, -signal])


def motion(t):
    """A rotation trajectory with a few distinct frequencies, none near any Nyquist here."""
    return Rotation.from_euler('zyx', np.column_stack([
        0.9 * np.sin(2 * np.pi * 0.7 * t),
        0.6 * np.sin(2 * np.pi * 1.1 * t),
        0.4 * np.sin(2 * np.pi * 1.7 * t)]))


def geodesic_error_deg(estimate, truth):
    relative = np.matmul(estimate.transpose(0, 2, 1), truth)
    trace = np.trace(relative, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))


class TestIMUTraceResampling(unittest.TestCase):
    def test_same_frequency_is_a_no_op(self):
        timestamps = np.arange(5, dtype=float)
        data = np.arange(15, dtype=float).reshape(5, 3)
        trace = IMUTrace(timestamps, data, data.copy(), data.copy())

        resampled = trace.resample(1.0)

        np.testing.assert_array_equal(resampled.timestamps, timestamps)
        np.testing.assert_array_equal(resampled.gyro, data)

    def test_a_ramp_survives_resampling_without_gain_error(self):
        """Replaces the old exact-linear-interpolation assertions.

        A ramp is the sharpest test of passband gain, because a gain error of eps shows up
        as an error growing linearly along it rather than as a constant offset. The
        polyphase FIR's default Kaiser window has 0.07% ripple, which this caught.
        """
        timestamps = np.linspace(0, 5, 501)
        ramp = np.column_stack([timestamps, 2 * timestamps, -timestamps])
        trace = IMUTrace(timestamps, ramp, ramp.copy(), ramp.copy())

        for new_rate in (50.0, 200.0):
            resampled = trace.resample(new_rate)
            expected = np.column_stack([resampled.timestamps, 2 * resampled.timestamps,
                                        -resampled.timestamps])
            np.testing.assert_allclose(resampled.gyro, expected, atol=1e-3)
            self.assertAlmostEqual(resampled.get_sample_frequency(), new_rate, places=6)

    def test_in_band_content_keeps_its_amplitude(self):
        """Linear interpolation lost 12.5% at 8 Hz going 40 -> 100. This must not."""
        timestamps, data = tone(40.0, 100.0, [8.0])
        resampled = IMUTrace(timestamps, data, data.copy(), data.copy()).resample(100.0)

        interior = resampled.gyro[500:-500, 0]
        self.assertAlmostEqual(np.abs(interior).max(), 1.0, places=2)

    def test_downsampling_rejects_content_above_the_new_nyquist(self):
        """A 30 Hz tone would land on 10 Hz if it were subsampled without a filter."""
        timestamps, data = tone(100.0, 100.0, [5.0, 30.0])
        resampled = IMUTrace(timestamps, data, data.copy(), data.copy()).resample(40.0)

        interior = resampled.gyro[500:-500, 0]
        spectrum = np.abs(np.fft.rfft(interior))
        frequencies = np.fft.rfftfreq(len(interior), 1 / 40.0)
        kept = spectrum[(frequencies > 4) & (frequencies < 6)].max()
        aliased = spectrum[(frequencies > 9) & (frequencies < 11)].max()

        self.assertLess(aliased / kept, 0.01)

    def test_downsampling_reduces_broadband_noise(self):
        """Decimation is not just discarding samples: the FIR averages over its support,
        so white noise drops by the bandwidth ratio rather than staying put."""
        rng = np.random.default_rng(0)
        timestamps = np.arange(100000) / 100.0
        noise = rng.normal(0, 1.0, (len(timestamps), 3))

        resampled = IMUTrace(timestamps, noise, noise.copy(), noise.copy()).resample(40.0)

        interior = resampled.gyro[500:-500, 0]
        self.assertLess(interior.std(), 0.75)
        self.assertGreater(interior.std(), 0.45)

    def test_a_fractional_time_shift_is_accurate(self):
        """Whole-sample alignment leaves 12.5 ms of error at 40 Hz, so the grid has to
        be shiftable by less than a sample."""
        timestamps, data = tone(100.0, 40.0, [3.0])
        trace = IMUTrace(timestamps, data, data.copy(), data.copy())

        shifted = trace.resample(100.0, time_shift=0.0033)

        expected = np.sin(2 * np.pi * 3.0 * shifted.timestamps)
        np.testing.assert_allclose(shifted.gyro[200:-200, 0], expected[200:-200], atol=1e-3)

    def test_values_outside_the_source_span_are_held_not_extrapolated(self):
        """A cubic run past the end of its data diverges as t^3.

        This is not hypothetical: asked for the 430 s of inertial record that precede the
        mocap in Subject01's walking trial, the spline returned marker positions of 1.3e10 m.
        Those frames are marked invalid, so no statistic was wrong -- but they live in the
        arrays filters integrate through, and the derivative of 1e10 is not inert. Held
        values reproduce the old index-clipping: the nearest real pose, motionless.
        """
        timestamps, data = tone(100.0, 10.0, [2.0])
        trace = IMUTrace(timestamps, data, data.copy(), data.copy())

        # Ask for ten times the record's own length.
        far_past_the_end = np.arange(-1000, 2000) / 10.0
        from src.toolchest.resampling import resample_values
        values = resample_values(data, timestamps, far_past_the_end,
                                 source_rate=100.0, target_rate=10.0)

        self.assertTrue(np.isfinite(values).all())
        self.assertLessEqual(np.abs(values).max(), 2 * np.abs(data).max())

        before = far_past_the_end < timestamps[0]
        after = far_past_the_end > timestamps[-1]
        np.testing.assert_allclose(np.diff(values[before], axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.diff(values[after], axis=0), 0.0, atol=1e-12)

    def test_a_trace_too_short_to_filter_still_resamples(self):
        """An order-8 Butterworth needs 27 samples. Short traces are legitimate -- test
        fixtures, and single segments carved out of a trial -- so the order drops to fit
        rather than the call failing."""
        for length in (4, 8, 15, 30):
            timestamps = np.linspace(0, 5, length)
            data = np.column_stack([timestamps, timestamps, timestamps])
            trace = IMUTrace(timestamps, data, data.copy(), data.copy())

            resampled = trace.resample(0.5)
            self.assertGreater(len(resampled), 0, f"failed at length {length}")


class TestWorldTraceResampling(unittest.TestCase):
    def test_pose_is_preserved_at_shared_timestamps(self):
        timestamps = np.linspace(0, 1, 101)
        positions = np.column_stack([timestamps, 2 * timestamps, np.zeros_like(timestamps)])
        rotations = Rotation.from_rotvec(
            timestamps[:, None] * np.array([0., 0., 1.])).as_matrix()

        resampled = WorldTrace(timestamps, positions, rotations).resample(50.0)

        self.assertAlmostEqual(resampled.get_sample_frequency(), 50.0, places=6)
        expected_positions = np.column_stack([resampled.timestamps, 2 * resampled.timestamps,
                                              np.zeros_like(resampled.timestamps)])
        expected_rotations = Rotation.from_rotvec(
            resampled.timestamps[:, None] * np.array([0., 0., 1.])).as_matrix()
        np.testing.assert_allclose(resampled.positions, expected_positions, atol=1e-3)
        np.testing.assert_allclose(resampled.rotations, expected_rotations, atol=1e-3)

    def test_resampled_rotations_are_still_rotations(self):
        timestamps = np.arange(2000) / 100.0
        trace = WorldTrace(timestamps, np.zeros((len(timestamps), 3)),
                           motion(timestamps).as_matrix())

        resampled = trace.resample(40.0)

        products = np.matmul(resampled.rotations.transpose(0, 2, 1), resampled.rotations)
        np.testing.assert_allclose(products, np.broadcast_to(np.eye(3), products.shape),
                                   atol=1e-9)
        np.testing.assert_allclose(np.linalg.det(resampled.rotations), 1.0, atol=1e-9)

    def test_yaw_past_pi_does_not_wrap(self):
        """Absolute axis-angle jumps the long way round past +/-pi and the filter smears
        the discontinuity across its whole support -- 57 deg of error, measured. World-frame
        yaw does exactly this over a long walk with turns, so the representation must not
        care where the trajectory happens to be on the circle."""
        timestamps = np.arange(6000) / 100.0
        turning = Rotation.from_euler('zyx', np.column_stack([
            2 * np.pi * 0.4 * timestamps,
            0.4 * np.sin(2 * np.pi * 0.5 * timestamps),
            0.3 * np.sin(2 * np.pi * 0.9 * timestamps)]))
        trace = WorldTrace(timestamps, np.zeros((len(timestamps), 3)), turning.as_matrix())

        resampled = trace.resample(40.0)

        truth = Rotation.from_euler('zyx', np.column_stack([
            2 * np.pi * 0.4 * resampled.timestamps,
            0.4 * np.sin(2 * np.pi * 0.5 * resampled.timestamps),
            0.3 * np.sin(2 * np.pi * 0.9 * resampled.timestamps)])).as_matrix()
        error = geodesic_error_deg(resampled.rotations[80:-80], truth[80:-80])
        self.assertLess(error.max(), 0.05)

    def test_quaternion_path_beats_filtering_matrix_entries(self):
        """A rotation of theta moves its quaternion by theta/2, so the chord that linear
        filtering cuts is half as long. Guards the choice of chart, not just the accuracy."""
        from src.toolchest.resampling import orthonormalize, resample_values

        timestamps = np.arange(6000) / 100.0
        fast = Rotation.from_euler('zyx', np.column_stack([
            1.5 * np.sin(2 * np.pi * 3.0 * timestamps),
            1.2 * np.sin(2 * np.pi * 4.0 * timestamps),
            0.9 * np.sin(2 * np.pi * 5.0 * timestamps)])).as_matrix()
        new_timestamps = np.arange(2400) / 40.0
        truth = Rotation.from_euler('zyx', np.column_stack([
            1.5 * np.sin(2 * np.pi * 3.0 * new_timestamps),
            1.2 * np.sin(2 * np.pi * 4.0 * new_timestamps),
            0.9 * np.sin(2 * np.pi * 5.0 * new_timestamps)])).as_matrix()

        interior = slice(80, -80)
        quaternion_error = geodesic_error_deg(
            resample_rotations(fast, timestamps, new_timestamps,
                               source_rate=100.0, target_rate=40.0)[interior], truth[interior])
        matrix_error = geodesic_error_deg(
            orthonormalize(resample_values(fast, timestamps, new_timestamps,
                                           source_rate=100.0,
                                           target_rate=40.0))[interior], truth[interior])

        self.assertLess(np.sqrt((quaternion_error ** 2).mean()),
                        np.sqrt((matrix_error ** 2).mean()))

    def test_the_validity_mask_widens_by_the_filter_reach(self):
        """Filtering across a dropout corrupts the samples around it, and the mask is the
        only thing that can say so. The invalid run must therefore GROW by the filter's
        support -- never shrink, and never drift away from the original damage."""
        timestamps = np.arange(200) / 100.0
        valid = np.ones(200, dtype=bool)
        valid[100:110] = False
        trace = WorldTrace(timestamps, np.zeros((200, 3)),
                           motion(timestamps).as_matrix(), valid=valid)

        resampled = trace.resample(50.0)

        self.assertEqual(len(resampled.valid), len(resampled))
        self.assertFalse(resampled.valid.all(), "the invalid run must survive downsampling")

        reach = contamination_width(100.0, 50.0) / 100.0
        invalid_times = resampled.timestamps[~resampled.valid]
        self.assertLessEqual(invalid_times.min(), timestamps[100])
        self.assertGreaterEqual(invalid_times.max(), timestamps[109])
        self.assertGreaterEqual(invalid_times.min(), timestamps[100] - reach - 1e-9)
        self.assertLessEqual(invalid_times.max(), timestamps[109] + reach + 1e-9)


class TestDerivativeWindow(unittest.TestCase):
    """The polyfit window is specified in SECONDS.

    It used to be a fixed 10 samples, which silently meant 100 ms at 100 Hz and 250 ms at
    40 Hz. Harmless while everything was resampled to 100 Hz; a per-trial difference
    masquerading as a per-subject one now that trials keep their native rate.
    """

    def test_the_default_window_is_a_fixed_duration(self):
        for rate, expected in ((40.0, 4), (100.0, 10), (200.0, 20)):
            timestamps = np.arange(10 * rate) / rate
            self.assertEqual(window_samples(timestamps, DEFAULT_WINDOW_SECONDS, 2), expected)

    def test_the_window_never_drops_below_a_smoothing_fit(self):
        """order+1 samples interpolates and fewer is underdetermined, so a short window at
        a low rate has to be floored or the fit stops smoothing at all."""
        timestamps = np.arange(100) / 20.0
        self.assertGreaterEqual(window_samples(timestamps, 0.01, 5), 7)

    def test_the_100hz_default_is_unchanged(self):
        """Every existing result was computed with the old fixed 10-sample window."""
        timestamps = np.arange(1000) / 100.0
        signal = np.sin(2 * np.pi * 2 * timestamps)

        np.testing.assert_array_equal(
            polynomial_fit_derivative(signal, timestamps, order=2, window_size=10),
            polynomial_fit_derivative(signal, timestamps, order=2))

    def test_accuracy_no_longer_depends_on_the_sample_rate(self):
        errors = []
        for rate in (40.0, 100.0, 200.0):
            timestamps = np.arange(int(10 * rate)) / rate
            signal = np.sin(2 * np.pi * 2 * timestamps)
            derivative = polynomial_fit_derivative(signal, timestamps, order=2)
            truth = 2 * np.pi * 2 * np.cos(2 * np.pi * 2 * timestamps)
            interior = slice(int(rate), -int(rate))
            errors.append(np.sqrt(((derivative[interior] - truth[interior]) ** 2).mean()))

        self.assertLess(max(errors) / min(errors), 1.3)


class TestTrialAlignment(unittest.TestCase):
    """Lag estimation and the common-grid policy, from TestPlateTrialSync."""

    TRUE_LAG = 5.0      # imu clock = true time + 5 ; world clock = true time

    def _streams(self, imu_rate, world_rate):
        world_time = np.arange(0, 50, 1 / world_rate)
        world = WorldTrace(world_time, np.zeros((len(world_time), 3)),
                           motion(world_time).as_matrix())
        # The IMU starts 5 s before the mocap and runs 5 s past it.
        imu_time = np.arange(-5, 55, 1 / imu_rate)
        synthetic = WorldTrace(imu_time, np.zeros((len(imu_time), 3)),
                               motion(imu_time).as_matrix()).calculate_imu_trace(
                                   skip_lin_acc=True)
        imu = IMUTrace(imu_time + self.TRUE_LAG, synthetic.gyro, synthetic.acc, synthetic.mag)
        return imu, world

    def test_lag_is_recovered_at_matched_rates(self):
        imu, world = self._streams(100.0, 100.0)
        self.assertAlmostEqual(assembly._lag_seconds(imu, world), self.TRUE_LAG, places=3)

    def test_lag_is_recovered_when_the_rates_differ(self):
        for imu_rate, world_rate in ((40.0, 100.0), (100.0, 40.0)):
            imu, world = self._streams(imu_rate, world_rate)
            self.assertAlmostEqual(assembly._lag_seconds(imu, world), self.TRUE_LAG,
                                   places=1, msg=f"{imu_rate}/{world_rate}")

    def test_the_trial_runs_at_the_slowest_stream(self):
        for imu_rate, world_rate in ((40.0, 100.0), (100.0, 40.0), (100.0, 100.0)):
            imu, world = self._streams(imu_rate, world_rate)
            target = min(imu_rate, world_rate)

            aligned_imu, aligned_world = assembly._to_common_grid(
                imu, world, target, self.TRUE_LAG)

            self.assertAlmostEqual(aligned_imu.get_sample_frequency(), target, places=6)
            self.assertAlmostEqual(aligned_world.get_sample_frequency(), target, places=6)
            np.testing.assert_allclose(aligned_imu.timestamps, aligned_world.timestamps,
                                       atol=1e-9)

    def test_t_zero_is_the_first_overlap_and_inertial_data_before_it_is_kept(self):
        imu, world = self._streams(100.0, 100.0)

        aligned_imu, aligned_world = assembly._to_common_grid(
            imu, world, 100.0, self.TRUE_LAG)

        # The IMU began 5 s before the mocap, so that record survives at negative time.
        self.assertAlmostEqual(aligned_imu.timestamps[0], -self.TRUE_LAG, places=2)
        zero_index = int(np.searchsorted(aligned_imu.timestamps, 0.0))
        self.assertTrue(aligned_world.valid[zero_index])
        self.assertFalse(aligned_world.valid[:zero_index].any(),
                         "nothing before t=0 has mocap coverage")

    def test_alignment_makes_the_two_gyro_signals_agree(self):
        imu, world = self._streams(100.0, 100.0)

        aligned_imu, aligned_world = assembly._to_common_grid(
            imu, world, 100.0, assembly._lag_seconds(imu, world))

        synthetic = aligned_world.calculate_imu_trace(skip_lin_acc=True)
        valid = aligned_world.valid
        residual = np.degrees(np.linalg.norm(
            synthetic.gyro[valid] - aligned_imu.gyro[valid], axis=1))
        self.assertLess(np.sqrt((residual ** 2).mean()), 1.0)

    def test_a_fractional_lag_is_recovered(self):
        """The whole point of estimating below the sample period. A lag of 2.5 samples at
        100 Hz is invisible to an integer search."""
        world_time = np.arange(0, 50, 0.01)
        world = WorldTrace(world_time, np.zeros((len(world_time), 3)),
                           motion(world_time).as_matrix())
        offset = 0.025
        imu_time = np.arange(0, 50, 0.01)
        synthetic = WorldTrace(imu_time, np.zeros((len(imu_time), 3)),
                               motion(imu_time - offset).as_matrix()).calculate_imu_trace(
                                   skip_lin_acc=True)
        imu = IMUTrace(imu_time, synthetic.gyro, synthetic.acc, synthetic.mag)

        self.assertAlmostEqual(assembly._lag_seconds(imu, world), offset, places=3)


class TestSyncSpreadGuard(unittest.TestCase):
    """One recording session has one lag, so scattered per-plate estimates mean the
    correlation found no peak -- and a median of noise is not an estimate."""

    @staticmethod
    def _paired(lags):
        """A plate per lag, each a rotating segment offset from the IMU by its own lag."""
        timestamps = np.arange(2000) / 100.0
        paired = {}
        for index, lag in enumerate(lags):
            world = WorldTrace(timestamps, np.zeros((len(timestamps), 3)),
                               motion(timestamps).as_matrix())
            synthetic = WorldTrace(timestamps, np.zeros((len(timestamps), 3)),
                                   motion(timestamps).as_matrix()).calculate_imu_trace(
                                       skip_lin_acc=True)
            imu = IMUTrace(timestamps + lag, synthetic.gyro, synthetic.acc, synthetic.mag)
            paired[f'plate{index}'] = (imu, world)
        return paired

    def test_agreeing_plates_give_their_shared_lag(self):
        lag = assembly._shared_lag(self._paired([2.0, 2.0, 2.0, 2.0]))
        self.assertAlmostEqual(lag, 2.0, places=2)

    def test_a_single_disagreeing_plate_does_not_veto_the_trial(self):
        """s24's t4_lat_step has fourteen plates at 1.9 s and one at -5.7. That is a bad
        plate, which the median exists to absorb -- a max-minus-min test would throw that
        protection away and fail the whole trial."""
        lag = assembly._shared_lag(self._paired([2.0, 2.0, 2.0, 2.0, -6.0]))
        self.assertAlmostEqual(lag, 2.0, places=2)

    def test_scattered_estimates_raise_rather_than_returning_a_median(self):
        with self.assertRaises(ValueError) as caught:
            assembly._shared_lag(self._paired([-6.0, -2.0, 1.0, 5.0, 9.0]))
        self.assertIn('scattered', str(caught.exception))


class TestSyncArrays(unittest.TestCase):
    """`_sync_arrays` is the integer-lag helper the scratch analyses still use."""

    def test_identical_arrays(self):
        first, second = assembly._sync_arrays(np.arange(1., 6.), np.arange(1., 6.))
        self.assertEqual((slice(0, 5), slice(0, 5)), (first, second))

    def test_offset_arrays(self):
        array1 = np.array([1., 2, 3, 4, 5, 6, 7, -4])
        array2 = np.array([0.1, 1, 2, 3, 4, 5, 6, 7])

        first, second = assembly._sync_arrays(array1, array2)
        self.assertEqual((slice(0, 7), slice(1, 8)), (first, second))

        # Swapping the arguments must swap the slices.
        first, second = assembly._sync_arrays(array2, array1)
        self.assertEqual((slice(1, 8), slice(0, 7)), (first, second))

    def test_no_overlap(self):
        first, second = assembly._sync_arrays(np.arange(1., 6.), np.arange(6., 11.))
        self.assertEqual((slice(0, 5), slice(0, 5)), (first, second))

    def test_a_long_signal_with_a_known_offset(self):
        length, start = 250, 100
        t = np.linspace(0, 100, length + start)
        base = np.sin(t) + np.sin(17 * t) - np.cos(3 * t) - np.cos(60 * t)

        first, second = assembly._sync_arrays(base, base[start:start + length])

        self.assertEqual((slice(start, start + length), slice(0, length)), (first, second))


if __name__ == '__main__':
    unittest.main()
