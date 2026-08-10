"""
Covers experiment_utils.compute_error_stats — the function that turns joint-angle
time series into every number the paper reports.

Three things here are easy to get wrong and impossible to notice downstream:

  * the error convention. r_imu * r_marker.inv() is a left/world-frame error; the
    other ordering (r_marker.inv() * r_imu, a body-frame error) is equally plausible
    and gives the same MAG but different X/Y/Z. Nothing checks it at runtime.
  * signed vs unsigned aggregation. A filter that oscillates symmetrically has a
    near-zero mean error and a large RMSE. Mixing the two up flatters or damns a
    method without changing anything about the plot's shape.
  * the melt-and-concat at the end, which reassembles ten separate group-by results
    into one long frame keyed by axis. Misalignment there would swap metrics between
    axes and still produce a full, plausible table.

All synthetic, all hand-checkable, no dataset needed.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from experiments.experiment_utils import compute_error_stats

GROUP_COLS = ['trial_type', 'method', 'joint_name', 'subject']
AXES = ['MAG', 'X', 'Y', 'Z']
METRICS = ['mean_rad', 'std_rad', 'rmse_rad', 'mae_rad', 'mad_rad',
           'min_rad', 'q25_rad', 'median_rad', 'q75_rad', 'max_rad']


def make_trial(error_rotvecs: np.ndarray, marker_rotvecs: np.ndarray = None,
               subject: str = 'Subject01', trial_type: str = 'walking',
               joint_name: str = 'R_Knee', method: str = 'mag_on',
               t0: float = 0.0) -> pd.DataFrame:
    """A marker trace and an IMU trace that differ by exactly `error_rotvecs`.

    The IMU orientation is built by LEFT-composing the error onto the marker
    orientation, matching compute_error_stats' r_imu * r_marker.inv(), so whatever
    error is asked for here is exactly what the function should recover.
    """
    n = len(error_rotvecs)
    marker_rotvecs = np.zeros((n, 3)) if marker_rotvecs is None else marker_rotvecs
    r_marker = Rotation.from_rotvec(marker_rotvecs)
    r_imu = Rotation.from_rotvec(error_rotvecs) * r_marker

    timestamps = t0 + np.arange(n) * 0.01
    frames = []
    for label, rotation in (('marker', r_marker), (method, r_imu)):
        rotvec = rotation.as_rotvec()
        frames.append(pd.DataFrame({
            'subject': subject, 'trial_type': trial_type, 'joint_name': joint_name,
            'timestamp': timestamps, 'method': label,
            'rx': rotvec[:, 0], 'ry': rotvec[:, 1], 'rz': rotvec[:, 2],
        }))
    return pd.concat(frames, ignore_index=True)


def cell(stats: pd.DataFrame, axis: str, metric: str, **filters) -> float:
    rows = stats[stats['axis'] == axis]
    for key, value in filters.items():
        rows = rows[rows[key] == value]
    assert len(rows) == 1, f"expected exactly one row, got {len(rows)}"
    return float(rows.iloc[0][metric])


class TestErrorConvention(unittest.TestCase):
    def test_a_constant_offset_is_recovered_on_the_right_axis(self):
        stats = compute_error_stats(make_trial(np.tile([0.0, 0.0, 0.1], (20, 1))))
        self.assertAlmostEqual(cell(stats, 'Z', 'mean_rad'), 0.1, places=12)
        self.assertAlmostEqual(cell(stats, 'X', 'mean_rad'), 0.0, places=12)
        self.assertAlmostEqual(cell(stats, 'Y', 'mean_rad'), 0.0, places=12)
        self.assertAlmostEqual(cell(stats, 'MAG', 'mean_rad'), 0.1, places=12)

    def test_the_error_is_left_composed_not_right_composed(self):
        """The discriminating case: a marker orientation that does not commute with the
        error. Under the code's r_imu * r_marker.inv() the error is [20 deg, 0, 0]
        exactly; under the body-frame ordering it would be that axis rotated by -30 deg
        about z, i.e. leakage into Y. MAG is identical either way, so only the
        per-axis numbers can tell the two apart."""
        n = 10
        error = np.tile(np.deg2rad([20.0, 0.0, 0.0]), (n, 1))
        marker = np.tile(np.deg2rad([0.0, 0.0, 30.0]), (n, 1))
        stats = compute_error_stats(make_trial(error, marker_rotvecs=marker))

        self.assertAlmostEqual(cell(stats, 'X', 'mean_rad'), np.deg2rad(20.0), places=12)
        self.assertAlmostEqual(cell(stats, 'Y', 'mean_rad'), 0.0, places=12)
        self.assertAlmostEqual(cell(stats, 'Z', 'mean_rad'), 0.0, places=12)

    def test_mag_is_the_geodesic_angle_between_the_orientations(self):
        rng = np.random.default_rng(0)
        error = rng.normal(scale=0.2, size=(50, 3))
        marker = rng.normal(scale=0.8, size=(50, 3))
        stats = compute_error_stats(make_trial(error, marker_rotvecs=marker))
        expected = np.linalg.norm(error, axis=1)
        self.assertAlmostEqual(cell(stats, 'MAG', 'mean_rad'), float(expected.mean()), places=10)
        self.assertAlmostEqual(cell(stats, 'MAG', 'max_rad'), float(expected.max()), places=10)

    def test_a_perfect_estimate_gives_zero_error(self):
        rng = np.random.default_rng(1)
        marker = rng.normal(scale=0.5, size=(30, 3))
        stats = compute_error_stats(make_trial(np.zeros((30, 3)), marker_rotvecs=marker))
        for axis in AXES:
            for metric in ('mean_rad', 'rmse_rad', 'mae_rad', 'max_rad'):
                with self.subTest(axis=axis, metric=metric):
                    self.assertAlmostEqual(cell(stats, axis, metric), 0.0, places=12)


class TestAggregations(unittest.TestCase):
    """Each metric against numpy on the same known error series, with a different
    distribution per axis so a misaligned melt/concat cannot pass."""

    def setUp(self):
        rng = np.random.default_rng(42)
        n = 400
        self.error = np.column_stack([
            rng.normal(loc=0.05, scale=0.010, size=n),   # X: small, biased
            rng.normal(loc=0.00, scale=0.050, size=n),   # Y: zero-mean, wide
            rng.normal(loc=-0.20, scale=0.002, size=n),  # Z: large, tight, negative
        ])
        self.stats = compute_error_stats(make_trial(self.error))
        self.truth = {
            'X': self.error[:, 0], 'Y': self.error[:, 1], 'Z': self.error[:, 2],
            'MAG': np.linalg.norm(self.error, axis=1),
        }

    def test_every_metric_matches_numpy_on_every_axis(self):
        for axis, values in self.truth.items():
            expected = {
                'mean_rad': values.mean(),
                'std_rad': values.std(ddof=1),   # pandas' default, not numpy's
                'rmse_rad': np.sqrt((values ** 2).mean()),
                'mae_rad': np.abs(values).mean(),
                'mad_rad': np.median(np.abs(values - np.median(values))),
                'min_rad': values.min(),
                'q25_rad': np.quantile(values, 0.25),
                'median_rad': np.median(values),
                'q75_rad': np.quantile(values, 0.75),
                'max_rad': values.max(),
            }
            for metric, want in expected.items():
                with self.subTest(axis=axis, metric=metric):
                    self.assertAlmostEqual(cell(self.stats, axis, metric), float(want), places=10)

    def test_no_metric_column_is_missing(self):
        for metric in METRICS:
            self.assertIn(metric, self.stats.columns)

    def test_one_row_per_axis(self):
        self.assertEqual(sorted(self.stats['axis'].tolist()), sorted(AXES))

    def test_mag_is_never_negative_even_when_the_axes_are(self):
        """Z is negative throughout here, so MAG mixing up sign with magnitude would
        show immediately."""
        self.assertLess(cell(self.stats, 'Z', 'mean_rad'), 0.0)
        self.assertGreater(cell(self.stats, 'MAG', 'min_rad'), 0.0)


class TestSignedVersusUnsigned(unittest.TestCase):
    def test_a_symmetric_oscillation_has_near_zero_mean_but_a_real_rmse(self):
        """Sign matters: this filter is wrong by 0.1 rad at every single sample, and
        mean_rad on a per-axis column says it is nearly perfect. Any figure that reports
        accuracy must read rmse/mae (or MAG), never a signed per-axis mean."""
        n = 200
        amplitude = 0.1
        error = np.zeros((n, 3))
        error[:, 2] = amplitude * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        stats = compute_error_stats(make_trial(error))

        self.assertAlmostEqual(cell(stats, 'Z', 'mean_rad'), 0.0, places=12)
        self.assertAlmostEqual(cell(stats, 'Z', 'rmse_rad'), amplitude, places=12)
        self.assertAlmostEqual(cell(stats, 'Z', 'mae_rad'), amplitude, places=12)
        self.assertAlmostEqual(cell(stats, 'MAG', 'mean_rad'), amplitude, places=12)

    def test_rmse_is_at_least_mae_which_is_at_least_the_absolute_mean(self):
        rng = np.random.default_rng(7)
        stats = compute_error_stats(make_trial(rng.normal(scale=0.1, size=(300, 3))))
        for axis in AXES:
            with self.subTest(axis=axis):
                rmse = cell(stats, axis, 'rmse_rad')
                mae = cell(stats, axis, 'mae_rad')
                self.assertGreaterEqual(rmse + 1e-12, mae)
                self.assertGreaterEqual(mae + 1e-12, abs(cell(stats, axis, 'mean_rad')))


class TestGrouping(unittest.TestCase):
    def test_subjects_joints_activities_and_methods_are_not_pooled(self):
        """Every cell gets its own error magnitude, so any missing group key would
        average two of them together into a value matching neither."""
        cells = [
            ('Subject01', 'walking', 'R_Knee', 'mag_on', 0.01),
            ('Subject01', 'walking', 'R_Knee', 'mag_off', 0.02),
            ('Subject01', 'walking', 'L_Hip', 'mag_on', 0.03),
            ('Subject01', 'complexTasks', 'R_Knee', 'mag_on', 0.04),
            ('Subject02', 'walking', 'R_Knee', 'mag_on', 0.05),
        ]
        df = pd.concat([
            make_trial(np.tile([0.0, 0.0, offset], (10, 1)), subject=subject,
                       trial_type=trial_type, joint_name=joint, method=method)
            for subject, trial_type, joint, method, offset in cells
        ], ignore_index=True)

        stats = compute_error_stats(df)
        self.assertEqual(len(stats), len(cells) * len(AXES))
        for subject, trial_type, joint, method, offset in cells:
            with self.subTest(cell=(subject, trial_type, joint, method)):
                self.assertAlmostEqual(
                    cell(stats, 'MAG', 'mean_rad', subject=subject, trial_type=trial_type,
                         joint_name=joint, method=method),
                    offset, places=12)

    def test_the_marker_reference_is_not_reported_as_a_method(self):
        stats = compute_error_stats(make_trial(np.tile([0.0, 0.0, 0.1], (10, 1))))
        self.assertNotIn('marker', stats['method'].unique())
        self.assertEqual(stats['method'].unique().tolist(), ['mag_on'])

    def test_the_marker_reference_is_shared_across_methods(self):
        """One marker trace serves every method, and the merge must not duplicate rows
        when several IMU methods claim the same timestamps."""
        base = make_trial(np.tile([0.0, 0.0, 0.1], (10, 1)), method='mag_on')
        second = base[base['method'] == 'mag_on'].assign(method='mag_off')
        stats = compute_error_stats(pd.concat([base, second], ignore_index=True))
        self.assertEqual(sorted(stats['method'].unique()), ['mag_off', 'mag_on'])
        self.assertEqual(len(stats), 2 * len(AXES))
        for method in ('mag_on', 'mag_off'):
            self.assertAlmostEqual(cell(stats, 'MAG', 'mean_rad', method=method), 0.1, places=12)

    def test_group_columns_are_all_present(self):
        stats = compute_error_stats(make_trial(np.tile([0.0, 0.0, 0.1], (10, 1))))
        for col in GROUP_COLS + ['axis']:
            self.assertIn(col, stats.columns)


class TestTimestampMerge(unittest.TestCase):
    def test_unmatched_timestamps_are_dropped_not_misaligned(self):
        """The merge is on timestamp, so an IMU sample with no marker sample at the same
        instant is discarded. The danger case is a positional join instead, which would
        pair each IMU sample with the wrong marker sample and produce a plausible but
        entirely fictitious error."""
        df = make_trial(np.tile([0.0, 0.0, 0.1], (10, 1)))
        marker = df[df['method'] == 'marker']
        imu = df[df['method'] == 'mag_on'].copy()
        imu['timestamp'] = imu['timestamp'] + 0.005  # offset onto the half-samples

        stats = compute_error_stats(pd.concat([marker, imu], ignore_index=True))
        self.assertTrue(stats.empty)

    def test_a_partial_overlap_keeps_only_the_shared_samples(self):
        good = make_trial(np.tile([0.0, 0.0, 0.1], (10, 1)))
        extra = make_trial(np.tile([0.0, 0.0, 5.0], (10, 1)), t0=100.0)
        extra = extra[extra['method'] == 'mag_on']  # IMU-only samples, no marker to pair with

        stats = compute_error_stats(pd.concat([good, extra], ignore_index=True))
        self.assertAlmostEqual(cell(stats, 'MAG', 'mean_rad'), 0.1, places=12)
        self.assertAlmostEqual(cell(stats, 'MAG', 'max_rad'), 0.1, places=12)


class TestDegenerateInputs(unittest.TestCase):
    def test_an_empty_frame_gives_an_empty_frame(self):
        self.assertTrue(compute_error_stats(pd.DataFrame()).empty)

    def test_marker_only_gives_an_empty_frame(self):
        df = make_trial(np.zeros((5, 3)))
        self.assertTrue(compute_error_stats(df[df['method'] == 'marker']).empty)

    def test_imu_only_gives_an_empty_frame(self):
        """No ground truth means no error. Returning empty is what makes a missing
        marker method show up as a hole in the results rather than as zeros."""
        df = make_trial(np.zeros((5, 3)))
        self.assertTrue(compute_error_stats(df[df['method'] != 'marker']).empty)

    def test_a_single_sample_gives_a_finite_mean_and_a_nan_std(self):
        """std is ddof=1, so one sample cannot have one. It must not silently become 0."""
        stats = compute_error_stats(make_trial(np.array([[0.0, 0.0, 0.1]])))
        self.assertAlmostEqual(cell(stats, 'Z', 'mean_rad'), 0.1, places=12)
        self.assertTrue(np.isnan(cell(stats, 'Z', 'std_rad')))


if __name__ == '__main__':
    unittest.main()
