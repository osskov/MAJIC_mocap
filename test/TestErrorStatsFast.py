"""The optimisations in compute_error_stats must not change a single number.

Three changes are covered here:

  1. relative_rotvec replaces the scipy Rotation composition (~9x)
  2. one quantile([0.25, 0.5, 0.75]) call replaces three separate ones (~2.5x)
  3. the label columns arrive as categoricals rather than str objects (7x less memory)

All three are refactors, so the bar is exact agreement with the pre-change behaviour --
recomputed here from scipy directly rather than trusted from a stored fixture, so the test
still means something if the implementation is rewritten again.

TestErrorStats.py covers what the statistics MEAN; this file only covers that making them
faster did not move them.
"""
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from experiments.experiment_utils import (as_categorical_labels, compute_error_stats,
                                          LABEL_COLUMNS)
from src.toolchest.gyro_utils import relative_rotvec

JOINTS = ['R_Knee', 'L_Ankle', 'Lumbar']
METHODS = ['marker', 'mag_on', 'mag_off']
SUBJECTS = ['Subject01', 'Subject02']
N_SAMPLES = 300


def _synthetic_joint_angles(seed=0):
    """A joint-angles frame shaped exactly like what load_all_joint_angles returns."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject in SUBJECTS:
        for trial_type in ('walking', 'complexTasks'):
            for joint in JOINTS:
                truth = np.cumsum(0.02 * rng.standard_normal((N_SAMPLES, 3)), axis=0)
                for method in METHODS:
                    # 'marker' is the reference; the others deviate from it by a
                    # method-dependent amount, including some large errors so the
                    # near-pi branch of the log map gets exercised.
                    if method == 'marker':
                        rotvec = truth
                    else:
                        scale = 0.05 if method == 'mag_on' else 0.9
                        rotvec = truth + scale * rng.standard_normal((N_SAMPLES, 3))
                    rows.append(pd.DataFrame({
                        'timestamp': 0.01 * np.arange(N_SAMPLES),
                        'joint_name': joint,
                        'rx': rotvec[:, 0], 'ry': rotvec[:, 1], 'rz': rotvec[:, 2],
                        'subject': subject, 'trial_type': trial_type, 'method': method,
                    }))
    return pd.concat(rows, ignore_index=True)


class TestRelativeRotvecMatchesScipy(unittest.TestCase):
    """relative_rotvec must reproduce the scipy expression it replaced."""

    def _check(self, a, b, tol=5e-15):
        expected = (Rotation.from_rotvec(a) * Rotation.from_rotvec(b).inv()).as_rotvec()
        np.testing.assert_allclose(relative_rotvec(a, b), expected, atol=tol)

    def test_generic_rotations(self):
        rng = np.random.default_rng(0)
        self._check(rng.normal(size=(5000, 3)), rng.normal(size=(5000, 3)))

    def test_uniformly_random_rotations(self):
        self._check(Rotation.random(5000, random_state=1).as_rotvec(),
                    Rotation.random(5000, random_state=2).as_rotvec())

    def test_near_identity(self):
        """Both from_rotvec and as_rotvec switch to a Taylor branch below 1e-3 rad."""
        rng = np.random.default_rng(3)
        self._check(1e-9 * rng.normal(size=(2000, 3)), 1e-9 * rng.normal(size=(2000, 3)))
        self._check(1e-4 * rng.normal(size=(2000, 3)), 1e-4 * rng.normal(size=(2000, 3)))

    def test_identical_rotations_give_zero(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(2000, 3))
        np.testing.assert_allclose(relative_rotvec(a, a), 0.0, atol=1e-14)

    def test_near_pi(self):
        """The log map is ill-conditioned at pi, and the w >= 0 canonicalisation is what
        keeps the result in [0, pi] rather than flipping sign arbitrarily."""
        rng = np.random.default_rng(5)
        axis = rng.normal(size=(2000, 3))
        axis /= np.linalg.norm(axis, axis=1)[:, None]
        near_pi = axis * (np.pi - 1e-7)
        self._check(near_pi, np.zeros_like(near_pi), tol=1e-6)
        angles = np.linalg.norm(relative_rotvec(near_pi, np.zeros_like(near_pi)), axis=1)
        self.assertTrue(np.all(angles <= np.pi + 1e-9))

    def test_returned_angle_is_in_the_principal_range(self):
        a = Rotation.random(5000, random_state=6).as_rotvec()
        b = Rotation.random(5000, random_state=7).as_rotvec()
        angles = np.linalg.norm(relative_rotvec(a, b), axis=1)
        self.assertTrue(np.all(angles <= np.pi + 1e-9))
        self.assertTrue(np.all(angles >= 0.0))


class TestComputeErrorStatsUnchanged(unittest.TestCase):
    """The whole function, against an independent reference implementation of the parts
    that were optimised."""

    def setUp(self):
        self.df = _synthetic_joint_angles()

    def _reference_stats(self, df):
        """compute_error_stats' aggregations, computed the slow explicit way."""
        marker = df[df['method'] == 'marker']
        imu = df[df['method'] != 'marker']
        merged = pd.merge(imu, marker,
                          on=['subject', 'trial_type', 'joint_name', 'timestamp'],
                          suffixes=('_imu', '_marker'))
        err = (Rotation.from_rotvec(merged[['rx_imu', 'ry_imu', 'rz_imu']].to_numpy())
               * Rotation.from_rotvec(
                   merged[['rx_marker', 'ry_marker', 'rz_marker']].to_numpy()).inv()
               ).as_rotvec()
        merged = merged.rename(columns={'method_imu': 'method'})
        merged['X'], merged['Y'], merged['Z'] = err[:, 0], err[:, 1], err[:, 2]
        merged['MAG'] = np.linalg.norm(err, axis=1)
        out = {}
        for keys, g in merged.groupby(['trial_type', 'method', 'joint_name', 'subject'],
                                      observed=True):
            for axis in ('MAG', 'X', 'Y', 'Z'):
                v = g[axis].to_numpy()
                out[keys + (axis,)] = {
                    'mean_rad': v.mean(), 'rmse_rad': np.sqrt((v ** 2).mean()),
                    'mae_rad': np.abs(v).mean(),
                    'median_rad': np.median(v),
                    'q25_rad': np.quantile(v, 0.25), 'q75_rad': np.quantile(v, 0.75),
                    'min_rad': v.min(), 'max_rad': v.max(),
                    'mad_rad': np.median(np.abs(v - np.median(v))),
                }
        return out

    def test_matches_an_independent_reference(self):
        actual = compute_error_stats(self.df)
        expected = self._reference_stats(self.df)
        self.assertEqual(len(actual), len(expected))
        indexed = actual.set_index(['trial_type', 'method', 'joint_name', 'subject', 'axis'])
        for key, metrics in expected.items():
            for metric, value in metrics.items():
                self.assertAlmostEqual(
                    float(indexed.loc[key, metric]), float(value), places=10,
                    msg=f"{metric} differs for {key}")

    def test_categorical_labels_give_identical_results(self):
        """The memory optimisation must be invisible in the output."""
        plain = compute_error_stats(self.df.copy())
        categorical = compute_error_stats(as_categorical_labels(self.df.copy()))
        pd.testing.assert_frame_equal(plain, categorical)

    def test_categorical_input_does_not_invent_empty_groups(self):
        """A categorical groupby without observed=True expands to the cartesian product
        of categories. Drop a combination and the output must shrink, not gain NaN rows."""
        subset = self.df[~((self.df['subject'] == 'Subject02')
                           & (self.df['joint_name'] == 'Lumbar'))]
        full = compute_error_stats(as_categorical_labels(self.df.copy()))
        partial = compute_error_stats(as_categorical_labels(subset.copy()))
        self.assertLess(len(partial), len(full))
        self.assertFalse(partial[['mean_rad', 'rmse_rad', 'median_rad']].isna().any().any())

    def test_label_columns_come_back_as_strings(self):
        """Whatever went in, downstream consumers get plain strings."""
        for frame in (self.df.copy(), as_categorical_labels(self.df.copy())):
            stats = compute_error_stats(frame)
            for col in ('trial_type', 'method', 'joint_name', 'subject'):
                self.assertFalse(isinstance(stats[col].dtype, pd.CategoricalDtype),
                                 f"{col} leaked a categorical dtype")

    def test_quantile_columns_are_ordered(self):
        stats = compute_error_stats(self.df)
        self.assertTrue((stats['q25_rad'] <= stats['median_rad'] + 1e-12).all())
        self.assertTrue((stats['median_rad'] <= stats['q75_rad'] + 1e-12).all())
        self.assertTrue((stats['min_rad'] <= stats['q25_rad'] + 1e-12).all())
        self.assertTrue((stats['q75_rad'] <= stats['max_rad'] + 1e-12).all())

    def test_marker_only_and_empty_inputs_still_return_empty(self):
        self.assertTrue(compute_error_stats(pd.DataFrame()).empty)
        self.assertTrue(
            compute_error_stats(self.df[self.df['method'] == 'marker']).empty)
        self.assertTrue(
            compute_error_stats(self.df[self.df['method'] != 'marker']).empty)


class TestAsCategoricalLabels(unittest.TestCase):
    def _present(self, df):
        """The label columns this frame actually has.

        Not all of LABEL_COLUMNS: `dataset` and `trial` joined it when the pipeline gained a
        second dataset, and a frame is not required to carry them — a single-trial caller has
        nothing to put in them, and artifacts written before they existed do not have them.
        Tolerating that is the documented contract (see the idempotence test below), so a test
        that indexed every name unconditionally would be pinning the opposite.
        """
        return [col for col in LABEL_COLUMNS if col in df.columns]

    def test_converts_only_the_label_columns(self):
        df = _synthetic_joint_angles()
        out = as_categorical_labels(df.copy())
        for col in self._present(df):
            self.assertIsInstance(out[col].dtype, pd.CategoricalDtype, col)
        for col in ('timestamp', 'rx', 'ry', 'rz'):
            self.assertEqual(out[col].dtype, np.float64)

    def test_it_converts_the_dataset_and_trial_labels_too(self):
        df = _synthetic_joint_angles().assign(dataset='alborno', trial='walking')
        out = as_categorical_labels(df)
        for col in ('dataset', 'trial'):
            self.assertIsInstance(out[col].dtype, pd.CategoricalDtype, col)

    def test_is_idempotent_and_tolerates_missing_columns(self):
        df = _synthetic_joint_angles().drop(columns=['trial_type'])
        once = as_categorical_labels(df.copy())
        twice = as_categorical_labels(once.copy())
        pd.testing.assert_frame_equal(once, twice)

    def test_values_are_preserved(self):
        df = _synthetic_joint_angles()
        out = as_categorical_labels(df.copy())
        for col in self._present(df):
            pd.testing.assert_series_equal(
                out[col].astype(str), df[col].astype(str), check_dtype=False)


class TestTrialIsPartOfTheKey(unittest.TestCase):
    """`trial` and `dataset` join the merge key and the grouping when they are present.

    The failure this guards is silent and quadratic. Al Borno has one trial per activity, so
    (subject, trial_type) identified a trial and nothing needed the distinction; IMoVE holds
    several takes of one activity per session. Without `trial` in the merge key, take A's
    estimate joins take B's ground truth on (subject, trial_type, joint, timestamp) — every
    pairing of the two — and the resulting error is the difference between two different
    walks.
    """

    def _two_trials(self):
        """One activity, two takes, with DIFFERENT ground truth in each.

        The second take's marker angles are offset, so a cross-join shows up as an error where
        there should be none: within each take the estimate equals its own marker exactly.
        """
        frames = []
        for trial, offset in (('t1_walking_001', 0.0), ('t1_walking_002', 0.5)):
            truth = offset + 0.01 * np.arange(N_SAMPLES)[:, None] * np.ones(3)
            for method in ('marker', 'mag_on'):
                frames.append(pd.DataFrame({
                    'timestamp': 0.01 * np.arange(N_SAMPLES),
                    'joint_name': 'R_Knee',
                    'rx': truth[:, 0], 'ry': truth[:, 1], 'rz': truth[:, 2],
                    'subject': 's13', 'trial': trial, 'trial_type': 'walking',
                    'method': method, 'dataset': 'imove',
                }))
        return pd.concat(frames, ignore_index=True)

    def test_each_trial_is_scored_against_its_own_ground_truth(self):
        stats = compute_error_stats(self._two_trials())
        self.assertEqual(sorted(stats['trial'].unique()),
                         ['t1_walking_001', 't1_walking_002'])
        # Each take's estimate IS its own marker, so every error is exactly zero. A cross-join
        # would put the 0.5 rad offset between the takes into these rows.
        np.testing.assert_allclose(stats['rmse_rad'].to_numpy(), 0.0, atol=1e-12)

    def test_dropping_the_trial_column_is_what_cross_joins(self):
        """The counterfactual, so the test above is known to be measuring the guard and not
        an accident of the fixture."""
        without = compute_error_stats(self._two_trials().drop(columns=['trial']))
        self.assertGreater(float(without['rmse_rad'].max()), 0.1)


if __name__ == '__main__':
    unittest.main()
