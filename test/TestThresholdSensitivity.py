"""
Unit tests for the mag_adapt threshold sweep and its supplementary figure.

The sweep's whole claim is a correspondence between three things that are computed in
different places and never checked against each other at runtime:

    the threshold in the METHOD NAME      -> what the filter actually gates on
    the threshold in the GATING TABLE     -> what the figure's duty-cycle axis says
    the o^J in the gating table           -> what _run_relative_filter compares against

If any pair drifts apart the figure still draws, the curve still looks smooth, and the
x-axis is simply wrong by some amount nobody can see. So the correspondences are pinned
here as equalities rather than left to the fact that both sides call the same helper.

Synthetic throughout — no dataset needed. The plate fixtures are shared with
test/TestExperimentPhysics.py, which pins o^J's physics; this file assumes that and tests
only the bookkeeping built on top of it.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd

import paths

from experiments.experiment_utils import (DEFAULT_MAG_ADAPT_THRESHOLD, TRIAL_DATASET,
                                          _calculate_observability_metric_, _run_relative_filter,
                                          project_pair_to_joint_center, resolve_method_spec)
from experiments.threshold_sensitivity import (EXPERIMENT_NAME, OBS_PERCENTILES, REFERENCE_METHODS,
                                               REFERENCE_VARIANT, THRESHOLDS,
                                               constants_disagreements, describe_disagreements,
                                               arm_constants, gating_tables,
                                               joint_observability, sweep_arms, sweep_stds,
                                               threshold_from_method, threshold_method)
from plotting.threshold_sensitivity import (BLOCK_COLS, METRIC, arm_table, block_curve,
                                            per_cell_optimum, pooled_duty_cycle, threshold_order,
                                            within_block_interval)
from test.TestExperimentPhysics import make_plate, make_shaking_plate


def make_plates() -> dict:
    """A two-sensor 'trial' spanning one joint (JOINTS['Lumbar']), moving enough that o^J
    spreads over a useful range instead of pinning to zero or to one value."""
    return {'pelvis_imu': make_shaking_plate('pelvis_imu', n=400, rate=5.0),
            'torso_imu': make_shaking_plate('torso_imu', n=400, rate=3.0,
                                            omega=np.array([0.2, -0.4, 1.0]))}


class TestThresholdGrid(unittest.TestCase):
    """The swept grid itself. Every property here is something a later edit to THRESHOLDS
    could quietly break while leaving the sweep runnable."""

    def test_the_shipped_default_is_one_of_the_swept_points(self):
        """The grid is anchored ON the default, not around it. If it were not, the figure
        would have to interpolate the default's RMSE between two measured points and every
        'the default costs X' number in the paper would be an estimate."""
        self.assertIn(DEFAULT_MAG_ADAPT_THRESHOLD, set(THRESHOLDS.tolist()))

    def test_every_threshold_survives_the_two_decimal_method_name(self):
        """THE correspondence the duty-cycle axis rests on. The filter's threshold comes
        from parsing the method name, the gating table's comes from the array, and the name
        carries only two decimals — so any grid value with more of them would gate at one
        number and be tabulated at another."""
        for threshold in THRESHOLDS:
            with self.subTest(threshold=threshold):
                self.assertEqual(float(f"{threshold:.2f}"), float(threshold))

    def test_method_names_round_trip_through_resolve_method_spec(self):
        for threshold in THRESHOLDS:
            with self.subTest(threshold=threshold):
                spec = resolve_method_spec(threshold_method(threshold))
                self.assertEqual(spec['mag_adapt_threshold'], float(threshold))
                self.assertEqual(spec['mag_mode'], 'adapt')

    def test_threshold_from_method_inverts_threshold_method(self):
        names = pd.Series([threshold_method(t) for t in THRESHOLDS])
        np.testing.assert_allclose(threshold_from_method(names).to_numpy(), THRESHOLDS)

    def test_threshold_from_method_is_nan_for_the_reference_arms(self):
        """mag_on/mag_off share the statistics file with the sweep and are selected out of
        it by threshold.notna(). A parse that returned 0.0 for them instead would silently
        add a tenth point to every curve."""
        parsed = threshold_from_method(pd.Series(REFERENCE_METHODS + ['marker', 'mag_adapt']))
        self.assertTrue(parsed.isna().all())

    def test_the_grid_brackets_the_default_by_more_than_a_decade_each_way(self):
        """A sweep that does not reach both saturations cannot show the curve converging to
        mag_off and mag_on, which is the figure's check on the whole pipeline."""
        self.assertLess(THRESHOLDS.min(), DEFAULT_MAG_ADAPT_THRESHOLD / 10.0)
        self.assertGreater(THRESHOLDS.max(), DEFAULT_MAG_ADAPT_THRESHOLD * 10.0)


class TestJointObservability(unittest.TestCase):
    """The o^J the duty-cycle table is built from must be the o^J the filter gates on."""

    def setUp(self):
        self.plates = make_plates()

    def test_it_matches_what_the_filter_gates_on(self):
        """THE test in this file. _run_relative_filter projects to the joint center and then
        compares o^J against the threshold; joint_observability has to reproduce that
        pipeline exactly, including the projection. Computing o^J on unprojected traces
        would still give a plausible curve, just of a different quantity."""
        _, from_filter = _run_relative_filter(self.plates['pelvis_imu'], self.plates['torso_imu'],
                                              project=True, mag_mode='adapt',
                                              return_observability=True)
        np.testing.assert_allclose(joint_observability(self.plates)['Lumbar'], from_filter,
                                   rtol=1e-12, atol=1e-12)

    def test_the_projection_is_not_a_no_op_on_this_fixture(self):
        """Guards the test above from passing for the wrong reason: if the two plates sat at
        the same point the projection would change nothing and the comparison would hold
        even for an unprojected implementation."""
        unprojected = _calculate_observability_metric_(self.plates['pelvis_imu'],
                                                       self.plates['torso_imu'])
        self.assertFalse(np.allclose(joint_observability(self.plates)['Lumbar'], unprojected))

    def test_joints_with_a_missing_sensor_are_skipped(self):
        """Matches _joint_angles_from_filter, so the gating table has a row for exactly the
        joints the accuracy table has rows for."""
        observability = joint_observability({'pelvis_imu': self.plates['pelvis_imu']})
        self.assertEqual(observability, {})


class TestGatingTables(unittest.TestCase):
    """The duty cycle, which is the only thing that gives the figure's x-axis a unit."""

    def setUp(self):
        self.plates = make_plates()
        self.thresholds = np.array([0.0, 10.0, 100.0, 1000.0, 1e9])
        self.tables = gating_tables('01', 'walking', self.plates, self.thresholds)
        self.gating = self.tables['gating']
        self.obs = joint_observability(self.plates)['Lumbar']

    def test_the_fraction_gated_is_the_fraction_over_the_threshold(self):
        for _, row in self.gating.iterrows():
            with self.subTest(threshold=row['threshold']):
                self.assertAlmostEqual(row['fraction_gated'],
                                       float(np.mean(self.obs > row['threshold'])), places=12)

    def test_it_is_monotonically_non_increasing_in_the_threshold(self):
        ordered = self.gating.sort_values('threshold')['fraction_gated'].to_numpy()
        self.assertTrue(np.all(np.diff(ordered) <= 1e-12))

    def test_the_comparison_is_strict_so_the_sample_zero_pad_is_never_gated(self):
        """o^J is padded with 0.0 at t=0 and the filter uses `obs > threshold`, so even a
        threshold of 0 leaves exactly that sample ungated. A `>=` here would report 100%
        and disagree with the filter by one sample at every threshold."""
        at_zero = self.gating.loc[self.gating['threshold'] == 0.0, 'fraction_gated'].iloc[0]
        self.assertAlmostEqual(at_zero, 1.0 - 1.0 / len(self.obs), places=12)

    def test_the_limits_of_the_sweep_are_mag_off_and_mag_on(self):
        """The claim that a low enough threshold IS mag_off and a high enough one IS mag_on,
        which is what licenses drawing them as the curve's limits."""
        fractions = self.gating.set_index('threshold')['fraction_gated']
        self.assertGreater(fractions[0.0], 0.99)
        self.assertEqual(fractions[1e9], 0.0)

    def test_the_percentile_grid_agrees_with_the_fractions(self):
        """The two tables describe the same distribution from opposite sides: the threshold
        at the (1 - fraction_gated) percentile must be the threshold itself. Checked away
        from the ends, where the discreteness of the sample set dominates."""
        percentiles = self.tables['observability'].set_index('percentile')['observability']
        for threshold in (10.0, 100.0, 1000.0):
            fraction = float(self.gating.loc[self.gating['threshold'] == threshold,
                                             'fraction_gated'].iloc[0])
            if not 0.05 < fraction < 0.95:
                continue
            with self.subTest(threshold=threshold):
                recovered = np.interp(100.0 * (1.0 - fraction), percentiles.index,
                                      percentiles.to_numpy())
                self.assertAlmostEqual(recovered, threshold, delta=0.02 * threshold)

    def test_the_percentile_grid_spans_the_full_range(self):
        self.assertEqual(OBS_PERCENTILES.min(), 0.0)
        self.assertEqual(OBS_PERCENTILES.max(), 100.0)
        self.assertTrue(np.all(np.diff(OBS_PERCENTILES) > 0))

    def test_a_trial_with_no_usable_joint_produces_no_tables(self):
        self.assertEqual(gating_tables('01', 'walking', {'pelvis_imu': make_plate()},
                                       self.thresholds), {})


def fake_sweep(rmse_by_threshold: dict, subjects=('Subject01', 'Subject02'),
               joints=('Lumbar', 'Ankle')) -> pd.DataFrame:
    """A minimal swept table in the shape load_sweep returns."""
    rows = [{'subject': s, 'joint_name': j, 'trial_type': 'walking',
             'threshold': float(t), METRIC: value + 0.1 * joints.index(j)}
            for s in subjects for j in joints for t, value in rmse_by_threshold.items()]
    return pd.DataFrame(rows)


def fake_references(rmse_by_method: dict, subjects=('Subject01', 'Subject02'),
                    joints=('Lumbar', 'Ankle')) -> pd.DataFrame:
    rows = [{'subject': s, 'joint_name': j, 'trial_type': 'walking', 'method': m,
             METRIC: value + 0.1 * joints.index(j)}
            for s in subjects for j in joints for m, value in rmse_by_method.items()]
    return pd.DataFrame(rows)


class TestWithinBlockInterval(unittest.TestCase):
    """The interval the accuracy panels are drawn with. Every arm is measured on the same
    blocks, so the between-block variance is common to all of them and has to be removed
    before the bands mean anything — see the comment above BLOCK_COLS."""

    def setUp(self):
        # Blocks that differ hugely in level (0, 10, 20, 30) but agree on the shape across
        # arms. This is the real data's situation: the between-block std dwarfs the effect.
        offsets = np.array([0.0, 10.0, 20.0, 30.0])
        shape = {'a': 1.0, 'b': 2.0, 'c': 1.5}
        self.wide = pd.DataFrame({arm: offsets + value for arm, value in shape.items()},
                                 index=[f'block{i}' for i in range(4)])

    def test_the_arm_means_are_untouched(self):
        """Centring rescales the spread only. If it moved the means, the figure's curve —
        and every number annotated on it — would be a different quantity from the table."""
        interval = within_block_interval(self.wide)
        np.testing.assert_allclose(interval['mean'].to_numpy(), self.wide.mean().to_numpy())

    def test_it_is_far_narrower_than_the_between_block_ci_when_blocks_differ_in_level(self):
        """THE reason it exists. Here the blocks agree exactly on the shape, so the
        within-block interval collapses to zero width while the naive one spans ±12."""
        interval = within_block_interval(self.wide)
        naive = 1.96 * self.wide.sem()
        self.assertGreater(float(naive.min()), 10.0)
        np.testing.assert_allclose((interval['upper'] - interval['lower']).to_numpy(), 0.0,
                                   atol=1e-12)

    def test_it_still_reports_spread_that_is_genuinely_within_block(self):
        """The complement of the test above: a block that disagrees about the shape must
        widen the interval. Otherwise the normalization would be hiding the effect rather
        than isolating it."""
        wide = self.wide.copy()
        wide.loc['block0', 'b'] += 6.0
        interval = within_block_interval(wide)
        self.assertGreater(float(interval.loc['b', 'upper'] - interval.loc['b', 'lower']), 1.0)

    def test_incomplete_blocks_are_dropped(self):
        """Same requirement as Friedman: a block present for some arms and not others would
        be centred against a mean it did not contribute to."""
        wide = self.wide.copy()
        wide.loc['block0', 'c'] = np.nan
        interval = within_block_interval(wide)
        self.assertTrue((interval['n_blocks'] == 3).all())
        np.testing.assert_allclose(interval['mean'].to_numpy(),
                                   self.wide.drop(index='block0').mean().to_numpy())

    def test_degenerate_shapes_return_zero_width_rather_than_nan(self):
        """One block, or one arm, is a legitimate state for a partially-run sweep. The
        Morey factor divides by (k - 1), so a single arm must be special-cased."""
        for wide in (self.wide.iloc[:1], self.wide[['a']]):
            with self.subTest(shape=wide.shape):
                interval = within_block_interval(wide)
                self.assertFalse(interval.isna().to_numpy().any())
                np.testing.assert_allclose((interval['upper'] - interval['lower']).to_numpy(), 0.0)

    def test_an_empty_table_returns_the_expected_columns(self):
        interval = within_block_interval(pd.DataFrame())
        self.assertTrue(interval.empty)
        self.assertEqual(list(interval.columns), ['mean', 'lower', 'upper', 'n_blocks'])


class TestArmTable(unittest.TestCase):
    """The one wide table both accuracy panels reduce. The references have to sit in it
    beside the swept arms, on the same blocks, or their bands are not comparable to the
    curve's."""

    def setUp(self):
        self.swept = fake_sweep({100.0: 5.0, 1000.0: 4.0})
        self.references = fake_references({'mag_off': 8.0, 'mag_on': 3.0})

    def test_it_carries_swept_arms_as_floats_and_references_as_strings(self):
        """block_curve separates the two by type, so a threshold arriving as a string (or a
        method as a float) would silently drop it from the curve."""
        columns = set(arm_table(self.swept, self.references).columns)
        self.assertEqual(columns, {100.0, 1000.0, 'mag_off', 'mag_on'})

    def test_it_is_indexed_by_block(self):
        wide = arm_table(self.swept, self.references)
        self.assertEqual(list(wide.index.names), BLOCK_COLS)
        self.assertEqual(len(wide), 4)

    def test_it_restricts_to_one_joint(self):
        wide = arm_table(self.swept, self.references, joint='Ankle')
        self.assertEqual(len(wide), 2)
        self.assertEqual(set(wide.index.get_level_values('joint_name')), {'Ankle'})

    def test_it_survives_missing_references(self):
        columns = set(arm_table(self.swept, pd.DataFrame()).columns)
        self.assertEqual(columns, {100.0, 1000.0})

    def test_block_curve_ignores_the_reference_arms(self):
        """The references are in the table so they share the centring, but they are not
        points on the curve — including them would put two arms at an undefined x."""
        curve = block_curve(self.swept, self.references)
        np.testing.assert_array_equal(curve['threshold'].to_numpy(), [100.0, 1000.0])


class TestFigureReductions(unittest.TestCase):
    """The three reductions the figure's numbers come out of. Each is a groupby whose
    failure mode is a plausible wrong number, not an exception."""

    def setUp(self):
        self.curve_values = {100.0: 5.0, 1000.0: 4.0, 10000.0: 6.0}
        self.swept = fake_sweep(self.curve_values)

    def test_block_curve_averages_over_blocks_not_over_rows(self):
        """Side and activity are averaged into a block first, so a joint recorded on both
        sides must not count twice. Here the two joints differ by 0.1 deg, so the mean is
        the swept value plus 0.05."""
        curve = block_curve(self.swept).set_index('threshold')
        for threshold, value in self.curve_values.items():
            with self.subTest(threshold=threshold):
                self.assertAlmostEqual(curve.loc[threshold, 'mean'], value + 0.05, places=12)
        self.assertTrue((curve['n_blocks'] == 4).all())  # 2 subjects x 2 joints

    def test_block_curve_is_sorted_by_threshold(self):
        """It is drawn with ax.plot, which connects points in row order — an unsorted curve
        would draw as a zigzag rather than fail."""
        thresholds = block_curve(self.swept)['threshold'].to_numpy()
        np.testing.assert_array_equal(thresholds, np.sort(thresholds))

    def test_per_cell_penalty_is_never_negative(self):
        """The default is one of the swept points, so its RMSE cannot beat the minimum over
        those points. A negative penalty would mean the two are being read off different
        groupings."""
        optima = per_cell_optimum(self.swept)
        self.assertFalse(optima.empty)
        self.assertTrue((optima['penalty'] >= 0).all())
        self.assertTrue((optima['best_rmse'] <= optima['default_rmse'] + 1e-12).all())

    def test_per_cell_optimum_finds_the_argmin(self):
        optima = per_cell_optimum(self.swept)
        self.assertTrue((optima['best_threshold'] == 1000.0).all())
        self.assertTrue(np.allclose(optima['penalty'], 0.0))

    def test_per_cell_optimum_has_one_row_per_block(self):
        optima = per_cell_optimum(self.swept)
        self.assertEqual(len(optima), 4)
        self.assertEqual(len(optima.drop_duplicates(BLOCK_COLS)), 4)

    def test_the_penalty_is_measured_against_the_default_not_the_best_neighbour(self):
        """A grid whose optimum is at an end: the penalty must be default - best, not the
        step between adjacent points."""
        swept = fake_sweep({100.0: 3.0, 1000.0: 5.0, 10000.0: 4.5})
        optima = per_cell_optimum(swept)
        self.assertTrue((optima['best_threshold'] == 100.0).all())
        self.assertTrue(np.allclose(optima['penalty'], 2.0))

    def test_threshold_order_is_numeric_not_lexicographic(self):
        """The group order handed to the significance test and used for the x order. Sorted
        as strings, '10000' would come before '1000' and the pairwise table would be
        labelled against the wrong neighbours."""
        self.assertEqual(threshold_order(self.swept), ['100', '1000', '10000'])

    def test_pooled_duty_cycle_averages_over_trial_joints(self):
        gating = pd.DataFrame([
            {'threshold': 100.0, 'fraction_gated': 0.8, 'joint_name': 'Ankle'},
            {'threshold': 100.0, 'fraction_gated': 0.4, 'joint_name': 'Lumbar'},
            {'threshold': 1000.0, 'fraction_gated': 0.2, 'joint_name': 'Ankle'},
            {'threshold': 1000.0, 'fraction_gated': 0.0, 'joint_name': 'Lumbar'},
        ])
        pooled = pooled_duty_cycle(gating)
        self.assertAlmostEqual(pooled[100.0], 0.6)
        self.assertAlmostEqual(pooled[1000.0], 0.1)

    def test_the_reductions_survive_an_empty_table(self):
        """The figure prints a warning and draws an empty panel when a sweep has not been
        run; none of these may raise on the way there."""
        empty = pd.DataFrame(columns=['subject', 'joint_name', 'threshold', METRIC])
        self.assertTrue(block_curve(empty).empty)
        self.assertTrue(per_cell_optimum(empty).empty)
        self.assertTrue(pooled_duty_cycle(pd.DataFrame()).empty)


class TestCrossArmConsistency(unittest.TestCase):
    """The guard on results/joint_angles/ being a shared namespace.

    This exists because the failure it catches happened twice in one day and was silent both
    times: another script rewrote mag_on/mag_off with retuned constants, every arm still
    loaded, and the only symptom was a reference line ~6 deg away from a swept arm that gates
    0.2% of samples and therefore IS mag_on."""

    BASE = {'gyro_std': 0.0045, 'acc_std': 0.037, 'mag_std': 0.03, 'mag_adapt_threshold': 1000.0}
    RETUNED = {'gyro_std': 0.0045, 'acc_std': 0.018, 'mag_std': 0.05, 'mag_adapt_threshold': 1000.0}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # BOTH roots. This experiment's arms now live under
        # results/experiments/<name>/joint_angles/, which resolves through EXPERIMENTS_DIR, so
        # patching JOINT_ANGLES_DIR alone let these tests write real manifests into the real
        # results tree — and then the two "nothing on disk" cases found the previous test's
        # leftovers and failed. Anything that redirects one root has to redirect the other.
        for attr in ('JOINT_ANGLES_DIR', 'EXPERIMENTS_DIR'):
            patcher = mock.patch.object(paths, attr, Path(self.tmp.name) / attr.lower())
            patcher.start()
            self.addCleanup(patcher.stop)

    def write(self, method, constants, subject='01', activity='walking', variant=None):
        # Written where `arm_constants` now looks: this experiment's own tree, not the
        # benchmark's canonical one.
        path = paths.manifest_path(
            paths.joint_angles_path(TRIAL_DATASET, subject, activity, method, variant=variant,
                                    experiment=EXPERIMENT_NAME))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'constants': constants}))

    def test_a_clean_sweep_is_a_single_group(self):
        for method in ('mag_on', 'mag_off', 'mag_adapt_th1000.00'):
            self.write(method, self.BASE)
        groups = constants_disagreements([('mag_on', None), ('mag_off', None), ('mag_adapt_th1000.00', None)],
                                         subjects=['01'], activities=['walking'])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(next(iter(groups.values()))), 3)

    def test_a_retuned_reference_arm_is_caught(self):
        """The exact incident: the swept arm keeps the old constants, mag_on is overwritten
        by another script with new ones."""
        self.write('mag_adapt_th1000.00', self.BASE)
        self.write('mag_on', self.RETUNED)
        groups = constants_disagreements([('mag_on', None), ('mag_adapt_th1000.00', None)],
                                         subjects=['01'], activities=['walking'])
        self.assertEqual(len(groups), 2)
        flat = {arm for arms in groups.values() for arm in arms}
        self.assertEqual(flat, {'Subject01/walking/mag_on',
                                'Subject01/walking/mag_adapt_th1000.00'})

    def test_the_swept_parameter_itself_is_not_a_disagreement(self):
        """mag_adapt_threshold legitimately differs per arm — it is what the sweep sweeps.
        Comparing on it would make every single arm its own group and the guard would fire
        on every healthy run, which is the fastest way to get a guard switched off."""
        for threshold in (100.0, 1000.0, 10000.0):
            self.write(threshold_method(threshold), {**self.BASE, 'mag_adapt_threshold': threshold})
        groups = constants_disagreements([(threshold_method(t), None) for t in (100.0, 1000.0, 10000.0)],
                                         subjects=['01'], activities=['walking'])
        self.assertEqual(len(groups), 1)

    def test_a_new_constant_appearing_counts_as_a_disagreement(self):
        """The real incident also added an 'oracle_std_scale' key. A comparison that only
        checked shared keys would have let that through."""
        self.write('mag_adapt_th1000.00', self.BASE)
        self.write('mag_on', {**self.BASE, 'oracle_std_scale': 0.01})
        groups = constants_disagreements([('mag_on', None), ('mag_adapt_th1000.00', None)],
                                         subjects=['01'], activities=['walking'])
        self.assertEqual(len(groups), 2)

    def test_missing_arms_are_skipped_not_treated_as_a_group(self):
        """A partially-run sweep is a normal state and must not trip the guard."""
        self.write('mag_adapt_th1000.00', self.BASE)
        groups = constants_disagreements([('mag_on', None), ('mag_off', None), ('mag_adapt_th1000.00', None)],
                                         subjects=['01'], activities=['walking'])
        self.assertEqual(len(groups), 1)

    def test_nothing_on_disk_is_not_a_disagreement(self):
        self.assertEqual(constants_disagreements([('mag_on', None)], subjects=['01'],
                                                 activities=['walking']), {})

    def test_the_limits_live_in_the_sweeps_own_variant(self):
        """The structural half of the fix. If the limits went back to the default namespace,
        the next benchmark run would overwrite them again and the guard would be the only
        thing standing between that and a published figure."""
        arms = dict(sweep_arms(np.array([100.0, 1000.0])))
        self.assertEqual(arms['mag_adapt_th100.00'], None)
        self.assertEqual(arms['mag_adapt_th1000.00'], None)
        for method in REFERENCE_METHODS:
            self.assertEqual(arms[method], REFERENCE_VARIANT)

    def test_a_variant_arm_does_not_collide_with_the_default_one(self):
        """Two different tunings of mag_on must be two files, not one. This is the property
        that makes the clobbering structurally impossible rather than merely detected."""
        self.write('mag_on', self.BASE)
        self.write('mag_on', self.RETUNED, variant=REFERENCE_VARIANT)
        self.assertEqual(arm_constants('01', 'walking', 'mag_on')['acc_std'], 0.037)
        self.assertEqual(arm_constants('01', 'walking', 'mag_on', REFERENCE_VARIANT)['acc_std'],
                         0.018)

    def test_sweep_stds_reads_the_tuning_off_the_swept_arms(self):
        """--references-only pins the limits to this, so it has to come from the arms on disk
        rather than from the module constants, which have since moved."""
        for threshold in (100.0, 1000.0):
            self.write(threshold_method(threshold),
                       {**self.BASE, 'mag_adapt_threshold': threshold})
        stds = sweep_stds(np.array([100.0, 1000.0]), subjects=['01'], activities=['walking'])
        self.assertEqual(stds, {'gyro_std': 0.0045, 'acc_std': 0.037, 'mag_std': 0.03})

    def test_sweep_stds_refuses_when_the_swept_arms_are_themselves_mixed(self):
        """Then there is no single tuning to match and only a full re-run will do — silently
        picking one of the two would bake the mismatch in permanently."""
        self.write(threshold_method(100.0), self.BASE)
        self.write(threshold_method(1000.0), self.RETUNED)
        with self.assertRaises(ValueError):
            sweep_stds(np.array([100.0, 1000.0]), subjects=['01'], activities=['walking'])

    def test_sweep_stds_refuses_when_nothing_is_on_disk(self):
        with self.assertRaises(ValueError):
            sweep_stds(np.array([100.0]), subjects=['01'], activities=['walking'])

    def test_the_guard_reads_each_arm_from_its_own_namespace(self):
        """A guard that looked for the limits in the default namespace would compare the
        wrong files and pass a genuinely mixed set."""
        self.write(threshold_method(1000.0), self.BASE)
        self.write('mag_on', self.BASE, variant=REFERENCE_VARIANT)
        self.write('mag_on', self.RETUNED)          # the stale default-namespace copy
        groups = constants_disagreements(sweep_arms(np.array([1000.0])),
                                         subjects=['01'], activities=['walking'])
        self.assertEqual(len(groups), 1)

    def test_the_message_names_the_differing_values_and_an_example_arm(self):
        """The message is the whole point — it has to say what differs, or the reader just
        re-runs blindly and hits it again."""
        self.write('mag_adapt_th1000.00', self.BASE)
        self.write('mag_on', self.RETUNED)
        message = describe_disagreements(constants_disagreements(
            [('mag_on', None), ('mag_adapt_th1000.00', None)], subjects=['01'],
            activities=['walking']))
        self.assertIn('0.037', message)
        self.assertIn('0.018', message)
        self.assertIn('Subject01/walking/mag_on', message)


if __name__ == '__main__':
    unittest.main()
