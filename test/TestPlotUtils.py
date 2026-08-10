"""
Covers plotting/utils.py — the shared statistics behind every figure: outlier
rejection, the block reduction that sets n, Holm correction, Wilcoxon effect sizes,
and the two-stage correction family that decides which significance brackets get
drawn.

This is where a silent error is most expensive, because the output is a p-value and a
bracket in a paper. Specifically: n is set by block_pivot's averaging (get the block
definition wrong and the whole figure overstates its precision), Holm is applied
twice across a family whose membership is decided by which panels survive the first
stage, and adjusted p-values are only assigned to survivors — so an unadjusted
p-value leaking into a bracket would look exactly like a real result.

Pure dataframe/array work; the drawing functions are not exercised.
"""
import os
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DISABLE_TQDM", "True")

import numpy as np
import pandas as pd
import scipy.stats as stats

from plotting.utils import (DEFAULT_BLOCK_COLS, PanelTest, _rank_biserial, block_pivot,
                            holm_bonferroni, order_present,
                            run_statistical_analysis, significance_report, test_panel,
                            test_panels)

METRIC = 'rmse_rad'
JOINTS = ['Lumbar', 'Hip', 'Knee', 'Ankle']


def panel_frame(offsets: dict, n_subjects: int = 11, joints=None, seed: int = 0,
                noise: float = 0.002, trial_types=('walking',)) -> pd.DataFrame:
    """One row per (subject, joint, activity, method). `offsets` maps method -> its
    mean metric value, so a method's advantage over another is exactly the difference
    of their offsets, consistent across every block."""
    joints = JOINTS if joints is None else joints
    rng = np.random.default_rng(seed)
    rows = []
    for subject_idx in range(n_subjects):
        for joint in joints:
            for trial_type in trial_types:
                # A per-block level, so blocks differ from each other but the method
                # ordering within a block stays consistent — the Friedman setup.
                block_level = rng.normal(scale=0.05)
                for method, offset in offsets.items():
                    rows.append({
                        'subject': f'Subject{subject_idx + 1:02d}',
                        'joint_name': joint,
                        'trial_type': trial_type,
                        'method': method,
                        METRIC: offset + block_level + rng.normal(scale=noise),
                    })
    return pd.DataFrame(rows)


class TestOrderPresent(unittest.TestCase):
    def test_it_follows_the_requested_order_not_the_data_order(self):
        self.assertEqual(order_present(['Knee', 'Lumbar', 'Ankle'], JOINTS),
                         ['Lumbar', 'Knee', 'Ankle'])

    def test_missing_values_are_dropped_and_extras_ignored(self):
        self.assertEqual(order_present(['Knee', 'Elbow'], JOINTS), ['Knee'])

    def test_empty_inputs(self):
        self.assertEqual(order_present([], JOINTS), [])
        self.assertEqual(order_present(JOINTS, []), [])

    def test_duplicates_in_the_data_do_not_duplicate_the_output(self):
        self.assertEqual(order_present(['Knee', 'Knee'], JOINTS), ['Knee'])


class TestHolmBonferroni(unittest.TestCase):
    def test_a_hand_worked_example(self):
        """p = [.01, .04, .03]: sorted (.01, .03, .04) x (3, 2, 1) = (.03, .06, .04),
        running max makes it (.03, .06, .06), then back into input order."""
        np.testing.assert_allclose(holm_bonferroni([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_a_single_p_value_is_unchanged(self):
        np.testing.assert_allclose(holm_bonferroni([0.023]), [0.023])

    def test_it_is_step_down_not_plain_bonferroni(self):
        """Bonferroni would multiply every p by 3; Holm only multiplies the smallest by
        3. If these ever agreed on the largest p, the correction would have silently
        become the more conservative one."""
        holm = holm_bonferroni([0.01, 0.02, 0.03])
        self.assertAlmostEqual(holm[0], 0.03)
        self.assertLess(holm[2], 0.03 * 3)

    def test_output_order_matches_input_order(self):
        p_values = [0.5, 0.001, 0.2, 0.04]
        adjusted = holm_bonferroni(p_values)
        by_rank = holm_bonferroni(sorted(p_values))
        np.testing.assert_allclose(np.sort(adjusted), np.sort(by_rank))
        self.assertEqual(int(np.argmin(adjusted)), int(np.argmin(p_values)))

    def test_it_never_lowers_a_p_value(self):
        rng = np.random.default_rng(3)
        p_values = rng.uniform(size=25)
        self.assertTrue(np.all(holm_bonferroni(p_values) >= p_values - 1e-15))

    def test_it_is_monotonic_in_the_raw_p_values(self):
        p_values = np.array([0.001, 0.01, 0.02, 0.3, 0.8])
        adjusted = holm_bonferroni(p_values)
        self.assertTrue(np.all(np.diff(adjusted) >= -1e-15))

    def test_it_is_capped_at_one(self):
        adjusted = holm_bonferroni([0.4, 0.6, 0.9])
        self.assertTrue(np.all(adjusted <= 1.0))

    def test_an_empty_family_is_handled(self):
        self.assertEqual(len(holm_bonferroni([])), 0)

    def test_ties_are_not_over_corrected(self):
        adjusted = holm_bonferroni([0.02, 0.02, 0.02])
        np.testing.assert_allclose(adjusted, [0.06, 0.06, 0.06])


class TestRankBiserial(unittest.TestCase):
    def test_a_clean_win_is_plus_one(self):
        self.assertAlmostEqual(_rank_biserial(np.array([2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0])), 1.0)

    def test_a_clean_loss_is_minus_one(self):
        self.assertAlmostEqual(_rank_biserial(np.array([1.0, 1.0, 1.0]), np.array([2.0, 3.0, 4.0])), -1.0)

    def test_it_is_antisymmetric(self):
        rng = np.random.default_rng(11)
        x, y = rng.normal(size=30), rng.normal(size=30)
        self.assertAlmostEqual(_rank_biserial(x, y), -_rank_biserial(y, x))

    def test_identical_inputs_are_zero_not_nan(self):
        """Every difference is an exact tie and gets dropped, leaving nothing to rank.
        The guard has to return 0.0 rather than divide by an empty sum."""
        x = np.array([1.0, 2.0, 3.0])
        self.assertEqual(_rank_biserial(x, x), 0.0)

    def test_a_balanced_split_is_near_zero(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 1.0, 4.0, 3.0])
        self.assertAlmostEqual(_rank_biserial(x, y), 0.0)

    def test_it_reflects_rank_not_magnitude(self):
        """One huge win against three small losses is still a negative effect — that is
        the point of a rank statistic, and the reason it travels with a mean difference
        in the report."""
        x = np.array([100.0, 1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 2.0, 2.0])
        self.assertLess(_rank_biserial(x, y), 0.0)

    def test_it_stays_within_range(self):
        rng = np.random.default_rng(5)
        for seed_shift in range(5):
            x = rng.normal(size=40)
            y = rng.normal(loc=0.3 * seed_shift, size=40)
            effect = _rank_biserial(x, y)
            self.assertGreaterEqual(effect, -1.0)
            self.assertLessEqual(effect, 1.0)


class TestBlockPivot(unittest.TestCase):
    def test_one_row_per_block_and_one_column_per_group(self):
        df = panel_frame({'mag_on': 0.10, 'mag_off': 0.20, 'ekf': 0.30})
        pivot = block_pivot(df, METRIC, 'method')
        self.assertEqual(pivot.shape, (11 * len(JOINTS), 3))
        self.assertEqual(sorted(pivot.columns), ['ekf', 'mag_off', 'mag_on'])

    def test_dimensions_outside_the_block_are_averaged_into_it(self):
        """Both activities are the same sensors in the same places, so they are averaged
        rather than counted as two blocks. Counting them would inflate n by ~1.5x."""
        df = pd.DataFrame({
            'subject': ['Subject01'] * 4,
            'joint_name': ['Knee'] * 4,
            'trial_type': ['walking', 'walking', 'complexTasks', 'complexTasks'],
            'method': ['mag_on', 'mag_off', 'mag_on', 'mag_off'],
            METRIC: [0.10, 0.20, 0.30, 0.60],
        })
        pivot = block_pivot(df, METRIC, 'method')
        self.assertEqual(len(pivot), 1)
        self.assertAlmostEqual(pivot['mag_on'].iloc[0], 0.20)
        self.assertAlmostEqual(pivot['mag_off'].iloc[0], 0.40)

    def test_the_default_block_is_subject_crossed_with_joint(self):
        df = panel_frame({'mag_on': 0.1, 'mag_off': 0.2}, n_subjects=3, joints=['Knee', 'Hip'])
        self.assertEqual(DEFAULT_BLOCK_COLS, ['subject', 'joint_name'])
        self.assertEqual(len(block_pivot(df, METRIC, 'method')), 3 * 2)

    def test_blocking_on_subject_alone_collapses_the_joints(self):
        df = panel_frame({'mag_on': 0.1, 'mag_off': 0.2}, n_subjects=3, joints=['Knee', 'Hip'])
        self.assertEqual(len(block_pivot(df, METRIC, 'method', block_cols=['subject'])), 3)

    def test_incomplete_blocks_are_dropped(self):
        """Friedman needs every group present in every block. A block missing one method
        has to go, not be silently filled."""
        df = panel_frame({'mag_on': 0.1, 'mag_off': 0.2, 'ekf': 0.3}, n_subjects=4,
                         joints=['Knee'])
        df = df.drop(df[(df['subject'] == 'Subject01') & (df['method'] == 'ekf')].index)
        pivot = block_pivot(df, METRIC, 'method')
        self.assertEqual(len(pivot), 3)
        self.assertNotIn('Subject01_Knee', pivot.index)

    def test_it_returns_empty_when_no_block_column_is_present(self):
        df = pd.DataFrame({'method': ['a', 'b'], METRIC: [1.0, 2.0]})
        self.assertTrue(block_pivot(df, METRIC, 'method').empty)

    def test_a_missing_block_column_is_tolerated_if_another_remains(self):
        df = panel_frame({'mag_on': 0.1, 'mag_off': 0.2}, n_subjects=3,
                         joints=['Knee']).drop(columns=['joint_name'])
        self.assertEqual(len(block_pivot(df, METRIC, 'method')), 3)


class TestTestPanel(unittest.TestCase):
    def test_a_consistent_ordering_is_detected(self):
        df = panel_frame({'mag_on': 0.20, 'mag_off': 0.30, 'ekf': 0.40})
        result = test_panel(df, METRIC, 'method', ['mag_on', 'mag_off', 'ekf'], key='all')

        self.assertIsNone(result.skipped_reason)
        self.assertEqual(result.n_blocks, 11 * len(JOINTS))
        self.assertLess(result.friedman_p, 1e-6)
        self.assertGreater(result.kendalls_w, 0.9)
        self.assertEqual(len(result.pairs), 3)
        self.assertTrue(all(p < 1e-4 for p in result.pair_p_raw))

    def test_n_is_subjects_times_joints_however_many_activities_there_are(self):
        """The design effect the block definition exists to avoid: walking and
        complexTasks are the same sensors on the same people, so adding the second
        activity must not double n and halve the standard error."""
        one = test_panel(panel_frame({'a': 0.2, 'b': 0.3, 'c': 0.4}, trial_types=('walking',)),
                         METRIC, 'method', ['a', 'b', 'c'])
        both = test_panel(panel_frame({'a': 0.2, 'b': 0.3, 'c': 0.4},
                                      trial_types=('walking', 'complexTasks')),
                          METRIC, 'method', ['a', 'b', 'c'])
        self.assertEqual(one.n_blocks, 11 * len(JOINTS))
        self.assertEqual(both.n_blocks, one.n_blocks)

    def test_effect_sign_and_mean_difference_follow_the_group_order(self):
        """mag_on is lower than mag_off by 0.10, so (mag_on, mag_off) must come out
        negative in both the effect size and the mean difference. A sign flip here would
        reverse every claim the figure makes about which method is better."""
        df = panel_frame({'mag_on': 0.20, 'mag_off': 0.30, 'ekf': 0.40})
        result = test_panel(df, METRIC, 'method', ['mag_on', 'mag_off', 'ekf'])
        idx = result.pairs.index(('mag_on', 'mag_off'))
        self.assertAlmostEqual(result.pair_effect[idx], -1.0)
        self.assertAlmostEqual(result.pair_mean_diff[idx], -0.10, places=2)

    def test_the_mean_difference_is_in_the_metric_s_own_units(self):
        df = panel_frame({'a': 0.10, 'b': 0.13, 'c': 0.40})
        result = test_panel(df, METRIC, 'method', ['a', 'b', 'c'])
        idx = result.pairs.index(('a', 'b'))
        self.assertAlmostEqual(result.pair_mean_diff[idx], -0.03, places=3)

    def test_groups_are_reported_in_the_requested_order(self):
        df = panel_frame({'a': 0.1, 'b': 0.2, 'c': 0.3})
        result = test_panel(df, METRIC, 'method', ['c', 'a', 'b'])
        self.assertEqual(result.groups, ['c', 'a', 'b'])
        self.assertEqual(result.pairs, [('c', 'a'), ('c', 'b'), ('a', 'b')])

    def test_groups_absent_from_the_data_are_dropped(self):
        df = panel_frame({'a': 0.1, 'b': 0.2, 'c': 0.3})
        result = test_panel(df, METRIC, 'method', ['a', 'b', 'c', 'nonexistent'])
        self.assertEqual(result.groups, ['a', 'b', 'c'])

    def test_it_returns_raw_p_values(self):
        """test_panel must NOT correct anything — correction is test_panels' job, across
        the whole figure. A panel that pre-corrected would be corrected twice."""
        df = panel_frame({'a': 0.20, 'b': 0.30, 'c': 0.40})
        result = test_panel(df, METRIC, 'method', ['a', 'b', 'c'])
        self.assertTrue(np.isnan(result.friedman_p_adj))
        self.assertEqual(result.pair_p_adj, [])
        self.assertEqual(result.significant_pairs, [])

    def test_two_groups_is_skipped_with_a_reason(self):
        """scipy's Friedman needs three or more groups. The skip has to be reported, not
        swallowed into a nan that reads as 'not significant'."""
        df = panel_frame({'a': 0.1, 'b': 0.2})
        result = test_panel(df, METRIC, 'method', ['a', 'b'])
        self.assertIsNotNone(result.skipped_reason)

    def test_too_few_blocks_is_skipped_with_a_reason(self):
        df = panel_frame({'a': 0.1, 'b': 0.2, 'c': 0.3}, n_subjects=1, joints=['Knee'])
        result = test_panel(df, METRIC, 'method', ['a', 'b', 'c'])
        self.assertEqual(result.n_blocks, 1)
        self.assertIsNotNone(result.skipped_reason)
        self.assertIn('blocks', result.skipped_reason)

    def test_one_group_is_skipped_with_a_reason(self):
        df = panel_frame({'a': 0.1})
        result = test_panel(df, METRIC, 'method', ['a'])
        self.assertIsNotNone(result.skipped_reason)

    def test_identical_groups_give_a_nan_p_value_not_a_small_one(self):
        """Three copies of the same column: every block is a perfect tie, so scipy's tie
        correction divides by zero and the p-value comes back nan (with a RuntimeWarning)
        rather than through the ValueError path that sets skipped_reason. Documented
        because the failure is only safe by luck downstream — see the companion test in
        TestTestPanels that nan never clears the alpha gate."""
        base = panel_frame({'a': 0.2}, noise=0.0)
        df = pd.concat([base.assign(method=m) for m in ('a', 'b', 'c')], ignore_index=True)
        with np.errstate(all='ignore'):
            result = test_panel(df, METRIC, 'method', ['a', 'b', 'c'])
        self.assertTrue(np.isnan(result.friedman_p))
        self.assertTrue(all(effect == 0.0 for effect in result.pair_effect))
        self.assertTrue(all(mean_diff == 0.0 for mean_diff in result.pair_mean_diff))


class TestTestPanels(unittest.TestCase):
    def setUp(self):
        self.panels = {
            'Knee': panel_frame({'a': 0.20, 'b': 0.30, 'c': 0.40}, joints=['Knee'], seed=1),
            'Hip': panel_frame({'a': 0.20, 'b': 0.20, 'c': 0.20}, joints=['Hip'], seed=2,
                               noise=0.05),
        }
        self.results = test_panels(self.panels, METRIC, 'method', ['a', 'b', 'c'])

    def test_a_real_effect_survives_both_correction_stages(self):
        knee = self.results['Knee']
        self.assertLess(knee.friedman_p_adj, 0.05)
        self.assertEqual(len(knee.pair_p_adj), 3)
        self.assertEqual(sorted(knee.significant_pairs), sorted(knee.pairs))

    def test_a_panel_failing_the_omnibus_gets_no_pairwise_verdicts(self):
        """The gate that stops a figure from drawing brackets off uncorrected pairwise
        p-values in a panel with no overall effect."""
        hip = self.results['Hip']
        self.assertGreater(hip.friedman_p_adj, 0.05)
        self.assertEqual(hip.pair_p_adj, [])
        self.assertEqual(hip.significant_pairs, [])

    def test_the_omnibus_family_is_the_set_of_panels(self):
        raw = [self.results[key].friedman_p for key in ('Knee', 'Hip')]
        adjusted = [self.results[key].friedman_p_adj for key in ('Knee', 'Hip')]
        np.testing.assert_allclose(adjusted, holm_bonferroni(raw))

    def test_the_pairwise_family_pools_only_the_surviving_panels(self):
        """Two panels with a real effect share one pairwise family of six, so each p is
        corrected against six comparisons and not three."""
        panels = {
            'Knee': panel_frame({'a': 0.20, 'b': 0.30, 'c': 0.40}, joints=['Knee'], seed=1),
            'Hip': panel_frame({'a': 0.20, 'b': 0.32, 'c': 0.45}, joints=['Hip'], seed=2),
        }
        results = test_panels(panels, METRIC, 'method', ['a', 'b', 'c'])
        pooled_raw = [p for key in panels for p in results[key].pair_p_raw]
        pooled_adj = [p for key in panels for p in results[key].pair_p_adj]
        self.assertEqual(len(pooled_adj), 6)
        np.testing.assert_allclose(pooled_adj, holm_bonferroni(pooled_raw))

    def test_a_single_panel_family_reduces_to_plain_holm_on_its_pairs(self):
        panels = {'only': panel_frame({'a': 0.20, 'b': 0.30, 'c': 0.40}, seed=4)}
        result = test_panels(panels, METRIC, 'method', ['a', 'b', 'c'])['only']
        self.assertAlmostEqual(result.friedman_p_adj, result.friedman_p)
        np.testing.assert_allclose(result.pair_p_adj, holm_bonferroni(result.pair_p_raw))

    def test_correction_only_ever_raises_a_p_value(self):
        for result in self.results.values():
            if result.skipped_reason is None:
                self.assertGreaterEqual(result.friedman_p_adj, result.friedman_p - 1e-15)
                for raw, adjusted in zip(result.pair_p_raw, result.pair_p_adj):
                    self.assertGreaterEqual(adjusted, raw - 1e-15)

    def test_skipped_panels_do_not_join_the_family(self):
        """A panel too small to test must not consume correction budget from the panels
        that were testable — that would inflate their adjusted p-values."""
        panels = {
            'good': panel_frame({'a': 0.20, 'b': 0.30, 'c': 0.40}, joints=['Knee'], seed=1),
            'tiny': panel_frame({'a': 0.2, 'b': 0.3, 'c': 0.4}, n_subjects=1, joints=['Hip']),
        }
        results = test_panels(panels, METRIC, 'method', ['a', 'b', 'c'])
        self.assertIsNotNone(results['tiny'].skipped_reason)
        self.assertAlmostEqual(results['good'].friedman_p_adj, results['good'].friedman_p)

    def test_all_panels_skipped_returns_the_skips_untouched(self):
        panels = {'tiny': panel_frame({'a': 0.2, 'b': 0.3, 'c': 0.4}, n_subjects=1,
                                      joints=['Hip'])}
        results = test_panels(panels, METRIC, 'method', ['a', 'b', 'c'])
        self.assertIsNotNone(results['tiny'].skipped_reason)
        self.assertTrue(np.isnan(results['tiny'].friedman_p_adj))

    def test_every_panel_appears_in_the_results(self):
        self.assertEqual(sorted(self.results), sorted(self.panels))

    def test_a_nan_omnibus_p_value_never_clears_the_gate(self):
        """The companion to the all-ties case in TestTestPanel: a fully-tied panel comes
        back with friedman_p = nan and no skip reason, so the only thing standing between
        it and a set of drawn brackets is that `nan < alpha` is False. Pinned here
        because a refactor to `not (p >= alpha)` would silently open that door."""
        base = panel_frame({'a': 0.2}, noise=0.0)
        tied = pd.concat([base.assign(method=m) for m in ('a', 'b', 'c')], ignore_index=True)
        with np.errstate(all='ignore'):
            results = test_panels({'tied': tied}, METRIC, 'method', ['a', 'b', 'c'])
        self.assertEqual(results['tied'].significant_pairs, [])
        self.assertEqual(results['tied'].pair_p_adj, [])


class TestSignificanceReport(unittest.TestCase):
    def setUp(self):
        self.panels = {
            'Knee': panel_frame({'a': 0.20, 'b': 0.30, 'c': 0.40}, joints=['Knee'], seed=1),
            'Hip': panel_frame({'a': 0.20, 'b': 0.20, 'c': 0.20}, joints=['Hip'], seed=2,
                               noise=0.05),
        }
        self.results = test_panels(self.panels, METRIC, 'method', ['a', 'b', 'c'])
        self.report = significance_report(self.results, METRIC)

    def test_it_reports_both_the_raw_and_the_corrected_p_value(self):
        """Both have to be citable. Reporting only the adjusted one hides the family
        size; only the raw one overstates the result."""
        for column in ('wilcoxon_p', 'wilcoxon_p_holm', 'friedman_p', 'friedman_p_holm'):
            self.assertIn(column, self.report.columns)

    def test_it_reports_n_and_an_effect_size_alongside_the_p_value(self):
        for column in ('n_blocks', 'rank_biserial', 'mean_diff', 'kendalls_w'):
            self.assertIn(column, self.report.columns)
        knee = self.report[self.report['panel'] == 'Knee']
        self.assertTrue((knee['n_blocks'] == 11).all())

    def test_one_row_per_comparison_per_panel(self):
        self.assertEqual(len(self.report[self.report['panel'] == 'Knee']), 3)
        self.assertEqual(len(self.report[self.report['panel'] == 'Hip']), 3)

    def test_the_significant_flag_agrees_with_the_corrected_p_value(self):
        for _, row in self.report.iterrows():
            if not np.isnan(row.get('wilcoxon_p_holm', np.nan)):
                self.assertEqual(bool(row['significant']), row['wilcoxon_p_holm'] < 0.05)

    def test_a_panel_that_failed_the_omnibus_is_flagged_insignificant(self):
        hip = self.report[self.report['panel'] == 'Hip']
        self.assertFalse(hip['significant'].any())
        self.assertTrue(hip['wilcoxon_p_holm'].isna().all())

    def test_a_skipped_panel_is_reported_with_its_reason(self):
        results = {'tiny': PanelTest(key='tiny', groups=[], n_blocks=0,
                                     skipped_reason='need >=2 blocks')}
        report = significance_report(results, METRIC)
        self.assertEqual(len(report), 1)
        self.assertEqual(report.iloc[0]['note'], 'need >=2 blocks')


class TestRunStatisticalAnalysis(unittest.TestCase):
    def test_it_matches_a_one_panel_test_panels_family(self):
        df = panel_frame({'a': 0.20, 'b': 0.30, 'c': 0.40}, seed=6)
        pairs = run_statistical_analysis(df, METRIC, 'method', ['a', 'b', 'c'])
        expected = test_panels({'': df}, METRIC, 'method', ['a', 'b', 'c'])[''].significant_pairs
        self.assertEqual(pairs, expected)

    def test_it_finds_nothing_when_there_is_nothing_to_find(self):
        df = panel_frame({'a': 0.20, 'b': 0.20, 'c': 0.20}, seed=7, noise=0.05)
        self.assertEqual(run_statistical_analysis(df, METRIC, 'method', ['a', 'b', 'c']), [])

    def test_alpha_is_honoured(self):
        df = panel_frame({'a': 0.20, 'b': 0.30, 'c': 0.40}, seed=8)
        self.assertEqual(run_statistical_analysis(df, METRIC, 'method', ['a', 'b', 'c'],
                                                  alpha=0.0), [])


class TestAgainstScipyDirectly(unittest.TestCase):
    """One end-to-end anchor: the pipeline's p-value for a panel must be the p-value a
    reader would get by calling scipy on the same block table by hand."""

    def test_the_friedman_and_wilcoxon_p_values_are_scipy_s(self):
        df = panel_frame({'a': 0.20, 'b': 0.24, 'c': 0.26}, seed=9, noise=0.03)
        pivot = block_pivot(df, METRIC, 'method')
        result = test_panel(df, METRIC, 'method', ['a', 'b', 'c'])

        _, expected_friedman = stats.friedmanchisquare(pivot['a'], pivot['b'], pivot['c'])
        self.assertAlmostEqual(result.friedman_p, float(expected_friedman), places=12)

        for (g1, g2), reported in zip(result.pairs, result.pair_p_raw):
            _, expected = stats.wilcoxon(pivot[g1], pivot[g2], alternative='two-sided',
                                         zero_method='zsplit')
            self.assertAlmostEqual(reported, float(expected), places=12)


if __name__ == '__main__':
    unittest.main()
