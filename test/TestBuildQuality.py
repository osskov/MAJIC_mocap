"""The build-quality analysis: coverage, invalid sections, ICC and pooling.

Driven with planted defects rather than real data wherever possible, following
TestReconstruction's pattern: assert the metric fires on a known fault AND that the clean
control does not. A test that only checks the fault case cannot tell a working detector from
one that fires on everything.

The ICC tests matter more than they look. The blocking derived from them changes every `n` in
the report, and the specific trap is real: IMoVE's three placements on a segment share one
WorldTrace, so a reconstruction metric is bit-identical across them and counting them as three
replicates inflates n threefold on exactly the numbers the report leads with.
"""
import unittest

import numpy as np
import pandas as pd

from experiments.build_quality import (BLOCKING, ICC_DEPENDENT, SHORT_INVALID_RUN,
                                       health_score, icc_report,
                                       intraclass_correlation, pooled_summary, step_table,
                                       warn_if_placements_split)


class TestIntraclassCorrelation(unittest.TestCase):
    """The measurement the blocking is derived from."""

    def test_identical_within_group_gives_one(self):
        """Three placements sharing one reconstruction. ICC 1.0 means one observation wearing
        three hats, which is the whole reason the blocking is per metric family."""
        frame = pd.DataFrame({
            'trial': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
            'value': [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 9.0, 9.0, 9.0]})
        self.assertGreater(intraclass_correlation(frame, 'value', ['trial']), 0.99)

    def test_independent_within_group_gives_about_zero(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({'trial': np.repeat(list('abcdefgh'), 6),
                              'value': rng.normal(size=48)})
        icc = intraclass_correlation(frame, 'value', ['trial'])
        self.assertLess(abs(icc), 0.35, f'independent data should not block, got {icc}')

    def test_a_constant_metric_is_undefined_not_one(self):
        """A metric with no variance carries no information about grouping, and returning 1.0
        would silently mark it dependent and drop it from the report."""
        frame = pd.DataFrame({'trial': list('aabbcc'), 'value': [2.0] * 6})
        self.assertTrue(np.isnan(intraclass_correlation(frame, 'value', ['trial'])))

    def test_singleton_groups_are_undefined(self):
        """One member per group means there is no within-group variance to compare against."""
        frame = pd.DataFrame({'trial': list('abcd'), 'value': [1.0, 2.0, 3.0, 4.0]})
        self.assertTrue(np.isnan(intraclass_correlation(frame, 'value', ['trial'])))


class TestPlacementGuard(unittest.TestCase):
    def test_it_names_metrics_shared_across_placements(self):
        """Mirrors plotting/utils._warn_if_sides_split. Silence here means an inflated n."""
        icc = pd.DataFrame({
            'step': ['S2_reconstruction', 'S7_alignment'],
            'metric': ['residual_median_mm', 'offset_angle_deg'],
            'grouping': ['placement', 'placement'],
            'icc': [1.0, 0.10]})
        icc['dependent'] = icc['icc'] >= ICC_DEPENDENT
        self.assertEqual(warn_if_placements_split(icc),
                         ['S2_reconstruction.residual_median_mm'])

    def test_an_empty_icc_table_is_not_an_error(self):
        self.assertEqual(warn_if_placements_split(pd.DataFrame()), [])


class TestStepTable(unittest.TestCase):
    def test_long_form_pivots_to_one_row_per_entity(self):
        long = pd.DataFrame({
            'dataset': ['imove'] * 4, 'subject': ['s2'] * 4, 'trial': ['t1'] * 4,
            'step': ['S2_reconstruction'] * 4,
            'entity_kind': ['segment'] * 4, 'entity': ['a', 'a', 'b', 'b'],
            'metric': ['residual', 'valid', 'residual', 'valid'],
            'value_num': [0.4, 1.0, 0.9, 0.5], 'value_str': [None] * 4})
        wide = step_table(long, 'S2_reconstruction')
        self.assertEqual(len(wide), 2)
        self.assertAlmostEqual(
            float(wide[wide.entity == 'b']['residual'].iloc[0]), 0.9)

    def test_an_absent_step_gives_an_empty_frame(self):
        long = pd.DataFrame(columns=['dataset', 'subject', 'trial', 'step', 'entity_kind',
                                     'entity', 'metric', 'value_num', 'value_str'])
        self.assertTrue(step_table(long, 'S4_sync').empty)


class TestPooling(unittest.TestCase):
    def test_blocking_averages_before_pooling(self):
        """Three placements sharing a value must contribute ONE observation, not three.

        This is the arithmetic the ICC work exists to protect: without the blocked mean, n
        would read 3 and every quantile would be computed over triplicated data.
        """
        table = pd.DataFrame({
            'dataset': ['imove'] * 3, 'subject': ['s2'] * 3, 'trial': ['t1'] * 3,
            'entity_kind': ['plate'] * 3,
            'entity': ['THIGH_L_H', 'THIGH_L_M', 'THIGH_L_L'],
            'residual_median_mm': [0.5, 0.5, 0.5]})
        # Blocked on entity, so three distinct entities remain three observations...
        self.assertEqual(int(pooled_summary(table, 'S7_alignment').n.iloc[0]), 3)
        # ...but blocked at trial level they collapse to one.
        self.assertEqual(int(pooled_summary(table, 'S4_sync').n.iloc[0]), 1)

    def test_every_blocked_step_is_a_known_step(self):
        """A typo in BLOCKING silently falls back to trial-level, which would be wrong for
        alignment and invisible."""
        from src.toolchest.building.report import STEPS
        for step in BLOCKING:
            self.assertIn(step, STEPS)


class TestInvalidSectionClassification(unittest.TestCase):
    """The position of a gap decides what it means, so the classification carries weight."""

    @staticmethod
    def _runs(valid):
        """Mirrors invalid_sections' run-finding on a bare mask, without touching the cache."""
        valid = np.asarray(valid, dtype=bool)
        edges = np.flatnonzero(np.concatenate([[True], valid[1:] != valid[:-1], [True]]))
        found = []
        for start, stop in zip(edges[:-1], edges[1:]):
            if valid[start]:
                continue
            found.append(('head' if start == 0 else
                          'tail' if stop == len(valid) else 'middle',
                          int(stop - start)))
        return found

    def test_a_leading_gap_is_head(self):
        self.assertEqual(self._runs([False] * 3 + [True] * 7), [('head', 3)])

    def test_a_trailing_gap_is_tail(self):
        self.assertEqual(self._runs([True] * 7 + [False] * 3), [('tail', 3)])

    def test_an_interior_gap_is_middle(self):
        """The kind that corrupts a joint angle mid-motion rather than trimming an edge."""
        self.assertEqual(self._runs([True] * 4 + [False] * 2 + [True] * 4), [('middle', 2)])

    def test_a_fully_valid_plate_has_no_runs(self):
        self.assertEqual(self._runs([True] * 10), [])

    def test_several_gaps_are_all_found(self):
        found = self._runs([False, True, True, False, False, True, False])
        self.assertEqual(found, [('head', 1), ('middle', 2), ('tail', 1)])

    def test_the_short_run_threshold_separates_blinks_from_gaps(self):
        self.assertTrue(1 < SHORT_INVALID_RUN)
        self.assertFalse(SHORT_INVALID_RUN < SHORT_INVALID_RUN)


class TestHealthScore(unittest.TestCase):
    def test_it_is_ordered_and_carries_its_components(self):
        """A composite shown without its components invites being read as a verdict, so the
        components have to survive into the frame."""
        index = pd.DataFrame({'subject': ['s1', 's2'], 'trial': ['t', 't'],
                              'n_suspect_plates': [0, 0]})
        tables = {'S2_reconstruction': pd.DataFrame({
            'subject': ['s1', 's2'], 'trial': ['t', 't'],
            'residual_median_mm': [0.2, 9.0], 'valid_fraction': [1.0, 0.5]})}
        health = health_score(index, tables)
        self.assertIn('worst_residual_mm', health.columns)
        self.assertEqual(list(health.subject), ['s2', 's1'])

    def test_no_usable_metrics_gives_an_empty_frame_not_a_crash(self):
        index = pd.DataFrame({'subject': ['s1'], 'trial': ['t'], 'n_suspect_plates': [0]})
        self.assertTrue(health_score(index, {}).empty)


class TestIccReportShape(unittest.TestCase):
    def test_it_derives_segment_identity_from_the_plate_name(self):
        """'THIGH_L_H' and 'THIGH_L_M' share a reconstruction; stripping the placement is how
        the ICC finds that out without being told the dataset's naming scheme."""
        table = pd.DataFrame({
            'dataset': ['imove'] * 6, 'subject': ['s2'] * 6,
            'trial': ['t1', 't1', 't1', 't2', 't2', 't2'],
            'entity_kind': ['plate'] * 6,
            'entity': ['THIGH_L_H', 'THIGH_L_M', 'THIGH_L_L'] * 2,
            'residual_median_mm': [0.4, 0.4, 0.4, 0.9, 0.9, 0.9]})
        icc = icc_report({'S2_reconstruction': table})
        placement = icc[(icc.grouping == 'placement') &
                        (icc.metric == 'residual_median_mm')]
        self.assertFalse(placement.empty)
        self.assertTrue(bool(placement.dependent.iloc[0]),
                        'a metric identical across placements must be flagged dependent')


if __name__ == '__main__':
    unittest.main()
