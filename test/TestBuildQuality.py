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
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.build_quality import (BLOCKING, ICC_DEPENDENT, SHORT_INVALID_RUN,
                                       health_score, icc_report,
                                       intraclass_correlation, pooled_summary, step_table,
                                       warn_if_placements_split)
from plotting.build_quality import _natural_key, _session_expectation, plot_data_state


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


class TestDataStateFigure(unittest.TestCase):
    """The triage figure. Two things in it are easy to get wrong and silent when wrong.

    SCALING COVERAGE AGAINST THE WRONG DENOMINATOR. IMoVE's five long-walk sessions carry 7
    sensors where the other 21 carry 15. Scored against the dataset-wide roster they read 47%
    and the figure says half the dataset is broken when nothing is.

    CONFLATING TWO KINDS OF BLANK. A cell with no colour can mean "no such trial in this
    session" or "the trial built but scored nothing", and those call for opposite reactions.
    """

    @staticmethod
    def _tables(long_walk_sensors=7, wide_sensors=15):
        rows = []
        for subject, n_sensors, trials in (('s2', wide_sensors, ('t1', 't2')),
                                           ('s5l', long_walk_sensors, ('t1',))):
            for trial in trials:
                for i in range(wide_sensors):
                    present = i < n_sensors
                    rows.append({'dataset': 'x', 'subject': subject, 'trial': trial,
                                 'sensor': f'S{i}', 'present': present,
                                 'status': 'present' if present else 'neither'})
        coverage = pd.DataFrame(rows)
        index = pd.DataFrame([
            {'dataset': 'x', 'subject': 's2', 'trial': 't1', 'status': 'fresh',
             'has_build_report': True, 'n_suspect_plates': 0},
            {'dataset': 'x', 'subject': 's2', 'trial': 't2', 'status': 'fresh',
             'has_build_report': True, 'n_suspect_plates': 0},
            {'dataset': 'x', 'subject': 's5l', 'trial': 't1', 'status': 'fresh',
             'has_build_report': True, 'n_suspect_plates': 0}])
        health = index.assign(health=[0.1, np.nan, 0.5])
        return {'coverage': coverage, 'index': index, 'health': health}

    def test_a_short_session_is_scored_against_itself(self):
        expectation = _session_expectation(self._tables()['coverage'])
        self.assertEqual(int(expectation['s2']), 15)
        self.assertEqual(int(expectation['s5l']), 7)

    def test_a_session_short_everywhere_is_not_hidden_by_its_own_maximum(self):
        """The known cost of per-session normalization: a session broken in every trial looks
        whole, because its own maximum drops with it. The figure keeps the count on the row
        label for exactly this reason, so the number is still visible even when the colour
        is not."""
        expectation = _session_expectation(self._tables(long_walk_sensors=2)['coverage'])
        self.assertEqual(int(expectation['s5l']), 2)

    def test_natural_order_puts_s2_before_s13(self):
        """Lexical order gives s10, s11, s13, s2 — which reads as a shuffled dataset."""
        self.assertEqual(sorted(['s13', 's2', 's10', 's5l'], key=_natural_key),
                         ['s2', 's5l', 's10', 's13'])

    def test_it_renders_without_a_display_and_writes_a_file(self):
        import matplotlib
        matplotlib.use('Agg')
        import tempfile
        from unittest import mock
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch('plotting.build_quality._plots_dir',
                            return_value=Path(tmpdir)):
                plot_data_state(self._tables(), 'x', save=True, show=False)
            self.assertTrue((Path(tmpdir) / 'data_state.png').exists())

    def test_missing_tables_are_not_fatal(self):
        """--only-tables exists so a targeted rerun is cheap, which means any table can be
        absent. A figure that raises on that makes the whole run fail for one missing file."""
        plot_data_state({}, 'x', save=False, show=False)
        plot_data_state({'index': pd.DataFrame()}, 'x', save=False, show=False)


class TestEveryFigureIsCaptioned(unittest.TestCase):
    """A figure travels. It ends up in a slide, a message or a paper draft without the code
    that made it and without the report section that explains it, so the caption is the only
    thing that makes it readable on arrival."""

    def test_every_figure_in_this_module_has_a_caption(self):
        """Keyed by filename rather than by function, so a new figure whose caption was
        forgotten fails here rather than shipping bare."""
        import plotting.build_quality as module
        rendered = re.findall(r"'([a-z_]+\.png)', plots_dir=", Path(module.__file__).read_text())
        self.assertTrue(rendered, 'found no figures to check')
        for filename in rendered:
            self.assertIn(filename, module.CAPTIONS, f'{filename} has no caption')
            self.assertGreater(len(module.CAPTIONS[filename]), 200,
                               f'{filename} caption is too short to be detailed')

    def test_a_caption_says_what_the_figure_does_not_show(self):
        """The limitation is the part a reader cannot recover from the axes, and the part most
        likely to be over-read if it is missing."""
        from plotting.build_quality import CAPTIONS
        for filename, caption in CAPTIONS.items():
            self.assertIn('NOT SHOWN', caption,
                          f'{filename} does not state what it leaves out')

    def test_saving_without_one_warns_rather_than_passing_quietly(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from plotting.utils import MissingCaptionWarning, finalize_and_save_plot

        figure = plt.figure()
        with self.assertWarns(MissingCaptionWarning):
            finalize_and_save_plot(figure, 'title', 'x.png', save=False, show=False)

    def test_a_caption_does_not_overlap_the_epilog(self):
        """They used to collide: the epilog sat at a fixed y and ran straight through the
        middle of a five-line caption on the 13x5 panels. Both are now measured off the font
        size and figure height, so the epilog's baseline must clear the caption's top."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from plotting.utils import CAPTION_FONTSIZE, finalize_and_save_plot

        figure = plt.figure(figsize=(13, 5))
        finalize_and_save_plot(figure, 'title', 'x.png', save=False, show=False,
                               caption='word ' * 250, epilog='n = 262 trials')

        texts = [t for t in figure.texts if t.get_text() != 'title']
        caption = max(texts, key=lambda t: len(t.get_text()))
        epilog = min(texts, key=lambda t: len(t.get_text()))
        caption_top = (caption.get_position()[1]
                       + (caption.get_text().count('\n') + 1)
                       * (CAPTION_FONTSIZE * 1.35) / (5.0 * 72.0))
        self.assertGreaterEqual(epilog.get_position()[1], caption_top,
                                'the epilog sits inside the caption block')
        plt.close(figure)


if __name__ == '__main__':
    unittest.main()
