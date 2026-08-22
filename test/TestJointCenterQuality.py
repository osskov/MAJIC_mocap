"""The joint-centre quality report: does it say the true thing about synthetic tables?

A REPORT'S FAILURE MODE IS SILENT WRONGNESS. It cannot crash into a wrong number the way a
computation can — it renders whatever it was handed, and an em dash where a number belongs, or a
dataset quietly missing from a table, reads exactly like a dataset with nothing to report. So
every test here plants a known state and asserts on the RENDERED MARKDOWN, not on an intermediate
frame: the string is what a reader acts on.

The two cases that matter most are the ones that look alike and are not:

  * NOT MEASURED     — the experiment has not been run. Actionable, and the report must print the
                       command.
  * RAN, FITTED NOTHING — the experiment ran over every trial and the recordings could not clear
                       the valid-frame floor. NOT actionable, and telling a reader to re-run is
                       telling them to waste an hour.

Tables are synthesized rather than loaded. The report reads eight parquets per dataset across a
tree that takes an hour to produce, and a test that needs that tree is a test that does not run.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from experiments.global_assumptions import DATASETS, get_dataset
from experiments.joint_center import MIN_FIT_FRAMES
from plotting import joint_center_quality as report


def _fits(dataset: str, joints, n_subjects: int = 3, converged: bool = True,
          n_valid: int = 5000) -> pd.DataFrame:
    """A `joint_fits` table with two trials per subject, so a between-trial spread exists.

    The offsets are deliberately NOT identical between a subject's trials: `offset_stability`
    returns nothing for a subject whose fits agree exactly, and a report tested only on that case
    would pass with its error-bar column empty.
    """
    rows = []
    for s in range(n_subjects):
        for t, jitter in enumerate((0.0, 0.004)):
            for joint in joints:
                rows.append({
                    'subject': f's{s}', 'trial': f't{t}', 'joint': joint, 'dataset': dataset,
                    'converged': converged, 'n_valid': n_valid,
                    'parent_x': 0.05 + jitter, 'parent_y': 0.10, 'parent_z': 0.0,
                    'child_x': -0.05, 'child_y': -0.10 - jitter, 'child_z': 0.0,
                    'parent_norm_mm': 111.8, 'child_norm_mm': 111.8,
                    'residual_rms_mm': 8.0, 'residual_holdout_mm': 9.0,
                    'unmasked_shift_mm': 4.0, 'excitation': 0.006,
                    'condition_number': 300.0, 'gain_median': 12.0, 'gain_p95': 30.0,
                })
    return pd.DataFrame(rows)


def _closure(fits: pd.DataFrame, sources=('own', 'own_holdout', 'cross_trial')) -> pd.DataFrame:
    """A `closure_by_source` table consistent with `fits` — same joint-trials, one row per source."""
    rows = []
    for _, row in fits.iterrows():
        for i, source in enumerate(sources):
            rows.append({'subject': row['subject'], 'trial': row['trial'], 'joint': row['joint'],
                         'source': source, 'in_sample': source == 'own',
                         'closure_mm': 8.0 + 3.0 * i,
                         'offset_from_own_mm': 0.0 if source == 'own' else 20.0 * i})
    return pd.DataFrame(rows)


def _tables(dataset: str, joints, **overrides) -> dict:
    fits = overrides.pop('fits', None)
    if fits is None:
        fits = _fits(dataset, joints)
    tables = {
        'fits': fits,
        'stability': report.offset_stability(fits),
        'has_landmarks': False,
        'has_curve': False,
        'closure_by_source': _closure(fits),
        'ambiguity': pd.DataFrame({'subject': fits['subject'], 'trial': fits['trial'],
                                   'joint': fits['joint'], 'worst_mm': 15.0, 'median_mm': 3.0,
                                   'best_mm': 1.0, 'anisotropy': 15.0}),
        'subject_fits': pd.DataFrame(),
    }
    tables.update(overrides)
    return tables


def _render(measured: dict) -> str:
    with TemporaryDirectory() as tmp:
        path = report.write_report(measured, Path(tmp) / 'r.md')
        return path.read_text()


class TestCoverage(unittest.TestCase):
    """The section that says where you stand. Every registered dataset has to appear."""

    def test_every_registered_dataset_is_listed_measured_or_not(self):
        """A dataset absent from the coverage table is indistinguishable from one with no
        problems, which is the failure this section exists to prevent."""
        spec = get_dataset('alborno')
        text = _render({'alborno': _tables('alborno', spec.primary_joints)})
        for name in DATASETS:
            self.assertIn(f'`{name}`', text, f"{name} missing from the report entirely")

    def test_unmeasured_dataset_gets_the_command_that_fixes_it(self):
        text = _render({'alborno': _tables('alborno', get_dataset('alborno').primary_joints)})
        self.assertIn('python -m experiments.joint_center --dataset', text)

    def test_ran_but_fitted_nothing_is_not_reported_as_unmeasured(self):
        """THE CASE THE TWO STATES COLLAPSE INTO ONE. A dataset whose ground truth is too short to
        fit has run, so telling the reader to run it is wrong; it must say what the floor was."""
        joints = get_dataset('imove_biplane').primary_joints
        short = _fits('imove_biplane', joints, converged=False, n_valid=180)
        tables = _tables('imove_biplane', joints, fits=short,
                         closure_by_source=pd.DataFrame(), ambiguity=pd.DataFrame())
        text = _render({'imove_biplane': tables})
        self.assertIn('fitted NOTHING', text)
        self.assertIn(str(MIN_FIT_FRAMES), text)
        self.assertIn('180', text, "the report must say what the data actually had")
        self.assertIn('correct answer rather than a failure', text)

    def test_zero_fit_dataset_contributes_no_per_joint_rows(self):
        """A table of em dashes would read as a measured joint with unknown quality."""
        joints = get_dataset('imove_biplane').primary_joints
        short = _fits('imove_biplane', joints, converged=False, n_valid=180)
        self.assertTrue(report.per_joint(_tables('imove_biplane', joints, fits=short),
                                        get_dataset('imove_biplane')).empty)

    def test_missing_inertial_arm_is_named_as_a_gap(self):
        """The inertial row is the only one that says whether this works without mocap, so its
        absence is a gap in the argument and not just in a table."""
        spec = get_dataset('alborno')
        text = _render({'alborno': _tables('alborno', spec.primary_joints)})
        self.assertIn('inertial_joint_center', text)


class TestPerJoint(unittest.TestCase):

    def test_between_trial_spread_is_populated_and_is_not_the_residual(self):
        """The number the report tells a reader to quote. If `offset_stability` ever returns
        nothing, this column silently becomes em dashes and the headline claim is unsupported."""
        spec = get_dataset('alborno')
        table = report.per_joint(_tables('alborno', spec.primary_joints), spec)
        self.assertTrue(np.isfinite(table['spread_mm']).all())
        self.assertFalse(np.allclose(table['spread_mm'], table['closure_own_mm']))

    def test_placement_variants_are_excluded(self):
        """IMoVE fits the same anatomical joint from three placements. Pooling them reports that
        joint three times and weights the dataset threefold."""
        spec = get_dataset('imove')
        variants = [j for j in spec.joints if j not in spec.primary_joints]
        self.assertTrue(variants, "the fixture assumes imove has placement variants")
        fits = _fits('imove', list(spec.primary_joints) + variants)
        table = report.per_joint(_tables('imove', None, fits=fits), spec)
        self.assertEqual(set(table.index), set(spec.primary_joints))

    def test_joints_are_ordered_by_the_spec_not_alphabetically(self):
        spec = get_dataset('alborno')
        table = report.per_joint(_tables('alborno', spec.primary_joints), spec)
        self.assertEqual(list(table.index),
                         [j for j in spec.primary_joints if j in table.index])

    def test_cost_is_gain_times_spread_not_times_residual(self):
        """The distinction section 6 of the experiment insists on: the residual is a model
        mismatch the fit already absorbed, the spread is what behaves like an error in r."""
        spec = get_dataset('alborno')
        table = report.per_joint(_tables('alborno', spec.primary_joints), spec)
        expected = table['gain'] * table['spread_mm'] / 1000.0
        np.testing.assert_allclose(table['acc_error'], expected)


class TestClosure(unittest.TestCase):

    def test_floor_ratio_is_relative_to_own(self):
        spec = get_dataset('alborno')
        summary = report.closure_summary(_tables('alborno', spec.primary_joints), spec)
        self.assertAlmostEqual(summary.loc['own', 'vs_floor'], 1.0)
        self.assertGreater(summary.loc['cross_trial', 'vs_floor'], 1.0)

    def test_sources_are_in_reported_order_not_alphabetical(self):
        """'own' is the floor and has to come first; alphabetically it lands third."""
        spec = get_dataset('alborno')
        summary = report.closure_summary(_tables('alborno', spec.primary_joints), spec)
        self.assertEqual(list(summary.index)[0], 'own')

    def test_every_rendered_source_is_explained(self):
        """A source label with no definition beside it is a column a reader cannot use."""
        spec = get_dataset('alborno')
        tables = _tables('alborno', spec.primary_joints,
                         closure_by_source=_closure(_fits('alborno', spec.primary_joints),
                                                    sources=report.SOURCE_ORDER))
        text = _render({'alborno': tables})
        for source in report.SOURCE_ORDER:
            self.assertIn(f'- **`{source}`**', text)

    def test_count_disagreement_between_tables_is_surfaced(self):
        """Two tables written by different runs. The reader would otherwise see two totals
        differing by a handful and have no way to tell which is stale."""
        spec = get_dataset('alborno')
        fits = _fits('alborno', spec.primary_joints)
        extra = _closure(fits)
        extra = pd.concat([extra, extra.head(1).assign(trial='t9')], ignore_index=True)
        text = _render({'alborno': _tables('alborno', None, fits=fits,
                                           closure_by_source=extra)})
        self.assertIn('written by different runs', text)


class TestRendering(unittest.TestCase):

    def test_no_nan_reaches_the_reader(self):
        """'nan' in a report is a number the reader will try to interpret."""
        spec = get_dataset('alborno')
        fits = _fits('alborno', spec.primary_joints, n_subjects=1)  # no cross-subject spread
        text = _render({'alborno': _tables('alborno', None, fits=fits,
                                           ambiguity=pd.DataFrame())})
        self.assertNotIn('nan', text.lower())

    def test_empty_table_body_renders_nothing_at_all(self):
        self.assertEqual(report._table(['a', 'b'], []), [])

    def test_single_trial_per_subject_says_there_is_no_error_bar(self):
        """With one trial each there is no between-trial spread, and a table of em dashes has to
        be labelled rather than left to be read as 'zero'."""
        spec = get_dataset('alborno')
        one = _fits('alborno', spec.primary_joints).query("trial == 't0'")
        text = _render({'alborno': _tables('alborno', None, fits=one)})
        self.assertIn('no between-trial spread', text)

    def test_report_names_its_own_regeneration_command(self):
        spec = get_dataset('alborno')
        text = _render({'alborno': _tables('alborno', spec.primary_joints)})
        self.assertIn('python -m plotting.joint_center_quality', text)

    def test_refuses_nothing_and_writes_a_file(self):
        spec = get_dataset('alborno')
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / 'nested' / 'r.md'
            report.write_report({'alborno': _tables('alborno', spec.primary_joints)}, target)
            self.assertTrue(target.exists())


if __name__ == '__main__':
    unittest.main()
