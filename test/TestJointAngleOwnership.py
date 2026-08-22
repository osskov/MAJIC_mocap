"""
Who is allowed to write results/joint_angles/.

The tree is keyed by DATASET, SUBJECT, TRIAL and METHOD — and a method name encodes mag mode,
oracles, normalization, threshold and distortion scale, but NOT which experiment ran it and NOT
the filter tuning it ran at. So `benchmark_experiment`, `threshold_sensitivity` and
`oracle_ablation` all wanted to write the same `mag_on.parquet`. While they compute the same
thing that is only wasteful; the moment a tuning differs between two runs, one experiment's arms
become a silent mix of two estimators. Every file still loads, every figure still renders, and
the only symptom is a number that should be impossible — it cost two full sweeps before anyone
worked out why.

The rule: the benchmark owns `results/joint_angles/`, everything else writes under
`results/experiments/<experiment>/joint_angles/`, and READING is open to all. These tests pin
the mechanism rather than the convention, because a convention is what was already in place.
"""
import os
import pathlib
import tempfile
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

import pandas as pd

import paths
from experiments import (distortion_tolerance, ekf_oracle_comparison, normalization_comparison,
                         normalized_benchmark, oracle_ablation, threshold_sensitivity)

TRIAL = ('alborno', '01', 'walking', 'mag_on')

# Every experiment that writes joint angles and is NOT the benchmark. Kept as a list of the
# modules themselves so that a new experiment which forgets to declare EXPERIMENT_NAME fails
# here rather than by quietly writing into the benchmark's tree.
SIBLINGS = (distortion_tolerance, ekf_oracle_comparison, normalization_comparison,
            normalized_benchmark, oracle_ablation, threshold_sensitivity)


class TestOnlyTheBenchmarkOwnsTheCanonicalTree(unittest.TestCase):

    def test_the_benchmark_writes_the_canonical_tree(self):
        self.assertEqual(paths.joint_angles_write_path(paths.BENCHMARK_EXPERIMENT, *TRIAL),
                         paths.joint_angles_path(*TRIAL))

    def test_every_other_experiment_writes_its_own(self):
        for module in SIBLINGS:
            name = module.EXPERIMENT_NAME
            with self.subTest(experiment=name):
                path = paths.joint_angles_write_path(name, *TRIAL)
                self.assertNotIn(paths.JOINT_ANGLES_DIR, [path, *path.parents],
                                 f"{name} would write into the benchmark's tree")
                self.assertEqual(path.parent.parent.parent.parent,
                                 paths.experiment_dir(name) / "joint_angles")

    def test_no_two_experiments_share_a_write_path(self):
        """The property that makes the clobbering structurally impossible, not merely detected."""
        names = [paths.BENCHMARK_EXPERIMENT] + [m.EXPERIMENT_NAME for m in SIBLINGS]
        written = [paths.joint_angles_write_path(n, *TRIAL) for n in names]
        self.assertEqual(len(set(written)), len(names))

    def test_an_undeclared_writer_raises(self):
        """No default. The script that forgets to say who it is is precisely the one that would
        overwrite the benchmark, and it would do so without a symptom."""
        for missing in (None, ''):
            with self.subTest(experiment=missing):
                with self.assertRaises(ValueError):
                    paths.joint_angles_write_path(missing, *TRIAL)

    def test_the_benchmark_sentinel_means_one_tree_for_read_and_write(self):
        """The bug this pins: `joint_angles_dir` mapped the sentinel to the canonical tree only
        inside the write helper, so the benchmark WROTE results/joint_angles/ and READ
        results/experiments/benchmark/joint_angles/ — a directory nothing ever creates. Its
        statistics phase found none of its own arms, and because the trial cache was stale at the
        time, nothing failed loudly enough to notice."""
        by_default = paths.joint_angles_path(*TRIAL)
        by_sentinel = paths.joint_angles_path(*TRIAL, experiment=paths.BENCHMARK_EXPERIMENT)
        written = paths.joint_angles_write_path(paths.BENCHMARK_EXPERIMENT, *TRIAL)
        self.assertEqual(by_default, by_sentinel)
        self.assertEqual(by_default, written)
        self.assertEqual(paths.joint_angles_dir(paths.BENCHMARK_EXPERIMENT),
                         paths.JOINT_ANGLES_DIR)

    def test_no_experiment_subdirectory_is_named_after_the_benchmark(self):
        """If the sentinel ever leaks into a path as a directory name, this is what it looks
        like — and it looks plausible, which is why it went unnoticed."""
        leaked = paths.experiment_dir(paths.BENCHMARK_EXPERIMENT) / "joint_angles"
        self.assertNotEqual(paths.joint_angles_dir(paths.BENCHMARK_EXPERIMENT), leaked)

    def test_reading_is_open(self):
        """An experiment comparing itself against the benchmark's arms has to be able to read
        them, so the read path defaults to the canonical tree and restricts nothing."""
        self.assertEqual(paths.joint_angles_path(*TRIAL),
                         paths.JOINT_ANGLES_DIR / 'alborno' / '01' / 'walking' / 'mag_on.parquet')
        own = paths.joint_angles_path(*TRIAL, experiment='oracle_ablation')
        self.assertEqual(own, paths.experiment_dir('oracle_ablation') / 'joint_angles'
                         / 'alborno' / '01' / 'walking' / 'mag_on.parquet')

    def test_a_variant_still_nests_inside_whichever_tree_owns_it(self):
        for experiment in (paths.BENCHMARK_EXPERIMENT, 'oracle_ablation'):
            with self.subTest(experiment=experiment):
                plain = paths.joint_angles_write_path(experiment, *TRIAL)
                tuned = paths.joint_angles_write_path(experiment, *TRIAL, variant='retuned')
                self.assertNotEqual(plain, tuned)
                self.assertIn('retuned', tuned.parts)

    def test_every_sibling_declares_a_name_matching_its_module(self):
        """The tree is named after the experiment, so a mismatch would put one experiment's arms
        in a directory named for another."""
        for module in SIBLINGS:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.EXPERIMENT_NAME, module.__name__.rsplit('.', 1)[-1])

    def test_no_sibling_claims_to_be_the_benchmark(self):
        for module in SIBLINGS:
            with self.subTest(module=module.__name__):
                self.assertNotEqual(module.EXPERIMENT_NAME, paths.BENCHMARK_EXPERIMENT)


class TestTheWorkerRefusesToGuess(unittest.TestCase):

    def test_generation_worker_requires_an_experiment(self):
        """Checked in the worker as well as in the path helper: the path is only reached after a
        trial's filters have run, and a three-minute run that then refuses to save has already
        wasted the compute."""
        from experiments.experiment_utils import generate_joint_angles_worker
        with self.assertRaises(ValueError) as caught:
            generate_joint_angles_worker(('01', 'walking'), ['mag_on'], {})
        self.assertIn('experiment', str(caught.exception))


class TestAnEmptyGenerationPhaseStopsTheRun(unittest.TestCase):

    def setUp(self):
        # A real directory, so the filters table is genuinely written rather than written to a
        # Mock. Patching `ensure_parent` away instead made `to_parquet` receive a MagicMock.
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stats_dir = pathlib.Path(self.tmp.name)
    """The guard that exists because its absence made a verification lie.

    The statistics phase reads whatever parquets are on disk. So when every generation cell
    fails — a stale trial cache does this to all 19 Al Borno trials at once — the benchmark used
    to score the PREVIOUS run's arms and print "Aggregated 19 trial(s)" over them. Nothing in
    the output distinguished that from a clean run.
    """

    def _run(self, outcome, methods=('marker', 'mag_on')):
        """Drives main() with the generation grid stubbed to a fixed per-cell outcome."""
        from unittest import mock
        import experiments.benchmark_experiment as be

        row_keys = [('01', 'walking')]
        state = {(key, method): outcome for key in row_keys for method in methods}

        with mock.patch.object(be, 'select_trials', return_value=row_keys), \
             mock.patch.object(be, 'orphaned_trials', return_value=[]), \
             mock.patch.object(be, 'run_tracked_grid',
                               return_value=(state, {})) as grid, \
             mock.patch.object(be, 'report_filter_mismatch', return_value=False), \
             mock.patch.object(be.paths, 'STATISTICS_DIR', self.stats_dir), \
             mock.patch.object(be, 'load_per_trial_statistics') as load_stats, \
             mock.patch.object(be, 'save_statistics') as save, \
             mock.patch('sys.argv', ['benchmark_experiment', '--dataset', 'alborno',
                                     '--methods', *methods]):
            code = be.main()
        return code, grid.call_count, load_stats.call_count, save.call_count

    def test_all_cells_failed_refuses_to_compile_statistics(self):
        code, grids, loaded, saved = self._run('Failed (load: trial cache is stale)')
        self.assertEqual(code, 1)
        self.assertEqual(grids, 1, "should stop after the generation grid")
        self.assertEqual(loaded, 0, "must not read the previous run's statistics")
        self.assertEqual(saved, 0, "must not write a summary")

    def test_a_successful_generation_proceeds(self):
        """The guard must not fire on a normal run."""
        code, grids, loaded, _ = self._run('Success')
        self.assertGreater(grids, 1, "should go on to the statistics grid")
        self.assertEqual(loaded, 1)

    def test_a_partial_failure_proceeds(self):
        """Two biplane trials legitimately have no bone poses; that is named in the shortfall
        report, not treated as a dead run."""
        from unittest import mock
        import experiments.benchmark_experiment as be
        row_keys = [('01', 'walking')]
        state = {(row_keys[0], 'marker'): 'Success',
                 (row_keys[0], 'mag_on'): 'Failed (something)'}
        with mock.patch.object(be, 'select_trials', return_value=row_keys), \
             mock.patch.object(be, 'orphaned_trials', return_value=[]), \
             mock.patch.object(be, 'run_tracked_grid', return_value=(state, {})), \
             mock.patch.object(be, 'report_filter_mismatch', return_value=False), \
             mock.patch.object(be.paths, 'STATISTICS_DIR', self.stats_dir), \
             mock.patch.object(be, 'load_per_trial_statistics') as load_stats, \
             mock.patch.object(be, 'save_statistics'), \
             mock.patch('sys.argv', ['benchmark_experiment', '--dataset', 'alborno',
                                     '--methods', 'marker', 'mag_on']):
            be.main()
        self.assertEqual(load_stats.call_count, 1)


if __name__ == '__main__':
    unittest.main()
