"""experiments/build_trials.py — the status-to-action decision, and the CLI's guards.

This file had no tests at all, which matters most for one case: a reader that raises must
produce `action='failed'` for that trial and let the other 260 carry on. A change that let an
exception escape the worker would abort an hour of building, and nothing would have noticed.

The worker takes a dict and returns a dict, so none of this needs real data — the source is
a stub and the expensive half is patched out.
"""
import unittest
from unittest import mock

from experiments import build_trials
from src.toolchest.building import sources


class _StubSource:
    """A TrialSource-shaped object whose load() does whatever the test needs."""

    def __init__(self, load):
        self.name = '_stub'
        self.load = load
        self.enumerate_trials = lambda: [('01', 'walking'), ('02', 'walking')]
        self.source_dir = lambda subject, trial: build_trials.paths.REPO_ROOT
        self.source_globs = ()


class TestWorkerDecisions(unittest.TestCase):
    """What build_trial_worker does with each cache status."""

    def _run(self, status, force=False, load=None, save=None):
        source = _StubSource(load or (lambda subject, trial: {'plate': object()}))
        with mock.patch.dict(sources.SOURCES, {'_stub': source}), \
             mock.patch.object(build_trials, 'cached_trial_status',
                               return_value=(status, None)), \
             mock.patch.object(build_trials, 'save_cached_trial',
                               side_effect=save or self._fake_save), \
             mock.patch.object(build_trials.paths, 'read_manifest', return_value={}):
            return build_trials.build_trial_worker(
                ('01', 'walking'), ['build'], {}, dataset='_stub', force=force)

    @staticmethod
    def _fake_save(plates, subject, trial, dataset):
        path = mock.MagicMock()
        path.stat.return_value.st_size = 1234
        return path

    def test_a_fresh_entry_is_skipped(self):
        self.assertEqual(self._run('fresh')['action'], 'skipped')

    def test_force_rebuilds_a_fresh_entry(self):
        """The only thing --force does. Without this the flag could stop working silently."""
        self.assertEqual(self._run('fresh', force=True)['action'], 'built')

    def test_a_missing_entry_is_built(self):
        self.assertEqual(self._run('missing')['action'], 'built')

    def test_an_absent_source_is_not_a_failure(self):
        """'absent' means the trial does not exist in the dataset, which is normal -- Al
        Borno's 05, 08 and 10 have walking only. Reporting it as 'failed' is what the old
        SUBJECTS x ACTIVITIES cross product did, and the distinction has no other guard."""
        result = self._run('absent')
        self.assertEqual(result['action'], 'absent')
        self.assertNotEqual(result['action'], 'failed')

    def test_a_raising_reader_fails_only_its_own_trial(self):
        """The exception must be caught and turned into a result, not escape the worker and
        take an hour-long build down with it."""
        def explode(subject, trial):
            raise ValueError('marker reconstruction gave up')

        result = self._run('missing', load=explode)

        self.assertEqual(result['action'], 'failed')
        self.assertIn('marker reconstruction gave up', result['error'])

    def test_a_failure_keeps_its_traceback(self):
        """str(e) alone routinely does not say which of several call paths raised, and the
        build that surfaced it costs minutes to reproduce."""
        def explode(subject, trial):
            raise ValueError('deep in the stack')

        result = self._run('missing', load=explode)

        self.assertIn('traceback', result)
        self.assertIn('ValueError', result['traceback'])
        self.assertIn('explode', result['traceback'])


class TestSelectionGuards(unittest.TestCase):
    """--subjects and --trials are validated against what is actually on disk."""

    def setUp(self):
        self.source = _StubSource(lambda subject, trial: {})
        self._patch = mock.patch.dict(sources.SOURCES, {'_stub': self.source})
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _main(self, *argv):
        with mock.patch('sys.argv', ['build_trials', '--dataset', '_stub', '--check', *argv]):
            return build_trials.main()

    def test_an_unknown_subject_is_rejected(self):
        """Checked against the enumerated set rather than a constant, so a typo cannot
        silently select nothing and report success."""
        self.assertEqual(self._main('--subjects', '99'), 1)

    def test_an_unknown_trial_is_rejected(self):
        self.assertEqual(self._main('--trials', 'not_a_trial'), 1)

    def test_a_known_subject_is_accepted(self):
        with mock.patch.object(build_trials, 'cached_trial_status',
                               return_value=('fresh', None)), \
             mock.patch.object(build_trials, '_has_suspect_plate', return_value=False):
            self.assertEqual(self._main('--subjects', '01'), 0)

    def test_check_exits_nonzero_when_work_remains(self):
        """--check reported and exited 0 regardless, so 'everything is stale' and
        'everything is fine' were indistinguishable to a script."""
        with mock.patch.object(build_trials, 'cached_trial_status',
                               return_value=('stale', 'toolchest_digest')):
            self.assertEqual(self._main(), 1)


class TestStatusGrid(unittest.TestCase):
    """The grid is the only thing that makes 262 trials legible at a glance."""

    KEYS = [('s2', 't1_walking_001'), ('s2', 't6_drop_jump_001'),
            ('s13l', 't12_longwalk_001')]

    def test_axes_are_in_natural_order(self):
        """'t10' sorts before 't2' as a string, which puts the columns in an order nobody
        reading them expects."""
        self.assertEqual(build_trials._natural_key('t2'), ['t', 2, ''])
        self.assertLess(build_trials._natural_key('t2'), build_trials._natural_key('t10'))
        self.assertLess(build_trials._natural_key('s4'), build_trials._natural_key('s13'))

    def test_a_trial_absent_from_a_session_renders_blank(self):
        """s13l has no t1, and a blank cell has to mean 'not in this dataset' rather than
        'not built yet' -- the long-walk sessions would otherwise look 90% unbuilt."""
        table = build_trials.status_grid(self.KEYS, 'imove',
                                         statuses={k: 'fresh' for k in self.KEYS})
        self.assertEqual(table.row_count, 2)

    def test_quiet_suppresses_everything(self):
        with mock.patch.object(build_trials, 'Console') as console:
            build_trials.print_status_grid(self.KEYS, 'imove', statuses={}, quiet=True)
        console.assert_not_called()


if __name__ == '__main__':
    unittest.main()
