"""Nothing ever writes under data/ — the repo's headline rule, tested.

paths.py opens by saying the read-only-inputs rule is "enforced mechanically, not by
convention". It was enforced by three functions that no test touched, so simplifying
`ensure_parent` to a bare `mkdir` would have kept the whole suite green while removing the
guarantee. `data/` is a large external download that nothing in the repo can regenerate.
"""
import unittest

import paths


class TestDataDirectoryIsReadOnly(unittest.TestCase):
    def test_ensure_parent_refuses_a_path_under_data(self):
        with self.assertRaises(ValueError) as caught:
            paths.ensure_parent(paths.raw_trial_dir('01', 'walking') / 'out.parquet')
        self.assertIn('read-only', str(caught.exception))

    def test_ensure_parent_refuses_data_itself(self):
        with self.assertRaises(ValueError):
            paths.ensure_parent(paths.DATA_DIR)

    def test_write_manifest_refuses_a_path_under_data(self):
        """The sidecar writer is a separate entry point, so it needs its own guard rather
        than relying on whoever created the parent directory."""
        with self.assertRaises(ValueError):
            paths.write_manifest(paths.raw_trial_dir('01', 'walking') / 'out.parquet')

    def test_a_path_that_merely_mentions_data_is_allowed(self):
        """The guard is on containment, not on the string. results/data_summary/ is fine."""
        allowed = paths.RESULTS_DIR / 'data_summary' / 'x.parquet'
        self.assertEqual(paths.ensure_parent(allowed), allowed)
        self.addCleanup(lambda: allowed.parent.rmdir() if allowed.parent.is_dir() else None)

    def test_results_paths_are_accepted(self):
        target = paths.RESULTS_DIR / '_guard_probe' / 'x.parquet'
        self.assertEqual(paths.ensure_parent(target), target)
        self.addCleanup(lambda: target.parent.rmdir() if target.parent.is_dir() else None)

    def test_a_symlink_out_of_results_into_data_is_still_refused(self):
        """resolve() before comparing, so the check cannot be walked around with '..'."""
        sneaky = paths.RESULTS_DIR / '..' / 'data' / 'Subject01' / 'x.parquet'
        with self.assertRaises(ValueError):
            paths.ensure_parent(sneaky)


if __name__ == '__main__':
    unittest.main()
