"""
Covers experiment_utils.resolve_method_spec.

Method names are the pipeline's only wiring between "what the user asked for" and
"which physics runs": the name picks the filter, the oracle inputs, and the output
filename all at once. A name that mis-parses does not fail — it silently runs a
different configuration and saves it under the requested label. These tests pin the
grammar and the round trip from every name the experiment scripts actually build.

Pure name parsing, so no data and no filter runs — fast, always runnable.
"""
import os
import unittest

os.environ.setdefault("DISABLE_TQDM", "True")

from experiments.experiment_utils import (DEFAULT_MAG_ADAPT_THRESHOLD, METHODS,
                                          resolve_method_spec)
from experiments.oracle_ablation import build_oracle_combos, oracle_method_name


class TestBaseMethods(unittest.TestCase):
    def test_every_base_resolves_to_its_own_entry(self):
        for name, entry in METHODS.items():
            with self.subTest(method=name):
                spec = resolve_method_spec(name)
                for key, value in entry.items():
                    self.assertEqual(spec[key], value)

    def test_bare_names_default_to_real_sensors(self):
        """No suffix must mean no oracle. If this ever flipped, every headline number
        in the paper would quietly become an oracle number."""
        for name in METHODS:
            with self.subTest(method=name):
                spec = resolve_method_spec(name)
                self.assertEqual(spec['acc_source'], 'real')
                self.assertEqual(spec['mag_source'], 'real')

    def test_bare_names_carry_no_threshold_override(self):
        for name in METHODS:
            with self.subTest(method=name):
                self.assertNotIn('mag_adapt_threshold', resolve_method_spec(name))

    def test_the_returned_spec_is_a_copy(self):
        """Callers mutate the spec (compute_joint_angles pops fields off it in effect);
        if it aliased the METHODS entry, one call would contaminate the next."""
        spec = resolve_method_spec('mag_on')
        spec['mag_mode'] = 'off'
        spec['acc_source'] = 'perfect'
        self.assertEqual(METHODS['mag_on']['mag_mode'], 'on')
        self.assertEqual(resolve_method_spec('mag_on')['mag_mode'], 'on')
        self.assertEqual(resolve_method_spec('mag_on')['acc_source'], 'real')


class TestOracleSuffixes(unittest.TestCase):
    def test_acc_oracle_only(self):
        spec = resolve_method_spec('mag_off_perfect_acc')
        self.assertEqual(spec['mag_mode'], 'off')
        self.assertEqual(spec['acc_source'], 'perfect')
        self.assertEqual(spec['mag_source'], 'real')

    def test_mag_oracle_only_with_explicit_real_acc(self):
        spec = resolve_method_spec('mag_on_real_acc_perfect_mag')
        self.assertEqual(spec['mag_mode'], 'on')
        self.assertEqual(spec['acc_source'], 'real')
        self.assertEqual(spec['mag_source'], 'perfect')

    def test_mag_oracle_only_with_the_acc_suffix_omitted(self):
        """'ekf_perfect_mag' (used by ekf_oracle_comparison and the failure-mode figure)
        is the same spec as 'ekf_real_acc_perfect_mag' (built by oracle_ablation)."""
        self.assertEqual(resolve_method_spec('ekf_perfect_mag'),
                         resolve_method_spec('ekf_real_acc_perfect_mag'))
        self.assertEqual(resolve_method_spec('ekf_perfect_mag')['mag_source'], 'perfect')

    def test_both_oracles(self):
        spec = resolve_method_spec('ekf_perfect_acc_perfect_mag')
        self.assertEqual(spec['kind'], 'ekf')
        self.assertEqual(spec['acc_source'], 'perfect')
        self.assertEqual(spec['mag_source'], 'perfect')

    def test_suffix_order_is_enforced(self):
        """acc before mag. The reversed name is a typo, not a synonym, and must not
        parse as the base with a longer name."""
        with self.assertRaises(ValueError):
            resolve_method_spec('ekf_perfect_mag_perfect_acc')

    def test_an_unknown_oracle_word_is_rejected(self):
        for name in ('ekf_ideal_acc', 'mag_on_true_mag', 'mag_on_perfect_gyro'):
            with self.subTest(method=name):
                with self.assertRaises(ValueError):
                    resolve_method_spec(name)


class TestThresholdSuffix(unittest.TestCase):
    def test_threshold_is_parsed_as_a_float(self):
        self.assertEqual(resolve_method_spec('mag_adapt_th50.00')['mag_adapt_threshold'], 50.0)
        self.assertEqual(resolve_method_spec('mag_adapt_th0')['mag_adapt_threshold'], 0.0)
        self.assertEqual(resolve_method_spec('mag_adapt_th1000.5')['mag_adapt_threshold'], 1000.5)

    def test_threshold_composes_with_the_oracle_suffixes(self):
        spec = resolve_method_spec('mag_adapt_th150.00_perfect_acc_perfect_mag')
        self.assertEqual(spec['mag_mode'], 'adapt')
        self.assertEqual(spec['mag_adapt_threshold'], 150.0)
        self.assertEqual(spec['acc_source'], 'perfect')
        self.assertEqual(spec['mag_source'], 'perfect')

    def test_the_names_threshold_sensitivity_builds_all_resolve(self):
        """Mirrors experiments/threshold_sensitivity.py's f"mag_adapt_th{threshold:.2f}"."""
        for threshold in (0.0, 12.5, 150.0, 1e4):
            name = f"mag_adapt_th{threshold:.2f}"
            with self.subTest(method=name):
                self.assertEqual(resolve_method_spec(name)['mag_adapt_threshold'], threshold)

    def test_the_default_threshold_round_trips(self):
        name = f"mag_adapt_th{DEFAULT_MAG_ADAPT_THRESHOLD:.2f}"
        self.assertEqual(resolve_method_spec(name)['mag_adapt_threshold'],
                         DEFAULT_MAG_ADAPT_THRESHOLD)

    def test_a_threshold_on_a_non_adapt_base_parses_but_does_nothing(self):
        """Documented, not endorsed: only the 'adapt' path reads the threshold, so
        'mag_on_th50.00' is 'mag_on' computed again under a second filename. Harmless
        but wasteful; the caller almost certainly meant mag_adapt."""
        spec = resolve_method_spec('mag_on_th50.00')
        self.assertEqual(spec['mag_mode'], 'on')
        self.assertEqual(spec['mag_adapt_threshold'], 50.0)


class TestRejectedNames(unittest.TestCase):
    def test_an_unknown_base_raises_and_names_the_base(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_method_spec('madgwick')
        self.assertIn('madgwick', str(ctx.exception))

    def test_a_near_miss_on_a_real_base_raises(self):
        for name in ('mag_onn', 'magon', 'Mag_On', 'ekf2'):
            with self.subTest(method=name):
                with self.assertRaises(ValueError):
                    resolve_method_spec(name)

    def test_the_empty_name_raises_valueerror(self):
        """The suffix regex cannot match an empty string, so this is the one input that
        reaches the body with no match object at all — it needs its own guard to fail the
        same way every other bad name does."""
        with self.assertRaises(ValueError):
            resolve_method_spec('')


class TestOracleAblationRoundTrip(unittest.TestCase):
    """oracle_ablation builds names from (base, acc_source, mag_source) and the pipeline
    parses them back. If the two ever disagree, the sweep silently mislabels its own
    cells — the ablation table would attribute one configuration's error to another."""

    def test_every_generated_combo_parses_back_to_what_was_requested(self):
        for name, base, acc_source, mag_source in build_oracle_combos(list(METHODS)):
            with self.subTest(method=name):
                spec = resolve_method_spec(name)
                self.assertEqual(spec['acc_source'], acc_source)
                self.assertEqual(spec['mag_source'], mag_source)
                for key, value in METHODS[base].items():
                    self.assertEqual(spec[key], value)

    def test_generated_names_are_unique_per_combo(self):
        combos = build_oracle_combos(list(METHODS))
        names = [name for name, *_ in combos]
        self.assertEqual(len(names), len(set(names)))

    def test_the_all_real_combo_is_the_bare_base_name(self):
        for base in METHODS:
            with self.subTest(base=base):
                self.assertEqual(oracle_method_name(base, 'real', 'real'), base)


if __name__ == '__main__':
    unittest.main()
