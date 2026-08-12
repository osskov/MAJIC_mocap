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

import numpy as np
import pandas as pd

from experiments.experiment_utils import (DEFAULT_MAG_ADAPT_THRESHOLD, METHODS,
                                          resolve_method_spec)
from experiments.distortion_tolerance import (DISTORTION_SCALES, distortion_method,
                                              scale_from_method)
from experiments.normalization_comparison import (NORMALIZATION_ARMS,
                                                  build_normalization_methods)
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


class TestDistortionSuffix(unittest.TestCase):
    """The '_dist<a>' suffix, which scales the magnetometer's estimated distortion. Its two
    endpoints coincide with arms the pipeline already has ('_dist0.00' is the mag oracle,
    '_dist1.00' is the real reading), so the grammar has to make that explicit rather than
    leave two spellings of the same configuration silently unrelated."""

    def test_the_scale_is_parsed_as_a_float_and_sets_the_scaled_source(self):
        for name, expected in (('mag_on_dist0.00', 0.0), ('mag_on_dist0.25', 0.25),
                               ('mag_on_dist1.00', 1.0), ('mag_on_dist2', 2.0)):
            with self.subTest(method=name):
                spec = resolve_method_spec(name)
                self.assertEqual(spec['mag_distortion_scale'], expected)
                self.assertEqual(spec['mag_source'], 'scaled')

    def test_bare_names_carry_no_distortion_scale(self):
        """The counterpart of the no-oracle-by-default test: a scale leaking onto a bare
        name would replace every headline arm's magnetometer with a reconstructed one."""
        for name in METHODS:
            with self.subTest(method=name):
                spec = resolve_method_spec(name)
                self.assertNotIn('mag_distortion_scale', spec)
                self.assertEqual(spec['mag_source'], 'real')

    def test_the_names_distortion_tolerance_builds_all_resolve(self):
        for scale in DISTORTION_SCALES:
            name = distortion_method(scale)
            with self.subTest(method=name):
                self.assertEqual(resolve_method_spec(name)['mag_distortion_scale'], float(scale))

    def test_the_scale_round_trips_through_the_column_the_sweep_plots(self):
        """distortion_method and scale_from_method are inverses; the figure's x-axis is the
        latter applied to method names written by the former. A mismatch would mislabel every
        point on the curve."""
        names = pd.Series([distortion_method(s) for s in DISTORTION_SCALES])
        np.testing.assert_allclose(scale_from_method(names).to_numpy(),
                                   np.asarray(DISTORTION_SCALES, dtype=float))

    def test_non_swept_methods_have_no_recoverable_scale(self):
        """The references share the statistics file with the sweep and are selected by the
        scale column being NaN."""
        parsed = scale_from_method(pd.Series(['mag_on', 'mag_off', 'marker', 'mag_adapt_th50.00',
                                              'ekf', 'mag_on_real_acc_perfect_mag']))
        self.assertTrue(parsed.isna().all())

    def test_the_distortion_and_oracle_suffixes_cannot_be_combined(self):
        """Both replace the magnetometer wholesale, so a name carrying both asks for two
        readings at once. It has to raise: whichever won silently, the sweep would read a
        curve off arms that are not the arms their names claim."""
        with self.assertRaises(ValueError):
            resolve_method_spec('mag_on_dist0.50_perfect_mag')

    def test_an_explicit_real_mag_alongside_a_scale_is_still_scaled(self):
        """'_real_mag' is the grammar's way of spelling the default out (see
        'mag_on_real_acc_perfect_mag'), so it names the absence of an ORACLE, not the absence
        of a scale. Unlike '_perfect_mag' it therefore composes rather than conflicting."""
        spec = resolve_method_spec('mag_on_dist0.50_real_mag')
        self.assertEqual(spec['mag_source'], 'scaled')
        self.assertEqual(spec['mag_distortion_scale'], 0.5)

    def test_the_distortion_suffix_composes_with_the_others(self):
        spec = resolve_method_spec('mag_adapt_normalized_th150.00_dist0.50_perfect_acc')
        self.assertEqual(spec['mag_mode'], 'adapt')
        self.assertTrue(spec['normalize_measurements'])
        self.assertEqual(spec['mag_adapt_threshold'], 150.0)
        self.assertEqual(spec['mag_distortion_scale'], 0.5)
        self.assertEqual(spec['acc_source'], 'perfect')

    def test_distortion_comes_after_the_threshold_and_before_the_oracles(self):
        """Fixed order, same as everywhere else in this grammar: the reversed spellings are
        typos and must not resolve to a base with a longer name."""
        for name in ('mag_adapt_dist0.50_th150.00', 'mag_on_perfect_acc_dist0.50'):
            with self.subTest(method=name):
                with self.assertRaises(ValueError):
                    resolve_method_spec(name)

    def test_a_malformed_scale_is_rejected(self):
        for name in ('mag_on_dist', 'mag_on_distx', 'mag_on_dist_0.5', 'mag_on_distortion0.5'):
            with self.subTest(method=name):
                with self.assertRaises(ValueError):
                    resolve_method_spec(name)


class TestNormalizationSuffix(unittest.TestCase):
    # The deliberate asymmetry: the EKF path needs a ~96x lighter accelerometer than the
    # relative filters do, and normalization is the mechanism. Pinned per base rather than
    # as a blanket default, because getting it backwards is silent — every method would
    # still run and produce plausible numbers at the wrong tuning.
    EXPECTED_BASE_DEFAULTS = {
        'marker': False, 'mag_on': False, 'mag_off': False, 'mag_adapt': False,
        'ekf': True,
    }

    def test_every_base_has_a_pinned_normalization_default(self):
        """Guards the table below against a base being added without a decision here."""
        self.assertEqual(set(METHODS), set(self.EXPECTED_BASE_DEFAULTS))

    def test_bare_names_take_their_base_default(self):
        for name, expected in self.EXPECTED_BASE_DEFAULTS.items():
            with self.subTest(method=name):
                self.assertEqual(resolve_method_spec(name)['normalize_measurements'], expected)

    def test_no_base_rescales_by_default(self):
        """rescale_stds is an experiment-only arm; nothing should inherit it."""
        for name in METHODS:
            with self.subTest(method=name):
                self.assertFalse(resolve_method_spec(name)['rescale_stds'])

    def test_the_ekf_oracle_variants_inherit_the_base_default(self):
        """The reason the flag lives on the base: 'ekf_perfect_acc' and friends have to
        match the plain EKF, or the oracle gap measures normalization instead of the
        oracle."""
        for name in ('ekf_perfect_acc', 'ekf_perfect_mag', 'ekf_perfect_acc_perfect_mag',
                     'ekf_real_acc_perfect_mag'):
            with self.subTest(method=name):
                self.assertTrue(resolve_method_spec(name)['normalize_measurements'])

    def test_mag_oracle_variants_stay_unnormalized(self):
        for name in ('mag_on_perfect_acc', 'mag_off_perfect_acc', 'mag_adapt_th50.00',
                     'mag_adapt_perfect_acc_perfect_mag'):
            with self.subTest(method=name):
                self.assertFalse(resolve_method_spec(name)['normalize_measurements'])

    def test_the_arms_parse_to_the_flags_they_name(self):
        self.assertTrue(resolve_method_spec('mag_off_normalized')['normalize_measurements'])
        self.assertFalse(resolve_method_spec('mag_off_unnormalized')['normalize_measurements'])

    def test_an_explicit_suffix_overrides_the_base_default_in_both_directions(self):
        """The suffix is an override, not an OR with the base. Turning normalization OFF
        for the EKF has to work, or normalization_comparison cannot run its control arm on
        the one base whose default is on."""
        self.assertFalse(resolve_method_spec('ekf_unnormalized')['normalize_measurements'])
        self.assertTrue(resolve_method_spec('ekf_normalized')['normalize_measurements'])
        self.assertTrue(resolve_method_spec('mag_off_normalized')['normalize_measurements'])

    def test_rescaled_normalizes_and_rescales(self):
        """'_rescaled' is '_normalized' plus the std conversion, so it must set BOTH
        flags. Setting only rescale_stds would divide the stds while the measurements kept
        their raw scale — the filter would over-trust every sensor ~96x instead of
        matching the control."""
        spec = resolve_method_spec('mag_off_rescaled')
        self.assertTrue(spec['normalize_measurements'])
        self.assertTrue(spec['rescale_stds'])

    def test_normalized_does_not_rescale(self):
        """The whole point of the third arm is that the second one leaves R alone."""
        self.assertFalse(resolve_method_spec('mag_off_normalized')['rescale_stds'])

    def test_unnormalized_is_the_bare_base_spec_where_the_base_does_not_normalize(self):
        """For those bases the explicit arm is the same physics as the bare name, only
        under a second filename — otherwise the comparison's control arm would not be the
        pipeline's actual behaviour. The EKF is excluded on purpose: its bare name
        normalizes, so 'ekf_unnormalized' is a genuinely different configuration."""
        for name, normalizes in self.EXPECTED_BASE_DEFAULTS.items():
            if normalizes:
                continue
            with self.subTest(method=name):
                self.assertEqual(resolve_method_spec(f'{name}_unnormalized'),
                                 resolve_method_spec(name))

    def test_the_ekf_control_arm_is_not_the_bare_ekf(self):
        """The counterpart of the above, stated positively so the asymmetry is not read as
        an oversight: normalization_comparison's EKF control differs from the shipped EKF."""
        self.assertNotEqual(resolve_method_spec('ekf_unnormalized'),
                            resolve_method_spec('ekf'))
        self.assertEqual(resolve_method_spec('ekf_normalized'),
                         resolve_method_spec('ekf'))

    def test_the_un_prefix_is_not_swallowed_by_the_base(self):
        """'unnormalized' contains 'normalized', so a greedier grammar could split
        'mag_off_unnormalized' as base 'mag_off_un' + '_normalized' and turn the control
        arm into the treatment arm."""
        self.assertFalse(resolve_method_spec('mag_off_unnormalized')['normalize_measurements'])
        with self.assertRaises(ValueError):
            resolve_method_spec('mag_off_un')

    def test_normalization_composes_with_the_other_suffixes(self):
        spec = resolve_method_spec('mag_adapt_normalized_th150.00_perfect_acc_perfect_mag')
        self.assertEqual(spec['mag_mode'], 'adapt')
        self.assertTrue(spec['normalize_measurements'])
        self.assertEqual(spec['mag_adapt_threshold'], 150.0)
        self.assertEqual(spec['acc_source'], 'perfect')
        self.assertEqual(spec['mag_source'], 'perfect')

    def test_normalization_comes_before_the_other_suffixes(self):
        """Fixed order, same as acc-before-mag: the reversed spelling is a typo."""
        for name in ('mag_adapt_th150.00_normalized', 'ekf_perfect_acc_normalized'):
            with self.subTest(method=name):
                with self.assertRaises(ValueError):
                    resolve_method_spec(name)

    def test_a_near_miss_on_the_arm_word_is_rejected(self):
        for name in ('mag_off_norm', 'mag_off_normalised', 'mag_off_denormalized',
                     'mag_off_rescale', 'mag_off_scaled'):
            with self.subTest(method=name):
                with self.assertRaises(ValueError):
                    resolve_method_spec(name)

    def test_the_arms_are_mutually_exclusive_configurations(self):
        """Exactly one arm per name — no spelling produces two normalization suffixes."""
        with self.assertRaises(ValueError):
            resolve_method_spec('mag_off_normalized_rescaled')


class TestNormalizationComparisonRoundTrip(unittest.TestCase):
    """normalization_comparison builds names from (base, arm) and the pipeline parses
    them back. A disagreement would swap the two arms of the comparison — the one
    failure mode the experiment cannot survive, since both arms otherwise look alike."""

    EXPECTED_FLAGS = {
        'unnormalized': (False, False),
        'normalized': (True, False),
        'rescaled': (True, True),
    }

    def test_every_generated_name_parses_back_to_what_was_requested(self):
        bases = list(METHODS)
        names = build_normalization_methods(bases)
        self.assertEqual(len(names), len(bases) * len(NORMALIZATION_ARMS))
        for name in names:
            with self.subTest(method=name):
                base, _, arm = name.rpartition('_')
                spec = resolve_method_spec(name)
                normalize, rescale = self.EXPECTED_FLAGS[arm]
                self.assertEqual(spec['normalize_measurements'], normalize)
                self.assertEqual(spec['rescale_stds'], rescale)
                # Everything except the normalization flags comes from the base. Those two
                # are excluded because the arm suffix is meant to override the base
                # default — which it must, or the EKF (whose default is on) could not be
                # run as this experiment's unnormalized control.
                for key, value in METHODS[base].items():
                    if key in ('normalize_measurements', 'rescale_stds'):
                        continue
                    self.assertEqual(spec[key], value)

    def test_every_arm_is_covered_by_the_expected_flag_table(self):
        """Guards the test above: a new arm added to NORMALIZATION_ARMS without a row here
        would otherwise KeyError rather than silently pass, but this says so directly."""
        self.assertEqual(set(NORMALIZATION_ARMS), set(self.EXPECTED_FLAGS))

    def test_generated_names_are_unique(self):
        names = build_normalization_methods(list(METHODS))
        self.assertEqual(len(names), len(set(names)))

    def test_the_arms_of_a_base_differ_only_in_the_normalization_flags(self):
        flag_keys = {'normalize_measurements', 'rescale_stds'}
        for base in METHODS:
            specs = [resolve_method_spec(f'{base}_{arm}') for arm in NORMALIZATION_ARMS]
            with self.subTest(base=base):
                # Every arm is a distinct flag combination...
                combos = {(s['normalize_measurements'], s['rescale_stds']) for s in specs}
                self.assertEqual(len(combos), len(NORMALIZATION_ARMS))
                # ...and nothing else about the method moves between them.
                rest = [{k: v for k, v in s.items() if k not in flag_keys} for s in specs]
                for other in rest[1:]:
                    self.assertEqual(rest[0], other)


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
