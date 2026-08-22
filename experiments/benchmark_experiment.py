"""
Vanilla full-grid pipeline (the benchmark experiment): computes joint angles for
every method in METHODS across every built trial of one dataset, then aggregates error
statistics against that dataset's ground truth. See experiments/experiment_utils.py for
the shared engine this is built on, and the rest of experiments/ for method-name
sweeps, noise-tuning sweeps, and other variations on this same pipeline.

    python -m experiments.benchmark_experiment --dataset alborno
    python -m experiments.benchmark_experiment --dataset imove --subjects s13 s13l
    python -m experiments.benchmark_experiment --dataset imove_biplane --methods marker mag_off ekf
    python -m experiments.benchmark_experiment --dataset alborno --stats-only
    python -m experiments.benchmark_experiment --dataset alborno --stats-only --anatomical-axes
    python -m experiments.benchmark_experiment --dataset alborno --pooled-angles

FOUR DATASETS, one --dataset at a time. Everything that differs between them — which sensor
pairs form a joint, which way is up, which sensor the magnetometer oracle is referenced
against, whether there is a magnetometer at all — comes from
`global_assumptions.tracking_spec`, so nothing below asks which dataset it is looking at:

    alborno              8 Xsens IMUs, marker ground truth, 7 joints incl. the lumbar
    imove                15 Xsens IMUs, marker ground truth, 6 joints (no torso sensor, so no
                         lumbar). The High and Low placements on each thigh and shank are NOT
                         benchmarked — see tracking_spec
    imove_biplane        4 MC10 BioStamps, FLUOROSCOPIC BONE POSES, 2 knees, no magnetometer
    imove_biplane_vicon  the same build and the same IMUs against the skin marker cluster
                         instead, so the pair measures what the ground-truth choice costs

WHAT THE DATASET DECIDES ABOUT METHODS. A dataset with no magnetometer supports neither
'mag_on' nor 'mag_adapt', and the reason to refuse them rather than run them is that they
SUCCEED: the reader fills mag with zeros, the filter drops a zero measurement out of its
update, and the result is mag_off's answer under mag_on's name. Left to the default method
list those arms are dropped with a printed note; asked for explicitly they are an error.
'ekf' is offered everywhere — it names an absolute filter, not a magnetometer.

Outputs
-------
THIS IS THE ONLY SCRIPT THAT WRITES results/joint_angles/. Every other experiment reads from it
freely but writes its own arms under results/experiments/<experiment>/joint_angles/, because a
method name is keyed by mag mode and oracles and nothing else — so a sweep and the benchmark
both want to write `mag_on.parquet`, and the moment their tunings differ one experiment's arms
become a silent mix of two estimators. `paths.joint_angles_write_path` enforces it.

    results/joint_angles/<dataset>/<subject>/<trial>/<method>.parquet
    results/statistics/per_subject/all_subject/<dataset>/<subject>/<trial>.parquet
    results/statistics/all_subject_<dataset>_statistics.parquet
    results/statistics/all_subject_<dataset>_filters.parquet

WHAT FILTER PRODUCED THESE NUMBERS is answered by the filters table, one row per method holding
its resolved spec (mag mode, oracles, normalization, threshold) alongside two tunings: the one
this run configured and the one its joint-angle sidecars actually carry. They should be equal.
When they are not, the statistics were computed from a different filter than their own manifest
claims — a `--stats-only` run over Al Borno's cached angles once reported acc_std/mag_std of
0.09695/0.009695 over parquets computed at 0.018/0.05, a 28x difference in magnetometer trust
whose only symptom was mag_on and mag_off agreeing at every joint. The run now stops instead;
--allow-stale-angles overrides it.

The summary is the CONCATENATION of the per-trial tables, not a second aggregation over
pooled samples. `compute_error_stats` already groups per trial, so the two are the same
numbers — and on Al Borno, where a subject has one trial per activity, this reproduces what
the pre-dataset pipeline wrote to all_subject_statistics.parquet row for row. The pooled
every-sample joint-angle table is what --pooled-angles adds, off by default: it is ~1 GB for
Al Borno and roughly ten times that for IMoVE, and nothing in the summary needs it.

ANATOMICAL AXES ARE A STATISTICS-STAGE OPTION, not a different run. `--anatomical-axes` splits
each error along the parent segment's flexion, adduction and rotation axes as well as its
magnitude, using the bases `experiments/anatomical_frames.py` measures from the source mocap's
landmarks. No filter is re-run and nothing existing is replaced — the new `axis` values sit beside
'MAG', 'X', 'Y', 'Z' — so `--stats-only --anatomical-axes` adds them to a finished benchmark.

THE PARQUET IS THE INTERFACE. Trials are enumerated from results/trials/<dataset>/ and read
through `experiment_utils.load_trial`, which refuses a stale artifact rather than parsing from
source. Build first:

    python -m experiments.build_trials --dataset <name>
"""
import os
os.environ["DISABLE_TQDM"] = "True"
import argparse
from functools import partial

import pandas as pd

import paths
from experiments.experiment_utils import (
    ALLOW_STALE_ENV, METHODS, compute_error_stats, generate_joint_angles_worker,
    compute_stats_worker, load_all_joint_angles, load_per_trial_statistics, resolve_method_spec,
    resolve_stds, run_tracked_grid, save_statistics,
)
from experiments.global_assumptions import (ALBORNO, DATASETS, orphaned_trials, select_trials,
                                            tracking_spec)

# Namespaces the per-trial statistics tree; the dataset is a directory inside it, and the
# top-level summary carries the dataset in its filename instead (see paths.statistics_path).
STATS_NAME = "all_subject"

# ==============================================================================
# CLI / ORCHESTRATOR
# ==============================================================================


def resolve_methods(requested, tracking, explicit: bool) -> list:
    """The methods to run, dropping the unsupported ones only when they were not asked for.

    The asymmetry is the point. A default method list should adapt to the dataset — otherwise
    `--dataset imove_biplane` with no --methods is simply unrunnable — but an EXPLICIT request
    for an arm this dataset cannot support has to fail, because the alternative is a run that
    silently covers less than the command line says it does. Whatever is dropped is named.
    """
    supported, dropped = [], []
    for method in requested:
        try:
            tracking.check_method(method)
        except ValueError as e:
            if explicit:
                raise
            dropped.append((method, str(e).split('.')[0]))
            continue
        supported.append(method)
    for method, reason in dropped:
        print(f"Dropping '{method}': {reason}.")
    return supported


def unscored_trials(summary_df, row_keys, tracking) -> list:
    """The requested trials that contributed no row to the summary.

    NO SILENT SHORTFALL. The summary is a concatenation of whatever per-trial tables exist, so a
    trial that failed or was skipped simply is not in it — and "Aggregated 377" against 379
    requested reads as complete unless the gap is named. Two of the biplane half's trials are
    legitimately in this state (built with their Vicon plates only, so the bone-pose spec has no
    reference in them at all), which is why the count alone is not enough.

    Matched on the LABELLED subject, because that is what the statistics tables carry: Al Borno's
    rows say 'Subject01' where its row_keys say '01' (see TrackingSpec.label_subject). Comparing
    the raw id against them reported all 19 trials as missing on the one dataset where the two
    differ — while the line above it said 19 aggregated.
    """
    if summary_df.empty or 'trial' not in summary_df.columns:
        return list(row_keys)
    scored = {(str(subject), str(trial)) for subject, trial
              in summary_df[['subject', 'trial']].drop_duplicates().to_numpy()}
    return [(subject, trial) for subject, trial in row_keys
            if (tracking.label_subject(subject), trial) not in scored]


# ==============================================================================
# Filter tuning: the rungs of experiments/filter_gains' information ladder
# ==============================================================================
# The shipped default is `experiment_utils.DATASET_STDS`, chosen by that experiment's sweep and
# used when --tuning is absent. --tuning re-runs the benchmark at one of the OTHER rungs, which
# is how a claim like "a static recording gets you within X of the acausal optimum" becomes a
# benchmark number rather than a sweep number.
#
# Read from the ladder table rather than re-derived or hardcoded, so the triple here is exactly
# the one that experiment scored. That matters because the rungs are MEASURED per dataset: the
# static floor is 0.0079/0.0044/0.0035 on Al Borno and 0.0085/0.0040/0.0046 on IMoVE, and a
# hardcoded copy would silently apply one dataset's instrument noise to another's.

def _slug(name: str) -> str:
    """'static mag x5', 'static_mag_x5' and 'Static Mag X5' are the same rung."""
    return str(name).strip().lower().replace(' ', '_')


def variant_name(tuning, normalized: bool):
    """The output namespace for a (tuning, normalization) pair, or None for the shipped default.

    THE ONE DEFINITION. The benchmark writes joint angles, per-trial statistics and the summary
    under this name, and `plotting/paper_figures.py` has to find the same string to draw them —
    two independent spellings of it would mean a figure silently drawn from the default-tuned
    file while its directory claimed otherwise. None keeps every path exactly where the
    pre-tuning pipeline put it.
    """
    if not tuning and not normalized:
        return None
    return '_'.join(filter(None, [_slug(tuning) if tuning else None,
                                  'normalized' if normalized else None]))


def ladder_tuning(dataset: str, rung: str, normalized: bool) -> dict:
    """The three stds for one rung of the ladder, IN THE UNITS THIS RUN'S FILTERS CONSUME.

    The arm decides the units, not the rung: a normalizing filter scales each vector measurement
    to unit length, so its accelerometer std is the physical one divided by |g| (see
    filter_gains.stds_for_arm). The ladder stores each arm's own converted triple, so this reads
    the matching arm instead of converting here — handing physical stds to a normalizing filter
    de-weights its accelerometer by 9.81, which is the mistake that experiment exists to document.

    `relative_normalized` and `ekf` hold identical triples (both normalize), which is what makes
    one triple valid for a whole run whose every filter arm normalizes.
    """
    arm = 'relative_normalized' if normalized else 'relative'
    path = paths.statistics_path(f"filter_gain_ladder_{dataset}_{arm}")
    if not path.exists():
        raise ValueError(
            f"No filter-gain ladder for {dataset}/{arm} at {path.name}. Measure one with "
            f"`python -m experiments.filter_gains --dataset {dataset} --arm {arm}`.")
    frame = pd.read_parquet(path, columns=['tuning', 'mag_mode', 'gyro_std', 'acc_std', 'mag_std'])
    match = frame[frame['tuning'].map(_slug) == _slug(rung)]
    if match.empty:
        available = ', '.join(sorted({_slug(v) for v in frame['tuning'].unique()}))
        raise ValueError(
            f"Unknown tuning '{rung}' for {dataset}/{arm}. Available: {available}.\n"
            f"Note that the 'static mag xN' rungs exist only where there IS a magnetometer to "
            f"measure a floor for — inflating an absent sensor's noise floor is not defined, so "
            f"the biplane datasets offer 'static_floor' and not 'static_mag_x5'.")
    triple = match[['gyro_std', 'acc_std', 'mag_std']].drop_duplicates()
    if len(triple) > 1:
        # 'swept optimum' is each mag_mode's own argmin, so it is not one tuning.
        raise ValueError(
            f"Tuning '{rung}' on {dataset}/{arm} differs by mag_mode ({len(triple)} distinct "
            f"triples), so it cannot be applied to a whole benchmark run. It is a per-arm "
            f"optimum — score it with `python -m experiments.filter_gains --stage ladder`.")
    return {key: float(value) for key, value in triple.iloc[0].items()}


def normalized_method_names(methods: list) -> list:
    """Every filter arm switched to its measurement-normalizing form.

    'marker' runs no filter and 'ekf' already normalizes by default (see METHODS), so both pass
    through. Anything else gets the '_normalized' suffix — but only if it is a BARE base name,
    because the suffix grammar is positional: '_normalized' comes directly after the base, so
    appending it to 'mag_on_perfect_mag' produces a name that does not parse. Such an arm has to
    be spelled out on the command line, and this says so rather than building a broken name.
    """
    out = []
    for method in methods:
        if method == 'marker' or method.startswith('ekf'):
            out.append(method)
        elif method in METHODS:
            out.append(f"{method}_normalized")
        else:
            raise ValueError(
                f"--normalized cannot be applied to '{method}' automatically: the normalization "
                f"suffix must come directly after the method base, so name the arm explicitly "
                f"(e.g. 'mag_on_normalized_perfect_mag') instead of relying on --normalized.")
    return out


# The spec fields worth recording per method — everything the method-name grammar can express
# about WHICH FILTER RAN. Listed rather than dumped wholesale so the table's columns are stable
# across method kinds: 'marker' has no mag_mode, the EKF has no project flag, and a spec dict
# straight from `resolve_method_spec` would give a ragged frame.
SPEC_FIELDS = ('kind', 'project', 'mag_mode', 'acc_source', 'mag_source',
               'normalize_measurements', 'rescale_stds', 'mag_adapt_threshold',
               'mag_distortion_scale')

STD_FIELDS = ('gyro_std', 'acc_std', 'mag_std')


def filter_configuration(methods, tracking, stds, variant, row_keys, tuning_label) -> pd.DataFrame:
    """One row per method: which filter it is, what tuning THIS RUN asked for, and what tuning
    the parquets on disk were actually computed with.

    The last part is the point, and it is why this reads the sidecars instead of just reporting
    its own arguments. `compute_stats_worker` stamps the statistics manifest with the tuning the
    command line requested whether or not the angles were regenerated, so on a `--stats-only`
    run — or any run whose generation phase partly failed — the manifest can describe a filter
    that produced none of the numbers underneath it. That is not hypothetical: Al Borno's cached
    angles carried acc_std=0.018 / mag_std=0.05 while the summary claimed 0.09695 / 0.009695, a
    28x difference in magnetometer trust, and the "mag_on and mag_off agree at every joint" that
    came out of it was pure artifact.

    `disk_*` columns are the MEDIAN over the trials' sidecars where they agree, and `n_disagree`
    counts trials whose recorded constants differ from this run's. A method with no parquet at
    all reports n_found = 0 rather than being dropped, because "this arm was never written" and
    "this arm matches" must not look the same.
    """
    requested = resolve_stds(stds, dataset=tracking.dataset)
    rows = []
    for method in methods:
        spec = resolve_method_spec(method)
        row = {'dataset': tracking.dataset, 'method': method, 'variant': variant or '',
               'tuning': tuning_label,
               **{f'spec_{field}': spec.get(field) for field in SPEC_FIELDS},
               **{f'requested_{field}': requested[field] for field in STD_FIELDS}}

        # 'marker' runs no filter, so it has no tuning to agree or disagree about.
        on_disk, n_found = [], 0
        if spec['kind'] != 'marker':
            for subject, trial in row_keys:
                manifest = paths.read_manifest(
                    paths.joint_angles_path(tracking.dataset, subject, trial, method,
                                            variant=variant,
                                            experiment=paths.BENCHMARK_EXPERIMENT))
                if manifest is None:
                    continue
                n_found += 1
                constants = manifest.get('constants') or {}
                on_disk.append(tuple(constants.get(field) for field in STD_FIELDS))
        row['n_found'] = n_found
        if on_disk:
            common = max(set(on_disk), key=on_disk.count)
            row.update({f'disk_{field}': value for field, value in zip(STD_FIELDS, common)})
            row['n_disagree'] = sum(1 for entry in on_disk if entry != common)
            row['matches_request'] = all(
                _close(disk, requested[field]) for disk, field in zip(common, STD_FIELDS))
        else:
            row.update({f'disk_{field}': None for field in STD_FIELDS})
            row['n_disagree'] = 0
            # None, not True: an arm with nothing on disk has not been checked, and calling that
            # a match is how a missing arm gets read as a verified one.
            row['matches_request'] = None if spec['kind'] != 'marker' else True
        rows.append(row)
    return pd.DataFrame(rows)


def _close(a, b, tol: float = 1e-9) -> bool:
    """Float comparison for stds read back out of JSON, where a round trip is not exact."""
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def report_filter_mismatch(config: pd.DataFrame) -> bool:
    """Prints any arm whose parquets were computed at a different tuning than this run asked for.
    Returns True if anything is wrong.

    This is the check the provenance was always capable of supporting and nobody was running: the
    sidecars recorded the constants all along, so the mismatch was discoverable by hand and
    therefore went undiscovered.
    """
    filtered = config[config['spec_kind'] != 'marker']
    stale = filtered[filtered['matches_request'] == False]          # noqa: E712 — None is distinct
    missing = filtered[filtered['matches_request'].isna()]
    mixed = filtered[filtered['n_disagree'] > 0]

    for _, row in stale.iterrows():
        print(f"  STALE  {row['method']}: on disk gyro/acc/mag = "
              f"{row['disk_gyro_std']:.6g}/{row['disk_acc_std']:.6g}/{row['disk_mag_std']:.6g}, "
              f"this run asked for {row['requested_gyro_std']:.6g}/"
              f"{row['requested_acc_std']:.6g}/{row['requested_mag_std']:.6g}")
    for _, row in mixed.iterrows():
        print(f"  MIXED  {row['method']}: {int(row['n_disagree'])} of {int(row['n_found'])} "
              f"trial(s) carry different constants — two estimators under one name")
    for _, row in missing.iterrows():
        print(f"  ABSENT {row['method']}: no joint angles on disk for any requested trial")

    if len(stale) or len(mixed):
        print("\nThe statistics below do NOT describe the filter this run configured. Re-run "
              "without --stats-only to regenerate the joint angles.")
        return True
    return bool(len(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=ALBORNO.name, choices=sorted(DATASETS),
                        help="Which built dataset to benchmark.")
    parser.add_argument("--subjects", nargs='+', default=None,
                        help="Restrict to these subject/session ids (default: everything built).")
    parser.add_argument("--trials", nargs='+', default=None,
                        help="Restrict to these trial names (default: everything built).")
    parser.add_argument("--methods", nargs='+', default=None,
                        help=f"Methods to process (default: {', '.join(METHODS)}, minus any the "
                             f"dataset cannot support).")
    parser.add_argument("--stats-only", action="store_true",
                        help="Skip joint-angle generation and only compile statistics.")
    parser.add_argument("--pooled-angles", action="store_true",
                        help="Also write the concatenated every-sample joint-angle table. Large "
                             "(~1 GB on alborno, ~10x on imove) and needed by nothing in the "
                             "summary — the time-series figures read it.")
    parser.add_argument("--tuning", default=None,
                        help="Re-run at a named rung of experiments/filter_gains' information "
                             "ladder (e.g. static_floor, static_mag_x5, innovation) instead of "
                             "the shipped DATASET_STDS. Output is namespaced under a variant, so "
                             "it never overwrites the default-tuned run.")
    parser.add_argument("--anatomical-axes", action="store_true",
                        help="Also break each error down along the parent segment's flexion, "
                             "adduction and rotation axes, as extra 'axis' rows beside MAG/X/Y/Z. "
                             "Needs `python -m experiments.anatomical_frames --dataset <name>` "
                             "to have run; the filters are untouched, so --stats-only is enough "
                             "to add them to a finished run.")
    parser.add_argument("--normalized", action="store_true",
                        help="Run every filter arm with unit-length acc/mag measurements: the "
                             "relative arms get their '_normalized' names and the stds are taken "
                             "in normalized units. 'ekf' already normalizes by default.")
    parser.add_argument("--allow-stale-angles", action="store_true",
                        help="Aggregate even when the joint angles on disk were computed at a "
                             "different tuning than this run configures. The summary then "
                             "misreports its own provenance — see the filters table.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    tracking = tracking_spec(args.dataset)

    # Fail before the grid rather than 19 trials into it: every worker would raise the same
    # FileNotFoundError, and a per-trial failure in that table reads like a data problem.
    if args.anatomical_axes:
        from experiments.anatomical_frames import bases_path, load_bases
        if load_bases(tracking.dataset) is None:
            print(f"Error: --anatomical-axes needs {bases_path(tracking.dataset)}. Run: "
                  f"python -m experiments.anatomical_frames --dataset {args.dataset}")
            return 1

    # THE VARIANT IS NOT OPTIONAL once the tuning moves. `results/joint_angles/` is keyed by
    # dataset, subject, trial and method — not by tuning — so a re-tuned run without one
    # overwrites the default-tuned parquets in place, and the only surviving evidence of which
    # estimator produced them is the manifest sidecar. The per-trial statistics tree and the
    # summary filename carry the same variant for the same reason.
    variant = variant_name(args.tuning, args.normalized)
    stds = None
    if args.tuning:
        try:
            stds = ladder_tuning(args.dataset, args.tuning, args.normalized)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    stats_name = STATS_NAME if variant is None else f"{STATS_NAME}_{variant}"

    # Validate up front, so a typo fails fast instead of surfacing as a per-cell
    # "Failed" status after workers have already spun up.
    explicit_methods = args.methods is not None
    for m in (args.methods or []):
        try:
            resolve_method_spec(m)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    try:
        methods = resolve_methods(args.methods or list(METHODS), tracking, explicit_methods)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    if not methods:
        print(f"Error: no requested method is supported on {args.dataset}.")
        return 1
    if args.normalized:
        try:
            methods = normalized_method_names(methods)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    if 'marker' not in methods:
        print("Note: 'marker' is not in the method list, so there is no ground truth to score "
              "against and the statistics phase will find nothing to merge.")

    try:
        row_keys = select_trials(args.dataset, args.subjects, args.trials)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    orphans = orphaned_trials(args.dataset)
    if orphans:
        names = ", ".join(f"{s}/{t}" for s, t in orphans[:3])
        print(f"Skipping {len(orphans)} built parquet(s) the {tracking.build_dataset} source no "
              f"longer enumerates ({names}{', …' if len(orphans) > 3 else ''}) — nothing can "
              f"rebuild them, so they are not failures.")

    print(f"Benchmarking {args.dataset}: {len(row_keys)} trial(s), "
          f"{len(tracking.joints)} joint(s), methods {', '.join(methods)}")
    if variant is None:
        print(f"Tuning: shipped DATASET_STDS['{args.dataset}'] "
              f"= {resolve_stds(dataset=args.dataset)}")
    else:
        shown = stds if stds is not None else resolve_stds(dataset=args.dataset)
        units = 'normalized' if args.normalized else 'physical'
        print(f"Tuning: {args.tuning or 'shipped'} ({units} units) = "
              f"{ {k: round(v, 6) for k, v in shown.items()} }")
        print(f"Writing under variant '{variant}', so the default-tuned run is untouched.")

    # 1. Joint-angle generation phase
    if not args.stats_only:
        print(f"Starting parallel generation of joint angles for {len(row_keys)} tasks using "
              f"{args.workers} workers...")
        state, _ = run_tracked_grid(row_keys, ['Subject', 'Trial'], ['load'] + methods,
                         partial(generate_joint_angles_worker, tracking=tracking,
                                 stds=stds, variant=variant,
                                 experiment=paths.BENCHMARK_EXPERIMENT),
                         args.workers, title=f"MAJIC MOCAP GENERATION ({args.dataset})")

        # A GENERATION PHASE THAT PRODUCED NOTHING IS NOT A SUCCESSFUL RUN. The statistics phase
        # reads whatever parquets are on disk, so when every cell fails it happily scores the
        # PREVIOUS run's arms and prints "Aggregated N trial(s)" over them. Observed for real: a
        # stale trial cache failed all 19 Al Borno trials while the summary reported complete
        # coverage from parquets three days old, which is indistinguishable from a good run and
        # is the same trap `--stats-only` sets deliberately.
        #
        # Only the all-failed case stops the run. A partial failure is a legitimate state — two
        # biplane trials have no bone poses at all — and is already named in the grid and in the
        # shortfall report.
        outcomes = [str(state.get((key, method), '')) for key in row_keys for method in methods]
        failed = [o for o in outcomes if o.startswith('Failed')]
        if outcomes and len(failed) == len(outcomes):
            print(f"\nError: every one of the {len(outcomes)} (trial, method) cells failed, so "
                  f"nothing was written. Refusing to compile statistics — they would score "
                  f"whatever the previous run left on disk and report it as this run's result.")
            reasons = sorted({o for o in failed})[:2]
            for reason in reasons:
                print(f"  {reason[:200]}")
            return 1
        print("Joint angles generation phase complete.\n")

    # 1b. The filter configuration this run's numbers actually rest on.
    #
    # Built BEFORE the statistics, because its whole value is the power to stop them. The
    # statistics phase reads whatever joint angles are on disk and `compute_stats_worker` stamps
    # its manifest with the tuning the command line asked for, so a run whose angles came from a
    # different filter produces a summary that misdescribes itself — and there is no symptom in
    # the output. Checking here turns the sidecars, which recorded the constants all along, into
    # a gate rather than an audit trail nobody walks.
    print("--- Filter configuration ---")
    filter_config = filter_configuration(methods, tracking, stds, variant, row_keys,
                                         args.tuning or 'shipped')
    config_path = paths.ensure_parent(paths.filter_config_path(stats_name, args.dataset))
    filter_config.to_parquet(config_path, engine='pyarrow', index=False)
    paths.write_manifest(config_path, constants=resolve_stds(stds, dataset=tracking.dataset),
                         dataset=args.dataset, variant=variant,
                         tuning=args.tuning or 'shipped', methods=methods,
                         n_trials_requested=len(row_keys),
                         world_frame_gravity=tracking.gravity.tolist(),
                         has_magnetometer=tracking.has_magnetometer,
                         joints=sorted(tracking.joints))
    print(f"Filter configuration saved to {config_path}")

    if report_filter_mismatch(filter_config):
        if not args.allow_stale_angles:
            print("\nRefusing to continue. Pass --allow-stale-angles to aggregate anyway "
                  "(the summary will then misreport its own tuning).")
            return 1
        print("\n--allow-stale-angles: continuing over the mismatch above.")

    # The flag also has to reach `experiment_utils`' own freshness checks, which now refuse a
    # stale artifact at every read rather than only here. Through the environment because the
    # statistics phase does its reading inside spawned worker processes, where a module global
    # set in this one never arrives -- see `stale_artifacts_allowed`.
    if args.allow_stale_angles:
        os.environ[ALLOW_STALE_ENV] = '1'

    # 2. Per-trial statistics phase
    print("--- Starting Statistics Aggregation Phase ---")
    run_tracked_grid(row_keys, ['Subject', 'Trial'], ['stats'],
                     partial(compute_stats_worker, methods=methods, stats_name=stats_name,
                             tracking=tracking, stds=stds, variant=variant,
                             anatomical_axes=args.anatomical_axes,
                             experiment=paths.BENCHMARK_EXPERIMENT),
                     args.workers, title=f"MAJIC MOCAP STATISTICS ({args.dataset})")

    # 3. Global aggregation phase
    print("\n--- Starting Global Aggregation & Statistics Phase ---")

    summary_stats_df = load_per_trial_statistics(
        stats_name, tracking.dataset, row_keys, methods=methods, variant=variant,
        experiment=paths.BENCHMARK_EXPERIMENT, stds=stds, tracking=tracking,
        anatomical_axes=args.anatomical_axes)
    if summary_stats_df.empty:
        print(f"Error: no per-trial statistics were written for {args.dataset}. Exiting.")
        return 1
    covered = summary_stats_df[['subject', 'trial']].drop_duplicates()
    print(f"Aggregated {len(covered)} trial(s) across "
          f"{covered['subject'].nunique()} subject(s).")

    absent = unscored_trials(summary_stats_df, row_keys, tracking)
    if absent:
        names = ", ".join(f"{s}/{t}" for s, t in absent[:5])
        print(f"{len(absent)} of {len(row_keys)} trial(s) contributed no statistics "
              f"({names}{', …' if len(absent) > 5 else ''}). A trial with none of this "
              f"dataset's sensors is skipped rather than failed; anything else is in the "
              f"grid above.")

    # `methods` and `subjects` are not passed: save_statistics reads them off the frame, which
    # is the honest version — what the summary CONTAINS, not what the run asked for.
    stats_path = save_statistics(summary_stats_df, f"{stats_name}_{args.dataset}",
                                 dataset=args.dataset, stds=stds,
                                 tuning=args.tuning, variant=variant,
                                 normalized_measurements=args.normalized,
                                 anatomical_axes=args.anatomical_axes,
                                 joints=sorted(tracking.joints),
                                 world_frame_gravity=tracking.gravity.tolist(),
                                 has_magnetometer=tracking.has_magnetometer,
                                 trials_dataset=tracking.build_dataset,
                                 n_trials=len(covered))
    print(f"Summary statistics saved to {stats_path}")

    if args.pooled_angles:
        print("\n--- Concatenating every-sample joint angles (this is the large one) ---")
        all_data_df = load_all_joint_angles(tracking.dataset, row_keys, methods,
                                           tracking=tracking, variant=variant,
                                           experiment=paths.BENCHMARK_EXPERIMENT, stds=stds)
        if all_data_df.empty:
            print("Warning: no joint angles were loaded, so no pooled table was written.")
        else:
            joint_angles_path = paths.ensure_parent(
                paths.all_subject_joint_angles_path(args.dataset))
            all_data_df.to_parquet(joint_angles_path, engine='pyarrow')
            paths.write_manifest(joint_angles_path, dataset=args.dataset, methods=methods,
                                 subjects=sorted({s for s, _ in row_keys}),
                                 trials=sorted(t for _, t in row_keys),
                                 n_rows=len(all_data_df))
            print(f"Concatenated DataFrame saved to {joint_angles_path}")

    print("\nPipeline finished successfully!")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
