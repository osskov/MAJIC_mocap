"""Per-subject cluster-to-IMU offsets for the taped sensors, derived from a completed build.

WHY PER SUBJECT. A taped sensor's offset is fitted per trial, and where that fit diverges the
build falls back to `NOMINAL_SENSOR_OFFSET_MM` -- one vector per sensor, the median over all 26
subjects. That fallback fires on 10.4% of taped plates and it is systematically wrong for any
subject whose taping differs from the cohort average, in the same direction every time.

Measured by leave-one-out over the 1876 trials whose own fit succeeded: hide a trial's answer,
predict it from the dataset nominal or from the median of that subject's OTHER trials.

    dataset nominal          median error 24.63 mm
    that subject's median    median error 12.29 mm      better in 82.4% of cases

Half the error, consistently across all ten sensors. This script writes the second table.

WHAT THIS DOES NOT CHANGE. Where a trial's own fit succeeds it is still used, because the
sensor genuinely moves between trials -- the within-subject spread does not shrink with
excitation (12.4 / 9.7 / 12.5 mm across low, medium and high) the way estimation noise would,
so pooling everything would smear a real effect. This only replaces the FALLBACK.

THE LOOP THIS CLOSES, AND WHY IT DOES NOT SPIN. Deriving a table from a build the build then
consumes is circular unless the table cannot change what the build measures, and the first
version of this did not clear that bar: it let the subject median serve as the plausibility
gate as well as the fallback, so a better reference admitted more fits, which moved the median,
which moved the gate. Measured over three passes it accepted 1890 then 1951 then 1954 fits
while one sensor swung 13.7 mm between the last two -- drifting, not settling.

`_sensor_offsets` therefore measures the GATE against the cohort nominal, a hand-checked
constant no build writes, and uses this table only for the VALUE substituted when the gate
rejects. The accepted set is then independent of the table and one pass is exact:

    build  ->  python -m experiments.derive_subject_offsets  ->  rebuild

A second iteration reproduces the table byte for byte, which `--check` asserts rather than
anyone trusting.

Writes a generated Python module rather than JSON so the toolchest digest covers it: the cache
key is an AST hash over the build modules, and a data file would let the table change without
invalidating a single artifact.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import paths
from experiments.experiment_utils import build_report_path
from src.toolchest.building.imove_mocap import NOMINAL_SENSOR_OFFSET_MM
from src.toolchest.building.sources import get_source

DATASET = 'imove'
TARGET = (paths.REPO_ROOT / 'src' / 'toolchest' / 'building'
          / 'imove_subject_offsets.py')

# Below this many successful fits a subject's median is itself noise, and the cohort nominal
# -- which is backed by hundreds -- is the better guess. Set where the leave-one-out gain
# stops being reliable rather than picked: at 3 the subject median already halves the error.
MIN_FITS_PER_SUBJECT = 3


def collect(dataset: str = DATASET) -> pd.DataFrame:
    """Every successful per-trial taped fit in the build reports, one row each."""
    rows = []
    for subject, trial in get_section(dataset):
        path = build_report_path(dataset, subject, trial)
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        lever = frame[frame.step == 'S8_lever_arm']
        if lever.empty:
            continue
        wide = lever.pivot_table(index='entity', columns='metric', values='value_num')
        for entity, row in wide.iterrows():
            if row.get('bolted', 0) or row.get('used_fallback', 1):
                continue
            if any(f'offset_mm_{i}' not in row for i in range(3)):
                continue
            rows.append({'subject': subject, 'sensor': entity,
                         **{f'o{i}': float(row[f'offset_mm_{i}']) for i in range(3)}})
    return pd.DataFrame(rows)


def get_section(dataset: str):
    return get_source(dataset).enumerate_trials()


def derive(fits: pd.DataFrame) -> dict:
    """{(subject, sensor): (x, y, z)} for every pairing with enough successful fits.

    COMPONENTWISE MEDIAN, not a mean of magnitudes. A mean is dragged by the diverged tail
    this table exists to protect against, and averaging magnitudes rather than components
    inflates the result whenever the noise is isotropic -- the same mistake that made an
    earlier version of the cohort constants read high.
    """
    table = {}
    for (subject, sensor), group in fits.groupby(['subject', 'sensor']):
        if sensor not in NOMINAL_SENSOR_OFFSET_MM or len(group) < MIN_FITS_PER_SUBJECT:
            continue
        table[(subject, sensor)] = tuple(
            round(float(group[f'o{i}'].median()), 1) for i in range(3))
    return table


def render(table: dict, n_fits: int, n_sessions: int) -> str:
    by_subject = defaultdict(dict)
    for (subject, sensor), value in table.items():
        by_subject[subject][sensor] = value
    lines = [
        '"""Per-subject cluster-to-IMU offsets for IMoVE\'s taped sensors, in millimetres.',
        '',
        'GENERATED by experiments/derive_subject_offsets.py. Do not hand-edit: a rebuild plus',
        'a re-run reproduces it, and an edit here would be silently overwritten.',
        '',
        'Read only when a trial\'s own per-trial fit diverges. Where the fit succeeds it wins,',
        'because a taped sensor genuinely moves between trials. See the deriving script for',
        'the leave-one-out measurement that justifies preferring these over the cohort',
        'nominal: 12.29 mm of error against 24.63.',
        f'',
        f'Derived from {n_fits} successful per-trial fits across {n_sessions} sessions.',
        f'{len(by_subject)} of those sessions reached the minimum fit count for at least one',
        f'sensor and appear below; the rest fall through to the cohort nominal.',
        '"""',
        'from typing import Dict, Tuple',
        '',
        'SUBJECT_SENSOR_OFFSET_MM: Dict[str, Dict[str, Tuple[float, float, float]]] = {',
    ]
    for subject in sorted(by_subject, key=lambda s: (len(s), s)):
        lines.append(f"    {subject!r}: {{")
        for sensor in sorted(by_subject[subject]):
            x, y, z = by_subject[subject][sensor]
            lines.append(f"        {sensor!r}: ({x}, {y}, {z}),")
        lines.append("    },")
    lines += ['}', '']
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=DATASET)
    parser.add_argument('--check', action='store_true',
                        help='fail if the table on disk differs from what this derives')
    args = parser.parse_args()

    fits = collect(args.dataset)
    if fits.empty:
        print('No successful taped fits in the build reports. Build first.')
        return 1
    table = derive(fits)
    text = render(table, len(fits), int(fits.subject.nunique()))

    print(f'{len(fits)} successful fits -> {len(table)} (subject, sensor) offsets '
          f'over {fits.subject.nunique()} sessions')
    covered = {s for _, s in table}
    missing = set(NOMINAL_SENSOR_OFFSET_MM) - covered
    if missing:
        print(f'  no subject reached {MIN_FITS_PER_SUBJECT} fits for: {sorted(missing)}')

    if args.check:
        current = TARGET.read_text() if TARGET.exists() else ''
        if current != text:
            print(f'{TARGET.relative_to(paths.REPO_ROOT)} is out of date. Re-run without '
                  f'--check, then rebuild.')
            return 1
        print('table is current')
        return 0

    TARGET.write_text(text)
    print(f'wrote {TARGET.relative_to(paths.REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
