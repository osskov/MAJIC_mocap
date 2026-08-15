"""What the build measures and then throws away.

Every step of a build already computes the numbers that say how well it went, and almost all
of them end up in a `print()` or a discarded local. `reconstruct_plate` returns a dict with
per-marker fault counts and fit residuals; `_shared_lag` estimates a lag per plate and keeps
only the median; `align_world_to_imu` solves a rotation and reports nothing about it. None of
that survives to the artifact, so a question like "which of these 281 trials should I not
trust" can only be answered by rebuilding and watching the terminal.

This is the accumulator that keeps them. It is threaded through the readers and assembly as
an optional argument: `None` means collect nothing, which is the default and costs one
`is not None` per call site, so an ordinary build is unchanged.

LONG FORM -- one row per (step, entity, metric) rather than a wide table per step. The
metrics are heterogeneous (floats, counts, strings, small vectors) and sparse: a plate that
never needed the un-flip pass has no un-flip statistics, and a trial with one mocap take has
no inter-take gap. A wide schema would be mostly nulls and would need migrating every time a
metric is added, whereas the analysis pivots per step anyway.

WHAT DOES NOT GO HERE: per-frame arrays. This sits beside the parquet and is read whole; the
per-frame series that figures want are re-derived by experiments/build_quality.py, which can
decimate them. Keeping a residual per frame per plate per trial here would be tens of GB.

The report is a DESCRIPTION of an artifact, never an input to it. It must not enter
`trial_cache_key`, or adding a metric would invalidate every trial on disk.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# The pipeline steps, in order. Carried on every row so `groupby('step')` reproduces the
# processing order rather than sorting alphabetically into nonsense.
STEPS = (
    'S0_discovery', 'S1_parse', 'S2_reconstruction', 'S3_pairing', 'S4_sync',
    'S5_resampling', 'S6_timeline', 'S7_alignment', 'S8_lever_arm', 'S9_serialization',
)

# Columns of the long-form table. `value_num` and `value_str` are exclusive: a metric is
# numeric or it is not, and splitting them keeps the numeric column a real float column
# rather than an object column that every consumer has to coerce.
COLUMNS = ('step', 'entity_kind', 'entity', 'metric', 'value_num', 'value_str')


@dataclass
class BuildReport:
    """Accumulates per-step build metrics. Optional everywhere; `None` collects nothing."""

    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, step: str, entity_kind: str, entity: str, **metrics: Any) -> None:
        """Record metrics for one entity at one step.

        `entity_kind` is what `entity` names -- 'trial', 'plate', 'segment', 'sensor',
        'marker', 'file', 'take' -- so the analysis can select "everything about plates"
        without parsing names.

        Values are stored numerically where possible. Sequences are expanded into indexed
        metrics (`fault_counts` of length 4 becomes `fault_counts_0..3`) rather than pickled,
        so the table stays queryable in SQL or pandas without unpacking anything.
        """
        for metric, value in metrics.items():
            self.rows.extend(_flatten(step, entity_kind, entity, metric, value))

    def extend(self, other: Optional['BuildReport']) -> None:
        """Absorb another report, for a reader that builds sub-reports per take."""
        if other is not None:
            self.rows.extend(other.rows)

    def to_frame(self):
        """The long-form table. Empty with the right columns when nothing was collected, so
        a caller never has to special-case the empty build."""
        import pandas as pd
        if not self.rows:
            return pd.DataFrame({name: pd.Series(dtype='object') for name in COLUMNS})
        return pd.DataFrame(self.rows, columns=list(COLUMNS))


def _flatten(step: str, entity_kind: str, entity: str, metric: str, value: Any
             ) -> List[Dict[str, Any]]:
    """One metric -> one or more long-form rows."""
    if value is None:
        return []

    if isinstance(value, (str, bytes)):
        return [_row(step, entity_kind, entity, metric, None, str(value))]

    if isinstance(value, (bool, np.bool_)):
        # Before the numeric branch: bool is an int in Python, and storing True as 1.0 loses
        # the distinction between a flag and a count when the analysis reads it back.
        return [_row(step, entity_kind, entity, metric, float(value), None)]

    if isinstance(value, dict):
        return [row for key, item in value.items()
                for row in _flatten(step, entity_kind, entity, f'{metric}_{key}', item)]

    if isinstance(value, (list, tuple, np.ndarray)):
        flat = np.asarray(value).reshape(-1)
        if flat.size == 0:
            return []
        # A long sequence is summarised rather than expanded. Frame indices of unresolved
        # glitches can run to thousands, and one row each would swamp the table with data
        # that belongs in the per-frame tier.
        if flat.size > _MAX_SEQUENCE:
            return [_row(step, entity_kind, entity, f'{metric}_count', float(flat.size), None)]
        return [row for index, item in enumerate(flat)
                for row in _flatten(step, entity_kind, entity, f'{metric}_{index}', item)]

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return [_row(step, entity_kind, entity, metric, None, str(value))]
    return [_row(step, entity_kind, entity, metric, numeric, None)]


_MAX_SEQUENCE = 16


def _row(step: str, entity_kind: str, entity: str, metric: str,
         value_num: Optional[float], value_str: Optional[str]) -> Dict[str, Any]:
    return {'step': step, 'entity_kind': entity_kind, 'entity': entity,
            'metric': metric, 'value_num': value_num, 'value_str': value_str}
