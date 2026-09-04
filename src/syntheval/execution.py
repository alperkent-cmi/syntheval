"""Structured execution records for metric-level SynthEval runs."""

from __future__ import annotations

import math
import traceback as traceback_module
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence


TERMINAL_STATES = frozenset(
    {"succeeded", "failed", "timed_out", "blocked", "not_applicable"}
)

_OPTIONAL_DIAGNOSTIC_PREFIXES = {
    "auroc": ("auroc_",),
    "auroc_v2": ("auroc_",),
    "statistical_parity": ("sp_",),
    "equalized_odds": ("eqo_",),
    "equal_opportunity": ("eo_",),
}


def _tuple(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in (values or ())))


def _finite(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _row_key(row: Any) -> Optional[str]:
    if isinstance(row, Mapping):
        value = row.get("metric")
    elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        value = row[0] if row else None
    else:
        value = None
    return None if value is None else str(value)


def _row_is_finite(row: Any) -> bool:
    if isinstance(row, Mapping):
        values = (row.get(name) for name in ("val", "err", "n_val", "n_err"))
    elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        values = row[2:6]
    else:
        return False
    return all(_finite(value) for value in values)


def _is_optional_diagnostic(key: str, expected: Sequence[str]) -> bool:
    return any(
        key.startswith(prefix)
        for expected_key in expected
        for prefix in _OPTIONAL_DIAGNOSTIC_PREFIXES.get(expected_key, ())
    )


@dataclass(frozen=True)
class MetricExecutionStatus:
    """Terminal status and key accounting for one metric method."""

    method: str
    state: str
    expected_keys: tuple[str, ...] = ()
    observed_keys: tuple[str, ...] = ()
    completed_keys: tuple[str, ...] = ()
    failed_keys: tuple[str, ...] = ()
    missing_keys: tuple[str, ...] = ()
    duplicate_keys: tuple[str, ...] = ()
    non_finite_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.state not in TERMINAL_STATES:
            raise ValueError(f"Unknown terminal metric state: {self.state!r}")

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def execution_complete(self) -> bool:
        """Whether all expected keys received a terminal outcome.

        A failed, missing, duplicate, or non-finite key is complete audit
        evidence but is not policy-eligible. This distinction lets callers
        reuse a diagnosed failure without treating it as a successful result.
        """
        return self.terminal and all(
            key in self.failed_keys or key in self.completed_keys
            for key in self.expected_keys
        )

    @property
    def succeeded(self) -> bool:
        """Whether this metric produced a complete, valid result."""
        return (
            self.state == "succeeded"
            and self.execution_complete
            and not self.failed_keys
            and not self.missing_keys
            and not self.duplicate_keys
            and not self.non_finite_keys
            and not self.unexpected_keys
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricExecution:
    """Raw and normalized outputs associated with one metric status."""

    method: str
    status: MetricExecutionStatus
    raw_result: Any = None
    formatted_output: Any = None
    normalized_rows: tuple[Mapping[str, Any], ...] = ()
    normalized_rows_v2: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.to_dict()
        result["normalized_rows"] = [dict(row) for row in self.normalized_rows]
        result["normalized_rows_v2"] = [dict(row) for row in self.normalized_rows_v2]
        return result


@dataclass(frozen=True)
class SynthEvalExecution:
    """Complete execution bundle for one SynthEval pass."""

    schema_version: str = "syntheval-execution-v1"
    pass_id: str = "native"
    target_view: str = "native"
    expected_manifest_digest: Optional[str] = None
    metric_executions: tuple[MetricExecution, ...] = ()
    normalized_table: Any = None
    normalized_table_v2: Any = None
    execution_complete: bool = False
    policy_eligible: bool = False
    preprocessing_fingerprint: Optional[str] = None
    preprocessing_metadata: Optional[Mapping[str, Any]] = None

    @property
    def succeeded(self) -> bool:
        """Whether every metric in this pass completed successfully."""
        return bool(self.metric_executions) and self.execution_complete and all(
            item.status.succeeded for item in self.metric_executions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pass_id": self.pass_id,
            "target_view": self.target_view,
            "expected_manifest_digest": self.expected_manifest_digest,
            "metric_executions": [item.to_dict() for item in self.metric_executions],
            "execution_complete": self.execution_complete,
            "succeeded": self.succeeded,
            "policy_eligible": self.policy_eligible,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "preprocessing_metadata": self.preprocessing_metadata,
        }


def build_metric_execution(
    method: str,
    key_result: Any,
    expected_keys: Optional[Sequence[str]] = None,
    *,
    status_key_result: Any = None,
    normalized_rows_v2: Any = None,
    raw_result: Any = None,
    formatted_output: Any = None,
    error: Optional[BaseException] = None,
    timed_out: bool = False,
    warnings_list: Optional[Sequence[str]] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
) -> MetricExecution:
    """Build a terminal metric record from legacy and versioned outputs."""
    expected = _tuple(expected_keys)
    legacy_rows = tuple(row for row in (key_result or ()) if isinstance(row, Mapping))
    v2_rows = tuple(row for row in (normalized_rows_v2 or ()) if isinstance(row, Mapping))
    status_rows = (
        legacy_rows
        if status_key_result is None
        else tuple(row for row in (status_key_result or ()) if isinstance(row, Mapping))
    )
    observed = tuple(_row_key(row) for row in status_rows)
    observed = tuple(key for key in observed if key is not None)
    counts = {key: observed.count(key) for key in set(observed)}
    duplicate = tuple(key for key, count in counts.items() if count > 1)
    non_finite = tuple(
        key for key, row in zip(observed, status_rows) if key is not None and not _row_is_finite(row)
    )
    unexpected = tuple(
        key
        for key in observed
        if expected and key not in expected and not _is_optional_diagnostic(key, expected)
    )
    missing = tuple(key for key in expected if key not in observed)
    failed = tuple(
        key
        for key in expected
        if key in missing or key in duplicate or key in non_finite
    )

    exception_type = None
    exception_message = None
    exception_traceback = None
    if error is not None:
        exception_type = type(error).__name__
        exception_message = str(error)
        exception_traceback = "".join(
            traceback_module.format_exception(type(error), error, error.__traceback__)
        )
        failed = expected or observed

    if timed_out:
        state = "timed_out"
        failed = expected or observed
    elif error is not None or failed or duplicate or non_finite or unexpected:
        state = "failed"
    elif expected and missing:
        state = "failed"
    else:
        state = "succeeded"

    completed = tuple(key for key in expected or observed if key in observed and key not in failed)
    status = MetricExecutionStatus(
        method=method,
        state=state,
        expected_keys=expected,
        observed_keys=observed,
        completed_keys=completed,
        failed_keys=_tuple(failed),
        missing_keys=missing,
        duplicate_keys=duplicate,
        non_finite_keys=non_finite,
        unexpected_keys=unexpected,
        warnings=_tuple(warnings_list),
        exception_type=exception_type,
        exception_message=exception_message,
        exception_traceback=exception_traceback,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed_seconds,
    )
    return MetricExecution(
        method=method,
        status=status,
        raw_result=raw_result,
        formatted_output=formatted_output,
        normalized_rows=legacy_rows,
        normalized_rows_v2=v2_rows,
    )


def manifest_for_methods(
    manifest: Optional[Mapping[str, Sequence[str]]], methods: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Normalize a method-to-local-key manifest without adding observed keys."""
    if manifest is None:
        return {method: () for method in methods}
    return {method: _tuple(manifest.get(method, ())) for method in methods}