import math

import pandas as pd

from syntheval.execution import build_metric_execution
from syntheval.syntheval import SynthEval
import syntheval.syntheval as syntheval_module


def test_metric_execution_accounts_for_missing_duplicate_and_unexpected_keys():
    execution = build_metric_execution(
        "metric_a",
        [
            {"metric": "expected", "val": 0.5},
            {"metric": "expected", "val": 0.6},
            {"metric": "extra", "val": 0.1},
        ],
        expected_keys=["expected", "missing"],
    )

    assert execution.status.state == "failed"
    assert execution.status.completed_keys == ()
    assert execution.status.succeeded is False
    assert execution.status.duplicate_keys == ("expected",)
    assert execution.status.missing_keys == ("missing",)
    assert execution.status.unexpected_keys == ("extra",)
    assert execution.status.execution_complete is True


def test_metric_execution_preserves_typed_failure_diagnostics():
    error = ValueError("bad metric input")
    execution = build_metric_execution(
        "metric_a",
        None,
        expected_keys=["expected"],
        error=error,
        warnings_list=["unstable estimate"],
    )

    assert execution.status.state == "failed"
    assert execution.status.failed_keys == ("expected",)
    assert execution.status.exception_type == "ValueError"
    assert execution.status.exception_message == "bad metric input"
    assert "ValueError: bad metric input" in execution.status.exception_traceback
    assert execution.status.warnings == ("unstable estimate",)


def test_metric_execution_accepts_declared_qualified_diagnostics():
    execution = build_metric_execution(
        "equalized_odds",
        [
            {"metric": "equalized_odds", "val": 0.1},
            {"metric": "eqo_target_group", "val": 0.2},
        ],
        expected_keys=["equalized_odds"],
    )

    assert execution.status.state == "succeeded"
    assert execution.status.succeeded is True
    assert execution.status.unexpected_keys == ()
    assert execution.status.observed_keys == ("equalized_odds", "eqo_target_group")


def test_non_finite_output_is_failed():
    execution = build_metric_execution(
        "metric_a",
        [{"metric": "expected", "val": math.nan}],
        expected_keys=["expected"],
    )

    assert execution.status.state == "failed"
    assert execution.status.non_finite_keys == ("expected",)


def test_metric_execution_keeps_legacy_rows_and_validates_v2_rows():
    legacy_rows = [{"metric": "legacy_metric", "val": 0.4}]
    v2_rows = [{"metric": "v2_metric", "val": 0.6, "n_val": 0.9}]

    execution = build_metric_execution(
        "metric_a",
        legacy_rows,
        expected_keys=["v2_metric"],
        status_key_result=v2_rows,
        normalized_rows_v2=v2_rows,
    )

    assert execution.status.state == "succeeded"
    assert execution.status.succeeded is True
    assert execution.status.completed_keys == ("v2_metric",)
    assert execution.normalized_rows == tuple(legacy_rows)
    assert execution.normalized_rows_v2 == tuple(v2_rows)


def test_structured_evaluation_forwards_group_context_to_metric(monkeypatch):
    real = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]})
    group_context = {
        "group_mode": "patient_group",
        "population_unit": "patient_group",
        "group_column": "patient_id",
    }

    class CaptureMetric:
        received_group_context = None

        def __init__(self, **kwargs):
            type(self).received_group_context = kwargs.get("group_context")

        def evaluate(self, **kwargs):
            return {"ok": True}

        def format_output(self):
            return []

        def normalize_output(self):
            return [{"metric": "capture", "dim": "u", "val": 0.5, "err": 0.0, "n_val": 0.5, "n_err": 0.0}]

        def normalize_output_v2(self):
            return [{"metric": "capture", "dim": "u", "val": 0.5, "err": 0.0, "n_val": 0.5, "n_err": 0.0}]

    monkeypatch.setitem(syntheval_module.loaded_metrics, "capture", CaptureMetric)
    evaluator = SynthEval(
        real,
        cat_cols=["target"],
        verbose=False,
        enable_plots=False,
        console="off",
    )

    execution = evaluator.evaluate(
        real.copy(),
        return_execution=True,
        expected_output_manifest={"capture": ("capture",)},
        group_context=group_context,
        capture={},
    )

    assert execution.execution_complete is True
    assert execution.metric_executions[0].status.state == "succeeded"
    assert execution.normalized_table_v2 is not None
    assert execution.normalized_table_v2.iloc[0]["metric"] == "capture"
    assert execution.preprocessing_fingerprint
    assert execution.preprocessing_metadata["fit_role"] == "train"
    assert execution.preprocessing_metadata["fingerprint"] == execution.preprocessing_fingerprint
    assert CaptureMetric.received_group_context == group_context