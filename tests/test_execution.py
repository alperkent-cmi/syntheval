import math

import pandas as pd

from syntheval.execution import build_metric_execution
from syntheval.metrics.fairness.metric_equal_opportunity import EqualOpportunity
from syntheval.metrics.fairness.metric_equalized_odds import EqualizedOdds
from syntheval.metrics.fairness.metric_statistical_parity import StatisticalParity
from syntheval.metrics.privacy.metric_AttrDis import AttributeDisclosure
from syntheval.syntheval import SynthEval
import syntheval.syntheval as syntheval_module
from syntheval.utils.configuration import AnalysisConfig, _analysis_target_parser


def test_analysis_config_keeps_sensitive_and_protected_roles_separate():
    data = pd.DataFrame({"target": [0, 1], "secret": ["a", "b"], "group": [0, 1]})

    config = AnalysisConfig(
        data, "target", sensitive_vars=["secret"], protected_vars=["group"]
    )

    assert config.sensitive_vars == ["secret"]
    assert config.protected_vars == ["group"]


def test_analysis_config_omitted_protected_vars_are_empty_and_json_loading_is_safe(tmp_path):
    data = pd.DataFrame({"target": [0, 1], "secret": ["a", "b"]})
    config = AnalysisConfig(data, "target", sensitive_vars=["secret"])
    assert config.sensitive_vars == ["secret"]
    assert config.protected_vars == []

    path = tmp_path / "legacy.json"
    path.write_text(
        '{"target_vars": ["target"], "target_types": {"target": 2}, '
        '"confounder_vars": {"target": []}, "sensitive_vars": ["secret"]}'
    )
    loaded = _analysis_target_parser(data, str(path))
    assert loaded.sensitive_vars == ["secret"]
    assert loaded.protected_vars == []


def test_analysis_config_keeps_sensitive_vars_for_disclosure_when_protected_is_omitted():
    data = pd.DataFrame({"target": [0, 1], "secret": ["a", "b"]})
    config = AnalysisConfig(data, "target", sensitive_vars=["secret"])

    assert config.sensitive_vars == ["secret"]
    assert config.protected_vars == []


def test_attribute_disclosure_keeps_using_sensitive_vars_when_protected_is_omitted():
    data = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "secret": ["a", "b", "a", "b"],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    config = AnalysisConfig(data, "target", sensitive_vars=["secret"])
    metric = AttributeDisclosure(
        data,
        data.copy(),
        cat_cols=["target", "secret"],
        num_cols=["feature"],
        analysis_target=config,
        do_preprocessing=False,
    )

    metric.evaluate()

    assert metric.analysis_target.sensitive_vars == ["secret"]
    assert metric.analysis_target.protected_vars == []


def test_fairness_reports_missing_protected_input_instead_of_using_sensitive_vars():
    data = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "secret": [0, 1, 0, 1],
        }
    )
    config = AnalysisConfig(data, "target", sensitive_vars=["secret"])

    try:
        StatisticalParity(
            data, data, analysis_target=config, do_preprocessing=False
        ).evaluate(folds=2)
    except ValueError as error:
        assert str(error) == (
            "SynthEval(stat parity): metric did not run, "
            "no protected variable specified!"
        )
    else:
        raise AssertionError("fairness metric unexpectedly used sensitive_vars")


def test_analysis_config_serializes_protected_vars(tmp_path):
    data = pd.DataFrame({"target": [0, 1], "secret": ["a", "b"], "group": [0, 1]})
    config = AnalysisConfig(data, "target", sensitive_vars=["secret"], protected_vars=["group"])
    config.save(str(tmp_path / "config"))

    saved = (tmp_path / "config.json").read_text()
    assert '"protected_vars": ["group"]' in saved
    loaded = _analysis_target_parser(data, str(tmp_path / "config.json"))
    assert loaded.sensitive_vars == ["secret"]
    assert loaded.protected_vars == ["group"]


def test_statistical_parity_uses_protected_role_not_disclosure_role():
    data = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "secret": [0, 1, 2, 0, 1, 2],
            "group": [0, 1, 0, 1, 0, 1],
            "feature": [1, 2, 3, 4, 5, 6],
        }
    )
    config = AnalysisConfig(
        data, "target", sensitive_vars=["secret"], protected_vars=["group"]
    )

    result = StatisticalParity(
        data, data, analysis_target=config, do_preprocessing=False
    ).evaluate(folds=2)

    assert result["raw results"]["protected_attribute"].tolist() == ["group"]


def test_equal_opportunity_uses_protected_role_not_disclosure_role():
    data = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "secret": [0, 1, 2, 0, 1, 2],
            "group": [0, 1, 0, 1, 0, 1],
            "feature": [1, 2, 3, 4, 5, 6],
        }
    )
    config = AnalysisConfig(
        data, "target", sensitive_vars=["secret"], protected_vars=["group"]
    )

    result = EqualOpportunity(
        data, data, analysis_target=config, do_preprocessing=False
    ).evaluate(folds=2)

    assert result["raw results"]["protected_attribute"].tolist() == ["group"]


def test_equalized_odds_uses_protected_role_not_disclosure_role():
    data = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "secret": [0, 1, 2, 0, 1, 2],
            "group": [0, 1, 0, 1, 0, 1],
            "feature": [1, 2, 3, 4, 5, 6],
        }
    )
    config = AnalysisConfig(
        data, "target", sensitive_vars=["secret"], protected_vars=["group"]
    )

    result = EqualizedOdds(
        data, data, analysis_target=config, do_preprocessing=False
    ).evaluate(folds=2)

    assert result["raw results"]["protected_attribute"].tolist() == ["group"]


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
