import importlib
import math

import pandas as pd
import pytest
import syntheval.syntheval as syntheval_module
from syntheval.execution import build_metric_execution
from syntheval.metrics.fairness.metric_equal_opportunity import EqualOpportunity
from syntheval.metrics.fairness.metric_equalized_odds import EqualizedOdds
from syntheval.metrics.fairness.metric_statistical_parity import StatisticalParity
from syntheval.metrics.privacy.metric_AttrDis import AttributeDisclosure
from syntheval.syntheval import SynthEval
from syntheval.utils.configuration import AnalysisConfig, _analysis_target_parser


def _multiclass_fairness_data():
    rows = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
        (2, 0, 2),
        (2, 1, 0),
    ] * 2
    return pd.DataFrame(rows, columns=["target", "group", "prediction"])


class _PredictionFeatureClassifier:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["prediction"].to_numpy()


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


@pytest.mark.parametrize(
    ("metric_class", "result_key", "per_class_values", "macro_value"),
    [
        (
            StatisticalParity,
            "statistical_parity",
            [-1 / 3, 2 / 3, -1 / 3],
            0.0,
        ),
        (
            EqualizedOdds,
            "equalized_odds",
            [0.5, 0.75, 0.5],
            7 / 12,
        ),
        (
            EqualOpportunity,
            "equal_opportunity",
            [-1.0, 1.0, -1.0],
            -1 / 3,
        ),
    ],
)
def test_multiclass_fairness_metrics_report_ovr_classes_and_versioned_macro(
    monkeypatch, metric_class, result_key, per_class_values, macro_value
):
    module = importlib.import_module(metric_class.__module__)
    monkeypatch.setattr(module, "RandomForestClassifier", _PredictionFeatureClassifier)
    data = _multiclass_fairness_data()
    config = AnalysisConfig(data, "target", protected_vars=["group"])
    metric = metric_class(
        data, data.copy(), analysis_target=config, do_preprocessing=False
    )

    result = metric.evaluate(folds=2, full_output=True)
    raw_results = result["raw results"].sort_values("target_class")
    macro_key = f"{result_key}_macro_ovr_v1"

    assert raw_results["target_class"].tolist() == [0, 1, 2]
    assert raw_results[result_key].tolist() == pytest.approx(per_class_values)
    assert result[macro_key] == pytest.approx(macro_value)
    normalized = metric.normalize_output()
    assert normalized[0]["metric"] == macro_key
    assert len(normalized) == 4
    assert all("ovr_v1" in row["metric"] for row in normalized[1:])

    if metric_class is EqualizedOdds:
        assert raw_results["tpr_difference"].tolist() == pytest.approx(
            [-1.0, 1.0, -1.0]
        )
        assert raw_results["fpr_difference"].tolist() == pytest.approx(
            [0.0, 0.5, 0.0]
        )


@pytest.mark.parametrize(
    "metric_class", [StatisticalParity, EqualizedOdds, EqualOpportunity]
)
@pytest.mark.parametrize("failure", ["missing_class", "missing_group"])
def test_multiclass_fairness_fails_on_missing_class_or_group_support(
    monkeypatch, metric_class, failure
):
    module = importlib.import_module(metric_class.__module__)
    monkeypatch.setattr(module, "RandomForestClassifier", _PredictionFeatureClassifier)
    real = _multiclass_fairness_data()
    synthetic = real.copy()
    if failure == "missing_class":
        synthetic = synthetic.loc[synthetic["target"] != 2].reset_index(drop=True)
    else:
        synthetic["group"] = 0
    config = AnalysisConfig(real, "target", protected_vars=["group"])
    metric = metric_class(
        real, synthetic, analysis_target=config, do_preprocessing=False
    )

    message = "missing classes" if failure == "missing_class" else "support"
    with pytest.raises(ValueError, match=message):
        metric.evaluate(folds=2)


@pytest.mark.parametrize(
    ("metric_class", "result_key"),
    [
        (StatisticalParity, "statistical_parity"),
        (EqualizedOdds, "equalized_odds"),
        (EqualOpportunity, "equal_opportunity"),
    ],
)
def test_binary_fairness_metric_keeps_legacy_identity(
    monkeypatch, metric_class, result_key
):
    module = importlib.import_module(metric_class.__module__)
    monkeypatch.setattr(module, "RandomForestClassifier", _PredictionFeatureClassifier)
    data = _multiclass_fairness_data()
    data = data.loc[data["target"] != 2].reset_index(drop=True)
    config = AnalysisConfig(data, "target", protected_vars=["group"])
    metric = metric_class(
        data, data.copy(), analysis_target=config, do_preprocessing=False
    )

    result = metric.evaluate(folds=2, full_output=True)

    assert result_key in result
    assert f"{result_key}_macro_ovr_v1" not in result
    assert metric.normalize_output()[0]["metric"] == result_key


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


def test_structured_evaluation_emits_safe_method_progress_for_success_failure_and_block(
    monkeypatch,
):
    train = pd.DataFrame({"category": ["known", "known"]})

    class CaptureMetric:
        def __init__(self, **_kwargs):
            pass

        def evaluate(self, **_kwargs):
            return {"ok": True}

        def format_output(self):
            return []

        def normalize_output(self):
            return [{"metric": "capture", "dim": "u", "val": 0.5, "n_val": 0.5}]

        def normalize_output_v2(self):
            return [{"metric": "capture", "dim": "u", "val": 0.5, "n_val": 0.5}]

    class FailingMetric(CaptureMetric):
        def evaluate(self, **_kwargs):
            raise ValueError("secret category value")

    class MustNotRun(CaptureMetric):
        def __init__(self, **_kwargs):
            raise AssertionError("blocked metric must not be constructed")

    monkeypatch.setitem(syntheval_module.loaded_metrics, "capture", CaptureMetric)
    monkeypatch.setitem(syntheval_module.loaded_metrics, "unexpected", FailingMetric)
    monkeypatch.setitem(syntheval_module.loaded_metrics, "cls_acc", MustNotRun)
    evaluator = SynthEval(
        train,
        holdout_dataframe=pd.DataFrame({"category": ["holdout-only"]}),
        cat_cols=["category"],
        verbose=False,
        enable_plots=False,
        console="off",
    )
    events = []

    evaluator.evaluate(
        train.copy(),
        return_execution=True,
        expected_output_manifest={
            "capture": ("capture",),
            "unexpected": ("unexpected",),
            "cls_acc": ("cls_acc",),
        },
        progress_callback=events.append,
        _dataset_name="model-a",
        capture={},
        unexpected={},
        cls_acc={},
    )

    assert [(item["method"], item["event"]) for item in events] == [
        ("capture", "started"),
        ("capture", "completed"),
        ("unexpected", "started"),
        ("unexpected", "failure"),
        ("cls_acc", "started"),
        ("cls_acc", "failure"),
    ]
    assert [item["outcome"] for item in events if item["event"] != "started"] == [
        "succeeded",
        "failed",
        "blocked",
    ]
    assert events[1]["duration_seconds"] >= 0
    assert events[3]["failure_class"] == "ValueError"
    assert events[5]["failure_class"] == "real_holdout_unknown_category"
    assert "secret category value" not in str(events)
    assert all(item["model_name"] == "model-a" for item in events)


def test_structured_evaluation_callback_failure_is_telemetry_only(monkeypatch, caplog):
    real = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]})

    class CaptureMetric:
        def __init__(self, **_kwargs):
            pass

        def evaluate(self, **_kwargs):
            return {"ok": True}

        def format_output(self):
            return []

        def normalize_output(self):
            return [{"metric": "capture", "dim": "u", "val": 0.5, "n_val": 0.5}]

        def normalize_output_v2(self):
            return [{"metric": "capture", "dim": "u", "val": 0.5, "n_val": 0.5}]

    monkeypatch.setitem(syntheval_module.loaded_metrics, "capture", CaptureMetric)
    evaluator = SynthEval(
        real,
        cat_cols=["target"],
        verbose=False,
        enable_plots=False,
        console="off",
    )

    def broken_callback(_event):
        raise ValueError("sensitive callback detail")

    with caplog.at_level("WARNING", logger=syntheval_module.__name__):
        execution = evaluator.evaluate(
            real.copy(),
            return_execution=True,
            expected_output_manifest={"capture": ("capture",)},
            progress_callback=broken_callback,
            capture={},
        )

    assert execution.execution_complete is True
    assert execution.metric_executions[0].status.state == "succeeded"
    assert caplog.text.count("progress callback failed") == 2
    assert "exception_type=ValueError" in caplog.text
    assert "sensitive callback detail" not in caplog.text


def test_structured_evaluation_uses_nominal_unknown_only_for_supported_holdout_metrics(
    monkeypatch,
):
    train = pd.DataFrame({"category": ["train-a", "train-b"]})
    synthetic = pd.DataFrame({"category": ["train-a", "train-b"]})
    holdout = pd.DataFrame({"category": ["holdout-only"]})

    class CaptureMetric:
        holdout_values = None

        def __init__(self, **kwargs):
            type(self).holdout_values = kwargs["hout_data"]["category"].tolist()

        def evaluate(self, **kwargs):
            return {"ok": True}

        def format_output(self):
            return []

        def normalize_output(self):
            return [
                {
                    "metric": "supported",
                    "dim": "p",
                    "val": 0.0,
                    "err": 0.0,
                    "n_val": 1.0,
                    "n_err": 0.0,
                }
            ]

        def normalize_output_v2(self):
            return [
                {
                    "metric": "supported",
                    "dim": "p",
                    "val": 0.0,
                    "err": 0.0,
                    "n_val": 1.0,
                    "n_err": 0.0,
                }
            ]

    class MustNotRun:
        def __init__(self, **kwargs):
            raise AssertionError("unsupported metric must be blocked before construction")

    monkeypatch.setitem(syntheval_module.loaded_metrics, "nndr", CaptureMetric)
    monkeypatch.setitem(syntheval_module.loaded_metrics, "cls_acc", MustNotRun)
    evaluator = SynthEval(
        train,
        holdout_dataframe=holdout,
        cat_cols=["category"],
        verbose=False,
        enable_plots=False,
        console="off",
    )

    execution = evaluator.evaluate(
        synthetic,
        return_execution=True,
        expected_output_manifest={"nndr": ("supported",), "cls_acc": ("unsupported",)},
        nndr={},
        cls_acc={},
    )

    assert CaptureMetric.holdout_values == [-1]
    assert execution.preprocessing_metadata["fit_role"] == "train"
    assert execution.preprocessing_metadata["categories"] == [["'train-a'", "'train-b'"]]
    assert execution.preprocessing_metadata["real_holdout_unknown_categories"] == {"category": 1}
    assert execution.preprocessing_metadata["real_holdout_unknown_representation"] == -1
    assert execution.preprocessing_metadata["real_holdout_unknown_nominal_metrics"] == [
        "nndr"
    ]
    assert execution.preprocessing_metadata["real_holdout_unknown_blocked_metrics"] == [
        "cls_acc"
    ]
    assert execution.preprocessing_metadata["real_holdout_unknown_policy"] == "blocked"
    assert execution.preprocessing_metadata["real_holdout_unknown_remapped_metrics"] == []
    assert execution.preprocessing_metadata["real_holdout_unknown_row_fraction"] == 1.0
    assert execution.metric_executions[0].status.state == "succeeded"
    blocked = execution.metric_executions[1]
    assert blocked.status.state == "blocked"
    assert blocked.status.exception_type == "RealHoldoutUnknownCategoryError"
    assert "real holdout" in blocked.status.exception_message.casefold()


def test_structured_evaluation_still_rejects_unknown_synthetic_category():
    train = pd.DataFrame({"category": ["train-a", "train-b"]})
    synthetic = pd.DataFrame({"category": ["synthetic-only"]})
    evaluator = SynthEval(
        train,
        holdout_dataframe=pd.DataFrame({"category": ["train-a"]}),
        cat_cols=["category"],
        verbose=False,
        enable_plots=False,
        console="off",
    )

    with pytest.raises(ValueError, match="Unknown categorical value.*role='evaluation'"):
        evaluator.evaluate(
            synthetic,
            return_execution=True,
            expected_output_manifest={"nndr": ("supported",)},
            nndr={},
        )


def test_legacy_evaluation_blocks_unsupported_real_holdout_unknown_category():
    train = pd.DataFrame({"category": ["train-a", "train-b"]})
    evaluator = SynthEval(
        train,
        holdout_dataframe=pd.DataFrame({"category": ["holdout-only"]}),
        cat_cols=["category"],
        verbose=False,
        enable_plots=False,
        console="off",
    )

    with pytest.raises(
        syntheval_module.RealHoldoutUnknownCategoryError, match="unsupported metrics"
    ):
        evaluator.evaluate(train.copy(), cls_acc={})


def _holdout_capture_metric(store, key):
    class CaptureMetric:
        def __init__(self, **kwargs):
            store[key] = kwargs["hout_data"]["category"].tolist()

        def evaluate(self, **kwargs):
            return {"ok": True}

        def format_output(self):
            return []

        def normalize_output(self):
            return [{"metric": key, "dim": "u", "val": 0.5, "n_val": 0.5}]

        def normalize_output_v2(self):
            return [{"metric": key, "dim": "u", "val": 0.5, "n_val": 0.5}]

    return CaptureMetric


def _rare_unknown_frames():
    # train-b is the mode; one of 40 holdout rows (2.5%) carries an unseen value.
    train = pd.DataFrame({"category": ["train-a"] * 3 + ["train-b"] * 5})
    holdout = pd.DataFrame({"category": ["train-a"] * 20 + ["train-b"] * 19 + ["holdout-only"]})
    return train, holdout


def test_structured_evaluation_remaps_rare_holdout_unknown_to_train_mode(monkeypatch):
    train, holdout = _rare_unknown_frames()
    seen = {}
    monkeypatch.setitem(
        syntheval_module.loaded_metrics, "nndr", _holdout_capture_metric(seen, "nndr")
    )
    monkeypatch.setitem(
        syntheval_module.loaded_metrics, "cls_acc", _holdout_capture_metric(seen, "cls_acc")
    )
    evaluator = SynthEval(
        train,
        holdout_dataframe=holdout,
        cat_cols=["category"],
        verbose=False,
        enable_plots=False,
        console="off",
    )

    execution = evaluator.evaluate(
        train.copy(),
        return_execution=True,
        expected_output_manifest={"nndr": ("nndr",), "cls_acc": ("cls_acc",)},
        nndr={},
        cls_acc={},
    )

    assert [item.status.state for item in execution.metric_executions] == [
        "succeeded",
        "succeeded",
    ]
    # Gower metrics keep the nominal unknown code; classifier metrics get the train mode.
    assert seen["nndr"][-1] == -1
    assert seen["cls_acc"][-1] == 1
    assert -1 not in seen["cls_acc"]
    metadata = execution.preprocessing_metadata
    assert metadata["real_holdout_unknown_categories"] == {"category": 1}
    assert metadata["real_holdout_unknown_row_count"] == 1
    assert metadata["real_holdout_unknown_row_fraction"] == pytest.approx(1 / 40)
    assert metadata["real_holdout_unknown_policy"] == "train_mode"
    assert metadata["real_holdout_unknown_nominal_metrics"] == ["nndr"]
    assert metadata["real_holdout_unknown_remapped_metrics"] == ["cls_acc"]
    assert metadata["real_holdout_unknown_blocked_metrics"] == []
    assert metadata["unknown_fallback_codes"] == [1]
    assert "holdout-only" not in str(metadata)


def test_structured_evaluation_blocks_holdout_unknown_above_row_fraction_cap(monkeypatch):
    train, holdout = _rare_unknown_frames()

    class MustNotRun:
        def __init__(self, **kwargs):
            raise AssertionError("blocked metric must not be constructed")

    monkeypatch.setitem(syntheval_module.loaded_metrics, "cls_acc", MustNotRun)
    evaluator = SynthEval(
        train,
        holdout_dataframe=holdout,
        cat_cols=["category"],
        verbose=False,
        enable_plots=False,
        console="off",
        max_holdout_unknown_row_fraction=0.0,
    )

    execution = evaluator.evaluate(
        train.copy(),
        return_execution=True,
        expected_output_manifest={"cls_acc": ("cls_acc",)},
        cls_acc={},
    )

    blocked = execution.metric_executions[0]
    assert blocked.status.state == "blocked"
    assert blocked.status.exception_type == "RealHoldoutUnknownCategoryError"
    assert execution.preprocessing_metadata["real_holdout_unknown_policy"] == "blocked"
    assert execution.preprocessing_metadata["real_holdout_unknown_blocked_metrics"] == [
        "cls_acc"
    ]


@pytest.mark.parametrize("value", [-0.1, 1.5, True, "0.05"])
def test_invalid_holdout_unknown_row_fraction_is_rejected(value):
    with pytest.raises(ValueError, match="max_holdout_unknown_row_fraction"):
        SynthEval(
            pd.DataFrame({"category": ["a", "b"]}),
            cat_cols=["category"],
            verbose=False,
            enable_plots=False,
            console="off",
            max_holdout_unknown_row_fraction=value,
        )


def test_legacy_evaluation_remaps_rare_holdout_unknown(monkeypatch):
    train, holdout = _rare_unknown_frames()
    seen = {}
    monkeypatch.setitem(
        syntheval_module.loaded_metrics, "cls_acc", _holdout_capture_metric(seen, "cls_acc")
    )
    evaluator = SynthEval(
        train,
        holdout_dataframe=holdout,
        cat_cols=["category"],
        verbose=False,
        enable_plots=False,
        console="off",
    )

    evaluator.evaluate(train.copy(), cls_acc={})

    assert seen["cls_acc"][-1] == 1


def test_train_mode_policy_uses_train_state_only():
    from syntheval.utils.preprocessing import TrainFittedPreprocessor

    train = pd.DataFrame(
        {"category": ["a", "b", "b", "c"], "constant": ["neg"] * 4, "x": [0.0, 1.0, 2.0, 3.0]}
    )
    holdout = pd.DataFrame(
        {"category": ["new", "a"], "constant": ["pos", "neg"], "x": [1.0, 2.0]}
    )
    preprocessor = TrainFittedPreprocessor.fit(train, ["category", "constant"], ["x"])
    refit = TrainFittedPreprocessor.fit(train, ["category", "constant"], ["x"])

    encoded = preprocessor.encode(holdout, role="real_holdout", unknown_policy="train_mode")

    assert encoded["category"].tolist() == [1, 0]
    # A column constant in train resolves to its only level.
    assert encoded["constant"].tolist() == [0, 0]
    assert preprocessor.unknown_row_count(holdout) == 1
    assert preprocessor.fingerprint == refit.fingerprint
    with pytest.raises(ValueError, match="Unknown categorical value"):
        preprocessor.encode(holdout, role="real_holdout")
    with pytest.raises(ValueError, match="unknown_policy"):
        preprocessor.encode(holdout, unknown_policy="drop")


def test_column_name_analysis_target_does_not_write_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0], "label": [0, 1, 0, 1]})

    config = _analysis_target_parser(real, "label")

    assert config.target_vars == ["label"]
    assert list(tmp_path.iterdir()) == []
