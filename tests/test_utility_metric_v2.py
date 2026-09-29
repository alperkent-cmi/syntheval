import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from syntheval.execution import build_metric_execution
from syntheval.metrics.utility import (
    metric_kolmogorov_smirnov,
    metric_mixed_correlation,
    metric_mutual_information,
)
from syntheval.metrics.utility.metric_accuracy_difference import ClassificationAccuracy
from syntheval.metrics.utility.metric_auroc_difference import (
    PredictionAUROCDifference,
    auroc_agreement_v2,
)
from syntheval.metrics.utility.metric_hellinger_distance import (
    HellingerDistance,
    _hellinger,
)
from syntheval.metrics.utility.metric_kolmogorov_smirnov import (
    KolmogorovSmirnovTest,
    _discrete_ks,
    _total_variation_distance,
)
from syntheval.metrics.utility.metric_max_mean_discrepancy import (
    MaximumMeanDiscrepancy,
    mixed_rbf_mmd_v2,
)
from syntheval.metrics.utility.metric_mixed_correlation import (
    MixedCorrelation,
    _correlation_ratio_v2,
    _cramers_V_v2,
    mixed_correlation_v2,
)
from syntheval.metrics.utility.metric_mutual_information import (
    MutualInformation,
    mutual_information_v2,
)
from syntheval.metrics.utility.metric_propensity_mse import (
    PropensityMeanSquaredError,
    normalize_pmse_v2,
)
from syntheval.metrics.utility.metric_quantile_mse import QuantileMSE


def test_auroc_v2_is_signed_and_zero_difference_is_perfect_agreement():
    assert auroc_agreement_v2(0.0) == 1.0
    assert auroc_agreement_v2(0.25) == 0.75
    assert auroc_agreement_v2(-0.25) == 0.75

    real = pd.DataFrame({'x': range(8), 'label': [0, 1] * 4})
    metric = PredictionAUROCDifference(
        real,
        real.copy(),
        real.copy(),
        cat_cols=['label'],
        num_cols=['x'],
        analysis_target='label',
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )
    result = metric.evaluate(model='log_reg')
    assert np.isclose(result['auroc_diff_v2'], 0.0)
    assert result['auroc_agreement_v2'] == 1.0
    assert result['auroc_version_v2'] == 'signed_synthetic_minus_real'
    assert metric.normalize_output_v2()[0]['n_val'] == 1.0
    assert metric.normalize_output_v2()[0]['metric'] == 'auroc_v2'


def test_auroc_multiclass_uses_macro_ovr_and_retains_classwise_evidence(monkeypatch):
    real_probabilities = np.array([
        [0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1],
        [0.3, 0.6, 0.1], [0.1, 0.2, 0.7], [0.2, 0.1, 0.7],
    ])
    synthetic_probabilities = np.array([
        [0.6, 0.2, 0.2], [0.4, 0.5, 0.1], [0.3, 0.6, 0.1],
        [0.2, 0.5, 0.3], [0.3, 0.2, 0.5], [0.1, 0.4, 0.5],
    ])

    class FixedProbabilityClassifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, features, labels):
            self.classes_ = np.unique(labels)
            self.probabilities = (
                real_probabilities if len(features) == 9 else synthetic_probabilities
            )
            return self

        def predict_proba(self, features):
            return self.probabilities[features['row'].to_numpy(dtype=int)]

    monkeypatch.setattr(
        'syntheval.metrics.utility.metric_auroc_difference.LogisticRegression',
        FixedProbabilityClassifier,
    )
    real = pd.DataFrame({'row': range(9), 'label': ['a', 'b', 'c'] * 3})
    synt = pd.DataFrame({'row': range(12), 'label': ['a', 'b', 'c'] * 4})
    hout = pd.DataFrame({'row': range(6), 'label': ['a', 'a', 'b', 'b', 'c', 'c']})
    metric = PredictionAUROCDifference(
        real,
        synt,
        hout,
        cat_cols=['label'],
        num_cols=['row'],
        analysis_target='label',
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )

    result = metric.evaluate(model='log_reg')
    classes = ['a', 'b', 'c']
    expected_class_differences = [
        roc_auc_score(
            (hout['label'] == label).astype(int), synthetic_probabilities[:, index]
        )
        - roc_auc_score(
            (hout['label'] == label).astype(int), real_probabilities[:, index]
        )
        for index, label in enumerate(classes)
    ]
    evidence = result['auroc_class_results_v3']['label']
    assert evidence['aggregation'] == 'macro_one_vs_rest'
    assert evidence['metric_version'] == 'macro_ovr_v3'
    assert [record['class_label'] for record in evidence['classes']] == classes
    np.testing.assert_allclose(
        [record['difference'] for record in evidence['classes']],
        expected_class_differences,
    )
    expected_macro = float(np.mean(expected_class_differences))
    assert np.isclose(result['auroc_diff_macro_ovr_v3'], expected_macro)
    assert result['auroc_version_v3'] == 'macro_one_vs_rest_synthetic_minus_real'

    normalized = metric.normalize_output_v2()
    assert normalized[0]['metric'] == 'auroc_macro_ovr_v3'
    assert normalized[0]['metric_version'] == 'macro_ovr_v3'
    class_rows = [row for row in normalized if '_class_' in row['metric']]
    assert len(class_rows) == 3
    execution = build_metric_execution(
        'auroc_diff',
        (),
        ('auroc_macro_ovr_v3',),
        status_key_result=normalized,
        normalized_rows_v2=normalized,
    )
    assert execution.status.succeeded


def test_auroc_multiclass_rejects_missing_class_support():
    real = pd.DataFrame({'x': range(9), 'label': ['a', 'b', 'c'] * 3})
    synt = pd.DataFrame({'x': range(9), 'label': ['a', 'b', 'a'] * 3})
    hout = pd.DataFrame({'x': range(6), 'label': ['a', 'a', 'b', 'b', 'c', 'c']})
    metric = PredictionAUROCDifference(
        real,
        synt,
        hout,
        cat_cols=['label'],
        num_cols=['x'],
        analysis_target='label',
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )
    with pytest.raises(ValueError, match='incompatible synthetic class support'):
        metric.evaluate(model='log_reg')


def test_classification_v2_uses_macro_default_and_paired_folds_without_equalizing_sizes():
    real = pd.DataFrame({'x': range(12), 'label': [0, 1] * 6})
    synt = pd.DataFrame({'x': range(18), 'label': [0, 1] * 9})
    metric = ClassificationAccuracy(
        real,
        synt,
        cat_cols=['label'],
        num_cols=['x'],
        analysis_target='label',
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )
    result = metric.evaluate(cls_models=['dt'], k_folds=3)
    row = result['train results'].iloc[0]
    assert result['score_type_v2'] == 'macro'
    assert row['score_type_v2'] == 'macro'
    assert np.isfinite(row['score_diff_err_v2'])
    assert metric.normalize_output_v2()[0]['metric'] == 'avg_macro_F1_diff_v2'

    balanced = ClassificationAccuracy(
        real,
        synt,
        cat_cols=['label'],
        num_cols=['x'],
        analysis_target='label',
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )
    balanced_result = balanced.evaluate(
        cls_models=['dt'], F1_type='balanced_accuracy', k_folds=3
    )
    assert balanced_result['score_type_v2'] == 'balanced_accuracy'
    assert balanced.normalize_output_v2()[0]['metric'] == 'avg_balanced_accuracy_diff_v2'


def test_classification_v2_rejects_missing_synthetic_class_support():
    real = pd.DataFrame({'x': range(6), 'label': [0, 1] * 3})
    synt = pd.DataFrame({'x': range(6), 'label': [0] * 6})
    metric = ClassificationAccuracy(
        real,
        synt,
        cat_cols=['label'],
        num_cols=['x'],
        analysis_target='label',
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )
    with pytest.raises(ValueError, match='synthetic classes'):
        metric.evaluate(cls_models=['dt'], k_folds=2)


def test_mixed_correlation_v2_uses_spearman_cramers_v_eta_and_invalid_counts():
    frame = pd.DataFrame({
        'n1': [1.0, 2.0, 4.0, 8.0],
        'n2': [10.0, 20.0, 30.0, 40.0],
        'constant': [1.0] * 4,
        'cat': ['a', 'a', 'b', 'b'],
    })
    assert np.isclose(_cramers_V_v2(['a', 'a', 'b', 'b'], ['x', 'x', 'y', 'y']), 1.0)
    assert np.isclose(_correlation_ratio_v2(['a', 'a', 'b', 'b'], [1, 1, 3, 3]), 1.0)
    matrix, valid = mixed_correlation_v2(frame, ['n1', 'n2', 'constant'], ['cat'])
    assert np.isclose(matrix.loc['n1', 'n2'], 1.0)
    assert not valid.loc['n1', 'constant']

    metric = MixedCorrelation(
        frame,
        frame.copy(),
        cat_cols=['cat'],
        num_cols=['n1', 'n2', 'constant'],
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )
    result = metric.evaluate()
    assert result['corr_valid_pairs_v2'] + result['corr_invalid_pairs_v2'] == result['corr_total_pairs_v2']
    assert result['corr_mat_diff_v2'] == 0.0
    assert metric.normalize_output_v2()[0]['n_val'] == 1.0


def _mixed_correlation_frame():
    size = 18
    frame = pd.DataFrame({
        f'cat{column}': [
            None if row % 11 == 0 else f'level_{(row + column) % 3}'
            for row in range(size)
        ]
        for column in range(5)
    })
    frame['num0'] = [np.nan if row == 4 else float(row) for row in range(size)]
    frame['num1'] = [float((row * 3) % 7) for row in range(size)]
    frame['constant'] = [1.0] * size
    frame['num_inf'] = [
        np.inf if row == 6 else float(size - row) for row in range(size)
    ]
    return frame


def _assert_results_equal(left, right):
    assert left.keys() == right.keys()
    for key, left_value in left.items():
        _assert_nested_equal(left_value, right[key])


def _assert_nested_equal(left, right):
    if isinstance(left, pd.DataFrame):
        assert left.equals(right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key, value in left.items():
            _assert_nested_equal(value, right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    elif np.isscalar(left) and pd.isna(left):
        assert pd.isna(right)
    else:
        assert left == right


def test_mixed_correlation_v2_parallel_matches_serial_for_matrices_and_metric(
    monkeypatch,
):
    real = _mixed_correlation_frame()
    synt = real.copy()
    synt['cat0'] = synt['cat0'].shift(1)
    synt['num1'] = synt['num1'].iloc[::-1].to_numpy()
    num_cols = ['num0', 'num1', 'constant', 'num_inf']
    cat_cols = [f'cat{column}' for column in range(5)]

    monkeypatch.setattr(metric_mixed_correlation, 'cpu_count', lambda: 1)
    serial_real, serial_real_valid = mixed_correlation_v2(real, num_cols, cat_cols)
    serial_synt, serial_synt_valid = mixed_correlation_v2(synt, num_cols, cat_cols)
    serial_metric = MixedCorrelation(
        real, synt, cat_cols=cat_cols, num_cols=num_cols, do_preprocessing=False,
        verbose=False, plot_figures=False,
    )
    serial_result = serial_metric.evaluate(return_mats=True)

    monkeypatch.setattr(metric_mixed_correlation, 'cpu_count', lambda: 2)
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '2')
    parallel_calls = []
    worker_pids = set()
    joblib_parallel = metric_mixed_correlation.Parallel

    def recording_parallel(*args, **kwargs):
        parallel_calls.append(kwargs)
        parallel = joblib_parallel(*args, **kwargs)

        class RecordingParallel:
            def __call__(self, tasks):
                results = parallel(tasks)

                def record_worker_pids():
                    for worker_pid, pair_values in results:
                        worker_pids.add(worker_pid)
                        yield worker_pid, pair_values

                return record_worker_pids()

        return RecordingParallel()

    monkeypatch.setattr(metric_mixed_correlation, 'Parallel', recording_parallel)
    parallel_real, parallel_real_valid = mixed_correlation_v2(real, num_cols, cat_cols)
    parallel_synt, parallel_synt_valid = mixed_correlation_v2(synt, num_cols, cat_cols)
    parallel_metric = MixedCorrelation(
        real, synt, cat_cols=cat_cols, num_cols=num_cols, do_preprocessing=False,
        verbose=False, plot_figures=False,
    )
    parallel_result = parallel_metric.evaluate(return_mats=True)

    assert len(parallel_calls) == 4
    assert all(call == {
        'n_jobs': 2,
        'backend': 'loky',
        'return_as': 'generator',
        'batch_size': 1,
        'pre_dispatch': 2,
    } for call in parallel_calls)
    assert worker_pids
    assert all(pid != os.getpid() for pid in worker_pids)
    assert parallel_real.equals(serial_real)
    assert parallel_synt.equals(serial_synt)
    assert parallel_real_valid.equals(serial_real_valid)
    assert parallel_synt_valid.equals(serial_synt_valid)
    _assert_results_equal(serial_result, parallel_result)
    assert parallel_real.equals(parallel_real.T)
    assert parallel_synt.equals(parallel_synt.T)
    assert np.diag(parallel_real).tolist() == [1.0] * len(parallel_real)
    assert np.diag(parallel_synt).tolist() == [1.0] * len(parallel_synt)


def test_mixed_correlation_v2_worker_count_is_bounded_and_serial_for_small_inputs(
    monkeypatch,
):
    monkeypatch.delenv('LOKY_MAX_CPU_COUNT', raising=False)
    monkeypatch.setattr(metric_mixed_correlation, 'cpu_count', lambda: 8)
    assert metric_mixed_correlation._v2_worker_count(31) == 1
    assert metric_mixed_correlation._v2_worker_count(32) == 8
    assert metric_mixed_correlation._v2_worker_count(3) == 1

    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '2')
    assert metric_mixed_correlation._v2_worker_count(1000) == 2

    def unexpected_parallel(*args, **kwargs):
        pytest.fail('small v2 input should use serial computation')

    monkeypatch.setattr(metric_mixed_correlation, 'Parallel', unexpected_parallel)
    small_frame = pd.DataFrame({'left': [1.0, 2.0, 3.0], 'right': [3.0, 2.0, 1.0]})
    small_matrix, small_valid = mixed_correlation_v2(small_frame, ['left', 'right'], [])
    assert small_matrix.loc['left', 'right'] == -1.0
    assert small_valid.loc['left', 'right']

    monkeypatch.setattr(metric_mixed_correlation, 'cpu_count', lambda: 2)
    assert metric_mixed_correlation._v2_worker_count(1000) == 2

    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '1')
    assert metric_mixed_correlation._v2_worker_count(1000) == 1

    monkeypatch.setattr(metric_mixed_correlation, 'cpu_count', lambda: 1)
    assert metric_mixed_correlation._v2_worker_count(1000) == 1


def test_mixed_correlation_v2_pair_chunks_are_bounded_and_ordered():
    chunks = metric_mixed_correlation._v2_pair_chunks(8, 5)
    first = next(chunks)
    assert first == [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    remaining_chunks = list(chunks)
    assert all(len(chunk) <= 5 for chunk in remaining_chunks)
    all_pairs = first + [pair for chunk in remaining_chunks for pair in chunk]
    assert all_pairs == [
        (left, right)
        for left in range(8)
        for right in range(left + 1, 8)
    ]
    batches = list(metric_mixed_correlation._v2_batches(iter(remaining_chunks), 2))
    assert all(len(batch) <= 2 for batch in batches)


def _fail_mixed_correlation_chunk(data, labels, pairs, numerical_set, categorical_set):
    left_index, right_index = pairs[0]
    left_label = labels[left_index]
    right_label = labels[right_index]
    try:
        raise ValueError('invalid pair data')
    except ValueError as exc:
        raise RuntimeError(
            f'Failed to compute v2 mixed-correlation pair '
            f'({left_label!r}, {right_label!r})'
        ) from exc


def test_mixed_correlation_v2_parallel_worker_failure_has_pair_context(monkeypatch):
    frame = _mixed_correlation_frame()
    monkeypatch.setattr(metric_mixed_correlation, 'cpu_count', lambda: 2)
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '2')
    monkeypatch.setattr(
        metric_mixed_correlation,
        '_mixed_correlation_chunk_v2',
        _fail_mixed_correlation_chunk,
    )
    with pytest.raises(
        RuntimeError, match='Failed to compute v2 mixed-correlation pair'
    ) as exc_info:
        mixed_correlation_v2(
            frame,
            ['num0', 'num1', 'constant', 'num_inf'],
            [f'cat{column}' for column in range(5)],
        )
    assert 'ValueError: invalid pair data' in str(exc_info.value.__cause__)


def test_mutual_information_v2_fits_numeric_bins_on_real_only():
    real = pd.DataFrame({'num': [0.0, 1.0, 2.0, 3.0], 'cat': ['a', 'a', 'b', 'b']})
    synt = pd.DataFrame({'num': [0.0, 1.0, 2.0, 100.0], 'cat': ['a', 'b', 'b', 'unknown']})
    real_mi, synt_mi, supports, support_hash = mutual_information_v2(
        real, synt, num_cols=['num'], cat_cols=['cat'], num_quants=2
    )
    assert supports['num']['bins'] == [0.0, 1.5, 3.0]
    assert supports['num']['overflow'] == 3
    assert supports['cat']['unknown'] == 2
    assert np.isfinite(real_mi.to_numpy()).all()
    assert np.isfinite(synt_mi.to_numpy()).all()
    assert len(support_hash) == 64


def _mi_v2_frames():
    rng = np.random.default_rng(842)
    real = pd.DataFrame({
        f'num{column}': rng.normal(loc=column, size=32)
        for column in range(5)
    })
    real.loc[3, 'num0'] = np.nan
    real.loc[5, 'num1'] = np.inf
    real['constant'] = 1.0
    real['missing'] = np.nan
    for column in range(3):
        real[f'cat{column}'] = [
            None if row % 13 == 0 else f'level_{(row + column) % 3}'
            for row in range(len(real))
        ]
    synt = real.copy()
    synt.loc[0, 'num2'] = 1000.0
    synt['cat0'] = synt['cat0'].shift(1)
    synt.loc[1, 'cat1'] = 'unknown'
    return real, synt


def _assert_frames_equal(left, right):
    assert left.index.equals(right.index)
    assert left.columns.equals(right.columns)
    np.testing.assert_equal(left.to_numpy(), right.to_numpy())


def test_mi_v2_worker_count_obeys_cpu_budget_and_small_pair_fallback(monkeypatch):
    monkeypatch.delenv('LOKY_MAX_CPU_COUNT', raising=False)
    monkeypatch.setattr(metric_mutual_information, 'cpu_count', lambda: 8)
    assert metric_mutual_information._mi_v2_worker_count(31) == 1
    assert metric_mutual_information._mi_v2_worker_count(32) == 8

    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '2')
    assert metric_mutual_information._mi_v2_worker_count(1000) == 2
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '1')
    assert metric_mutual_information._mi_v2_worker_count(1000) == 1

    def unexpected_parallel(*args, **kwargs):
        pytest.fail('small v2 input should use serial computation')

    monkeypatch.setattr(metric_mutual_information, 'Parallel', unexpected_parallel)
    small = pd.DataFrame({'a': [1, 2], 'b': [2, 1], 'c': [1, 1]})
    matrix = metric_mutual_information._pairwise_nmi_v2(small)
    assert matrix.loc['a', 'b'] == 1.0

    monkeypatch.setattr(metric_mutual_information, 'cpu_count', lambda: 1)
    assert metric_mutual_information._mi_v2_worker_count(1000) == 1
    eligible = pd.DataFrame(np.arange(18).reshape(2, 9))
    assert metric_mutual_information._pairwise_nmi_v2(eligible).shape == (9, 9)


def test_mi_v2_parallel_matches_serial_full_outputs_and_uses_loky_processes(monkeypatch):
    real, synt = _mi_v2_frames()
    monkeypatch.setattr(metric_mutual_information, 'cpu_count', lambda: 1)
    joblib_parallel = metric_mutual_information.Parallel

    def unexpected_parallel(*args, **kwargs):
        pytest.fail('one-worker MI v2 workload should use serial computation')

    monkeypatch.setattr(metric_mutual_information, 'Parallel', unexpected_parallel)
    serial_real, serial_synt, serial_supports, serial_hash = mutual_information_v2(
        real,
        synt,
        num_cols=['num0', 'num1', 'num2', 'num3', 'num4', 'constant', 'missing'],
        cat_cols=['cat0', 'cat1', 'cat2'],
        num_quants=5,
    )
    serial_metric = MutualInformation(
        real, synt, cat_cols=['cat0', 'cat1', 'cat2'],
        num_cols=['num0', 'num1', 'num2', 'num3', 'num4', 'constant', 'missing'],
        do_preprocessing=False, verbose=False, plot_figures=False,
    )
    serial_result = serial_metric.evaluate(num_quants=5)

    monkeypatch.setattr(metric_mutual_information, 'cpu_count', lambda: 2)
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '2')
    worker_pids = set()
    parallel_calls = []
    monkeypatch.setattr(metric_mutual_information, 'Parallel', joblib_parallel)

    def recording_parallel(*args, **kwargs):
        parallel_calls.append(kwargs)
        parallel = joblib_parallel(*args, **kwargs)

        class RecordingParallel:
            def __call__(self, tasks):
                results = parallel(tasks)

                def record_worker_pids():
                    for worker_pid, pair_values in results:
                        worker_pids.add(worker_pid)
                        yield worker_pid, pair_values

                return record_worker_pids()

        return RecordingParallel()

    monkeypatch.setattr(metric_mutual_information, 'Parallel', recording_parallel)
    parallel_real, parallel_synt, parallel_supports, parallel_hash = mutual_information_v2(
        real,
        synt,
        num_cols=['num0', 'num1', 'num2', 'num3', 'num4', 'constant', 'missing'],
        cat_cols=['cat0', 'cat1', 'cat2'],
        num_quants=5,
    )
    parallel_metric = MutualInformation(
        real, synt, cat_cols=['cat0', 'cat1', 'cat2'],
        num_cols=['num0', 'num1', 'num2', 'num3', 'num4', 'constant', 'missing'],
        do_preprocessing=False, verbose=False, plot_figures=False,
    )
    parallel_result = parallel_metric.evaluate(num_quants=5)

    assert parallel_calls
    assert all(call == {
        'n_jobs': 2,
        'backend': 'loky',
        'return_as': 'generator',
        'batch_size': 1,
        'pre_dispatch': 2,
    } for call in parallel_calls)
    assert worker_pids
    assert all(pid != os.getpid() for pid in worker_pids)
    _assert_frames_equal(parallel_real, serial_real)
    _assert_frames_equal(parallel_synt, serial_synt)
    assert parallel_supports == serial_supports
    assert parallel_hash == serial_hash
    _assert_results_equal(serial_result, parallel_result)


def test_mi_v2_one_row_and_pair_chunks_preserve_invalids_and_order(monkeypatch):
    chunks = metric_mutual_information._mi_v2_pair_chunks(5, 3)
    pairs = [pair for chunk in chunks for pair in chunk]
    assert pairs == [(i, j) for i in range(5) for j in range(i + 1, 5)]
    assert all(len(chunk) <= 3 for chunk in metric_mutual_information._mi_v2_pair_chunks(5, 3))

    one_row = pd.DataFrame(np.arange(10).reshape(1, 10))
    monkeypatch.setattr(metric_mutual_information, 'cpu_count', lambda: 1)
    serial_matrix = metric_mutual_information._pairwise_nmi_v2(one_row)
    monkeypatch.setattr(metric_mutual_information, 'cpu_count', lambda: 2)
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '2')
    parallel_budget_matrix = metric_mutual_information._pairwise_nmi_v2(one_row)
    _assert_frames_equal(serial_matrix, parallel_budget_matrix)
    assert np.diag(parallel_budget_matrix.to_numpy()).tolist() == [1.0] * 10
    assert np.isnan(parallel_budget_matrix.to_numpy()[~np.eye(10, dtype=bool)]).all()
    distance, valid_pairs, invalid_pairs, total_pairs = (
        metric_mutual_information._upper_triangle_rms_v2(
            parallel_budget_matrix, serial_matrix
        )
    )
    assert np.isnan(distance)
    assert (valid_pairs, invalid_pairs, total_pairs) == (0, 45, 45)


def _raise_mi_v2_pair_chunk(codes, pairs):
    raise RuntimeError('deliberate MI worker failure')


def test_mi_v2_worker_failure_propagates(monkeypatch):
    monkeypatch.setattr(metric_mutual_information, 'cpu_count', lambda: 2)
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '2')
    monkeypatch.setattr(
        metric_mutual_information,
        '_mi_v2_pair_chunk',
        _raise_mi_v2_pair_chunk,
    )
    codes = pd.DataFrame(np.arange(90).reshape(9, 10))
    with pytest.raises(RuntimeError, match='deliberate MI worker failure'):
        metric_mutual_information._pairwise_nmi_v2(codes)


def test_ks_worker_count_and_serial_fallback_use_cpu_budget(monkeypatch):
    monkeypatch.delenv('LOKY_MAX_CPU_COUNT', raising=False)
    monkeypatch.setattr(metric_kolmogorov_smirnov, 'cpu_count', lambda: 8)
    assert metric_kolmogorov_smirnov._ks_v2_worker_count(49, 10000, 0, 25) == 1
    assert metric_kolmogorov_smirnov._ks_v2_worker_count(50, 10000, 0, 25) == 8
    # The observed 100-column x 500-row case remains serial despite its width.
    assert metric_kolmogorov_smirnov._ks_v2_worker_count(100, 1000, 0, 25) == 1
    assert metric_kolmogorov_smirnov._ks_v2_worker_count(100, 1000, 10, 25) == 1
    assert metric_kolmogorov_smirnov._ks_v2_worker_count(100, 1000, 10, 100) == 8
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '2')
    assert metric_kolmogorov_smirnov._ks_v2_worker_count(100, 10000, 0, 25) == 2
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '1')
    assert metric_kolmogorov_smirnov._ks_v2_worker_count(100, 10000, 0, 25) == 1

    def unexpected_parallel(*args, **kwargs):
        pytest.fail('small KS input should use serial computation')

    monkeypatch.setattr(metric_kolmogorov_smirnov, 'Parallel', unexpected_parallel)
    real = pd.DataFrame({'x': [0.0, 1.0], 'empty': [np.nan, np.nan]})
    metric = KolmogorovSmirnovTest(
        real, real.copy(), cat_cols=[], num_cols=['x', 'empty'],
        do_preprocessing=False, verbose=False, plot_figures=False,
    )
    assert metric.evaluate()['ks_invalid_tests_v2'] == 1


def test_ks_parallel_matches_serial_order_seeds_and_full_outputs(monkeypatch):
    rng = np.random.default_rng(731)
    size = 1200
    real = pd.DataFrame({
        f'num{column}': rng.normal(loc=column / 10, size=size)
        for column in range(48)
    })
    real['cat_first'] = ['a' if row % 2 else 'b' for row in range(len(real))]
    real['empty'] = np.nan
    synt = real.copy()
    synt['num0'] = synt['num0'] + 0.25
    synt['cat_first'] = synt['cat_first'].shift(1)
    cat_cols = ['cat_first']
    num_cols = [column for column in real.columns if column not in cat_cols]

    monkeypatch.setattr(metric_kolmogorov_smirnov, 'cpu_count', lambda: 1)
    joblib_parallel = metric_kolmogorov_smirnov.Parallel

    def unexpected_parallel(*args, **kwargs):
        pytest.fail('one-worker KS workload should use serial computation')

    monkeypatch.setattr(metric_kolmogorov_smirnov, 'Parallel', unexpected_parallel)
    serial_metric = KolmogorovSmirnovTest(
        real, synt, cat_cols=cat_cols, num_cols=num_cols,
        do_preprocessing=False, verbose=False, plot_figures=False,
    )
    serial_result = serial_metric.evaluate(n_perms=25, random_state=19)

    monkeypatch.setattr(metric_kolmogorov_smirnov, 'cpu_count', lambda: 2)
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '2')
    parallel_calls = []
    worker_pids = set()
    monkeypatch.setattr(metric_kolmogorov_smirnov, 'Parallel', joblib_parallel)

    def recording_parallel(*args, **kwargs):
        parallel_calls.append(kwargs)
        parallel = joblib_parallel(*args, **kwargs)

        class RecordingParallel:
            def __call__(self, tasks):
                results = parallel(tasks)

                def record_worker_pids():
                    for worker_pid, result in results:
                        worker_pids.add(worker_pid)
                        yield worker_pid, result

                return record_worker_pids()

        return RecordingParallel()

    monkeypatch.setattr(metric_kolmogorov_smirnov, 'Parallel', recording_parallel)
    parallel_metric = KolmogorovSmirnovTest(
        real, synt, cat_cols=cat_cols, num_cols=num_cols,
        do_preprocessing=False, verbose=False, plot_figures=False,
    )
    parallel_result = parallel_metric.evaluate(n_perms=25, random_state=19)

    assert parallel_calls == [{
        'n_jobs': 2,
        'backend': 'loky',
        'return_as': 'generator',
        'batch_size': 1,
        'pre_dispatch': 2,
    }]
    assert worker_pids
    assert all(pid != os.getpid() for pid in worker_pids)
    _assert_results_equal(serial_result, parallel_result)
    column_results = parallel_result['ks_column_results_v2']
    assert [row['column'] for row in column_results] == list(real.columns)
    assert parallel_result['ks_invalid_columns_v2'] == ('empty',)
    assert parallel_result['ks_column_results_v2'][0]['pvalue'] == serial_result[
        'ks_column_results_v2'
    ][0]['pvalue']
    assert parallel_result['sigs cols'] == serial_result['sigs cols']


def _raise_ks_column_worker(*task):
    raise RuntimeError('deliberate KS worker failure')


def test_ks_worker_failure_propagates(monkeypatch):
    monkeypatch.setattr(metric_kolmogorov_smirnov, 'cpu_count', lambda: 2)
    monkeypatch.setenv('LOKY_MAX_CPU_COUNT', '2')
    monkeypatch.setattr(
        metric_kolmogorov_smirnov,
        '_evaluate_one_column_in_worker',
        _raise_ks_column_worker,
    )
    real = pd.DataFrame(np.arange(100_000).reshape(2000, 50))
    metric = KolmogorovSmirnovTest(
        real, real.copy(), cat_cols=[], num_cols=list(real.columns),
        do_preprocessing=False, verbose=False, plot_figures=False,
    )
    with pytest.raises(RuntimeError, match='deliberate KS worker failure'):
        metric.evaluate()


def test_qmse_v2_skips_categoricals_and_keeps_numeric_overflow_and_missing():
    real = pd.DataFrame({'num': [0.0, 1.0, 2.0, 3.0, np.nan], 'cat': ['a', 'a', 'b', 'b', 'a']})
    synt = pd.DataFrame({'num': [0.0, 1.0, 2.0, 100.0, np.nan], 'cat': ['a', 'b', 'unknown', 'b', 'a']})
    metric = QuantileMSE(
        real,
        synt,
        cat_cols=['cat'],
        num_cols=['num'],
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )
    result = metric.evaluate(num_quants=2, cat_mse=False)
    assert result['qMSE_columns_v2'] == ('num',)
    assert result['qMSE_details_v2'][0]['support']['overflow'] == 3
    assert result['qMSE_details_v2'][0]['support']['missing'] == 4
    assert metric.normalize_output_v2()[0]['metric'] == 'avg_qMSE_v2'


def test_hellinger_v2_uses_shared_support_and_is_finite_for_degenerate_columns():
    real = pd.DataFrame({'constant': [1.0] * 3, 'missing': [np.nan] * 3, 'cat': ['a', 'a', 'b']})
    synt = pd.DataFrame({'constant': [1.0, 2.0, 1.0], 'missing': [np.nan] * 3, 'cat': ['a', 'b', 'unknown']})
    metric = HellingerDistance(
        real,
        synt,
        cat_cols=['cat'],
        num_cols=['constant', 'missing'],
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )
    result = metric.evaluate()
    assert np.isfinite(result['avg'])
    assert np.isfinite(result['err'])
    assert _hellinger([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert _hellinger([0.0, 0.0], [1.0, 0.0]) == 1.0


def test_pmse_v2_uses_actual_synthetic_prevalence_and_calibration_metadata():
    real = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0], 'cat': ['a', 'a', 'b', 'b']})
    synt = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 'cat': ['a', 'b', 'a', 'b', 'a', 'b']})
    metric = PropensityMeanSquaredError(
        real,
        synt,
        cat_cols=['cat'],
        num_cols=['x'],
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )
    result = metric.evaluate(k_folds=2)
    assert result['pMSE_prevalence_c_v2'] == 0.6
    assert np.isclose(result['pMSE_normalizer_v2'], 0.24)
    assert result['pMSE_oof_n_v2'] == 10
    assert np.isfinite(result['pMSE_calibration_residual_v2'])
    assert normalize_pmse_v2(0.0, 0.6) == 1.0
    assert metric.normalize_output_v2()[0]['metric'] == 'avg_pMSE_v2'


def test_ks_v2_is_full_precision_seeded_and_reports_valid_tests():
    assert np.isclose(_total_variation_distance(['a', 'a', 'b'], ['a', 'b', 'b']), 1 / 3)
    assert _discrete_ks(['a', 'a', 'b'], ['a', 'b', 'b'], n_perms=25, random_state=7) == _discrete_ks(
        ['a', 'a', 'b'], ['a', 'b', 'b'], n_perms=25, random_state=7
    )
    real = pd.DataFrame({'cat': ['a', 'a', 'b', 'b'], 'num': [0.0, 1.0, 2.0, np.nan], 'empty': [np.nan] * 4})
    synt = pd.DataFrame({'cat': ['a', 'b', 'b', 'b'], 'num': [0.0, 1.0, 3.0, 4.0], 'empty': [np.nan] * 4})
    metric = KolmogorovSmirnovTest(
        real,
        synt,
        cat_cols=['cat'],
        num_cols=['num', 'empty'],
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )
    result = metric.evaluate(n_perms=25, random_state=7)
    assert result['ks_valid_tests_v2'] == 2
    assert result['ks_invalid_tests_v2'] == 1
    assert 'empty' in result['ks_invalid_columns_v2']
    assert metric.normalize_output_v2()[0]['metadata']['valid_tests'] == 2


def test_mmd_v2_identical_frames_are_perfect_and_exposes_audit_estimate():
    frame = pd.DataFrame({'x': [0.0, 1.0, 2.0], 'rank': [1, 2, 3], 'kind': ['a', 'b', 'a']})
    result = mixed_rbf_mmd_v2(frame, frame.copy(), continuous_columns=['x'],
                              ordinal_columns=['rank'], nominal_columns=['kind'])
    assert result['score'] == 1.0
    assert result['b_mmd'] == 0.0
    assert np.isfinite(result['u_mmd'])
    assert result['fit_role'] == 'train'


def test_mmd_v2_nominal_recoding_does_not_change_result():
    train = pd.DataFrame({'kind': ['low', 'high', 'low']})
    candidate = pd.DataFrame({'kind': ['high', 'low', 'high']})
    recoded_train = train.replace({'low': 100, 'high': -4})
    recoded_candidate = candidate.replace({'low': 100, 'high': -4})
    first = mixed_rbf_mmd_v2(train, candidate, nominal_columns=['kind'])
    second = mixed_rbf_mmd_v2(recoded_train, recoded_candidate, nominal_columns=['kind'])
    assert first['b_mmd'] == second['b_mmd']


def test_mmd_v2_ordinal_distance_is_sensitive_and_fit_bandwidth_is_train_only():
    train = pd.DataFrame({'rank': [0, 1, 2]})
    near = pd.DataFrame({'rank': [0, 1, 2]})
    far = pd.DataFrame({'rank': [2, 2, 2]})
    near_result = mixed_rbf_mmd_v2(train, near, ordinal_columns=['rank'])
    far_result = mixed_rbf_mmd_v2(train, far, ordinal_columns=['rank'])
    assert near_result['b_mmd'] < far_result['b_mmd']
    with_tuning = mixed_rbf_mmd_v2(train, far, ordinal_columns=['rank'],
                                   tuning=pd.DataFrame({'rank': [100]}), fit_role='final')
    assert with_tuning['fit_role'] == 'train+tuning'
    assert with_tuning['bandwidth'] != near_result['bandwidth']


def test_mmd_v2_uses_declared_ordinal_order_and_rejects_unknown_or_null_values():
    train = pd.DataFrame({'rank': ['middle', 'low', 'high']})
    result = mixed_rbf_mmd_v2(train, train.copy(), ordinal_columns=['rank'],
                              ordinal_orders={'rank': ['low', 'middle', 'high']})
    assert result['score'] == 1.0
    with pytest.raises(ValueError, match='unseen values'):
        mixed_rbf_mmd_v2(train, pd.DataFrame({'rank': ['other']}), ordinal_columns=['rank'],
                         ordinal_orders={'rank': ['low', 'middle', 'high']})
    with pytest.raises(ValueError, match='null'):
        mixed_rbf_mmd_v2(train, pd.DataFrame({'rank': [None]}), ordinal_columns=['rank'],
                         ordinal_orders={'rank': ['low', 'middle', 'high']})


def test_mmd_v2_rejects_bad_continuous_values_schema_and_weights():
    train = pd.DataFrame({'x': [0.0, 1.0]})
    with pytest.raises(ValueError, match='finite'):
        mixed_rbf_mmd_v2(train, pd.DataFrame({'x': [np.inf]}), continuous_columns=['x'])
    with pytest.raises(KeyError, match='missing schema'):
        mixed_rbf_mmd_v2(train, pd.DataFrame({'y': [1.0]}), continuous_columns=['x'])
    with pytest.raises(ValueError, match='fixed equal weights'):
        mixed_rbf_mmd_v2(train, train.copy(), continuous_columns=['x'], weights={'continuous': 0})
    with pytest.raises(ValueError, match='fixed equal weights'):
        mixed_rbf_mmd_v2(train, train.copy(), continuous_columns=['x'], weights={'continuous': 1, 'unknown': 0})


def test_mmd_v2_candidate_and_final_fit_roles_are_explicit():
    train = pd.DataFrame({'x': [0.0, 1.0, 2.0]})
    tuning = pd.DataFrame({'x': [100.0]})
    candidate = mixed_rbf_mmd_v2(train, train.copy(), continuous_columns=['x'])
    final = mixed_rbf_mmd_v2(train, train.copy(), continuous_columns=['x'],
                             tuning=tuning, fit_role='final')
    assert candidate['fit_role'] == 'train'
    assert final['fit_role'] == 'train+tuning'
    with pytest.raises(ValueError, match='must not receive tuning'):
        mixed_rbf_mmd_v2(train, train.copy(), continuous_columns=['x'], tuning=tuning)


def test_mmd_v2_final_mode_uses_combined_reference_for_all_kernel_terms():
    train = pd.DataFrame({'x': [0.0, 1.0]})
    tuning = pd.DataFrame({'x': [10.0]})
    combined = pd.concat([train, tuning], ignore_index=True)
    final_match = mixed_rbf_mmd_v2(train, combined, continuous_columns=['x'],
                                   tuning=tuning, fit_role='final')
    final_train_only = mixed_rbf_mmd_v2(train, train.copy(), continuous_columns=['x'],
                                        tuning=tuning, fit_role='final')
    assert final_match['b_mmd'] == 0.0
    assert final_match['b_mmd_raw'] == 0.0
    assert final_train_only['b_mmd'] > 0.0


def test_mmd_v2_supports_singletons_for_biased_audit_only():
    result = mixed_rbf_mmd_v2(pd.DataFrame({'x': [0.0]}), pd.DataFrame({'x': [0.0]}),
                              continuous_columns=['x'])
    assert result['b_mmd'] == 0.0
    assert np.isnan(result['u_mmd'])


def test_mmd_v2_rejects_invalid_populations_and_keeps_biased_primary():
    result = mixed_rbf_mmd_v2(pd.DataFrame({'x': [0, 1]}), pd.DataFrame({'x': [0]}), continuous_columns=['x'])
    assert np.isnan(result['u_mmd'])
    assert np.isfinite(result['b_mmd'])


def test_mmd_metric_v2_api_returns_higher_is_better_score():
    frame = pd.DataFrame({'x': [0.0, 1.0, 2.0]})
    metric = MaximumMeanDiscrepancy(frame, frame.copy(), cat_cols=[], num_cols=['x'],
                                    do_preprocessing=False, verbose=False, plot_figures=False)
    assert metric.evaluate(version='mixed_rbf_mmd_v2', use_cats=False)['score'] == 1.0
