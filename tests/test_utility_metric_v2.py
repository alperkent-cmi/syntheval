import numpy as np
import pandas as pd
import pytest

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
from syntheval.metrics.utility.metric_mixed_correlation import (
    MixedCorrelation,
    _correlation_ratio_v2,
    _cramers_V_v2,
    mixed_correlation_v2,
)
from syntheval.metrics.utility.metric_mutual_information import mutual_information_v2
from syntheval.metrics.utility.metric_propensity_mse import (
    PropensityMeanSquaredError,
    normalize_pmse_v2,
)
from syntheval.metrics.utility.metric_quantile_mse import QuantileMSE
from syntheval.metrics.utility.metric_max_mean_discrepancy import (
    MaximumMeanDiscrepancy,
    mixed_rbf_mmd_v2,
)


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
    assert metric.normalize_output_v2()[0]['n_val'] == 1.0


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
