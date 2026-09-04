import numpy as np
import pandas as pd

from syntheval.metrics.privacy.metric_epsilon_identifiability import EpsilonIdentifiability
from syntheval.metrics.privacy.metric_nn_adversarial_accuracy import (
    NearestNeighbourAdversarialAccuracy,
)
from syntheval.metrics.privacy.metric_nn_distance_ratio import NearestNeighbourDistanceRatio
from syntheval.utils.nn_distance import _knn_distance


def test_equal_values_in_separate_frames_are_cross_dataset_neighbors():
    real = pd.DataFrame({'value': [0.0, 1.0, 2.0]})
    synthetic = real.copy()

    cross_distance = _knn_distance(
        real,
        synthetic,
        [],
        1,
        'euclid',
    )[0]
    within_distance = _knn_distance(real, real, [], 1, 'euclid')[0]

    assert np.array_equal(cross_distance, np.zeros(len(real)))
    assert np.all(within_distance > 0)


def test_epsilon_holdout_risk_uses_holdout_population_size():
    real = pd.DataFrame({'value': [0.0, 1.0, 2.0, 3.0]})
    synthetic = real.copy()
    holdout = pd.DataFrame({'value': [10.0, 11.0]})
    metric = EpsilonIdentifiability(
        real,
        synthetic,
        holdout,
        cat_cols=[],
        num_cols=['value'],
        nn_dist='euclid',
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )

    result = metric.evaluate()

    assert result['eps_risk'] == 1.0
    assert result['priv_loss'] == 1.0


def test_nndr_privacy_loss_reports_propagated_uncertainty():
    real = pd.DataFrame({'value': [0.0, 1.0, 2.0]})
    synthetic = real.copy()
    holdout = pd.DataFrame({'value': [10.0, 11.0, 12.0]})
    metric = NearestNeighbourDistanceRatio(
        real,
        synthetic,
        holdout,
        cat_cols=[],
        num_cols=['value'],
        nn_dist='euclid',
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )

    result = metric.evaluate()
    normalized = metric.normalize_output()

    assert np.isfinite(result['err'])
    assert normalized[1]['n_err'] == result['priv_loss_err']


def test_nnaa_is_a_privacy_metric_with_finite_single_round_error():
    real = pd.DataFrame({'value': [0.0, 1.0, 2.0, 3.0]})
    synthetic = pd.DataFrame({'value': [0.0, 1.0]})
    metric = NearestNeighbourAdversarialAccuracy(
        real,
        synthetic,
        cat_cols=[],
        num_cols=['value'],
        nn_dist='euclid',
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )

    result = metric.evaluate(n_resample=1)
    normalized = metric.normalize_output()

    assert NearestNeighbourAdversarialAccuracy.type() == 'privacy'
    assert np.isfinite(result['err'])
    assert normalized[0]['dim'] == 'p'


def test_nnaa_resampling_is_reproducible():
    real = pd.DataFrame({'value': [0.0, 1.0, 2.0, 3.0]})
    synthetic = pd.DataFrame({'value': [0.0, 1.0]})
    first = evaluate_nnaa(real, synthetic, seed=41)
    second = evaluate_nnaa(real, synthetic, seed=41)

    assert first == second


def evaluate_nnaa(real, synthetic, seed):
    metric = NearestNeighbourAdversarialAccuracy(
        real,
        synthetic,
        cat_cols=[],
        num_cols=['value'],
        nn_dist='euclid',
        do_preprocessing=False,
        verbose=False,
        plot_figures=False,
    )
    return metric.evaluate(n_resample=4, seed=seed)