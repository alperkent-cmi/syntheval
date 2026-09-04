# Description: propensity mean squared error
# Author: Anton D. Lautrup
# Date: 23-08-2023

import numpy as np
import pandas as pd
from numbers import Integral

from syntheval.metrics.core.metric import MetricClass

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score

from syntheval.utils.preprocessing import stack


def _sem(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def normalize_pmse_v2(pmse, synthetic_prevalence):
    """Normalize pMSE against the actual mixture variance c(1-c)."""
    denominator = float(synthetic_prevalence) * (1.0 - float(synthetic_prevalence))
    if denominator <= 0.0 or not np.isfinite(pmse):
        return np.nan
    return float(np.clip(1.0 - float(pmse) / denominator, 0.0, 1.0))


def _fold_features(train_frame, test_frame, numerical_columns):
    train = pd.get_dummies(train_frame, dummy_na=True, dtype=float)
    test = pd.get_dummies(test_frame, dummy_na=True, dtype=float)
    test = test.reindex(columns=train.columns, fill_value=0.0)
    medians = train.median(numeric_only=True)
    train = train.fillna(medians).fillna(0.0)
    test = test.fillna(medians).fillna(0.0)
    scale_columns = [column for column in numerical_columns if column in train.columns]
    if scale_columns:
        scaler = StandardScaler().fit(train[scale_columns])
        train.loc[:, scale_columns] = scaler.transform(train[scale_columns])
        test.loc[:, scale_columns] = scaler.transform(test[scale_columns])
    return train, test


def _legacy_pmse(real_data, synt_data, numerical_columns, k_folds, max_iter, solver):
    data = stack(real_data, synt_data).drop(['index'], axis=1)
    labels = data.pop('real')
    features = pd.get_dummies(data, dummy_na=True, dtype=float)
    scale_columns = [column for column in numerical_columns if column in features.columns]
    if scale_columns:
        features = features.astype({column: float for column in scale_columns})
        features.loc[:, scale_columns] = StandardScaler().fit_transform(
            features[scale_columns]
        )
    discriminator = LogisticRegression(max_iter=max_iter, solver=solver, random_state=42)
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    values, accuracies = [], []
    for train_index, test_index in kf.split(features, labels):
        model = discriminator.fit(features.iloc[train_index], labels.iloc[train_index])
        probabilities = model.predict_proba(features.iloc[test_index])
        test_labels = labels.iloc[test_index]
        num_synthetic = len(test_labels) - np.count_nonzero(test_labels)
        values.append(float(np.mean(
            (probabilities[:, 0] - num_synthetic / len(test_labels)) ** 2
        )))
        accuracies.append(float(f1_score(
            test_labels,
            model.predict(features.iloc[test_index]),
            average='macro',
            zero_division=0,
        )))
    return float(np.mean(values)), _sem(values), float(np.mean(accuracies)), _sem(accuracies)

class PropensityMeanSquaredError(MetricClass):
    """The Metric Class is an abstract class that interfaces with 
    SynthEval. When initialised the class has the following attributes:

    Attributes:
    self.real_data : DataFrame
    self.synt_data : DataFrame
    self.hout_data : DataFrame
    self.cat_cols  : list of strings
    self.num_cols  : list of strings

    self.nn_dist   : string keyword
    self.analysis_target: variable name
    """

    def name() -> str:
        """ Name/keyword to reference the metric"""
        return 'p_mse'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, k_folds=5, max_iter=100, solver='liblinear') -> float | dict:
        """Train a a discriminator to distinguish between real and fake data.
        
        Args:
            k_folds (int): Number of cross-validation folds
            max_iter (int): Maximum number of iterations for logistic regression
            solver (str): Solver for the logistic regression (see sklearn documentation)
        
        Returns:
            dict: Propensity mean squared error (pMSE) and classifier accuracy
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3, 2], 'b': [4, 5, 6, 4]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3, 2], 'b': [4, 5, 6, 4]})
            >>> PMSE = PropensityMeanSquaredError(real, fake, num_cols=['a', 'b'], do_preprocessing=False)
            >>> PMSE.evaluate(k_folds=2) # doctest: +ELLIPSIS
            {'avg pMSE': 0.0, ...}
        """

        if not isinstance(k_folds, Integral) or k_folds < 2:
            raise ValueError("SynthEval(p_mse): k_folds must be an integer of at least 2.")
        if not isinstance(max_iter, Integral) or max_iter < 1:
            raise ValueError("SynthEval(p_mse): max_iter must be a positive integer.")
        if len(self.real_data) == 0 or len(self.synt_data) == 0:
            raise ValueError("SynthEval(p_mse): real and synthetic data must be non-empty.")

        features = pd.concat([self.real_data, self.synt_data], ignore_index=True)
        labels = np.concatenate([
            np.zeros(len(self.real_data), dtype=int),
            np.ones(len(self.synt_data), dtype=int),
        ])
        counts = np.bincount(labels, minlength=2)
        if np.any(counts < k_folds):
            raise ValueError(
                f"SynthEval(p_mse): each source needs at least {k_folds} rows; "
                f"real={counts[0]}, synthetic={counts[1]}."
            )
        synthetic_prevalence = float(len(self.synt_data) / len(features))
        legacy_pmse, legacy_pmse_err, legacy_acc, legacy_acc_err = _legacy_pmse(
            self.real_data,
            self.synt_data,
            list(self.num_cols or []),
            k_folds,
            max_iter,
            solver,
        )
        fold_pmses, fold_acc = [], []
        oof_predictions = np.full(len(features), np.nan, dtype=float)
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(features, labels):
            x_train, x_test = _fold_features(
                features.iloc[train_index],
                features.iloc[test_index],
                list(self.num_cols or []),
            )
            y_train = labels[train_index]
            y_test = labels[test_index]
            discriminator = LogisticRegression(
                max_iter=max_iter, solver=solver, random_state=42
            )
            discriminator.fit(x_train, y_train)
            probabilities = discriminator.predict_proba(x_test)[:, 1]
            predictions = discriminator.predict(x_test)
            oof_predictions[test_index] = probabilities
            fold_pmses.append(
                float(np.mean((probabilities - synthetic_prevalence) ** 2))
            )
            fold_acc.append(float(f1_score(y_test, predictions, average='macro', zero_division=0)))

        pmse = float(np.mean((oof_predictions - synthetic_prevalence) ** 2))
        mean_predicted_prevalence = float(np.mean(oof_predictions))
        calibration_residual = mean_predicted_prevalence - synthetic_prevalence
        calibration_residual_mse = float(calibration_residual ** 2)
        brier_mse = float(np.mean((oof_predictions - labels) ** 2))
        self.results = {
            'avg pMSE': legacy_pmse,
            'pMSE err': legacy_pmse_err,
            'avg acc': legacy_acc,
            'acc err': legacy_acc_err,
            'avg_pMSE_v2': pmse,
            'pMSE_err_v2': _sem(fold_pmses),
            'pMSE_prevalence_c_v2': synthetic_prevalence,
            'pMSE_normalizer_v2': synthetic_prevalence * (1.0 - synthetic_prevalence),
            'pMSE_normalized_v2': normalize_pmse_v2(pmse, synthetic_prevalence),
            'pMSE_calibration_residual_v2': calibration_residual,
            'pMSE_calibration_residual_mse_v2': calibration_residual_mse,
            'pMSE_mean_predicted_prevalence_v2': mean_predicted_prevalence,
            'pMSE_brier_mse_v2': brier_mse,
            'pMSE_oof_n_v2': len(oof_predictions),
            'pMSE_fold_pMSE_v2': tuple(fold_pmses),
            'pMSE_fold_count_v2': len(fold_pmses),
            'pMSE_seed_v2': 42,
        }
        return self.results

    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        rows =[
            ("utility", "Propensity mean squared error (pMSE)", self.results['avg pMSE'], self.results['pMSE err']),
            ("utility", "  -> average pMSE classifier accuracy", self.results['avg acc'], self.results['acc err']),
        ]
        return rows
    
    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            return [{'metric': 'avg_pMSE', 'dim': 'u', 
                     'val': self.results['avg pMSE'], 
                     'err': self.results['pMSE err'], 
                     'n_val': 1-4*self.results['avg pMSE'], 
                     'n_err': 4*self.results['pMSE err'], 
                     }]
        else: pass

    def normalize_output_v2(self) -> list:
        """Return pMSE normalized by the actual real/synthetic mixture prevalence."""
        if self.results == {}:
            return []
        return [{
            'metric': 'avg_pMSE_v2',
            'dim': 'u',
            'val': self.results['avg_pMSE_v2'],
            'err': self.results['pMSE_err_v2'],
            'n_val': self.results['pMSE_normalized_v2'],
            'n_err': self.results['pMSE_err_v2'],
            'metric_version': 'v2',
            'raw_value': self.results['avg_pMSE_v2'],
            'normalized_value': self.results['pMSE_normalized_v2'],
            'metadata': {
                'synthetic_prevalence_c': self.results['pMSE_prevalence_c_v2'],
                'normalizer_c_times_one_minus_c': self.results['pMSE_normalizer_v2'],
                'calibration_residual': self.results['pMSE_calibration_residual_v2'],
                'calibration_residual_mse': self.results['pMSE_calibration_residual_mse_v2'],
                'mean_predicted_prevalence': self.results['pMSE_mean_predicted_prevalence_v2'],
                'brier_mse': self.results['pMSE_brier_mse_v2'],
                'oof_n': self.results['pMSE_oof_n_v2'],
                'fold_count': self.results['pMSE_fold_count_v2'],
                'seed': self.results['pMSE_seed_v2'],
            },
        }]
