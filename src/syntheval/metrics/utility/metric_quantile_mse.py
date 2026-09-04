# Description: Template script for making new metric classes
# Author: Anton D. Lautrup
# Date: 21-08-2023

import numpy as np
import hashlib
import json
import pandas as pd

from syntheval.metrics.core.metric import MetricClass


def _quantile_edges_v2(values, num_quants):
    numeric = pd.to_numeric(pd.Series(values), errors='coerce')
    finite = numeric[np.isfinite(numeric)].to_numpy(dtype=float)
    if len(finite) == 0:
        return np.array([], dtype=float)
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, num_quants + 1))
    return np.unique(quantiles.astype(float))


def _numeric_codes_v2(values, edges):
    numeric = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    missing = np.isnan(numeric)
    if len(edges) == 0:
        codes = np.ones(len(numeric), dtype=int)
        codes[np.isneginf(numeric)] = 2
        codes[np.isposinf(numeric)] = 3
        codes[missing] = 0
        return codes, {
            'kind': 'numeric',
            'bins': [],
            'underflow': 2,
            'overflow': 3,
            'missing': 0,
            'support_size': 4,
        }
    bin_count = len(edges) if len(edges) == 1 else len(edges) - 1
    if len(edges) == 1:
        codes = np.zeros(len(numeric), dtype=int)
        codes[numeric < edges[0]] = bin_count
        codes[numeric > edges[0]] = bin_count + 1
    else:
        codes = np.digitize(numeric, edges[1:-1], right=False).astype(int)
        codes[numeric < edges[0]] = bin_count
        codes[numeric > edges[-1]] = bin_count + 1
    codes[missing] = bin_count + 2
    return codes, {
        'kind': 'numeric',
        'bins': edges.tolist(),
        'underflow': bin_count,
        'overflow': bin_count + 1,
        'missing': bin_count + 2,
        'support_size': bin_count + 3,
    }


def _is_missing_scalar(value):
    missing = pd.isna(value)
    return bool(missing) if np.isscalar(missing) else False


def _categorical_codes_v2(values, support):
    series = pd.Series(values).reset_index(drop=True)
    codes = np.full(len(series), len(support), dtype=int)
    for code, value in enumerate(support):
        codes[series.eq(value).fillna(False).to_numpy(dtype=bool)] = code
    codes[series.isna().to_numpy(dtype=bool)] = len(support) + 1
    return codes, {
        'kind': 'categorical',
        'support': [repr(value) for value in support],
        'unknown': len(support),
        'missing': len(support) + 1,
        'support_size': len(support) + 2,
    }


def _fraction_vector(codes, support_size):
    return np.bincount(codes, minlength=support_size).astype(float) / len(codes)


def _sem(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _resolve_columns_v2(real_data, num_cols, cat_cols):
    columns = list(real_data.columns)
    numerical = [column for column in (num_cols or []) if column in columns]
    categorical = [column for column in (cat_cols or []) if column in columns]
    assigned = set(numerical) | set(categorical)
    for column in columns:
        if column in assigned:
            continue
        if pd.api.types.is_numeric_dtype(real_data[column]):
            numerical.append(column)
        else:
            categorical.append(column)
    return numerical, categorical


def _legacy_qmse(real_data, synt_data, numerical, categorical, num_quants, cat_mse):
    columns = list(numerical)
    if cat_mse:
        columns.extend(column for column in categorical if column not in columns)
    if not columns:
        raise ValueError(
            "Quantile mse did not run, no nummerical attributes, or cat_mse not enabled!"
        )
    values = []
    try:
        for column in columns:
            if column in categorical and cat_mse:
                real_items = real_data[column].unique()
                synt_fraction = np.array([
                    np.sum(synt_data[column] == item) for item in real_items
                ]) / len(synt_data)
                real_fraction = np.array([
                    np.sum(real_data[column] == item) for item in real_items
                ]) / len(real_data)
                values.append(float(np.mean((synt_fraction - real_fraction) ** 2)))
            else:
                quantiles = np.quantile(
                    real_data[column], np.linspace(0, 1, num_quants + 1)
                )
                synthetic_histogram, _ = np.histogram(
                    synt_data[column], bins=quantiles.tolist()
                )
                synthetic_fraction = synthetic_histogram / len(synt_data)
                values.append(float(np.mean(
                    (synthetic_fraction - 1 / num_quants) ** 2
                )))
    except (TypeError, ValueError) as exc:
        return np.nan, np.nan, f'{type(exc).__name__}: {exc}'
    error = np.nan if len(values) < 2 else float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return float(np.mean(values)), error, None

class QuantileMSE(MetricClass):
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

    self.verbose   : bool (mainly for supressing prints and plots)

    """

    def name() -> str:
        """ Name/keyword to reference the metric"""
        return 'q_mse'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, num_quants=10, cat_mse=False) -> float | dict:
        """Function for executing the quantile mse metric.

        Args:
            num_quants (int): Number of quantiles to divide the data into
            cat_mse (bool): Enable categorical mse
    
        Returns:
            dict : holds avg. and standard error of the mean (SE)
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> QMSE = QuantileMSE(real, fake, cat_cols=[], num_cols=[], do_preprocessing=False)
            >>> QMSE.evaluate(num_quants=5, cat_mse=True) # doctest: +ELLIPSIS
            {'avg qMSE': ...
        """
        if not isinstance(num_quants, (int, np.integer)) or num_quants < 1:
            raise ValueError("Quantile mse did not run, num_quants must be a positive integer!")
        if len(self.real_data) == 0 or len(self.synt_data) == 0:
            raise ValueError("Quantile mse did not run, real and synthetic data must be non-empty!")

        numerical, categorical = _resolve_columns_v2(
            self.real_data, self.num_cols, self.cat_cols
        )
        selected = list(numerical)
        if cat_mse:
            selected.extend(column for column in categorical if column not in selected)
        if not selected:
            raise ValueError("Quantile mse did not run, no columns selected for qMSE!")

        legacy_average, legacy_error, legacy_error_message = _legacy_qmse(
            self.real_data,
            self.synt_data,
            numerical,
            categorical,
            int(num_quants),
            bool(cat_mse),
        )

        q_mse_values = []
        details = []
        for column in selected:
            if column in categorical and cat_mse:
                support = [
                    value for value in pd.unique(self.real_data[column])
                    if not _is_missing_scalar(value)
                ]
                real_codes, support_metadata = _categorical_codes_v2(
                    self.real_data[column], support
                )
                synt_codes, _ = _categorical_codes_v2(self.synt_data[column], support)
            else:
                edges = _quantile_edges_v2(self.real_data[column], int(num_quants))
                real_codes, support_metadata = _numeric_codes_v2(
                    self.real_data[column], edges
                )
                synt_codes, _ = _numeric_codes_v2(self.synt_data[column], edges)
            support_size = support_metadata['support_size']
            real_fraction = _fraction_vector(real_codes, support_size)
            synt_fraction = _fraction_vector(synt_codes, support_size)
            value = float(np.mean(np.square(synt_fraction - real_fraction)))
            q_mse_values.append(value)
            details.append({
                'column': column,
                'qMSE_v2': value,
                'support': support_metadata,
            })

        average = float(np.mean(q_mse_values))
        support_hash = hashlib.sha256(
            json.dumps(
                {item['column']: item['support'] for item in details},
                sort_keys=True,
                separators=(',', ':'),
            ).encode()
        ).hexdigest()
        self.results = {
            'avg qMSE': legacy_average,
            'qMSE err': legacy_error,
            'qMSE_legacy_error': legacy_error_message,
            'avg_qMSE_v2': average,
            'qMSE_err_v2': _sem(q_mse_values),
            'qMSE_columns_v2': tuple(selected),
            'qMSE_valid_columns_v2': len(selected),
            'qMSE_invalid_columns_v2': 0,
            'qMSE_details_v2': details,
            'qMSE_support_hash_v2': support_hash,
            'qMSE_num_quants_v2': int(num_quants),
            'qMSE_cat_mse_v2': bool(cat_mse),
        }
        return self.results

    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        if self.results != {}:
            row = ('utility', 'Quantile mean squared error (qMSE)', 
                   self.results['avg qMSE'], self.results['qMSE err'])
            return [row]
        else: pass

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            return [{'metric': 'avg_qMSE', 'dim': 'u', 
                     'val': self.results['avg qMSE'], 
                     'err': self.results['qMSE err'], 
                     'n_val': 1-self.results['avg qMSE'], 
                     'n_err': self.results['qMSE err'], 
                     }]
        else: pass

    def normalize_output_v2(self) -> list:
        """Return the real-support qMSE and bounded agreement identity."""
        if self.results == {}:
            return []
        value = self.results['avg_qMSE_v2']
        normalized = float(np.clip(1.0 - value, 0.0, 1.0))
        return [{
            'metric': 'avg_qMSE_v2',
            'dim': 'u',
            'val': value,
            'err': self.results['qMSE_err_v2'],
            'n_val': normalized,
            'n_err': self.results['qMSE_err_v2'],
            'metric_version': 'v2',
            'raw_value': value,
            'normalized_value': normalized,
            'metadata': {
                'columns': list(self.results['qMSE_columns_v2']),
                'valid_columns': self.results['qMSE_valid_columns_v2'],
                'invalid_columns': self.results['qMSE_invalid_columns_v2'],
                'support_hash': self.results['qMSE_support_hash_v2'],
                'num_quants': self.results['qMSE_num_quants_v2'],
                'cat_mse': self.results['qMSE_cat_mse_v2'],
            },
        }]
