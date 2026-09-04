# Description: Hellinger distance metric class
# Author: Anton D. Lautrup
# Date: 18-08-2023

import numpy as np
import hashlib
import json
import pandas as pd

from syntheval.metrics.core.metric import MetricClass

def _scott_ref_rule(set1,set2):
    """Function for doing the Scott reference rule to calcualte number of bins needed to 
    represent the nummerical values.
    
    Args:
        set1 (array-like): Real data
        set2 (array-like): Synthetic data
    
    Returns:
        array : bin edges
    
    Example:
        >>> _scott_ref_rule([1,2,3,4,5],[1,2,3,4,5])
        array([1., 2., 3., 4., 5.])
    """
    samples = np.concatenate((set1, set2)).astype(float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return np.array([], dtype=float)
    std = np.std(samples)
    n = len(samples)
    if np.min(samples) == np.max(samples):
        value = float(samples[0])
        width = max(abs(value) * 1e-6, 0.5)
        return np.array([value - width, value + width], dtype=float)
    if np.percentile(samples, 75) - np.percentile(samples, 25) == 0:
        bins = np.percentile(samples, [0, 10, 25, 75, 90, 100])
        bins = np.unique(bins)
        if len(bins) < 2:
            width = max(abs(float(samples[0])) * 1e-6, 0.5)
            bins = np.array([samples[0] - width, samples[0] + width], dtype=float)
        return bins
    else:
        bin_width = max(
            1,
            np.ceil(
                n**(1/3) * std /
                (3.5 * (np.percentile(samples, 75) - np.percentile(samples, 25)))
            ).astype(int),
        )
        min_edge = min(samples); max_edge = max(samples)
        N = max(1, min(abs(int((max_edge-min_edge)/bin_width)),10000))
        bins = np.linspace(min_edge, max_edge, N + 1)
        return bins

def _hellinger(p,q):
    """Hellinger distance between distributions
    
    Args:
        p (array-like): Real data
        q (array-like): Synthetic data
    
    Returns:
        float : Hellinger distance
    
    Example:
        >>> _hellinger([1,2,3,4,5],[1,2,3,4,5])
        0.0
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("Hellinger distributions must use the same support shape.")
    p_sum = float(np.sum(p))
    q_sum = float(np.sum(q))
    if p_sum <= 0.0 and q_sum <= 0.0:
        return 0.0
    if p_sum <= 0.0 or q_sum <= 0.0:
        return 1.0
    p = p / p_sum
    q = q / q_sum
    sqrt_pdf1 = np.sqrt(np.clip(p, 0.0, None))
    sqrt_pdf2 = np.sqrt(np.clip(q, 0.0, None))
    diff = sqrt_pdf1 - sqrt_pdf2
    return float(np.clip(1/np.sqrt(2)*np.linalg.norm(diff), 0.0, 1.0))


def _is_missing_scalar(value):
    missing = pd.isna(value)
    return bool(missing) if np.isscalar(missing) else False


def _categorical_counts_v2(real_values, synt_values):
    support = [
        value for value in pd.unique(pd.concat([
            pd.Series(real_values), pd.Series(synt_values)
        ], ignore_index=True)) if not _is_missing_scalar(value)
    ]

    def _codes(values):
        series = pd.Series(values).reset_index(drop=True)
        codes = np.full(len(series), len(support), dtype=int)
        for code, value in enumerate(support):
            codes[series.eq(value).fillna(False).to_numpy(dtype=bool)] = code
        codes[series.isna().to_numpy(dtype=bool)] = len(support) + 1
        return codes

    support_size = len(support) + 2
    return (
        np.bincount(_codes(real_values), minlength=support_size),
        np.bincount(_codes(synt_values), minlength=support_size),
        {'kind': 'categorical', 'support': [repr(value) for value in support],
         'unknown': len(support), 'missing': len(support) + 1,
         'support_size': support_size},
    )


def _numeric_counts_v2(real_values, synt_values):
    real_numeric = pd.to_numeric(pd.Series(real_values), errors='coerce')
    synt_numeric = pd.to_numeric(pd.Series(synt_values), errors='coerce')
    finite = np.concatenate([
        real_numeric[np.isfinite(real_numeric)].to_numpy(dtype=float),
        synt_numeric[np.isfinite(synt_numeric)].to_numpy(dtype=float),
    ])
    edges = _scott_ref_rule(finite, [])
    if len(edges) == 0:
        def _empty_codes(values):
            numeric = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
            codes = np.ones(len(numeric), dtype=int)
            codes[np.isneginf(numeric)] = 2
            codes[np.isposinf(numeric)] = 3
            codes[np.isnan(numeric)] = 0
            return codes

        real_codes = _empty_codes(real_values)
        synt_codes = _empty_codes(synt_values)
        support = {
            'kind': 'numeric', 'bins': [], 'underflow': 2, 'overflow': 3,
            'missing': 0, 'support_size': 4,
        }
    else:
        bin_count = len(edges) - 1

        def _codes(values):
            numeric = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
            missing = np.isnan(numeric)
            codes = np.digitize(numeric, edges[1:-1], right=False).astype(int)
            codes[numeric < edges[0]] = bin_count
            codes[numeric > edges[-1]] = bin_count + 1
            codes[missing] = bin_count + 2
            return codes

        real_codes = _codes(real_values)
        synt_codes = _codes(synt_values)
        support = {
            'kind': 'numeric', 'bins': edges.tolist(), 'underflow': bin_count,
            'overflow': bin_count + 1, 'missing': bin_count + 2,
            'support_size': bin_count + 3,
        }
    return (
        np.bincount(real_codes, minlength=support['support_size']),
        np.bincount(synt_codes, minlength=support['support_size']),
        support,
    )


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


def _sem(values):
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _legacy_hellinger(real_data, synt_data, numerical, categorical):
    distances = []
    fallback_columns = []
    try:
        for column in categorical:
            class_num = len(np.unique(real_data[column]))
            try:
                real_histogram = np.histogram(real_data[column], bins=class_num)[0]
                synt_histogram = np.histogram(synt_data[column], bins=class_num)[0]
            except (TypeError, ValueError):
                real_items = pd.unique(real_data[column])
                real_histogram = np.array([
                    np.sum(real_data[column] == item) for item in real_items
                ])
                synt_histogram = np.array([
                    np.sum(synt_data[column] == item) for item in real_items
                ])
            distances.append(_hellinger(real_histogram, synt_histogram))
        for column in numerical:
            edges = _scott_ref_rule(
                np.asarray(real_data[column], dtype=float),
                np.asarray(synt_data[column], dtype=float),
            )
            if len(edges) < 2:
                real_histogram, synt_histogram, _ = _numeric_counts_v2(
                    real_data[column], synt_data[column]
                )
                fallback_columns.append(column)
            else:
                real_histogram = np.histogram(real_data[column], bins=edges)[0]
                synt_histogram = np.histogram(synt_data[column], bins=edges)[0]
            distances.append(_hellinger(real_histogram, synt_histogram))
    except (TypeError, ValueError) as exc:
        return np.nan, np.nan, f'{type(exc).__name__}: {exc}'
    if not distances:
        return np.nan, np.nan, 'No legacy Hellinger columns were selected.'
    error = np.nan if len(distances) < 2 else float(
        np.std(distances, ddof=1) / np.sqrt(len(distances))
    )
    note = (
        f'v2 stable support fallback for legacy columns: {fallback_columns}'
        if fallback_columns else None
    )
    return float(np.mean(distances)), error, note

class HellingerDistance(MetricClass):

    def name() -> str:
        """name/keyword to reference the metric"""
        return 'h_dist'

    def type() -> str:
        """privacy or utility"""
        return 'utility'

    def evaluate(self) -> float | dict:
        """ Function for evaluating the metric
        
        Returns:
            dict: Average Hellinger distance and standard error of the mean
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [0, 1, 0], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [0, 1, 0], 'b': [4, 5, 6]})
            >>> HD = HellingerDistance(real, fake, cat_cols=['a'], num_cols=['b'], do_preprocessing=False)
            >>> HD.evaluate()
            {'avg': 0.0, 'err': 0.0, ...}
        """
        numerical, categorical = _resolve_columns_v2(
            self.real_data, self.num_cols, self.cat_cols
        )
        selected = categorical + [column for column in numerical if column not in categorical]
        if not selected:
            raise ValueError("Hellinger distance did not run, no columns were selected!")

        legacy_average, legacy_error, legacy_error_message = _legacy_hellinger(
            self.real_data, self.synt_data, numerical, categorical
        )

        distances = []
        details = []
        for column in selected:
            if column in categorical:
                real_counts, synt_counts, support = _categorical_counts_v2(
                    self.real_data[column], self.synt_data[column]
                )
            else:
                real_counts, synt_counts, support = _numeric_counts_v2(
                    self.real_data[column], self.synt_data[column]
                )
            distance = _hellinger(real_counts, synt_counts)
            distances.append(distance)
            details.append({'column': column, 'hellinger_v2': distance, 'support': support})

        average = float(np.mean(distances))
        support_hash = hashlib.sha256(
            json.dumps(
                {item['column']: item['support'] for item in details},
                sort_keys=True,
                separators=(',', ':'),
            ).encode()
        ).hexdigest()
        self.results = {
            'avg': legacy_average,
            'err': legacy_error,
            'hellinger_legacy_error': legacy_error_message,
            'avg_v2': average,
            'err_v2': _sem(distances),
            'hellinger_columns_v2': tuple(selected),
            'hellinger_valid_columns_v2': len(selected),
            'hellinger_invalid_columns_v2': 0,
            'hellinger_details_v2': details,
            'hellinger_support_hash_v2': support_hash,
        }
        return self.results

    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        row = ('utility', 'Average empirical Hellinger distance', 
               self.results['avg'], self.results['err'])
        return [row]

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            return [{'metric': 'avg_h_dist', 'dim': 'u', 
                     'val': self.results['avg'], 
                     'err': self.results['err'], 
                     'n_val': 1-self.results['avg'], 
                     'n_err': self.results['err'], 
                     }]
        else: pass

    def normalize_output_v2(self) -> list:
        """Return bounded v2 Hellinger similarity and support metadata."""
        if self.results == {}:
            return []
        value = self.results['avg_v2']
        normalized = float(np.clip(1.0 - value, 0.0, 1.0))
        return [{
            'metric': 'avg_h_dist_v2',
            'dim': 'u',
            'val': value,
            'err': self.results['err_v2'],
            'n_val': normalized,
            'n_err': self.results['err_v2'],
            'metric_version': 'v2',
            'raw_value': value,
            'normalized_value': normalized,
            'metadata': {
                'columns': list(self.results['hellinger_columns_v2']),
                'valid_columns': self.results['hellinger_valid_columns_v2'],
                'invalid_columns': self.results['hellinger_invalid_columns_v2'],
                'support_hash': self.results['hellinger_support_hash_v2'],
            },
        }]
