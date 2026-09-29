# Description: Mutual information metric and plot
# Author: Anton D. Lautrup
# Date: 21-08-2023

import hashlib
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from sklearn.metrics import normalized_mutual_info_score
from syntheval.metrics.core.metric import MetricClass
from syntheval.utils.plot_metrics import plot_matrix_heatmap

#: Below this column count, the row-parallel path isn't worth the loky
#: process-pool overhead (~0.1-0.5s) -- e.g. small doctest-sized inputs.
_PARALLEL_MIN_COLS = 50
_V2_PARALLEL_MIN_PAIRS = 32
_V2_MAX_CHUNK_PAIRS = 512

def _pairwise_attributes_mutual_information(data):
    """Compute normalized mutual information for all pairwise attributes.

    Elements borrowed from: 
    Ping H, Stoyanovich J, Howe B. DataSynthesizer: privacy-preserving synthetic datasets. 2017
    Presented at: Proceedingsof the 29th International Conference on Scientific and Statistical Database Management; 2017; Chicago.
    [doi:10.1145/3085504.3091117]
    
    Args:
        data (DataFrame): Data
    
    Returns:
        DataFrame : Matrix
    
    Example:
        >>> _pairwise_attributes_mutual_information(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})) # doctest: +NORMALIZE_WHITESPACE
            a    b
        a  1.0  1.0
        b  1.0  1.0
    """
    labs = sorted(data.columns)
    n = len(labs)

    codes = {lab: pd.factorize(data[lab], sort=False)[0] for lab in labs}

    # normalized_mutual_info_score(a, b) == normalized_mutual_info_score(b, a),
    # so only the upper triangle (incl. diagonal) needs computing -- halves
    # the number of pairwise calls.
    def _row(i):
        return [normalized_mutual_info_score(codes[labs[i]], codes[labs[j]], average_method='arithmetic') for j in range(i, n)]

    if n >= _PARALLEL_MIN_COLS:
        # Process-based (loky) parallelism -- threads were measured *slower*
        # than sequential here (sklearn/pandas overhead doesn't release the
        # GIL enough), while loky gave ~8x on a 24-core machine. n_jobs=-2
        # matches the outer `benchmark()` Parallel's own choice (leaves one
        # core free); safe to nest -- joblib does not force this back to
        # sequential just because it's called from inside another Parallel.
        rows = Parallel(n_jobs=-2, backend='loky')(delayed(_row)(i) for i in range(n))
    else:
        rows = [_row(i) for i in range(n)]

    mat = np.empty((n, n), dtype=float)
    for i, row in enumerate(rows):
        for k, j in enumerate(range(i, n)):
            mat[i, j] = row[k]
            mat[j, i] = row[k]
    return pd.DataFrame(mat, columns=labs, index=labs)


def _quantile_edges_v2(values, num_quants):
    numeric = pd.to_numeric(pd.Series(values), errors='coerce')
    finite = numeric[np.isfinite(numeric)].to_numpy(dtype=float)
    if len(finite) == 0:
        return np.array([], dtype=float)
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, num_quants + 1))
    return np.unique(quantiles.astype(float))


def _encode_numeric_v2(values, edges):
    numeric = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    missing = ~np.isfinite(numeric)
    if len(edges) == 0:
        codes = np.where(missing, 0, 1).astype(int)
        return codes, {'kind': 'numeric', 'bins': [], 'underflow': 1, 'overflow': 1, 'missing': 0}
    bin_count = len(edges) if len(edges) == 1 else len(edges) - 1
    codes = np.zeros(len(numeric), dtype=int)
    if len(edges) == 1:
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
    }


def _is_missing_scalar(value):
    missing = pd.isna(value)
    return bool(missing) if np.isscalar(missing) else False


def _encode_categorical_v2(values, support):
    series = pd.Series(values).reset_index(drop=True)
    codes = np.full(len(series), len(support), dtype=int)
    for code, value in enumerate(support):
        matches = series.eq(value).fillna(False).to_numpy(dtype=bool)
        codes[matches] = code
    missing = series.isna().to_numpy(dtype=bool)
    codes[missing] = len(support) + 1
    return codes, {
        'kind': 'categorical',
        'support': [repr(value) for value in support],
        'unknown': len(support),
        'missing': len(support) + 1,
    }


def _resolve_columns_v2(real_data, num_cols, cat_cols):
    columns = list(real_data.columns)
    categorical = [column for column in (cat_cols or []) if column in columns]
    numerical = [column for column in (num_cols or []) if column in columns]
    assigned = set(categorical) | set(numerical)
    for column in columns:
        if column in assigned:
            continue
        if pd.api.types.is_numeric_dtype(real_data[column]):
            numerical.append(column)
        else:
            categorical.append(column)
    return numerical, categorical


def _mi_v2_worker_count(pair_count):
    """Choose workers bounded by joblib and the configured per-model budget."""
    if pair_count < _V2_PARALLEL_MIN_PAIRS:
        return 1
    available = cpu_count()
    configured_limit = os.environ.get('LOKY_MAX_CPU_COUNT')
    if configured_limit is not None:
        available = min(available, int(configured_limit))
    return min(max(1, available), pair_count)


def _mi_v2_pair_chunks(column_count, chunk_size):
    """Yield bounded upper-triangle coordinates in matrix order."""
    chunk = []
    for left_index in range(column_count):
        for right_index in range(left_index + 1, column_count):
            chunk.append((left_index, right_index))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _mi_v2_pair_chunk(codes, pairs):
    """Compute one bounded pair chunk and retain each pair's matrix position."""
    values = []
    for left_index, right_index in pairs:
        value = normalized_mutual_info_score(
            codes[:, left_index], codes[:, right_index], average_method='arithmetic'
        )
        values.append((left_index, right_index, value))
    return os.getpid(), values


def _pairwise_nmi_v2(codes):
    columns = list(codes.columns)
    matrix = np.full((len(columns), len(columns)), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)

    def store_pair_value(left_index, right_index, value):
        if np.isfinite(value):
            matrix[left_index, right_index] = float(np.clip(value, 0.0, 1.0))
            matrix[right_index, left_index] = matrix[left_index, right_index]

    pair_count = len(columns) * (len(columns) - 1) // 2
    workers = _mi_v2_worker_count(pair_count)
    code_values = codes.to_numpy(copy=False)
    if workers > 1 and len(codes) >= 2:
        chunk_size = max(1, (pair_count + workers * 4 - 1) // (workers * 4))
        chunk_size = min(chunk_size, _V2_MAX_CHUNK_PAIRS)
        parallel = Parallel(
            n_jobs=workers,
            backend='loky',
            return_as='generator',
            batch_size=1,
            pre_dispatch=workers,
        )
        chunks = _mi_v2_pair_chunks(len(columns), chunk_size)
        tasks = (delayed(_mi_v2_pair_chunk)(code_values, pairs) for pairs in chunks)
        for _worker_pid, pair_values in parallel(tasks):
            for left_index, right_index, value in pair_values:
                store_pair_value(left_index, right_index, value)
    elif len(codes) >= 2:
        for left_index in range(len(columns)):
            for right_index in range(left_index + 1, len(columns)):
                value = normalized_mutual_info_score(
                    code_values[:, left_index],
                    code_values[:, right_index],
                    average_method='arithmetic',
                )
                store_pair_value(left_index, right_index, value)
    return pd.DataFrame(matrix, columns=columns, index=columns)


def mutual_information_v2(real_data, synt_data, num_cols=None, cat_cols=None, num_quants=10):
    """Compute arithmetic NMI using supports fitted from the real reference frame."""
    if not isinstance(num_quants, (int, np.integer)) or num_quants < 1:
        raise ValueError('SynthEval(mi_diff): num_quants must be a positive integer.')
    if list(real_data.columns) != list(synt_data.columns):
        raise ValueError('SynthEval(mi_diff): real and synthetic columns must match.')
    numerical, categorical = _resolve_columns_v2(real_data, num_cols, cat_cols)
    numerical_set = set(numerical)
    categorical_set = set(categorical)
    real_codes = pd.DataFrame(index=real_data.index)
    synt_codes = pd.DataFrame(index=synt_data.index)
    supports = {}
    for column in real_data.columns:
        if column in numerical_set:
            edges = _quantile_edges_v2(real_data[column], int(num_quants))
            real_codes[column], support = _encode_numeric_v2(real_data[column], edges)
            synt_codes[column], _ = _encode_numeric_v2(synt_data[column], edges)
        elif column in categorical_set:
            support = [
                value for value in pd.unique(real_data[column])
                if not _is_missing_scalar(value)
            ]
            real_codes[column], support = _encode_categorical_v2(real_data[column], support)
            synt_codes[column], _ = _encode_categorical_v2(synt_data[column], support)
        supports[column] = support
    real_mi = _pairwise_nmi_v2(real_codes)
    synt_mi = _pairwise_nmi_v2(synt_codes)
    support_hash = hashlib.sha256(
        json.dumps(supports, sort_keys=True, separators=(',', ':'), default=repr).encode()
    ).hexdigest()
    return real_mi, synt_mi, supports, support_hash


def _upper_triangle_rms_v2(real_matrix, synt_matrix):
    size = len(real_matrix)
    upper = np.triu(np.ones((size, size), dtype=bool), k=1)
    differences = real_matrix.to_numpy() - synt_matrix.to_numpy()
    valid = upper & np.isfinite(differences)
    total_pairs = int(np.count_nonzero(upper))
    valid_pairs = int(np.count_nonzero(valid))
    invalid_pairs = total_pairs - valid_pairs
    if total_pairs == 0:
        distance = 0.0
    elif valid_pairs == 0:
        distance = np.nan
    else:
        distance = float(np.sqrt(np.mean(np.square(np.clip(differences[valid], -1.0, 1.0)))))
    return distance, valid_pairs, invalid_pairs, total_pairs

class MutualInformation(MetricClass):

    def name() -> str:
        """name/keyword to reference the metric"""
        return 'mi_diff'

    def type() -> str:
        """privacy or utility"""
        return 'utility'

    def evaluate(self, axs_lim=(0,1), axs_scale='Blues', num_quants=10) -> float | dict:
        """ Function for evaluating the metric
        
        Args:
            axs_lim (tuple): Axis limits (for plotting)
            axs_scale (str): Color scale (for plotting)
        
        Returns:
            dict: Mutual information matrix difference
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> M = MutualInformation(real, fake, do_preprocessing=False, plot_figures=False)
            >>> M.evaluate()
            {'mutual_inf_diff': 0.0, 'mi_mat_dims': 2, ...}
        """
        r_mi = _pairwise_attributes_mutual_information(self.real_data)
        f_mi = _pairwise_attributes_mutual_information(self.synt_data)

        mi_mat = r_mi - f_mi
        if self.plot_figures: plot_matrix_heatmap(mi_mat,'Mutual information matrix difference', 'mi', axs_lim, axs_scale)

        real_mi_v2, synt_mi_v2, supports_v2, support_hash_v2 = mutual_information_v2(
            self.real_data,
            self.synt_data,
            num_cols=self.num_cols,
            cat_cols=self.cat_cols,
            num_quants=num_quants,
        )
        mi_diff_v2, valid_pairs_v2, invalid_pairs_v2, total_pairs_v2 = _upper_triangle_rms_v2(
            real_mi_v2, synt_mi_v2
        )
        
        self.results = {'mutual_inf_diff': float(np.linalg.norm(mi_mat, ord='fro')),'mi_mat_dims': len(mi_mat)}
        self.results.update({
            'mutual_inf_diff_v2': mi_diff_v2,
            'mi_mat_dims_v2': len(real_mi_v2),
            'mi_valid_pairs_v2': valid_pairs_v2,
            'mi_invalid_pairs_v2': invalid_pairs_v2,
            'mi_total_pairs_v2': total_pairs_v2,
            'mi_supports_v2': supports_v2,
            'mi_support_hash_v2': support_hash_v2,
            'mi_num_quants_v2': int(num_quants),
            'mi_definition_v2': 'real_fitted_quantile_support_arithmetic_nmi',
        })
        return self.results

    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        row = [('utility','Pairwise mutual information difference', self.results['mutual_inf_diff'], None)]
        return row

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            n_elements = int(self.results['mi_mat_dims']*(self.results['mi_mat_dims']-1)/2)
            return [{'metric': 'mutual_inf_diff', 'dim': 'u', 
                     'val': self.results['mutual_inf_diff'], 
                     'n_val': 1-self.results['mutual_inf_diff']/n_elements, 
                     }]
        else: pass

    def normalize_output_v2(self) -> list:
        """Return bounded v2 MI distance and real-fitted support metadata."""
        if self.results == {}:
            return []
        distance = self.results['mutual_inf_diff_v2']
        normalized = float(np.clip(1.0 - distance, 0.0, 1.0)) if np.isfinite(distance) else np.nan
        return [{
            'metric': 'mutual_inf_diff_v2',
            'dim': 'u',
            'val': distance,
            'err': None,
            'n_val': normalized,
            'n_err': None,
            'metric_version': 'v2',
            'raw_value': distance,
            'normalized_value': normalized,
            'metadata': {
                'valid_pairs': self.results['mi_valid_pairs_v2'],
                'invalid_pairs': self.results['mi_invalid_pairs_v2'],
                'total_pairs': self.results['mi_total_pairs_v2'],
                'support_hash': self.results['mi_support_hash_v2'],
                'num_quants': self.results['mi_num_quants_v2'],
                'definition': self.results['mi_definition_v2'],
            },
        }]
