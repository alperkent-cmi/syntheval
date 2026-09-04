# Description: Script for calculating the mixed correlation
# Author: Anton D. Lautrup
# Date: 23-08-2023

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from syntheval.metrics.core.metric import MetricClass

from scipy.stats import chi2_contingency
from syntheval.utils.plot_metrics import plot_matrix_heatmap

#: Below this column count, the row-parallel path isn't worth the loky
#: process-pool overhead (~0.1-0.5s) -- e.g. small doctest-sized inputs.
_PARALLEL_MIN_COLS = 50

def _cramers_V_legacy(var1,var2) :
    """function for calculating Cramers V between two categorial variables
    credit: https://www.kaggle.com/code/chrisbss1/cramer-s-v-correlation-matrix

    Args:
        var1 (array-like): Real data
        var2 (array-like): Synthetic data
    
    Returns:
        float : Cramers V
    
    Example:
        >>> _cramers_V([1,2,3,4,5],[1,2,3,4,5])
        1.0...
    """
    crosstab =np.array(pd.crosstab(var1, var2, rownames=None, colnames=None)) # Cross table building
    stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab) # Number of observations
    mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
    return float((stat/(obs*mini+1e-16)))

def _apply_mat(data,func,labs1,labs2):
    """Help function for constructing a matrix based on func accross labels 1 and 2
    
    Args:
        data (DataFrame): Data
        func (function): Function to apply
        labs1 (list): Labels 1
        labs2 (list): Labels 2

    Returns:
        DataFrame : Matrix
    
    Example:
        >>> _apply_mat(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), _cramers_V ,['a','b'], ['a','b']) # doctest: +NORMALIZE_WHITESPACE
            a    b
        a  1.0  1.0
        b  1.0  1.0
    """
    # Process-based (loky) parallelism, split one task per row -- threads were
    # measured *slower* than sequential for these funcs (pandas/scipy overhead
    # doesn't release the GIL enough), while loky gave ~8x on a 24-core
    # machine. n_jobs=-2 matches the outer `benchmark()` Parallel's own choice
    # (leaves one core free); safe to nest -- joblib does not force this back
    # to sequential just because it's called from inside another Parallel.
    # Below `_PARALLEL_MIN_COLS` the pool startup isn't worth it, so fall back
    # to a plain sequential loop (e.g. doctest-sized inputs).
    n1, n2 = len(labs1), len(labs2)
    use_parallel = max(n1, n2) >= _PARALLEL_MIN_COLS

    if labs1 == labs2:
        # Same label set on both axes -> the matrix is symmetric for both funcs
        # this helper is called with here (_cramers_V and mutual information),
        # so only the upper triangle needs computing -- halves the O(d^2) calls
        # that dominate wide datasets (e.g. loris's 822 categorical columns).
        n = n1
        def _row(i):
            return [func(data[labs1[i]], data[labs2[j]]) for j in range(i, n)]

        rows = (Parallel(n_jobs=-2, backend='loky')(delayed(_row)(i) for i in range(n))
                if use_parallel else [_row(i) for i in range(n)])
        mat = np.empty((n, n), dtype=float)
        for i, row in enumerate(rows):
            for k, j in enumerate(range(i, n)):
                mat[i, j] = row[k]
                mat[j, i] = row[k]
        return pd.DataFrame(mat, columns=labs2, index=labs1)

    def _row_full(lab1):
        return [func(data[lab1], data[lab2]) for lab2 in labs2]

    rows = (Parallel(n_jobs=-2, backend='loky')(delayed(_row_full)(lab1) for lab1 in labs1)
            if use_parallel else [_row_full(lab1) for lab1 in labs1])
    return pd.DataFrame(np.array(rows, dtype=float).reshape(n1, n2), columns=labs2, index=labs1)

def _correlation_ratio_legacy(categories, measurements):
    """Function for calculating the correlation ration eta^2 of categorial and nummerical data
    
    Args:
        categories (array): Categories
        measurements (array): Measurements
    
    Returns:
        float : Eta^2
    
    Example:
        >>> _correlation_ratio_legacy(np.array([0,1,0,1]),np.array([1,2,3,4]))
        0.2
    """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[fcat == i]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return float(eta)

def mixed_correlation_legacy(data,num_cols,cat_cols):
    """Function for calculating a correlation matrix of mixed datatypes.
    Spearman's rho is used for rank-based correlation, Cramer's V is used for categorical variables, 
    and correlation ratio is used for categorical and continuous variables.

    Args:
        data (DataFrame): Data
        num_cols (list): Numerical columns
        cat_cols (list): Categorical columns

    Returns:
        DataFrame : Correlation matrix

    Example:
        >>> mixed_correlation(pd.DataFrame({'num': [1, 2, 3], 'cat': [4, 5, 6]}),['num'],['cat']) # doctest: +NORMALIZE_WHITESPACE
            cat  num
        cat  1.0  1.0
        num  1.0  1.0
    """
    corr_num_num = data[num_cols].corr()
    corr_cat_cat = _apply_mat(data,_cramers_V_legacy,cat_cols,cat_cols)
    corr_cat_num = _apply_mat(data,_correlation_ratio_legacy,cat_cols,num_cols)
    if corr_cat_cat.empty: corr = corr_num_num
    elif corr_num_num.empty: corr = corr_cat_cat
    else:
        top_row = pd.concat([corr_cat_cat,corr_cat_num],axis=1)
        bot_row = pd.concat([corr_cat_num.transpose(),corr_num_num],axis=1)
        corr = pd.concat([top_row,bot_row],axis=0)
    return corr + np.diag(1-np.diag(corr))


def _safe_values_v2(values):
    series = pd.Series(values).reset_index(drop=True).astype(object)
    return series.where(~series.isna(), '__syntheval_missing__')


def _cramers_V_v2(var1, var2):
    """Compute conventional square-root Cramer's V for two categorical values."""
    left = _safe_values_v2(var1)
    right = _safe_values_v2(var2)
    if len(left) != len(right) or len(left) < 2:
        return np.nan
    crosstab = pd.crosstab(left, right, dropna=False)
    if min(crosstab.shape) < 2:
        return np.nan
    statistic = chi2_contingency(crosstab, correction=False)[0]
    denominator = len(left) * (min(crosstab.shape) - 1)
    value = np.sqrt(max(0.0, float(statistic) / denominator))
    return float(np.clip(value, 0.0, 1.0))


def _correlation_ratio_v2(categories, measurements):
    """Compute eta, rather than eta-squared, for categorical/numeric values."""
    category_values = pd.Series(categories).reset_index(drop=True)
    numeric_values = pd.to_numeric(
        pd.Series(measurements).reset_index(drop=True), errors='coerce'
    )
    valid = (~category_values.isna()) & np.isfinite(numeric_values)
    category_values = category_values[valid]
    numeric_values = numeric_values[valid]
    if len(category_values) < 2 or category_values.nunique(dropna=True) < 2:
        return np.nan
    total_mean = float(numeric_values.mean())
    denominator = float(np.sum((numeric_values.to_numpy() - total_mean) ** 2))
    if denominator <= 0.0:
        return np.nan
    grouped = pd.DataFrame({'category': category_values, 'value': numeric_values}).groupby(
        'category', observed=True
    )['value']
    numerator = float(np.sum(grouped.size().to_numpy() * (grouped.mean().to_numpy() - total_mean) ** 2))
    return float(np.clip(np.sqrt(max(0.0, numerator / denominator)), 0.0, 1.0))


def _spearman_v2(left, right):
    left = pd.to_numeric(pd.Series(left), errors='coerce')
    right = pd.to_numeric(pd.Series(right), errors='coerce')
    valid = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return np.nan
    value = left.corr(right, method='spearman')
    return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else np.nan


def mixed_correlation_v2(data, num_cols, cat_cols):
    """Return a Spearman/eta/Cramer's-V matrix and its valid-pair mask."""
    numerical = list(num_cols or [])
    categorical = list(cat_cols or [])
    labels = categorical + numerical
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    valid = np.zeros((len(labels), len(labels)), dtype=bool)
    np.fill_diagonal(matrix, 1.0)
    np.fill_diagonal(valid, True)

    numerical_set = set(numerical)
    categorical_set = set(categorical)
    for left_index, left_label in enumerate(labels):
        for right_index in range(left_index + 1, len(labels)):
            right_label = labels[right_index]
            if left_label in numerical_set and right_label in numerical_set:
                value = _spearman_v2(data[left_label], data[right_label])
            elif left_label in categorical_set and right_label in categorical_set:
                value = _cramers_V_v2(data[left_label], data[right_label])
            elif left_label in categorical_set:
                value = _correlation_ratio_v2(data[left_label], data[right_label])
            else:
                value = _correlation_ratio_v2(data[right_label], data[left_label])
            if np.isfinite(value):
                matrix[left_index, right_index] = value
                matrix[right_index, left_index] = value
                valid[left_index, right_index] = True
                valid[right_index, left_index] = True
    return pd.DataFrame(matrix, columns=labels, index=labels), pd.DataFrame(
        valid, columns=labels, index=labels
    )


def _upper_triangle_rms_v2(real_matrix, synt_matrix, real_valid, synt_valid):
    size = len(real_matrix)
    upper = np.triu(np.ones((size, size), dtype=bool), k=1)
    valid_pairs = upper & real_valid.to_numpy() & synt_valid.to_numpy()
    total_pairs = int(np.count_nonzero(upper))
    n_valid = int(np.count_nonzero(valid_pairs))
    n_invalid = total_pairs - n_valid
    if total_pairs == 0:
        score = 0.0
    elif n_valid == 0:
        score = np.nan
    else:
        differences = real_matrix.to_numpy() - synt_matrix.to_numpy()
        score = float(np.sqrt(np.mean(np.square(differences[valid_pairs] / 2.0))))
    return score, n_valid, n_invalid, total_pairs


def _cramers_V(var1, var2):
    """Compute conventional square-root Cramer's V."""
    return _cramers_V_v2(var1, var2)


def _correlation_ratio(categories, measurements):
    """Compute eta, rather than eta-squared, for categorical/numeric values."""
    return _correlation_ratio_v2(categories, measurements)


def mixed_correlation(data, num_cols, cat_cols):
    """Compute the corrected Spearman/Cramer's-V/eta correlation matrix."""
    numerical = list(num_cols or [])
    categorical = list(cat_cols or [])
    corr_num_num = data[numerical].corr(method='spearman')
    corr_cat_cat = _apply_mat(data, _cramers_V, categorical, categorical)
    corr_cat_num = _apply_mat(data, _correlation_ratio, categorical, numerical)
    if corr_cat_cat.empty:
        corr = corr_num_num
    elif corr_num_num.empty:
        corr = corr_cat_cat
    else:
        top_row = pd.concat([corr_cat_cat, corr_cat_num], axis=1)
        bot_row = pd.concat([corr_cat_num.transpose(), corr_num_num], axis=1)
        corr = pd.concat([top_row, bot_row], axis=0)
    return corr + np.diag(1 - np.diag(corr))

class MixedCorrelation(MetricClass):
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
        return 'corr_diff'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, mixed_corr=True, return_mats=False, axs_lim=(-1,1), axs_scale="RdBu") -> dict:
        """Function for calculating the (mixed) correlation matrix difference.
        This calculation uses spearmans rho for numerical-numerical, Cramer's V for categories,
        and correlation ratio (eta) for numerical-categorials.
                
        Args:
            mixed_corr (bool): Use mixed correlation
            return_mats (bool): Return the individual correlation matrices
            axs_lim (tuple): Axis limits (for plotting)
            axs_scale (str): Axis scale (for plotting)

        Returns:
            dict: Frobenius norm of the correlation matrix difference
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'num': [1, 2, 3], 'cat': [0, 1, 0]})
            >>> fake = pd.DataFrame({'num': [1, 2, 3], 'cat': [0, 1, 0]})
            >>> MC = MixedCorrelation(real, fake, cat_cols=['cat'], num_cols=['num'], do_preprocessing=False, plot_figures=False)
            >>> MC.evaluate(mixed_corr=True)
            {'corr_mat_diff': 0.0, 'corr_mat_dims': 2, ...}
        """
        self.mixed_corr = mixed_corr
        if mixed_corr:
            r_corr = mixed_correlation_legacy(self.real_data,self.num_cols,self.cat_cols)
            f_corr = mixed_correlation_legacy(self.synt_data,self.num_cols,self.cat_cols)
            corr_mat = r_corr-f_corr
            if self.plot_figures: plot_matrix_heatmap(corr_mat,'Mixed correlation matrix difference', 'corr', axs_lim, axs_scale)
        else:
            r_corr = self.real_data[self.num_cols].corr()
            f_corr = self.synt_data[self.num_cols].corr()
            corr_mat = r_corr-f_corr
            if self.plot_figures: plot_matrix_heatmap(corr_mat,'Correlation matrix difference (nums only)', 'corr', axs_lim, axs_scale)

        v2_num_cols = self.num_cols if mixed_corr else list(self.num_cols or [])
        v2_cat_cols = self.cat_cols if mixed_corr else []
        real_corr_v2, real_valid_v2 = mixed_correlation_v2(
            self.real_data, v2_num_cols, v2_cat_cols
        )
        synt_corr_v2, synt_valid_v2 = mixed_correlation_v2(
            self.synt_data, v2_num_cols, v2_cat_cols
        )
        corr_diff_v2, valid_pairs_v2, invalid_pairs_v2, total_pairs_v2 = _upper_triangle_rms_v2(
            real_corr_v2, synt_corr_v2, real_valid_v2, synt_valid_v2
        )
        
        self.results = {'corr_mat_diff': float(np.linalg.norm(corr_mat,ord='fro')), 'corr_mat_dims': len(corr_mat)}
        self.results.update({
            'corr_mat_diff_v2': corr_diff_v2,
            'corr_mat_dims_v2': len(real_corr_v2),
            'corr_valid_pairs_v2': valid_pairs_v2,
            'corr_invalid_pairs_v2': invalid_pairs_v2,
            'corr_total_pairs_v2': total_pairs_v2,
            'corr_definition_v2': 'spearman_cramers_v_eta_upper_triangle_rms',
        })
        if return_mats: self.results['real_cor_mat'] = r_corr
        if return_mats: self.results['synt_cor_mat'] = f_corr
        if return_mats: self.results['diff_cor_mat'] = corr_mat
        if return_mats: self.results['real_cor_mat_v2'] = real_corr_v2
        if return_mats: self.results['synt_cor_mat_v2'] = synt_corr_v2
        if return_mats: self.results['diff_cor_mat_v2'] = real_corr_v2 - synt_corr_v2
        return self.results

    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        if self.mixed_corr:
            row = [('utility','Mixed correlation matrix difference', self.results['corr_mat_diff'], None)]
        else:
            row = [('utility','Correlation difference (nums only)', self.results['corr_mat_diff'], None)]
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
            n_elements = int(self.results['corr_mat_dims']*(self.results['corr_mat_dims']-1)/2)
            return [{'metric': 'corr_mat_diff', 'dim': 'u', 
                     'val': self.results['corr_mat_diff'], 
                     'n_val': 1-self.results['corr_mat_diff']/n_elements, 
                     }]
        else: pass

    def normalize_output_v2(self) -> list:
        """Return bounded v2 correlation distance with valid-pair metadata."""
        if self.results == {}:
            return []
        distance = self.results['corr_mat_diff_v2']
        normalized = float(np.clip(1.0 - distance, 0.0, 1.0)) if np.isfinite(distance) else np.nan
        return [{
            'metric': 'corr_mat_diff_v2',
            'dim': 'u',
            'val': distance,
            'err': None,
            'n_val': normalized,
            'n_err': None,
            'metric_version': 'v2',
            'raw_value': distance,
            'normalized_value': normalized,
            'metadata': {
                'valid_pairs': self.results['corr_valid_pairs_v2'],
                'invalid_pairs': self.results['corr_invalid_pairs_v2'],
                'total_pairs': self.results['corr_total_pairs_v2'],
                'definition': self.results['corr_definition_v2'],
            },
        }]
