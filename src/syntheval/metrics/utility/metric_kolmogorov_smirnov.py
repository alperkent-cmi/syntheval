# Description: Kolmogorov–Smirnov test implementation 
# Author: Anton D. Lautrup
# Date: 21-08-2023

import os
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from scipy.stats import ks_2samp, permutation_test
from syntheval.metrics.core.metric import MetricClass
from syntheval.utils.plot_metrics import plot_significantly_dissimilar_variables

# Avoid process startup for low-work KS inputs even when they are wide. The
# estimate counts numeric sorting work and categorical permutation work.
_PARALLEL_MIN_COLS = 50
_PARALLEL_MIN_WORK = 1_250_000


def _ks_v2_worker_count(column_count, sample_count, categorical_count, n_perms):
    """Choose bounded workers only when estimated per-column work merits them.

    ``sample_count`` is the combined real and synthetic row count. Numeric KS
    work scales with sorting; categorical TVD permutation work scales with
    both samples and the requested permutation count.
    """
    if column_count < _PARALLEL_MIN_COLS or column_count < 1:
        return 1
    sample_count = max(1, sample_count)
    categorical_count = min(column_count, max(0, categorical_count))
    numeric_count = column_count - categorical_count
    sort_factor = int(np.ceil(np.log2(max(2, sample_count))))
    estimated_work = sample_count * (
        numeric_count * sort_factor + categorical_count * n_perms
    )
    if estimated_work < _PARALLEL_MIN_WORK:
        return 1
    available = cpu_count()
    configured_limit = os.environ.get('LOKY_MAX_CPU_COUNT')
    if configured_limit is not None:
        available = min(available, int(configured_limit))
    return min(max(1, available), column_count)


def _is_missing_scalar(value):
    missing = pd.isna(value)
    return bool(missing) if np.isscalar(missing) else False


def _canonical_values(values):
    return np.asarray([
        '__syntheval_missing__' if _is_missing_scalar(value) else value
        for value in list(values)
    ], dtype=object)

def _total_variation_distance(x,y):
    """Function for calculating the TVD (KS statistic equivalent)
    
    Args:
        x (array-like): Real data
        y (array-like): Synthetic data
    
    Returns:
        float : Total variation distance
    
    Example:
        >>> _total_variation_distance([1,2,3,4,5],[1,2,3,4,5])
        0.0
    """
    x_values = _canonical_values(x)
    y_values = _canonical_values(y)
    if len(x_values) == 0 or len(y_values) == 0:
        raise ValueError('TVD requires non-empty samples.')
    X, Y = Counter(x_values), Counter(y_values)
    merged = X + Y

    return float(0.5 * sum(
        abs(X[key] / len(x_values) - Y[key] / len(y_values))
        for key in merged.keys()
    ))


def _discrete_ks(x, y, n_perms=1000, random_state=42):
    """Function for doing permutation test of discrete values in the KS test
    
    Args:
        x (array-like): Real data
        y (array-like): Synthetic data
        n_perms (int): Number of permutations
    
    Returns:
        float : KS statistic
        float : p-value
    
    Example:
        >>> _discrete_ks([1,2,3,4,5],[1,2,3,4,5])
        (0.0, 1.0)
    """
    x_values = _canonical_values(x)
    y_values = _canonical_values(y)
    if len(x_values) == 0 or len(y_values) == 0:
        raise ValueError('Discrete KS requires non-empty samples.')
    codes = pd.factorize(np.concatenate([x_values, y_values]), sort=False)[0]
    x_codes = codes[:len(x_values)]
    y_codes = codes[len(x_values):]
    res = permutation_test(
        (x_codes, y_codes),
        _total_variation_distance,
        n_resamples=n_perms,
        vectorized=False,
        permutation_type='independent',
        alternative='greater',
        rng=np.random.default_rng(random_state),
    )

    return float(res.statistic), float(res.pvalue)


def _evaluate_one_column(category, R, F, is_categorical, n_perms, random_state):
    """Run the (discrete or continuous) KS test for a single column.

    Standalone module-level function (rather than a method/closure) so it
    can be dispatched via joblib's 'loky' (process) backend without needing
    to pickle the whole metric instance -- only the two column Series are
    sent to the worker.

    Returns (category, is_categorical, statistic, pvalue, valid, test_name).
    """
    if is_categorical:
        real_values = _canonical_values(R)
        synt_values = _canonical_values(F)
        if len(real_values) == 0 or len(synt_values) == 0:
            return category, is_categorical, np.nan, np.nan, False, 'tvd_permutation'
        statistic, pvalue = _discrete_ks(
            synt_values, real_values, n_perms, random_state=random_state
        )
        test_name = 'tvd_permutation'
    else:
        real_values = pd.to_numeric(pd.Series(R), errors='coerce').dropna().to_numpy()
        synt_values = pd.to_numeric(pd.Series(F), errors='coerce').dropna().to_numpy()
        if len(real_values) == 0 or len(synt_values) == 0:
            return category, is_categorical, np.nan, np.nan, False, 'ks_2samp'
        KstestResult = ks_2samp(real_values, synt_values, nan_policy='omit')
        statistic, pvalue = KstestResult.statistic, KstestResult.pvalue
        test_name = 'ks_2samp'
    return category, is_categorical, float(statistic), float(pvalue), True, test_name


def _evaluate_one_column_in_worker(*task):
    """Return worker PID with result so process dispatch is auditable in tests."""
    return os.getpid(), _evaluate_one_column(*task)


class KolmogorovSmirnovTest(MetricClass):
    """The Metric Class is an abstract class that interfaces with 
    SynthEval. When initialised the class has the following attributes:

    Attributes:
    self.real_data : DataFrame
    self.synt_data : DataFrame
    self.hout_data : DataFrame
    self.cat_cols  : list of strings
    self.num_cols  : list of strings

    self.nn_dist   : string keyword
    
    """

    def name() -> str:
        """ Name/keyword to reference the metric"""
        return 'ks_test'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, sig_lvl=0.05, n_perms = 1000, random_state=42) -> float | dict:
        """Function for executing the Kolmogorov-Smirnov test.

        Args:
            sig_lvl (float): Significance level
            n_perms (int): Number of permutations
        
        Returns:
            dict: Average KS statistic and standard error of the mean

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> KST = KolmogorovSmirnovTest(real, fake, cat_cols=['a'], num_cols=['b'], do_preprocessing=False)
            >>> KST.evaluate(sig_lvl=0.05, n_perms=10) # doctest: +ELLIPSIS
            {'avg stat': 0.0, ...}
        """
        if not 0.0 <= sig_lvl <= 1.0:
            raise ValueError('SynthEval(ks_test): sig_lvl must be between 0 and 1.')
        if not isinstance(n_perms, (int, np.integer)) or n_perms < 1:
            raise ValueError('SynthEval(ks_test): n_perms must be a positive integer.')
        if not isinstance(random_state, (int, np.integer)):
            raise ValueError('SynthEval(ks_test): random_state must be an integer.')
        n_dists, c_dists = [], []
        legacy_c_dists = []
        pvals = []
        sig_cols = []
        column_results = []
        
        self.sig_lvl = sig_lvl

        columns = list(self.real_data.columns)
        cat_col_set = set(self.cat_cols)
        tasks = [
            (
                category,
                self.real_data[category],
                self.synt_data[category],
                category in cat_col_set,
                n_perms,
                int(random_state) + index,
            )
            for index, category in enumerate(columns)
        ]

        workers = _ks_v2_worker_count(
            len(columns),
            len(self.real_data) + len(self.synt_data),
            len(cat_col_set.intersection(columns)),
            n_perms,
        )
        if workers > 1:
            worker_results = Parallel(
                n_jobs=workers,
                backend='loky',
                return_as='generator',
                batch_size=1,
                pre_dispatch=workers,
            )(delayed(_evaluate_one_column_in_worker)(*task) for task in tasks)
            results = (result for _worker_pid, result in worker_results)
        else:
            results = [_evaluate_one_column(*task) for task in tasks]

        for category, is_categorical, statistic, pvalue, valid, test_name in results:
            column_results.append({
                'column': category,
                'test': test_name,
                'is_categorical': is_categorical,
                'statistic': statistic,
                'pvalue': pvalue,
                'valid': valid,
            })
            if not valid:
                continue
            if is_categorical:
                c_dists.append(statistic)
                legacy_c_dists.append(float(np.round(statistic, 4)))
            else:
                n_dists.append(statistic)
            pvals.append(pvalue)
            if pvalue < sig_lvl:
                sig_cols.append(category)

        def _mean_and_se(values):
            if len(values) == 0:
                return np.nan, np.nan
            mean_val = float(np.mean(values))
            if len(values) < 2:
                return mean_val, np.nan
            se_val = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            return mean_val, se_val

        avg_ks, err_ks = _mean_and_se(n_dists)
        avg_tvd, err_tvd = _mean_and_se(c_dists)
        avg_stat, err_stat = _mean_and_se(n_dists + c_dists)
        legacy_avg_tvd, legacy_err_tvd = _mean_and_se(legacy_c_dists)
        legacy_avg_stat, legacy_err_stat = _mean_and_se(n_dists + legacy_c_dists)
        avg_pval, err_pval = _mean_and_se(pvals)
        valid_columns = [row['column'] for row in column_results if row['valid']]
        invalid_columns = [row['column'] for row in column_results if not row['valid']]
        frac_sigs = float(len(sig_cols) / len(pvals)) if pvals else np.nan

        ### Calculate number of significant tests, and fraction of significant tests
        self.results = {'avg stat' : float(legacy_avg_stat), 'stat err' : float(legacy_err_stat),
                'avg ks'   : float(avg_ks), 'ks err'   : float(err_ks),
                'avg tvd'  : float(legacy_avg_tvd), 'tvd err'  : float(legacy_err_tvd),
                        'avg pval' : float(avg_pval), 'pval err' : float(err_pval),
                        'num sigs' : len(sig_cols),
                        'frac sigs': frac_sigs,
                        'sigs cols': sig_cols,
                        'avg stat_v2': float(avg_stat),
                        'stat err_v2': float(err_stat),
                        'avg ks_v2': float(avg_ks),
                        'ks err_v2': float(err_ks),
                        'avg tvd_v2': float(avg_tvd),
                        'tvd err_v2': float(err_tvd),
                        'avg pval_v2': float(avg_pval),
                        'pval err_v2': float(err_pval),
                        'num sigs_v2': len(sig_cols),
                        'frac sigs_v2': frac_sigs,
                        'ks_valid_columns_v2': tuple(valid_columns),
                        'ks_invalid_columns_v2': tuple(invalid_columns),
                        'ks_valid_tests_v2': len(valid_columns),
                        'ks_invalid_tests_v2': len(invalid_columns),
                        'ks_column_results_v2': column_results,
                        'ks_n_perms_v2': int(n_perms),
                        'ks_seed_v2': int(random_state),
                        }

        if (self.plot_figures and sig_cols != []): plot_significantly_dissimilar_variables(self.real_data, self.synt_data, sig_cols, self.cat_cols)
        return self.results
    
    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        rows = [
            ("utility", "General Kolmogorov–Smirnov Statistic", self.results['avg stat'], self.results['stat err']),
            ("utility", "  -> Avg. Kolmogorov–Smirnov dist.", self.results['avg ks'], self.results['ks err']),
            ("utility", "  -> Avg. Total Variation Distance", self.results['avg tvd'], self.results['tvd err']),
            ("utility", "Fraction of Significant KS Tests", self.results['frac sigs'], None),
            ("utility", f"  -> # of Significant Tests at a={self.sig_lvl:.2f}", self.results['num sigs'], None),
            ("utility", "  -> Avg. combined p-value", self.results['avg pval'], self.results['pval err']),
        ]
        return rows
    
#     def format_output(self) -> str:
#         """ Return string for formatting the output, when the
#         metric is part of SynthEval. 
# |                                          :                    |"""
#         R = self.results
#         if self.results != {}:
#             string = """\
# | Kolmogorov–Smirnov / Total Variation Distance test            |
# |   -> average combined statistic          :   %.4f  %.4f   |
# |       -> avg. Kolmogorov–Smirnov dist.   :   %.4f  %.4f   |
# |       -> avg. Total Variation Distance   :   %.4f  %.4f   |
# |   -> average combined p-value            :   %.4f  %.4f   |
# |       -> # significant tests at a=%.2f   :   %2d               |
# |       -> fraction of significant tests   :   %.4f           |""" % (R['avg stat'], R['stat err'],
#                                                                       R['avg ks'], R['ks err'],
#                                                                       R['avg tvd'], R['tvd err'],
#                                                                       R['avg pval'], R['pval err'], 
#                                                                       self.sig_lvl, R['num sigs'],
#                                                                       R['frac sigs'])
#             return string
#         else: pass

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            R = self.results

            return [{'metric': 'ks_tvd_stat', 'dim': 'u', 
                     'val': R['avg stat'], 
                     'err': R['stat err'], 
                     'n_val': 1-R['avg stat'], 
                     'n_err': R['stat err'], 
                     },
                     {'metric': 'frac_ks_sigs', 'dim': 'u', 
                     'val': R['frac sigs'], 
                     'n_val': 1-R['frac sigs'], 
                     }]
        else: pass

    def normalize_output_v2(self) -> list:
        """Return full-precision KS/TVD identities with valid-test metadata."""
        if self.results == {}:
            return []
        R = self.results
        stat = R['avg stat_v2']
        frac = R['frac sigs_v2']
        return [
            {
                'metric': 'ks_tvd_stat_v2',
                'dim': 'u',
                'val': stat,
                'err': R['stat err_v2'],
                'n_val': float(np.clip(1.0 - stat, 0.0, 1.0)) if np.isfinite(stat) else np.nan,
                'n_err': R['stat err_v2'],
                'metric_version': 'v2',
                'raw_value': stat,
                'normalized_value': float(np.clip(1.0 - stat, 0.0, 1.0)) if np.isfinite(stat) else np.nan,
                'metadata': {
                    'valid_tests': R['ks_valid_tests_v2'],
                    'invalid_tests': R['ks_invalid_tests_v2'],
                    'valid_columns': list(R['ks_valid_columns_v2']),
                    'invalid_columns': list(R['ks_invalid_columns_v2']),
                    'n_perms': R['ks_n_perms_v2'],
                    'seed': R['ks_seed_v2'],
                },
            },
            {
                'metric': 'frac_ks_sigs_v2',
                'dim': 'u',
                'val': frac,
                'err': None,
                'n_val': float(np.clip(1.0 - frac, 0.0, 1.0)) if np.isfinite(frac) else np.nan,
                'n_err': None,
                'metric_version': 'v2',
                'raw_value': frac,
                'normalized_value': float(np.clip(1.0 - frac, 0.0, 1.0)) if np.isfinite(frac) else np.nan,
                'metadata': {
                    'valid_tests': R['ks_valid_tests_v2'],
                    'invalid_tests': R['ks_invalid_tests_v2'],
                    'seed': R['ks_seed_v2'],
                },
            },
        ]
