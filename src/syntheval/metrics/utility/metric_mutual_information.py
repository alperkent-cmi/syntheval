# Description: Mutual information metric and plot
# Author: Anton D. Lautrup
# Date: 21-08-2023

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from syntheval.metrics.core.metric import MetricClass

from syntheval.utils.plot_metrics import plot_matrix_heatmap
from sklearn.metrics import normalized_mutual_info_score

#: Below this column count, the row-parallel path isn't worth the loky
#: process-pool overhead (~0.1-0.5s) -- e.g. small doctest-sized inputs.
_PARALLEL_MIN_COLS = 50

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

    # Encode each column to string once up front (same semantics as the previous
    # per-pair `.astype(str)` -- NaNs still collapse to the shared 'nan' string
    # category) instead of re-casting/relabeling it for every pair it appears
    # in. That was O(d^2) redundant string work (e.g. d=1038 -> ~2M wasted
    # casts) instead of O(d); this is the dominant cost at large d.
    codes = {lab: pd.factorize(data[lab].astype(str))[0] for lab in labs}

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

class MutualInformation(MetricClass):

    def name() -> str:
        """name/keyword to reference the metric"""
        return 'mi_diff'

    def type() -> str:
        """privacy or utility"""
        return 'utility'

    def evaluate(self, axs_lim=(0,1), axs_scale='Blues') -> float | dict:
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
            {'mutual_inf_diff': 0.0, 'mi_mat_dims': 2}
        """
        r_mi = _pairwise_attributes_mutual_information(self.real_data)
        f_mi = _pairwise_attributes_mutual_information(self.synt_data)

        mi_mat = r_mi - f_mi
        if self.plot_figures: plot_matrix_heatmap(mi_mat,'Mutual information matrix difference', 'mi', axs_lim, axs_scale)
        
        self.results = {'mutual_inf_diff': float(np.linalg.norm(mi_mat, ord='fro')),'mi_mat_dims': len(mi_mat)}
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