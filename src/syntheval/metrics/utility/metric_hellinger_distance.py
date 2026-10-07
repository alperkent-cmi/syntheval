# Description: Hellinger distance metric class
# Author: Anton D. Lautrup
# Date: 18-08-2023

import numpy as np

from syntheval.metrics.core.metric import MetricClass

def _scott_ref_rule(set1,set2):
    """Shared histogram bin edges for real and synthetic numeric values, by
    Scott's reference rule (bin width 3.5 * std * n^(-1/3), Scott 1979) over
    the pooled range.

    Upstream computed the width as n^(1/3) * std / (3.5 * IQR), rounded up to
    an integer, which gives a single bin on [0, 1]-scaled data, so the
    Hellinger distance of every numeric column was exactly 0.
    
    Args:
        set1 (array-like): Real data
        set2 (array-like): Synthetic data
    
    Returns:
        array : bin edges
    
    Example:
        >>> _scott_ref_rule([1,2,3,4,5],[1,2,3,4,5])
        array([1., 3., 5.])
    """
    samples = np.concatenate((np.asarray(set1, dtype=float), np.asarray(set2, dtype=float)))
    samples = samples[np.isfinite(samples)]
    if samples.min() == samples.max():
        return np.array([samples.min() - 0.5, samples.max() + 0.5])
    edges = np.histogram_bin_edges(samples, bins='scott')
    if len(edges) > 10001:
        edges = np.linspace(samples.min(), samples.max(), 10001)
    return edges

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
    sqrt_pdf1 = np.sqrt(p)
    sqrt_pdf2 = np.sqrt(q)
    diff = sqrt_pdf1 - sqrt_pdf2
    return float(1/np.sqrt(2)*np.linalg.norm(diff))

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
            {'avg': 0.0, 'err': 0.0}
        """
        H_dist = []
    
        for category in self.cat_cols:
            # Count each category in both tables over one shared category set;
            # separate np.histogram calls binned each table over its own range.
            levels = np.union1d(np.unique(self.real_data[category]), np.unique(self.synt_data[category]))
            pdfR = self.real_data[category].value_counts().reindex(levels, fill_value=0).to_numpy()
            pdfF = self.synt_data[category].value_counts().reindex(levels, fill_value=0).to_numpy()
            H_dist.append(_hellinger(pdfR/sum(pdfR),pdfF/sum(pdfF)))
        
        for category in self.num_cols:
            n_bins = _scott_ref_rule(self.real_data[category],self.synt_data[category]) # Scott rule for finding bin width

            pdfR = np.histogram(self.real_data[category], bins=n_bins)[0]
            pdfF = np.histogram(self.synt_data[category], bins=n_bins)[0]
            H_dist.append(_hellinger(pdfR/sum(pdfR),pdfF/sum(pdfF)))

        self.results = {'avg': float(np.mean(H_dist)), 'err': float(np.std(H_dist,ddof=1)/np.sqrt(len(H_dist)))}
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
