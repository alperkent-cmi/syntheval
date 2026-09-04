# Description: Maximum Mean Discrepancy metric for utility evaluation of synthetic data
# Author: Anton D. Lautrup
# Date: 06-02-2026

import numpy as np
import pandas as pd

from syntheval.metrics.core.metric import MetricClass
from typing import List, Literal

from scipy.spatial.distance import cdist
from syntheval.utils.preprocessing import MixedSchemaPreprocessor


def _mixed_kernel(left, right, bandwidth, weights):
    """Build role-safe mixed-schema RBF kernel from component distances."""
    distance = np.zeros((len(left["continuous"]), len(right["continuous"])), dtype=float)
    for role in ("continuous", "ordinal"):
        if left[role].shape[1]:
            distance += weights[role] * cdist(left[role], right[role], "sqeuclidean")
    if left["nominal"].shape[1]:
        nominal = np.zeros_like(distance)
        for column in range(left["nominal"].shape[1]):
            nominal += left["nominal"][:, column, None] != right["nominal"][None, :, column]
        distance += weights["nominal"] * nominal
    return np.exp(-distance / (2.0 * bandwidth**2))


def mixed_rbf_mmd_v2(train, candidate, *, continuous_columns=None,
                     ordinal_columns=None, nominal_columns=None,
                     ordinal_orders=None, tuning=None, fit_role="candidate", weights=None):
    """Calculate mixed-schema RBF MMD using state fitted on train roles.

    ``tuning`` is optional. When supplied, final fit state uses train+tuning;
    candidate representation and bandwidth otherwise use train only. Returned
    ``b_mmd`` is primary; ``u_mmd`` is retained for audit only.
    """
    if fit_role not in ("candidate", "final"):
        raise ValueError("fit_role must be 'candidate' or 'final'.")
    if train.empty or candidate.empty:
        raise ValueError("MMD v2 requires non-empty train and candidate populations.")
    if fit_role == "final" and (tuning is None or tuning.empty):
        raise ValueError("Final MMD v2 mode requires non-empty tuning data.")
    if fit_role == "candidate" and tuning is not None:
        raise ValueError("Candidate MMD v2 mode must not receive tuning data.")
    continuous_columns = list(continuous_columns or [])
    ordinal_columns = list(ordinal_columns or [])
    nominal_columns = list(nominal_columns or [])
    fit_frame = pd.concat([train, tuning], ignore_index=True) if fit_role == "final" else train
    preprocessor = MixedSchemaPreprocessor.fit(
        fit_frame, continuous_columns, ordinal_columns, nominal_columns, ordinal_orders
    )
    train_values = preprocessor.transform(train, role="train")
    candidate_values = preprocessor.transform(candidate, role="candidate")
    fit_values = preprocessor.transform(fit_frame, role=fit_role)
    active = [role for role, values in fit_values.items() if values.shape[1]]
    fixed_weights = {role: (1.0 / len(active) if role in active else 0.0)
                     for role in ("continuous", "ordinal", "nominal")}
    if weights is not None:
        if set(weights) != set(fixed_weights) or any(
                not np.isclose(weights[key], fixed_weights[key]) for key in fixed_weights):
            raise ValueError("MMD v2 uses fixed equal weights across active roles; overrides are not allowed.")
    weights = fixed_weights
    fit_kernel = _mixed_kernel(fit_values, fit_values, 1.0, weights)
    fit_distances = -2.0 * np.log(np.maximum(fit_kernel[np.triu_indices(len(fit_kernel), 1)], 1e-300))
    positive = fit_distances[fit_distances > 0]
    bandwidth = float(np.sqrt(np.median(positive))) if positive.size else 1.0
    bandwidth = max(bandwidth, np.finfo(float).eps)
    reference_values = fit_values if fit_role == "final" else train_values
    kxx = _mixed_kernel(reference_values, reference_values, bandwidth, weights)
    kyy = _mixed_kernel(candidate_values, candidate_values, bandwidth, weights)
    kxy = _mixed_kernel(reference_values, candidate_values, bandwidth, weights)
    raw_biased = float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
    biased = max(raw_biased, 0.0)
    reference_size = len(reference_values["continuous"])
    if reference_size < 2 or len(candidate) < 2:
        unbiased = float("nan")
    else:
        unbiased = float((kxx.sum() - np.trace(kxx)) / (reference_size * (reference_size - 1))
                         + (kyy.sum() - np.trace(kyy)) / (len(candidate) * (len(candidate) - 1))
                         - 2.0 * kxy.mean())
    return {
        "version": "mixed_rbf_mmd_v2",
        "b_mmd": biased,
        "b_mmd_raw": raw_biased,
        "b_mmd_clip": max(biased, 0.0),
        "u_mmd": unbiased,
        "u_mmd_clip": max(unbiased, 0.0) if np.isfinite(unbiased) else np.nan,
        "score": max(0.0, 1.0 - np.sqrt(max(biased, 0.0))),
        "bandwidth": bandwidth,
        "weights": weights,
        "fit_role": "train+tuning" if tuning is not None else "train",
        "preprocessing": preprocessor.metadata(),
    }

def _linear_kernel(A, B):
    """ Linear kernel, K(x, y) = x^T y """
    return A @ B.T

def _polynomial_kernel(A, B, degree=3, coef0=1):
    """ Polynomial kernel, K(x, y) = (x^T y + coef0)^degree """
    return (A @ B.T + coef0) ** degree

def _rbf_kernel(A, B, gamma):
    """ RBF kernel, K(x, y) = exp(-gamma * ||x-y||^2) """
    sq_dists = cdist(A, B, "sqeuclidean")
    return np.exp(-gamma * sq_dists / 2)

def _off_diagonal_mean(K):
    """ get the unbiased estimate of the mean of the off-diagonal elements of a kernel matrix K """
    n = K.shape[0]
    return (K.sum() - np.trace(K)) / (n * (n - 1))

class MaximumMeanDiscrepancy(MetricClass):
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
        return 'mmd'

    def type() -> str:
        """ Set to 'privacy', 'utility' or 'fairness' """
        return 'utility'

    def evaluate(self,
                 use_cats: bool = True,
                 kernel: Literal['rbf', 'linear', 'poly'] = 'rbf',
                 gamma: float = None,
                 degree: int = 3,
                 coef0: float = 1,
                 version: str = 'legacy',
                 tuning_data=None,
                 ordinal_cols=None,
                 ordinal_orders=None,
                 fit_role='candidate',
                 weights=None,
                 ) -> dict:
        """ Function for calculating Maximum Mean Discrepancy (MMD) between the real and synthetic datasets.
        Calculates both the biased (V statistic) and unbiased (U statistic) estimates of MMD^2. 
        Note that the U statistic can be negative why we provide max(U, 0) as a more stable estimate. 
        
        Args:
            use_cats (bool): Whether to include categorical columns in the MMD calculation. Default is True.
            kernel (str): The kernel to use for MMD calculation. Options are 'rbf', 'linear', and 'poly'. Default is 'rbf'.
            gamma (float): The gamma parameter for the RBF kernel. Default sets it to 1 / (2 * median of pairwise squared distances) as a heuristic.
            degree (int): The degree parameter for the polynomial kernel. Default is 3.
            coef0 (float): The coef0 parameter for the polynomial kernel. Default is 1.
        
        Returns:
            dict: A dictionary containing the MMD value and related information.

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> MMD = MaximumMeanDiscrepancy(real, fake, cat_cols=[], num_cols=['a', 'b'], analysis_target='a', do_preprocessing=False)
            >>> MMD.evaluate(use_cats=False, kernel='linear') # doctest: +ELLIPSIS
            {'kernel': 'linear', 'u_mmd': ..., 'u_mmd_clip': 0.0, 'b_mmd': 0.0, 'b_mmd_clip': 0.0}
        """
        if version == 'mixed_rbf_mmd_v2':
            result = mixed_rbf_mmd_v2(
                self.real_data,
                self.synt_data,
                continuous_columns=self.num_cols or [],
                ordinal_columns=ordinal_cols or [],
                nominal_columns=self.cat_cols or [],
                tuning=tuning_data,
                ordinal_orders=ordinal_orders,
                fit_role=fit_role,
                weights=weights,
            )
            self.results.update(result)
            return self.results
        if version != 'legacy':
            raise ValueError("version must be 'legacy' or 'mixed_rbf_mmd_v2'.")
        try:
            assert use_cats and len(self.cat_cols) > 0 or not use_cats, "Categorical columns must be specified if use_cats is True."
            assert len(self.num_cols) > 0 or use_cats, "Numerical columns must be specified if use_cats is False."
            assert kernel in ['rbf', 'linear', 'poly'], "Kernel must be one of 'rbf', 'linear', or 'poly'."
        except AssertionError as e:
            raise ValueError(str(e))
        
        if use_cats:
            X = self.real_data[self.cat_cols + self.num_cols].values
            Y = self.synt_data[self.cat_cols + self.num_cols].values
        else:
            X = self.real_data[self.num_cols].values
            Y = self.synt_data[self.num_cols].values

        # Standardize the data based on the real data statistics
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        Y_scaled = (Y - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        match kernel:
            case "linear":
                Kxx = _linear_kernel(X_scaled, X_scaled)
                Kyy = _linear_kernel(Y_scaled, Y_scaled)
                Kxy = _linear_kernel(X_scaled, Y_scaled)
            case "poly":
                self.results['poly_degree'] = degree
                self.results['poly_coef0'] = coef0
                Kxx = _polynomial_kernel(X_scaled, X_scaled, degree, coef0)
                Kyy = _polynomial_kernel(Y_scaled, Y_scaled, degree, coef0)
                Kxy = _polynomial_kernel(X_scaled, Y_scaled, degree, coef0)
            case "rbf":
                if gamma is None:
                    Z = np.vstack([X_scaled, Y_scaled])
                    dists = cdist(Z, Z, "sqeuclidean")
                    gamma = 1.0 / (2 * np.median(dists[dists > 0]))
                    if gamma == 0:  # Handle case where all points are identical
                        gamma = 1.0

                self.results['rbf_gamma'] = float(gamma)
                Kxx = _rbf_kernel(X_scaled, X_scaled, gamma)
                Kyy = _rbf_kernel(Y_scaled, Y_scaled, gamma)
                Kxy = _rbf_kernel(X_scaled, Y_scaled, gamma)

        self.results['kernel'] = kernel

        self.results['u_mmd'] = float(_off_diagonal_mean(Kxx) + _off_diagonal_mean(Kyy) - 2 * Kxy.mean())
        self.results['u_mmd_clip'] = float(max(self.results['u_mmd'], 0))  # U-statistics can be negative at finite sample sizes due to variance
        self.results['b_mmd'] = float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())
        self.results['b_mmd_clip'] = float(max(self.results['b_mmd'], 0))  # V-statistics can also be negative at finite sample sizes due to variance

        return self.results

    def format_output(self) -> List[tuple]:
        """ Return a list of tuples for printing results to the rich console.
        Required format: [(metric type in lowercase, string (max 43 characters), value, error)]
        Example:
            [('utility', 'Metric description', 0.1234, 0.0123)]
        """
        if self.results.get('version') == 'mixed_rbf_mmd_v2':
            return [('utility', 'Mixed-schema RBF MMD^2 (biased)', self.results['b_mmd_clip'], None),
                    ('utility', ' -> MMD v2 score', self.results['score'], None),
                    ('utility', ' -> Unbiased audit estimate', self.results['u_mmd_clip'], None)]
        match self.results.get('kernel', None):
            case 'linear':
                kernel_desc = 'linear'
            case 'poly':
                kernel_desc = f'poly d={self.results["poly_degree"]}'
            case 'rbf':
                kernel_desc = f'rbf g={self.results["rbf_gamma"]:.2f}'

        rows = [('utility', f'Maximum Mean Discrepancy^2 ({kernel_desc})', None, None),
                ('utility', f' -> Biased estimate (V statistic)', self.results['b_mmd_clip'], None),
                ('utility', f' -> Unbiased estimate (U statistic)', self.results['u_mmd_clip'], None),
                ]
        return rows

    def normalize_output(self) -> List[dict]:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).
        
        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results.get('version') == 'mixed_rbf_mmd_v2':
            return [{'metric': 'mixed_rbf_mmd_v2', 'dim': 'u',
                     'val': self.results['b_mmd_clip'], 'n_val': self.results['score']}]
        if self.results != {}:
            return [{'metric': 'b_mmd', 'dim': 'u', 'val': self.results['b_mmd_clip'], 'n_val': max(0, 1-self.results['b_mmd_clip']**0.5)},
                    {'metric': 'u_mmd', 'dim': 'u', 'val': self.results['u_mmd_clip'], 'n_val': max(0, 1-self.results['u_mmd_clip']**0.5)}]
        else: pass
