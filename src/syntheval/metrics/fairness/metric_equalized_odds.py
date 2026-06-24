# Description: Equalized Odds fairness metric based on classification
# Author: Adapted from StatisticalParity (Tobias Hyrup)
# Date: 2026-06-24

from itertools import product
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from syntheval.metrics.core.metric import MetricClass


class EqualizedOdds(MetricClass):
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
        """Name/keyword to reference the metric"""
        return "equalized_odds"

    def type() -> str:
        """Set to 'privacy', 'utility', or 'fairness'"""
        return "fairness"

    @staticmethod
    def rate_difference(
        X: pd.DataFrame,
        S: str,
        y_true: np.ndarray | pd.Series,
        preds: np.ndarray | pd.Series,
        condition_label: int,
        positive_pred: int = 1,
    ) -> float:
        """Difference in the rate of positive predictions between groups, conditioned on the true label.

        For ``condition_label == positive_pred`` this is the true positive rate (TPR) difference, and for
        the other label it is the false positive rate (FPR) difference. Each rate is
        P(Yhat = positive_pred | Y = condition_label, S = group), and the returned value is rate(S=1) - rate(S=0).

        Args:
            X (pd.DataFrame): the data to compute the metric on (must contain the sensitive attribute column)
            S (str): the name of the sensitive attribute column in X
            y_true (np.ndarray | pd.Series): the ground-truth labels aligned with the predictions
            preds (np.ndarray | pd.Series): the predictions from a classifier
            condition_label (int): the true-label value to condition on (positive label -> TPR, other -> FPR)
            positive_pred (int, optional): the positive class of the classifier, either 0 or 1, by default 1

        Returns:
            float: the conditional positive-prediction rate difference between the groups, or ``np.nan`` if
            either group has no samples with the conditioned label

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> X = pd.DataFrame({'S': [0, 0, 1, 1]})
            >>> y_true = pd.Series([1, 1, 1, 1])
            >>> preds = np.array([1, 0, 1, 1])
            >>> EqualizedOdds.rate_difference(X, 'S', y_true, preds, condition_label=1)
            0.5
        """
        sensitive = np.asarray(X[S])
        y_true = np.asarray(y_true)
        preds = np.asarray(preds)

        positive_outcome = preds == positive_pred
        conditioned = y_true == condition_label

        def _rate(group_value: int) -> float:
            mask = conditioned & (sensitive == group_value)
            if mask.sum() == 0:
                return np.nan
            return float(positive_outcome[mask].mean())

        return _rate(1) - _rate(0)

    @staticmethod
    def equalized_odds(
        X: pd.DataFrame,
        S: str,
        y_true: np.ndarray | pd.Series,
        preds: np.ndarray | pd.Series,
        positive_pred: int = 1,
    ) -> float:
        """Function for computing the equalized odds difference between the protected and unprotected group.

        Equalized odds (Hardt et al., 2016) requires equal true positive rates (TPR) and false positive rates
        (FPR) across groups. This metric summarises both constraints as the mean of the absolute TPR gap and
        the absolute FPR gap, so 0 indicates equalized odds is satisfied.

        Args:
            X (pd.DataFrame): the data to compute the metric on (must contain the sensitive attribute column)
            S (str): the name of the sensitive attribute column in X
            y_true (np.ndarray | pd.Series): the ground-truth labels aligned with the predictions
            preds (np.ndarray | pd.Series): the predictions from a classifier to compute the metric on
            positive_pred (int, optional): the positive class of the classifier, either 0 or 1, by default 1

        Returns:
            float: the mean of the absolute TPR and FPR differences between the groups, or ``np.nan`` if
            neither gap can be estimated

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> X = pd.DataFrame({'S': [0, 0, 1, 1, 0, 0, 1, 1]})
            >>> y_true = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
            >>> preds = np.array([1, 0, 1, 1, 0, 0, 1, 0])
            >>> EqualizedOdds.equalized_odds(X, 'S', y_true, preds)
            0.5
        """
        assert len(X) == len(
            preds
        )  # Check that the length of the data and the predictions are the same
        assert len(X) == len(y_true)  # Check that the labels align with the data
        assert S in X.columns  # Check that the sensitive attribute is in the data
        assert positive_pred in [
            0,
            1,
        ]  # Check that the positive prediction is either 0 or 1

        if len(np.unique(preds)) > 2:
            warn(
                "The predictions are not binary. Running the metric on the positive class."
            )

        negative_pred = 1 - positive_pred
        tpr_gap = EqualizedOdds.rate_difference(
            X, S, y_true, preds, positive_pred, positive_pred
        )
        fpr_gap = EqualizedOdds.rate_difference(
            X, S, y_true, preds, negative_pred, positive_pred
        )

        gaps = [abs(gap) for gap in (tpr_gap, fpr_gap) if not np.isnan(gap)]
        if not gaps:
            return np.nan
        return float(np.mean(gaps))

    def evaluate(
        self, positive_class: int = 1, folds: int = 5, full_output: bool = False
    ) -> float | dict:
        """Function for evaluating the equalized odds of a classifier trained on the synthetic data.

        Args:
            positive_class (int): The positive class of the classifier. Default is 1
            folds (int, optional): Number of folds to use in the cross-validation, by default 5
            full_output (bool, optional): Whether to return full output

        Returns:
            dict: Dictionary with the equalized odds difference and its standard error
        """
        try:
            assert self.analysis_target is not None, (
                "SynthEval(equalized odds): metric did not run, no analysis target variable object specified!"
            )
            assert self.analysis_target.sensitive_vars is not None, (
                "SynthEval(equalized odds): metric did not run, no sensitive variable specified!"
            )

            target_vars = [
                key
                for (key, value) in self.analysis_target.target_types.items()
                if isinstance(value, int) and value == 2
            ]

            assert target_vars != [], (
                "SynthEval(equalized odds): metric did not run, no categorical target variables with exactly 2 unique values!"
            )

            protected_attributes = [
                var
                for var in self.analysis_target.sensitive_vars
                if self.real_data[var].nunique() == 2
            ]
            assert protected_attributes != [], (
                "SynthEval(equalized odds): metric did not run, no sensitive variables with exactly 2 unique values!"
            )

            assert positive_class in [0, 1], (
                "SynthEval(equalized odds): metric did not run, the positive class argument must be either 0 or 1"
            )
        except AssertionError as e:
            raise ValueError(str(e))

        self.full_output = full_output
        negative_class = 1 - positive_class
        result_rows = []
        for target_var, protected_attribute in product(
            target_vars, protected_attributes
        ):
            # Drop confounder variables for the current target variable (if any)
            confounders = self.analysis_target.confounder_vars[target_var]
            synt_data = self.synt_data.drop(confounders, axis=1)

            fake_x, fake_y = synt_data.drop([target_var], axis=1), synt_data[target_var]

            # Train a classifier for each fold
            differences, tpr_gaps, fpr_gaps = [], [], []
            for train_idxs, test_idxs in KFold(folds).split(fake_x, fake_y):
                X_train, X_test = fake_x.iloc[train_idxs], fake_x.iloc[test_idxs]
                y_train, y_test = fake_y.iloc[train_idxs], fake_y.iloc[test_idxs]

                # Train a classifier
                clf = RandomForestClassifier(n_estimators=100)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)

                differences.append(
                    self.equalized_odds(
                        X_test, protected_attribute, y_test, preds, positive_class
                    )
                )
                tpr_gaps.append(
                    self.rate_difference(
                        X_test,
                        protected_attribute,
                        y_test,
                        preds,
                        positive_class,
                        positive_class,
                    )
                )
                fpr_gaps.append(
                    self.rate_difference(
                        X_test,
                        protected_attribute,
                        y_test,
                        preds,
                        negative_class,
                        positive_class,
                    )
                )

            differences = np.asarray(differences, dtype=float)
            valid = differences[~np.isnan(differences)]
            if valid.size == 0:
                warn(
                    f"SynthEval(equalized odds): no fold yielded an estimable rate for "
                    f"'{target_var}' x '{protected_attribute}' (no conditioned samples in a group)."
                )
                mean_difference = np.nan
                se_difference = np.nan
            else:
                mean_difference = float(np.mean(valid))
                se_difference = (
                    float(np.std(valid, ddof=1) / np.sqrt(valid.size))
                    if valid.size > 1
                    else 0.0
                )

            target_var = target_var.replace(" ", "_").lower()
            result_rows.append(
                {
                    "target_var": target_var,
                    "protected_attribute": protected_attribute,
                    "equalized_odds": mean_difference,
                    "equalized_odds_se": se_difference,
                    "tpr_difference": float(np.nanmean(tpr_gaps))
                    if np.any(~np.isnan(tpr_gaps))
                    else np.nan,
                    "fpr_difference": float(np.nanmean(fpr_gaps))
                    if np.any(~np.isnan(fpr_gaps))
                    else np.nan,
                }
            )

        columns = [
            "target_var",
            "protected_attribute",
            "equalized_odds",
            "equalized_odds_se",
            "tpr_difference",
            "fpr_difference",
        ]

        row_values = [row["equalized_odds"] for row in result_rows]
        row_errors = [row["equalized_odds_se"] for row in result_rows]
        self.results["equalized_odds"] = float(np.nanmean(row_values))
        self.results["equalized_odds_se"] = float(
            np.sqrt(np.nansum([err**2 for err in row_errors])) / len(result_rows)
        )
        self.results["raw results"] = pd.DataFrame.from_records(
            result_rows, columns=columns
        )
        return self.results

    def format_output(self) -> list:
        """Return a list of tuples for printing results to the rich console."""
        rows = (
            "fairness",
            "Equalized Odds difference",
            self.results["equalized_odds"],
            self.results["equalized_odds_se"],
        )
        return [rows]

    def normalize_output(self) -> list:
        """This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            output = [
                {
                    "metric": "equalized_odds",
                    "dim": "f",
                    "val": self.results["equalized_odds"],
                    "err": self.results["equalized_odds_se"],
                    "n_val": 1 - abs(self.results["equalized_odds"]),
                    "n_err": self.results["equalized_odds_se"],
                }
            ]
            if self.full_output:
                for idx, row in self.results["raw results"].iterrows():
                    output.append(
                        {
                            "metric": "eqo_"
                            + row["target_var"]
                            + "_"
                            + row["protected_attribute"],
                            "dim": "f",
                            "val": row["equalized_odds"],
                            "err": row["equalized_odds_se"],
                            "n_val": 1 - abs(row["equalized_odds"]),
                            "n_err": row["equalized_odds_se"],
                        }
                    )
            return output
        else:
            pass
