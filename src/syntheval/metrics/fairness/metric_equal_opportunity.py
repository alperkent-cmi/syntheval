# Description: Equal Opportunity fairness metric based on classification
# Author: Adapted from StatisticalParity (Tobias Hyrup)
# Date: 2026-06-24

from itertools import product
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from syntheval.metrics.core.metric import MetricClass


class EqualOpportunity(MetricClass):
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
        return "equal_opportunity"

    def type() -> str:
        """Set to 'privacy', 'utility', or 'fairness'"""
        return "fairness"

    @staticmethod
    def equal_opportunity(
        X: pd.DataFrame,
        S: str,
        y_true: np.ndarray | pd.Series,
        preds: np.ndarray | pd.Series,
        positive_pred: int = 1,
    ) -> float:
        """Function for computing the equal opportunity difference between the protected and unprotected group.

        Equal opportunity (Hardt et al., 2016) requires equal true positive rates (TPR) across groups, where
        TPR = P(Yhat = positive | Y = positive, S = group). The metric is the difference TPR(S=1) - TPR(S=0).

        Args:
             X (pd.DataFrame): the data to compute the metric on (must contain the protected attribute column)
             S (str): the name of the protected attribute column in X
            y_true (np.ndarray | pd.Series): the ground-truth labels aligned with the predictions
            preds (np.ndarray | pd.Series): the predictions from a classifier to compute the metric on
            positive_pred (int, optional): the positive class of the classifier, either 0 or 1, by default 1

        Returns:
            float: the true positive rate difference between the protected and unprotected group, or
            ``np.nan`` if either group has no positive ground-truth samples to estimate a TPR from

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> X = pd.DataFrame({'S': [0, 0, 1, 1]})
            >>> y_true = pd.Series([1, 1, 1, 1])
            >>> preds = np.array([1, 0, 1, 1])
            >>> EqualOpportunity.equal_opportunity(X, 'S', y_true, preds)
            0.5
        """
        assert len(X) == len(
            preds
        )  # Check that the length of the data and the predictions are the same
        assert len(X) == len(y_true)  # Check that the labels align with the data
        assert S in X.columns  # Check that the protected attribute is in the data
        assert positive_pred in [
            0,
            1,
        ]  # Check that the positive prediction is either 0 or 1

        if len(np.unique(preds)) > 2:
            warn(
                "The predictions are not binary. Running the metric on the positive class."
            )

        sensitive = np.asarray(X[S])
        y_true = np.asarray(y_true)
        preds = np.asarray(preds)

        positive_outcome = preds == positive_pred
        actual_positive = y_true == positive_pred

        def _tpr(group_value: int) -> float:
            mask = actual_positive & (sensitive == group_value)
            if mask.sum() == 0:
                return np.nan
            return float(positive_outcome[mask].mean())

        return _tpr(1) - _tpr(0)

    def evaluate(
        self, positive_class: int = 1, folds: int = 5, full_output: bool = False
    ) -> float | dict:
        """Function for evaluating the equal opportunity of a classifier trained on the synthetic data.

        Args:
            positive_class (int): The positive class of the classifier. Default is 1
            folds (int, optional): Number of folds to use in the cross-validation, by default 5
            full_output (bool, optional): Whether to return full output

        Returns:
            dict: Dictionary with the equal opportunity difference and its standard error
        """
        try:
            assert self.analysis_target is not None, (
                "SynthEval(equal opportunity): metric did not run, no analysis target variable object specified!"
            )
            assert self.analysis_target.protected_vars, (
                "SynthEval(equal opportunity): metric did not run, no protected variable specified!"
            )

            target_vars = [
                key
                for (key, value) in self.analysis_target.target_types.items()
                if isinstance(value, int) and value >= 2
            ]

            assert target_vars != [], (
                "SynthEval(equal opportunity): metric did not run, no categorical target variables with at least 2 unique values!"
            )

            protected_attributes = [
                var
                for var in self.analysis_target.protected_vars
                if self.real_data[var].nunique() == 2
            ]
            assert protected_attributes != [], (
                "SynthEval(equal opportunity): metric did not run, no protected variables with exactly 2 unique values!"
            )

            assert positive_class in [0, 1], (
                "SynthEval(equal opportunity): metric did not run, the positive class argument must be either 0 or 1"
            )
        except AssertionError as e:
            raise ValueError(str(e))

        self.full_output = full_output
        result_rows = []
        multiclass_rows = []
        for target_var, protected_attribute in product(
            target_vars, protected_attributes
        ):
            # Drop confounder variables for the current target variable (if any)
            confounders = self.analysis_target.confounder_vars[target_var]
            synt_data = self.synt_data.drop(confounders, axis=1)

            fake_x, fake_y = synt_data.drop([target_var], axis=1), synt_data[target_var]

            if self.analysis_target.target_types[target_var] > 2:
                expected_classes = list(pd.unique(self.real_data[target_var]))
                synthetic_classes = list(pd.unique(fake_y))
                missing_classes = [value for value in expected_classes if value not in synthetic_classes]
                unexpected_classes = [value for value in synthetic_classes if value not in expected_classes]
                if missing_classes or unexpected_classes:
                    raise ValueError(
                        f"SynthEval(equal opportunity): target '{target_var}' class mismatch in "
                        f"synthetic data; missing classes {missing_classes!r}, unexpected "
                        f"classes {unexpected_classes!r}."
                    )

                fold_values = {value: [] for value in expected_classes}
                for train_idxs, test_idxs in KFold(folds).split(fake_x, fake_y):
                    X_train, X_test = fake_x.iloc[train_idxs], fake_x.iloc[test_idxs]
                    y_train, y_test = fake_y.iloc[train_idxs], fake_y.iloc[test_idxs]
                    absent_train_classes = [
                        value for value in expected_classes if not (y_train == value).any()
                    ]
                    if absent_train_classes:
                        raise ValueError(
                            f"SynthEval(equal opportunity): target '{target_var}' fold training data "
                            f"is missing classes {absent_train_classes!r}."
                        )

                    clf = RandomForestClassifier(n_estimators=100)
                    clf.fit(X_train, y_train)
                    preds = np.asarray(clf.predict(X_test))
                    sensitive = np.asarray(X_test[protected_attribute])
                    actual = np.asarray(y_test)

                    for class_value in expected_classes:
                        rates = {}
                        for group_value in (0, 1):
                            mask = (actual == class_value) & (sensitive == group_value)
                            if not np.any(mask):
                                raise ValueError(
                                    f"SynthEval(equal opportunity): class {class_value!r} for "
                                    f"target '{target_var}' lacks positive-class support in "
                                    f"protected group {group_value} ('{protected_attribute}')."
                                )
                            rates[group_value] = float(np.mean(preds[mask] == class_value))
                        fold_values[class_value].append(rates[1] - rates[0])

                target_name = target_var.replace(" ", "_").lower()
                for class_value in expected_classes:
                    values = np.asarray(fold_values[class_value], dtype=float)
                    multiclass_rows.append(
                        {
                            "target_var": target_name,
                            "protected_attribute": protected_attribute,
                            "target_class": class_value,
                            "equal_opportunity": float(np.mean(values)),
                            "equal_opportunity_se": (
                                float(np.std(values, ddof=1) / np.sqrt(values.size))
                                if values.size > 1
                                else 0.0
                            ),
                        }
                    )
                continue

            # Train a classifier for each fold
            differences = []
            for train_idxs, test_idxs in KFold(folds).split(fake_x, fake_y):
                X_train, X_test = fake_x.iloc[train_idxs], fake_x.iloc[test_idxs]
                y_train, y_test = fake_y.iloc[train_idxs], fake_y.iloc[test_idxs]

                # Train a classifier
                clf = RandomForestClassifier(n_estimators=100)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                differences.append(
                    self.equal_opportunity(
                        X_test, protected_attribute, y_test, preds, positive_class
                    )
                )

            differences = np.asarray(differences, dtype=float)
            valid = differences[~np.isnan(differences)]
            if valid.size == 0:
                warn(
                    f"SynthEval(equal opportunity): no fold yielded an estimable TPR for "
                    f"'{target_var}' x '{protected_attribute}' (no positive samples in a group)."
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
                    "equal_opportunity": mean_difference,
                    "equal_opportunity_se": se_difference,
                }
            )

        columns = [
            "target_var",
            "protected_attribute",
            "equal_opportunity",
            "equal_opportunity_se",
        ]

        if result_rows:
            row_values = [row["equal_opportunity"] for row in result_rows]
            row_errors = [row["equal_opportunity_se"] for row in result_rows]
            self.results["equal_opportunity"] = float(np.nanmean(row_values))
            self.results["equal_opportunity_se"] = float(
                np.sqrt(np.nansum([err**2 for err in row_errors])) / len(result_rows)
            )
        if multiclass_rows:
            self.results["equal_opportunity_macro_ovr_v1"] = float(
                np.mean([row["equal_opportunity"] for row in multiclass_rows])
            )
            self.results["equal_opportunity_macro_ovr_v1_se"] = float(
                np.sqrt(sum(row["equal_opportunity_se"] ** 2 for row in multiclass_rows))
                / len(multiclass_rows)
            )
        self.results["raw results"] = pd.DataFrame.from_records(
            result_rows + multiclass_rows, columns=columns + ["target_class"]
        )
        return self.results

    def format_output(self) -> list:
        """Return a list of tuples for printing results to the rich console."""
        rows = []
        if "equal_opportunity" in self.results:
            rows.append((
                "fairness",
                "Equal Opportunity difference",
                self.results["equal_opportunity"],
                self.results["equal_opportunity_se"],
            ))
        if "equal_opportunity_macro_ovr_v1" in self.results:
            rows.append((
                "fairness",
                "Equal Opportunity macro OvR difference",
                self.results["equal_opportunity_macro_ovr_v1"],
                self.results["equal_opportunity_macro_ovr_v1_se"],
            ))
        return rows

    def normalize_output(self) -> list:
        """This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            output = []
            if "equal_opportunity" in self.results:
                output.append(
                    {
                        "metric": "equal_opportunity",
                        "dim": "f",
                        "val": self.results["equal_opportunity"],
                        "err": self.results["equal_opportunity_se"],
                        "n_val": 1 - abs(self.results["equal_opportunity"]),
                        "n_err": self.results["equal_opportunity_se"],
                    }
                )
            if "equal_opportunity_macro_ovr_v1" in self.results:
                value = self.results["equal_opportunity_macro_ovr_v1"]
                error = self.results["equal_opportunity_macro_ovr_v1_se"]
                output.append(
                    {
                        "metric": "equal_opportunity_macro_ovr_v1",
                        "dim": "f",
                        "val": value,
                        "err": error,
                        "n_val": 1 - abs(value),
                        "n_err": error,
                    }
                )
            if self.full_output:
                for _, row in self.results["raw results"].iterrows():
                    if pd.notna(row.get("target_class")):
                        class_name = str(row["target_class"]).replace(" ", "_").lower()
                        output.append(
                            {
                                "metric": f"eo_ovr_v1_{row['target_var']}_{class_name}_{row['protected_attribute']}",
                                "dim": "f",
                                "val": row["equal_opportunity"],
                                "err": row["equal_opportunity_se"],
                                "n_val": 1 - abs(row["equal_opportunity"]),
                                "n_err": row["equal_opportunity_se"],
                            }
                        )
                        continue
                    output.append(
                        {
                            "metric": "eo_"
                            + row["target_var"]
                            + "_"
                            + row["protected_attribute"],
                            "dim": "f",
                            "val": row["equal_opportunity"],
                            "err": row["equal_opportunity_se"],
                            "n_val": 1 - abs(row["equal_opportunity"]),
                            "n_err": row["equal_opportunity_se"],
                        }
                    )
            return output
        else:
            pass
