# Description: Metric implementation of the classification accuracy difference.
# Author: Anton D. Lautrup
# Date: 05-03-2023

import copy
from numbers import Integral

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from typing import Literal
from syntheval.metrics.core.metric import MetricClass

# Below this many folds, joblib's per-task process-pool dispatch overhead
# (via the 'loky' backend) outweighs the benefit -- keep small/doctest-sized
# inputs (e.g. k_folds=2) sequential.
_PARALLEL_MIN_FOLDS = 3

model_name_dict = {
    'dt': 'DecisionTreeClassifier',
    'svm': 'SupportVectorMachine',
    'rf': 'RandomForestClassifier', 
    'adaboost': 'AdaBoostClassifier', 
    'logreg': 'LogisticRegression'
    }

def _get_model(model_name: str) -> object:
    """Function for returning a classification model based on the input string"""
    match model_name:
        case 'dt':
            return DecisionTreeClassifier(max_depth=15, random_state=42)
        case 'rf':
            return RandomForestClassifier(n_estimators=10, max_depth=15, random_state=42)
        case 'svm':
            return SVC(kernel='rbf', random_state=42)
        case 'adaboost':
            return AdaBoostClassifier(n_estimators=10, learning_rate=1, random_state=42)
        case 'logreg':
            return LogisticRegression(solver='saga', max_iter=5000, random_state=42)
        case _:
            raise ValueError(f"SynthEval(cls_acc): Model {model_name} not currently implemented!")


def _propagated_err(err_values: pd.Series) -> float:
    """Propagate independent errors for a mean estimate."""
    values = np.asarray(err_values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.nansum(values ** 2)) / values.size)


def _series_sem(values: pd.Series) -> float:
    """Return SEM using sample std (ddof=1), mirroring pandas semantics."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return np.nan
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))


def _classification_score(y_true, y_pred, score_type, labels=None):
    if score_type == 'balanced_accuracy':
        return float(balanced_accuracy_score(y_true, y_pred))
    return float(f1_score(
        y_true,
        y_pred,
        average=score_type,
        labels=labels,
        zero_division=0,
    ))


def _v2_score_type(score_type):
    if score_type == 'balanced_accuracy':
        return 'balanced_accuracy'
    return 'macro'


def _class_support(values):
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError("SynthEval(cls_acc): target values cannot contain missing labels.")
    counts = series.value_counts(dropna=False)
    return tuple(counts.index.tolist()), counts.to_dict()


def _validate_target_support(target_var, real_y, fake_y, hout_y, k_folds):
    real_classes, real_counts = _class_support(real_y)
    fake_classes, fake_counts = _class_support(fake_y)
    if len(real_classes) < 2:
        raise ValueError(
            f"SynthEval(cls_acc): target {target_var!r} needs at least two real classes; "
            f"observed {real_classes!r}."
        )
    if set(fake_classes) != set(real_classes):
        raise ValueError(
            f"SynthEval(cls_acc): target {target_var!r} synthetic classes {fake_classes!r} "
            f"do not match real classes {real_classes!r}."
        )
    for role, counts in (("real", real_counts), ("synthetic", fake_counts)):
        insufficient = {
            label: int(count)
            for label, count in counts.items()
            if count < k_folds
        }
        if insufficient:
            raise ValueError(
                f"SynthEval(cls_acc): target {target_var!r} has insufficient {role} class "
                f"support for {k_folds} folds: {insufficient!r}."
            )
    if hout_y is not None:
        holdout_classes, _ = _class_support(hout_y)
        if set(holdout_classes) != set(real_classes):
            raise ValueError(
                f"SynthEval(cls_acc): target {target_var!r} holdout classes "
                f"{holdout_classes!r} do not match real classes {real_classes!r}."
            )
    return tuple(real_classes)


def class_test(real_models, fake_models, real, fake, test, F1_type, labels=None):
    """Function for running a training session and getting predictions 
    on the SciPy model provided, and data.
    
    Args:
        real_models (list): List of SciPy models
        fake_models (list): List of SciPy models
        real (list): List of real data
        fake (list): List of synthetic data
        test (list): List of test data
        F1_type (str): Type of F1 score to compute

    Returns:
        np.array: F1 scores for real and fake data
    
    Example:
        >>> import numpy as np
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> real_models = [DecisionTreeClassifier(), DecisionTreeClassifier()]
        >>> fake_models = [DecisionTreeClassifier(), DecisionTreeClassifier()]
        >>> real = [np.array([[1, 2], [3, 4]]) , np.array([0, 1])]
        >>> fake = [np.array([[1, 2], [3, 4]]) , np.array([0, 1])]
        >>> test = [np.array([[1, 2], [3, 4]]) , np.array([0, 1])]

        >>> class_test(real_models, fake_models, real, fake, test, 'weighted') # doctest: +ELLIPSIS
        array([[1., 1.],...])
    """
    res = []
    for r_mod_, f_mod_ in zip(real_models, fake_models):
        r_mod, f_mod = copy.copy(r_mod_), copy.copy(f_mod_)
        r_mod.fit(real[0],real[1])
        f_mod.fit(fake[0],fake[1])

        pred_real = r_mod.predict(test[0])
        pred_fake = f_mod.predict(test[0])

        f1_real = _classification_score(test[1], pred_real, F1_type, labels=labels)
        f1_fake = _classification_score(test[1], pred_fake, F1_type, labels=labels)

        res.append([f1_real, f1_fake])
    return np.array(res).T

def _evaluate_one_fold(
    train_index_real, test_index_real, train_index_fake,
    real_x_sub, real_y_sub, fake_x_sub, fake_y_sub,
    real_models, fake_models, F1_type, v2_score_type, labels,
):
    """Run class_test for a single CV fold. Standalone module-level function
    (rather than inlined in the loop) so it can be dispatched via joblib's
    'loky' (process) backend -- each fold's train/test split and model
    fit/predict is fully independent of every other fold.
    """
    real_x_train, real_y_train = real_x_sub.iloc[train_index_real], real_y_sub.iloc[train_index_real]
    real_x_test, real_y_test = real_x_sub.iloc[test_index_real], real_y_sub.iloc[test_index_real]
    fake_x_train, fake_y_train = fake_x_sub.iloc[train_index_fake], fake_y_sub.iloc[train_index_fake]
    legacy_scores = class_test(
        real_models,
        fake_models,
        [real_x_train, real_y_train],
        [fake_x_train, fake_y_train],
        [real_x_test, real_y_test],
        F1_type,
        labels=labels,
    )
    v2_scores = class_test(
        real_models,
        fake_models,
        [real_x_train, real_y_train],
        [fake_x_train, fake_y_train],
        [real_x_test, real_y_test],
        v2_score_type,
        labels=labels,
    )
    return np.concatenate([legacy_scores, v2_scores], axis=0)

class ClassificationAccuracy(MetricClass):
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
        return 'cls_acc'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, cls_models = ['rf', 'adaboost', 'svm', 'logreg'],
                 F1_type: Literal['micro', 'macro', 'weighted', 'balanced_accuracy'] = 'macro',
                 k_folds: int = 5,
                 full_output: bool = False,
                 ) -> dict:

        """ Function for evaluating the metric

        Args:
            cls_models (list): List of classification models to use
                - 'dt' : Decision Tree Classifier
                - 'rf' : Random Forest Classifier
                - 'adaboost' : AdaBoost Classifier
                - 'svm' : Support Vector Machine Classifier
                - 'logreg' : Logistic Regression Classifier
            F1_type (str): Classification score to use
                - 'micro' : Calculate metrics globally by counting the total true positives, false negatives and false positives.
                - 'macro' : Calculate metrics for each label, and find their unweighted mean. I.e, emphasize the importance of rare labels.
                - 'weighted' : Calculate metrics for each label, and find their average weighted by support.
                - 'balanced_accuracy' : Calculate the mean recall over classes.
            k_folds (int): Number of folds to use in cross-validation

        Returns:
            dict: result variables for the metric

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3, 2], 'b': [4, 5, 6, 4], 'label': [1, 0, 1, 0]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3, 1], 'b': [4, 5, 6, 1], 'label': [1, 0, 1, 0]})
            >>> cls_acc = ClassificationAccuracy(real, fake, cat_cols=['label'], analysis_target='label', do_preprocessing=False)
            >>> cls_acc.evaluate(cls_models = ['rf'], k_folds=2) # doctest: +ELLIPSIS
            {'train results': ...
        """
        if self.analysis_target is None:
            raise AssertionError("SynthEval(cls_acc): Analysis target variable(s) not set!")
        if F1_type not in {'micro', 'macro', 'weighted', 'balanced_accuracy'}:
            raise ValueError(
                "SynthEval(cls_acc): F1_type must be 'macro', 'balanced_accuracy', "
                "'weighted', or legacy 'micro'."
            )
        if not isinstance(k_folds, Integral) or k_folds < 2:
            raise ValueError("SynthEval(cls_acc): k_folds must be an integer of at least 2.")
        
        target_vars = [
            key for (key, value) in self.analysis_target.target_types.items() 
            if isinstance(value, Integral) and value >= 2
            ]

        if target_vars == []:
            raise AssertionError("SynthEval(cls_acc): No categorical target variables with 2 or more unique values!")
        if cls_models == []:
            raise AssertionError("SynthEval(cls_acc): No classification models provided!")

        train_rows, test_rows = [], []

        self.k_folds = k_folds
        self.models = cls_models
        self.full_output = full_output
        self.legacy_score_type = F1_type
        self.score_type_v2 = _v2_score_type(F1_type)

        for target_var in target_vars:
            # Drop confounder variables for the current target variable (if any)
            confounders = self.analysis_target.confounder_vars[target_var]
            real_data = self.real_data.drop(confounders, axis=1)
            synt_data = self.synt_data.drop(confounders, axis=1)

            real_x, real_y = real_data.drop([target_var], axis=1), real_data[target_var]
            fake_x, fake_y = synt_data.drop([target_var], axis=1), synt_data[target_var]

            if self.hout_data is not None:
                hout_data = self.hout_data.drop(confounders, axis=1)

                hout_x, hout_y = hout_data.drop([target_var], axis=1), hout_data[target_var]
            else:
                hout_x, hout_y = None, None

            labels = _validate_target_support(
                target_var, real_y, fake_y, hout_y, k_folds
            )
            v2_score_type = self.score_type_v2

            real_models = [_get_model(model_name) for model_name in cls_models]
            fake_models = [_get_model(model_name) for model_name in cls_models]
            target_var = target_var.replace(' ', '_').lower()

            real_kf = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
            fake_kf = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
            splits = list(zip(real_kf.split(real_x, real_y), fake_kf.split(fake_x, fake_y)))
            fold_args = [
                (train_index_real, test_index_real, train_index_fake,
                 real_x, real_y, fake_x, fake_y,
                 real_models, fake_models, F1_type, v2_score_type, labels)
                for (train_index_real, test_index_real), (train_index_fake, _) in splits
            ]

            if len(fold_args) >= _PARALLEL_MIN_FOLDS:
                from joblib import Parallel, delayed
                res = Parallel(n_jobs=-2, backend='loky')(
                    delayed(_evaluate_one_fold)(*args) for args in fold_args
                )
            else:
                res = [
                    _evaluate_one_fold(*args)
                    for args in tqdm(fold_args, desc='cls_acc', disable=not self.verbose)
                ]

            fold_scores = np.asarray(res, dtype=float)
            legacy_fold_scores = fold_scores[:, :2, :]
            v2_fold_scores = fold_scores[:, 2:, :]
            class_avg = np.mean(legacy_fold_scores, axis=0)
            class_err = np.std(legacy_fold_scores, axis=0, ddof=1) / np.sqrt(k_folds)
            paired_differences = legacy_fold_scores[:, 1, :] - legacy_fold_scores[:, 0, :]
            class_diff = np.mean(paired_differences, axis=0)
            class_diff_err = np.std(paired_differences, axis=0, ddof=1) / np.sqrt(k_folds)
            v2_class_avg = np.mean(v2_fold_scores, axis=0)
            v2_class_err = np.std(v2_fold_scores, axis=0, ddof=1) / np.sqrt(k_folds)
            v2_paired_differences = v2_fold_scores[:, 1, :] - v2_fold_scores[:, 0, :]
            v2_class_diff = np.mean(v2_paired_differences, axis=0)
            v2_class_diff_err = np.std(
                v2_paired_differences, axis=0, ddof=1
            ) / np.sqrt(k_folds)

            for i, model in enumerate(cls_models):
                train_rows.append({
                    'target_var': target_var,
                    'model': model,
                    'TRTR_acc': class_avg[0, i],
                    'TRTR_err': class_err[0, i],
                    'TSTR_acc': class_avg[1, i],
                    'TSTR_err': class_err[1, i],
                    'acc_diff': class_diff[i],
                    'acc_diff_err': class_diff_err[i],
                    'score_type_v2': v2_score_type,
                    'score_trtr_v2': v2_class_avg[0, i],
                    'score_trtr_err_v2': v2_class_err[0, i],
                    'score_tstr_v2': v2_class_avg[1, i],
                    'score_tstr_err_v2': v2_class_err[1, i],
                    'score_diff_v2': v2_class_diff[i],
                    'score_diff_err_v2': v2_class_diff_err[i],
                })

            if self.hout_data is not None:
                holdout_legacy = class_test(
                    real_models,
                    fake_models,
                    [real_x, real_y],
                    [fake_x, fake_y],
                    [hout_x, hout_y],
                    F1_type,
                    labels=labels,
                )
                holdout_v2 = class_test(
                    real_models,
                    fake_models,
                    [real_x, real_y],
                    [fake_x, fake_y],
                    [hout_x, hout_y],
                    v2_score_type,
                    labels=labels,
                )
                for i, model in enumerate(cls_models):
                    test_rows.append({
                        'target_var': target_var,
                        'model': model,
                        'TRTR_acc': holdout_legacy[0, i],
                        'TSTR_acc': holdout_legacy[1, i],
                        'acc_diff': holdout_legacy[1, i] - holdout_legacy[0, i],
                        'acc_diff_err': np.nan,
                        'score_type_v2': v2_score_type,
                        'score_trtr_v2': holdout_v2[0, i],
                        'score_tstr_v2': holdout_v2[1, i],
                        'score_diff_v2': holdout_v2[1, i] - holdout_v2[0, i],
                        'score_diff_err_v2': np.nan,
                    })

        train_cols = [
            'target_var', 'model', 'TRTR_acc', 'TRTR_err', 'TSTR_acc', 'TSTR_err',
            'acc_diff', 'acc_diff_err', 'score_type_v2', 'score_diff_v2',
            'score_diff_err_v2', 'score_trtr_v2', 'score_trtr_err_v2',
            'score_tstr_v2', 'score_tstr_err_v2',
        ]
        test_cols = [
            'target_var', 'model', 'TRTR_acc', 'TSTR_acc', 'acc_diff',
            'acc_diff_err', 'score_type_v2', 'score_diff_v2', 'score_diff_err_v2',
            'score_trtr_v2', 'score_tstr_v2',
        ]
        results_df_train = pd.DataFrame.from_records(train_rows, columns=train_cols)
        results_df_test = pd.DataFrame.from_records(test_rows, columns=test_cols)

        self.results['train results'] = results_df_train
        self.results['test results'] = results_df_test

        self.results['avg diff'] = float(results_df_train['acc_diff'].mean())
        self.results['avg diff err'] = _propagated_err(results_df_train['acc_diff_err'])
        self.results['score_type_v2'] = self.score_type_v2
        self.results['legacy_score_type'] = self.legacy_score_type
        self.results['avg diff v2'] = float(results_df_train['score_diff_v2'].mean())
        self.results['avg diff err v2'] = _propagated_err(results_df_train['score_diff_err_v2'])

        if len(results_df_test) > 0:
            self.results['avg diff hout'] = float(results_df_test['acc_diff'].mean())
            self.results['avg diff err hout'] = _series_sem(results_df_test['acc_diff'])
            self.results['avg diff hout v2'] = float(results_df_test['score_diff_v2'].mean())
            self.results['avg diff err hout v2'] = _series_sem(results_df_test['score_diff_v2'])
        return self.results
    
    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        if self.results !={}:
            multiple_targets_flag = len(self.results['train results']['target_var'].unique()) > 1
            rows = [('prediction', 'Accuracy Diff. (%d-fold cross val.)' % (self.k_folds), "", "")]
            train_results = self.results['train results']
            if not multiple_targets_flag:
                for model in self.models:
                    model_rows = train_results[train_results['model'] == model]
                    if len(model_rows) == 0:
                        continue

                    trtr_avg = float(model_rows['TRTR_acc'].mean())
                    tstr_avg = float(model_rows['TSTR_acc'].mean())
                    diff_avg = float(model_rows['acc_diff'].mean())
                    diff_err = _propagated_err(model_rows['acc_diff_err'])

                    rows.append((
                        'prediction',
                        f"{model_name_dict.get(model, model):<22} | RR {trtr_avg:.2f} | FR {tstr_avg:.2f}",
                        diff_avg,
                        diff_err,
                    ))

                if len(train_results) > 1:
                    rows.append((
                        'prediction',
                        f"{'Averages':<22} | RR {train_results['TRTR_acc'].mean():.2f} | FR {train_results['TSTR_acc'].mean():.2f}",
                        self.results['avg diff'],
                        self.results['avg diff err'],
                    ))
                
            else:
                target_averages = train_results.groupby('target_var').agg({
                    'TRTR_acc': 'mean',
                    'TSTR_acc': 'mean',
                    'acc_diff': 'mean'
                }).reset_index()

                for _, target_row in target_averages.iterrows():
                    target_rows = train_results[train_results['target_var'] == target_row['target_var']]
                    target_err = _propagated_err(target_rows['acc_diff_err'])

                    rows.append((
                        'prediction',
                        f"Avg. for {target_row['target_var']:<13.13} | RR {target_row['TRTR_acc']:.2f} | FR {target_row['TSTR_acc']:.2f}",
                        float(target_row['acc_diff']),
                        target_err,
                    ))

                rows.append((
                    'prediction',
                    f"{'Global averages':<22} | RR {train_results['TRTR_acc'].mean():.2f} | FR {train_results['TSTR_acc'].mean():.2f}",
                    self.results['avg diff'],
                    self.results['avg diff err'],
                ))

            if len(self.results['test results']) > 0:
                test_results = self.results['test results']
                rows.append(('prediction', 'Holdout Data Results', "", ""))

                if not multiple_targets_flag:
                    for model in self.models:
                        model_rows = test_results[test_results['model'] == model]
                        if len(model_rows) == 0:
                            continue

                        trtr_avg = float(model_rows['TRTR_acc'].mean())
                        tstr_avg = float(model_rows['TSTR_acc'].mean())
                        diff_avg = float(model_rows['acc_diff'].mean())

                        rows.append((
                            'prediction',
                            f"{model_name_dict.get(model, model):<22} | RR {trtr_avg:.2f} | FR {tstr_avg:.2f}",
                            diff_avg,
                            None,
                        ))
                    if len(test_results) > 1:
                        rows.append((
                            'prediction',
                            f"{'Averages':<22} | RR {test_results['TRTR_acc'].mean():.2f} | FR {test_results['TSTR_acc'].mean():.2f}",
                            self.results['avg diff hout'],
                            self.results['avg diff err hout']
                        ))
                else:
                    target_test_averages = test_results.groupby('target_var').agg(
                        TRTR_acc=('TRTR_acc', 'mean'),
                        TSTR_acc=('TSTR_acc', 'mean'),
                        acc_diff=('acc_diff', 'mean'),
                        acc_diff_sem=('acc_diff', 'sem'),
                    ).reset_index()

                    for _, target_row in target_test_averages.iterrows():
                        rows.append((
                            'prediction',
                            f"Avg. for {target_row['target_var']:<13.13} | RR {target_row['TRTR_acc']:.2f} | FR {target_row['TSTR_acc']:.2f}",
                            float(target_row['acc_diff']),
                            float(target_row['acc_diff_sem'])
                        ))

                    rows.append((
                        'prediction',
                        f"{'Global averages':<22} | RR {test_results['TRTR_acc'].mean():.2f} | FR {test_results['TSTR_acc'].mean():.2f}",
                        self.results['avg diff hout'],
                        self.results['avg diff err hout']
                    ))
            return rows

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).
        
        The required format is:
        metric  val  err  n_val  n_err idx_val idx_err
            name1  0.0  0.0    0.0    0.0    None    None
            name2  0.0  0.0    0.0    0.0    0.0     0.0
        """
        if self.results !={}:
            train_results = self.results['train results']
            test_results = self.results['test results']
            multiple_targets_flag = len(train_results['target_var'].unique()) > 1

            avg_diff = self.results.get('avg diff', float(train_results['acc_diff'].mean()))
            avg_diff_err = self.results.get('avg diff err', _propagated_err(train_results['acc_diff_err']))

            output = [{'metric': 'avg_F1_diff', 'dim': 'u',
                       'val': avg_diff,
                       'err': avg_diff_err,
                       'n_val': 1-abs(avg_diff),
                       'n_err': avg_diff_err,
                       }]

            target_groups_train = [
                (target_var, train_results[train_results['target_var'] == target_var])
                for target_var in train_results['target_var'].unique()
            ] if multiple_targets_flag else [(None, train_results)]

            if self.full_output:
                for target_var, target_rows in target_groups_train:
                    for model in self.models:
                        model_rows = target_rows[target_rows['model'] == model]
                        if len(model_rows) == 0:
                            continue

                        metric_prefix = f'{target_var}_{model}' if target_var is not None else model
                        syn_f1 = float(model_rows['TSTR_acc'].mean())
                        syn_f1_err = _propagated_err(model_rows['TSTR_err'])
                        model_diff = float(model_rows['acc_diff'].mean())
                        model_diff_err = _propagated_err(model_rows['acc_diff_err'])

                        output.extend([{'metric': f'{metric_prefix}_syn_F1', 'dim': 'u',
                            'val': syn_f1,
                            'err': syn_f1_err,
                            'n_val': syn_f1,
                            'n_err': syn_f1_err,
                            }])

                        output.extend([
                            {'metric': f'{metric_prefix}_F1_diff', 'dim': 'u',
                                'val': model_diff,
                                'err': model_diff_err,
                                'n_val': 1-abs(model_diff),
                                'n_err': model_diff_err,
                            }])
            if len(test_results) > 0:
                avg_diff_hout = self.results.get('avg diff hout', float(test_results['acc_diff'].mean()))
                output.extend([{'metric': 'avg_F1_diff_hout', 'dim': 'u',
                       'val': avg_diff_hout,
                       'err': self.results.get('avg diff err hout', None),
                       'n_val': 1-abs(avg_diff_hout),
                       'n_err': self.results.get('avg diff err hout', None),
                       }])

                target_groups_test = [
                    (target_var, test_results[test_results['target_var'] == target_var])
                    for target_var in test_results['target_var'].unique()
                ] if multiple_targets_flag else [(None, test_results)]

                if self.full_output:
                    for target_var, target_rows in target_groups_test:
                        for model in self.models:
                            model_rows = target_rows[target_rows['model'] == model]
                            if len(model_rows) == 0:
                                continue

                            metric_prefix = f'{target_var}_{model}' if target_var is not None else model
                            syn_f1_hout = float(model_rows['TSTR_acc'].mean())
                            model_diff_hout = float(model_rows['acc_diff'].mean())

                            output.extend([{'metric': f'{metric_prefix}_syn_F1_hout', 'dim': 'u',
                                'val': syn_f1_hout,
                                'n_val': syn_f1_hout,
                                }])

                            output.extend([
                                {'metric': f'{metric_prefix}_F1_diff_hout', 'dim': 'u',
                                    'val': model_diff_hout,
                                    'n_val': 1-abs(model_diff_hout),
                                }])
            return output
        else: pass

    def normalize_output_v2(self) -> list:
        """Return score-specific v2 classification differences and agreement."""
        if self.results == {}:
            return []
        score_name = (
            'balanced_accuracy'
            if self.results['score_type_v2'] == 'balanced_accuracy'
            else f"{self.results['score_type_v2']}_F1"
        )
        metric_name = f'avg_{score_name}_diff_v2'
        difference = self.results['avg diff v2']
        difference_err = self.results.get('avg diff err v2')
        rows = [{
            'metric': metric_name,
            'dim': 'u',
            'val': difference,
            'err': difference_err,
            'n_val': float(np.clip(1.0 - abs(difference), 0.0, 1.0)),
            'n_err': difference_err,
            'metric_version': 'v2',
            'raw_value': difference,
            'normalized_value': float(np.clip(1.0 - abs(difference), 0.0, 1.0)),
            'metadata': {
                'score': self.results['score_type_v2'],
                'uncertainty': 'paired_fold_difference_sem',
                'primary_f1_is_not_micro': self.results['score_type_v2'] != 'micro',
            },
        }]
        if len(self.results['test results']) > 0:
            holdout_difference = self.results['avg diff hout v2']
            holdout_err = self.results.get('avg diff err hout v2')
            holdout_score = float(np.clip(1.0 - abs(holdout_difference), 0.0, 1.0))
            rows.append({
                'metric': f'{metric_name}_hout',
                'dim': 'u',
                'val': holdout_difference,
                'err': holdout_err,
                'n_val': holdout_score,
                'n_err': holdout_err,
                'metric_version': 'v2',
                'raw_value': holdout_difference,
                'normalized_value': holdout_score,
                'metadata': {
                    'score': self.results['score_type_v2'],
                    'population': 'holdout',
                },
            })
        return rows