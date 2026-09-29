# Description: Implementation of auroc metric and plot of roc curves
# Author: Anton D. Lautrup
# Date: 07-11-2023

import numpy as np
import pandas as pd
from numbers import Integral

from syntheval.metrics.core.metric import MetricClass
from syntheval.utils.plot_metrics import plot_roc_curves

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.utils import resample


def _binary_support(values):
    series = pd.Series(values)
    return tuple(value for value in pd.unique(series.dropna()))


def _validate_binary_support(target_var, real_data, synt_data, hout_data):
    real_support = _binary_support(real_data[target_var])
    synt_support = _binary_support(synt_data[target_var])
    hout_support = _binary_support(hout_data[target_var])
    if len(real_support) != 2:
        raise ValueError(
            f"SynthEval(auroc): target {target_var!r} needs two real-training classes; "
            f"observed {real_support!r}."
        )
    if len(synt_support) != 2 or set(synt_support) != set(real_support):
        raise ValueError(
            f"SynthEval(auroc): target {target_var!r} has incompatible synthetic class "
            f"support {synt_support!r}; expected {real_support!r}."
        )
    if len(hout_support) != 2 or set(hout_support) != set(real_support):
        raise ValueError(
            f"SynthEval(auroc): target {target_var!r} has incompatible holdout class "
            f"support {hout_support!r}; expected {real_support!r}."
        )
    return real_support


def _validate_multiclass_support(target_var, real_data, synt_data, hout_data):
    real_support = _binary_support(real_data[target_var])
    synt_support = _binary_support(synt_data[target_var])
    hout_support = _binary_support(hout_data[target_var])
    if len(real_support) < 3:
        raise ValueError(
            f"SynthEval(auroc): target {target_var!r} needs at least three real-training classes; "
            f"observed {real_support!r}."
        )
    if len(synt_support) != len(real_support) or set(synt_support) != set(real_support):
        raise ValueError(
            f"SynthEval(auroc): target {target_var!r} has incompatible synthetic class "
            f"support {synt_support!r}; expected {real_support!r}."
        )
    if len(hout_support) != len(real_support) or set(hout_support) != set(real_support):
        raise ValueError(
            f"SynthEval(auroc): target {target_var!r} has incompatible holdout class "
            f"support {hout_support!r}; expected {real_support!r}."
        )
    return real_support


def _sem(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _sem_or_none(values):
    """Return SEM when estimable, otherwise retain missing uncertainty as None."""
    return None if len(values) < 2 else _sem(values)


def _new_auc_classifier(model):
    if model == 'rf_cls':
        return RandomForestClassifier(random_state=42)
    if model == 'log_reg':
        return LogisticRegression(random_state=42, max_iter=100)
    raise ValueError(f"Unrecognised AUROC model {model!r}.")


def _multiclass_ovr_result(
    target_var, support, model, num_boots, real_x, real_y, fake_x, fake_y, hout_x, hout_y
):
    per_class = {label: {'real_auc': [], 'synthetic_auc': [], 'differences': []} for label in support}
    bootstrap_differences = []
    for bootstrap in range(num_boots):
        if num_boots != 1:
            real_x_sub, real_y_sub = resample(
                real_x, real_y, n_samples=len(real_x), stratify=real_y,
                random_state=bootstrap,
            )
            fake_x_sub, fake_y_sub = resample(
                fake_x, fake_y, n_samples=len(fake_x), stratify=fake_y,
                random_state=bootstrap,
            )
        else:
            real_x_sub, real_y_sub = real_x, real_y
            fake_x_sub, fake_y_sub = fake_x, fake_y

        real_model = _new_auc_classifier(model)
        synthetic_model = _new_auc_classifier(model)
        real_model.fit(real_x_sub, real_y_sub)
        synthetic_model.fit(fake_x_sub, fake_y_sub)
        bootstrap_class_differences = []
        for label in support:
            if label not in real_model.classes_ or label not in synthetic_model.classes_:
                raise ValueError(
                    f"SynthEval(auroc): target {target_var!r} class {label!r} is absent "
                    f"from bootstrap {bootstrap} training support."
                )
            real_index = int(np.flatnonzero(real_model.classes_ == label)[0])
            synthetic_index = int(np.flatnonzero(synthetic_model.classes_ == label)[0])
            real_auc = float(
                roc_auc_score(
                    (hout_y == label).astype(int),
                    real_model.predict_proba(hout_x)[:, real_index],
                )
            )
            synthetic_auc = float(
                roc_auc_score(
                    (hout_y == label).astype(int),
                    synthetic_model.predict_proba(hout_x)[:, synthetic_index],
                )
            )
            difference = synthetic_auc - real_auc
            per_class[label]['real_auc'].append(real_auc)
            per_class[label]['synthetic_auc'].append(synthetic_auc)
            per_class[label]['differences'].append(difference)
            bootstrap_class_differences.append(difference)
        bootstrap_differences.append(float(np.mean(bootstrap_class_differences)))

    class_results = []
    for class_index, label in enumerate(support):
        values = per_class[label]
        difference = float(np.mean(values['differences']))
        class_label = label.item() if isinstance(label, np.generic) else label
        class_results.append({
            'class_index': class_index,
            'class_label': class_label,
            'real_auc': float(np.mean(values['real_auc'])),
            'synthetic_auc': float(np.mean(values['synthetic_auc'])),
            'difference': difference,
            'difference_err': _sem_or_none(values['differences']),
            'bootstrap_real_auc': tuple(values['real_auc']),
            'bootstrap_synthetic_auc': tuple(values['synthetic_auc']),
            'bootstrap_differences': tuple(values['differences']),
        })

    macro_difference = float(np.mean(bootstrap_differences))
    return {
        'target_var': target_var,
        'aggregation': 'macro_one_vs_rest',
        'metric_version': 'macro_ovr_v3',
        'classes': tuple(class_results),
        'bootstrap_differences': tuple(bootstrap_differences),
        'difference': macro_difference,
        'difference_err': _sem_or_none(bootstrap_differences),
        'agreement': auroc_agreement_v2(macro_difference),
    }


def auroc_agreement_v2(signed_difference):
    """Return the bounded agreement score for a signed synthetic-minus-real AUC difference."""
    difference = float(signed_difference)
    if not np.isfinite(difference):
        return np.nan
    return float(np.clip(1.0 - abs(difference), 0.0, 1.0))


class PredictionAUROCDifference(MetricClass):
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

    self.verbose   : bool (mainly for supressing prints and plots)

    """

    def name() -> str:
        """ Name/keyword to reference the metric"""
        return 'auroc_diff'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, model = 'log_reg', num_boots = 1, full_output: bool = False) -> float | dict:
        """ Metric that calculates the AUROC difference between a Random Forest model trained on 
        real data and one trained on fake data. Also plots the ROC curves if verbose
        
        Args:
            model (str): 'log_reg' or 'rf_cls'
            num_boots (int): Number of bootstraps runs of the model
            full_output (bool): whether to return the full results dictionary or just the auroc_diff value
        
        Returns:
            dict: AUROC difference between the two models

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'label': [0, 1, 0]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'label': [0, 1, 0]})
            >>> hout = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'label': [0, 1, 0]})
            >>> AUROC = PredictionAUROCDifference(real, fake, hout, analysis_target='label',
            ...     verbose=False, do_preprocessing=False, plot_figures=False)
            >>> AUROC.evaluate(model='log_reg', num_boots=1) # doctest: +ELLIPSIS
            {'model': 'log_reg', 'auroc results': ..., 'auroc_diff': 0.0, ...}
        """
        try:
            assert self.analysis_target is not None, "SynthEval(auroc): metric did not run, no analysis target variable(s) supplied!"

            target_vars = [
                key for (key, value) in self.analysis_target.target_types.items()
                if isinstance(value, Integral) and value >= 2
            ]
            
            assert target_vars != [], "SynthEval(auroc): metric did not run, no categorical target variables with at least 2 unique values!"
            assert self.hout_data is not None, "SynthEval(auroc): metric did not run, no holdout data supplied!"
            assert model in ['rf_cls', 'log_reg'], "SynthEval(auroc): metric did not run, unrecognised model name supplied! Use 'rf_cls' or 'log_reg'."
            if not isinstance(num_boots, Integral) or num_boots < 1:
                raise ValueError("SynthEval(auroc): num_boots must be a positive integer.")
        except AssertionError as e:
            raise AssertionError(e)
        else:
            self.full_output = full_output
            result_rows = []
            multiclass_results = {}
            for target_var in target_vars:
                class_count = self.analysis_target.target_types[target_var]
                if class_count == 2:
                    support = _validate_binary_support(
                        target_var, self.real_data, self.synt_data, self.hout_data
                    )
                else:
                    support = _validate_multiclass_support(
                        target_var, self.real_data, self.synt_data, self.hout_data
                    )
                # Drop confounder variables for the current target variable (if any)
                confounders = self.analysis_target.confounder_vars[target_var]
                real_data = self.real_data.drop(confounders, axis=1)
                synt_data = self.synt_data.drop(confounders, axis=1)
                hout_data = self.hout_data.drop(confounders, axis=1)

                real_x, real_y = real_data.drop([target_var], axis=1), real_data[target_var]
                fake_x, fake_y = synt_data.drop([target_var], axis=1), synt_data[target_var]
                hout_x, hout_y = hout_data.drop([target_var], axis=1), hout_data[target_var]
                target_name = target_var.replace(' ', '_').lower()
                if len(support) > 2:
                    multiclass_result = _multiclass_ovr_result(
                        target_var, support, model, num_boots,
                        real_x, real_y, fake_x, fake_y, hout_x, hout_y,
                    )
                    multiclass_results[target_name] = multiclass_result
                    result_rows.append({
                        'target_var': target_name,
                        'model': model,
                        'auroc_diff': multiclass_result['difference'],
                        'auroc_diff_v2': multiclass_result['difference'],
                        'auroc_diff_err_v2': multiclass_result['difference_err'],
                        'auroc_agreement_v2': multiclass_result['agreement'],
                    })
                    continue

                hout_y_binary = (hout_y == support[1]).astype(int)

                match model:
                    case 'rf_cls':
                        model1 = RandomForestClassifier(random_state=42)
                        model2 = RandomForestClassifier(random_state=42)
                    case 'log_reg':
                        model1 = LogisticRegression(random_state=42, max_iter=100)
                        model2 = LogisticRegression(random_state=42, max_iter=100)

                roc_curves_real = []
                roc_curves_fake = []
                v2_diffs = []
                for i in range(num_boots):
                    if num_boots != 1:
                        real_x_sub, real_y_sub = resample(
                            real_x, real_y, n_samples=len(real_x), stratify=real_y, random_state=i
                        )
                        fake_x_sub, fake_y_sub = resample(
                            fake_x, fake_y, n_samples=len(fake_x), stratify=fake_y, random_state=i
                        )
                    else:
                        real_x_sub, real_y_sub = real_x, real_y
                        fake_x_sub, fake_y_sub = fake_x, fake_y
                
                    model1.fit(real_x_sub, real_y_sub)
                    model2.fit(fake_x_sub, fake_y_sub)
                    positive_index_real = int(np.flatnonzero(model1.classes_ == support[1])[0])
                    positive_index_fake = int(np.flatnonzero(model2.classes_ == support[1])[0])
                    y1_probs = model1.predict_proba(hout_x)[:, positive_index_real]
                    y2_probs = model2.predict_proba(hout_x)[:, positive_index_fake]
                    real_auc_v2 = roc_auc_score(hout_y_binary, y1_probs)
                    synt_auc_v2 = roc_auc_score(hout_y_binary, y2_probs)
                    v2_diffs.append(float(synt_auc_v2 - real_auc_v2))
                
                    # Calculate ROC curve for the subsampled model
                    fpr1, tpr1, _ = roc_curve(hout_y, y1_probs, pos_label=support[1])
                    fpr2, tpr2, _ = roc_curve(hout_y, y2_probs, pos_label=support[1])

                    roc_curves_real.append((fpr1, tpr1))
                    roc_curves_fake.append((fpr2, tpr2))
            
                mean_fpr = np.linspace(0, 1, len(fpr1))

                tprs_real, tprs_fake = [], []

                for fpr, tpr in roc_curves_real:
                    tprs_real.append(np.interp(mean_fpr, fpr, tpr))

                mean_tpr_real = np.mean(tprs_real, axis=0)
                std_tpr_real = np.std(tprs_real, axis=0)

                for fpr, tpr in roc_curves_fake:
                    tprs_fake.append(np.interp(mean_fpr, fpr, tpr))

                mean_tpr_fake = np.mean(tprs_fake, axis=0)
                std_tpr_fake = np.std(tprs_fake, axis=0)

                # Calculate AUROC for the mean ROC curve
                roc_auc_mean_real = auc(mean_fpr, mean_tpr_real)
                roc_auc_mean_fake = auc(mean_fpr, mean_tpr_fake)

                if self.plot_figures: plot_roc_curves([mean_fpr, mean_tpr_real, roc_auc_mean_real], 
                                                [mean_fpr, mean_tpr_real, std_tpr_real], 
                                                [mean_fpr, mean_tpr_fake, roc_auc_mean_fake],
                                                [mean_fpr, mean_tpr_fake, std_tpr_fake],
                                                f"{model}, predicting {target_name}", 'roc_curves_'+target_name)
                
                result_rows.append({
                    'target_var': target_name,
                    'model': model,
                    'auroc_diff': float(roc_auc_mean_fake - roc_auc_mean_real),
                    'auroc_diff_v2': float(np.mean(v2_diffs)),
                    'auroc_diff_err_v2': _sem(v2_diffs),
                    'auroc_agreement_v2': auroc_agreement_v2(np.mean(v2_diffs)),
                })

            self.results['model'] = model

            columns = [
                'target_var', 'model', 'auroc_diff', 'auroc_diff_v2',
                'auroc_diff_err_v2', 'auroc_agreement_v2',
            ]
            self.results['auroc results'] = pd.DataFrame.from_records(result_rows, columns=columns)

            self.results['auroc_diff'] = float(self.results['auroc results']['auroc_diff'].mean())
            if len(self.results['auroc results']) > 1:
                self.results['auroc_diff_err'] = float(self.results['auroc results']['auroc_diff'].sem()) 
            self.results['auroc_diff_v2'] = float(self.results['auroc results']['auroc_diff_v2'].mean())
            self.results['auroc_diff_err_v2'] = _sem(self.results['auroc results']['auroc_diff_v2'])
            self.results['auroc_agreement_v2'] = auroc_agreement_v2(self.results['auroc_diff_v2'])
            if not multiclass_results:
                self.results['auroc_version_v2'] = 'signed_synthetic_minus_real'
            else:
                self.results['auroc_class_results_v3'] = multiclass_results
                self.results['auroc_diff_macro_ovr_v3'] = float(
                    np.mean([item['difference'] for item in multiclass_results.values()])
                )
                self.results['auroc_diff_err_macro_ovr_v3'] = (
                    _sem([item['difference'] for item in multiclass_results.values()])
                    if len(multiclass_results) > 1 else None
                )
                self.results['auroc_agreement_macro_ovr_v3'] = auroc_agreement_v2(
                    self.results['auroc_diff_macro_ovr_v3']
                )
                self.results['auroc_version_v3'] = 'macro_one_vs_rest_synthetic_minus_real'
            # self.results = {'model': model, 'auroc_diff': float(roc_auc_mean_fake - roc_auc_mean_real)}
            return self.results
        
    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        row = ("prediction", f"Prediction AUROC difference ({self.results['model']:<7})", self.results['auroc_diff'], self.results.get('auroc_diff_err'))
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
            output = [{'metric': 'auroc', 'dim': 'u', 
                     'val': self.results['auroc_diff'],
                     'err': self.results.get('auroc_diff_err'),
                     'n_val': np.tanh(2*self.results['auroc_diff']+1), 
                     'n_err': self.results.get('auroc_diff_err')
                     }]
            if self.full_output and len(self.results['auroc results']) > 1:
                for index, row in self.results['auroc results'].iterrows():
                    output.append({'metric': 'auroc_'+row['target_var'], 'dim': 'u', 
                                'val': row['auroc_diff'],
                                'n_val': np.tanh(2*row['auroc_diff']+1), 
                                })
            return output
        else: pass

    def normalize_output_v2(self) -> list:
        """Return the versioned signed-difference and agreement identities."""
        if self.results == {}:
            return []
        if 'auroc_class_results_v3' in self.results:
            rows = [{
                'metric': 'auroc_macro_ovr_v3',
                'dim': 'u',
                'val': self.results['auroc_diff_macro_ovr_v3'],
                'err': self.results.get('auroc_diff_err_macro_ovr_v3'),
                'n_val': self.results['auroc_agreement_macro_ovr_v3'],
                'n_err': self.results.get('auroc_diff_err_macro_ovr_v3'),
                'metric_version': 'macro_ovr_v3',
                'raw_value': self.results['auroc_diff_macro_ovr_v3'],
                'normalized_value': self.results['auroc_agreement_macro_ovr_v3'],
                'metadata': {
                    'difference': 'synthetic_minus_real',
                    'aggregation': 'macro_one_vs_rest',
                    'class_results': self.results['auroc_class_results_v3'],
                },
            }]
            for target_name, target_result in self.results['auroc_class_results_v3'].items():
                rows.append({
                    'metric': f'auroc_{target_name}_macro_ovr_v3',
                    'dim': 'u',
                    'val': target_result['difference'],
                    'err': target_result['difference_err'],
                    'n_val': target_result['agreement'],
                    'n_err': target_result['difference_err'],
                    'metric_version': 'macro_ovr_v3',
                    'raw_value': target_result['difference'],
                    'normalized_value': target_result['agreement'],
                    'metadata': {
                        'target_var': target_name,
                        'aggregation': target_result['aggregation'],
                        'class_results': target_result['classes'],
                    },
                })
                for class_result in target_result['classes']:
                    rows.append({
                        'metric': (
                            f"auroc_{target_name}_class_{class_result['class_index']}_ovr_v3"
                        ),
                        'dim': 'u',
                        'val': class_result['difference'],
                        'err': class_result['difference_err'],
                        'n_val': auroc_agreement_v2(class_result['difference']),
                        'n_err': class_result['difference_err'],
                        'metric_version': 'macro_ovr_v3',
                        'raw_value': class_result['difference'],
                        'normalized_value': auroc_agreement_v2(class_result['difference']),
                        'metadata': {
                            'target_var': target_name,
                            'class_index': class_result['class_index'],
                            'class_label': class_result['class_label'],
                            'real_auc': class_result['real_auc'],
                            'synthetic_auc': class_result['synthetic_auc'],
                            'bootstrap_differences': class_result['bootstrap_differences'],
                        },
                    })
            binary_rows = self.results['auroc results'].loc[
                ~self.results['auroc results']['target_var'].isin(
                    self.results['auroc_class_results_v3']
                )
            ]
            if not binary_rows.empty:
                binary_differences = binary_rows['auroc_diff_v2'].to_numpy(dtype=float)
                binary_difference = float(np.mean(binary_differences))
                binary_agreement = auroc_agreement_v2(binary_difference)
                rows.append({
                    'metric': 'auroc_v2',
                    'dim': 'u',
                    'val': binary_difference,
                    'err': _sem(binary_differences),
                    'n_val': binary_agreement,
                    'n_err': _sem(binary_differences),
                    'metric_version': 'v2',
                    'raw_value': binary_difference,
                    'normalized_value': binary_agreement,
                    'metadata': {
                        'difference': 'synthetic_minus_real',
                        'agreement': '1-abs(difference)',
                    },
                })
                if self.full_output:
                    for _, row in binary_rows.iterrows():
                        rows.append({
                            'metric': f"auroc_{row['target_var']}_v2",
                            'dim': 'u',
                            'val': float(row['auroc_diff_v2']),
                            'err': row['auroc_diff_err_v2'],
                            'n_val': float(row['auroc_agreement_v2']),
                            'n_err': row['auroc_diff_err_v2'],
                            'metric_version': 'v2',
                            'raw_value': float(row['auroc_diff_v2']),
                            'normalized_value': float(row['auroc_agreement_v2']),
                            'metadata': {'target_var': row['target_var']},
                        })
            return rows
        rows = [{
            'metric': 'auroc_v2',
            'dim': 'u',
            'val': self.results['auroc_diff_v2'],
            'err': self.results.get('auroc_diff_err_v2'),
            'n_val': self.results['auroc_agreement_v2'],
            'n_err': self.results.get('auroc_diff_err_v2'),
            'metric_version': 'v2',
            'raw_value': self.results['auroc_diff_v2'],
            'normalized_value': self.results['auroc_agreement_v2'],
            'metadata': {
                'difference': 'synthetic_minus_real',
                'agreement': '1-abs(difference)',
            },
        }]
        if self.full_output:
            for _, row in self.results['auroc results'].iterrows():
                rows.append({
                    'metric': f"auroc_{row['target_var']}_v2",
                    'dim': 'u',
                    'val': float(row['auroc_diff_v2']),
                    'err': row['auroc_diff_err_v2'],
                    'n_val': float(row['auroc_agreement_v2']),
                    'n_err': row['auroc_diff_err_v2'],
                    'metric_version': 'v2',
                    'raw_value': float(row['auroc_diff_v2']),
                    'normalized_value': float(row['auroc_agreement_v2']),
                    'metadata': {'target_var': row['target_var']},
                })
        return rows
