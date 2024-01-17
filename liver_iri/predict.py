import warnings

import numpy as np
import pandas as pd
import xarray as xr
from cmtf_pls.cmtf import ctPLS
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import (ElasticNet, ElasticNetCV, LogisticRegression,
                                  LogisticRegressionCV)
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import (LeaveOneOut, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict)

warnings.filterwarnings("ignore")


OPTIMAL_TPLS = 9
rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=10)
skf = StratifiedKFold(n_splits=10)


def oversample(tensors: list[np.ndarray], labels: pd.Series):
    """
    Over-/under-samples tensor data to form balanced dataset.

    Parameters:
        tensors (list of np.ndarray): coupled tensors
        labels (pd.Series): labels for patients in tensors

    Returns:
        Over-/under-sampled tensors and labels
    """
    undersampler = RandomUnderSampler(sampling_strategy=1 / 3, random_state=42)
    undersampler.fit_resample(labels.values.reshape(-1, 1), labels)
    tensors = [tensor[undersampler.sample_indices_, :, :] for tensor in tensors]
    labels = labels.iloc[undersampler.sample_indices_]
    oversampler = RandomOverSampler(random_state=42)
    oversampler.fit_resample(labels.values.reshape(-1, 1), labels)
    tensors = [tensor[oversampler.sample_indices_, :, :] for tensor in tensors]
    labels = labels.iloc[oversampler.sample_indices_]

    return tensors, labels


def run_coupled_tpls_classification(
    tensors: list[np.ndarray],
    labels: pd.Series,
    rank: int = OPTIMAL_TPLS,
    return_proba: bool = False,
):
    """
    Fits coupled tPLS model to provided data and labels.

    Parameters:
        tensors (list of np.ndarray): coupled tensors
        labels (pd.Series): labels to regress data against
        rank (int, default:OPTIMAL_TPLS): number of components to use in tPLS
        return_proba (bool, default:False): returns probability of each
            patient's classification

    Returns:
        models (tuple[tPLS, LR classifier]): tuple of trained tPLS and LR model
        acc (float): accuracy achieved over cross-validation
        pred (pd.Series, only returns if not return_proba): predicted value for
            each patient
        proba (pd.Series, only returns if return_proba): probability of positive
            classification for each patient
    """
    np.random.seed(215)
    tpls = ctPLS(n_components=rank)
    tpls.fit(tensors, labels.values)

    predicted = pd.Series(0, index=labels.index)
    if return_proba:
        probabilities = predicted.copy()

    model = LogisticRegressionCV(
        l1_ratios=np.linspace(0, 1, 6),
        Cs=10,
        solver="saga",
        penalty="elasticnet",
        n_jobs=-1,
        cv=rskf,
        max_iter=100000,
        scoring="accuracy",
        multi_class="ovr",
    )
    model.fit(tpls.Xs_factors[0][0], labels)

    lr_model = LogisticRegression(
        C=model.C_[0],
        l1_ratio=model.l1_ratio_[0],
        solver="saga",
        penalty="elasticnet",
        max_iter=100000,
    )
    for train_index, test_index in skf.split(labels, labels):
        train_data = [tensor[train_index, :, :] for tensor in tensors]
        test_data = [tensor[test_index, :, :] for tensor in tensors]
        train_labels = labels.iloc[train_index].values
        tpls.fit(train_data, train_labels)

        train_transformed = tpls.transform(train_data)
        test_transformed = tpls.transform(test_data)
        lr_model.fit(train_transformed, train_labels)
        predicted.iloc[test_index] = lr_model.predict(test_transformed)

        if return_proba:
            probabilities.iloc[test_index] = lr_model.predict_proba(
                test_transformed
            )[:, 1]

    acc = accuracy_score(labels, predicted)
    tpls.fit(tensors, labels.values)
    lr_model.fit(tpls.transform(tensors), labels)

    if return_proba:
        return (tpls, lr_model), acc, probabilities
    else:
        return (tpls, lr_model), acc, predicted


def predict_continuous(data: xr.Dataset, labels: pd.Series):
    """
    Fits Elastic Net model and hyperparameters to provided data.

    Parameters:
        data (pandas.DataFrame): Data to predict
        labels (pandas.Series): Labels for provided data

    Returns:
        q2y (float): Accuracy for best-performing model
        model (sklearn.LogisticRegressionCV)
    """
    labels = labels.reset_index(drop=True)
    labels = labels[labels != "Unknown"]

    if isinstance(data, pd.Series):
        data = data.iloc[labels.index]
        data = data.values.reshape(-1, 1)
    elif isinstance(data, pd.DataFrame):
        data = data.iloc[labels.index, :]
    else:
        data = data[labels.index, :]

    model = ElasticNetCV()
    model.fit(data, labels)

    loo = LeaveOneOut()
    model = ElasticNet(alpha=model.alpha_, l1_ratio=model.l1_ratio_)
    predicted = cross_val_predict(model, data, labels, cv=loo)
    numerator = sum(predicted**2)
    denominator = sum(labels**2)
    q2y = 1 - numerator / denominator

    return q2y, model


def predict_categorical(
    data: pd.DataFrame,
    labels: pd.Series,
    return_coef: bool = False,
    return_pred: bool = False,
    balanced_resample: bool = True,
):
    """
    Fits Logistic Regression model and hyperparameters to provided data.

    Parameters:
        data (pandas.DataFrame): Data to predict
        labels (pandas.Series): Labels for provided data
        return_coef (bool, default: False): Return model coefficients
        return_pred (bool, default: False): Return predictions
        balanced_resample (bool, default:False): under-/over-samples data to
            form balanced dataset

    Returns:
        score (float): Accuracy for best-performing model
        model (sklearn.LogisticRegression): Fitted model with hyperparameters
            optimized to predict provided data and labels
        return_coef (numpy.array): LR coefficients for each feature
    """
    np.random.seed(21517)

    if isinstance(labels, pd.Series):
        labels = labels.reset_index(drop=True)
    else:
        labels = pd.Series(labels)

    labels = labels[labels != "Unknown"]

    if isinstance(data, pd.Series):
        data = data.iloc[labels.index]
        data = data.values.reshape(-1, 1)
    elif isinstance(data, pd.DataFrame):
        data = data.iloc[labels.index, :]
    else:
        data = data[labels.index, :]

    if balanced_resample:
        undersampler = RandomUnderSampler(
            sampling_strategy=0.25, random_state=42
        )
        data, labels = undersampler.fit_resample(data, labels)
        oversampler = RandomOverSampler(random_state=42)
        data, labels = oversampler.fit_resample(data, labels)

    model = LogisticRegressionCV(
        l1_ratios=np.linspace(0, 1, 6),
        Cs=10,
        solver="saga",
        penalty="elasticnet",
        n_jobs=-1,
        cv=rskf,
        max_iter=100000,
        scoring="accuracy",
        multi_class="ovr",
    )
    model.fit(data, labels)
    coef = model.coef_[0]
    scores = np.mean(list(model.scores_.values())[0], axis=0)

    model = LogisticRegression(
        C=model.C_[0],
        l1_ratio=model.l1_ratio_[0],
        solver="saga",
        penalty="elasticnet",
        max_iter=100000,
    )
    model.fit(data, labels)

    if return_coef:
        return np.max(scores), model, coef
    elif return_pred:
        return np.max(scores), model, model.predict(data), labels
    else:
        return np.max(scores), model


def get_probabilities(model, data, labels):
    """
    Returns probabilities of positive classification via cross-validation.

    Parameters:
        model (sklearn.BaseEstimator): sklearn model; must have predict_proba()
        data (pandas.DataFrame): Data to predict
        labels (pandas.Series): Labels for provided data

    Returns:
        pd.Series: probability of positive class for each sample
    """
    np.random.seed(215)

    if isinstance(labels, pd.Series):
        labels = labels.reset_index(drop=True)
    else:
        labels = pd.Series(labels)

    labels = labels[labels != "Unknown"]

    if isinstance(data, pd.Series):
        data = data.iloc[labels.index]
        data = data.values.reshape(-1, 1)
    elif isinstance(data, pd.DataFrame):
        data = data.iloc[labels.index, :].values
    else:
        data = data[labels.index, :]

    probabilities = np.zeros(data.shape[0])
    for train_index, test_index in skf.split(data, labels):
        model.fit(data[train_index, :], labels[train_index])
        probabilities[test_index] = model.predict_proba(data[test_index, :])[
            :, 1
        ]

    return probabilities
