import warnings

import numpy as np
import pandas as pd
import xarray as xr
from cmtf_pls.cmtf import ctPLS
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (ElasticNet, ElasticNetCV, LogisticRegression,
                                  LogisticRegressionCV)
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import (LeaveOneOut, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict)
from sklearn.svm import SVC

from liver_iri.tensor import run_coupled

warnings.filterwarnings("ignore")


OPTIMAL_TPLS = 10
rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=10)
skf = StratifiedKFold(n_splits=10)


def cp_impute(data, labels, rank=2):
    """Imputes missing values via coupled CP decomposition."""
    _, cp = run_coupled(data, rank)
    reconstructed = cp.reconstruct()
    tensors = []
    for var in data.data_vars:
        tensor = data[var].sel(Patient=labels.index).to_numpy()
        imputed = reconstructed[var].sel(Patient=labels.index).to_numpy()
        mask = np.isnan(tensor)
        tensor[mask] = imputed[mask]
        tensors.append(tensor)

    return tensors


def run_coupled_tpls_classification(
    data,
    labels,
    rank=OPTIMAL_TPLS,
    impute_method=None,
    return_proba=False,
    return_pred=False,
    oversample=True,
):
    """
    Fits coupled tPLS model to provided data and labels.

    Parameters:
        data (xr.Dataset): dataset to evaluate
        labels (pd.Series): labels to regress data against
        rank (int, default:OPTIMAL_TPLS): number of components to use in tPLS
        impute_method (str, default:None): specify how to handle missing values;
            must be one of 'cp', 'drop', 'zero', or None
        return_proba(bool, default:False): returns probability of each patient's
            classification

    Returns:
        models (tuple[tPLS, LR classifier]): tuple of trained tPLS and LR model
        acc (float): accuracy achieved over cross-validation
        proba (pd.Series, only returns if return_proba): probability of positive
            classification for each patient
    """
    if impute_method not in ["cp", "drop", "zero", None]:
        raise ValueError('impute_method must be one of "cp", "drop", or "zero"')

    shared_patients = sorted(list(set(data.Patient.values) & set(labels.index)))
    data = data.sel(Patient=shared_patients)
    labels = labels.loc[shared_patients]

    if impute_method == "drop":
        tensors = [data[var].to_numpy() for var in data.data_vars]
        patients_all = np.array(
            [np.isfinite(tensor).any(axis=1).any(axis=1) for tensor in tensors]
        ).all(axis=0)
        data = data.sel(Patient=data.Patient.values[patients_all])
        labels = labels.loc[data.Patient.values]
        tensors = [
            data[var].sel(Patient=labels.index).to_numpy()
            for var in data.data_vars
        ]
    elif impute_method == "zero":
        tensors = [
            data[var].sel(Patient=labels.index).to_numpy()
            for var in data.data_vars
        ]
        for index, tensor in enumerate(tensors):
            all_missing = np.isnan(tensor).all(axis=1).all(axis=1)
            tensors[index][all_missing, :, :] = 0
    elif impute_method == "cp":
        tensors = cp_impute(data, labels)
    else:
        tensors = [
            data[var].sel(Patient=labels.index).to_numpy()
            for var in data.data_vars
        ]

    if oversample:
        undersampler = RandomUnderSampler(
            sampling_strategy=1 / 3, random_state=42
        )
        undersampler.fit_resample(labels.values.reshape(-1, 1), labels)
        tensors = [
            tensor[undersampler.sample_indices_, :, :] for tensor in tensors
        ]
        labels = labels.iloc[undersampler.sample_indices_]
        oversampler = RandomOverSampler(random_state=42)
        oversampler.fit_resample(labels.values.reshape(-1, 1), labels)
        tensors = [
            tensor[oversampler.sample_indices_, :, :] for tensor in tensors
        ]
        labels = labels.iloc[oversampler.sample_indices_]

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
        if impute_method == "cp":
            train_data = data.sel(Patient=labels.iloc[train_index].index)
            test_data = data.sel(Patient=labels.iloc[test_index].index)

            train_data = cp_impute(train_data, labels.iloc[train_index])
            test_data = cp_impute(test_data, labels.iloc[test_index])
        else:
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

    acc = balanced_accuracy_score(labels, predicted)
    tpls.fit(tensors, labels.values)
    lr_model.fit(tpls.transform(tensors), labels)

    if return_proba:
        return (tpls, lr_model), acc, probabilities
    elif return_pred:
        return (tpls, lr_model), predicted, labels
    else:
        return (tpls, lr_model), acc, data


def predict_continuous(data, labels):
    """
    Fits Elastic Net model and hyperparameters to provided data.

    Parameters:
        data (pandas.DataFrame): Data to predict
        labels (pandas.Series): Labels for provided data

    Returns:
        q2y (float): Accuracy for best-performing model
        model (sklearn.LogisticRegressionCV)
    """
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
    data, labels, return_coef=False, return_pred=False, oversample=True
):
    """
    Fits Logistic Regression model and hyperparameters to provided data.

    Parameters:
        data (pandas.DataFrame): Data to predict
        labels (pandas.Series): Labels for provided data
        return_coef (bool, default: False): Return model coefficients
        return_pred (bool, default: False): Return predictions
        oversample (bool, default: True): Over/under sample dataset for class
            balance

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

    if oversample:
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


def optimize_parameters(classifier, data, labels, params):
    """
    Optimizes parameters for provided classifier.

    Parameters:
        classifier (sklearn.estimator): sklearn classifier
        data (pandas.DataFrame): Data to predict
        labels (pandas.Series): Labels for provided data
        params (dict[str:Iterable]): Maps parameter name to values to test

    Returns:
        params (dict[str:Iterable]): Parameters that lead to highest accuracy
            via cross-validation
    """
    key = list(params.keys())[0]
    grid = np.zeros(shape=[len(params[key])])
    for index, param in enumerate(params[key]):
        model = classifier(**{key: param})
        scores = []
        for train_index, test_index in rskf.split(data, labels):
            model.fit(data.iloc[train_index, :], labels.iloc[train_index])
            predicted = model.predict(data.iloc[test_index, :])
            scores.append(
                balanced_accuracy_score(labels.iloc[test_index], predicted)
            )
        grid[index] = np.mean(scores)

    return {
        key: params[key][int(np.argmax(grid))],
    }


def predict_categorical_rf(data, labels, return_coef=False):
    """
    Fits Random Forest classifier model and hyperparameters to provided data.

    Parameters:
        data (pandas.DataFrame): Data to predict
        labels (pandas.Series): Labels for provided data
        return_coef (bool, default: False): Return model coefficients

    Returns:
        score (float): Accuracy for best-performing model
        model (sklearn.LogisticRegression): Fitted model with hyperparameters
            optimized to predict provided data and labels
        return_coef (numpy.array): LR coefficients for each feature
    """
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

    best_params = optimize_parameters(
        RandomForestClassifier,
        data,
        labels,
        {"n_estimators": np.arange(50, 160, 10)},
    )
    model = RandomForestClassifier(**best_params)
    scores = np.zeros(shape=(rskf.cvargs["n_splits"] * rskf.n_repeats))
    coefs = np.zeros(
        shape=(rskf.cvargs["n_splits"] * rskf.n_repeats, data.shape[1])
    )
    i = 0
    for train_index, test_index in rskf.split(data, labels):
        model.fit(data.iloc[train_index, :], labels.iloc[train_index])
        predicted = model.predict(data.iloc[test_index, :])
        scores[i] = balanced_accuracy_score(labels.iloc[test_index], predicted)

        coefs[i, :] = model.feature_importances_
        i += 1

    score = np.mean(scores)
    coefs = np.mean(coefs, axis=0)

    if return_coef:
        return score, model, coefs
    else:
        return score, model


def predict_categorical_svc(data, labels):
    """
    Fits SVC model and hyperparameters to provided data.

    Parameters:
        data (pandas.DataFrame): Data to predict
        labels (pandas.Series): Labels for provided data

    Returns:
        score (float): Accuracy for best-performing model
        model (sklearn.LogisticRegression): Fitted model with hyperparameters
            optimized to predict provided data and labels
        return_coef (numpy.array): LR coefficients for each feature
    """
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

    best_params = optimize_parameters(
        SVC, data, labels, {"C": np.logspace(-4, 4, 9)}
    )
    model = SVC(**best_params)
    scores = np.zeros(shape=(rskf.cvargs["n_splits"] * rskf.n_repeats))
    i = 0
    for train_index, test_index in rskf.split(data, labels):
        model.fit(data.iloc[train_index, :], labels.iloc[train_index])
        predicted = model.predict(data.iloc[test_index, :])
        scores[i] = balanced_accuracy_score(labels.iloc[test_index], predicted)

        i += 1

    score = np.mean(scores)
    return score, model
