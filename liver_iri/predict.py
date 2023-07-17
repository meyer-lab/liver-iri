from cmtf_pls.cmtf import ctPLS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression, \
    LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_predict, LeaveOneOut, \
    RepeatedStratifiedKFold, StratifiedKFold
from sklearn.svm import SVC
import xarray as xr

OPTIMAL_TPLS = 2
rskf = RepeatedStratifiedKFold(
    n_repeats=5,
    n_splits=10
)
skf = StratifiedKFold(
    n_splits=10
)


def run_coupled_tpls_classification(data, labels, rank=OPTIMAL_TPLS,
                                    drop_missing=True, return_proba=False):
    """
    Fits coupled tPLS model to provided data and labels.

    Parameters:
        data (xr.Dataset): dataset to evaluate
        labels (pd.Series): labels to regress data against
        rank (int, default:OPTIMAL_TPLS): number of components to use in tPLS
        drop_missing (bool, default:True): drop patients missing ALL
            measurements in ANY dimension
        return_proba(bool, default:False): returns probability of each patient's
            classification

    Returns:
        models (tuple[tPLS, LR classifier]): tuple of trained tPLS and LR model
        acc (float): accuracy achieved over cross-validation
        proba (pd.Series, only returns if return_proba): probability of positive
            classification for each patient
    """
    shared_patients = sorted(list(set(data.Patient.values) & set(labels.index)))
    data = data.sel(Patient=shared_patients)
    labels = labels.loc[shared_patients]

    if drop_missing:
        tensors = [data[var].to_numpy() for var in data.data_vars]
        patients_all = np.array(
            [np.isfinite(tensor).any(axis=1).any(axis=1) for tensor in tensors]
        ).all(axis=0)
        data = data.sel(Patient=data.Patient.values[patients_all])
        labels = labels.loc[data.Patient.values]
        tensors = [
            data[var].sel(Patient=labels.index).to_numpy() for
            var in data.data_vars
        ]
    else:
        tensors = [
            data[var].sel(Patient=labels.index).to_numpy() for
            var in data.data_vars
        ]
        for index, tensor in enumerate(tensors):
            all_missing = np.isnan(tensor).all(axis=1).all(axis=1)
            tensors[index][all_missing, :, :] = 0

    np.random.seed(42)
    tpls = ctPLS(n_components=rank)

    predicted = pd.Series(0, index=labels.index)
    if return_proba:
        probabilities = predicted.copy()

    lr_model = LogisticRegression()
    for train_index, test_index in skf.split(tensors[0], labels):
        train_data = [
            tensor[train_index, :, :] for tensor in tensors
        ]
        test_data = [
            tensor[test_index, :, :] for tensor in tensors
        ]
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

    acc = accuracy_score(
        labels,
        predicted
    )
    tpls.fit(tensors, labels.values)
    lr_model.fit(
        tpls.transform(tensors),
        labels
    )

    if return_proba:
        return (tpls, lr_model), acc, probabilities
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

    labels = labels[labels != 'Unknown']

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
    model = ElasticNet(
        alpha=model.alpha_,
        l1_ratio=model.l1_ratio_
    )
    predicted = cross_val_predict(
        model,
        data,
        labels,
        cv=loo
    )
    numerator = sum(predicted ** 2)
    denominator = sum(labels ** 2)
    q2y = 1 - numerator / denominator

    return q2y, model


def predict_categorical(data, labels, return_coef=False):
    """
    Fits Logistic Regression model and hyperparameters to provided data.

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
    np.random.seed(21517)

    if isinstance(labels, pd.Series):
        labels = labels.reset_index(drop=True)
    else:
        labels = pd.Series(labels)

    labels = labels[labels != 'Unknown']

    if isinstance(data, pd.Series):
        data = data.iloc[labels.index]
        data = data.values.reshape(-1, 1)
    elif isinstance(data, pd.DataFrame):
        data = data.iloc[labels.index, :]
    else:
        data = data[labels.index, :]

    model = LogisticRegressionCV(
        l1_ratios=np.linspace(0, 1, 6),
        Cs=10,
        solver="saga",
        penalty="elasticnet",
        n_jobs=-1,
        cv=rskf,
        max_iter=100000,
        scoring='accuracy',
        multi_class='ovr'
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

    labels = labels[labels != 'Unknown']

    if isinstance(data, pd.Series):
        data = data.iloc[labels.index]
        data = data.values.reshape(-1, 1)
    elif isinstance(data, pd.DataFrame):
        data = data.iloc[labels.index, :].values
    else:
        data = data[labels.index, :]

    probabilities = np.zeros(data.shape[0])
    predicted = probabilities.copy()
    for train_index, test_index in skf.split(data, labels):
        model.fit(data[train_index, :], labels[train_index])
        probabilities[test_index] = model.predict_proba(
            data[test_index, :]
        )[:, 1]
        predicted[test_index] = model.predict(data[test_index, :])

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
    grid = np.zeros(
        shape=[len(params[key])]
    )
    for index, param in enumerate(params[key]):
        model = classifier(**{key: param})
        scores = []
        for train_index, test_index in rskf.split(data, labels):
            model.fit(
                data.iloc[train_index, :],
                labels.iloc[train_index]
            )
            predicted = model.predict(data.iloc[test_index, :])
            scores.append(
                balanced_accuracy_score(
                    labels.iloc[test_index],
                    predicted
                )
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

    labels = labels[labels != 'Unknown']

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
        {'n_estimators': np.arange(50, 160, 10)}
    )
    model = RandomForestClassifier(**best_params)
    scores = np.zeros(shape=(rskf.cvargs['n_splits'] * rskf.n_repeats))
    coefs = np.zeros(
        shape=(
            rskf.cvargs['n_splits'] * rskf.n_repeats,
            data.shape[1]
        )
    )
    i = 0
    for train_index, test_index in rskf.split(data, labels):
        model.fit(
            data.iloc[train_index, :],
            labels.iloc[train_index]
        )
        predicted = model.predict(data.iloc[test_index, :])
        scores[i] = \
            balanced_accuracy_score(
                labels.iloc[test_index],
                predicted
            )

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

    labels = labels[labels != 'Unknown']

    if isinstance(data, pd.Series):
        data = data.iloc[labels.index]
        data = data.values.reshape(-1, 1)
    elif isinstance(data, pd.DataFrame):
        data = data.iloc[labels.index, :]
    else:
        data = data[labels.index, :]

    best_params = optimize_parameters(
        SVC,
        data,
        labels,
        {'C': np.logspace(-4, 4, 9)}
    )
    model = SVC(**best_params)
    scores = np.zeros(shape=(rskf.cvargs['n_splits'] * rskf.n_repeats))
    i = 0
    for train_index, test_index in rskf.split(data, labels):
        model.fit(
            data.iloc[train_index, :],
            labels.iloc[train_index]
        )
        predicted = model.predict(data.iloc[test_index, :])
        scores[i] = \
            balanced_accuracy_score(
                labels.iloc[test_index],
                predicted
            )

        i += 1

    score = np.mean(scores)
    return score, model
