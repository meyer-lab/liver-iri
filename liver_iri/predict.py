from cmtf_pls.cmtf import ctPLS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression, \
    LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, \
    precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict, LeaveOneOut, \
    RepeatedStratifiedKFold, StratifiedKFold
from sklearn.svm import SVC
from tensorpack.tpls import calcR2X, tPLS
import xarray as xr

OPTIMAL_TPLS = 4
skf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5
)
skf = StratifiedKFold(
    n_splits=5
)


def run_coupled_tpls_classification(data, labels, rank=OPTIMAL_TPLS):
    tensors = [
        data['Cytokine Measurements'].to_numpy(),
        data['RNA Measurements'].to_numpy()
    ]
    patients_both = np.array(
        [
            np.isfinite(tensors[0]).all(axis=1).all(axis=1),
            np.isfinite(tensors[1]).all(axis=1).all(axis=1)
        ]
    ).all(axis=0)
    data = data.sel(Patient=data.Patient.values[patients_both])
    labels = labels.loc[data.Patient.values]

    tensors = [
        data['Cytokine Measurements'].to_numpy(),
        data['RNA Measurements'].to_numpy()
    ]

    np.random.seed(42)
    tpls = ctPLS(n_components=rank)

    loo = LeaveOneOut()
    predicted = np.zeros(labels.shape)
    for train_index, test_index in loo.split(labels):
        train_data = [
            tensors[0][train_index, :, :],
            tensors[1][train_index, :, :]
        ]
        test_data = [
            tensors[0][test_index, :, :],
            tensors[1][test_index, :, :]
        ]
        train_labels = labels.iloc[train_index, :].values

        tpls.fit(train_data, train_labels)
        predicted[test_index, :] = tpls.predict(test_data)

    acc = balanced_accuracy_score(
        labels.values.argmax(axis=1),
        predicted.argmax(axis=1)
    )
    tpls.fit(tensors, labels.values)

    return tpls, acc, data


def run_tpls_classification(data, labels, rank=OPTIMAL_TPLS):
    """
    Runs CP_PLSR on provided data and labels.

    Parameters:
        data:
        labels:
        rank:

    Returns:
    """
    labels.index = labels.index.astype(str)
    shared = list(set(data.Patient.values) & set(labels.index.astype(str)))
    data = data.sel(Patient=list(shared))
    labels = labels.loc[shared, :]

    if isinstance(data, xr.Dataset):
        data = data.to_array().squeeze()
        data = data.to_numpy()
    elif isinstance(data, xr.DataArray):
        data = data.to_numpy()
    elif not isinstance(data, np.ndarray):
        raise TypeError('Unrecognized data format provided')

    missing = ~np.isnan(data).any(axis=1).any(axis=1)
    data = data[missing, :, :]
    labels = labels.loc[missing, :]

    missing_labels = np.isfinite(labels).all(axis=1).values
    data = data[missing_labels, :, :]
    labels = labels.loc[missing_labels, :].values

    np.random.seed(42)
    tpls = tPLS(
        n_components=rank
    )

    predicted = np.zeros(labels.shape)
    for train_index, test_index in skf.split(data, labels.argmax(axis=1)):
        train_data, train_labels = data[train_index, :, :], labels[train_index, :]
        test_data, test_labels = data[test_index, :, :], labels[test_index, :]

        tpls.fit(train_data, train_labels)
        predicted[test_index, :] = tpls.predict(test_data)

    acc = balanced_accuracy_score(
        labels.argmax(axis=1),
        predicted.argmax(axis=1)
    )

    return tpls, acc


def run_coupled_tpls(data, labels, rank=OPTIMAL_TPLS):
    """
    Runs coupled tPLS.
    Args:
        data:
        labels:
        rank:

    Returns:
    """
    tensors = [
        data['Cytokine Measurements'].to_numpy(),
        data['RNA Measurements'].to_numpy()
    ]
    patients_both = np.array(
        [
            np.isfinite(tensors[0]).all(axis=1).all(axis=1),
            np.isfinite(tensors[1]).all(axis=1).all(axis=1)
        ]
    ).all(axis=0)
    tensors[0] = tensors[0][patients_both, :, :]
    tensors[1] = tensors[1][patients_both, :, :]
    labels = labels.loc[patients_both, :]

    missing_labels = ~labels.isna().any(axis=1)
    labels = labels.loc[missing_labels, :]
    tensors[0] = tensors[0][missing_labels, :, :]
    tensors[1] = tensors[1][missing_labels, :, :]

    np.random.seed(42)
    tpls = ctPLS(n_components=rank)

    loo = LeaveOneOut()
    predicted = np.zeros(labels.shape)
    for train_index, test_index in loo.split(labels):
        train_data = [
            tensors[0][train_index, :, :],
            tensors[1][train_index, :, :]
        ]
        test_data = [
            tensors[0][test_index, :, :],
            tensors[1][test_index, :, :]
        ]
        train_labels = labels.iloc[train_index, :].values

        tpls.fit(train_data, train_labels)
        predicted[test_index, :] = tpls.predict(test_data)

    q2y = calcR2X(np.exp(labels), np.exp(predicted))
    return tpls, q2y


def run_tpls(data, labels, rank=OPTIMAL_TPLS):
    """
    Runs CP_PLSR on provided data and labels.

    Parameters:
        data:
        labels:
        rank:

    Returns:
    """
    labels.index = labels.index.astype(str)
    shared = list(set(data.Patient.values) & set(labels.index.astype(str)))
    data = data.sel(Patient=list(shared))
    labels = labels.loc[shared, :]

    if isinstance(data, xr.Dataset):
        data = data.to_array().squeeze()
        data = data.to_numpy()
    elif isinstance(data, xr.DataArray):
        data = data.to_numpy()
    elif not isinstance(data, np.ndarray):
        raise TypeError('Unrecognized data format provided')

    missing = ~np.isnan(data).any(axis=1).any(axis=1)
    data = data[missing, :, :]
    labels = labels.loc[missing, :]

    missing_labels = np.isfinite(labels).all(axis=1).values
    # missing_labels = ~labels.isna().any(axis=1)
    data = data[missing_labels, :, :]
    labels = labels.loc[missing_labels, :].values

    np.random.seed(42)
    # tpls = CP_PLSR(
    #     n_components=rank,
    #     random_state=42
    # )
    tpls = tPLS(
        n_components=rank
    )

    loo = LeaveOneOut()
    predicted = np.zeros(labels.shape)
    for train_index, test_index in loo.split(labels):
        train_data, train_labels = data[train_index, :, :], labels[train_index, :]
        test_data, test_labels = data[test_index, :, :], labels[test_index, :]

        tpls.fit(train_data, train_labels)
        predicted[test_index, :] = tpls.predict(test_data)

    # q2y = mean_absolute_error(labels, predicted)
    q2y = calcR2X(np.exp(labels), np.exp(predicted))

    return tpls, q2y


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
    for train_index, test_index in skf.split(data, labels):
        model.fit(data[train_index, :], labels[train_index])
        probabilities[test_index] = model.predict_proba(
            data[test_index, :]
        )[:, 1]

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
