import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression, \
    LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict, LeaveOneOut, \
    RepeatedStratifiedKFold
from sklearn.svm import SVC

skf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5
)


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
        cv=skf,
        max_iter=100000,
        scoring='balanced_accuracy',
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
        for train_index, test_index in skf.split(data, labels):
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
    scores = np.zeros(shape=(skf.cvargs['n_splits'] * skf.n_repeats))
    coefs = np.zeros(
        shape=(
            skf.cvargs['n_splits'] * skf.n_repeats,
            data.shape[1]
        )
    )
    i = 0
    for train_index, test_index in skf.split(data, labels):
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
    scores = np.zeros(shape=(skf.cvargs['n_splits'] * skf.n_repeats))
    i = 0
    for train_index, test_index in skf.split(data, labels):
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
