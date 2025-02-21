import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from cmtf_pls.cmtf import ctPLS
from imblearn.over_sampling import RandomOverSampler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")

OPTIMAL_TPLS = 2
kf = KFold(n_splits=10)
skf = StratifiedKFold(n_splits=20)


def oversample(
    tensors: list[np.ndarray],
    labels: pd.Series,
    column: str | None = None,
):
    """
    Over-/under-samples tensor data to form balanced dataset.

    Parameters:
        tensors (list of np.ndarray): coupled tensors
        labels (pd.Series): labels for patients in tensors
        column (str): column to balance

    Returns:
        Over-/under-sampled tensors and labels
    """
    oversampler = RandomOverSampler(random_state=42)
    if column is None:
        oversampler.fit_resample(labels.to_numpy().reshape(-1, 1), labels)
    else:
        oversampler.fit_resample(
            labels.loc[:, column].values.reshape(-1, 1), labels.loc[:, column]
        )

    tensors = [tensor[oversampler.sample_indices_, :, :] for tensor in tensors]
    labels = labels.iloc[oversampler.sample_indices_]

    return tensors, labels


def run_coupled_tpls_classification(
    tensors: list[np.ndarray],
    labels: pd.Series,
    rank: int = OPTIMAL_TPLS,
    return_proba: bool = False,
    return_components: bool = False,
):
    """
    Fits coupled tPLS model to provided data and labels.

    Parameters:
        tensors (list of np.ndarray): coupled tensors
        labels (pd.Series): labels to regress data against
        rank (int, default:OPTIMAL_TPLS): number of components to use in tPLS
        return_proba (bool, default:False): returns probability of each
            patient's classification
        return_components (bool, default:False): returns patient component
            factors

    Returns:
        models (tuple[tPLS, LR classifier]): tuple of trained tPLS and LR model
        acc (float): accuracy achieved over cross-validation
        pred (pd.Series, only returns if not return_proba): predicted value for
            each patient
        proba (pd.Series, only returns if return_proba): probability of positive
            classification for each patient
    """
    if return_proba and return_components:
        return_components = False

    np.random.seed(215)
    tpls = ctPLS(n_components=rank)
    tpls.fit(tensors, labels.values)

    predicted = pd.Series(0, index=labels.index)
    components = pd.DataFrame(
        0, index=labels.index, columns=np.arange(rank) + 1
    )

    model = LogisticRegressionCV(
        l1_ratios=np.linspace(0, 1, 11),
        Cs=11,
        solver="saga",
        penalty="elasticnet",
        n_jobs=-1,
        cv=skf,
        max_iter=100000,
        scoring="balanced_accuracy",
        multi_class="ovr",
    )
    model.fit(tpls.Xs_factors[0][0], labels.values)
    model = LogisticRegression(
        C=model.C_[0],
        l1_ratio=model.l1_ratio_[0],
        solver="saga",
        penalty="elasticnet",
        max_iter=100000,
    )
    for train_index, test_index in skf.split(labels, labels):
        train_data = [tensor[train_index, :, :] for tensor in tensors]
        test_data = [tensor[test_index, :, :] for tensor in tensors]
        train_labels = labels.iloc[train_index]

        train_data, train_labels = oversample(train_data, train_labels)
        tpls.fit(train_data, train_labels.values)

        train_transformed = tpls.transform(train_data)
        test_transformed = tpls.transform(test_data)
        model.fit(train_transformed, train_labels)

        if return_proba:
            predicted.iloc[test_index] = model.predict_proba(test_transformed)[
                :, 1
            ]
        elif return_components:
            components.iloc[test_index, :] = test_transformed
        else:
            predicted.iloc[test_index] = model.predict(test_transformed)

    if return_proba:
        acc = accuracy_score(labels, predicted.round().astype(int))
    else:
        acc = accuracy_score(labels, predicted)

    tpls.fit(tensors, labels.values)
    model.fit(tpls.transform(tensors), labels)

    if return_proba:
        return (tpls, model), acc, predicted
    elif return_components:
        return (tpls, model), acc, components
    else:
        return (tpls, model), acc, predicted


def run_tpls_survival(
    tensors: list[np.ndarray], labels: pd.DataFrame, rank: int = OPTIMAL_TPLS
):
    """
    Runs survival regression via coupled tPLS.

    Parameters:
        tensors (list of np.ndarray): coupled tensors
        labels (pd.DataFrame): labels to regress data against
        rank (int, default:OPTIMAL_TPLS): number of components to use in tPLS

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
    model = CoxPHFitter(penalizer=0.05, l1_ratio=0.2)

    for train_index, test_index in skf.split(
        labels.loc[:, "graft_death"], labels.loc[:, "graft_death"]
    ):
        train_data = [tensor[train_index, :, :] for tensor in tensors]
        test_data = [tensor[test_index, :, :] for tensor in tensors]
        train_labels = labels.iloc[train_index, :]

        train_data, train_labels = oversample(
            train_data, train_labels, column="graft_death"
        )
        tpls.fit(train_data, train_labels.values)

        train_transformed = tpls.transform(train_data)
        test_transformed = tpls.transform(test_data)

        train_transformed = pd.DataFrame(
            train_transformed,
            index=train_labels.index,
            columns=np.arange(tpls.n_components) + 1,
        )
        train_transformed = pd.concat([train_transformed, train_labels], axis=1)
        model.fit(
            train_transformed,
            duration_col="survival_time",
            event_col="graft_death",
        )
        predicted.iloc[test_index] = model.predict_expectation(test_transformed)

    c_index = concordance_index(
        labels.loc[:, "survival_time"], predicted, labels.loc[:, "graft_death"]
    )

    return (tpls, model), c_index, predicted


def run_survival(data: pd.DataFrame, labels: pd.DataFrame):
    """
    Runs survival regression.

    Parameters:
        data (pd.DataFrame): data to regress
        labels (pd.DataFrame): labels to regress data against
    """
    np.random.seed(215)
    model = CoxPHFitter(penalizer=0.05, l1_ratio=0.2)
    oversampler = RandomOverSampler(random_state=42)

    data = data.dropna(axis=0)
    labels = labels.loc[data.index, :]
    predicted = pd.Series(0, index=labels.index)

    for train_index, test_index in skf.split(
        labels.loc[:, "graft_death"], labels.loc[:, "graft_death"]
    ):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        train_labels = labels.iloc[train_index]

        if isinstance(train_data, pd.Series):
            train_data = train_data.to_frame()
            test_data = test_data.to_frame()

        oversampler.fit_resample(train_data, train_labels.loc[:, "graft_death"])
        train_data = train_data.iloc[oversampler.sample_indices_]
        train_labels = train_labels.iloc[oversampler.sample_indices_]
        train_data = pd.concat([train_data, train_labels], axis=1)
        model.fit(
            train_data, duration_col="survival_time", event_col="graft_death"
        )
        predicted.iloc[test_index] = model.predict_expectation(test_data)

    c_index = concordance_index(
        labels.loc[:, "survival_time"], predicted, labels.loc[:, "graft_death"]
    )

    return model, c_index, predicted


def predict_continuous(data: pd.DataFrame | pd.Series, labels: pd.Series):
    """
    Fits Elastic Net model and hyperparameters to provided data.

    Parameters:
        data (xr.Dataset): Data to predict
        labels (pandas.Series): Labels for provided data

    Returns:
        q2y (float): Accuracy for best-performing model
        model (sklearn.LogisticRegressionCV)
    """
    labels = labels.reset_index(drop=True)
    labels = labels.loc[labels != "Unknown"]

    if isinstance(data, pd.DataFrame):
        sm_data = data.iloc[labels.index, :]
    else:
        data = data.iloc[labels.index]
        sm_data = data.to_numpy().reshape(-1, 1)

    sm_data = sm.add_constant(sm_data)
    model = sm.OLS(labels, sm_data, missing="drop")
    results = model.fit()

    return results.rsquared_adj, results


def predict_categorical(
    data: pd.DataFrame,
    labels: pd.Series,
    return_proba: bool = False,
    balanced_resample: bool = True,
):
    """
    Fits Logistic Regression model and hyperparameters to provided data.

    Parameters:
        data (pandas.DataFrame): Data to predict
        labels (pandas.Series): Labels for provided data
        return_proba (bool, default: False): Return predictions
        balanced_resample (bool, default:False): under-/over-samples data to
            form balanced dataset

    Returns:
        score (float): Accuracy for best-performing model
        model (sklearn.LogisticRegression): Fitted model with hyperparameters
            optimized to predict provided data and labels
        return_coef (numpy.array): LR coefficients for each feature
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()

    model = LogisticRegressionCV(
        l1_ratios=np.linspace(0, 1, 6),
        Cs=10,
        solver="saga",
        penalty="elasticnet",
        n_jobs=-1,
        cv=skf,
        max_iter=100000,
        scoring="balanced_accuracy",
        multi_class="ovr",
    )
    if data.shape[1] < 2:
        model.fit(data.values.reshape(-1, 1), labels)
    else:
        model.fit(data, labels)

    oversampler = RandomOverSampler(random_state=42)
    model = LogisticRegression(
        C=model.C_[0],
        l1_ratio=model.l1_ratio_[0],
        solver="saga",
        penalty="elasticnet",
        max_iter=100000,
    )
    predicted = pd.Series(index=labels.index)

    for train_index, test_index in skf.split(data, labels):
        train_data = data.iloc[train_index, :]
        train_labels = labels.iloc[train_index]
        test_data = data.iloc[test_index, :]

        if balanced_resample:
            train_data, train_labels = oversampler.fit_resample(  # type: ignore # noqa
                train_data, train_labels
            )

        if data.shape[1] < 2:
            model.fit(train_data.to_numpy().reshape(-1, 1), train_labels)
        else:
            model.fit(train_data, train_labels)

        if return_proba:
            predicted.iloc[test_index] = model.predict_proba(test_data)[:, 1]
        else:
            predicted.iloc[test_index] = model.predict(test_data)

    if return_proba:
        acc = balanced_accuracy_score(labels, predicted.round())
    else:
        acc = balanced_accuracy_score(labels, predicted)

    model.fit(data, labels)

    return acc, model, predicted


def predict_clinical(
    data: pd.Series,
    labels: pd.Series,
    balanced_resample: bool = True,
    return_proba: bool = False,
):
    """
    Fits Logistic Regression model and hyperparameters to provided data.

    Parameters:
        data (pandas.Series): Data to predict
        labels (pandas.Series): Labels for provided data
        balanced_resample (bool, default:False): under-/over-samples data to
            form balanced dataset

    Returns:
        score (float): Prediction accuracy
    """
    oversampler = RandomOverSampler(random_state=42)
    model = LogisticRegression()
    predicted = pd.Series(index=labels.index)
    for train_index, test_index in skf.split(data, labels):
        train_data = data.iloc[train_index]
        train_labels = labels.iloc[train_index]
        test_data = data.iloc[test_index]

        if balanced_resample:
            train_data, train_labels = oversampler.fit_resample(  # type: ignore # noqa
                train_data.to_numpy().reshape(-1, 1), train_labels
            )

        model.fit(train_data, train_labels)

        if return_proba:
            predicted.iloc[test_index] = model.predict_proba(
                test_data.values.reshape(-1, 1)
            )[:, 1]
        else:
            predicted.iloc[test_index] = model.predict(
                test_data.values.reshape(-1, 1)
            )

    if return_proba:
        return predicted
    else:
        acc = balanced_accuracy_score(labels, predicted)
        return acc
