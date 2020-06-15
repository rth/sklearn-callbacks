import warnings

from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn_callbacks import DebugCallback


def test_pipeline():
    X, y = load_iris(return_X_y=True)

    callback = DebugCallback(verbose=False)
    print("")

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3))
    pipe._set_callbacks(callback)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipe.fit(X, y)

    log_expected = [
        "fit_begin Pipeline",
        "fit_begin StandardScaler",
        "fit_begin StandardScaler",  # why second time?
        r"fit_begin LogisticRegression\(max_iter=3\)",
        "iter_end coef=.*",
        "iter_end coef=.*",
        "iter_end coef=.*",
    ]
    callback.check_log_expected(log_expected)


def test_pipeline_column_transformer():
    X, y = load_iris(return_X_y=True)

    callback = DebugCallback(verbose=False)

    pipe = make_pipeline(
        make_column_transformer(
            (StandardScaler(), [0, 1]), (MinMaxScaler(), [2, 3]),
        ),
        LogisticRegression(max_iter=3),
    )
    pipe._set_callbacks(callback)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipe.fit(X, y)

    log_expected = [
        "fit_begin Pipeline",
        "fit_begin ColumnTransformer",
        "fit_begin StandardScaler",
        "fit_begin StandardScaler",  # why second time?
        "fit_begin MinMaxScaler",
        r"fit_begin LogisticRegression\(max_iter=3\)",
        "iter_end coef=.*",
        "iter_end coef=.*",
        "iter_end coef=.*",
    ]
    callback.check_log_expected(log_expected)
