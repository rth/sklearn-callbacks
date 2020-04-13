import warnings
import re

from sklearn.linear_model import Ridge
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.exceptions import ConvergenceWarning

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
        "fit Pipeline",
        "fit StandardScaler",
        "fit StandardScaler",  # why second time?
        r"fit LogisticRegression\(max_iter=3\)",
        "call coef=.*",
        "call coef=.*",
        "call coef=.*",
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
        "fit Pipeline",
        "fit ColumnTransformer",
        "fit StandardScaler",
        "fit StandardScaler",  # why second time?
        "fit MinMaxScaler",
        r"fit LogisticRegression\(max_iter=3\)",
        "call coef=.*",
        "call coef=.*",
        "call coef=.*",
    ]
    callback.check_log_expected(log_expected)
