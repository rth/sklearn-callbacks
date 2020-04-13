import warnings
import re

from sklearn.linear_model import Ridge
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from sklearn_callbacks import DebugCallback


def test_debug_callback():
    X, y = load_iris(return_X_y=True)

    callback = DebugCallback(verbose=False)

    est = LogisticRegression(max_iter=3)
    est._set_callbacks(callback)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        est.fit(X, y)

    log_expected = [
        r"fit LogisticRegression\(max_iter=3\)",
        "call coef=.*",
        "call coef=.*",
        "call coef=.*",
    ]
    callback.check_log_expected(log_expected)
