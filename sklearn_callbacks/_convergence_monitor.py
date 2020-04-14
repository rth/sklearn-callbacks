from typing import List, Dict

from sklearn._callbacks import BaseCallback
from sklearn.base import clone
import sklearn.metrics
from sklearn.linear_model._base import LinearModel


class ConvergenceMonitor(BaseCallback):
    """Monitor model convergence.

    Currently only a few linear models are supported
    (e.g. ``Ridge(solver="sag")``)

    Parameters
    ----------
    metric
         metric to evaluate
    X_test, y_test
         optional validation data
    """

    def __init__(self, metric: str, X_test=None, y_test=None):
        self.metric = metric
        self.metric_func = getattr(sklearn.metrics, metric, None)
        if self.metric_func is None:
            raise ValueError(f"uknown metric={metric}")
        self.data: List[Dict] = []
        self.X_test = X_test
        self.y_test = y_test

    def fit(self, estimator, X, y):
        if not isinstance(estimator, LinearModel):
            # not implemented
            return
        self.X_train = X
        self.y_train = y
        # Explicitly clone later, so that estimator
        # attributes can still be modified in fit
        self.estimator = estimator

    def __call__(self, **kwargs):
        coef = kwargs.get("coef", None)
        intercept = kwargs.get("intercept", None)
        if coef is None or intercept is None:
            raise NotImplementedError

        # create a new estimator with updated coefs
        est = clone(self.estimator)
        est.coef_ = coef.reshape(-1)
        est.intercept_ = intercept.reshape(-1)

        y_pred = est.predict(self.X_train)
        score_train = self.metric_func(self.y_train, y_pred)
        res = {"score_train": score_train}
        if self.X_test is not None:
            y_pred = est.predict(self.X_test)
            score_test = self.metric_func(self.y_test, y_pred)
            res["score_test"] = score_test
        self.data.append(res)

    def plot(self, ax=None):
        import pandas as pd
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        df = pd.DataFrame(self.data)
        df.plot(ax=ax)
        ax.set_xlabel("Number of iterations")
        ax.set_ylabel(self.metric)
        with sklearn.config_context(print_changed_only=True):
            ax.set_title(str(self.estimator))
