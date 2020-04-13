import re
from typing import List

from sklearn._callbacks import BaseCallback
import sklearn


class DebugCallback(BaseCallback):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.log = []

    def add_message(self, msg):
        self.log.append(msg)
        if self.verbose:
            print("[DebugCallback] " + msg)

    def fit(self, estimator, X, y):
        with sklearn.config_context(print_changed_only=True):
            self.add_message("fit " + str(estimator))

    def check_log_expected(self, log: List[str]):
        """Check that the recored log matches expected values
        
        Parameters
        ----------
        log
           list of regexp with the expected lines for each log entry.
        """
        assert len(self.log) == len(log)

        for val, expected in zip(self.log, log):
            if not re.match(expected, val):
                raise AssertionError(
                    f"Expected regexp {expected} does not match '{val}'."
                )

    def __call__(self, **kwargs):

        self.add_message(
            "call " + ", ".join(f"{key}={val}" for key, val in kwargs.items())
        )
