import logging
import re
import sys
from typing import List

from sklearn._callbacks import BaseCallback


class DebugCallback(BaseCallback):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(message)s",
        )
        self.log = []
        self.handler = logging.StreamHandler(stream=sys.stdout)
        self.handler.setFormatter(self.formatter)
        self.logger = logging.getLogger("sklearn")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)

    def add_message(self, msg):
        self.log.append(msg)
        if self.verbose:
            self.logger.info(msg)

    def on_fit_begin(self, estimator, X, y):
        self.add_message("fit_begin " + str(estimator))

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

    def on_iter_end(self, **kwargs):

        self.add_message(
            "iter_end "
            + ", ".join(f"{key}={val}" for key, val in kwargs.items())
        )
