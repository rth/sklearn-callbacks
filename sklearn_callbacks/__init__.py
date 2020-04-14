# pkg_resources is installed with setuptools
from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

from ._debug import DebugCallback
from ._progressbar import ProgressBar
from ._convergence_monitor import ConvergenceMonitor

__all__ = ["DebugCallback", "ProgressBar", "ConvergenceMonitor"]
