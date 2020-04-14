# sklearn-callbacks

[![CircleCI](https://circleci.com/gh/rth/sklearn-callbacks.svg?style=svg)](https://circleci.com/gh/rth/sklearn-callbacks)<Paste>

Experimental callbacks for scikit-learn: progress bars, monitoring convergence etc.

## Install

This package require a patched scikit-learn 0.23.0dev0,
```
pip install https://github.com/rth/scikit-learn/archive/progress-bar.zip
pip install git+https://github.com/rth/sklearn-callbacks.git
```

## Usage

### Progress bars

This package implements progress bars for estimators with iterative solvers,
```py
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn_callbacks import ProgressBar

X, y = make_classification(n_samples=200000, n_features=200, random_state=0)

est = SGDClassifier(max_iter=100, tol=1e-4)
est._set_callbacks(ProgressBar())

est.fit(X, y)
```
![SGD progress bar](./doc/static/img/progressbar-sgd.gif?raw=true "SGD progress bar")

more complex scikit-learn pipelines are also supported,
```py
# see details for full list of imports
from sklearn_callbacks import ProgressBar

X, y = make_classification(n_samples=500000, n_features=200, random_state=0)

pipe = make_pipeline(
    SimpleImputer(),
    make_column_transformer(
        (StandardScaler(), slice(0, 80)),
        (MinMaxScaler(), slice(80, 120)),
        (StandardScaler(with_mean=False), slice(120, 180)),
    ),
    LogisticRegression(),
)


pipe._set_callbacks(ProgressBar())
pipe.fit(X, y)
```

<details>

```py
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn_callbacks import ProgressBar

X, y = make_classification(n_samples=500000, n_features=200, random_state=0)

pipe = make_pipeline(
    SimpleImputer(),
    make_column_transformer(
        (StandardScaler(), slice(0, 80)),
        (MinMaxScaler(), slice(80, 120)),
        (StandardScaler(with_mean=False), slice(120, 180)),
    ),
    LogisticRegression(),
)

pipe._set_callbacks(ProgressBar())

pipe.fit(X, y)
```
</details>

![pipeline progress bar](./doc/static/img/progressbar-pipeline.gif?raw=true "pipeline progress bar")

### Monitoring convergence

```py
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_callbacks import ConvergenceMonitor

X, y = make_regression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

conv_mon = ConvergenceMonitor("mean_absolute_error", X_test, y_test)

pipe = make_pipeline(StandardScaler(), Ridge(solver="sag", alpha=1))
pipe._set_callbacks(conv_mon)
_ = pipe.fit(X_train, y_train)

conv_mon.plot()
```

![convergence monitor](./doc/static/img/convergence-monitor.png?raw=true "convergence monitor")



## License

This project is distributed under the BSD 3-clause license.
