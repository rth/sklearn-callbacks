# sklearn-callbacks
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


## Estimators with callbacks

Following estimatators currently have some support of callbacks,



## License

This project is distributed under the BSD 3-clause license.
