from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

from sklearn_callbacks import ProgressBar


X, y = make_classification(n_samples=200000, n_features=200, random_state=0)

est = SGDClassifier(max_iter=100, tol=1e-4)

pbar = ProgressBar()
est._set_callbacks(pbar)

est.fit(X, y)

pbar.pbar.close()
