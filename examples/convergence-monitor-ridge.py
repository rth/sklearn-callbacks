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
