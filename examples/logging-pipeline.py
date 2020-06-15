from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn_callbacks import DebugCallback

X, y = make_classification(n_samples=10000, n_features=100, random_state=0)

pipe = make_pipeline(
    SimpleImputer(),
    make_column_transformer(
        (StandardScaler(), slice(0, 80)), (MinMaxScaler(), slice(80, 90)),
    ),
    SGDClassifier(max_iter=20),
    verbose=1,
)


pbar = DebugCallback()
# pipe._set_callbacks(pbar)

_ = pipe.fit(X, y)
