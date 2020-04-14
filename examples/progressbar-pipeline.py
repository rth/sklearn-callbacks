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


pbar = ProgressBar()
pipe._set_callbacks(pbar)

_ = pipe.fit(X, y)
