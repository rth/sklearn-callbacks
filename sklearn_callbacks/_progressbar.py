from sklearn._callbacks import BaseCallback


class ProgressBar(BaseCallback):
    def __init__(self):
        self.pbar = None

    def fit(self, estimator, X, y):
        self.estimator = estimator
        max_iter = estimator.get_params().get("max_iter", None)
        if max_iter is not None:
            self.max_iter = max_iter

    def __call__(self, **kwargs):
        from tqdm.auto import tqdm

        if hasattr(self, "max_iter"):
            max_iter = self.max_iter
        if "max_iter" in kwargs:
            max_iter = kwargs["max_iter"]

        desc = self.estimator.__class__.__name__
        if "loss" in kwargs:
            desc += f", loss={kwargs['loss']:8g}"
        elif "score" in kwargs:
            desc += f", score={kwargs['score']:.4f}"

        if self.pbar is None:
            self.pbar = tqdm(total=max_iter, desc=desc)
        self.pbar.set_description(desc)
        self.pbar.update(1)
