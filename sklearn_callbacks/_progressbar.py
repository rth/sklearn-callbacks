from sklearn._callbacks import BaseCallback


class ProgressBar(BaseCallback):
    def __init__(self):
        self.pbar = None

    def fit(X, y):
        pass

    def __call__(self, **kwargs):
        from tqdm.auto import tqdm

        if self.pbar is None:
            self.pbar = tqdm(total=kwargs.get("n_iter_total"))
        self.pbar.update(1)
