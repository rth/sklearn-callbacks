from time import sleep

from sklearn._callbacks import BaseCallback

from ._computational_graph import ComputeGraph


class TqdmPbar:
    def __init__(self, name, **kwargs):
        from tqdm.auto import tqdm

        self.pbar = tqdm(**kwargs)
        self.n_steps = kwargs.get("total", None)
        self.n_iter = 0
        self.name = name

    def update(self, node, n_iter=None, **kwargs):
        desc = node.name
        if "loss" in kwargs:
            desc += f", loss={kwargs['loss']:8g}"
        elif "score" in kwargs:
            desc += f", score={kwargs['score']:.4f}"
        if desc is not None:
            self.pbar.set_description(desc)
        if n_iter is None:
            n_iter = node.n_iter + 1
        self.pbar.update(max(n_iter - self.n_iter, 0))
        self.n_iter = max(n_iter, self.n_iter)

    def close(self):
        self.pbar.close()

    def finalize(self):
        if self.n_steps and not None and self.n_iter < self.n_steps:
            self.pbar.update(self.n_steps - self.n_iter)
        return self


def _get_node_at_depth(node_init, depth=1):
    if depth > node_init.depth:
        raise ValueError

    node = node_init
    while True:
        node_depth = node.depth
        if node_depth == 1:
            return node
        node = node.parent


class ProgressBar(BaseCallback):
    def __init__(self):
        self.pbar = None
        self.pbar2 = None
        self.compute_graph = None

    def fit(self, estimator, X, y):
        if self.compute_graph is None:
            # assume this first call was made from the root node.
            self.compute_graph = ComputeGraph.from_estimator(estimator)
        self.compute_graph.update_state(estimator)
        root = self.compute_graph.root_node
        if self.pbar is None:
            self.pbar = TqdmPbar(
                total=root.n_steps, name=root.name, desc=root.name, leave=False
            )
        self.pbar.update(root)
        current_node = self.compute_graph.current_node
        if current_node.depth >= 1:
            node = _get_node_at_depth(current_node, depth=1)

            if self.pbar2 is not None and self.pbar2.name != node.name:
                self.pbar2.finalize().close()
                self.pbar2 = None

            if self.pbar2 is None:
                self.pbar2 = TqdmPbar(
                    total=node.n_steps,
                    name=node.name,
                    desc=node.name,
                    leave=False,
                )

            self.pbar2.update(node)

        else:
            if self.pbar2 is not None:
                self.pbar2.close()

    def __call__(self, **kwargs):
        self.compute_graph.current_node.n_iter += 1

        root = self.compute_graph.root_node

        self.pbar.update(root, **kwargs)

        current_node = self.compute_graph.current_node
        if current_node.depth >= 1 and self.pbar2 is not None:
            node = _get_node_at_depth(current_node, depth=1)
            self.pbar2.update(node, **kwargs)
