# Building an approximate graph for scikit-learn calculations.
#
# The goal is to determine for each callback call which estimator
# it belongs to, and what impact it has on the overall progress
# of computations. Very experimental.
from sklearn.base import BaseEstimator

from collections.abc import Iterator


class ComputeNode:
    """A node in a scikit-learn computational graph"""

    def __init__(self, _id, name, parent=None, children=None, max_iter=None):
        self._id = _id
        self.name = name
        if children is None:
            self.children = []
        else:
            self.children = children
        self.parent = parent
        self.n_iter = 0
        self.max_iter = max_iter

    @classmethod
    def from_estimator(cls, estimator, max_depth=0, parent=None):
        """Create a computational node from estimator.

        Set max_depth=0, to avoid recursive build. And max_depth=-1
        to remove limitation on the recursion depth
        """
        name = estimator.__class__.__name__
        _id = id(estimator)
        max_iter = estimator.get_params().get("max_iter", None)
        node = cls(_id, name, max_iter=max_iter, parent=parent)

        if max_depth == 0:
            return node

        if max_depth == -1:
            child_max_depth = -1
        else:
            child_max_depth = max_depth - 1
        for attr_name in getattr(estimator, "_required_parameters", []):
            # likely a meta-estimator
            if attr_name not in ["steps", "transformers"]:
                continue
            for attr in getattr(estimator, attr_name):
                if isinstance(attr, BaseEstimator):
                    node.children.append(
                        cls.from_estimator(
                            attr, max_depth=child_max_depth, parent=node
                        )
                    )
                elif (
                    hasattr(attr, "__len__")
                    and len(attr) >= 2
                    and isinstance(attr[1], BaseEstimator)
                ):
                    # e.g. Pipeline or ColumnTransformer
                    node.children.append(
                        cls.from_estimator(
                            attr[1], max_depth=child_max_depth, parent=node
                        )
                    )
        return node

    @property
    def root(self):
        """Find the root node"""
        if self.parent is None:
            return self
        else:
            return self.parent.root

    @property
    def depth(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.depth + 1

    @property
    def n_steps(self):
        if len(self.children):
            return len(self.children)
        elif self.max_iter is not None:
            return self.max_iter
        else:
            return 1

    def __repr__(self):
        return f"{self.name} {self.n_iter} / {self.n_steps}"

    def next(self) -> "ComputeNode":
        """Depth first tree traversal"""
        if self.children:
            # go down the graph
            return self.children[0]
        # one (or several levels up) and onto the next child
        prev_node = self
        while True:
            if prev_node.parent is None:
                raise StopIteration
            node = prev_node.parent
            idx = node.children.index(prev_node)
            if idx + 1 < len(node.children):
                return node.children[idx + 1]
            else:
                # go one level up
                prev_node = node


class ComputeGraph(Iterator):
    def __init__(self, root: ComputeNode):
        self.root_node = root
        self.current_node = root
        self._id_map = {node._id: node for node in self}

    @classmethod
    def from_estimator(cls, estimator, max_depth=-1):
        """Build the computational graph from the root estimator"""
        root_node = ComputeNode.from_estimator(estimator, max_depth=max_depth)
        return cls(root_node)

    def __next__(self):
        node = self.current_node.next()
        self.current_node = node
        return node

    def __iter__(self):
        yield self.root_node
        node_prev = self.root_node
        while True:
            try:
                node = node_prev.next()
            except StopIteration:
                break
            yield node
            node_prev = node

    def __repr__(self):
        out = []
        for node in self:
            indent = node.depth
            out.append("{}- {}".format("  " * indent, node))
        return "\n".join(out)

    def __len__(self):
        return len([None for el in self])

    def update_state(self, estimator):
        """Set the next active state
        
        All earlier node are assumed to be computed, but only
        parent n_iter is properly updated.
        """
        node = self._id_map.get(id(estimator), None)
        name = estimator.__class__.__name__
        if node is None:
            next_node = self.current_node.next()
            if self.current_node.name == name:
                return
            elif next_node.name == name:
                node = next_node
            else:
                raise ValueError(f"Could not identify state for {estimator}")

        self.current_node = node
        # update parent progress
        if node.parent is not None:
            idx = node.parent.children.index(node)
            node.parent.n_iter = idx
