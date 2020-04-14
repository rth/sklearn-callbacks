from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn_callbacks._computational_graph import ComputeGraph


def test_graph_single_estimator():
    est = StandardScaler()
    graph = ComputeGraph.from_estimator(est)
    assert str(graph.root_node) == "StandardScaler 0 / 1"
    assert len(graph) == 1
    assert graph.root_node.depth == 0


def test_graph_pipeline():
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=7))
    graph = ComputeGraph.from_estimator(pipe)
    assert len(graph) == 3
    graph_flat_str = [(node.depth, str(node)) for node in graph]
    assert graph_flat_str == [
        (0, "Pipeline 0 / 2"),
        (1, "StandardScaler 0 / 1"),
        (1, "LogisticRegression 0 / 7"),
    ]


def test_graph_pipeline_column_transformer():
    pipe_0 = make_pipeline(MinMaxScaler())
    pipe = make_pipeline(
        make_column_transformer((StandardScaler(), [0, 1]), (pipe_0, [2, 3]),),
        LogisticRegression(max_iter=7),
    )

    graph = ComputeGraph.from_estimator(pipe)
    assert len(graph) == 6
    graph_flat_str = [(node.depth, str(node)) for node in graph]
    assert graph_flat_str == [
        (0, "Pipeline 0 / 2"),
        (1, "ColumnTransformer 0 / 2"),
        (2, "StandardScaler 0 / 1"),
        (2, "Pipeline 0 / 1"),
        (3, "MinMaxScaler 0 / 1"),
        (1, "LogisticRegression 0 / 7"),
    ]
    graph.update_state(pipe_0)

    graph_flat_str = [(node.depth, str(node)) for node in graph]
    assert graph_flat_str == [
        (0, "Pipeline 0 / 2"),
        (1, "ColumnTransformer 1 / 2"),  # parent progress changed
        (2, "StandardScaler 0 / 1"),
        (2, "Pipeline 0 / 1"),
        (3, "MinMaxScaler 0 / 1"),
        (1, "LogisticRegression 0 / 7"),
    ]
