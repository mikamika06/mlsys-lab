import sys

sys.path.insert(0, ".")

from onnxcalc.broadcast import compute_broadcast_shape
from onnxcalc.checker import triage_graph


def test_broadcast_rank_alignment():
    res = compute_broadcast_shape([4, 8, 16], [8, 16])
    assert res == [4, 8, 16], f"expected [4, 8, 16], got {res}"


def test_broadcast_incompatible_dimensions():
    try:
        compute_broadcast_shape([3, 4], [3, 5])
        assert False, "expected ValueError for incompatible dimensions [3, 4] and [3, 5]"
    except ValueError:
        pass


def test_triage_graph_invalid_broadcast():
    graph = {
        "inputs": [
            {"name": "A", "shape": [3, 4], "type": "float32"},
            {"name": "B", "shape": [3, 5], "type": "float32"}
        ],
        "nodes": [
            {"op": "Add", "inputs": ["A", "B"], "outputs": ["C"]}
        ]
    }
    res = triage_graph(graph)
    assert not res["valid"], "graph with incompatible broadcast shapes should be invalid"
