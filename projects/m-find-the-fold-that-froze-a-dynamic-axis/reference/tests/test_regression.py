import sys

sys.path.insert(0, ".")
from graphfix.fold_finder import find_frozen_dynamic_folds
from graphfix.graph_sweep import sweep_dead_and_orphans
from graphfix.metrics import compute_simplification_payoff


def test_detects_frozen_dynamic_axis():
    graph = {
        "inputs": [{"name": "input_ids", "shape": ["batch_size", 128]}],
        "nodes": [
            {
                "name": "node_shape",
                "op": "Shape",
                "inputs": ["input_ids"],
                "outputs": ["shape_out"],
                "is_folded": False
            },
            {
                "name": "node_fold_freeze",
                "op": "ConstantFold",
                "inputs": ["shape_out"],
                "outputs": ["frozen_dim"],
                "is_folded": True
            }
        ],
        "initializers": {},
        "outputs": ["frozen_dim"]
    }
    frozen = find_frozen_dynamic_folds(graph)
    assert "node_fold_freeze" in frozen, f"Expected node_fold_freeze in {frozen}"


def test_sweep_removes_unreachable_nodes_and_initializers():
    graph = {
        "inputs": [{"name": "x", "shape": [1, 64]}],
        "nodes": [
            {"name": "n1", "op": "Relu", "inputs": ["x"], "outputs": ["y"]},
            {"name": "dead_node", "op": "Add", "inputs": ["orphan_param"], "outputs": ["dead_out"]}
        ],
        "initializers": {"orphan_param": [1.0, 2.0, 3.0], "valid_param": [0.5]},
        "outputs": ["y"]
    }
    cleaned = sweep_dead_and_orphans(graph)
    node_names = [n["name"] for n in cleaned["nodes"]]
    assert "dead_node" not in node_names
    assert "orphan_param" not in cleaned["initializers"]


def test_simplification_payoff_calculation():
    before = {
        "nodes": [{"name": "n1"}, {"name": "n2"}, {"name": "n3"}],
        "initializers": {"p1": [1, 2, 3, 4], "p2": [5, 6]}
    }
    after = {
        "nodes": [{"name": "n1"}],
        "initializers": {"p1": [1, 2, 3, 4]}
    }
    payoff = compute_simplification_payoff(before, after)
    assert payoff["nodes_removed"] == 2
    assert payoff["initializers_removed"] == 1
    assert payoff["bytes_saved"] == 2
