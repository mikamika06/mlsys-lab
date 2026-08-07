"""Regression tests for un-evaluated graph retention and compilation overhead."""

import sys
sys.path.insert(0, ".")

from mlxgraph.graph import LazyGraphNode, evaluate_and_clean_graph, simulate_lazy_graph_retention
from mlxgraph.bench import measure_mlx_recompilation_cost


def test_unevaluated_graph_references_cleaned():
    nodes = []
    prev = None
    for _ in range(5):
        node = LazyGraphNode((512, 512), parent=prev)
        nodes.append(node)
        prev = node

    freed_bytes = evaluate_and_clean_graph(nodes)
    assert freed_bytes > 0, f"Expected freed_bytes > 0, got {freed_bytes}"
    assert len(nodes) == 0, f"Expected node list to be emptied, got {len(nodes)}"

    sim_retained = simulate_lazy_graph_retention(10, 1024, retain_references=True)
    sim_cleared = simulate_lazy_graph_retention(10, 1024, retain_references=False)
    assert sim_retained[-1]["rss_bytes"] > sim_cleared[-1]["rss_bytes"]


def test_recompilation_penalty_ratio():
    spec = {"num_ops": 4}
    shapes = [(16, 64), (16, 64), (32, 64), (16, 64), (64, 64)]
    res = measure_mlx_recompilation_cost(spec, shapes)
    assert res["recompile_count"] == 3
    assert res["cached_count"] == 2
    assert res["recompile_penalty_ratio"] >= 1.5, f"Expected penalty >= 1.5, got {res['recompile_penalty_ratio']}"
