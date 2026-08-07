import sys

sys.path.insert(0, ".")
from pt2e_counts.analyzer import analyze_graph_counts, compute_conversion_deltas
from pt2e_counts.nodes import classify_node, extract_node_stats


def test_analyze_graph_counts_non_empty():
    g = {"nodes": [{"op": "placeholder", "target": "x"}, {"op": "output", "target": "out"}]}
    stats = analyze_graph_counts(g)
    assert "total" in stats
    assert stats["total"] == 2


def test_compute_conversion_deltas_structure():
    orig = {"nodes": [{"op": "placeholder", "target": "x"}]}
    prep = {"nodes": [{"op": "placeholder", "target": "x"}, {"op": "call_module", "target": "obs"}]}
    conv = {"nodes": [{"op": "placeholder", "target": "x"}, {"op": "call_function", "target": "q"}]}
    res = compute_conversion_deltas(orig, prep, conv)
    assert "prep_delta" in res
    assert "conv_delta" in res


def test_classify_node_observer():
    node = {"op": "call_module", "target": "quantize_per_tensor_observer"}
    assert classify_node(node) == "observer_quant"


def test_total_nodes_consistency():
    g = {"nodes": [{"op": "call_module", "target": "m1"}, {"op": "call_function", "target": "f1"}]}
    stats = extract_node_stats(g)
    cat_sum = sum(v for k, v in stats.items() if k != "total")
    assert stats["total"] == cat_sum
