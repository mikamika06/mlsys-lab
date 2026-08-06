import sys

sys.path.insert(0, ".")
from edge_export.lower import lower_model
from edge_export.census import compute_census
from edge_export.repair import repair_graph

def test_lowering_produces_pte_format():
    spec = {"nodes": [{"name": "conv_1", "op": "conv2d"}]}
    res = lower_model(spec)
    assert res["format"] == "pte"
    assert len(res["nodes"]) == 1

def test_census_counts_correctly():
    nodes = [{"target": "xnnpack"}, {"target": "cpu_fallback"}, {"target": "xnnpack"}]
    c = compute_census(nodes)
    assert c["delegated"] == 2
    assert c["fallback"] == 1
    assert c["ratio"] > 0.6

def test_repair_fixes_unsupported_ops():
    graph = {"nodes": [{"name": "n1", "op": "unsupported_clamp", "type": "fp64"}]}
    rep = repair_graph(graph)
    assert rep["nodes"][0]["op"] == "relu6"
    assert rep["nodes"][0]["type"] == "fp32"
    assert rep["verified"] is True
