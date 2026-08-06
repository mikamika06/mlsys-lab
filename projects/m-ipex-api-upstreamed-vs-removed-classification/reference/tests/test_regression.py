"""Regression tests for IPEX API migration and op-graph analysis."""

import sys

sys.path.insert(0, ".")

from ipexaudit.classify import classify_api_call
from ipexaudit.graph import diff_op_graphs


def test_api_classification_upstreamed():
    res = classify_api_call("ipex.optimize")
    assert res["status"] == "upstreamed"
    assert res["target"] == "torch.xpu.optimize"

    res_removed = classify_api_call("ipex.core.enable_auto_dnnl")
    assert res_removed["status"] == "removed"
    assert res_removed["target"] is None


def test_op_graph_diff_detects_redundant_copies():
    manual = {
        "nodes": [
            {"id": 0, "op": "input", "output_bytes": 100},
            {"id": 1, "op": "to", "target_format": "channels_last", "output_bytes": 100},
            {"id": 2, "op": "conv2d", "weight_bytes": 50, "output_bytes": 200},
        ]
    }
    ipex = {
        "nodes": [
            {"id": 0, "op": "input", "output_bytes": 100},
            {"id": 1, "op": "ipex_conv2d", "weight_bytes": 50, "output_bytes": 200},
        ]
    }
    diff = diff_op_graphs(manual, ipex)
    assert diff["redundant_copies_removed"] == 1
    assert diff["manual_node_count"] == 3
    assert diff["ipex_node_count"] == 2


def test_unoptimized_graph_flag():
    unoptimized_manual = {
        "nodes": [
            {"id": 0, "op": "input"},
            {"id": 1, "op": "to", "target_format": "channels_last", "output_bytes": 100},
            {"id": 2, "op": "conv2d"},
        ]
    }
    unoptimized_ipex_copy = {
        "nodes": [
            {"id": 0, "op": "input"},
            {"id": 1, "op": "to", "target_format": "channels_last", "output_bytes": 100},
            {"id": 2, "op": "conv2d"},
        ]
    }
    diff = diff_op_graphs(unoptimized_manual, unoptimized_ipex_copy)
    assert not diff["is_ipex_optimized"]
    assert diff["redundant_copies_removed"] == 0
