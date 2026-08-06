import numpy as np
from converter.rewrite import apply_decomposition_table


def test_decomposition_correctness():
    def gelu_decomp(node):
        return [
            {"id": f"{node['id']}_mul", "op_type": "Mul", "inputs": [node["inputs"][0], "0.5"]},
            {"id": f"{node['id']}_erf", "op_type": "Erf", "inputs": [f"{node['id']}_mul"]},
        ]

    table = {
        "GELU": {"decompose_fn": gelu_decomp}
    }
    graph = {
        "nodes": [
            {"id": "node_0", "op_type": "GELU", "inputs": ["input_x"]}
        ]
    }
    res = apply_decomposition_table(graph, table)
    assert res["decomposed_count"] == 1
    assert len(res["nodes"]) == 2
    assert res["nodes"][0]["op_type"] == "Mul"
    assert res["nodes"][1]["op_type"] == "Erf"
