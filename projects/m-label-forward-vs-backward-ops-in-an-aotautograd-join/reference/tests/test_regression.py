import sys
sys.path.insert(0, ".")
from aot_tools.functionalize import functionalize_graph

SAMPLE_GRAPH = {
    "graph_id": "test_g",
    "nodes": [
        {"id": "n0", "op": "add_", "in_place": True},
        {"id": "n1", "op": "mul", "in_place": False}
    ]
}


def test_functionalization_removes_mutations():
    res = functionalize_graph(SAMPLE_GRAPH)
    for node in res["nodes"]:
        assert not node.get("in_place", False), f"Node {node['id']} is still marked in_place"
        assert not node["op"].endswith("_"), f"Node {node['id']} op {node['op']} has trailing underscore"
