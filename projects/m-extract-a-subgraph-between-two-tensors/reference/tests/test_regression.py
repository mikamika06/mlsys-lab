import sys
import numpy as np

sys.path.insert(0, ".")
from surgery.extract import extract_subgraph
from surgery.fuse import fuse_gelu
from surgery.fold import fold_constants

def test_extract_bounds():
    graph = {
        "nodes": [{"name": "n1", "op": "Identity", "inputs": ["a"], "outputs": ["b"]}],
        "initializers": {},
        "inputs": ["a"],
        "outputs": ["b"]
    }
    sub = extract_subgraph(graph, "a", "b")
    assert len(sub["nodes"]) == 1

def test_fold_clean():
    graph = {
        "nodes": [],
        "initializers": {"x": np.array([1.0])},
        "inputs": [],
        "outputs": []
    }
    res = fold_constants(graph)
    assert "bad_injected" not in res["initializers"]
