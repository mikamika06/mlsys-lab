"""Regression tests for constant folding and graph surgery."""
import sys
sys.path.insert(0, ".")
from surgery.fold import fold_constants
from surgery.extract import extract_subgraph
from surgery.fuse import fuse_gelu
import numpy as np

def test_constant_folding_removes_initializer_nodes():
    graph = {
        "nodes": [{"name": "n1", "op_type": "Add", "inputs": ["A", "B"], "outputs": ["C"]}],
        "initializer": {"A": np.array([1.0], dtype=np.float32), "B": np.array([2.0], dtype=np.float32)},
        "input": [],
        "output": ["C"]
    }
    folded = fold_constants(graph)
    assert len(folded["nodes"]) == 0
    assert "C" in folded["initializer"]
    assert folded["initializer"]["C"] == 3.0

def test_subgraph_extraction_retains_correct_outputs():
    graph = {
        "nodes": [
            {"name": "n1", "op_type": "Mul", "inputs": ["X", "W1"], "outputs": ["T1"]},
            {"name": "n2", "op_type": "Add", "inputs": ["T1", "W2"], "outputs": ["Y"]}
        ],
        "initializer": {"W1": np.ones((2,), dtype=np.float32), "W2": np.ones((2,), dtype=np.float32)},
        "input": [{"name": "X"}],
        "output": ["Y"]
    }
    sub = extract_subgraph(graph, "T1", "Y")
    assert len(sub["nodes"]) == 1
    assert sub["nodes"][0]["op_type"] == "Add"

def test_gelu_fusion_preserves_op_type():
    graph = {
        "nodes": [{"name": "e1", "op_type": "Erf", "inputs": ["X"], "outputs": ["Y"]}],
        "initializer": {},
        "input": [{"name": "X"}],
        "output": ["Y"]
    }
    fused = fuse_gelu(graph)
    assert any(n["op_type"] == "Gelu" for n in fused["nodes"])
