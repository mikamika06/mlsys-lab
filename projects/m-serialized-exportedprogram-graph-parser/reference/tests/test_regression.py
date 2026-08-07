import sys

sys.path.insert(0, ".")
from gparser.parser import parse_graph
from gparser.inspector import inspect_shapes, inspect_dtypes, inspect_io
from gparser.optimizer import optimize_graph

SAMPLE_PROG = {
    "metadata": {"version": 1},
    "nodes": [
        {"name": "x", "op": "placeholder", "target": "x", "inputs": [], "outputs": [{"name": "x_out", "shape": [2, 2], "dtype": "float32"}]},
        {"name": "y", "op": "placeholder", "target": "y", "inputs": [], "outputs": [{"name": "y_out", "shape": [2, 2], "dtype": "float32"}]},
        {"name": "add", "op": "call_function", "target": "add", "inputs": ["x_out", "y_out"], "outputs": [{"name": "add_out", "shape": [2, 2], "dtype": "float32"}]},
        {"name": "out", "op": "output", "target": "output", "inputs": ["add_out"], "outputs": []}
    ]
}


def test_parse_nodes_count():
    parsed = parse_graph(SAMPLE_PROG)
    assert len(parsed["nodes"]) == 4


def test_inspect_shapes():
    parsed = parse_graph(SAMPLE_PROG)
    shapes = inspect_shapes(parsed)
    assert shapes["x_out"] == (2, 2)


def test_inspect_dtypes():
    parsed = parse_graph(SAMPLE_PROG)
    dtypes = inspect_dtypes(parsed)
    assert dtypes["x_out"] == "float32"


def test_optimize_graph():
    parsed = parse_graph(SAMPLE_PROG)
    opt = optimize_graph(parsed)
    assert len(opt["nodes"]) == 4
