import sys

sys.path.insert(0, ".")
from exportops.counter import count_aten_ops, identify_decompositions
from exportops.mutations import capture_export_mutation_error


def test_count_aten_ops_filters_correctly():
    graph = [
        {"op": "call_function", "target": "aten.add.Tensor"},
        {"op": "call_module", "target": "linear_layer"},
        {"op": "call_function", "target": "torch.ops.aten.mul.Tensor"},
        {"op": "call_function", "target": "aten.add.Tensor"}
    ]
    counts = count_aten_ops(graph)
    assert counts.get("aten.add.Tensor", 0) == 2
    assert counts.get("torch.ops.aten.mul.Tensor", 0) == 1
    assert "linear_layer" not in counts
    assert len(counts) == 2


def test_identify_decompositions():
    graph = [
        {"op": "call_function", "target": "aten.add.Tensor"},
        {"op": "call_function", "target": "aten.mul.Tensor"}
    ]
    targets = ["torch.ops.higher_order.cond", "aten.native_layer_norm.default"]
    res = identify_decompositions(graph, targets)
    assert res["fully_decomposed"] is True
    assert res["counts"]["torch.ops.higher_order.cond"] == 0


def _dummy_export(m):
    if m == "mutating_model":
        raise ValueError("Unsupported mutation on global state")
    return "ExportedProgram"


def test_mutation_capture():
    res1 = capture_export_mutation_error("mutating_model", _dummy_export)
    assert res1 == ("ValueError", "Unsupported mutation on global state")

    res2 = capture_export_mutation_error("pure_model", _dummy_export)
    assert res2 is None
