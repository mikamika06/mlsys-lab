import sys
sys.path.insert(0, ".")
from tvm_compare.latency import compute_latency_ratio
from tvm_compare.frontend import import_or_catch_error


def test_latency_ratio_bounds():
    ratio = compute_latency_ratio("model_a", [10.0, 12.0], [5.0, 6.0])
    assert ratio > 0.0
    assert ratio == 2.0


def test_unsupported_op_error_format():
    res = import_or_catch_error({"unsupported_op": "nn.CustomOp"})
    assert res["success"] is False
    assert res["error_type"] == "UnsupportedError"
    assert res["op"] == "nn.CustomOp"
