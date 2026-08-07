import sys

sys.path.insert(0, ".")
from gguf_pred.inventory import tensor_bytes, predict_output_size
from gguf_pred.recipe import resolve_recipe
from gguf_pred.delta import explain_delta


def test_tensor_bytes_positive():
    t = {"name": "test.weight", "shape": [256, 256], "qtype": "Q4_K"}
    assert tensor_bytes(t) > 0


def test_predict_output_size_aggregation():
    inv = {"tensors": [{"name": "a", "shape": [256, 256], "qtype": "F32"}]}
    assert predict_output_size(inv) == 256 * 256 * 4


def test_resolve_recipe_default():
    assert resolve_recipe("output.weight", "Q4_K") == "Q4_K"


def test_explain_delta_positive():
    t = {"name": "test.weight", "shape": [256, 256], "qtype": "Q4_K"}
    assert explain_delta(t) == (256 * 256 // 256) * 8
