import sys
sys.path.insert(0, ".")
from gguf_quant.analyzer import quantify_requantize_error, compare_recipes
from gguf_quant.validator import validate_pruned_model


def test_requantize_error_non_negative():
    w = [0.1, 0.5, 0.9, 1.2, 3.4]
    err = quantify_requantize_error(w, 0.25)
    assert err >= 0.0, f"extra error is negative: {err}"


def test_recipe_comparison_keys():
    w = [1.0, 2.0, 3.0]
    res = compare_recipes(w)
    assert "default" in res
    assert "pure" in res


def test_validator_catches_retained_pruned_layer():
    model = {"header": "GGUF_VALID", "tensors": {"layer_0.weight": [1, 2], "layer_1.weight": [3, 4]}}
    assert not validate_pruned_model(model, [1])
