import sys

sys.path.insert(0, ".")
from kvquant.metrics import compare_quality
from kvquant.recipe import make_recipe
from kvquant.serving import build_serving_args


def test_recipe_contains_fp8_dtype():
    cfg = {"model_name": "test-model", "num_layers": 16, "hidden_size": 2048}
    recipe = make_recipe(cfg, scheme="fp8", block_size=16)
    assert recipe["kv_cache_dtype"] == "fp8"
    assert recipe["kv_cache_scheme"] == "fp8"


def test_serving_args_include_dtype_and_scheme():
    recipe = {"model": "test", "kv_cache_dtype": "fp8", "kv_cache_scheme": "fp8", "block_size": 16}
    args = build_serving_args(recipe, "/path")
    assert "--kv-cache-dtype" in args
    assert "fp8" in args
    assert "--kv-cache-scheme" in args


def test_quality_comparison_detects_drift():
    base = [1.0, 2.0, 3.0, 4.0]
    quant_good = [1.01, 1.99, 3.02, 3.98]
    quant_bad = [10.0, -5.0, 20.0, -15.0]
    res_good = compare_quality(base, quant_good)
    res_bad = compare_quality(base, quant_bad)
    assert res_good["valid"] is True
    assert res_bad["valid"] is False
