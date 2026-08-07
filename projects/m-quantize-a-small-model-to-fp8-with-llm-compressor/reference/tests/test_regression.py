import sys

sys.path.insert(0, ".")
from fp8quant.recipe import build_recipe
from fp8quant.pipeline import run_compression


def test_recipe_is_fp8():
    r = build_recipe()
    assert r.get("quant_method") == "fp8", f"expected fp8 quant_method, got {r.get('quant_method')}"
    assert r.get("weight_dtype") == "fp8_e4m3", f"expected fp8_e4m3 weight_dtype, got {r.get('weight_dtype')}"
    assert r.get("input_dtype") == "fp8_e4m3", f"expected fp8_e4m3 input_dtype, got {r.get('input_dtype')}"


def test_compression_ratio_target():
    ratio = run_compression()
    assert ratio <= 0.55, f"compression ratio {ratio} exceeds target 0.55"
