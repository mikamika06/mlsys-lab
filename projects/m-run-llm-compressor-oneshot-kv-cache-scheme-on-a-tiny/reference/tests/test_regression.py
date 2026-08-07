import sys
sys.path.insert(0, ".")
import numpy as np
from compressor_kv.oneshot import compute_kv_scales
from compressor_kv.debug import repair_recipe
from compressor_kv.evaluate import relative_error

def test_oneshot_scales_non_trivial():
    cfg = {"layers": 2, "hidden_dim": 32, "num_heads": 2}
    acts = [np.random.standard_normal((16, 32)).astype(np.float32) * 5.0]
    res = compute_kv_scales(cfg, acts)
    assert res["scale"] != 1.0

def test_repair_recipe_fixes_ones():
    acts = [np.random.standard_normal((16, 32)).astype(np.float32) * 5.0]
    recipe = {"scale": 1.0, "scheme": "fp8_e4m3"}
    fixed = repair_recipe(recipe, acts)
    assert fixed["scale"] != 1.0

def test_relative_error_bound():
    ref_out = np.ones((10, 10), dtype=np.float32)
    test_out = np.ones((10, 10), dtype=np.float32) * 1.05
    err = relative_error(ref_out, test_out)
    assert err > 0.0
    assert err < 0.2
