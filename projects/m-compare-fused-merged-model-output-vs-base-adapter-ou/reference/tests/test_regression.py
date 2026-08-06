import numpy as np
from loraeval.compare import compute_output_error
from loraeval.diff import diff_adapter_configs


def test_diff_configs():
    c1 = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}
    c2 = {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]}
    res = diff_adapter_configs(c1, c2)
    assert "r" in res
    assert "lora_alpha" in res
    assert res["r"]["config1"] == 8
    assert res["r"]["config2"] == 16


def test_compare_output():
    rng = np.random.default_rng(42)
    bw = rng.normal(size=(16, 16))
    aa = rng.normal(size=(16, 4))
    ab = rng.normal(size=(4, 16))
    x = rng.normal(size=(2, 16))
    err = compute_output_error(bw, aa, ab, scaling=1.0, x=x)
    assert err < 1e-5
