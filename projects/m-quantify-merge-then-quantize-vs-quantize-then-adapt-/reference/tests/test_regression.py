import sys
sys.path.insert(0, ".")
from loratool.divergence import measure_divergence
from loratool.sizing import adapter_size_bytes
from loratool.scaling import verify_output_shift

def test_divergence_non_negative():
    cfg = {"rank": 8, "alpha": 16.0, "modules": ["q_proj"], "hidden_dim": 256, "dtype_bytes": 2}
    val = measure_divergence(cfg)
    assert val >= 0.0

def test_adapter_size_positive():
    cfg = {"rank": 8, "alpha": 16.0, "modules": ["q_proj", "v_proj"], "hidden_dim": 256, "dtype_bytes": 2}
    size = adapter_size_bytes(cfg)
    assert size > 0
    assert size == len(cfg["modules"]) * (256 * 8 * 2 + 8 * 256 * 2)

def test_scaling_verification():
    assert verify_output_shift(1.0, 2.0, 1.5, 4.0) is True
