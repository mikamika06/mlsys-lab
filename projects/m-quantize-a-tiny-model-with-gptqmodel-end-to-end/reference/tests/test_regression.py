import sys
import numpy as np

sys.path.insert(0, ".")
from gptqquant.config import make_config
from gptqquant.quantize import quantize_weights
from gptqquant.export import calculate_size_ratio

def test_config_defaults():
    cfg = make_config()
    d = cfg.to_dict()
    assert d["bits"] == 4
    assert d["group_size"] == 128

def test_quantization_shape_and_bounds():
    rng = np.random.default_rng(42)
    w = rng.normal(size=(32, 64)).astype(np.float32)
    cfg = make_config(bits=4, group_size=32)
    qw, scales, zeros = quantize_weights(w, cfg)
    assert qw.shape == w.shape
    assert np.all(qw >= -8) and np.all(qw <= 7)

def test_size_ratio_reduction():
    rng = np.random.default_rng(42)
    w = rng.normal(size=(64, 128)).astype(np.float32)
    cfg = make_config(bits=4, group_size=64)
    qw, scales, zeros = quantize_weights(w, cfg)
    ratio = calculate_size_ratio(w, qw, scales, zeros, cfg)
    assert ratio < 0.5
