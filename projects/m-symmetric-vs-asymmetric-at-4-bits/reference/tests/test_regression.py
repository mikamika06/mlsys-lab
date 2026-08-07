import sys

sys.path.insert(0, ".")
from quant.dynamic import match_dynamic_override
from quant.engine import determine_group_size
from quant.config import quantize_weights


def test_dynamic_override_matching():
    assert match_dynamic_override(".*layer.*", "model.layers.0.self_attn") is True
    assert match_dynamic_override("nomatch", "model.layers.0.self_attn") is False


def test_group_size_bounds():
    case = {"total_params": 1024, "target_bits": 4.0, "symmetric": True}
    gs = determine_group_size(case)
    assert gs > 0


def test_quantize_symmetry():
    cfg = {"bits": 4, "symmetric": True, "group_size": 128, "weights": [0.1, -0.2]}
    res = quantize_weights(cfg)
    assert res["symmetric"] is True
