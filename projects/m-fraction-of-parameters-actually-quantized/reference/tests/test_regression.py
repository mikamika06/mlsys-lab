import sys
sys.path.insert(0, ".")
from quant_target.targeting import filter_target_modules
from quant_target.metrics import compute_quantized_fraction
from quant_target.analyzer import analyze_model_quantization

CONFIG = {
    "is_multimodal": True,
    "language_only_quant": True,
    "layers": [
        {"name": "vision_encoder.proj", "type": "linear", "params": 1000000},
        {"name": "model.layers.0.self_attn.q_proj", "type": "linear", "params": 2000000},
        {"name": "model.layers.0.mlp.gate_proj", "type": "linear", "params": 2000000},
        {"name": "lm_head", "type": "linear", "params": 5000000}
    ]
}


def test_multimodal_vision_ignored():
    targets = filter_target_modules(CONFIG, ignore_list=[])
    names = [t["name"] for t in targets]
    assert "vision_encoder.proj" not in names


def test_lm_head_included_by_default():
    targets = filter_target_modules(CONFIG, ignore_list=[])
    names = [t["name"] for t in targets]
    assert "lm_head" in names


def test_fraction_calculation_bounds():
    targets = filter_target_modules(CONFIG, ignore_list=[])
    frac, cost = compute_quantized_fraction(CONFIG, targets)
    assert 0.0 <= frac <= 1.0
    assert cost == 5000000


def test_analyzer_returns_expected_keys():
    res = analyze_model_quantization(CONFIG)
    assert "fraction" in res
    assert "targets" in res
    assert res["total_params"] == 10000000
