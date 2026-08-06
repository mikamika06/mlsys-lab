import sys
import numpy as np

sys.path.insert(0, ".")
from lorameasure.params import count_trainable_params
from lorameasure.expansion import expand_target_modules
from lorameasure.stochasticity import measure_dropout_stochasticity


def test_param_count_scales_linearly_with_rank():
    model = {
        "modules": {
            "layer0.q_proj": {"type": "linear", "in_features": 256, "out_features": 256},
            "layer0.v_proj": {"type": "linear", "in_features": 256, "out_features": 256},
        }
    }
    c4 = count_trainable_params(model, ["q_proj", "v_proj"], r=4)
    c8 = count_trainable_params(model, ["q_proj", "v_proj"], r=8)
    assert c8 == 2 * c4, f"Expected params for r=8 ({c8}) to be double r=4 ({c4})"


def test_all_linear_expansion_excludes_lm_head():
    model = {
        "lm_head_name": "lm_head",
        "modules": {
            "model.layers.0.self_attn.q_proj": {"type": "linear", "in_features": 128, "out_features": 128},
            "model.layers.0.mlp.gate_proj": {"type": "linear", "in_features": 128, "out_features": 512},
            "lm_head": {"type": "linear", "in_features": 128, "out_features": 1000},
        }
    }
    targets = expand_target_modules(model, "all-linear")
    assert "lm_head" not in targets, "lm_head should be excluded from all-linear expansion"
    assert len(targets) == 2, f"Expected 2 targets, got {targets}"


def test_zero_dropout_is_deterministic():
    x = np.ones((2, 16), dtype=np.float64)
    w_a = np.ones((4, 16), dtype=np.float64)
    w_b = np.ones((16, 4), dtype=np.float64)
    res = measure_dropout_stochasticity(x, w_a, w_b, lora_alpha=8, lora_dropout=0.0, num_samples=5)
    assert res["mean_variance"] == 0.0, "Zero dropout must have 0 output variance"
    assert not res["is_stochastic"], "Zero dropout should not be stochastic"
