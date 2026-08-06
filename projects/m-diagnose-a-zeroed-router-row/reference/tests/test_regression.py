"""Regression tests for MoE parameter counts and router behavior."""

import numpy as np
from moerouter.params import count_parameters
from moerouter.routing import route_tokens


def test_parameter_counts():
    cfg = {
        "hidden_size": 1024,
        "num_layers": 4,
        "moe_layer_frequency": 2,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "ffn_hidden_size": 4096,
        "expert_hidden_size": 2048,
        "non_ffn_layer_params": 1000,
    }
    res = count_parameters(cfg)
    assert res["total_params"] > res["active_params"]
    assert isinstance(res["total_params"], int)
    assert isinstance(res["active_params"], int)


def test_router_handles_extreme_masking():
    logits = np.array([[1.0, 2.0, 3.0, 4.0], [-10.0, -20.0, -30.0, -40.0]])
    mask = np.array(
        [[False, False, False, False], [True, True, True, True]]
    )
    res = route_tokens(logits, top_k=2, mask=mask)

    assert 0 in res["zero_row_diagnosed"]
    assert np.allclose(np.sum(res["weights"], axis=1), 1.0)
    assert res["weights"].shape == (2, 2)
