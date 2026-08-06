import numpy as np


def get_cases():
    rng = np.random.default_rng(42)
    w1 = rng.normal(size=(10, 10))
    w2 = rng.normal(size=(10, 10))
    cases_m1 = [
        ({"model.embed_tokens.weight": w1, "lm_head.weight": w1}, True),
        ({"model.embed_tokens.weight": w1, "lm_head.weight": w2}, False),
    ]

    hf_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.unsupported.weight",
    ]
    mapped = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
    }
    cases_m2 = (hf_keys, mapped, ["model.layers.0.unsupported.weight"])

    moe_names = [
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.1.w1.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
    ]
    return cases_m1, cases_m2, moe_names
