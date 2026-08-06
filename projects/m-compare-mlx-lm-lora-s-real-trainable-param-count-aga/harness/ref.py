import numpy as np


def generate_audit_fixture(seed=42):
    np.random.seed(seed)
    specs = [
        {"in_features": 128, "out_features": 256},
        {"in_features": 256, "out_features": 512},
        {"in_features": 512, "out_features": 512},
    ]
    r = 8

    layers_lora = {}
    for i, spec in enumerate(specs):
        in_dim = spec["in_features"]
        out_dim = spec["out_features"]
        layers_lora[f"layer_{i}"] = {
            "lora_a": {"weight": np.random.randn(r, in_dim), "trainable": True},
            "lora_b": {"weight": np.random.randn(out_dim, r), "trainable": True},
            "base": {"weight": np.random.randn(out_dim, in_dim), "trainable": False}
        }

    layers_dora = {}
    for i, spec in enumerate(specs):
        in_dim = spec["in_features"]
        out_dim = spec["out_features"]
        layers_dora[f"layer_{i}"] = {
            "lora_a": {"weight": np.random.randn(r, in_dim), "trainable": True},
            "lora_b": {"weight": np.random.randn(out_dim, r), "trainable": True},
            "m": {"weight": np.random.randn(out_dim, 1), "trainable": True},
            "base": {"weight": np.random.randn(out_dim, in_dim), "trainable": False}
        }

    return {
        "specs": specs,
        "r": r,
        "layers_lora": layers_lora,
        "layers_dora": layers_dora
    }


def generate_fuse_fixture(seed=99):
    np.random.seed(seed)
    in_dim, out_dim, r = 64, 128, 4
    scale = 2.0
    base = np.random.randn(out_dim, in_dim)
    lora_a = np.random.randn(r, in_dim)
    lora_b = np.random.randn(out_dim, r)

    delta = (lora_b @ lora_a) * scale
    fused_lora = base + delta

    w_comb = base + delta
    norm_w = np.linalg.norm(w_comb, axis=1, keepdims=True)
    m_vec = np.linalg.norm(base, axis=1, keepdims=True)
    fused_dora = m_vec * (w_comb / norm_w)

    return {
        "base": base,
        "lora_a": lora_a,
        "lora_b": lora_b,
        "scale": scale,
        "m_vec": m_vec,
        "fused_lora": fused_lora,
        "fused_dora": fused_dora
    }
