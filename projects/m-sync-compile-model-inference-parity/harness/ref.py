import numpy as np


def generate_model_config(seed=42):
    rng = np.random.RandomState(seed)
    in_dim = 16
    hidden_dim = 32
    out_dim = 8
    W1 = rng.randn(in_dim, hidden_dim).astype(np.float32)
    b1 = rng.randn(hidden_dim).astype(np.float32)
    W2 = rng.randn(hidden_dim, out_dim).astype(np.float32)
    b2 = rng.randn(out_dim).astype(np.float32)
    return {
        "input_shape": (4, in_dim),
        "layers": [
            {"weights": W1, "bias": b1, "activation": "relu"},
            {"weights": W2, "bias": b2, "activation": "linear"},
        ],
    }


def reference_infer(model_config, inputs):
    x = np.asarray(inputs, dtype=np.float32)
    expected = tuple(model_config["input_shape"])
    if x.shape != expected:
        raise ValueError(f"Shape mismatch: expected {expected}, got {x.shape}")
    for layer in model_config["layers"]:
        x = np.dot(x, layer["weights"]) + layer["bias"]
        if layer.get("activation") == "relu":
            x = np.maximum(0.0, x)
    return x


def compute_rel_err(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.maximum(np.abs(b), 1e-7)
    return float(np.max(np.abs(a - b) / denom))


TEST_CONFIGS = [generate_model_config(10 + i) for i in range(3)]

RNG = np.random.RandomState(99)
TEST_INPUTS = [RNG.randn(4, 16).astype(np.float32) for _ in range(5)]
