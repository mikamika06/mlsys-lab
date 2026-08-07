import numpy as np


def check(workdir):
    out = {"loss_decreases": 0.0, "weights_frozen": 0.0, "adapters_changed": 0.0}
    try:
        from qlora.layer import LinearQLoRA
        from qlora.train import train_20_steps
    except ImportError as e:
        out["_note"] = f"Could not import: {e}"
        return out

    rng = np.random.default_rng(999)
    X = rng.normal(0, 1, size=(8, 16)).astype(np.float32)
    target = rng.normal(0, 1, size=(8, 32)).astype(np.float32)

    try:
        layer = LinearQLoRA(16, 32, seed=123)
    except Exception as e:
        out["_note"] = f"Layer init failed: {e}"
        return out

    w_before = layer.weight.copy()
    s_before = layer.scale.copy()
    a_before = layer.lora_A.copy()
    b_before = layer.lora_B.copy()

    try:
        losses = train_20_steps(layer, X, target, lr=0.1)
    except Exception as e:
        out["_note"] = f"Training failed: {e}"
        return out

    if len(losses) > 0 and losses[-1] < losses[0]:
        out["loss_decreases"] = 1.0
    elif len(losses) > 0:
        out["_note"] = "Loss did not decrease over 20 steps"

    if np.array_equal(layer.weight, w_before) and np.array_equal(layer.scale, s_before):
        out["weights_frozen"] = 1.0

    if not np.allclose(layer.lora_A, a_before) and not np.allclose(layer.lora_B, b_before):
        out["adapters_changed"] = 1.0

    return out
