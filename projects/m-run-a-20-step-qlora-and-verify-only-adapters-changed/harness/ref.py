import numpy as np

def get_model():
    np.random.seed(42)
    return {
        "base_weight": np.random.randn(64, 64).astype(np.float32),
        "lora_a": np.zeros((8, 64), dtype=np.float32),
        "lora_b": np.zeros((64, 8), dtype=np.float32)
    }

def get_data():
    np.random.seed(123)
    return [np.random.randn(64).astype(np.float32) for _ in range(25)]

def run_qlora_steps(model, data, steps=20, lr=0.01):
    m = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in model.items()}
    for i in range(steps):
        x = data[i % len(data)]
        adapter_out = m["lora_b"] @ (m["lora_a"] @ x)
        grad_adapter = 2.0 * adapter_out
        grad_b = np.outer(grad_adapter, (m["lora_a"] @ x))
        grad_a = np.outer((m["lora_b"].T @ grad_adapter), x)
        m["lora_b"] -= lr * grad_b
        m["lora_a"] -= lr * grad_a
    return m

def verify_adapters_changed(initial_model, final_model):
    base_diff = np.max(np.abs(initial_model["base_weight"] - final_model["base_weight"]))
    a_diff = np.max(np.abs(initial_model["lora_a"] - final_model["lora_a"]))
    b_diff = np.max(np.abs(initial_model["lora_b"] - final_model["lora_b"]))
    base_ok = base_diff < 1e-7
    adapters_ok = (a_diff > 1e-7) or (b_diff > 1e-7)
    return bool(base_ok and adapters_ok)
