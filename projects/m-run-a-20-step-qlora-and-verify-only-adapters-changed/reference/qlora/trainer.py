import numpy as np

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
