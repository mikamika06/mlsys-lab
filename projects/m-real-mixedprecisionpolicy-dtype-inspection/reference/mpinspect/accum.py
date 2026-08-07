import numpy as np

def simulate_accumulation(steps, dtype_mode):
    """Simulate numerical accumulation error."""
    val = 0.0
    for _ in range(steps):
        if dtype_mode == "bf16":
            val = np.float32(val + np.float32(0.1))
        else:
            val += 0.1
    return float(val)
