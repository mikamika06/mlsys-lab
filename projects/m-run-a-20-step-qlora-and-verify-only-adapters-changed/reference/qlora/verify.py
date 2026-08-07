import numpy as np

def verify_adapters_changed(initial_model, final_model):
    base_diff = np.max(np.abs(initial_model["base_weight"] - final_model["base_weight"]))
    a_diff = np.max(np.abs(initial_model["lora_a"] - final_model["lora_a"]))
    b_diff = np.max(np.abs(initial_model["lora_b"] - final_model["lora_b"]))
    base_ok = base_diff < 1e-7
    adapters_ok = (a_diff > 1e-7) or (b_diff > 1e-7)
    return bool(base_ok and adapters_ok)
