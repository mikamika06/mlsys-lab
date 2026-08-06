import os
import numpy as np

def get_sample_data():
    rng = np.random.default_rng(42)
    base = {"layer.weight": rng.standard_normal((64, 64), dtype=np.float32)}
    adapter_A = {"layer.lora_A": rng.standard_normal((8, 64), dtype=np.float32)}
    adapter_B = {"layer.lora_B": rng.standard_normal((64, 8), dtype=np.float32)}
    return base, adapter_A, adapter_B

def measure_file_sizes(base_dict, adapter_dict, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    b_path = os.path.join(tmp_dir, "base.npz")
    a_path = os.path.join(tmp_dir, "adapter.npz")
    np.savez(b_path, **base_dict)
    np.savez(a_path, **adapter_dict)
    b_size = os.path.getsize(b_path)
    a_size = os.path.getsize(a_path)

    merged = {"layer.weight": base_dict["layer.weight"] + np.matmul(adapter_dict["layer.lora_B"], adapter_dict["layer.lora_A"])}
    m_path = os.path.join(tmp_dir, "merged.npz")
    np.savez(m_path, **merged)
    m_size = os.path.getsize(m_path)
    return {"base_size": b_size, "adapter_size": a_size, "merged_size": m_size, "size_ratio": m_size / (b_size + a_size)}

def simulate_quantization(weights, num_bits=4):
    levels = 2 ** num_bits
    min_val, max_val = np.min(weights), np.max(weights)
    if min_val == max_val:
        return weights, 0.0
    scaled = (weights - min_val) / (max_val - min_val)
    quantized = np.round(scaled * (levels - 1)) / (levels - 1)
    dequantized = quantized * (max_val - min_val) + min_val
    mse = float(np.mean((weights - dequantized) ** 2))
    return dequantized, mse

def compute_quantization_error(base_dict, adapter_dict):
    merged = base_dict["layer.weight"] + np.matmul(adapter_dict["layer.lora_B"], adapter_dict["layer.lora_A"])
    _, err_merged = simulate_quantization(merged, num_bits=4)
    deq_base, _ = simulate_quantization(base_dict["layer.weight"], num_bits=4)
    deq_adapter, _ = simulate_quantization(np.matmul(adapter_dict["layer.lora_B"], adapter_dict["layer.lora_A"]), num_bits=4)
    summed_approx = deq_base + deq_adapter
    err_summed = float(np.mean((merged - summed_approx) ** 2))
    return {"error_merged": err_merged, "error_summed": err_summed, "diff": err_merged - err_summed}

def merge_and_unload(state_dict):
    new_dict = {}
    lora_prod = None
    if "layer.lora_A" in state_dict and "layer.lora_B" in state_dict:
        lora_prod = np.matmul(state_dict["layer.lora_B"], state_dict["layer.lora_A"])
    for k, v in state_dict.items():
        if "lora" in k or "adapter" in k:
            continue
        if k == "layer.weight" and lora_prod is not None:
            new_dict[k] = v + lora_prod
        else:
            new_dict[k] = v
    return new_dict
