import numpy as np

def generate_tensor():
    np.random.seed(42)
    return np.random.randn(256, 256).astype(np.float32)

def measure_footprint(tensor, block_size=64):
    arr = np.asarray(tensor, dtype=np.float32)
    orig_bytes = arr.nbytes
    numel = arr.size
    n_blocks = (numel + block_size - 1) // block_size
    data_bytes = (numel + 1) // 2
    absmax_bytes = n_blocks * 2
    quant_bytes = data_bytes + absmax_bytes
    return {"orig_bytes": orig_bytes, "quant_bytes": quant_bytes}

def compare_nf4_uniform(tensor, block_size=64):
    arr = np.asarray(tensor, dtype=np.float32)
    nf4_levels = np.array([
        -1.0, -0.6961928, -0.52507305, -0.39491748,
        -0.28444138, -0.18477343, -0.09105004, 0.0,
        0.0795803, 0.1609302, 0.2461123, 0.33791524,
        0.44070983, 0.562617, 0.72295684, 1.0
    ], dtype=np.float32)
    uniform_levels = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    max_val = np.max(np.abs(arr)) + 1e-8
    norm_arr = arr / max_val
    nf4_diffs = np.abs(norm_arr[..., None] - nf4_levels)
    nf4_idx = np.argmin(nf4_diffs, axis=-1)
    nf4_recon = nf4_levels[nf4_idx] * max_val
    nf4_mse = np.mean((arr - nf4_recon) ** 2)
    uni_diffs = np.abs(norm_arr[..., None] - uniform_levels)
    uni_idx = np.argmin(uni_diffs, axis=-1)
    uni_recon = uniform_levels[uni_idx] * max_val
    uni_mse = np.mean((arr - uni_recon) ** 2)
    return {"nf4_mse": float(nf4_mse), "uniform_mse": float(uni_mse), "nf4_beats_uniform": bool(nf4_mse < uni_mse)}

def double_quant_size(tensor, block_size=64, second_block_size=256):
    arr = np.asarray(tensor, dtype=np.float32)
    numel = arr.size
    n_blocks = (numel + block_size - 1) // block_size
    n_sec_blocks = (n_blocks + second_block_size - 1) // second_block_size
    absmax_data_bytes = n_blocks * 1
    absmax_scales_bytes = n_sec_blocks * 4
    total_double_absmax_bytes = absmax_data_bytes + absmax_scales_bytes
    standard_absmax_bytes = n_blocks * 4
    return {
        "standard_absmax_bytes": int(standard_absmax_bytes),
        "double_absmax_bytes": int(total_double_absmax_bytes),
        "savings_ratio": float(total_double_absmax_bytes / standard_absmax_bytes)
    }
