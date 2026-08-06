import numpy as np

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
