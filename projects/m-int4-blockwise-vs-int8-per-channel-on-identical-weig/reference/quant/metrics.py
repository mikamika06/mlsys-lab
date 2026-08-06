import numpy as np


def compute_mse(w_orig: np.ndarray, w_dequant: np.ndarray) -> float:
    return float(np.mean((w_orig.astype(np.float32) - w_dequant.astype(np.float32)) ** 2))


def compute_bit_size(w_shape: tuple[int, int], mode: str, block_size: int = 32) -> int:
    out_ch, in_ch = w_shape
    total_elements = out_ch * in_ch

    if mode == "int8_per_channel":
        weight_bits = total_elements * 8
        scale_bits = out_ch * 32
        return weight_bits + scale_bits
    elif mode == "int4_blockwise":
        weight_bits = total_elements * 4
        num_blocks_per_row = (in_ch + block_size - 1) // block_size
        scale_bits = out_ch * num_blocks_per_row * 16
        return weight_bits + scale_bits
    else:
        raise ValueError(f"Unknown mode: {mode}")
