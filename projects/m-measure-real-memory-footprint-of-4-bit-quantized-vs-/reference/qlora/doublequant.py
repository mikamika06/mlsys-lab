import numpy as np


def double_quantize_absmax(absmax_tensor, inner_block_size=256):
    flat = absmax_tensor.astype(np.float32).flatten()
    numel = flat.size
    num_outer_blocks = (numel + inner_block_size - 1) // inner_block_size
    padded_size = num_outer_blocks * inner_block_size
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[:numel] = flat
    blocks = padded.reshape(-1, inner_block_size)

    meta_absmax = np.max(np.abs(blocks), axis=1)
    meta_absmax = np.where(meta_absmax == 0, 1e-5, meta_absmax)

    normalized = blocks / meta_absmax[:, None]
    quantized_fp8 = np.clip(np.round(normalized * 127.0), -128, 127).astype(np.int8)

    fp8_bytes = quantized_fp8.size * 1
    meta_scale_bytes = num_outer_blocks * 4
    total_dq_bytes = fp8_bytes + meta_scale_bytes
    orig_absmax_bytes = numel * 4

    return {
        "orig_bytes": int(orig_absmax_bytes),
        "dq_bytes": int(total_dq_bytes),
        "compression_ratio": float(orig_absmax_bytes / total_dq_bytes),
    }
