import numpy as np


def dequantize_nf4(qweight, absmax, codebook, block_size=64):
    qweight = np.asarray(qweight, dtype=np.int32)
    absmax = np.asarray(absmax, dtype=np.float32)
    codebook = np.asarray(codebook, dtype=np.float32)

    num_blocks = len(absmax)
    flattened_indices = qweight.reshape(num_blocks, block_size)
    dequantized_blocks = codebook[flattened_indices] * absmax[:, None]
    return dequantized_blocks.reshape(-1)


def lora_nf4_forward(x, qweight, absmax, codebook, lora_a, lora_b, scaling, compute_dtype="float32", block_size=64):
    target_dt = np.dtype(compute_dtype)
    x = np.asarray(x, dtype=target_dt)

    w_flat = dequantize_nf4(qweight, absmax, codebook, block_size=block_size)
    out_features, in_features = lora_b.shape[0], lora_a.shape[1]
    w_dequant = w_flat.reshape(out_features, in_features).astype(target_dt)

    base_out = x @ w_dequant.T

    lora_a = np.asarray(lora_a, dtype=target_dt)
    lora_b = np.asarray(lora_b, dtype=target_dt)

    adapter_out = (x @ lora_a.T) @ lora_b.T
    return base_out + target_dt.type(scaling) * adapter_out
