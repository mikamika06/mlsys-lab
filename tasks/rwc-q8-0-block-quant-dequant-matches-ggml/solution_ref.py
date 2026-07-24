import numpy as np

def q8_0_quantize(x):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    block_size = 32
    num_blocks = n // block_size
    codes = np.empty_like(x, dtype=np.int8)
    scales = np.empty(num_blocks, dtype=np.float16)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        block = x[start:end]
        absmax = np.max(np.abs(block))
        d = absmax / 127.0 if absmax != 0 else 0.0
        scales[b] = np.float16(d)
        c = np.round(block / d) if d != 0 else np.zeros_like(block)
        c = np.clip(c, -127, 127).astype(np.int8)
        codes[start:end] = c
    return codes, scales

def q8_0_dequantize(codes, scales):
    block_size = 32
    num_blocks = len(scales)
    x_hat = np.empty_like(codes, dtype=np.float32)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        c_block = codes[start:end].astype(np.int8).astype(np.float32)
        d = scales[b].astype(np.float32)
        x_hat[start:end] = c_block * d
    return x_hat
