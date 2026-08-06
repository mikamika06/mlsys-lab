import numpy as np

def nested_absmax_quantize(tensor, block_size=256):
    flat = tensor.astype(np.float64)
    n = len(flat)
    padded_len = ((n + block_size - 1) // block_size) * block_size
    padded = np.pad(flat, (0, padded_len - n), mode='constant')
    blocks = padded.reshape(-1, block_size)
    absmaxs = np.max(np.abs(blocks), axis=1)
    absmaxs_scale = np.max(np.abs(absmaxs))
    if absmaxs_scale == 0:
        absmaxs_quantized = np.zeros_like(absmaxs)
        scale2 = 1.0
    else:
        scale2 = absmaxs_scale / 127.0
        absmaxs_quantized = np.round(absmaxs / scale2).clip(-127, 127)
    absmaxs_dequant = absmaxs_quantized * scale2
    quantized_blocks = np.zeros_like(blocks)
    for i in range(len(blocks)):
        scale1 = absmaxs_dequant[i]
        if scale1 == 0:
            quantized_blocks[i] = 0
        else:
            q = np.round(blocks[i] / (scale1 / 7.0)).clip(-7, 7)
            quantized_blocks[i] = q * (scale1 / 7.0)
    dequant = quantized_blocks.flatten()[:n]
    return dequant
