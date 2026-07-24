import numpy as np


def double_quant_nf4(W, block_size=64):
    codebook = np.array(
        [-1.0, -0.6961928, -0.52507305, -0.3949175,
         -0.28444138, -0.18477343, -0.09105004, 0.0,
         0.0795803, 0.1609302, 0.2461123, 0.33791524,
         0.44070983, 0.562617, 0.72295684, 1.0],
        dtype=np.float64,
    )

    shape = W.shape
    flat = np.asarray(W, dtype=np.float64).ravel()
    n = flat.size
    padded = int(np.ceil(n / block_size) * block_size)
    x = np.zeros(padded, dtype=np.float64)
    x[:n] = flat

    blocks = x.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1)
    scales[scales == 0] = 1.0

    normalized = blocks / scales[:, None]
    codes = np.argmin(
        np.abs(normalized[:, :, None] - codebook[None, None, :]),
        axis=2,
    )

    max_scale = np.max(scales)
    scale_codes = np.rint(scales / max_scale * 255).astype(np.uint8)
    dequant_scales = scale_codes.astype(np.float64) * max_scale / 255.0

    out = (codebook[codes] * dequant_scales[:, None]).reshape(-1)[:n]
    bits = float(4.0 + 8.0 / block_size + 32.0 / (block_size * 256.0))
    return out.reshape(shape), bits
