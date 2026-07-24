import numpy as np


def _nf4_codebook() -> np.ndarray:
    rng = np.random.RandomState(0)
    sample = rng.randn(1_000_000)
    q = (np.arange(16) + 0.5) / 16
    return np.quantile(sample, q)


_CODEBOOK = _nf4_codebook()


def _nf4_quantize(flat: np.ndarray, block_size: int):
    n = len(flat)
    n_blocks = int(np.ceil(n / block_size))
    codes = np.empty(n, dtype=np.uint8)
    c1 = np.empty(n_blocks, dtype=np.float64)
    for bi, i in enumerate(range(0, n, block_size)):
        blk = flat[i:i + block_size]
        m = float(np.max(np.abs(blk)))
        if m == 0.0:
            m = 1.0
        c1[bi] = m
        y = blk / m
        idx = np.abs(y[:, None] - _CODEBOOK[None, :]).argmin(axis=1)
        codes[i:i + block_size] = idx.astype(np.uint8)
    return codes, c1


def _affine8_quant_dequant(x: np.ndarray, outer_block: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, outer_block):
        grp = x[i:i + outer_block]
        lo, hi = float(np.min(grp)), float(np.max(grp))
        scale = (hi - lo) / 255.0
        if scale == 0.0:
            scale = 1.0
        zp = round(-lo / scale)
        code = np.clip(np.round(grp / scale + zp), 0, 255)
        out[i:i + outer_block] = (code - zp) * scale
    return out


def nf4_double_quant_dequant(weights: np.ndarray, block_size: int, outer_block: int):
    """
    QLoRA-style double quantization.

    weights: array of any shape.
    block_size: level-1 NF4 block size (number of weight elements sharing
        one fp32 absmax constant c1).
    outer_block: number of consecutive c1 values grouped together for
        the level-2 8-bit (asymmetric, min-max) blockwise quantization of
        the absmax constants themselves.

    Level 1: split `weights` into blocks of `block_size`; each block's
    absmax c1 = max(|block|) (or 1.0 for an all-zero block) normalizes
    the block before snapping every value to the 16-level NF4 codebook.

    Level 2: instead of storing each c1 as a full fp32, group every
    `outer_block` consecutive c1 values and quantize/dequantize THAT
    group with asymmetric (affine) min-max 8-bit quantization, needing
    just one more fp32 scale per group.

    Returns (reconstructed_weights, bits_per_param) where
    bits_per_param = 4 + 8/block_size + 32/(block_size*outer_block)
    (4 bits per NF4 code, an 8-bit c1 code per block_size weights, and a
    32-bit c2 scale per block_size*outer_block weights).
    """
    weights = np.asarray(weights, dtype=np.float64)
    flat = weights.ravel()
    n = len(flat)

    codes, c1 = _nf4_quantize(flat, block_size)
    c1_hat = _affine8_quant_dequant(c1, outer_block)

    block_idx = np.arange(n) // block_size
    recon = (_CODEBOOK[codes] * c1_hat[block_idx]).reshape(weights.shape)

    bits_per_param = 4.0 + 8.0 / block_size + 32.0 / (block_size * outer_block)
    return recon, bits_per_param
