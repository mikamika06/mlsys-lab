import numpy as np


def round_e2m1(x: np.ndarray) -> np.ndarray:
    magnitudes = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    sign = np.sign(x)
    sign[sign == 0] = 1.0
    abs_x = np.abs(x)
    diff = np.abs(abs_x[..., None] - magnitudes)
    idx = np.argmin(diff, axis=-1)
    return sign * magnitudes[idx]


def mxfp4(x: np.ndarray, block_size: int = 32) -> np.ndarray:
    blocks = x.reshape(-1, block_size)
    m = np.max(np.abs(blocks), axis=1, keepdims=True)
    m = np.maximum(m, 1e-12)
    s = 2.0 ** np.ceil(np.log2(m / 6.0))
    scaled = blocks / s
    q = round_e2m1(scaled)
    return (q * s).reshape(x.shape)


def nvfp4(x: np.ndarray, block_size: int = 16, super_block: int = 256) -> np.ndarray:
    sup_blocks = x.reshape(-1, super_block)
    m_sup = np.max(np.abs(sup_blocks), axis=1, keepdims=True)
    m_sup = np.maximum(m_sup, 1e-12)
    s_sup = 2.0 ** np.ceil(np.log2(m_sup / 6.0))
    scaled_sup = sup_blocks / s_sup

    blocks = scaled_sup.reshape(-1, block_size)
    m_blk = np.max(np.abs(blocks), axis=1, keepdims=True)
    m_blk = np.maximum(m_blk, 1e-12)

    log2_s = np.ceil(np.log2(m_blk / 6.0))
    log2_s = np.clip(log2_s, -15, 0)
    s_vec = 2.0 ** log2_s

    scaled_blk = blocks / s_vec
    q = round_e2m1(scaled_blk)

    out = (q * s_vec).reshape(sup_blocks.shape) * s_sup
    return out.reshape(x.shape)


def effective_bits(fmt: str) -> float:
    if fmt == "mxfp4":
        return 4.0 + 8.0 / 32.0
    elif fmt == "nvfp4":
        return 4.0 + 4.0 / 16.0 + 8.0 / 256.0
    raise ValueError("Unknown format")
