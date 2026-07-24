import numpy as np


def _quant4(x: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    nb = n // block_size
    xb = x.reshape(nb, block_size)
    scales = np.max(np.abs(xb), axis=1) / 7.0
    scales = np.where(scales == 0, 1.0, scales)
    codes = np.round(xb / scales[:, None])
    codes = np.clip(codes, -7, 7).astype(np.int64)
    offset = (codes + 8).astype(np.uint8).reshape(n)  # in [1, 15]
    low = offset[0::2]
    high = offset[1::2]
    packed = (low | (high << 4)).astype(np.uint8)
    return packed, scales.astype(np.float32)


def _dequant4(packed: np.ndarray, scales: np.ndarray, block_size: int) -> np.ndarray:
    nb = scales.shape[0]
    n = nb * block_size
    low2 = (packed & 0x0F).astype(np.int64) - 8
    high2 = ((packed >> 4) & 0x0F).astype(np.int64) - 8
    unpacked = np.empty(n, dtype=np.int64)
    unpacked[0::2] = low2
    unpacked[1::2] = high2
    xhat = (unpacked.reshape(nb, block_size).astype(np.float64)
            * scales.astype(np.float64)[:, None]).reshape(n)
    return xhat


def adamw_4bit_step(
    p: np.ndarray, grad: np.ndarray,
    m_packed: np.ndarray, m_scales: np.ndarray,
    v_packed: np.ndarray, v_scales: np.ndarray,
    step: int, block_size: int = 32,
    lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
    eps: float = 1e-8, weight_decay: float = 0.01,
) -> dict:
    """Dequantize m,v (4-bit blockwise), take one AdamW step, requantize m,v."""
    p = np.asarray(p, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)

    m_prev = _dequant4(m_packed, m_scales, block_size)
    v_prev = _dequant4(v_packed, v_scales, block_size)

    m = beta1 * m_prev + (1.0 - beta1) * grad
    v = beta2 * v_prev + (1.0 - beta2) * grad * grad

    m_hat = m / (1.0 - beta1 ** step)
    v_hat = v / (1.0 - beta2 ** step)

    update = lr * m_hat / (np.sqrt(v_hat) + eps)
    p_new = p * (1.0 - lr * weight_decay) - update

    m_packed_new, m_scales_new = _quant4(m, block_size)
    v_packed_new, v_scales_new = _quant4(v, block_size)

    return {
        "p_new": p_new,
        "m_packed": m_packed_new,
        "m_scales": m_scales_new,
        "v_packed": v_packed_new,
        "v_scales": v_scales_new,
    }
