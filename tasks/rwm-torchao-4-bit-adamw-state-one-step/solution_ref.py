import math
import numpy as np


def _quant4(x: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    nb = n // block_size
    scales_list = []
    offsets_list = []
    for b in range(nb):
        max_abs = 0.0
        for i in range(block_size):
            val = abs(float(x[b * block_size + i]))
            if val > max_abs:
                max_abs = val
        scale = max_abs / 7.0
        if scale == 0.0:
            scale = 1.0
        scales_list.append(scale)
        for i in range(block_size):
            val = float(x[b * block_size + i])
            code = round(val / scale)
            if code < -7:
                code = -7
            elif code > 7:
                code = 7
            offset = int(code + 8)
            offsets_list.append(offset)
    scales = np.array(scales_list, dtype=np.float32)
    packed_list = []
    for i in range(0, n, 2):
        low = offsets_list[i]
        high = offsets_list[i + 1]
        packed_val = low | (high << 4)
        packed_list.append(packed_val)
    packed = np.array(packed_list, dtype=np.uint8)
    return packed, scales


def _dequant4(packed: np.ndarray, scales: np.ndarray, block_size: int) -> np.ndarray:
    nb = scales.shape[0]
    n = nb * block_size
    unpacked = [0] * n
    for idx, byte in enumerate(packed):
        low = int(byte & 0x0F) - 8
        high = int((byte >> 4) & 0x0F) - 8
        unpacked[2 * idx] = low
        unpacked[2 * idx + 1] = high
    
    xhat_list = []
    for b in range(nb):
        scale = float(scales[b])
        for i in range(block_size):
            val = float(unpacked[b * block_size + i]) * scale
            xhat_list.append(val)
    return np.array(xhat_list, dtype=np.float64)


def adamw_4bit_step(
    p: np.ndarray, grad: np.ndarray,
    m_packed: np.ndarray, m_scales: np.ndarray,
    v_packed: np.ndarray, v_scales: np.ndarray,
    step: int, block_size: int = 32,
    lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
    eps: float = 1e-8, weight_decay: float = 0.01,
) -> dict:
    """Dequantize m,v (4-bit blockwise), take one AdamW step, requantize m,v."""
    p_arr = np.asarray(p, dtype=np.float64)
    grad_arr = np.asarray(grad, dtype=np.float64)
    n = p_arr.shape[0]

    m_prev = _dequant4(m_packed, m_scales, block_size)
    v_prev = _dequant4(v_packed, v_scales, block_size)

    beta1_pow_step = beta1 ** step
    beta2_pow_step = beta2 ** step
    denom1 = 1.0 - beta1_pow_step
    denom2 = 1.0 - beta2_pow_step
    weight_mult = 1.0 - lr * weight_decay

    m_list = []
    v_list = []
    p_new_list = []

    for i in range(n):
        mp = float(m_prev[i])
        vp = float(v_prev[i])
        g = float(grad_arr[i])
        pv = float(p_arr[i])

        m_val = beta1 * mp + (1.0 - beta1) * g
        v_val = beta2 * vp + (1.0 - beta2) * g * g

        m_list.append(m_val)
        v_list.append(v_val)

        m_hat = m_val / denom1
        v_hat = v_val / denom2

        update = lr * m_hat / (math.sqrt(v_hat) + eps)
        p_new_val = pv * weight_mult - update
        p_new_list.append(p_new_val)

    m = np.array(m_list, dtype=np.float64)
    v = np.array(v_list, dtype=np.float64)
    p_new = np.array(p_new_list, dtype=np.float64)

    m_packed_new, m_scales_new = _quant4(m, block_size)
    v_packed_new, v_scales_new = _quant4(v, block_size)

    return {
        "p_new": p_new,
        "m_packed": m_packed_new,
        "m_scales": m_scales_new,
        "v_packed": v_packed_new,
        "v_scales": v_scales_new,
    }
