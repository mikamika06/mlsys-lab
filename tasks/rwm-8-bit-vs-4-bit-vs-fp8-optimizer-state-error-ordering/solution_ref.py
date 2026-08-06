import math
import numpy as np


def _quant_8bit(x: np.ndarray, block_size: int) -> tuple[np.ndarray, int]:
    n = x.shape[0]
    nb = n // block_size
    codes_arr = np.empty(n, dtype=np.int8)
    scales_arr = np.empty(nb, dtype=np.float32)
    xhat_arr = np.empty(n, dtype=np.float64)

    for i in range(nb):
        start = i * block_size
        end = start + block_size
        max_abs = 0.0
        for j in range(start, end):
            val = x[j]
            a = abs(val)
            if a > max_abs:
                max_abs = a
        scale = max_abs / 127.0
        if scale == 0.0:
            scale = 1.0
        scales_arr[i] = scale

        for j in range(start, end):
            val = x[j]
            c = round(val / scale)
            if c < -127:
                c = -127
            elif c > 127:
                c = 127
            codes_arr[j] = c
            xhat_arr[j] = float(c) * scale

    nbytes = int(codes_arr.nbytes + scales_arr.nbytes)
    return xhat_arr, nbytes


def _quant_4bit(x: np.ndarray, block_size: int) -> tuple[np.ndarray, int]:
    n = x.shape[0]
    nb = n // block_size
    scales_arr = np.empty(nb, dtype=np.float32)
    codes_arr = np.empty(n, dtype=np.int64)

    for i in range(nb):
        start = i * block_size
        end = start + block_size
        max_abs = 0.0
        for j in range(start, end):
            val = x[j]
            a = abs(val)
            if a > max_abs:
                max_abs = a
        scale = max_abs / 7.0
        if scale == 0.0:
            scale = 1.0
        scales_arr[i] = scale

        for j in range(start, end):
            val = x[j]
            c = round(val / scale)
            if c < -7:
                c = -7
            elif c > 7:
                c = 7
            codes_arr[j] = c

    offset_arr = np.empty(n, dtype=np.uint8)
    for j in range(n):
        offset_arr[j] = int(codes_arr[j] + 8) & 0xFF

    n_packed = n // 2
    packed_arr = np.empty(n_packed, dtype=np.uint8)
    for k in range(n_packed):
        low_val = offset_arr[2 * k]
        high_val = offset_arr[2 * k + 1]
        packed_arr[k] = (low_val | (high_val << 4)) & 0xFF

    unpacked_arr = np.empty(n, dtype=np.int64)
    for k in range(n_packed):
        p = packed_arr[k]
        low2 = (p & 0x0F) - 8
        high2 = ((p >> 4) & 0x0F) - 8
        unpacked_arr[2 * k] = low2
        unpacked_arr[2 * k + 1] = high2

    xhat_arr = np.empty(n, dtype=np.float64)
    for i in range(nb):
        start = i * block_size
        end = start + block_size
        scale = float(scales_arr[i])
        for j in range(start, end):
            xhat_arr[j] = float(unpacked_arr[j]) * scale

    nbytes = int(packed_arr.nbytes + scales_arr.nbytes)
    return xhat_arr, nbytes


def _quant_fp8(x: np.ndarray, mantissa_bits: int = 3, bias: int = 6) -> tuple[np.ndarray, int]:
    n = x.shape[0]
    e_min = -bias
    e_max = (2 ** 4 - 1) - bias - 1
    m_min = 2.0 ** e_min
    max_frac = 2.0 - (2.0 ** -mantissa_bits)
    m_max = max_frac * (2.0 ** e_max)
    mult_factor = 2.0 ** mantissa_bits

    recon = np.empty(n, dtype=np.float64)
    for i in range(n):
        val = x[i]
        if val == 0.0:
            recon[i] = 0.0
            continue

        if val > 0.0:
            sign = 1.0
        else:
            sign = -1.0

        m = abs(val)

        if m < m_min:
            m_clamped = m_min
        elif m > m_max:
            m_clamped = m_max
        else:
            m_clamped = m

        e = math.floor(math.log2(m_clamped))
        if e < e_min:
            e = e_min
        elif e > e_max:
            e = e_max

        scale2 = 2.0 ** e
        frac = m_clamped / scale2
        frac_q = round(frac * mult_factor) / mult_factor

        if frac_q >= 2.0 and e < e_max:
            frac_q = frac_q / 2.0
            e = e + 1

        if frac_q < 1.0:
            frac_q = 1.0
        elif frac_q > max_frac:
            frac_q = max_frac

        recon[i] = sign * frac_q * (2.0 ** e)

    nbytes = int(n * 1)
    return recon, nbytes


def optimizer_state_quant_compare(v: np.ndarray, block_size: int = 32) -> dict:
    """
    Quantize `v` with 8-bit blockwise, 4-bit blockwise (nibble-packed), and
    fp8-style formats; return reconstruction MSE and storage bytes for each.
    """
    v = np.asarray(v, dtype=np.float64)

    xhat8, b8 = _quant_8bit(v, block_size)
    xhat4, b4 = _quant_4bit(v, block_size)
    xhatf, bf = _quant_fp8(v)

    n = v.shape[0]
    sum_sq_8 = 0.0
    sum_sq_4 = 0.0
    sum_sq_f = 0.0
    for i in range(n):
        d8 = v[i] - xhat8[i]
        sum_sq_8 += d8 * d8
        d4 = v[i] - xhat4[i]
        sum_sq_4 += d4 * d4
        df = v[i] - xhatf[i]
        sum_sq_f += df * df

    return {
        "mse_8bit": float(sum_sq_8 / n),
        "mse_4bit": float(sum_sq_4 / n),
        "mse_fp8": float(sum_sq_f / n),
        "bytes_8bit": b8,
        "bytes_4bit": b4,
        "bytes_fp8": bf,
    }
