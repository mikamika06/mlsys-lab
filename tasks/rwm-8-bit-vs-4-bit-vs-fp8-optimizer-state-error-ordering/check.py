import numpy as np

from mlsys import scorers


def _gen_state(seed: int = 0, n: int = 4096, T: int = 50, beta2: float = 0.999) -> np.ndarray:
    """Deterministic Adam-style second-moment EMA over synthetic gradients."""
    rng = np.random.default_rng(seed)
    v = np.zeros(n, dtype=np.float64)
    for _ in range(T):
        g = rng.standard_normal(n)
        v = beta2 * v + (1.0 - beta2) * g * g
    return v


def _oracle_8bit(x: np.ndarray, block_size: int) -> tuple[np.ndarray, int]:
    n = x.shape[0]
    nb = n // block_size
    xb = x.reshape(nb, block_size)
    scales = np.max(np.abs(xb), axis=1) / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    codes = np.round(xb / scales[:, None])
    codes = np.clip(codes, -127, 127).astype(np.int8)
    xhat = (codes.astype(np.float64) * scales[:, None]).reshape(n)
    nbytes = int(codes.nbytes + scales.astype(np.float32).nbytes)
    return xhat, nbytes


def _oracle_4bit(x: np.ndarray, block_size: int) -> tuple[np.ndarray, int]:
    n = x.shape[0]
    nb = n // block_size
    xb = x.reshape(nb, block_size)
    scales = np.max(np.abs(xb), axis=1) / 7.0
    scales = np.where(scales == 0, 1.0, scales)
    codes = np.round(xb / scales[:, None])
    codes = np.clip(codes, -7, 7).astype(np.int64)
    offset = (codes + 8).astype(np.uint8).reshape(n)
    low = offset[0::2]
    high = offset[1::2]
    packed = (low | (high << 4)).astype(np.uint8)

    low2 = (packed & 0x0F).astype(np.int64) - 8
    high2 = ((packed >> 4) & 0x0F).astype(np.int64) - 8
    unpacked = np.empty(n, dtype=np.int64)
    unpacked[0::2] = low2
    unpacked[1::2] = high2
    xhat = (unpacked.reshape(nb, block_size).astype(np.float64) * scales[:, None]).reshape(n)
    nbytes = int(packed.nbytes + scales.astype(np.float32).nbytes)
    return xhat, nbytes


def _oracle_fp8(x: np.ndarray, mantissa_bits: int = 3, bias: int = 6) -> tuple[np.ndarray, int]:
    n = x.shape[0]
    sign = np.sign(x)
    m = np.abs(x)

    e_min = -bias
    e_max = (2 ** 4 - 1) - bias - 1
    m_min = 2.0 ** e_min
    m_max = (2.0 - 2.0 ** -mantissa_bits) * 2.0 ** e_max

    m_clamped = np.clip(m, m_min, m_max)
    e = np.floor(np.log2(m_clamped)).astype(np.int64)
    e = np.clip(e, e_min, e_max)
    scale2 = np.exp2(e.astype(np.float64))
    frac = m_clamped / scale2
    frac_q = np.round(frac * (2 ** mantissa_bits)) / (2 ** mantissa_bits)

    overflow = (frac_q >= 2.0) & (e < e_max)
    frac_q = np.where(overflow, frac_q / 2.0, frac_q)
    e = np.where(overflow, e + 1, e)
    frac_q = np.clip(frac_q, 1.0, 2.0 - 2.0 ** -mantissa_bits)

    recon = sign * frac_q * np.exp2(e.astype(np.float64))
    recon = np.where(x == 0, 0.0, recon)
    nbytes = int(n * 1)
    return recon, nbytes


def _oracle_compare(v: np.ndarray, block_size: int) -> dict:
    xhat8, b8 = _oracle_8bit(v, block_size)
    xhat4, b4 = _oracle_4bit(v, block_size)
    xhatf, bf = _oracle_fp8(v)
    return {
        "mse_8bit": float(np.mean((v - xhat8) ** 2)),
        "mse_4bit": float(np.mean((v - xhat4) ** 2)),
        "mse_fp8": float(np.mean((v - xhatf) ** 2)),
        "bytes_8bit": b8,
        "bytes_4bit": b4,
        "bytes_fp8": bf,
    }


def grade(sol, fx) -> dict:
    block_size = 32
    v = _gen_state(seed=0, n=4096, T=50, beta2=0.999)
    oracle = _oracle_compare(v, block_size)

    keys_mse = ["mse_8bit", "mse_4bit", "mse_fp8"]
    keys_bytes = ["bytes_8bit", "bytes_4bit", "bytes_fp8"]

    try:
        got = sol.optimizer_state_quant_compare(np.array(v, dtype=np.float64), block_size)
        got_mse = np.array([float(got[k]) for k in keys_mse], dtype=np.float64)
        got_bytes = [int(got[k]) for k in keys_bytes]
    except Exception:
        return {"rel_err": float("inf"), "order_match": 0.0, "bytes_ok": 0.0}

    oracle_mse = np.array([oracle[k] for k in keys_mse], dtype=np.float64)
    oracle_bytes = [oracle[k] for k in keys_bytes]

    rel = scorers.rel_err(oracle_mse, got_mse)

    order_oracle = np.argsort(oracle_mse)
    order_got = np.argsort(got_mse)
    order_match = 1.0 if np.array_equal(order_oracle, order_got) else 0.0

    bytes_ok = 1.0 if got_bytes == oracle_bytes else 0.0

    return {"rel_err": rel, "order_match": order_match, "bytes_ok": bytes_ok}
