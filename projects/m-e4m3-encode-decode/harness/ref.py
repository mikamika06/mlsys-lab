import numpy as np

E4M3_MAX = 448.0
E4M3_BIAS = 7


def _build_e4m3_table():
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0x0F
        mant = b & 0x07
        if exp == 0:
            val = sign * (2.0 ** (-6)) * (mant / 8.0)
        elif exp == 15 and mant == 7:
            val = np.nan
        else:
            val = sign * (2.0 ** (exp - E4M3_BIAS)) * (1.0 + mant / 8.0)
        lut[b] = val
    return lut


_E4M3_LUT = _build_e4m3_table()
_E4M3_FINITE_MASK = ~np.isnan(_E4M3_LUT)
_E4M3_VALS = _E4M3_LUT[_E4M3_FINITE_MASK]
_E4M3_BYTES = np.arange(256, dtype=np.uint8)[_E4M3_FINITE_MASK]


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    flat_x = np.asarray(x, dtype=np.float32).ravel()
    nan_mask = np.isnan(flat_x)
    diffs = np.abs(flat_x[:, None] - _E4M3_VALS[None, :])
    nearest_idx = np.argmin(diffs, axis=1)
    out_bytes = _E4M3_BYTES[nearest_idx]
    out_bytes[nan_mask] = 0x7F
    return out_bytes.reshape(x.shape)


def decode_e4m3(b: np.ndarray) -> np.ndarray:
    b_arr = np.asarray(b, dtype=np.uint8)
    return _E4M3_LUT[b_arr].reshape(b_arr.shape)


def quantize_and_descale(
    x: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    scaled_x = x * scale
    clamped_x = np.clip(scaled_x, -E4M3_MAX, E4M3_MAX)
    q_bytes = encode_e4m3(clamped_x)
    deq = decode_e4m3(q_bytes)
    reconstructed = deq / scale
    return q_bytes, reconstructed


def compute_scale(x: np.ndarray) -> float:
    max_val = float(np.max(np.abs(x)))
    if max_val == 0.0:
        return 1.0
    return E4M3_MAX / max_val


def find_optimal_scale(
    x: np.ndarray, candidates: list[float]
) -> tuple[float, float]:
    best_scale = float(candidates[0])
    best_mse = float("inf")
    x_f32 = np.asarray(x, dtype=np.float32)
    for s in candidates:
        _, recon = quantize_and_descale(x_f32, float(s))
        mse = float(np.mean((x_f32 - recon) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_scale = float(s)
    return best_scale, best_mse


def generate_test_tensors():
    np.random.seed(42)
    t1 = np.random.randn(16, 32).astype(np.float32) * 50.0
    t2 = np.random.uniform(-500.0, 500.0, size=(8, 16)).astype(np.float32)
    t3 = np.zeros((4, 4), dtype=np.float32)
    return [t1, t2, t3]
