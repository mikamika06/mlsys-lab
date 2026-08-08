import numpy as np

E4M3_MAX = 448.0
E4M3_BIAS = 7


def _build_e4m3_lut():
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


_E4M3_LUT = _build_e4m3_lut()
_E4M3_FINITE_MASK = ~np.isnan(_E4M3_LUT)
_E4M3_VALS = _E4M3_LUT[_E4M3_FINITE_MASK]
_E4M3_BYTES = np.arange(256, dtype=np.uint8)[_E4M3_FINITE_MASK]


def decode_e4m3(b: np.ndarray) -> np.ndarray:
    b_arr = np.asarray(b, dtype=np.uint8)
    return _E4M3_LUT[b_arr].reshape(b_arr.shape)


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    flat_x = np.asarray(x, dtype=np.float32).ravel()
    nan_mask = np.isnan(flat_x)
    diffs = np.abs(flat_x[:, None] - _E4M3_VALS[None, :])
    nearest_idx = np.argmin(diffs, axis=1)
    out_bytes = _E4M3_BYTES[nearest_idx]
    out_bytes[nan_mask] = 0x7F
    return out_bytes.reshape(x.shape)
