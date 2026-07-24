import numpy as np


def _e4m3_values():
    vals = []
    for bits in range(256):
        sign = -1.0 if bits & 0x80 else 1.0
        exp = (bits >> 3) & 0x0F
        mant = bits & 0x07
        if exp == 0:
            vals.append(sign * (mant / 8.0) * 2 ** -6)
        elif exp < 15:
            vals.append(sign * (1.0 + mant / 8.0) * 2 ** (exp - 7))
        else:
            vals.append(sign * 448.0)
    return np.array(vals, dtype=np.float64)


_E4 = _e4m3_values()
_E2 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -0.0],
    dtype=np.float64,
)


def _encode_e4m3(x):
    return np.argmin(np.abs(np.asarray(x).reshape(-1, 1) - _E4), axis=1).astype(np.uint8)


def _decode_e4m3(c):
    return _E4[np.asarray(c, dtype=np.uint8)]


def quantize_nvfp4(x):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    blocks = np.asarray(
        [np.max(np.abs(x[i:i + 16])) for i in range(0, n, 16)],
        dtype=np.float64,
    )
    global_scale = float(np.max(blocks) / 448.0) if np.max(blocks) != 0 else 1.0
    block_scales = _encode_e4m3(blocks / global_scale)
    decoded = _decode_e4m3(block_scales)

    codes = np.empty(n, dtype=np.uint8)
    reconstruction = np.empty(n, dtype=np.float32)
    for b in range(len(decoded)):
        start = b * 16
        end = min(n, start + 16)
        values = x[start:end].astype(np.float64) / (decoded[b] * global_scale)
        codes[start:end] = np.argmin(
            np.abs(values.reshape(-1, 1) - _E2.reshape(1, -1)),
            axis=1,
        )
        reconstruction[start:end] = (
            _E2[codes[start:end]] * decoded[b] * global_scale
        ).astype(np.float32)
    return codes, block_scales, global_scale, reconstruction
