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
    x = np.asarray(x, dtype=np.float64)
    idx = np.argmin(np.abs(x.reshape(-1, 1) - _E4.reshape(1, -1)), axis=1)
    return idx.astype(np.uint8)


def _decode_e4m3(c):
    return _E4[np.asarray(c, dtype=np.uint8)]


def _oracle(x):
    x = np.asarray(x, dtype=np.float32)
    blocks = []
    for i in range(0, len(x), 16):
        blocks.append(np.max(np.abs(x[i:i + 16])))
    blocks = np.asarray(blocks, dtype=np.float64)
    global_scale = float(np.max(blocks) / 448.0) if np.max(blocks) != 0 else 1.0
    raw = blocks / global_scale
    scale_codes = _encode_e4m3(raw)
    decoded_blocks = _decode_e4m3(scale_codes)

    codes = np.empty(len(x), dtype=np.uint8)
    recon = np.empty(len(x), dtype=np.float32)
    for b in range(len(blocks)):
        start = b * 16
        end = min(len(x), start + 16)
        q = x[start:end].astype(np.float64) / (global_scale * decoded_blocks[b])
        c = np.argmin(np.abs(q.reshape(-1, 1) - _E2.reshape(1, -1)), axis=1)
        codes[start:end] = c
        recon[start:end] = (_E2[c] * global_scale * decoded_blocks[b]).astype(np.float32)
    return codes, scale_codes, global_scale, recon


def grade(sol, fx) -> dict:
    cases = [
        np.linspace(-12, 12, 33, dtype=np.float32),
        np.array([0, 1, -1, 3.2, -7.5, 100], dtype=np.float32),
        np.arange(64, dtype=np.float32) / 3 - 8,
    ]
    code_ok = 1.0
    scale_err = 0.0
    recon_err = 0.0
    for x in cases:
        ref_codes, ref_scales, ref_g, ref_y = _oracle(x)
        try:
            got_codes, got_scales, got_g, got_y = sol.quantize_nvfp4(x)
        except Exception:
            return {"code_exact": 0.0, "scale_max_abs_err": 1e9, "max_abs_err": 1e9}
        code_ok *= float(np.array_equal(np.asarray(got_codes), ref_codes))
        scale_err = max(scale_err, float(np.max(np.abs(_decode_e4m3(got_scales) - _decode_e4m3(ref_scales)))))
        recon_err = max(recon_err, float(np.max(np.abs(np.asarray(got_y, dtype=np.float32) - ref_y))))
    return {
        "code_exact": code_ok,
        "scale_max_abs_err": scale_err,
        "max_abs_err": recon_err,
    }
