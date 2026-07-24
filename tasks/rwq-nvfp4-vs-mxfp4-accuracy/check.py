import numpy as np


_CODEBOOK = np.array(
    [-1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0],
    dtype=np.float64,
)


def _quantize_oracle(x, block_size, pow2_scale):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    for start in range(0, len(x), block_size):
        end = min(start + block_size, len(x))
        block = x[start:end]
        max_mag = float(np.max(np.abs(block))) if len(block) else 0.0
        scale = max_mag / np.max(np.abs(_CODEBOOK))
        if scale == 0:
            q_scale = 1.0
        elif pow2_scale:
            q_scale = 2.0 ** np.ceil(np.log2(scale))
        else:
            q_scale = scale
        codes = np.argmin(
            np.abs(block[:, None] / q_scale - _CODEBOOK[None, :]),
            axis=1,
        )
        out[start:end] = q_scale * _CODEBOOK[codes]
    return out


def _reference(weight):
    nv = _quantize_oracle(weight, 16, False)
    mx = _quantize_oracle(weight, 32, True)
    nv_rmse = float(np.sqrt(np.mean((weight.astype(np.float64) - nv) ** 2)))
    mx_rmse = float(np.sqrt(np.mean((weight.astype(np.float64) - mx) ** 2)))
    return nv_rmse, mx_rmse


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    weight = rng.normal(0, 1, 160).astype(np.float32)
    weight[:16] *= 0.25
    ref_nv, ref_mx = _reference(weight)
    try:
        got_nv, got_mx = sol.fp4_accuracy_comparison(weight)
        got_nv = float(got_nv)
        got_mx = float(got_mx)
    except Exception:
        return {
            "nvfp4_rmse_error": float("inf"),
            "mxfp4_rmse_error": float("inf"),
            "nvfp4_better": 0.0,
        }
    return {
        "nvfp4_rmse_error": abs(got_nv - ref_nv),
        "mxfp4_rmse_error": abs(got_mx - ref_mx),
        "nvfp4_better": 1.0 if got_nv < got_mx else 0.0,
    }
