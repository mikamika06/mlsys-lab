import numpy as np

BLOCK_SIZE = 16
FP4_MAX = 6.0     # max magnitude representable by an E2M1 element
FP8_MAX = 448.0   # max magnitude representable by an E4M3 block scale

E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def _e4m3_nonneg_grid() -> np.ndarray:
    """All non-negative finite E4M3 representable magnitudes."""
    vals = set()
    for exp in range(16):
        for mant in range(8):
            if exp == 15 and mant == 7:
                continue  # NaN pattern
            if exp == 0:
                val = (2.0 ** -6) * (mant / 8.0)          # subnormal
            else:
                val = (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)  # normal
            vals.add(val)
    return np.array(sorted(vals), dtype=np.float64)


E4M3_GRID = _e4m3_nonneg_grid()


def _snap(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    vals = np.atleast_1d(np.asarray(vals, dtype=np.float64))
    diffs = np.abs(vals[:, None] - grid[None, :])
    idx = np.argmin(diffs, axis=1)
    return grid[idx]


def _oracle(w: np.ndarray, block_size: int):
    n = w.shape[0]
    nb = n // block_size
    wb = w.reshape(nb, block_size)

    tensor_amax = float(np.max(np.abs(w)))
    global_scale = tensor_amax / (FP4_MAX * FP8_MAX)

    block_amax = np.max(np.abs(wb), axis=1)
    block_scale_fp32 = block_amax / FP4_MAX
    block_scale_scaled = np.where(block_amax == 0, 0.0, block_scale_fp32 / global_scale)
    block_scale_scaled = np.clip(block_scale_scaled, 0.0, FP8_MAX)
    block_scales_e4m3 = _snap(block_scale_scaled, E4M3_GRID)

    eff_scale = block_scales_e4m3 * global_scale
    eff_scale_safe = np.where(eff_scale == 0, 1.0, eff_scale)

    normalized = wb / eff_scale_safe[:, None]
    sign = np.sign(normalized)
    mag = np.clip(np.abs(normalized), 0.0, FP4_MAX)
    mag_snapped = _snap(mag.reshape(-1), E2M1_MAG).reshape(mag.shape)
    codes = sign * mag_snapped

    dequant = (codes * eff_scale[:, None]).reshape(n)
    return global_scale, block_scales_e4m3, codes.reshape(n), dequant


def _fail():
    return {
        "global_scale_err": float("inf"),
        "block_scale_max_abs_err": float("inf"),
        "codes_max_abs_err": float("inf"),
        "max_abs_err": float("inf"),
    }


def grade(sol, fx) -> dict:
    w = fx["nv_w"]
    gs_ref, bs_ref, codes_ref, deq_ref = _oracle(w, BLOCK_SIZE)

    try:
        out = sol.nvfp4_two_level_quantize(w.copy(), BLOCK_SIZE)
    except Exception:
        return _fail()

    try:
        gs_got, bs_got, codes_got, deq_got = out
        gs_got = float(gs_got)
        bs_got = np.asarray(bs_got, dtype=np.float64).reshape(-1)
        codes_got = np.asarray(codes_got, dtype=np.float64).reshape(-1)
        deq_got = np.asarray(deq_got, dtype=np.float64).reshape(-1)
    except Exception:
        return _fail()

    if (
        bs_got.shape != bs_ref.shape
        or codes_got.shape != codes_ref.shape
        or deq_got.shape != deq_ref.shape
    ):
        return _fail()

    return {
        "global_scale_err": abs(gs_got - gs_ref),
        "block_scale_max_abs_err": float(np.max(np.abs(bs_got - bs_ref))),
        "codes_max_abs_err": float(np.max(np.abs(codes_got - codes_ref))),
        "max_abs_err": float(np.max(np.abs(deq_got - deq_ref))),
    }
