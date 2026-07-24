import numpy as np


def _decode_e4m3(codes):
    codes = np.asarray(codes, dtype=np.int64)
    S = (codes >> 7) & 1
    E = (codes >> 3) & 0xF
    M = (codes & 0x7).astype(np.float64)
    sign = np.where(S == 0, 1.0, -1.0)
    normal = sign * np.ldexp(1.0 + M / 8.0, (E - 7).astype(np.int64))
    sub = sign * np.ldexp(M / 8.0, -6)
    return np.where(E == 0, sub, normal)


def _decode_e2m1(codes):
    codes = np.asarray(codes, dtype=np.int64)
    S = (codes >> 3) & 1
    E = (codes >> 1) & 0x3
    M = (codes & 0x1).astype(np.float64)
    sign = np.where(S == 0, 1.0, -1.0)
    normal = sign * np.ldexp(1.0 + M / 2.0, (E - 1).astype(np.int64))
    sub = sign * (M / 2.0)
    return np.where(E == 0, sub, normal)


def _oracle(global_scale, e4m3_block_codes, e2m1_codes):
    g = float(global_scale)
    s_b = _decode_e4m3(e4m3_block_codes)          # (n_blocks,)
    q = _decode_e2m1(e2m1_codes)                  # (n_blocks, 16)
    return g * s_b[:, None] * q


def grade(sol, fx) -> dict:
    global_scale = fx["global_scale"]
    e4m3_block_codes = fx["e4m3_block_codes"]
    e2m1_codes = fx["e2m1_codes"]

    ref = _oracle(global_scale, e4m3_block_codes, e2m1_codes)

    try:
        got = np.asarray(
            sol.nvfp4_reconstruct(global_scale, e4m3_block_codes.copy(), e2m1_codes.copy()),
            dtype=np.float64,
        )
    except Exception:
        return {"max_abs_err": float("inf")}

    if got.shape != ref.shape:
        return {"max_abs_err": float("inf")}

    return {"max_abs_err": float(np.max(np.abs(got - ref)))}
