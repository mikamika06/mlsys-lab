"""Grader: fp32 -> bf16 packing (round-to-nearest-even) and back.

Oracle: for every fp32 input the two bracketing bf16 candidates are formed from
the raw bit pattern, their distances are compared in float64, and the nearer one
wins (ties -> even low bit). Nothing is hardcoded.
"""
import numpy as np

from mlsys import scorers, probe  # noqa: F401


def _oracle_codes(x: np.ndarray) -> np.ndarray:
    """Correctly rounded bf16 codes for finite float32 `x`, by explicit search."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lo = (u & np.uint32(0xFFFF0000)).astype(np.uint32)          # truncate toward zero
    hi = (lo + np.uint32(0x00010000)).astype(np.uint32)         # next bf16 magnitude up
    lo_f = lo.view(np.float32).astype(np.float64)
    hi_f = hi.view(np.float32).astype(np.float64)
    xf = x.astype(np.float64)
    d_lo = np.abs(xf - lo_f)
    d_hi = np.abs(hi_f - xf)
    lo_code = (lo >> np.uint32(16)).astype(np.uint32)
    tie_even_picks_hi = (lo_code & np.uint32(1)).astype(bool)   # lo is odd -> hi is even
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & tie_even_picks_hi)
    codes = np.where(pick_hi, hi >> np.uint32(16), lo_code)
    return codes.astype(np.uint16)


def _oracle_decode(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes, dtype=np.uint16).astype(np.uint32)
    return (c << np.uint32(16)).astype(np.uint32).view(np.float32)


def _edge_cases(rng) -> list:
    cases = []
    cases.append(np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32))
    # exactly representable bf16 values (1 + k/128)
    cases.append((1.0 + np.arange(8) / 128.0).astype(np.float32))
    # exact midpoints between neighbouring bf16 values -> ties-to-even fires
    base = (1.0 + np.arange(16) / 128.0).astype(np.float32)
    mid = (base.astype(np.float64) + (2.0 ** -8)).astype(np.float32)
    cases.append(mid)
    cases.append((-mid).astype(np.float32))
    # powers of two across the exponent range
    cases.append((2.0 ** np.arange(-30, 30, dtype=np.float64)).astype(np.float32))
    # tiny and huge magnitudes
    cases.append(np.array([1e-30, -1e-30, 1e30, -1e30, 5.960464e-08], dtype=np.float32))
    # random spread
    cases.append(rng.normal(0.0, 1.0, size=257).astype(np.float32))
    cases.append(rng.uniform(-1e4, 1e4, size=129).astype(np.float32))
    return cases


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    W = np.asarray(fx["W"], dtype=np.float32)

    fail = {
        "size_ratio": 0.0,
        "code_exact_fraction": 0.0,
        "max_rel_err": float("inf"),
        "max_abs_err": float("inf"),
        "decode_max_abs_err": float("inf"),
    }

    # ---- size ratio on the fixture -------------------------------------
    try:
        codes_W = sol.pack_bf16(W)
        codes_W = np.asarray(codes_W)
    except Exception:
        return fail
    if codes_W.shape != W.shape:
        return fail
    try:
        ratio = scorers.size_ratio(W, codes_W)
    except Exception:
        return fail

    # ---- code exactness vs oracle --------------------------------------
    cases = [W] + _edge_cases(rng)
    total = 0
    same = 0
    max_rel = 0.0
    max_abs = 0.0
    for x in cases:
        ref_codes = _oracle_codes(x)
        try:
            got = np.asarray(sol.pack_bf16(x))
        except Exception:
            return fail
        total += ref_codes.size
        if got.shape != ref_codes.shape or got.dtype != np.uint16:
            continue
        same += int(np.count_nonzero(got.astype(np.uint16) == ref_codes))

        # round trip error against the ORIGINAL values
        try:
            rt = np.asarray(sol.unpack_bf16(got), dtype=np.float64)
        except Exception:
            return fail
        if rt.shape != x.shape:
            return fail
        xf = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(rt) & np.isfinite(xf)
        if not np.all(finite):
            return fail
        diff = np.abs(rt - xf)
        max_abs = max(max_abs, float(np.max(diff)))
        nz = np.abs(xf) > 0
        if np.any(nz):
            max_rel = max(max_rel, float(np.max(diff[nz] / np.abs(xf[nz]))))

    if total == 0:
        return fail

    # ---- decoder exercised on grader-built raw codes --------------------
    raw = rng.integers(0, 1 << 16, size=4096, dtype=np.uint64).astype(np.uint16)
    ref_dec = _oracle_decode(raw)
    keep = np.isfinite(ref_dec)
    try:
        got_dec = np.asarray(sol.unpack_bf16(raw))
    except Exception:
        return fail
    if got_dec.shape != raw.shape:
        return fail
    dec_err = scorers.max_abs_err(ref_dec[keep].astype(np.float64),
                                  np.asarray(got_dec)[keep].astype(np.float64))

    return {
        "size_ratio": float(ratio),
        "code_exact_fraction": float(same) / float(total),
        "max_rel_err": float(max_rel),
        "max_abs_err": float(max_abs),
        "decode_max_abs_err": float(dec_err),
    }
