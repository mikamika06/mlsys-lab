import numpy as np

from mlsys import scorers

FP16_MAX = 65504.0
E_LO, E_HI = -64, 64


def _cases():
    """Deterministic gradient tensors spanning several orders of magnitude."""
    rng = np.random.default_rng(0)
    out = []

    # 1. small gradients: every element underflows a naive fp16 cast
    out.append((1e-8 * rng.standard_normal(256)).astype(np.float32))

    # 2. extremely small gradients (deep in fp16 underflow territory)
    out.append((1e-11 * rng.standard_normal(128)).astype(np.float32))

    # 3. mixed per-row magnitudes (a stack of "layers" with different scales)
    rows = [10.0 ** p * rng.standard_normal(64) for p in (-9.0, -7.0, -6.0)]
    out.append(np.stack(rows).astype(np.float32))

    # 4. large gradients: a naive fp16 cast OVERFLOWS, so the scale must be < 1
    out.append((1e6 * rng.standard_normal((8, 16))).astype(np.float32))

    # 5. moderate gradients
    out.append((3e-2 * rng.standard_normal(200)).astype(np.float32))

    return out


def _oracle_scale(g):
    """Largest S = 2**e (integer e) with max(|g|) * S <= FP16_MAX; 1.0 if g is all-zero."""
    m = float(np.max(np.abs(np.asarray(g, dtype=np.float64))))
    if m == 0.0:
        return 1.0
    best = None
    for e in range(E_LO, E_HI + 1):
        s = 2.0 ** e
        if m * s <= FP16_MAX:
            best = s
    return float(best) if best is not None else 2.0 ** E_LO


def _fail(**over):
    out = {
        "rel_err": float("inf"),
        "scale_ok": 0.0,
        "dtype_ok": 0.0,
        "naive_rel_err": 0.0,
    }
    out.update(over)
    return out


def grade(sol, fx) -> dict:
    cases = _cases()

    # info: what happens with NO loss scaling on the smallest-gradient case
    g0 = cases[0]
    naive = np.asarray(np.asarray(g0, dtype=np.float16), dtype=np.float32)
    naive_rel = scorers.rel_err(g0, naive)

    worst = 0.0
    scale_ok = 1.0
    dtype_ok = 1.0

    for g in cases:
        ref_scale = _oracle_scale(g)

        try:
            s = float(sol.pick_loss_scale(g.copy()))
        except Exception:
            return _fail(naive_rel_err=float(naive_rel))
        if not np.isfinite(s) or s != ref_scale:
            scale_ok = 0.0

        try:
            packed = sol.to_fp16_grads(g.copy(), ref_scale)
            packed = np.asarray(packed)
        except Exception:
            return _fail(naive_rel_err=float(naive_rel))
        if packed.dtype != np.float16 or packed.shape != g.shape:
            dtype_ok = 0.0

        try:
            back = sol.unscale_grads(np.asarray(packed).copy(), ref_scale)
            back = np.asarray(back)
        except Exception:
            return _fail(naive_rel_err=float(naive_rel))
        if back.dtype != np.float32 or back.shape != g.shape:
            dtype_ok = 0.0

        # the real oracle: the untouched fp32 gradients
        worst = max(worst, scorers.rel_err(g, back))

    return {
        "rel_err": float(worst),
        "scale_ok": float(scale_ok),
        "dtype_ok": float(dtype_ok),
        "naive_rel_err": float(naive_rel),
    }
