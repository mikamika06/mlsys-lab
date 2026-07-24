import numpy as np

from mlsys import scorers

E2M1_MAX = 6.0


def _e4m3_nonneg_grid() -> np.ndarray:
    """All non-negative finite E4M3 magnitudes (1s.4e.3m, bias 7, no inf,
    NaN only at exponent==15 & mantissa==7)."""
    bias = 7
    vals = set()
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue  # NaN
            if e == 0:
                v = (m / 8.0) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + m / 8.0) * (2.0 ** (e - bias))
            vals.add(v)
    return np.array(sorted(vals), dtype=np.float64)


_GRID = _e4m3_nonneg_grid()


def _round_to_e4m3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    idx = np.argmin(np.abs(x[..., None] - _GRID[None, :]), axis=-1)
    return _GRID[idx]


def _oracle(W: np.ndarray, group_size: int, per_tensor_scale: float) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    n_blocks = W.shape[0] // group_size
    blocks = W[:n_blocks * group_size].reshape(n_blocks, group_size)
    absmax = np.max(np.abs(blocks), axis=1)
    raw = absmax / (E2M1_MAX * per_tensor_scale)
    return _round_to_e4m3(raw)


def _build_cases():
    cases = []
    for seed, n, group_size, scale, mag in [
        (0, 16 * 20, 16, 0.5, 1.0),
        (1, 16 * 12, 16, 2.0, 50.0),
        (2, 16 * 8, 16, 0.1, 0.02),
    ]:
        rng = np.random.default_rng(seed)
        W = rng.standard_normal(n) * mag
        cases.append((W, group_size, scale))
    return cases


def grade(sol, fx) -> dict:
    all_ref = []
    all_got = []
    for W, group_size, per_tensor_scale in _build_cases():
        ref = _oracle(W, group_size, per_tensor_scale)
        try:
            got = np.asarray(sol.nvfp4_block_scales(W.copy(), group_size, per_tensor_scale), dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"rel_err": float("inf")}

        all_ref.append(ref)
        all_got.append(got)

    return {"rel_err": scorers.rel_err(np.concatenate(all_ref), np.concatenate(all_got))}
