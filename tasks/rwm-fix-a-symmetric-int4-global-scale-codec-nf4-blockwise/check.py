import numpy as np

from mlsys import scorers

NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float64)


def _oracle_nf4(w: np.ndarray, block_size: int) -> np.ndarray:
    n = w.shape[0]
    nb = n // block_size
    wb = w.reshape(nb, block_size)
    scales = np.max(np.abs(wb), axis=1)
    scales = np.where(scales == 0, 1.0, scales)
    normalized = wb / scales[:, None]
    diffs = np.abs(normalized[:, :, None] - NF4_LEVELS[None, None, :])
    idx = np.argmin(diffs, axis=-1)
    return (NF4_LEVELS[idx] * scales[:, None]).reshape(n)


def _oracle_naive_broken(w: np.ndarray) -> np.ndarray:
    levels = np.linspace(-1.0, 1.0, 16)
    scale = np.max(np.abs(w))
    scale = 1.0 if scale == 0 else scale
    normalized = w / scale
    diffs = np.abs(normalized[:, None] - levels[None, :])
    idx = np.argmin(diffs, axis=-1)
    return levels[idx] * scale


def grade(sol, fx) -> dict:
    block_size = 64
    rng = np.random.default_rng(0)
    w = (rng.standard_normal(4096) * 0.02).astype(np.float64)

    ref = _oracle_nf4(w, block_size)
    naive = _oracle_naive_broken(w)
    mse_naive = float(np.mean((w - naive) ** 2))

    try:
        got = sol.nf4_blockwise_dequant(np.array(w, dtype=np.float64), block_size)
        got = np.asarray(got, dtype=np.float64)
        if got.shape != w.shape:
            return {"rel_err": float("inf"), "beats_naive": 0.0}
        rel = scorers.rel_err(ref, got)
        mse_got = float(np.mean((w - got) ** 2))
    except Exception:
        return {"rel_err": float("inf"), "beats_naive": 0.0}

    beats_naive = 1.0 if mse_got < mse_naive else 0.0
    return {"rel_err": rel, "beats_naive": beats_naive}
