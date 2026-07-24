import numpy as np

NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float64)


def _oracle_nf4_indices(w: np.ndarray, block_size: int) -> np.ndarray:
    n = w.shape[0]
    nb = n // block_size
    wb = w.reshape(nb, block_size)
    scales = np.max(np.abs(wb), axis=1)
    scales = np.where(scales == 0, 1.0, scales)
    normalized = wb / scales[:, None]
    diffs = np.abs(normalized[:, :, None] - NF4_LEVELS[None, None, :])
    idx = np.argmin(diffs, axis=-1)
    return idx.reshape(n).astype(np.int64)


def _build_cases():
    cases = []
    for seed, n in [(0, 4096), (1, 2048), (2, 6400)]:
        rng = np.random.default_rng(seed)
        w = (rng.standard_normal(n) * 0.02).astype(np.float64)
        cases.append((w, 64))
    return cases


def grade(sol, fx) -> dict:
    total = 0
    matched = 0

    for w, block_size in _build_cases():
        ref = _oracle_nf4_indices(w, block_size)

        try:
            got = sol.nf4_quantize_indices(w.copy(), block_size)
        except Exception:
            return {"index_match_frac": 0.0}

        try:
            got = np.asarray(got).astype(np.int64).reshape(-1)
        except Exception:
            return {"index_match_frac": 0.0}

        if got.shape != ref.shape:
            return {"index_match_frac": 0.0}

        total += ref.size
        matched += int(np.sum(got == ref))

    frac = matched / total if total else 0.0
    return {"index_match_frac": frac}
