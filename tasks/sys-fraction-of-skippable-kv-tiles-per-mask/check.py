import numpy as np


def _analytic_skippable_fraction(mask: np.ndarray, tile_size: int) -> float:
    """Real oracle: partition `mask` into tile_size x tile_size blocks and
    compute the fraction of blocks that are entirely False (no attention
    weight anywhere in the block -> a tiled kernel could skip it)."""
    n = mask.shape[0]
    nt = n // tile_size
    blocks = mask.reshape(nt, tile_size, nt, tile_size)
    block_has_any = blocks.any(axis=(1, 3))
    return float(1.0 - block_has_any.mean())


def _causal_mask(n):
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return j <= i


def _sliding_window_mask(n, window):
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return (j <= i) & (j > i - window)


def _block_sparse_mask(n, tile_size, rng, keep_prob):
    nt = n // tile_size
    tile_pattern = rng.random((nt, nt)) < keep_prob
    return np.repeat(np.repeat(tile_pattern, tile_size, axis=0), tile_size, axis=1)


def _random_elementwise_mask(n, rng, keep_prob):
    return rng.random((n, n)) < keep_prob


def _cases():
    cases = []

    n, ts = 12, 3
    cases.append(("causal", _causal_mask(n), ts))

    n, ts, w = 16, 4, 5
    cases.append(("sliding_window", _sliding_window_mask(n, w), ts))

    n, ts = 15, 5
    rng = np.random.default_rng(3)
    cases.append(("block_sparse", _block_sparse_mask(n, ts, rng, keep_prob=0.4), ts))

    n, ts = 10, 2
    rng = np.random.default_rng(4)
    cases.append(("random_elementwise", _random_elementwise_mask(n, rng, keep_prob=0.3), ts))

    n, ts = 9, 3
    cases.append(("all_masked", np.zeros((n, n), dtype=bool), ts))

    n, ts = 8, 4
    cases.append(("all_kept", np.ones((n, n), dtype=bool), ts))

    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for name, mask, tile_size in _cases():
        expected = _analytic_skippable_fraction(mask, tile_size)
        try:
            got = float(sol.skippable_kv_tile_fraction(mask.copy(), tile_size))
        except Exception:
            return {"size_ratio": float("inf")}
        if not np.isfinite(got):
            return {"size_ratio": float("inf")}
        worst = max(worst, abs(got - expected))
    return {"size_ratio": worst}
