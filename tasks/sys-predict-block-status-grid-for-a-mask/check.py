import numpy as np

def _oracle(mask, tile_h, tile_w):
    """Compute the ground-truth block-status grid via NumPy reshape."""
    H, W = mask.shape
    n_h = H // tile_h
    n_w = W // tile_w
    area = tile_h * tile_w
    reshaped = mask.reshape(n_h, tile_h, n_w, tile_w)
    tile_sums = reshaped.sum(axis=(1, 3))
    status = np.full((n_h, n_w), 1, dtype=np.int32)
    status[tile_sums == 0] = 2
    status[tile_sums == area] = 0
    return status

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(42)

    cases = [
        (np.ones((8, 8), dtype=bool), 2, 2),
        (np.zeros((8, 8), dtype=bool), 2, 2),
        (rng.random((8, 8)) > 0.5, 2, 2),
        (rng.random((6, 12)) > 0.5, 3, 4),
        (rng.random((5, 7)) > 0.5, 1, 1),
        (rng.random((16, 16)) > 0.5, 4, 4),
        (rng.random((12, 8)) > 0.3, 4, 2),
        (np.triu(np.ones((10, 10), dtype=bool)), 2, 5),
    ]

    ok = 1.0
    for mask, th, tw in cases:
        try:
            got = sol.block_status_grid(mask, th, tw)
            ref = _oracle(mask, th, tw)
            if not np.array_equal(np.asarray(got, dtype=np.int32), ref):
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break

    return {"exact_match": ok}
