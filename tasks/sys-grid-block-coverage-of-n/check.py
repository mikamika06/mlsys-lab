import numpy as np

def _ref(n, block_size):
    grid = (n + block_size - 1) // block_size
    mask = np.zeros((grid, block_size), dtype=bool)
    idx = np.arange(n)
    rows = idx // block_size
    cols = idx % block_size
    mask[rows, cols] = True
    return grid, mask

def grade(sol, fx):
    cases = [
        (0, 4),
        (1, 4),
        (3, 4),
        (4, 4),
        (5, 4),
        (10, 4),
        (15, 7),
        (23, 8)
    ]
    ok = 1.0
    for n, b in cases:
        try:
            got_grid, got_mask = sol.block_coverage(n, b)
        except Exception:
            return {"exact_match": 0.0}
        ref_grid, ref_mask = _ref(n, b)
        if got_grid != ref_grid or not np.array_equal(got_mask, ref_mask):
            ok = 0.0
            break
    return {"exact_match": ok}
