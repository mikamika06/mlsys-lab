import numpy as np

def grade(sol, fx) -> dict:
    """
    Verify that sol.launch_indices(block_dim, grid_dim)
    returns a NumPy array equal to np.arange(block_dim*grid_dim,dtype=np.int64).
    """
    ok = 1.0
    cases = [
        (1, 1),
        (4, 3),
        (5, 2),
        (7, 9),
        (10, 11)
    ]
    for block_dim, grid_dim in cases:
        try:
            got = sol.launch_indices(block_dim, grid_dim)
            expected = np.arange(block_dim * grid_dim, dtype=np.int64)
        except Exception:
            ok = 0.0
            break
        if not isinstance(got, np.ndarray):
            ok = 0.0
            break
        if got.shape != expected.shape or got.dtype != expected.dtype:
            ok = 0.0
            break
        if not np.array_equal(got, expected):
            ok = 0.0
            break
    return {"exact_match": ok}
