import numpy as np

def grade(sol, fx) -> dict:
    """
    Verify that sol.launch_indices(block_dim, grid_dim)
    returns a list equal to the expected reference.
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
            expected = np.arange(block_dim * grid_dim, dtype=np.int64).tolist()
        except Exception:
            ok = 0.0
            break
        if not isinstance(got, list):
            ok = 0.0
            break
        if len(got) != len(expected):
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
