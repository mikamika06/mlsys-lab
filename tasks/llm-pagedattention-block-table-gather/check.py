import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_error = 0.0
    for _ in range(3):
        d = rng.integers(8, 32)
        block_size = 4
        num_blocks = rng.integers(5, 10)
        N = num_blocks * block_size
        keys_contig = rng.standard_normal((N, d))
        vals_contig = rng.standard_normal((N, d))
        blocks_keys = [keys_contig[i*block_size:(i+1)*block_size] for i in range(num_blocks)]
        blocks_vals = [vals_contig[i*block_size:(i+1)*block_size] for i in range(num_blocks)]
        blocks = list(zip(blocks_keys, blocks_vals))
        block_table = {i: (i // block_size, i % block_size) for i in range(N)}
        indices = rng.choice(N, size=rng.integers(5, N), replace=False)
        expected_keys = keys_contig[indices]
        expected_vals = vals_contig[indices]
        try:
            got_keys, got_vals = sol.paged_gather(blocks, block_table, indices)
        except Exception:
            return {"max_abs_err": float("inf")}
        err_k = max_abs_err(got_keys, expected_keys)
        err_v = max_abs_err(got_vals, expected_vals)
        max_error = max(max_error, err_k, err_v)
    return {"max_abs_err": max_error}
