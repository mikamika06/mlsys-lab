import numpy as np


def _cases():
    rng = np.random.default_rng(29)
    cases = []
    for _ in range(6):
        block_size = int(rng.integers(3, 7))
        seq_len = int(rng.integers(5, 30))
        d = int(rng.integers(2, 8))

        n_logical_blocks = (seq_len + block_size - 1) // block_size
        num_physical_blocks = n_logical_blocks + int(rng.integers(1, 4))
        block_table = rng.permutation(num_physical_blocks)[:n_logical_blocks].tolist()

        original = rng.standard_normal((seq_len, d))

        pool_rows = num_physical_blocks * block_size
        kv_pool = rng.standard_normal((pool_rows, d)) * 1000.0  # unrelated noise everywhere

        for pos in range(seq_len):
            logical_block = pos // block_size
            offset = pos % block_size
            slot = block_table[logical_block] * block_size + offset
            kv_pool[slot] = original[pos]

        cases.append((kv_pool, block_table, block_size, seq_len, original))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for kv_pool, block_table, block_size, seq_len, original in _cases():
        try:
            got = np.asarray(
                sol.reconstruct_contiguous_kv(kv_pool.copy(), list(block_table), block_size, seq_len),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != original.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - original))))
    return {"max_abs_err": worst}
