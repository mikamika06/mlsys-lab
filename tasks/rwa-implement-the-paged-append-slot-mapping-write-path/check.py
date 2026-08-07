import numpy as np


def _slot_of(pos, block_size, block_table):
    logical_block = pos // block_size
    offset = pos % block_size
    return block_table[logical_block] * block_size + offset


def _oracle_output(existing_k, existing_v, new_k, new_v, q):
    full_k = np.concatenate([existing_k, new_k], axis=0)
    full_v = np.concatenate([existing_v, new_v], axis=0)
    d = full_k.shape[1]
    scores = (q @ full_k.T) / np.sqrt(d)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / np.sum(probs)
    return probs @ full_v


def _cases():
    rng = np.random.default_rng(23)
    cases = []
    for _ in range(6):
        block_size = int(rng.integers(3, 6))
        existing_len = int(rng.integers(1, block_size))  # < 1 block, forces the append to cross a boundary
        T = int(rng.integers(block_size, 2 * block_size + 1))  # long enough to span >= 2 blocks
        d = int(rng.integers(3, 8))

        total_len = existing_len + T
        n_logical_blocks = (total_len + block_size - 1) // block_size
        num_physical_blocks = n_logical_blocks + int(rng.integers(0, 3))
        block_table = rng.permutation(num_physical_blocks)[:n_logical_blocks].tolist()

        existing_k = rng.standard_normal((existing_len, d))
        existing_v = rng.standard_normal((existing_len, d))
        new_k = rng.standard_normal((T, d))
        new_v = rng.standard_normal((T, d))
        q = rng.standard_normal(d)

        pool_rows = num_physical_blocks * block_size
        kv_pool_k = rng.standard_normal((pool_rows, d)) * 1000.0  # garbage elsewhere in the pool
        kv_pool_v = rng.standard_normal((pool_rows, d)) * 1000.0
        for pos in range(existing_len):
            s = _slot_of(pos, block_size, block_table)
            kv_pool_k[s] = existing_k[pos]
            kv_pool_v[s] = existing_v[pos]

        ref_out = _oracle_output(existing_k, existing_v, new_k, new_v, q)
        cases.append((kv_pool_k, kv_pool_v, block_table, block_size, existing_len,
                       new_k, new_v, q, ref_out))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for kv_pool_k, kv_pool_v, block_table, block_size, existing_len, new_k, new_v, q, ref_out in _cases():
        try:
            got = np.asarray(
                sol.paged_append_and_attend(
                    kv_pool_k.copy().tolist(), kv_pool_v.copy().tolist(), list(block_table), block_size,
                    existing_len, new_k.copy().tolist(), new_v.copy().tolist(), q.copy().tolist(),
                ),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref_out.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref_out))))
    return {"max_abs_err": worst}
