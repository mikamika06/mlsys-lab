import numpy as np


def _oracle(q, k_cache, v_cache, block_tables, seq_lens, block_size):
    q = np.asarray(q, dtype=np.float64)
    k_cache = np.asarray(k_cache, dtype=np.float64)
    v_cache = np.asarray(v_cache, dtype=np.float64)
    block_tables = np.asarray(block_tables)
    seq_lens = np.asarray(seq_lens)

    batch, d = q.shape
    out = np.zeros((batch, d), dtype=np.float64)
    scale = np.sqrt(float(d))

    for b in range(batch):
        length = int(seq_lens[b])
        k_list = []
        v_list = []
        for t in range(length):
            logical_block = t // block_size
            offset = t % block_size
            physical_block = int(block_tables[b, logical_block])
            k_list.append(k_cache[physical_block, offset])
            v_list.append(v_cache[physical_block, offset])

        k = np.asarray(k_list, dtype=np.float64)
        v = np.asarray(v_list, dtype=np.float64)
        scores = (k @ q[b]) / scale
        scores = scores - np.max(scores)
        weights = np.exp(scores)
        weights = weights / np.sum(weights)
        out[b] = weights @ v

    return out


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = []

    for batch, d, block_size, num_blocks in [
        (2, 4, 2, 5),
        (3, 3, 3, 8),
        (4, 8, 4, 10),
    ]:
        max_blocks = 3
        q = rng.normal(size=(batch, d)).astype(np.float32)
        k_cache = rng.normal(size=(num_blocks, block_size, d)).astype(np.float32)
        v_cache = rng.normal(size=(num_blocks, block_size, d)).astype(np.float32)

        block_tables = np.empty((batch, max_blocks), dtype=np.int64)
        for b in range(batch):
            block_tables[b] = rng.choice(num_blocks, size=max_blocks, replace=False)

        seq_lens = rng.integers(1, max_blocks * block_size + 1, size=batch)
        cases.append((q, k_cache, v_cache, block_tables, seq_lens, block_size))

    worst = 0.0
    for args in cases:
        ref = _oracle(*args)
        try:
            q_list = args[0].tolist()
            k_cache_list = args[1].tolist()
            v_cache_list = args[2].tolist()
            block_tables_list = args[3].tolist()
            seq_lens_list = args[4].tolist()
            block_size_val = args[5]

            got = np.asarray(sol.batched_paged_decode(
                q_list,
                k_cache_list,
                v_cache_list,
                block_tables_list,
                seq_lens_list,
                block_size_val
            ), dtype=np.float64)

            if got.shape != ref.shape:
                return {"max_abs_err": float("inf")}
            err = float(np.max(np.abs(got - ref)))
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)

    return {"max_abs_err": worst}
