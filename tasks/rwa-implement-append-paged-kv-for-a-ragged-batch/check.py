import numpy as np


def _scatter_write(pool, values, block_table, base_pos, block_size):
    """Reference scatter: write `values[i]` to the physical slot for
    logical position `base_pos + i`, following `block_table`."""
    for i in range(values.shape[0]):
        pos = base_pos + i
        logical_block = pos // block_size
        slot = pos % block_size
        phys = int(block_table[logical_block])
        pool[phys, slot] = values[i]


def _gather_read(pool, block_table, total_len, block_size):
    """Reference gather: read back the logical sequence of length
    `total_len` from the pool via `block_table`."""
    d = pool.shape[-1]
    out = np.empty((total_len, d), dtype=np.float64)
    for pos in range(total_len):
        logical_block = pos // block_size
        slot = pos % block_size
        phys = int(block_table[logical_block])
        out[pos] = pool[phys, slot]
    return out


def _causal_attend(Q, K, V, q_offset):
    """Q: (nq, d) queries at absolute sequence positions q_offset..q_offset+nq-1.
    K, V: (nk, d) full context. Row i of Q attends to K[j] for j <= q_offset + i."""
    nq, d = Q.shape
    scores = (Q @ K.T) / np.sqrt(d)
    row_abs = (q_offset + np.arange(nq))[:, None]
    col = np.arange(K.shape[0])[None, :]
    scores = np.where(col <= row_abs, scores, -np.inf)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V


def _make_case(rng):
    block_size = 4
    d = 5
    num_requests = 3
    old_lens = rng.integers(1, 9, size=num_requests)
    new_lens = rng.integers(1, 6, size=num_requests)
    total_lens = old_lens + new_lens
    blocks_needed = [(int(t) + block_size - 1) // block_size for t in total_lens]
    num_physical_blocks = sum(blocks_needed) + 5  # a few spare/unused blocks
    phys_ids = rng.permutation(num_physical_blocks)

    block_tables = []
    cursor = 0
    for nb in blocks_needed:
        block_tables.append(phys_ids[cursor:cursor + nb].astype(np.int64))
        cursor += nb

    k_pool = np.zeros((num_physical_blocks, block_size, d), dtype=np.float64)
    v_pool = np.zeros((num_physical_blocks, block_size, d), dtype=np.float64)

    old_k, old_v, new_k_list, new_v_list = [], [], [], []
    for r in range(num_requests):
        ok = rng.standard_normal((int(old_lens[r]), d))
        ov = rng.standard_normal((int(old_lens[r]), d))
        nk = rng.standard_normal((int(new_lens[r]), d))
        nv = rng.standard_normal((int(new_lens[r]), d))
        old_k.append(ok)
        old_v.append(ov)
        new_k_list.append(nk)
        new_v_list.append(nv)
        _scatter_write(k_pool, ok, block_tables[r], 0, block_size)
        _scatter_write(v_pool, ov, block_tables[r], 0, block_size)

    new_k = np.concatenate(new_k_list, axis=0)
    new_v = np.concatenate(new_v_list, axis=0)
    cu_new_seqlens = np.concatenate([[0], np.cumsum(new_lens)]).astype(np.int64)

    return dict(
        block_size=block_size, d=d, num_requests=num_requests,
        old_lens=old_lens, new_lens=new_lens, block_tables=block_tables,
        k_pool=k_pool, v_pool=v_pool, old_k=old_k, old_v=old_v,
        new_k_list=new_k_list, new_v_list=new_v_list,
        new_k=new_k, new_v=new_v, cu_new_seqlens=cu_new_seqlens,
    )


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(21)
    exact = 1.0
    worst_attn = 0.0

    for _ in range(5):
        case = _make_case(rng)
        block_size = case["block_size"]
        block_tables = case["block_tables"]
        num_requests = case["num_requests"]
        old_lens = case["old_lens"]
        new_lens = case["new_lens"]

        oracle_k_pool = case["k_pool"].copy()
        oracle_v_pool = case["v_pool"].copy()
        for r in range(num_requests):
            _scatter_write(oracle_k_pool, case["new_k_list"][r], block_tables[r], int(old_lens[r]), block_size)
            _scatter_write(oracle_v_pool, case["new_v_list"][r], block_tables[r], int(old_lens[r]), block_size)

        student_k_pool = case["k_pool"].copy()
        student_v_pool = case["v_pool"].copy()
        try:
            sol.append_paged_kv(
                student_k_pool, student_v_pool,
                case["new_k"].copy(), case["new_v"].copy(),
                case["cu_new_seqlens"].copy(),
                old_lens.copy(),
                [bt.copy() for bt in block_tables],
                block_size,
            )
        except Exception:
            return {"exact_match": 0.0, "max_abs_err": float("inf")}

        if (not isinstance(student_k_pool, np.ndarray) or not isinstance(student_v_pool, np.ndarray)
                or student_k_pool.shape != oracle_k_pool.shape
                or student_v_pool.shape != oracle_v_pool.shape):
            return {"exact_match": 0.0, "max_abs_err": float("inf")}

        if not (np.array_equal(student_k_pool, oracle_k_pool) and np.array_equal(student_v_pool, oracle_v_pool)):
            exact = 0.0

        for r in range(num_requests):
            total_len = int(old_lens[r] + new_lens[r])
            try:
                gk = _gather_read(student_k_pool, block_tables[r], total_len, block_size)
                gv = _gather_read(student_v_pool, block_tables[r], total_len, block_size)
            except Exception:
                return {"exact_match": 0.0, "max_abs_err": float("inf")}
            Qr = rng.standard_normal((int(new_lens[r]), case["d"]))
            got_attn = _causal_attend(Qr, gk, gv, int(old_lens[r]))

            ref_k = np.concatenate([case["old_k"][r], case["new_k_list"][r]], axis=0)
            ref_v = np.concatenate([case["old_v"][r], case["new_v_list"][r]], axis=0)
            ref_attn = _causal_attend(Qr, ref_k, ref_v, int(old_lens[r]))

            worst_attn = max(worst_attn, float(np.max(np.abs(got_attn - ref_attn))))

    return {"exact_match": exact, "max_abs_err": worst_attn}
