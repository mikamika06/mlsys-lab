import numpy as np


def _oracle_append(k_pool, v_pool, block_table, free_blocks, new_k, new_v, block_size):
    new_k = np.asarray(new_k, dtype=np.float64)
    new_v = np.asarray(new_v, dtype=np.float64)
    n_before = len(block_table) * block_size
    L = new_k.shape[0]
    for i in range(L):
        pos = n_before + i
        logical_block = pos // block_size
        slot = pos % block_size
        if logical_block >= len(block_table):
            block_table.append(free_blocks.pop(0))
        phys = block_table[logical_block]
        k_pool[phys, slot] = new_k[i]
        v_pool[phys, slot] = new_v[i]
    return n_before + L


def _oracle_attend(k_pool, v_pool, block_table, block_size, seq_len, q):
    k_pool = np.asarray(k_pool, dtype=np.float64)
    v_pool = np.asarray(v_pool, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    idx = np.asarray(block_table, dtype=np.int64)
    k_logical = k_pool[idx].reshape(-1, k_pool.shape[-1])[:seq_len]
    v_logical = v_pool[idx].reshape(-1, v_pool.shape[-1])[:seq_len]
    D = q.shape[0]
    scores = (k_logical @ q) / np.sqrt(D)
    scores = scores - np.max(scores)
    w = np.exp(scores)
    w = w / np.sum(w)
    return w @ v_logical


def _make_pools(rng, num_phys, block_size, D):
    k_pool = rng.standard_normal((num_phys, block_size, D))
    v_pool = rng.standard_normal((num_phys, block_size, D))
    return k_pool, v_pool


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    bookkeeping_ok = 1.0
    worst_err = 0.0

    for _ in range(4):
        D = int(rng.integers(3, 6))
        block_size = int(rng.integers(2, 5))
        num_phys = int(rng.integers(8, 14))

        # reference pools/state
        k_ref, v_ref = _make_pools(rng, num_phys, block_size, D)
        bt_ref = []
        free_ref = list(range(num_phys))
        rng.shuffle(free_ref)  # arbitrary initial free order

        # candidate pools/state -- independent copies, identical initial content
        k_got = k_ref.copy()
        v_got = v_ref.copy()
        bt_got = []
        free_got = list(free_ref)

        seq_len_ref = 0
        seq_len_got = None

        # several append rounds, growing the sequence
        n_rounds = int(rng.integers(2, 4))
        for _ in range(n_rounds):
            L = int(rng.integers(1, 2 * block_size + 1))
            new_k = rng.standard_normal((L, D))
            new_v = rng.standard_normal((L, D))

            seq_len_ref = _oracle_append(
                k_ref, v_ref, bt_ref, free_ref, new_k.copy(), new_v.copy(), block_size
            )
            try:
                seq_len_got = sol.paged_append(
                    k_got, v_got, bt_got, free_got, new_k.copy(), new_v.copy(), block_size
                )
            except Exception:
                return {"bookkeeping_exact": 0.0, "max_abs_err": float("inf")}

            if (
                seq_len_got != seq_len_ref
                or list(bt_got) != list(bt_ref)
                or list(free_got) != list(free_ref)
            ):
                bookkeeping_ok = 0.0

            if not np.allclose(k_got, k_ref) or not np.allclose(v_got, v_ref):
                bookkeeping_ok = 0.0

        # now attend with a fresh query
        q = rng.standard_normal(D)
        ref_out = _oracle_attend(k_ref, v_ref, bt_ref, block_size, seq_len_ref, q)
        try:
            got_out = np.asarray(
                sol.gather_and_attend(k_got, v_got, bt_got, block_size, seq_len_got, q),
                dtype=np.float64,
            )
        except Exception:
            return {"bookkeeping_exact": bookkeeping_ok, "max_abs_err": float("inf")}

        if got_out.shape != ref_out.shape:
            return {"bookkeeping_exact": bookkeeping_ok, "max_abs_err": float("inf")}

        worst_err = max(worst_err, float(np.max(np.abs(got_out - ref_out))))

    return {"bookkeeping_exact": bookkeeping_ok, "max_abs_err": worst_err}
