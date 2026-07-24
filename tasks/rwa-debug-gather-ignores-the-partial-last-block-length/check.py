import numpy as np

from mlsys import scorers


def _oracle(k_phys, v_phys, block_table, seq_len, q):
    H = k_phys.shape[-1]
    k_logical = k_phys[block_table].reshape(-1, H).astype(np.float64)[:seq_len]
    v_logical = v_phys[block_table].reshape(-1, H).astype(np.float64)[:seq_len]
    qf = np.asarray(q, dtype=np.float64)
    scores = (k_logical @ qf) / np.sqrt(H)
    scores = scores - np.max(scores)
    w = np.exp(scores)
    w = w / np.sum(w)
    return (w[:, None] * v_logical).sum(axis=0)


def _make_case(rng, num_phys, block_size, H, valid_logical_blocks, seq_len, garbage_scale=100.0):
    k_phys = rng.normal(size=(num_phys, block_size, H)).astype(np.float64)
    v_phys = rng.normal(size=(num_phys, block_size, H)).astype(np.float64)
    block_table = rng.permutation(num_phys)[:valid_logical_blocks].astype(np.int64)
    q = rng.normal(size=(H,)).astype(np.float64)

    valid_in_last = seq_len - (valid_logical_blocks - 1) * block_size
    assert 0 < valid_in_last <= block_size

    # Stale slots at/after `seq_len` in the last logical block: aligned with q
    # so their dot-product score is guaranteed large regardless of q's random
    # signs, and their value is a distinct constant so any leakage is obvious.
    if valid_in_last < block_size:
        last_phys = block_table[-1]
        k_phys[last_phys, valid_in_last:, :] = garbage_scale * np.sign(q)[None, :]
        v_phys[last_phys, valid_in_last:, :] = 999.0

    return k_phys, v_phys, block_table, seq_len, q


def _cases():
    rng = np.random.default_rng(20260723)
    specs = [
        (6, 4, 8, 3, 11),   # last block missing 1 of 4 slots
        (8, 3, 5, 4, 10),   # last block missing 2 of 3 slots
        (5, 2, 4, 2, 4),    # exact multiple of block_size: no stale rows
        (10, 6, 6, 5, 25),  # last block has only 1 valid row of 6
    ]
    out = []
    for num_phys, block_size, H, vlb, seq_len in specs:
        out.append(_make_case(rng, num_phys, block_size, H, vlb, seq_len))
    return out


def grade(sol, fx) -> dict:
    worst = 0.0
    for k_phys, v_phys, block_table, seq_len, q in _cases():
        expected = _oracle(k_phys, v_phys, block_table, seq_len, q)
        try:
            got = np.asarray(
                sol.gathered_attention(k_phys.copy(), v_phys.copy(), block_table.copy(), seq_len, q.copy()),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != expected.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(scorers.max_abs_err(expected, got)))
    return {"max_abs_err": worst}
