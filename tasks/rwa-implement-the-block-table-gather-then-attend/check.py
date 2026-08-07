import numpy as np


def _oracle_attend(q, Kseq, Vseq):
    d = q.shape[0]
    scores = (Kseq @ q) / np.sqrt(d)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / np.sum(probs)
    return probs @ Vseq


def _make_case(rng, block_size, d):
    """A single request's KV cache scattered across a SHUFFLED physical pool,
    with the tail of its last logical block, and every physical block NOT
    referenced by block_table, filled with adversarial 'poison' values --
    reading any of it would blow up the softmax and be trivially detectable."""
    seq_len = int(rng.integers(1, 3 * block_size + 1))
    num_logical_blocks = (seq_len + block_size - 1) // block_size
    num_physical_blocks = num_logical_blocks + int(rng.integers(2, 5))
    phys_ids = rng.permutation(num_physical_blocks)
    block_table = phys_ids[:num_logical_blocks].astype(np.int64)

    shape = (num_physical_blocks, block_size, d)
    k_pool = rng.uniform(3.0, 6.0, size=shape) * rng.choice([-1.0, 1.0], size=shape)
    v_pool = rng.uniform(3.0, 6.0, size=shape) * rng.choice([-1.0, 1.0], size=shape)

    Kseq = rng.standard_normal((seq_len, d))
    Vseq = rng.standard_normal((seq_len, d))
    for pos in range(seq_len):
        lb = pos // block_size
        slot = pos % block_size
        phys = int(block_table[lb])
        k_pool[phys, slot] = Kseq[pos]
        v_pool[phys, slot] = Vseq[pos]

    q = rng.standard_normal(d)
    return q, k_pool, v_pool, block_table, seq_len, Kseq, Vseq


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(5)
    block_size = 4
    worst = 0.0

    for _ in range(10):
        d = int(rng.integers(3, 8))
        q, k_pool, v_pool, block_table, seq_len, Kseq, Vseq = _make_case(rng, block_size, d)
        ref = _oracle_attend(q, Kseq, Vseq)

        try:
            got = np.asarray(
                sol.paged_attention(
                    q.tolist(),
                    k_pool.tolist(),
                    v_pool.tolist(),
                    block_table.tolist(),
                    seq_len,
                    block_size,
                ),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": worst}
