import numpy as np


def _oracle(k_phys, v_phys, block_table, q):
    k_logical = k_phys[block_table].reshape(-1, k_phys.shape[-1])
    v_logical = v_phys[block_table].reshape(-1, v_phys.shape[-1])
    scores = (k_logical @ q.astype(np.float64)) / np.sqrt(k_logical.shape[-1])
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / np.sum(weights)
    return np.sum(weights[:, None] * v_logical, axis=0).astype(np.float64)


def grade(sol, fx) -> dict:
    cases = [
        (3, 2, 4),
        (4, 3, 5),
        (5, 1, 3),
    ]
    worst = 0.0
    rng = np.random.default_rng(12345)
    for blocks, block_size, head_dim in cases:
        k_phys = rng.normal(size=(blocks, block_size, head_dim)).astype(np.float32)
        v_phys = rng.normal(size=(blocks, block_size, head_dim)).astype(np.float32)
        table = np.arange(blocks - 1, -1, -1, dtype=np.int64)
        if blocks > 2:
            table = np.roll(table, 1)
        q = rng.normal(size=(head_dim,)).astype(np.float32)

        expected = _oracle(k_phys, v_phys, table, q)
        try:
            got = np.asarray(sol.gather_attention(k_phys, v_phys, table, q), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != expected.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - expected))))
    return {"max_abs_err": worst}
