import numpy as np

from mlsys import scorers


def _oracle(k, v, q, n_kv_heads):
    """Contiguous GQA attention over the original (un-paged) sequence."""
    n_q_heads, D = q.shape
    group = n_q_heads // n_kv_heads
    scale = 1.0 / np.sqrt(D)
    out = np.zeros((n_q_heads, D), dtype=np.float64)
    for h in range(n_q_heads):
        kv_h = h // group
        K = k[:, kv_h, :]
        V = v[:, kv_h, :]
        scores = (K @ q[h]) * scale
        scores = scores - scores.max()
        w = np.exp(scores)
        w = w / w.sum()
        out[h] = w @ V
    return out


def _build_paged_case(rng, k, v, block_size, n_kv_heads, num_extra_phys=3, garbage=777.0):
    seq_len, H_kv, D = k.shape
    assert seq_len % block_size == 0
    L_b = seq_len // block_size
    num_phys = L_b + num_extra_phys

    block_table = rng.permutation(num_phys)[:L_b].astype(np.int64)

    k_phys = np.full((num_phys, block_size, H_kv, D), garbage, dtype=np.float64)
    v_phys = np.full((num_phys, block_size, H_kv, D), garbage, dtype=np.float64)
    for i in range(L_b):
        phys = int(block_table[i])
        k_phys[phys] = k[i * block_size:(i + 1) * block_size]
        v_phys[phys] = v[i * block_size:(i + 1) * block_size]

    return k_phys, v_phys, block_table


def grade(sol, fx) -> dict:
    k = np.asarray(fx["k"], dtype=np.float64)
    v = np.asarray(fx["v"], dtype=np.float64)
    q = np.asarray(fx["q"], dtype=np.float64)
    n_kv_heads = k.shape[1]

    rng = np.random.default_rng(2026)
    expected = _oracle(k, v, q, n_kv_heads)

    worst = 0.0
    for block_size in (16, 32, 8):
        k_phys, v_phys, block_table = _build_paged_case(rng, k, v, block_size, n_kv_heads)
        try:
            got = np.asarray(
                sol.paged_gqa_attention(k_phys.copy(), v_phys.copy(), block_table.copy(), q.copy(), n_kv_heads),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != expected.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        worst = max(worst, float(scorers.max_abs_err(expected, got)))

    return {"max_abs_err": worst}
