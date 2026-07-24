import numpy as np


def _oracle_mse(Q, K, n_rep, pool):
    n_heads, seq_q, d = Q.shape
    _, seq_k, _ = K.shape
    n_kv = n_heads // n_rep
    scale = 1.0 / np.sqrt(d)

    Kg = K.reshape(n_kv, n_rep, seq_k, d)
    if pool == "mean":
        Kp = Kg.mean(axis=1)
    else:  # "pick" -- baseline: keep only the first head of each group
        Kp = Kg[:, 0]
    Kp_rep = np.repeat(Kp, n_rep, axis=0)

    orig_logits = np.einsum("hqd,hkd->hqk", Q, K) * scale
    recon_logits = np.einsum("hqd,hkd->hqk", Q, Kp_rep) * scale
    return float(np.mean((recon_logits - orig_logits) ** 2))


def _cases():
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(6):
        n_rep = int(rng.integers(2, 5))
        n_kv = int(rng.integers(1, 4))
        n_heads = n_kv * n_rep
        seq_q = int(rng.integers(2, 6))
        seq_k = int(rng.integers(3, 8))
        d = int(rng.integers(2, 8))
        Q = rng.standard_normal((n_heads, seq_q, d))
        K = rng.standard_normal((n_heads, seq_k, d))
        cases.append((Q, K, n_rep))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for Q, K, n_rep in _cases():
        ref_mean = _oracle_mse(Q, K, n_rep, "mean")
        ref_pick = _oracle_mse(Q, K, n_rep, "pick")
        # Sanity: mean-pooling each group provably minimizes sum-of-squares
        # distance to the group's K heads, which (for these random, roughly
        # isotropic Q) strictly beats keeping a single representative head.
        assert ref_mean < ref_pick

        try:
            got = float(sol.mean_pool_gqa_logit_mse(Q.copy(), K.copy(), n_rep))
        except Exception:
            return {"rel_err": 1.0}

        rel = abs(got - ref_mean) / max(abs(ref_mean), 1e-12)
        worst = max(worst, rel)

    return {"rel_err": worst}
