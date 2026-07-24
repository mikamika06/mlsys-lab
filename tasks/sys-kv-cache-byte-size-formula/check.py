import numpy as np


def _oracle(L, n_q, n_kv, d, seq, dtype_bytes):
    kv_bytes = 2 * L * n_kv * d * seq * dtype_bytes
    mha_bytes = 2 * L * n_q * d * seq * dtype_bytes
    ratio = kv_bytes / mha_bytes
    return kv_bytes, ratio


def grade(sol, fx) -> dict:
    """
    Builds several seeded random (L, n_q, n_kv, d, seq, dtype_bytes) configs
    -- spanning MHA (n_kv==n_q), MQA (n_kv==1), and general GQA -- computes
    the exact reference kv_bytes = 2*L*n_kv*d*seq*dtype_bytes and
    ratio_vs_mha = n_kv/n_q directly, and compares them to the submission's
    kv_cache_size(...) output. Reports the worst-case abs error in the
    ratio ("size_ratio") and the worst-case relative error in kv_bytes
    ("rel_err") across all trials.
    """
    rng = np.random.default_rng(0)
    configs = []
    for _ in range(8):
        L = int(rng.integers(1, 64))
        n_kv = int(rng.integers(1, 16))
        group = int(rng.integers(1, 8))
        n_q = n_kv * group
        d = int(rng.integers(16, 256))
        seq = int(rng.integers(128, 8192))
        dtype_bytes = int(rng.choice([1, 2, 4]))
        configs.append((L, n_q, n_kv, d, seq, dtype_bytes))
    configs.append((12, 8, 8, 64, 2048, 2))  # explicit MHA case (n_kv == n_q)

    worst_ratio_err = 0.0
    worst_rel_err = 0.0
    for L, n_q, n_kv, d, seq, dtype_bytes in configs:
        exp_bytes, exp_ratio = _oracle(L, n_q, n_kv, d, seq, dtype_bytes)
        try:
            got_bytes, got_ratio = sol.kv_cache_size(L, n_q, n_kv, d, seq, dtype_bytes)
            got_bytes = float(got_bytes)
            got_ratio = float(got_ratio)
        except Exception:
            return {"size_ratio": 1e9, "rel_err": 1e9}

        worst_ratio_err = max(worst_ratio_err, abs(got_ratio - exp_ratio))
        worst_rel_err = max(worst_rel_err, abs(got_bytes - exp_bytes) / (abs(exp_bytes) + 1e-12))

    return {"size_ratio": worst_ratio_err, "rel_err": worst_rel_err}
