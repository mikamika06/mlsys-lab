import numpy as np


def _oracle_intensity(g, n_q, d, seq_len):
    n_kv = np.asarray(n_q, dtype=np.float64) / np.asarray(g, dtype=np.float64)
    flops = 4.0 * np.asarray(n_q, dtype=np.float64) * np.asarray(seq_len, dtype=np.float64) * np.asarray(d, dtype=np.float64)
    kv_bytes = 4.0 * np.asarray(seq_len, dtype=np.float64) * np.asarray(d, dtype=np.float64) * n_kv
    q_bytes = 2.0 * np.asarray(n_q, dtype=np.float64) * np.asarray(d, dtype=np.float64)
    return float(flops / (kv_bytes + q_bytes))


def grade(sol, fx) -> dict:
    cases = [
        (1, 32, 128, 2048),
        (4, 32, 128, 2048),
        (8, 32, 128, 2048),
        (32, 32, 128, 2048),
        (1, 64, 80, 512),
        (8, 64, 80, 512),
        (64, 64, 80, 512),
    ]
    passed = 0
    for g, n_q, d, seq_len in cases:
        try:
            got = float(sol.decode_arithmetic_intensity(g, n_q, d, seq_len))
        except Exception:
            continue
        ref = _oracle_intensity(g, n_q, d, seq_len)
        if np.isfinite(got) and abs(got - ref) <= 1e-6:
            passed += 1
    return {"modeled_arith_intensity": passed / len(cases)}
