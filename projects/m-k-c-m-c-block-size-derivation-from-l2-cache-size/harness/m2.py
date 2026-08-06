"""Milestone 2 harness check."""

import count_ops
import numpy as np


def check(workdir):
    from blocking.cache import derive_l2_blocking
    from blocking.matmul import matmul_cache_blocked, matmul_naive
    from blocking.registers import derive_register_tile

    out = {
        "naive_correctness": 0.0,
        "blocked_correctness": 0.0,
        "speedup_achieved": 0.0,
    }

    rng = np.random.default_rng(42)
    m, k, n = 64, 64, 64
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)

    expected = np.matmul(a, b)

    got_naive = matmul_naive(a, b)
    if not np.allclose(got_naive, expected, atol=1e-3, rtol=1e-3):
        out["_note"] = "matmul_naive output does not match reference result"
        return out
    out["naive_correctness"] = 1.0

    m_r, n_r = derive_register_tile(16, 32, 4)
    m_c, k_c = derive_l2_blocking(128 * 1024, m_r, n_r, 4, 0.75)

    got_blocked = matmul_cache_blocked(a, b, m_r, n_r, m_c, k_c)
    if not np.allclose(got_blocked, expected, atol=1e-3, rtol=1e-3):
        out["_note"] = "matmul_cache_blocked output does not match reference result"
        return out
    out["blocked_correctness"] = 1.0

    ops_naive = count_ops.count_naive_element_accesses(m, k, n)
    ops_blocked = count_ops.count_blocked_element_accesses(m, k, n, m_r, n_r, m_c, k_c)

    if ops_blocked < ops_naive * 0.5:
        out["speedup_achieved"] = 1.0
    else:
        out["_note"] = f"blocked memory access operations ({ops_blocked}) not sufficiently lower than naive ({ops_naive})"

    return out
