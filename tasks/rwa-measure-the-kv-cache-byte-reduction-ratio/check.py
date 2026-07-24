def grade(sol, fx) -> dict:
    cases = [
        (8, 16, 64),
        (4, 8, 32),
        (12, 24, 128),
        (1, 2, 256),
        (10, 5, 64),   # n_kv > n_q
    ]
    ok = 1.0
    for n_kv, n_q, dim in cases:
        try:
            got = sol.kv_cache_stats(n_kv, n_q, dim)
            if not isinstance(got, tuple) or len(got) != 3:
                ok = 0.0
                break
            kv_bytes, mha_bytes, ratio = got
            # reference using same formula but with numpy for consistency
            import numpy as np
            ref_kv = int(2 * n_kv * dim * 4)
            ref_mha = int(2 * n_q * dim * 4)
            ref_ratio = ref_kv / ref_mha
            if kv_bytes != ref_kv or mha_bytes != ref_mha:
                ok = 0.0
                break
            # relative error tolerance
            rel_err = abs(ratio - ref_ratio) / (abs(ref_ratio) + 1e-12)
            if rel_err > 1e-12:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
