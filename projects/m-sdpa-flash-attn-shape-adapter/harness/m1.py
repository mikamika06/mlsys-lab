import ref
import numpy as np


def check(workdir):
    from adapter.shape import sdpa_to_flash

    cases = ref.generate_cases()
    matched = 0
    for q, k, v in cases:
        try:
            got_q, got_k, got_v = sdpa_to_flash(q, k, v)
            want_q, want_k, want_v = ref.oracle_sdpa_to_flash(q, k, v)
            if (
                got_q.shape == want_q.shape
                and got_k.shape == want_k.shape
                and got_v.shape == want_v.shape
                and np.allclose(got_q, want_q)
                and np.allclose(got_k, want_k)
                and np.allclose(got_v, want_v)
            ):
                matched += 1
        except Exception:
            pass
    return {"shapes_matched": float(matched)}
