import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from rope.core import compute_rope_frequencies, apply_rope, rope_dot_product

    out = {"max_abs_err": 0.0}
    dim = 64
    max_len = 4096
    freqs = compute_rope_frequencies(dim, max_len)
    ref_freqs = ref.compute_rope_frequencies(dim, max_len)

    freq_err = float(np.max(np.abs(freqs[0] - ref_freqs[0]) + np.abs(freqs[1] - ref_freqs[1])))
    if freq_err > 1e-5:
        out["max_abs_err"] = freq_err
        out["_note"] = f"Frequency calculation mismatch: err={freq_err}"
        return out

    rng = np.random.RandomState(123)
    q = rng.randn(dim)
    k = rng.randn(dim)

    m1, n1 = 50, 20
    m2, n2 = 150, 120

    dp1 = rope_dot_product(q, k, m1, n1, freqs)
    dp2 = rope_dot_product(q, k, m2, n2, freqs)
    ref_dp1 = ref.rope_dot_product(q, k, m1, n1, ref_freqs)

    err_ref = float(np.abs(dp1 - ref_dp1))
    err_invariance = float(np.abs(dp1 - dp2))

    total_err = max(err_ref, err_invariance)
    out["max_abs_err"] = total_err
    if total_err > 1e-5:
        out["_note"] = f"Invariance check failed: dp1={dp1}, dp2={dp2}, ref={ref_dp1}"

    return out
