import ref

def check(workdir):
    from alibi_attn.matrix import check_support_matrix
    from alibi_attn.overflow import measure_overflow_rate
    import numpy as np

    match = 1
    for backend in ref.BACKENDS:
        for test_mods in [
            ["alibi", "causal"],
            ["alibi", "softcap"],
            ["sliding_window"],
            ["alibi", "softcap", "sliding_window"]
        ]:
            want = ref.check_support(backend, test_mods)
            got = check_support_matrix(backend, test_mods)
            if want != got:
                match = 0
                break

    scores = np.array([-100.0, -50.0, 0.0, 50.0, 100.0, 70000.0], dtype=np.float32)
    rate_no_cap = measure_overflow_rate(scores, threshold=65504.0, softcap=None)
    want_no_cap = ref.compute_overflow_rate(scores, threshold=65504.0, softcap=None)

    rate_capped = measure_overflow_rate(scores, threshold=65504.0, softcap=50.0)
    want_capped = ref.compute_overflow_rate(scores, threshold=65504.0, softcap=50.0)

    if abs(rate_no_cap - want_no_cap) > 1e-6 or abs(rate_capped - want_capped) > 1e-6:
        match = 0

    return {"matrix_matches": float(match)}
