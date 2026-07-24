def _ref(batch_sizes, seq_lengths):
    P_peak = 200e9
    B_mem = 50e9
    d_model = 4096
    d_ff = 11008
    out = []
    for b, s in zip(batch_sizes, seq_lengths):
        I = (2 * d_model * d_ff) / (4 * d_model + 4 * d_ff)
        I *= (s / (s + 64)) * (b / (b + 8)) * 0.25
        if B_mem * I < P_peak:
            out.append("memory-bound")
        else:
            out.append("compute-bound")
    return out


def grade(sol, fx) -> dict:
    cases = [
        ([1, 16, 32, 64], [8, 64, 256, 1024]),
        ([1, 4, 8, 16], [1, 8, 64, 512]),
        ([2, 8, 32], [4, 128, 512]),
    ]
    ok = 1.0
    for batches, seqs in cases:
        try:
            got = sol.roofline_phase_classify(batches, seqs)
        except Exception:
            ok = 0.0
            break
        ref = _ref(batches, seqs)
        if list(got) != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
