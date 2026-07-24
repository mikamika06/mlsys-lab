def roofline_phase_classify(batch_sizes, seq_lengths):
    P_peak = 200e9
    B_mem = 50e9
    d_model = 4096
    d_ff = 11008
    results = []
    for b, s in zip(batch_sizes, seq_lengths):
        I = (2 * d_model * d_ff) / (4 * d_model + 4 * d_ff)
        I *= (s / (s + 64)) * (b / (b + 8)) * 0.25
        if B_mem * I < P_peak:
            results.append("memory-bound")
        else:
            results.append("compute-bound")
    return results
