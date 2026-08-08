def compute_throughput_ratio(baseline_traces, cprofile_traces, pyspy_traces):
    ratios = []
    for b, c, p in zip(baseline_traces, cprofile_traces, pyspy_traces):
        if c <= 0 or p <= 0:
            continue
        cp_ratio = b / c
        ps_ratio = b / p
        ratios.append(cp_ratio / ps_ratio if ps_ratio > 0 else 1.0)
    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)
