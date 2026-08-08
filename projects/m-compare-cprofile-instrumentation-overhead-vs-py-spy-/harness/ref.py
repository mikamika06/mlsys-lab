BASELINE_TRACES = [1000.0, 1500.0, 1200.0, 1800.0]
CPROFILE_TRACES = [200.0, 300.0, 240.0, 360.0]
PYSPY_TRACES = [900.0, 1350.0, 1080.0, 1620.0]

FLAG_MEASUREMENTS = {
    "record_shapes": 0.18,
    "with_stack": 0.52,
    "profile_memory": 0.31,
    "with_flops": 0.04
}

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

def rank_profiler_flags(flag_measurements):
    sorted_flags = sorted(flag_measurements.items(), key=lambda x: x[1])
    return [flag for flag, overhead in sorted_flags]
