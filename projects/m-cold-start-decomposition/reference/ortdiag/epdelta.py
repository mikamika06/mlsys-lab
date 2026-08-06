def compute_ep_delta(cpu_runs, alt_runs):
    if not cpu_runs or not alt_runs:
        return 0.0
    mean_cpu = sum(cpu_runs) / len(cpu_runs)
    mean_alt = sum(alt_runs) / len(alt_runs)
    return float(mean_cpu - mean_alt)
