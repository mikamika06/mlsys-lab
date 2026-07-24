import math

def _reference_amat(l3_hit, miss_rate, local_dram, remote_base, hops, per_hop, local_frac):
    """Compute the reference AMAT using the closed-form NUMA model."""
    remote_latency = remote_base + max(hops - 1, 0) * per_hop
    dram_latency = local_frac * local_dram + (1.0 - local_frac) * remote_latency
    return l3_hit + miss_rate * dram_latency

def grade(sol, fx) -> dict:
    """Grade compute_numa_amat against the reference formula on multiple inputs."""
    test_cases = [
        # (l3_hit, miss_rate, local_dram, remote_base, hops, per_hop, local_frac)
        (10.0, 0.05, 80.0, 100.0, 2, 25.0, 0.7),
        (8.0, 0.10, 70.0, 120.0, 3, 20.0, 0.5),
        (12.0, 0.02, 90.0, 110.0, 1, 30.0, 0.9),
        (6.0, 0.15, 60.0, 150.0, 4, 15.0, 0.3),
        (10.0, 0.00, 80.0, 100.0, 2, 25.0, 0.7),   # zero miss rate edge
        (10.0, 0.05, 80.0, 100.0, 1, 25.0, 1.0),    # all-local edge
        (10.0, 0.05, 80.0, 100.0, 1, 25.0, 0.0),    # all-remote edge
    ]

    max_rel_err = 0.0
    for tc in test_cases:
        try:
            learner_val = sol.compute_numa_amat(*tc)
        except Exception:
            return {"rel_err": 1.0}

        ref_val = _reference_amat(*tc)

        if ref_val == 0.0:
            err = 0.0 if learner_val == 0.0 else 1.0
        else:
            err = abs(learner_val - ref_val) / abs(ref_val)
        max_rel_err = max(max_rel_err, err)

    return {"rel_err": max_rel_err}
