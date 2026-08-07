import ref

def check(workdir):
    m = {"imbalance_detected": 0.0}
    b = ref.get_reference_balancer(4, 64)
    w = b.get_workload_per_step()
    if isinstance(w, list) and len(w) == 4:
        has_zeros = any(0 in step for step in w)
        if has_zeros:
            m["imbalance_detected"] = 1.0
    return m
