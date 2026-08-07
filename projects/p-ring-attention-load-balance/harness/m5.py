import ref

def check(workdir):
    m = {"below_threshold": 0.0}
    b = ref.get_reference_balancer(4, 64)
    w = b.get_workload_per_step()
    new_w = b.rebalance(w)
    if b.check_imbalance_threshold(new_w, threshold=0.9):
        m["below_threshold"] = 1.0
    return m
