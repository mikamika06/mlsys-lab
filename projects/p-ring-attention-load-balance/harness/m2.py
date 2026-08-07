import ref

def check(workdir):
    m = {"rebalanced": 0.0}
    b = ref.get_reference_balancer(4, 64)
    w = b.get_workload_per_step()
    new_w = b.rebalance(w)
    if isinstance(new_w, list):
        m["rebalanced"] = 1.0
    return m
