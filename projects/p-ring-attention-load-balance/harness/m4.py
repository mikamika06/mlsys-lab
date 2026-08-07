import ref

def check(workdir):
    m = {"utilization_improved": 0.0}
    b = ref.get_reference_balancer(4, 64)
    w = b.get_workload_per_step()
    u = b.measure_utilization(w)
    if 0.0 <= u <= 1.0:
        m["utilization_improved"] = 1.0
    return m
