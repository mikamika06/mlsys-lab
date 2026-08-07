import ref

def check(workdir):
    from sched.pipeline import PipelineScheduler
    m = {"traffic_util_ok": 0.0}
    ps = PipelineScheduler(4, 8)
    workload = [0.85, 0.90, 0.88]
    got = ps.evaluate_traffic(workload)
    expected = ref.oracle_traffic(workload)
    if abs(got - expected) < 1e-5:
        m["traffic_util_ok"] = 1.0
    return m
