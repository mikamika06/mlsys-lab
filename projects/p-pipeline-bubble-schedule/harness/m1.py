import ref

def check(workdir):
    from sched.pipeline import PipelineScheduler
    m = {"utilization_ok": 0.0}
    ps = PipelineScheduler(4, 8)
    expected = ref.oracle_gpipe(4, 8)
    got = ps.gpipe_utilization()
    if abs(got - expected) < 1e-5:
        m["utilization_ok"] = 1.0
    return m
