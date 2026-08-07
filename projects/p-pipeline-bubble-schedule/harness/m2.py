import ref

def check(workdir):
    from sched.pipeline import PipelineScheduler
    m = {"steady_state_ok": 0.0}
    ps = PipelineScheduler(4, 8)
    sched = ps.schedule_1f1b()
    expected = ref.oracle_1f1b(4, 8)
    if len(sched) == len(expected):
        m["steady_state_ok"] = 1.0
    return m
