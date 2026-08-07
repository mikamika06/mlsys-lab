import ref

def check(workdir):
    from sched.pipeline import PipelineScheduler
    m = {"zb_schedule_ok": 0.0}
    ps = PipelineScheduler(4, 8)
    res = ps.zero_bubble_schedule()
    if isinstance(res, dict) and res.get("valid") is True:
        m["zb_schedule_ok"] = 1.0
    return m
