import ref

def check(workdir):
    from sched.pipeline import PipelineScheduler
    m = {"interleaved_ok": 0.0}
    ps = PipelineScheduler(4, 8)
    got = ps.interleaved_memory(2)
    expected = ref.oracle_interleaved(4, 2)
    if got == expected:
        m["interleaved_ok"] = 1.0
    return m
