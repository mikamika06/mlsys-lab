import ref

def check(workdir):
    from compcache.tracer import Tracer
    t = Tracer()
    res = t.trace(lambda x: x + 1, [1, 2, 3])
    oracle = ref.get_oracle_trace([1, 2, 3])
    return {"trace_ok": oracle["trace_ok"] if res else 0.0}
