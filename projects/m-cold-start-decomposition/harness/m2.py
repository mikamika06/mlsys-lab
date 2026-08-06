import ref

def check(workdir):
    from ort_perf.profiler import measure_breakdown, ep_delta

    out = {
        "creation_rel_err": 1.0,
        "first_run_rel_err": 1.0,
        "steady_rel_err": 1.0,
        "speedup_rel_err": 1.0
    }

    class Factory:
        def __init__(self, kind, ts):
            self.kind = kind
            self.ts = ts
        def __call__(self):
            return ref.MockSession(self.kind, self.ts)

    def make_time_fn(ts):
        def time_fn():
            return ts["now"]
        return time_fn

    try:
        ts1 = {"now": 0.0}
        ts2 = {"now": 0.0}
        gpu_f_got = Factory("GPU", ts1)
        gpu_f_want = Factory("GPU", ts2)

        got_br = measure_breakdown(gpu_f_got, None, make_time_fn(ts1), 5)
        want_br = ref.measure_breakdown(gpu_f_want, None, make_time_fn(ts2), 5)

        ts3 = {"now": 0.0}
        ts4 = {"now": 0.0}
        cpu_f_got = Factory("CPU", ts3)
        gpu_f2_got = Factory("GPU", ts3)

        cpu_f_want = Factory("CPU", ts4)
        gpu_f2_want = Factory("GPU", ts4)

        got_delta = ep_delta(cpu_f_got, gpu_f2_got, None, make_time_fn(ts3), 5)
        want_delta = ref.ep_delta(cpu_f_want, gpu_f2_want, None, make_time_fn(ts4), 5)

        def rel_err(g, w):
            return abs(g - w) / abs(w) if w != 0 else abs(g)

        out["creation_rel_err"] = rel_err(got_br["creation"], want_br["creation"])
        out["first_run_rel_err"] = rel_err(got_br["first_run"], want_br["first_run"])
        out["steady_rel_err"] = rel_err(got_br["steady_step"], want_br["steady_step"])
        out["speedup_rel_err"] = rel_err(got_delta["steady_speedup"], want_delta["steady_speedup"])
    except Exception as e:
        out["_note"] = str(e)

    return out
