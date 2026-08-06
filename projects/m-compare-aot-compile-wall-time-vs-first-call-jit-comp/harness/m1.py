import sys
import os
import time
import ref

def check(workdir):
    for k in list(sys.modules.keys()):
        if k == "aot_compare" or k.startswith("aot_compare."):
            del sys.modules[k]

    sys.path.insert(0, workdir)

    out = {"time_ratio_rel_err": 1.0}

    class DeterministicClock:
        def __init__(self):
            self.now = 100.0

        def perf_counter(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    try:
        from aot_compare.profiling import compare_compilation_timings

        clock_got = DeterministicClock()
        orig_perf_counter = time.perf_counter
        orig_sleep = time.sleep
        time.perf_counter = clock_got.perf_counter
        time.sleep = clock_got.sleep

        try:
            mock_fn = ref.MockJitFunction()
            got = compare_compilation_timings(mock_fn, 1.0)
        finally:
            time.perf_counter = orig_perf_counter
            time.sleep = orig_sleep

        clock_want = DeterministicClock()
        time.perf_counter = clock_want.perf_counter
        time.sleep = clock_want.sleep
        try:
            mock_fn_ref = ref.MockJitFunction()
            want = ref.compare_compilation_timings(mock_fn_ref, 1.0)
        finally:
            time.perf_counter = orig_perf_counter
            time.sleep = orig_sleep

        if not isinstance(got, dict):
            out["_note"] = "compare_compilation_timings must return a dict"
            return out

        required_keys = ["aot_compile_time", "jit_first_call_time", "jit_cached_time", "overhead_ratio"]
        for k in required_keys:
            if k not in got:
                out["_note"] = f"missing key '{k}' in result dict"
                return out

        got_ratio = got["overhead_ratio"]
        want_ratio = want["overhead_ratio"]
        rel_err = abs(got_ratio - want_ratio) / (abs(want_ratio) + 1e-6)
        out["time_ratio_rel_err"] = float(rel_err)

    except Exception as e:
        out["_note"] = f"raised {type(e).__name__}: {e}"

    return out
