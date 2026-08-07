import sys


def check(workdir):
    sys.path.insert(0, workdir)
    m = {"trace_verification_ok": 0.0}

    try:
        from zero_planner.planner import ZeroPlanner

        planner = ZeroPlanner(num_params=10**8, bytes_per_param=2, bytes_per_optim_state=12)
        trace_good = {"peak_memory_bytes": 1000000000}
        trace_bad = {"peak_memory_bytes": 1500000000}

        v_good = planner.verify_trace(trace_good, predicted_mem_bytes=1050000000)
        v_bad = planner.verify_trace(trace_bad, predicted_mem_bytes=1050000000)

        if v_good is True and v_bad is False:
            m["trace_verification_ok"] = 1.0
    except Exception:
        pass

    return m
