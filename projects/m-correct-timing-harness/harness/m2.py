import ref


def check(workdir):
    from timing.audit import audit_benchmark
    out = {"audit_detected": 0.0}

    flawed_code = "def run():\n    start = time.time()\n    kernel()\n    return time.time() - start"
    good_code = "def run():\n    start.record()\n    kernel()\n    end.record()\n    torch.cuda.synchronize()\n    return start.elapsed_time(end)"

    try:
        res_flawed = audit_benchmark(flawed_code)
        res_good = audit_benchmark(good_code)
        if res_flawed.get("is_flawed") is True and res_good.get("is_flawed") is False:
            out["audit_detected"] = 1.0
        else:
            out["_note"] = f"audit failed: flawed={res_flawed}, good={res_good}"
    except Exception as e:
        out["_note"] = f"audit_benchmark raised {type(e).__name__}: {str(e)[:120]}"
    return out
