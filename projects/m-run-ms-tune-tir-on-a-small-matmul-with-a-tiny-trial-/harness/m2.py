import ref


def check(workdir):
    out = {"speedup_valid": 0.0}
    try:
        from tuntir.tune import compare_schedules
        res = compare_schedules(workdir)
        if isinstance(res, dict) and "default_latency" in res and "tuned_latency" in res:
            if res["tuned_latency"] <= res["default_latency"]:
                out["speedup_valid"] = 1.0
            else:
                out["_note"] = f"tuned latency {res['tuned_latency']} not better than default {res['default_latency']}"
        else:
            out["_note"] = "compare_schedules did not return expected dictionary structure"
    except Exception as e:
        out["_note"] = f"exception during compare_schedules: {type(e).__name__}: {e}"
    return out
