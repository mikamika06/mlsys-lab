import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from bench.measure import measure_warmup_bias
    except ImportError as e:
        return {"_note": f"ImportError: {e}", "steady_err": 1.0, "bias_err": 1.0}

    engine_ref = ref.MockEngine("mlx")
    engine_stu = ref.MockEngine("mlx")

    want_s, want_b = ref.measure_warmup_bias(engine_ref, 128, 2, 5, "fp16")
    try:
        got_s, got_b = measure_warmup_bias(engine_stu, 128, 2, 5, "fp16")
    except Exception as e:
        return {"_note": f"Error running measure_warmup_bias: {e}", "steady_err": 1.0, "bias_err": 1.0}

    out = {}
    out["steady_err"] = abs(want_s - got_s) / max(1e-5, want_s)
    out["bias_err"] = abs(want_b - got_b) / max(1e-5, want_b)
    if out["steady_err"] > 1e-3 or out["bias_err"] > 1e-3:
        out["_note"] = f"Expected ({want_s}, {want_b}), got ({got_s}, {got_b})"

    return out
