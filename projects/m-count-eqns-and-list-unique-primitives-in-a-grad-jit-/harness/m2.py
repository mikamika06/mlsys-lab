import ref


def check(workdir):
    from jaxpr_tools.tracer import run_with_mutable_closure

    out = {"leak_detected": 0.0, "safe_execution": 0.0}
    try:
        status, _ = run_with_mutable_closure(ref.sample_function, 1.0)
        if status == "leaked":
            out["leak_detected"] = 1.0
            out["safe_execution"] = 1.0
        elif status == "safe":
            out["safe_execution"] = 1.0
    except Exception:
        pass
    return out
