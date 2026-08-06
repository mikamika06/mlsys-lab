import ref


def check(workdir):
    from allocator.metrics import run_trace

    out = {"traces_passed": 0.0, "exact_matches": 0.0}
    ok = 0
    for i, trace in enumerate(ref.TRACES):
        ref_res = ref.get_ref_metrics(i)
        try:
            got_res = run_trace(trace)
            if got_res == ref_res:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"Trace {i} mismatch: got {got_res}, want {ref_res}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Trace {i} raised {type(e).__name__}: {e}"
    out["traces_passed"] = float(ok)
    out["exact_matches"] = float(ok)
    return out
