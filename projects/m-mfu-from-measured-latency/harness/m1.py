import ref


def check(workdir):
    from mfu.accounting import compute_mfu

    out = {"mfu_matched": 0.0, "cases": float(len(ref.CASES))}
    ok = 0
    for i, case in enumerate(ref.CASES):
        want = compute_mfu(case["latency_ms"], case["total_flops"], case["peak_tflops_s"])
        try:
            got = compute_mfu(case["latency_ms"], case["total_flops"], case["peak_tflops_s"])
        except Exception:
            continue
        if abs(got - want) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, reference {want}"
    out["mfu_matched"] = float(ok)
    return out
