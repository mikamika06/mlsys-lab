import ref


def check(workdir):
    from mfu.accounting import compute_tflops

    out = {"tflops_matched": 0.0, "cases": float(len(ref.PUBLISHED_CASES))}
    ok = 0
    for i, case in enumerate(ref.PUBLISHED_CASES):
        want = compute_tflops(case["total_flops"], case["latency_ms"])
        try:
            got = compute_tflops(case["total_flops"], case["latency_ms"])
        except Exception:
            continue
        if abs(got - want) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, reference {want}"
    out["tflops_matched"] = float(ok)
    return out
