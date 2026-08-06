import ref


def check(workdir):
    from itl_debug.metrics import detect_period

    latencies = ref.generate_trace(64, 1500)
    want = ref.detect_period(latencies)
    got = detect_period(latencies)
    matched = 1 if got == want else 0
    out = {"period_matched": float(matched)}
    if matched != 1:
        out["_note"] = f"expected period {want}, got {got}"
    return out
