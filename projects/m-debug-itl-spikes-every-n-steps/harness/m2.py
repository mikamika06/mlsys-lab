import ref


def check(workdir):
    from itl_debug.analyzer import find_root_cause

    latencies = ref.generate_trace(32, 1000)
    want = ref.find_root_cause(latencies, 32)
    got = find_root_cause(latencies, 32)
    matched = 1 if got and got.get("period") == want.get("period") else 0
    out = {"root_cause_matched": float(matched)}
    if matched != 1:
        out["_note"] = f"root cause analysis mismatch: got {got}"
    return out
