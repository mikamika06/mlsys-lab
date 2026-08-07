import ref


def check(workdir):
    from fp8read.analysis import compute_ratios
    from fp8read.parser import parse_logs

    records = parse_logs(ref.RAW_LOGS)
    got = compute_ratios(records)
    want = ref.compute_ratios(records)
    out = {"ratio_match": 0.0}
    if got == want:
        out["ratio_match"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
