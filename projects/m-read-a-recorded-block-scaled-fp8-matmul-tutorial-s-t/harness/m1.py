import ref


def check(workdir):
    from fp8read.parser import parse_logs

    got = parse_logs(ref.RAW_LOGS)
    want = ref.parse_logs(ref.RAW_LOGS)
    out = {"records_parsed": 0.0}
    if got == want and len(got) == len(ref.RAW_LOGS):
        out["records_parsed"] = float(len(got))
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
