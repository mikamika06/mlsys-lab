import ref


def check(workdir):
    from vllm_pred.parser import parse_log

    out = {"logs_parsed": 0.0}
    ok = 0
    for log_text in ref.LOGS:
        want = ref.parse_log(log_text)
        got = parse_log(log_text)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"got {got}, want {want}"
    out["logs_parsed"] = float(ok)
    return out
