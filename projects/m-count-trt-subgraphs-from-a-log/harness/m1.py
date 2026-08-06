import ref


def check(workdir):
    from trtlog.parser import parse_log
    out = {"subgraphs_matched": 0.0, "total": float(len(ref.LOGS))}
    ok = 0
    for i, log in enumerate(ref.LOGS):
        want = ref.parse_log(log)
        got = parse_log(log)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"log {i}: got {got}, want {want}"
    out["subgraphs_matched"] = float(ok)
    return out
