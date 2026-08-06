import ref

def check(workdir):
    from elastic.parser import parse_nccl_log
    out = {"logs_diagnosed": 0.0, "total": float(len(ref.LOGS))}
    ok = 0
    for log_text, expected in ref.LOGS:
        got = parse_nccl_log(log_text)
        if got == expected:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"log '{log_text}': got {got}, want {expected}"
    out["logs_diagnosed"] = float(ok)
    return out
