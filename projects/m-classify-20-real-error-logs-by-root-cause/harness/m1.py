import ref


def check(workdir):
    from gguf_triage.classifier import classify_log

    out = {"classifications_matched": 0.0, "total": float(len(ref.LOGS))}
    ok = 0
    for i, (log, want) in enumerate(ref.LOGS):
        try:
            got = classify_log(log)
        except Exception:
            got = None
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"log {i}: got {got}, reference {want}"
    out["classifications_matched"] = float(ok)
    return out
