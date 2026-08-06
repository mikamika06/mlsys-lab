import ref


def check(workdir):
    from gguf_triage.classifier import get_fixing_command

    out = {"commands_matched": 0.0, "total": float(len(ref.LOGS))}
    ok = 0
    for i, (log, cause) in enumerate(ref.LOGS):
        want = ref.FIXES.get(cause)
        try:
            got = get_fixing_command(cause)
        except Exception:
            got = None
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"cause {cause}: got {got}, reference {want}"
    out["commands_matched"] = float(ok)
    return out
