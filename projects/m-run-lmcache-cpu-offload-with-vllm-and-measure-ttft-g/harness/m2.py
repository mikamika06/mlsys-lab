import ref


def check(workdir):
    from offload.diagnose import diagnose_transfer_log

    out = {"diagnoses_matched": 0.0}
    ok = 0
    for logs in ref.LOG_BATCHES:
        want = ref.diagnose_transfer_log(logs)
        try:
            got = diagnose_transfer_log(logs)
        except Exception as e:
            out["_note"] = f"diagnose raised {type(e).__name__}: {e}"
            return out
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"got {got}, want {want}"
    out["diagnoses_matched"] = float(ok)
    return out
