import ref


def check(workdir):
    from deploy.diagnose import diagnose_failure

    out = {"diagnoses_matched": 0.0}
    ok = 0

    test_cases = [
        (ref.LOGS[0], ref.SPECS[1]),
        (ref.LOGS[1], ref.SPECS[0]),
        (ref.LOGS[2], ref.SPECS[0])
    ]

    for i, (log, spec) in enumerate(test_cases):
        want = ref.diagnose_failure(log, spec)
        try:
            got = diagnose_failure(log, spec)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} raised error: {type(e).__name__}: {str(e)[:100]}"
            continue

        if isinstance(got, dict) and set(got.get("issues", [])) == set(want.get("issues", [])):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"

    out["diagnoses_matched"] = float(ok)
    return out
