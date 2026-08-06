import ref


def check(workdir):
    from errorclass.signature import classify_signature

    cases = ref.get_test_cases()
    matched = 0
    out = {"signatures_matched": 0.0}

    for i, (r, t, meta, expected) in enumerate(cases):
        try:
            got = classify_signature(r, t, meta)
            if got == expected:
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"case {i}: got {got}, expected {expected}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} raised error: {type(e).__name__}: {str(e)[:100]}"

    out["signatures_matched"] = float(matched)
    return out
