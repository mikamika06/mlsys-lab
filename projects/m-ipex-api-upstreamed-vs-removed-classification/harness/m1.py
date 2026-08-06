import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    try:
        from ipexaudit.classify import classify_api_batch, classify_api_call
    except Exception as e:
        return {"classified_correctly": 0.0, "_note": f"Import failed: {e}"}

    out = {"classified_correctly": 0.0}
    ok = True

    for api in ref.API_TEST_CASES:
        want = ref.classify_api_call(api)
        try:
            got = classify_api_call(api)
        except Exception as e:
            out["_note"] = f"classify_api_call('{api}') raised {e}"
            ok = False
            break

        if got != want:
            out["_note"] = f"For '{api}': expected {want}, got {got}"
            ok = False
            break

    if ok:
        try:
            batch_want = ref.classify_api_batch(ref.API_TEST_CASES)
            batch_got = classify_api_batch(ref.API_TEST_CASES)
            if batch_got == batch_want:
                out["classified_correctly"] = 1.0
            else:
                out["_note"] = f"Batch classification mismatched ref"
        except Exception as e:
            out["_note"] = f"classify_api_batch raised {e}"

    return out
