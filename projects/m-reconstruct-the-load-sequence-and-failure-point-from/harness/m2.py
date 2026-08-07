import ref


def check(workdir):
    from runner.diagnostics import classify_missing_asset

    contexts = ref.generate_error_contexts()
    want = [classify_missing_asset(c) for c in contexts]

    try:
        from skeleton.runner.diagnostics import classify_missing_asset as skeleton_classify
        got_skel = [skeleton_classify(c) for c in contexts]
    except Exception:
        got_skel = None

    if got_skel == want:
        return {"classification_matched": 0.0, "_note": "skeleton passes check"}

    got = []
    for c in contexts:
        try:
            got.append(classify_missing_asset(c))
        except Exception:
            got.append(None)

    matched = 1.0 if got == want else 0.0
    out = {"classification_matched": matched}
    if matched == 0.0:
        out["_note"] = f"got {got}, want {want}"
    return out
