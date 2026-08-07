import ref


def check(workdir):
    from vllm_compat.bnb import detect_backend

    matched = 0
    total = len(ref.BNB_TESTS)
    for t in ref.BNB_TESTS:
        got = detect_backend(t["env"])
        want = ref.detect_bnb(t["env"])
        got_features = sorted(got.get("features", []))
        want_features = sorted(want.get("features", []))
        if (got.get("backend") == want.get("backend") and
            got.get("supported") == want.get("supported") and
            got_features == want_features):
            matched += 1

    return {"backends_matched": float(matched), "total": float(total)}
