import ref


def check(workdir):
    from specdec.metrics import expected_accepted_tokens

    out = {"expected_tokens_match": 0.0}
    ok = 0
    for case in ref.TEST_CASES:
        g = case["gamma"]
        p = case["p"]
        want = ref.compute_expected(g, p)
        try:
            got = expected_accepted_tokens(g, p)
        except Exception as e:
            out["_note"] = f"raised {type(e).__name__}: {e}"
            return out
        if isinstance(got, (int, float)) and abs(float(got) - float(want)) < 1e-5:
            ok += 1
        else:
            out["_note"] = f"expected_accepted_tokens({g}, {p}) got {got}, want {want}"
            return out
    out["expected_tokens_match"] = float(ok)
    return out
