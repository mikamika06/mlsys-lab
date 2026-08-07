import ref


def check(workdir):
    from runner_audit.audit import detect_context_mismatch
    data = ref.make_test_cases()

    out = {"context_mismatch_detected": 0.0}
    try:
        res_good = detect_context_mismatch(data["matching_a"], data["matching_b"])
        res_bad = detect_context_mismatch(data["mismatch_a"], data["mismatch_b"])
        if not res_good and res_bad:
            out["context_mismatch_detected"] = 1.0
        else:
            out["_note"] = f"expected False for matching and True for mismatched, got {res_good} and {res_bad}"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {e}"
    return out
