import ref


def check(workdir):
    from cacheval.verify import verify_prefill_update

    cases = ref.get_test_cases()
    passed = 0
    for ref_cache, cand_cache, tol in cases:
        if verify_prefill_update(ref_cache, cand_cache, tol):
            passed += 1
    out = {"verifications_passed": float(passed)}
    if passed < len(cases):
        out["_note"] = f"passed {passed} out of {len(cases)} verification test cases"
    return out
