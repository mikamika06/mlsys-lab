import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)

    from runnerdiag.skew import check_version_skew, identify_skew_field

    out = {"skew_detected": 0.0, "field_named": 0.0}

    cases = ref.generate_skew_test_cases()
    passed_skew = 0
    for client_v, payload in cases:
        got = check_version_skew(client_v, payload)
        expected_skew = (client_v != payload.get("version"))
        if isinstance(got, dict) and got.get("has_skew") == expected_skew:
            passed_skew += 1

    if passed_skew == len(cases):
        out["skew_detected"] = 1.0
    else:
        out["_note"] = f"Failed skew cases: passed {passed_skew}/{len(cases)}"

    if identify_skew_field() == "version":
        out["field_named"] = 1.0
    else:
        out["_note"] = f"Expected skew field 'version', got '{identify_skew_field()}'"

    return out
