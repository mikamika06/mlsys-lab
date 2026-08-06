import ref


def check(workdir):
    from buildfix.abi import predict_abi_mismatch
    ok = 0
    for t1, t2, want in ref.TEST_CASES_ABI:
        got = predict_abi_mismatch(t1, t2)
        if bool(got) == bool(want):
            ok += 1
    return {"abi_matched": 1.0 if ok == len(ref.TEST_CASES_ABI) else 0.0}
