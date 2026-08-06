import ref


def check(workdir):
    from ggml_isa.benchmark import compare_performance

    out = {"speedup_match": 0.0, "efficiency_valid": 0.0}
    test_pairs = [(125.5, 80.2), (45.0, 44.8), (200.0, 100.0)]
    matched = 0
    valid = 0
    for amx, avx in test_pairs:
        want = ref.compare_performance(amx, avx)
        got = compare_performance(amx, avx)
        if isinstance(got, dict) and abs(got.get("speedup", 0) - want["speedup"]) < 1e-5:
            matched += 1
        if got.get("efficient") == want["efficient"]:
            valid += 1

    out["speedup_match"] = 1.0 if matched == len(test_pairs) else 0.0
    out["efficiency_valid"] = 1.0 if valid == len(test_pairs) else 0.0
    return out
