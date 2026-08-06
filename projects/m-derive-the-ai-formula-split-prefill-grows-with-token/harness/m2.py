import ref

def check(workdir):
    from profiling.classify import classify_phase_bound
    cases = ref.get_test_cases()
    matches = 0
    total = len(cases) * 2

    for case in cases:
        for phase in ["prefill", "decode"]:
            want = ref.classify_phase(case, phase, tokens=64 if phase == "prefill" else 10)
            got = classify_phase_bound(
                case,
                phase,
                tokens=64 if phase == "prefill" else 10,
                peak_flops=case["peak_flops"],
                peak_bandwidth_gbps=case["peak_bandwidth_gbps"]
            )
            if got.get("bound") == want["bound"]:
                matches += 1

    match_ratio = float(matches) / float(total)
    out = {"classification_match": match_ratio}
    if match_ratio < 1.0:
        out["_note"] = f"Classification match ratio {match_ratio:.2f} is below 1.0."
    return out
