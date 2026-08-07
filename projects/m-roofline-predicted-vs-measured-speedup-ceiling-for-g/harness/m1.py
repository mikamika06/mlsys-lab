import ref


def check(workdir):
    from roofline.calc import compute_intensity

    cases = ref.get_test_cases()
    matched = 0
    out = {}
    for i, c in enumerate(cases):
        got = compute_intensity(c["flops_per_token"], c["bytes_per_token"])
        want = c["expected_intensity"]
        if abs(got - want) / (abs(want) + 1e-9) < 1e-5:
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"

    out["intensity_match"] = 1.0 if matched == len(cases) else 0.0
    if matched != len(cases) and "_note" not in out:
        out["_note"] = f"matched {matched}/{len(cases)} intensities"
    return out
