import ref

def check(workdir):
    from prefill.roofline import analyze_roofline
    cases = ref.get_roofline_cases()
    match_count = 0
    for i, c in enumerate(cases):
        want = ref.analyze_roofline(c)
        try:
            got = analyze_roofline(c)
        except Exception as e:
            return {"roofline_match": 0.0, "_note": f"case {i} raised {type(e).__name__}: {e}"}
        if got is None or not isinstance(got, dict):
            continue
        b_match = got.get("bound") == want["bound"]
        i_err = abs(got.get("arithmetic_intensity", 0.0) - want["arithmetic_intensity"]) / (want["arithmetic_intensity"] + 1e-9)
        if b_match and i_err < 0.05:
            match_count += 1
    return {"roofline_match": float(match_count)}
