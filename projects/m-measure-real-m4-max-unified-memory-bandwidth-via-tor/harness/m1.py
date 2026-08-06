import ref

def check(workdir):
    from mprofiler.bandwidth import compute_bandwidth_percentage
    out = {"bandwidth_matched": 0.0}
    ok = 0
    for case in ref.TEST_CASES:
        got = compute_bandwidth_percentage(case["bytes"], case["duration"], case["peak"])
        want = (case["bytes"] / 1e9 / case["duration"] / case["peak"]) * 100.0
        if abs(got - want) < 1e-3:
            ok += 1
    out["bandwidth_matched"] = float(ok)
    return out
