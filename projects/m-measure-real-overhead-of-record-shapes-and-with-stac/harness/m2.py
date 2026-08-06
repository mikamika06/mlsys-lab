import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from profoverhead.measure import calculate_doubling_ratio

    out = {"throughput_ratio": 0.0}

    test_cases = [
        (200, 10000, 2000),
        (500, 50000, 5000),
        (100, 12000, 1200),
        (350, 8000, 8000),
    ]

    ok = True
    for base_t, total_e, active_e in test_cases:
        want = ref.calculate_doubling_ratio(base_t, total_e, active_e)
        got = calculate_doubling_ratio(base_t, total_e, active_e)
        if abs(float(got) - want) > 1e-5:
            ok = False
            out["_note"] = f"for inputs ({base_t}, {total_e}, {active_e}): expected {want}, got {got}"
            break

    if ok:
        out["throughput_ratio"] = 1.0
    return out
