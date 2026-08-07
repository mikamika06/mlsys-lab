import ref


def check(workdir):
    from triton_profiler.parser import compute_region_times
    tree = ref.get_test_tree()

    try:
        got = compute_region_times(tree)
    except Exception as e:
        return {"region_percentages_matched": 0.0, "_note": f"raised {type(e).__name__}"}

    want = {
        "root": 0.0,
        "load": 25.0,
        "compute": 60.0,
        "store": 15.0
    }

    if not isinstance(got, dict):
        return {"region_percentages_matched": 0.0, "_note": "did not return a dict"}

    matched = 1.0
    for k, v in want.items():
        if k not in got or abs(float(got[k]) - float(v)) > 1e-4:
            matched = 0.0
            break

    return {"region_percentages_matched": matched}
