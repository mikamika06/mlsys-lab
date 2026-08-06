import ref


def check(workdir):
    from coalesce.traffic import min_dram_traffic

    out = {"traffic_matched": 0.0}
    ok = 0
    for case in ref.TEST_CASES:
        want = ref.min_dram_traffic(case["num_elements"], case["element_size"], case["stride"])
        got = min_dram_traffic(case["num_elements"], case["element_size"], case["stride"])
        if want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"stride {case['stride']}: got {got}, want {want}"
    out["traffic_matched"] = float(ok)
    return out
