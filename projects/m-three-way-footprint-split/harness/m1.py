import ref


def check(workdir):
    from footprint.split import analyze_three_way_footprint

    out = {"splits_matched": 0.0}
    ok = 0
    for i, (b_info, r_cfg, tensors) in enumerate(ref.SPLIT_TEST_CASES):
        want = ref.analyze_three_way_footprint(b_info, r_cfg, tensors)
        got = analyze_three_way_footprint(b_info, r_cfg, tensors)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, reference {want}"

    out["splits_matched"] = float(ok)
    return out
