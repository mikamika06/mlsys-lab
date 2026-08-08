import ref


def check(workdir):
    from speculative.eval import classify_pairing, compare_families

    out = {"comparison_match": 0.0, "threshold_match": 0.0, "_note": ""}
    comp_ok = 0
    thresh_ok = 0
    for i, pair in enumerate(ref.PAIRS):
        want_comp = ref.compare_families(pair["same"], pair["cross"])
        got_comp = compare_families(pair["same"], pair["cross"])
        if abs(want_comp - got_comp) < 1e-5:
            comp_ok += 1

        want_thresh = ref.classify_pairing(pair["same"], 0.7)
        got_thresh = classify_pairing(pair["same"], 0.7)
        if want_thresh == got_thresh:
            thresh_ok += 1

    if comp_ok == len(ref.PAIRS):
        out["comparison_match"] = 1.0
    if thresh_ok == len(ref.PAIRS):
        out["threshold_match"] = 1.0
    return out
