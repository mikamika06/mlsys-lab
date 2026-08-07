import ref


def check(workdir):
    from ringattn.imbalance import compute_imbalance

    out = {"imbalance_matched": 0.0}
    cases = ref.get_test_cases()
    matched = 0
    for c in cases:
        want = ref.compute_imbalance(c["seq_len"], c["num_ranks"])
        got = compute_imbalance(c["seq_len"], c["num_ranks"])
        if got and isinstance(got, dict) and got.get("imbalance_ratio") == want["imbalance_ratio"]:
            matched += 1
    out["imbalance_matched"] = float(matched)
    return out
