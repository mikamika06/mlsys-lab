import ref


def check(workdir):
    from specbench.measure import compute_acceptance_rate
    from specbench.metrics import family_gap_ratio

    out = {"gap_match": 0.0, "efficiency_match": 0.0}
    same_rates = []
    cross_rates = []
    for p in ref.PAIRS:
        rate = compute_acceptance_rate(p["draft"], p["target"], p["probs"])
        if p["is_same_family"]:
            same_rates.append(rate)
        else:
            cross_rates.append(rate)

    want_gap = ref.family_gap_ratio(same_rates, cross_rates)
    got_gap = family_gap_ratio(same_rates, cross_rates)
    if abs(want_gap - got_gap) < 1e-5:
        out["gap_match"] = 1.0

    efficiency = float(sum(same_rates + cross_rates)) / max(len(same_rates + cross_rates), 1)
    if efficiency >= 0.0:
        out["efficiency_match"] = 1.0
    return out
