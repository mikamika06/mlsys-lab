import ref

def check(workdir):
    from trtopt.profile import split_wide_profile, evaluate_profile_latency

    out = {"profiles_split_valid": 0.0, "latency_ratio": 1.0}

    split_profs = split_wide_profile(ref.WIDE_PROFILE)
    if isinstance(split_profs, list) and len(split_profs) >= 2:
        p1, p2 = split_profs[0], split_profs[1]
        valid_cover = (p1["min"][0] <= ref.WIDE_PROFILE["min"][0] and
                       p2["max"][0] >= ref.WIDE_PROFILE["max"][0] and
                       p1["opt"][0] != p2["opt"][0])
        if valid_cover:
            out["profiles_split_valid"] = 1.0

    baseline_latency = evaluate_profile_latency([ref.WIDE_PROFILE], ref.QUERY_SHAPES, ref.cost_function)
    split_latency = evaluate_profile_latency(split_profs, ref.QUERY_SHAPES, ref.cost_function)

    if baseline_latency > 0:
        ratio = float(split_latency / baseline_latency)
        out["latency_ratio"] = ratio
    else:
        out["_note"] = "Baseline latency calculated as 0"

    return out
