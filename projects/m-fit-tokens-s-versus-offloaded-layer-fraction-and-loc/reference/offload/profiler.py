def find_offload_cliff(profiles):
    """Locate the offload fraction where tok/s drops dramatically.

    profiles is a dict mapping offload_fraction (float 0.0 to 1.0) to tok_s (float).
    The cliff fraction is the lowest offload fraction (highest GPU layer offload drop)
    where throughput drops by > 30% relative to pure GPU or preceding smooth degradation.
    """
    sorted_fracs = sorted(profiles.keys())
    if not sorted_fracs:
        return 0.0

    max_tok = max(profiles.values())
    cliff_frac = 1.0

    for i in range(1, len(sorted_fracs)):
        prev_f = sorted_fracs[i - 1]
        curr_f = sorted_fracs[i]
        prev_tok = profiles[prev_f]
        curr_tok = profiles[curr_f]

        if prev_tok > 0 and (prev_tok - curr_tok) / prev_tok >= 0.30:
            return curr_f

    return cliff_frac
