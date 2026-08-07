import ref


def check(workdir):
    from kvblock.trace import find_optimal_block_size, total_overhead

    out = {"argmin_index": 0.0, "traces_matched": 0.0, "total_traces": float(len(ref.TRACES))}
    matched = 0

    for i, trace in enumerate(ref.TRACES):
        want_costs = ref.total_overhead(trace, ref.CANDIDATES)
        want_best = ref.find_optimal_block_size(trace, ref.CANDIDATES)

        got_costs = total_overhead(trace, ref.CANDIDATES)
        got_best = find_optimal_block_size(trace, ref.CANDIDATES)

        if got_costs == want_costs and got_best == want_best:
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"trace {i}: expected best {want_best}, got {got_best}"

    out["traces_matched"] = float(matched)
    if matched == len(ref.TRACES):
        out["argmin_index"] = 1.0

    return out
