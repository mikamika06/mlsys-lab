import ref


def check(workdir):
    from kvblock.simulator import find_optimal_block_size, simulate_block_sweep

    out = {"sweep_costs_matched": 0.0, "optimal_block_size_matched": 0.0}

    want_costs = ref.ref_simulate_block_sweep(
        ref.TRACE, ref.CANDIDATE_BLOCK_SIZES, ref.TOTAL_MEMORY_BLOCKS, ref.HIT_PENALTY_WEIGHT
    )
    got_costs = simulate_block_sweep(
        ref.TRACE, ref.CANDIDATE_BLOCK_SIZES, ref.TOTAL_MEMORY_BLOCKS, ref.HIT_PENALTY_WEIGHT
    )

    costs_ok = True
    for b_size in ref.CANDIDATE_BLOCK_SIZES:
        if b_size not in got_costs or abs(want_costs[b_size] - got_costs[b_size]) > 1e-5:
            costs_ok = False
            out["_note"] = f"cost mismatch at block_size={b_size}: got {got_costs.get(b_size)}, want {want_costs[b_size]}"
            break

    if costs_ok:
        out["sweep_costs_matched"] = 1.0

    want_best = ref.ref_find_optimal_block_size(
        ref.TRACE, ref.CANDIDATE_BLOCK_SIZES, ref.TOTAL_MEMORY_BLOCKS, ref.HIT_PENALTY_WEIGHT
    )
    got_best = find_optimal_block_size(
        ref.TRACE, ref.CANDIDATE_BLOCK_SIZES, ref.TOTAL_MEMORY_BLOCKS, ref.HIT_PENALTY_WEIGHT
    )

    if want_best == got_best:
        out["optimal_block_size_matched"] = 1.0
    else:
        out["_note"] = f"optimal block_size mismatch: got {got_best}, want {want_best}"

    return out
