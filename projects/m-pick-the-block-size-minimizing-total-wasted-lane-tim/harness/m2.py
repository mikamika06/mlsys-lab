import ref


def check(workdir):
    from lanewaste.optimizer import select_best_block_size

    out = {
        "optimal_index_matched": 0.0,
        "total": float(len(ref.TEST_WORKLOADS)),
    }
    matched = 0
    for i, item in enumerate(ref.TEST_WORKLOADS):
        n = item["n"]
        overhead = item["launch_overhead"]
        cands = item["candidates"]
        want_idx, want_val = ref.select_best_block_size(n, cands, overhead)
        got_idx, got_val = select_best_block_size(n, cands, overhead)
        if got_idx == want_idx and abs(got_val - want_val) < 1e-5:
            matched += 1
        elif "_note" not in out:
            out["_note"] = (
                f"workload {i}: got (idx={got_idx}, val={got_val}), want (idx={want_idx}, val={want_val})"
            )
    out["optimal_index_matched"] = float(matched)
    return out
