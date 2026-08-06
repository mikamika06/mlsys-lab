import ref


def check(workdir):
    from embedopt.batching import select_optimal_batch_size

    out = {"optimal_batch_selected": 0.0}
    candidates, profile_fn, memory_cap = ref.generate_benchmark_data(seed=42)

    want_bs, want_tp = ref.compute_optimal_batch_size(candidates, profile_fn, memory_cap)
    try:
        got_bs, got_tp = select_optimal_batch_size(candidates, profile_fn, memory_cap)
        if got_bs == want_bs and abs(got_tp - want_tp) < 1e-5:
            out["optimal_batch_selected"] = 1.0
        else:
            out["_note"] = f"Expected batch size {want_bs} with throughput {want_tp}, got {got_bs} with {got_tp}"
    except Exception as e:
        out["_note"] = f"Error during batch selection: {type(e).__name__}: {str(e)}"

    return out
