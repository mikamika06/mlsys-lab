import ref


def check(workdir):
    from threadsweep.sweep import find_optimal_threads

    threads, samples = ref.generate_sweep_data()
    want_thread, want_idx = ref.optimal_thread_count(threads, samples)

    out = {"argmin_index": 0.0}
    try:
        got_thread, got_idx = find_optimal_threads(threads, samples)
    except Exception as e:
        out["_note"] = f"execution failed: {type(e).__name__}: {str(e)[:100]}"
        return out

    if got_idx == want_idx and got_thread == want_thread:
        out["argmin_index"] = 1.0
    else:
        out["_note"] = f"got thread {got_thread} (idx {got_idx}), want thread {want_thread} (idx {want_idx})"
    return out
