import ref

def check(workdir):
    from ort_tune.threads import measure_thread_scaling
    m = {"thread_scaling_ok": 0.0}
    w = ref.get_workload()
    counts = [1, 2, 4, 8]
    res = measure_thread_scaling(w, counts)
    expected = ref.oracle_thread_scaling(w, counts)
    if isinstance(res, dict) and len(res) == len(counts):
        match = all(abs(res[t] - expected[t]) < 1e-5 for t in counts)
        if match:
            m["thread_scaling_ok"] = 1.0
    return m
