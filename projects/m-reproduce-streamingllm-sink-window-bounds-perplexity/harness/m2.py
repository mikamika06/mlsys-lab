import ref


def check(workdir):
    from streamllm.sweep import find_optimal_sink

    seq_len = 8192
    window_size = 1024
    candidates = [0, 2, 4, 8, 16, 32]

    want = ref.sweep_sinks(seq_len, window_size, candidates)
    try:
        got = int(find_optimal_sink(seq_len, window_size, candidates))
    except Exception as e:
        return {"optimal_sink_match": 0.0, "_note": f"execution error: {e}"}

    match = 1.0 if got == want else 0.0
    return {"optimal_sink_match": match}
