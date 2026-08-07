import ref


def check(workdir):
    from streamllm.cache import compute_perplexity

    seq_len = 4096
    sink_size = 4
    window_size = 512

    want = ref.simulate_ppl(seq_len, sink_size, window_size, "sink_window")
    try:
        got = float(compute_perplexity(seq_len, sink_size, window_size, "sink_window"))
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"execution error: {e}"}

    rel = abs(got - want) / (abs(want) + 1e-8)
    return {"rel_err": float(rel)}
