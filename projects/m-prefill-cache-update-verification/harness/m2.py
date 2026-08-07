import ref


def check(workdir):
    from cacheverify.metrics import compute_latencies, peak_memory_delta
    p_len, d_steps = 128, 10
    s_params = {"base_overhead": 1.5, "per_token": 0.01, "decode_per_token": 0.05}
    sl_params = {"base_overhead": 1.0, "per_token": 0.012, "decode_per_token": 0.05, "io_penalty": 0.02}

    got_lat = compute_latencies(p_len, d_steps, s_params, sl_params)
    want_lat = ref.compute_latencies(p_len, d_steps, s_params, sl_params)
    lat_match = 1.0 if abs(got_lat["stateful_total"] - want_lat["stateful_total"]) < 1e-5 and abs(got_lat["stateless_total"] - want_lat["stateless_total"]) < 1e-5 else 0.0

    shape = (32, 2, 512, 64)
    dtype_b = 2
    got_mem = peak_memory_delta(shape, dtype_b)
    want_mem = ref.peak_memory_delta(shape, dtype_b)
    mem_match = 1.0 if abs(got_mem - want_mem) < 1e-5 else 0.0

    return {"latency_match": lat_match, "memory_delta_match": mem_match}
