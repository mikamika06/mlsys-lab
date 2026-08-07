import ref


def check(workdir):
    from serve.runner import simulate_decode_step
    from serve.metrics import compute_throughput_ratio, compute_memory_delta

    out = {"throughput_ratio_match": 0.0, "memory_delta_match": 0.0}
    cfg_fp = ref.CONFIGS[0]
    cfg_w4 = ref.CONFIGS[1]

    res_fp = simulate_decode_step(cfg_fp)
    res_w4 = simulate_decode_step(cfg_w4)

    want_ratio = ref.compute_throughput_ratio(
        res_fp["tokens_per_sec"], res_w4["tokens_per_sec"]
    )
    got_ratio = compute_throughput_ratio(
        res_fp["tokens_per_sec"], res_w4["tokens_per_sec"]
    )

    want_delta = ref.compute_memory_delta(
        res_fp["memory_bytes"], res_w4["memory_bytes"]
    )
    got_delta = compute_memory_delta(
        res_fp["memory_bytes"], res_w4["memory_bytes"]
    )

    if abs(got_ratio - want_ratio) < 1e-5:
        out["throughput_ratio_match"] = 1.0
    if abs(got_delta - want_delta) < 1e-5:
        out["memory_delta_match"] = 1.0

    return out
