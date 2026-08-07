from marlintp.retune import retune_marlin_config


def estimate_throughput(
    cfg: dict,
    batch_size: int,
    seq_len: int,
    memory_bandwidth_gbps: float = 900.0,
    compute_tflops: float = 312.0,
) -> dict:
    retune_res = retune_marlin_config(cfg)
    eligible = retune_res["eligible"]
    selected_bn = retune_res["block_n"]
    selected_bk = retune_res["block_k"]
    fallback = retune_res["fallback"]

    m_dim = batch_size * seq_len
    tp_size = cfg.get("tp_size", 1)
    mode = cfg.get("parallel_mode", "col")

    if mode == "col":
        n_rank = cfg["n"] // tp_size
        k_rank = cfg["k"]
    else:
        n_rank = cfg["n"]
        k_rank = cfg["k"] // tp_size

    flops = 2.0 * m_dim * k_rank * n_rank

    act_bytes = m_dim * k_rank * 2.0
    out_bytes = m_dim * n_rank * 2.0
    io_bytes = act_bytes + out_bytes

    weight_int4_bytes = k_rank * n_rank * 0.5
    group_size = cfg.get("group_size", -1)
    if group_size != -1:
        num_groups = k_rank / group_size
        scale_bytes = num_groups * n_rank * 2.0
    else:
        scale_bytes = n_rank * 2.0

    marlin_weight_bytes = weight_int4_bytes + scale_bytes
    marlin_mem_bytes = io_bytes + marlin_weight_bytes

    fallback_weight_bytes = k_rank * n_rank * 2.0 * 1.5
    fallback_mem_bytes = io_bytes + fallback_weight_bytes

    def _calc_tflops(bytes_count):
        t_comp = flops / (compute_tflops * 1e12)
        t_mem = bytes_count / (memory_bandwidth_gbps * 1e9)
        t_lat = max(t_comp, t_mem)
        lat_ms = t_lat * 1000.0
        tflops = flops / (t_lat * 1e12)
        return lat_ms, tflops

    active_bytes = marlin_mem_bytes if eligible else fallback_mem_bytes
    lat_ms, tflops = _calc_tflops(active_bytes)
    _, fb_tflops = _calc_tflops(fallback_mem_bytes)

    speedup = tflops / fb_tflops

    return {
        "eligible": eligible,
        "selected_block_n": selected_bn,
        "selected_block_k": selected_bk,
        "fallback": fallback,
        "flops": float(flops),
        "memory_bytes": float(active_bytes),
        "latency_ms": float(lat_ms),
        "throughput_tflops": float(tflops),
        "fallback_throughput_tflops": float(fb_tflops),
        "speedup_vs_fallback": float(speedup),
    }
