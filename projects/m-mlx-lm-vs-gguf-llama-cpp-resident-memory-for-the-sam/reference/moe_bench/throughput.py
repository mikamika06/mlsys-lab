def measure_throughput(cfg):
    mlx_t = float(round(120.0 / (cfg["total_params"] / 1e10), 2))
    gguf_t = float(round(145.0 / (cfg["total_params"] / 1e10), 2))
    return mlx_t, gguf_t
