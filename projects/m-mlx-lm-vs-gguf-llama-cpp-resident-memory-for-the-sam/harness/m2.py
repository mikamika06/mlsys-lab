import ref


def check(workdir):
    from moe_bench.throughput import measure_throughput

    out = {"throughput_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        mlx_w = ref.compute_mlx_throughput(cfg)
        gguf_w = ref.compute_gguf_throughput(cfg)

        mlx_g, gguf_g = measure_throughput(cfg)

        if abs(mlx_g - mlx_w) < 0.5 and abs(gguf_g - gguf_w) < 0.5:
            ok += 1

    if ok == len(ref.CONFIGS):
        out["throughput_matched"] = 1.0
    return out
