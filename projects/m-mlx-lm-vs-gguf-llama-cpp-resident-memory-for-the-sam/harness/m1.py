import ref


def check(workdir):
    from moe_bench.memory import estimate_resident_memory

    out = {"memory_ratios_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        mlx_want = ref.compute_mlx_memory(cfg, 16)
        gguf_want = ref.compute_gguf_memory(cfg, 4)

        mlx_got, gguf_got = estimate_resident_memory(cfg)

        if abs(mlx_got - mlx_want) < 1000 and abs(gguf_got - gguf_want) < 1000:
            ok += 1

    out["memory_ratios_matched"] = float(ok)
    return out
