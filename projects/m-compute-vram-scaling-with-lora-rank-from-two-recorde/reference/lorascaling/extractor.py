def extract_rank_scaling(log1, log2):
    r1, r2 = float(log1["rank"]), float(log2["rank"])
    if r1 == r2:
        raise ValueError("Logs must have distinct LoRA ranks")

    dr = r2 - r1

    v1, v2 = float(log1["peak_vram_bytes"]), float(log2["peak_vram_bytes"])
    vram_slope = (v2 - v1) / dr
    vram_base = v1 - vram_slope * r1

    f1, f2 = float(log1["step_flops"]), float(log2["step_flops"])
    flops_slope = (f2 - f1) / dr
    flops_base = f1 - flops_slope * r1

    return {
        "vram_base": vram_base,
        "vram_slope": vram_slope,
        "flops_base": flops_base,
        "flops_slope": flops_slope,
    }
