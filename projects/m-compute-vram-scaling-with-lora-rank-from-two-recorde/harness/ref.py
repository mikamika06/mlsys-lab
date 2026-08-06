import numpy as np

LOG_PAIRS = [
    (
        {"rank": 8, "peak_vram_bytes": 14_000_000_000.0, "step_flops": 1.2e14},
        {"rank": 32, "peak_vram_bytes": 18_800_000_000.0, "step_flops": 1.8e14},
        64,
    ),
    (
        {"rank": 16, "peak_vram_bytes": 22_000_000_000.0, "step_flops": 3.5e14},
        {"rank": 64, "peak_vram_bytes": 31_600_000_000.0, "step_flops": 4.7e14},
        128,
    ),
    (
        {"rank": 4, "peak_vram_bytes": 8_500_000_000.0, "step_flops": 8.0e13},
        {"rank": 16, "peak_vram_bytes": 10_300_000_000.0, "step_flops": 1.1e14},
        256,
    ),
]


def oracle_extract(log1, log2):
    r1, r2 = float(log1["rank"]), float(log2["rank"])
    dr = r2 - r1

    v1, v2 = float(log1["peak_vram_bytes"]), float(log2["peak_vram_bytes"])
    v_slope = (v2 - v1) / dr
    v_base = v1 - v_slope * r1

    f1, f2 = float(log1["step_flops"]), float(log2["step_flops"])
    f_slope = (f2 - f1) / dr
    f_base = f1 - f_slope * r1

    return {
        "vram_base": v_base,
        "vram_slope": v_slope,
        "flops_base": f_base,
        "flops_slope": f_slope,
    }


def oracle_predict(params, target_rank):
    tr = float(target_rank)
    return {
        "predicted_vram_bytes": float(params["vram_base"]) + float(params["vram_slope"]) * tr,
        "predicted_step_flops": float(params["flops_base"]) + float(params["flops_slope"]) * tr,
    }
