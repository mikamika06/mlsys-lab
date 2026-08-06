def predict_rank_requirements(params, target_rank):
    tr = float(target_rank)
    vram_est = float(params["vram_base"]) + float(params["vram_slope"]) * tr
    flops_est = float(params["flops_base"]) + float(params["flops_slope"]) * tr
    return {
        "predicted_vram_bytes": vram_est,
        "predicted_step_flops": flops_est,
    }
