def triage_oom(timelines, vram_limit):
    fixes = []
    for timeline in timelines:
        peak = 0
        max_step = None
        for step in timeline:
            total = step["weights"] + step["hessian"] + step["activations"]
            if total > peak:
                peak = total
                max_step = step

        if peak <= vram_limit:
            fixes.append("ok")
        else:
            components = {
                "weights": max_step["weights"],
                "hessian": max_step["hessian"],
                "activations": max_step["activations"]
            }
            bottleneck = max(components.items(), key=lambda x: x[1])[0]
            if bottleneck == "weights":
                fixes.append("cpu_offload")
            elif bottleneck == "hessian":
                fixes.append("block_quant")
            else:
                fixes.append("reduce_calib_batch")
    return fixes
