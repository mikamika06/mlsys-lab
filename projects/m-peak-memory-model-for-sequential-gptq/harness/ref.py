VRAM_LIMIT = 2 * 1024**3

CONFIGS = [
    {"in_features": 4096, "out_features": 4096, "calib_samples": 128, "seq_len": 2048},
    {"in_features": 2048, "out_features": 2048, "calib_samples": 256, "seq_len": 4096},
    {"in_features": 8192, "out_features": 8192, "calib_samples": 64, "seq_len": 2048},
    {"in_features": 4096, "out_features": 1024, "calib_samples": 512, "seq_len": 1024},
    {"in_features": 32768, "out_features": 1024, "calib_samples": 8, "seq_len": 128},
    {"in_features": 16384, "out_features": 16384, "calib_samples": 4, "seq_len": 128},
    {"in_features": 24576, "out_features": 4096, "calib_samples": 2, "seq_len": 256},
    {"in_features": 32768, "out_features": 8192, "calib_samples": 1, "seq_len": 64},
    {"in_features": 1024, "out_features": 131072, "calib_samples": 1, "seq_len": 32},
    {"in_features": 2048, "out_features": 262144, "calib_samples": 1, "seq_len": 16},
    {"in_features": 512, "out_features": 524288, "calib_samples": 2, "seq_len": 8},
    {"in_features": 1024, "out_features": 1024, "calib_samples": 8, "seq_len": 128},
    {"in_features": 2048, "out_features": 2048, "calib_samples": 16, "seq_len": 256},
    {"in_features": 512, "out_features": 512, "calib_samples": 128, "seq_len": 128},
    {"in_features": 4096, "out_features": 4096, "calib_samples": 4, "seq_len": 512},
]

def simulate_timeline(in_features, out_features, calib_samples, seq_len):
    w_fp16 = in_features * out_features * 2
    w_q4 = (in_features * out_features) // 2
    acts = calib_samples * seq_len * in_features * 2
    hessian = in_features * in_features * 4

    return [
        {"phase": "load_weights", "weights": w_fp16, "hessian": 0, "activations": 0},
        {"phase": "load_activations", "weights": w_fp16, "hessian": 0, "activations": acts},
        {"phase": "compute_hessian", "weights": w_fp16, "hessian": hessian, "activations": acts},
        {"phase": "quantize", "weights": w_q4, "hessian": hessian, "activations": acts},
        {"phase": "done", "weights": w_q4, "hessian": 0, "activations": 0}
    ]

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
