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
