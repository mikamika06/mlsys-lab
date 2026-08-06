import numpy as np


def make_sample_layers(seed=42):
    rng = np.random.RandomState(seed)
    layers = []
    shapes = [(32, 64), (64, 32), (16, 64)]
    for i, (out_d, in_d) in enumerate(shapes):
        w = rng.randn(out_d, in_d).astype(np.float32)
        b = rng.randn(out_d).astype(np.float32)
        rmin, rmax = float(np.min(w)), float(np.max(w))
        layers.append({
            "name": f"layer_{i}",
            "weights": w,
            "bias": b,
            "calibration_range": (rmin, rmax)
        })
    return layers


def make_hw_config():
    return {
        "memory_bandwidth_gbps": 15.0,
        "fp32_tflops": 0.5,
        "fp16_tflops": 1.0,
        "int8_tops": 3.0,
        "dynamic_quant_overhead_us": 4.0
    }


def make_test_inputs(seed=42):
    rng = np.random.RandomState(seed + 100)
    return [rng.randn(4, 64).astype(np.float32) for _ in range(3)]


def calculate_layer_size_ref(weights, bias, mode):
    w_count = weights.size
    b_count = bias.size
    if mode == "fp32":
        return w_count * 4 + b_count * 4
    elif mode == "fp16":
        return w_count * 2 + b_count * 2
    elif mode == "dynamic_int8":
        return w_count * 1 + b_count * 4 + 4
    elif mode == "full_int8":
        return w_count * 1 + b_count * 4 + 12
    raise ValueError(f"Unknown mode {mode}")


def quantize_fp16_ref(arr):
    return arr.astype(np.float16).astype(np.float32)


def quantize_int8_weights_ref(weights):
    max_val = float(np.max(np.abs(weights)))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    q = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
    dequant = q.astype(np.float32) * scale
    return q, scale, dequant


def quantize_int8_activations_ref(x, rmin, rmax):
    if rmax == rmin:
        scale = 1.0
        zero_point = 0
    else:
        scale = float(rmax - rmin) / 255.0
        zero_point = int(np.round(-rmin / scale))
        zero_point = max(0, min(255, zero_point))
    q = np.clip(np.round(x / scale) + zero_point, 0, 255).astype(np.uint8)
    dequant = (q.astype(np.float32) - zero_point) * scale
    return q, scale, zero_point, dequant


def evaluate_mode_output_ref(weights, bias, x, mode, calibration_range=None):
    if mode == "fp32":
        return x @ weights.T + bias
    elif mode == "fp16":
        w16 = quantize_fp16_ref(weights)
        b16 = quantize_fp16_ref(bias)
        x16 = quantize_fp16_ref(x)
        return x16 @ w16.T + b16
    elif mode == "dynamic_int8":
        _, _, w_dequant = quantize_int8_weights_ref(weights)
        return x @ w_dequant.T + bias
    elif mode == "full_int8":
        _, _, w_dequant = quantize_int8_weights_ref(weights)
        if calibration_range is None:
            rmin, rmax = float(np.min(x)), float(np.max(x))
        else:
            rmin, rmax = calibration_range
        _, _, _, x_dequant = quantize_int8_activations_ref(x, rmin, rmax)
        return x_dequant @ w_dequant.T + bias
    else:
        raise ValueError(f"Unknown mode: {mode}")


def calculate_model_size_ref(layers, mode):
    return sum(calculate_layer_size_ref(layer["weights"], layer["bias"], mode) for layer in layers)


def estimate_model_latency_ref(layers, mode, hw_config):
    bw_bps = hw_config["memory_bandwidth_gbps"] * 1e9
    if mode == "fp32":
        tflops = hw_config["fp32_tflops"]
    elif mode == "fp16":
        tflops = hw_config["fp16_tflops"]
    elif mode == "dynamic_int8":
        tflops = hw_config["fp32_tflops"]
    elif mode == "full_int8":
        tflops = hw_config["int8_tops"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ops_per_sec = tflops * 1e12
    dynamic_overhead = hw_config.get("dynamic_quant_overhead_us", 5.0) if mode == "dynamic_int8" else 0.0

    total_latency_us = 0.0
    for layer in layers:
        w = layer["weights"]
        flops = 2 * w.shape[0] * w.shape[1]
        w_bytes = calculate_layer_size_ref(w, layer["bias"], mode)
        t_mem_us = (w_bytes / bw_bps) * 1e6
        t_comp_us = (flops / ops_per_sec) * 1e6
        layer_lat = max(t_mem_us, t_comp_us) + dynamic_overhead
        total_latency_us += layer_lat

    return total_latency_us


def build_ptq_summary_table_ref(layers, hw_config, test_inputs):
    modes = ["fp32", "fp16", "dynamic_int8", "full_int8"]
    baseline_size = calculate_model_size_ref(layers, "fp32")

    table = {}
    for mode in modes:
        size_bytes = calculate_model_size_ref(layers, mode)
        size_ratio = float(size_bytes) / float(baseline_size)
        latency_us = estimate_model_latency_ref(layers, mode, hw_config)

        mse_list = []
        for x0 in test_inputs:
            x_fp = x0.copy()
            for layer in layers:
                x_fp = evaluate_mode_output_ref(layer["weights"], layer["bias"], x_fp, "fp32")

            x_m = x0.copy()
            for layer in layers:
                cal = layer.get("calibration_range")
                x_m = evaluate_mode_output_ref(layer["weights"], layer["bias"], x_m, mode, cal)

            mse = float(np.mean((x_m - x_fp) ** 2))
            mse_list.append(mse)

        table[mode] = {
            "size_bytes": size_bytes,
            "size_ratio": size_ratio,
            "mse": float(np.mean(mse_list)),
            "latency_us": latency_us,
        }

    return table


def rank_ptq_modes_ref(table, max_mse_threshold=0.01):
    valid = [mode for mode, metrics in table.items() if metrics["mse"] <= max_mse_threshold]
    valid.sort(key=lambda m: table[m]["latency_us"])
    return valid
