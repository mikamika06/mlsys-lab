import numpy as np

from quanteval.modes import evaluate_mode_output


def calculate_layer_size(weights, bias, mode):
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
    else:
        raise ValueError(f"Unknown mode: {mode}")


def calculate_model_size(layers, mode):
    return sum(calculate_layer_size(layer["weights"], layer["bias"], mode) for layer in layers)


def estimate_model_latency(layers, mode, hw_config):
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
        w_bytes = calculate_layer_size(w, layer["bias"], mode)
        t_mem_us = (w_bytes / bw_bps) * 1e6
        t_comp_us = (flops / ops_per_sec) * 1e6
        layer_lat = max(t_mem_us, t_comp_us) + dynamic_overhead
        total_latency_us += layer_lat

    return total_latency_us


def build_ptq_summary_table(layers, hw_config, test_inputs):
    modes = ["fp32", "fp16", "dynamic_int8", "full_int8"]
    baseline_size = calculate_model_size(layers, "fp32")

    table = {}
    for mode in modes:
        size_bytes = calculate_model_size(layers, mode)
        size_ratio = float(size_bytes) / float(baseline_size)
        latency_us = estimate_model_latency(layers, mode, hw_config)

        mse_list = []
        for x0 in test_inputs:
            x_fp = x0.copy()
            for layer in layers:
                x_fp = evaluate_mode_output(layer["weights"], layer["bias"], x_fp, "fp32")

            x_m = x0.copy()
            for layer in layers:
                cal = layer.get("calibration_range")
                x_m = evaluate_mode_output(layer["weights"], layer["bias"], x_m, mode, cal)

            mse = float(np.mean((x_m - x_fp) ** 2))
            mse_list.append(mse)

        table[mode] = {
            "size_bytes": size_bytes,
            "size_ratio": size_ratio,
            "mse": float(np.mean(mse_list)),
            "latency_us": latency_us,
        }

    return table


def rank_ptq_modes(table, max_mse_threshold=0.01):
    valid = [mode for mode, metrics in table.items() if metrics["mse"] <= max_mse_threshold]
    valid.sort(key=lambda m: table[m]["latency_us"])
    return valid
