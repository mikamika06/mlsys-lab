import numpy as np


def predict_tok_s(model_config, offload_fraction, hardware_config):
    """Predict generation throughput given offload fraction and hardware specs."""
    total_layers = model_config["num_layers"]
    bytes_per_layer = model_config["layer_weight_bytes"]
    offloaded_layers = total_layers * offload_fraction
    cpu_layers = total_layers - offloaded_layers

    gpu_bw = hardware_config["gpu_bandwidth_bytes_s"]
    cpu_bw = hardware_config["cpu_bandwidth_bytes_s"]
    pcie_bw = hardware_config["pcie_bandwidth_bytes_s"]

    gpu_time = (offloaded_layers * bytes_per_layer) / gpu_bw
    cpu_time = (cpu_layers * bytes_per_layer) / cpu_bw
    transfer_time = (cpu_layers * model_config["activation_bytes_per_layer"]) / pcie_bw

    total_time_per_token = max(gpu_time, cpu_time + transfer_time)
    if total_time_per_token <= 0:
        return 0.0
    return 1.0 / total_time_per_token


def locate_performance_cliff(model_config, hardware_config, step=0.05):
    """Find the offload fraction with the steepest drop/gain in throughput."""
    fractions = np.arange(0.0, 1.0 + step / 2.0, step)
    tok_s_list = [predict_tok_s(model_config, f, hardware_config) for f in fractions]

    max_drop = -1.0
    cliff_fraction = 0.0
    for i in range(len(fractions) - 1):
        drop = abs(tok_s_list[i + 1] - tok_s_list[i])
        if drop > max_drop:
            max_drop = drop
            cliff_fraction = float(fractions[i + 1])
    return cliff_fraction
