def compute_tp_scaling_table(model_config, tp_degrees, sequence_length, batch_size, precision_bytes=2):
    num_layers = model_config["num_layers"]
    h = model_config["hidden_size"]
    i = model_config["intermediate_size"]

    results = []
    for tp in tp_degrees:
        weight_params_per_rank = num_layers * (4 * h * h + 2 * h * i) / tp
        weight_mem = weight_params_per_rank * precision_bytes

        act_elements_per_rank = num_layers * batch_size * sequence_length * (2 * h + (6 * h + i) / tp)
        act_mem = act_elements_per_rank * precision_bytes

        if tp > 1:
            comm_bytes_per_layer = 4 * ((tp - 1) / tp) * batch_size * sequence_length * h * precision_bytes
        else:
            comm_bytes_per_layer = 0.0
        comm_bytes_total = num_layers * comm_bytes_per_layer

        results.append({
            "tp_degree": tp,
            "weight_memory_bytes": float(weight_mem),
            "activation_memory_bytes": float(act_mem),
            "comm_bytes_per_step": float(comm_bytes_total),
        })
    return results


def find_optimal_tp_degree(model_config, available_tp_degrees, sequence_length, batch_size, precision_bytes, gpu_memory_bytes, interconnect_bandwidth_bytes_per_sec, compute_flops_per_sec):
    scaling_table = compute_tp_scaling_table(model_config, available_tp_degrees, sequence_length, batch_size, precision_bytes)
    num_layers = model_config["num_layers"]
    h = model_config["hidden_size"]
    i = model_config["intermediate_size"]

    fwd_flops = num_layers * 2.0 * (4 * h * h + 2 * h * i) * batch_size * sequence_length

    best_tp = None
    min_latency = float("inf")
    valid_degrees = []

    for row in scaling_table:
        tp = row["tp_degree"]
        tot_mem = row["weight_memory_bytes"] + row["activation_memory_bytes"]
        if tot_mem <= gpu_memory_bytes:
            valid_degrees.append(tp)
            compute_time = fwd_flops / (tp * compute_flops_per_sec)
            comm_time = row["comm_bytes_per_step"] / interconnect_bandwidth_bytes_per_sec if interconnect_bandwidth_bytes_per_sec > 0 else 0.0
            tot_latency = compute_time + comm_time
            if tot_latency < min_latency:
                min_latency = tot_latency
                best_tp = tp

    return {
        "optimal_tp_degree": best_tp,
        "min_latency_sec": float(min_latency) if best_tp is not None else float("inf"),
        "valid_degrees": valid_degrees,
    }
