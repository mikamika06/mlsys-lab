def simulate_tp_sweep(
    config: dict, tp_list: list[int], workload: dict, hardware: dict
) -> list[dict]:
    layers = config["num_layers"]
    hidden = config["hidden_size"]
    attn_heads = config["num_attention_heads"]
    kv_heads = config["num_kv_heads"]
    inter = config["intermediate_size"]
    head_dim = config.get("head_dim", hidden // attn_heads)

    batch_size = workload["batch_size"]
    prompt_len = workload["prompt_len"]
    gen_len = workload["gen_len"]

    flops_per_sec = hardware["tflops_per_gpu"] * 1e12
    bw_bytes_sec = hardware["interconnect_gbps"] * 1e9
    comm_penalty_s = hardware["comm_overhead_us"] * 1e-6

    kv_dim = kv_heads * head_dim
    flops_per_layer_per_token = 4 * hidden * hidden + 4 * hidden * kv_dim + 6 * hidden * inter
    total_flops_per_token = layers * flops_per_layer_per_token

    results = []
    for tp in tp_list:
        if tp <= 0:
            continue

        prefill_comp = (batch_size * prompt_len * total_flops_per_token / tp) / flops_per_sec
        if tp > 1:
            prefill_comm_bytes = 2 * layers * 2.0 * ((tp - 1) / tp) * (2 * batch_size * prompt_len * hidden)
            prefill_comm = (prefill_comm_bytes / bw_bytes_sec) + (2 * layers * comm_penalty_s)
        else:
            prefill_comm = 0.0
        prefill_latency = prefill_comp + prefill_comm

        decode_step_comp = (batch_size * total_flops_per_token / tp) / flops_per_sec
        if tp > 1:
            decode_step_comm_bytes = 2 * layers * 2.0 * ((tp - 1) / tp) * (2 * batch_size * hidden)
            decode_step_comm = (decode_step_comm_bytes / bw_bytes_sec) + (2 * layers * comm_penalty_s)
        else:
            decode_step_comm = 0.0
        decode_step_latency = decode_step_comp + decode_step_comm
        decode_latency = gen_len * decode_step_latency

        total_latency = prefill_latency + decode_latency

        total_tokens = batch_size * (prompt_len + gen_len)
        sys_throughput = total_tokens / total_latency if total_latency > 0 else 0.0
        per_gpu_throughput = sys_throughput / tp

        results.append({
            "tp": tp,
            "prefill_latency_s": float(prefill_latency),
            "decode_latency_s": float(decode_latency),
            "total_latency_s": float(total_latency),
            "system_throughput_tps": float(sys_throughput),
            "per_gpu_throughput_tps": float(per_gpu_throughput),
        })

    return results


def find_optimal_tp(sweep_results: list[dict], metric: str = "throughput") -> int:
    if not sweep_results:
        return 1

    if metric == "throughput":
        best = max(sweep_results, key=lambda x: x["system_throughput_tps"])
    elif metric == "per_gpu_efficiency":
        best = max(sweep_results, key=lambda x: x["per_gpu_throughput_tps"])
    elif metric == "latency":
        best = min(sweep_results, key=lambda x: x["total_latency_s"])
    else:
        best = max(sweep_results, key=lambda x: x["system_throughput_tps"])

    return best["tp"]
