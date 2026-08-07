CONFIGS = [
    {"num_attention_heads": 32, "hidden_dim": 4096, "base_seq_latency": 120.0, "comm_overhead": 1.5, "base_throughput": 800.0},
    {"num_attention_heads": 16, "hidden_dim": 2048, "base_seq_latency": 80.0, "comm_overhead": 1.0, "base_throughput": 1200.0},
    {"num_attention_heads": 8, "hidden_dim": 1024, "base_seq_latency": 50.0, "comm_overhead": 0.5, "base_throughput": 2000.0}
]

TP_DEGREES_LIST = [[1, 2, 4, 8], [1, 2, 4, 8, 16], [1, 2, 4, 8]]

SAMPLE_LOGS = [
    ["INFO: TensorParallel layer q_proj shape=(4096, 4096)", "INFO: TensorParallel layer k_proj shape=(1024, 4096)"],
    ["INFO: TensorParallel layer q_proj shape=(2048, 2048)", "INFO: TensorParallel layer v_proj shape=(512, 2048)"],
    ["INFO: TensorParallel layer q_proj shape=(1024, 1024)"]
]

def sweep_tp_performance(config, tp_degrees):
    results = []
    num_heads = config["num_attention_heads"]
    hidden_dim = config["hidden_dim"]
    for tp in tp_degrees:
        if num_heads % tp != 0 or hidden_dim % tp != 0:
            continue
        base_latency = config["base_seq_latency"] * (1.0 / tp) + config["comm_overhead"] * (tp - 1)
        estimated_throughput = config["base_throughput"] * tp * (1.0 / (1.0 + 0.05 * (tp - 1)))
        results.append({"tp": tp, "latency": base_latency, "throughput": estimated_throughput})
    return results

def max_valid_tp_degree(num_kv_heads, num_attention_heads):
    valid = []
    for tp in range(1, num_attention_heads + 1):
        if num_attention_heads % tp == 0 and num_kv_heads % tp == 0:
            valid.append(tp)
    return max(valid) if valid else 1

def verify_server_log_partitions(log_lines, expected_tp):
    verified_layers = 0
    for line in log_lines:
        if "TensorParallel" in line or "partition" in line.lower():
            parts = line.split()
            for p in parts:
                if "shape=" in p:
                    dims = p.split("=")[1].strip("()[]").split(",")
                    if len(dims) > 0:
                        try:
                            dim_val = int(dims[0])
                            if dim_val % expected_tp == 0:
                                verified_layers += 1
                        except ValueError:
                            pass
    return verified_layers
