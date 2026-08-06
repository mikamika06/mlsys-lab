CONFIGS = [
    {"num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128},
    {"num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128},
    {"num_attention_heads": 16, "num_key_value_heads": 16, "head_dim": 64},
]


def max_cp_degree(config):
    return min(config["num_key_value_heads"], config["num_attention_heads"])


def hybrid_usp_bandwidth(config, nvlink_bw, infiniband_bw):
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    return float((nvlink_bw * kv_heads + infiniband_bw * heads) / (nvlink_bw + infiniband_bw + 1e-5))


def evaluate_throughput(config, cp_degree, base_tput):
    if cp_degree > max_cp_degree(config) or cp_degree <= 0:
        raise ValueError("invalid cp degree")
    return float(base_tput / (1.0 + 0.05 * cp_degree))
