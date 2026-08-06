def max_cp_degree(config):
    return min(config["num_key_value_heads"], config["num_attention_heads"])


def hybrid_usp_score(config, ring_bw, ulysses_bw):
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    return float((ulysees_bw * kv_heads + ring_bw * heads) / (ring_bw + ulysses_bw + 1e-5))


def predict_throughput(config, cp_degree, base_tput):
    return base_tput * (1.0 / (1.0 + 0.05 * cp_degree))
