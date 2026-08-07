CONFIGS = [
    {"hidden_size": 2048, "num_layers": 22, "num_kv_heads": 4, "head_dim": 128, "vocab_size": 32000},
    {"hidden_size": 4096, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "vocab_size": 32000},
    {"hidden_size": 8192, "num_layers": 80, "num_kv_heads": 8, "head_dim": 128, "vocab_size": 32000},
]


def model_bytes(config, bytes_per_param=2):
    h = config["hidden_size"]
    l = config["num_layers"]
    v = config["vocab_size"]
    attn_weights = l * (4 * h * h)
    mlp_weights = l * (3 * h * (4 * h))
    embed_weights = 2 * v * h
    total_params = attn_weights + mlp_weights + embed_weights
    return total_params * bytes_per_param


def kv_bytes(config, seq_len, bytes_per_elem=2):
    l = config["num_layers"]
    kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    return 2 * l * seq_len * kv_heads * head_dim * bytes_per_elem


def sequential_cost(config, seq_len, gamma, bandwidth_gbs):
    m_bytes = model_bytes(config)
    k_bytes = kv_bytes(config, seq_len)
    total_traffic = (m_bytes + k_bytes) * (gamma + 1)
    return total_traffic / (bandwidth_gbs * 1e9)


def parallel_cost(config, seq_len, gamma, bandwidth_gbs):
    m_bytes = model_bytes(config)
    k_bytes = kv_bytes(config, seq_len + gamma)
    total_traffic = m_bytes + k_bytes
    return total_traffic / (bandwidth_gbs * 1e9)


def execution_cost_ratio(config, seq_len, gamma, bandwidth_gbs):
    s_cost = sequential_cost(config, seq_len, gamma, bandwidth_gbs)
    p_cost = parallel_cost(config, seq_len, gamma, bandwidth_gbs)
    return s_cost / p_cost
