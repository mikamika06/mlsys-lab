import math


def param_count(config):
    hidden = config["hidden_size"]
    kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    layers = config["n_layers"]

    attn = 2 * hidden * hidden + 2 * hidden * kv_heads * head_dim
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    embedding = vocab * hidden
    head = 0 if config.get("tie_embeddings", False) else vocab * hidden

    return embedding + layers * (attn + mlp + norms) + hidden + head


def kv_cache_bytes(config, context_length, kv_bytes_per_element=2):
    return (config["n_layers"] * 2 * config["n_kv_heads"] * config["head_dim"]
            * context_length * kv_bytes_per_element)


def estimate_bytes(config, context_length, bytes_per_param):
    weight_bytes = math.ceil(param_count(config) * bytes_per_param)
    return int(weight_bytes) + kv_cache_bytes(config, context_length)
