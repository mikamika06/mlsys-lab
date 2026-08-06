import numpy as np

CONFIGS = [
    {
        "hidden_size": 2048,
        "num_layers": 16,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 128,
        "vocab_size": 32000,
        "num_experts": 8,
        "active_experts": 2,
        "expert_hidden_size": 4096,
        "bytes_per_param": 2,
        "context_len": 4096,
        "batch_size": 4,
    },
    {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
        "num_experts": 16,
        "active_experts": 4,
        "expert_hidden_size": 8192,
        "bytes_per_param": 2,
        "context_len": 8192,
        "batch_size": 2,
    },
]

def compute_total_memory(cfg):
    h = cfg["hidden_size"]
    l = cfg["num_layers"]
    kv_heads = cfg["num_kv_heads"]
    hd = cfg["head_dim"]
    n_exp = cfg["num_experts"]
    a_exp = cfg["active_experts"]
    exp_h = cfg["expert_hidden_size"]
    bpp = cfg["bytes_per_param"]
    ctx = cfg["context_len"]
    bs = cfg["batch_size"]
    attn_weight_params = l * (4 * h * h)
    ffn_weight_params = l * n_exp * (3 * h * exp_h)
    embed_params = cfg["vocab_size"] * h
    total_params = attn_weight_params + ffn_weight_params + embed_params
    weight_memory = total_params * bpp
    kv_cache_memory = 2 * l * bs * ctx * kv_heads * hd * bpp
    activation_memory = bs * ctx * h * 4 * bpp
    return float(weight_memory + kv_cache_memory + activation_memory)

def compute_crossover_len(cfg):
    h = cfg["hidden_size"]
    l = cfg["num_layers"]
    kv_heads = cfg["num_kv_heads"]
    hd = cfg["head_dim"]
    n_exp = cfg["num_experts"]
    a_exp = cfg["active_experts"]
    exp_h = cfg["expert_hidden_size"]
    attn_flops_per_token = 4 * l * h
    ffn_flops_per_token = 2 * l * a_exp * (3 * h * exp_h)
    ratio = ffn_flops_per_token / (2 * l * kv_heads * hd)
    return float(ratio)

def compute_latency(cfg, contexts):
    latencies = []
    for ctx in contexts:
        prefill = 0.001 * ctx + 0.05
        decode = 0.0002 * ctx + 0.02
        latencies.append((float(prefill), float(decode)))
    return latencies
