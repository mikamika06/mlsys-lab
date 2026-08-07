def get_sample_config():
    return {
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "intermediate_size": 11008,
        "num_layers": 32
    }

def reference_layer_flops(config, seq_len):
    hidden_size = config["hidden_size"]
    num_heads = config["num_heads"]
    num_kv_heads = config.get("num_kv_heads", num_heads)
    intermediate_size = config["intermediate_size"]
    head_dim = hidden_size // num_heads

    q_flops = 2 * seq_len * hidden_size * (num_heads * head_dim)
    k_flops = 2 * seq_len * hidden_size * (num_kv_heads * head_dim)
    v_flops = 2 * seq_len * hidden_size * (num_kv_heads * head_dim)
    o_flops = 2 * seq_len * (num_heads * head_dim) * hidden_size
    attn_proj_flops = q_flops + k_flops + v_flops + o_flops
    attn_score_flops = 2 * num_heads * seq_len * seq_len * head_dim
    mlp_flops = 6 * seq_len * hidden_size * intermediate_size
    return attn_proj_flops + attn_score_flops + mlp_flops

def reference_total_flops(config, prefill_len, decode_steps):
    num_layers = config["num_layers"]
    layer_prefill = reference_layer_flops(config, prefill_len)
    total_prefill = num_layers * layer_prefill
    decode_total = 0.0
    for step in range(decode_steps):
        current_seq = prefill_len + step
        decode_total += num_layers * reference_layer_flops(config, 1)
        decode_total += num_layers * 2 * config["num_heads"] * current_seq * (config["hidden_size"] // config["num_heads"])
    return total_prefill + decode_total
