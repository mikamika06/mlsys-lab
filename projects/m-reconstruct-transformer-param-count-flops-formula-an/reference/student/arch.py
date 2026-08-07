import numpy as np


def compute_params(config):
    layers = config.get("num_hidden_layers", 12)
    hidden = config.get("hidden_size", 768)
    vocab = config.get("vocab_size", 32000)
    intermediate = config.get("intermediate_size", 4 * hidden)
    num_heads = config.get("num_attention_heads", 12)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = hidden // num_heads

    embed = vocab * hidden
    q_proj = hidden * (num_heads * head_dim)
    k_proj = hidden * (num_kv_heads * head_dim)
    v_proj = hidden * (num_kv_heads * head_dim)
    o_proj = (num_heads * head_dim) * hidden
    attn_total = q_proj + k_proj + v_proj + o_proj

    gate_proj = hidden * intermediate
    up_proj = hidden * intermediate
    down_proj = intermediate * hidden
    mlp_total = gate_proj + up_proj + down_proj

    input_layernorm = hidden
    post_attention_layernorm = hidden
    layer_total = attn_total + mlp_total + input_layernorm + post_attention_layernorm

    total = embed + layers * layer_total + hidden + vocab * hidden
    return int(total)


def compute_flops(config, seq_len):
    layers = config.get("num_hidden_layers", 12)
    hidden = config.get("hidden_size", 768)
    vocab = config.get("vocab_size", 32000)
    intermediate = config.get("intermediate_size", 4 * hidden)
    num_heads = config.get("num_attention_heads", 12)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = hidden // num_heads

    q_params = hidden * (num_heads * head_dim)
    k_params = hidden * (num_kv_heads * head_dim)
    v_params = hidden * (num_kv_heads * head_dim)
    o_params = (num_heads * head_dim) * hidden

    attn_flops = seq_len * (
        q_params + k_params + v_params + o_params + seq_len * hidden
    )
    mlp_flops = seq_len * (
        hidden * intermediate
        + hidden * intermediate
        + intermediate * hidden
    )
    layer_flops = attn_flops + mlp_flops
    total_flops = layers * layer_flops + seq_len * vocab
    return int(total_flops)


def compression_ratio(teacher_config, student_config):
    tp = compute_params(teacher_config)
    sp = compute_params(student_config)
    return float(tp / sp)
