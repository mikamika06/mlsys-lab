def compute_prefill_flops(config: dict) -> int:
    hidden_size = config["hidden_size"]
    num_hidden_layers = config["num_hidden_layers"]
    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    intermediate_size = config["intermediate_size"]
    head_dim = hidden_size // num_attention_heads

    q_proj = hidden_size * num_attention_heads * head_dim
    k_proj = hidden_size * num_key_value_heads * head_dim
    v_proj = hidden_size * num_key_value_heads * head_dim
    o_proj = num_attention_heads * head_dim * hidden_size
    attn_flops = q_proj + k_proj + v_proj + o_proj

    attn_score_flops = 2 * num_attention_heads * head_dim

    mlp_gate = hidden_size * intermediate_size
    mlp_up = hidden_size * intermediate_size
    mlp_down = intermediate_size * hidden_size
    mlp_flops = mlp_gate + mlp_up + mlp_down

    layer_flops = 2 * (attn_flops + attn_score_flops + mlp_flops)
    return num_hidden_layers * layer_flops
