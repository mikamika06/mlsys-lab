"""Prefill FLOPs computation."""


def compute_prefill_flops_per_token(config: dict, seq_len: int) -> float:
    """Compute prefill floating point operations per token."""
    h = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    d_head = h // n_heads
    i = config.get("intermediate_size", 4 * h)

    q_flops = 2 * h * (n_heads * d_head)
    k_flops = 2 * h * (n_kv_heads * d_head)
    v_flops = 2 * h * (n_kv_heads * d_head)
    out_flops = 2 * (n_heads * d_head) * h

    attn_score_flops = 2 * n_heads * d_head * seq_len
    attn_val_flops = 2 * n_heads * seq_len * d_head

    attn_flops = q_flops + k_flops + v_flops + out_flops + attn_score_flops + attn_val_flops

    gate_flops = 2 * h * i
    up_flops = 2 * h * i
    down_flops = 2 * i * h
    mlp_flops = gate_flops + up_flops + down_flops

    layer_flops = attn_flops + mlp_flops
    total_flops_per_token = layer_flops * n_layers

    return float(total_flops_per_token)
