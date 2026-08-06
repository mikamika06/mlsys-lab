import ref


def check(workdir):
    from mlx_lora_util.verify import verify_safetensors_shapes

    cfg = ref.CONFIGS[0]
    rank = 4
    targets = ["q_proj"]
    hidden_dim = cfg.get("hidden_size", 512)
    num_layers = cfg.get("num_hidden_layers", 2)
    num_heads = cfg.get("num_attention_heads", 8)
    head_dim = hidden_dim // num_heads
    dim_out = num_heads * head_dim

    tensors = {}
    for l in range(num_layers):
        tensors[f"model.layers.{l}.self_attn.q_proj.lora_a.weight"] = (rank, hidden_dim)
        tensors[f"model.layers.{l}.self_attn.q_proj.lora_b.weight"] = (dim_out, rank)

    res = verify_safetensors_shapes(tensors, cfg, rank, targets)
    out = {"shapes_matched": 1.0 if res else 0.0}
    return out
