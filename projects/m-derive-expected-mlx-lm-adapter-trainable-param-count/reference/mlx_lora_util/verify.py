def verify_safetensors_shapes(tensors_dict, config, rank, target_modules):
    hidden_dim = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    num_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = hidden_dim // num_heads
    for l in range(num_layers):
        for mod in target_modules:
            if mod in ["q_proj", "query"]:
                dim_in = hidden_dim
                dim_out = num_heads * head_dim
            elif mod in ["k_proj", "v_proj", "key", "value"]:
                dim_in = hidden_dim
                dim_out = num_kv_heads * head_dim
            elif mod in ["o_proj", "output"]:
                dim_in = num_heads * head_dim
                dim_out = hidden_dim
            elif mod in ["gate_proj", "up_proj"]:
                intermediate_dim = config.get("intermediate_size", 11008)
                dim_in = hidden_dim
                dim_out = intermediate_dim
            elif mod in ["down_proj"]:
                intermediate_dim = config.get("intermediate_size", 11008)
                dim_in = intermediate_dim
                dim_out = hidden_dim
            else:
                continue
            k_a = f"model.layers.{l}.self_attn.{mod}.lora_a.weight" if "proj" in mod and mod != "down_proj" and mod != "up_proj" and mod != "gate_proj" else f"model.layers.{l}.mlp.{mod}.lora_a.weight"
            if mod in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                k_a = f"model.layers.{l}.self_attn.{mod}.lora_a.weight"
                k_b = f"model.layers.{l}.self_attn.{mod}.lora_b.weight"
            else:
                k_a = f"model.layers.{l}.mlp.{mod}.lora_a.weight"
                k_b = f"model.layers.{l}.mlp.{mod}.lora_b.weight"
            if k_a not in tensors_dict or k_b not in tensors_dict:
                return False
            if tuple(tensors_dict[k_a]) != (rank, dim_in):
                return False
            if tuple(tensors_dict[k_b]) != (dim_out, rank):
                return False
    return True
