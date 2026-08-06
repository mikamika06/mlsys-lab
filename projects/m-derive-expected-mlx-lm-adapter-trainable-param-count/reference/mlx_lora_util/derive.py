def expected_adapter_parameters(config, rank, target_modules):
    hidden_dim = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    num_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = hidden_dim // num_heads
    total = 0
    for mod in target_modules:
        if mod in ["q_proj", "query"]:
            dim_in = hidden_dim
            dim_out = num_heads * head_dim
            total += num_layers * (dim_in * rank + rank * dim_out)
        elif mod in ["k_proj", "key"]:
            dim_in = hidden_dim
            dim_out = num_kv_heads * head_dim
            total += num_layers * (dim_in * rank + rank * dim_out)
        elif mod in ["v_proj", "value"]:
            dim_in = hidden_dim
            dim_out = num_kv_heads * head_dim
            total += num_layers * (dim_in * rank + rank * dim_out)
        elif mod in ["o_proj", "output"]:
            dim_in = num_heads * head_dim
            dim_out = hidden_dim
            total += num_layers * (dim_in * rank + rank * dim_out)
        elif mod in ["gate_proj", "up_proj"]:
            intermediate_dim = config.get("intermediate_size", 11008)
            dim_in = hidden_dim
            dim_out = intermediate_dim
            total += num_layers * (dim_in * rank + rank * dim_out)
        elif mod in ["down_proj"]:
            intermediate_dim = config.get("intermediate_size", 11008)
            dim_in = intermediate_dim
            dim_out = hidden_dim
            total += num_layers * (dim_in * rank + rank * dim_out)
    return total
