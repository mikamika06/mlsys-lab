def count_parameters(config):
    hidden_dim = config["hidden_dim"]
    num_experts = config["num_experts"]
    top_k = config["top_k"]
    intermediate_dim = config["intermediate_dim"]
    num_layers = config.get("num_layers", 1)

    router_params = hidden_dim * num_experts
    expert_params_per_expert = (hidden_dim * intermediate_dim) + (intermediate_dim * hidden_dim)
    total_expert_params = num_experts * expert_params_per_expert
    total_layer_params = router_params + total_expert_params
    total_params = total_layer_params * num_layers

    active_expert_params_per_token = top_k * expert_params_per_expert
    active_layer_params = router_params + active_expert_params_per_token
    active_params = active_layer_params * num_layers

    return {
        "total_parameters": int(total_params),
        "active_parameters": int(active_params),
    }
