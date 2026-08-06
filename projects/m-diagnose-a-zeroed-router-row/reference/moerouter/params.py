"""MoE parameter counting utilities."""


def count_parameters(config):
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    moe_freq = config.get("moe_layer_frequency", 1)
    num_experts = config["num_experts"]
    top_k = config["num_experts_per_tok"]
    ffn_hidden = config["ffn_hidden_size"]
    expert_hidden = config["expert_hidden_size"]
    non_ffn_params = config.get("non_ffn_layer_params", 0)

    dense_ffn_params = 2 * hidden_size * ffn_hidden
    expert_params = 2 * hidden_size * expert_hidden

    total_params = 0
    active_params = 0

    for l in range(num_layers):
        is_moe = (l % moe_freq) == (moe_freq - 1)
        layer_non_ffn = non_ffn_params
        if is_moe:
            layer_total = layer_non_ffn + (num_experts * expert_params)
            layer_active = layer_non_ffn + (top_k * expert_params)
        else:
            layer_total = layer_non_ffn + dense_ffn_params
            layer_active = layer_non_ffn + dense_ffn_params

        total_params += layer_total
        active_params += layer_active

    return {
        "total_params": int(total_params),
        "active_params": int(active_params),
    }
