"""MoE parameter counting utilities."""


def count_parameters(config):
    """Calculate total and active parameter counts for an MoE model config.

    Args:
        config (dict): Configuration dictionary containing:
            - hidden_size (int)
            - num_layers (int)
            - moe_layer_frequency (int): Every Nth layer is an MoE layer
            - num_experts (int)
            - num_experts_per_tok (int): Top-k active experts per token
            - ffx_hidden_size (int): Hidden dimension of non-MoE FFN
            - expert_hidden_size (int): Hidden dimension of expert FFN
            - non_ffn_layer_params (int): Base parameters per layer excluding FFN/MoE

    Returns:
        dict: {
            "total_params": int,
            "active_params": int
        }
    """
    raise NotImplementedError
