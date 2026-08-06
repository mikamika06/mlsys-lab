"""MoE top-k router with zeroed row diagnosis and recovery."""


def route_tokens(logits, top_k, mask=None):
    """Compute top-k expert indices and normalized gating weights.

    Args:
        logits (np.ndarray): Shape (num_tokens, num_experts)
        top_k (int): Number of experts to select per token
        mask (np.ndarray, optional): Boolean array of shape (num_tokens, num_experts).
            True indicates an expert is available/valid; False indicates masked out.

    Returns:
        dict: {
            "indices": np.ndarray of shape (num_tokens, top_k) - int,
            "weights": np.ndarray of shape (num_tokens, top_k) - float,
            "zero_row_diagnosed": list of int indices of tokens that had zeroed rows before recovery
        }
    """
    raise NotImplementedError
