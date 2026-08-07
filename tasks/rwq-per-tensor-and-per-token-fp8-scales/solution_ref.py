def fp8_scales(W: list[list[float]], X: list) -> tuple[float, list[float]]:
    """
    Compute per‑tensor and per‑token FP8 scales.

    Parameters
    ----------
    W : list[list[float]]
        Weight matrix of shape (out_dim, in_dim).
    X : list
        Activation tensor. Tokens are rows along all axes except the last one.

    Returns
    -------
    tuple[float, list[float]]
        Per‑tensor scale and per‑token scales.
    """
    max_w = -float('inf')
    for row in W:
        for val in row:
            abs_val = abs(val)
            if abs_val > max_w:
                max_w = abs_val
    tensor_scale = max_w / 448.0

    def get_tokens(tensor):
        if not isinstance(tensor, list):
            return []
        if len(tensor) == 0:
            return []
        if not isinstance(tensor[0], list):
            return [tensor]
        if not isinstance(tensor[0][0], list):
            return tensor
        tokens = []
        for sub in tensor:
            tokens.extend(get_tokens(sub))
        return tokens

    tokens = get_tokens(X)

    token_scales = []
    for token in tokens:
        max_x = -float('inf')
        for val in token:
            abs_val = abs(val)
            if abs_val > max_x:
                max_x = abs_val
        token_scales.append(max_x / 448.0)

    return tensor_scale, token_scales
