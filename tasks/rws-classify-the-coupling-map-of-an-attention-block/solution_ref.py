def classify_coupling_map(
    q_shape,
    k_shape,
    v_shape,
    o_proj_shape,
    head_index
):
    """
    Compute the slice indices for a given attention head.

    Parameters
    ----------
    q_shape, k_shape, v_shape : tuple[int, ...]
        Shapes of the query/key/value tensors.  The last dimension is H * d_k.
    o_proj_shape : tuple[int, ...]
        Shape of the output projection matrix W_o.  The first dimension is H * d_v.
    head_index : int
        Zero‑based index of the head to classify.

    Returns
    -------
    dict[str, tuple[int, int]]
        Mapping from tensor name to a half‑open slice (start, end).
    """
    # Infer number of heads and per‑head dimension from shapes
    Hq = q_shape[-1]
    Hv = o_proj_shape[0]

    import math
    H = math.gcd(Hq, Hv)
    if H == 0:
        raise ValueError("Invalid tensor shapes")

    d_k = Hq // H
    d_v = Hv // H

    start_qk = head_index * d_k
    end_qk = (head_index + 1) * d_k

    start_o = head_index * d_v
    end_o = (head_index + 1) * d_v

    return {
        "q": (start_qk, end_qk),
        "k": (start_qk, end_qk),
        "v": (start_qk, end_qk),
        "o_proj_input": (start_o, end_o)
    }
