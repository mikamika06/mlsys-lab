def param_counts(input_dim, output_dim, hidden_size):
    """
    Return the number of learnable parameters for a vanilla and a gated FFN.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    output_dim : int
        Dimensionality of the output vector.
    hidden_size : int
        Size of the hidden representation (number of neurons).

    Returns
    -------
    vanilla_params : int
        Number of parameters in a vanilla two‑layer FFN.
    gated_params : int
        Number of parameters in a gated (SwiGLU/GeGLU) FFN.
    """
    # Vanilla: one projection from input to hidden, then hidden to output
    vanilla = input_dim * hidden_size + hidden_size * output_dim

    # Gated: two parallel projections into the same hidden space,
    # followed by a single projection to the output
    gated = 2 * input_dim * hidden_size + hidden_size * output_dim

    return vanilla, gated
