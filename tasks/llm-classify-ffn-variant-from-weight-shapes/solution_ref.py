def classify_ffn_variant(weight_shapes):
    """
    Classify an FFN variant based solely on the shapes of its linear layers.

    Parameters
    ----------
    weight_shapes : list[tuple[int, int]]
        Each tuple is (out_dim, in_dim) for a linear layer, in order.

    Returns
    -------
    str
        One of "vanilla", "swi_glu" or "geglu".

    Raises
    ------
    ValueError
        If the shapes do not match any known variant.
    """
    n = len(weight_shapes)
    if n == 3:
        h0, _ = weight_shapes[0]
        h1, _ = weight_shapes[1]
        out2, in2 = weight_shapes[2]
        if h0 != h1:
            raise ValueError("GeGLU gate and act dims differ")
        if in2 != h0:
            raise ValueError("GeGLU hidden size mismatch with output layer")
        return "geglu"
    elif n == 2:
        h0, _ = weight_shapes[0]
        out1, in1 = weight_shapes[1]
        if h0 == in1:
            return "vanilla"
        elif h0 == 2 * in1:
            return "swi_glu"
        else:
            raise ValueError("Unknown variant")
    else:
        raise ValueError("Unsupported number of layers")
