def param_savings(vocab_size, d_model):
    """
    Return the number of parameters for a language model with and without weight tying.
    
    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (V).
    d_model : int
        Dimensionality of the hidden representation (d).
    
    Returns
    -------
    tuple[int, int]
        (tied_params, untied_params)
    """
    tied = vocab_size * d_model          # shared embedding / head matrix
    untied = 2 * vocab_size * d_model   # separate matrices
    return (tied, untied)
