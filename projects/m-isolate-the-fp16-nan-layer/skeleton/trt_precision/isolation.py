def locate_fp16_overflow_layers(graph):
    """
    Identifies names of layers that produce activations outside FP16 range or contain NaNs/Infs.
    """
    raise NotImplementedError
