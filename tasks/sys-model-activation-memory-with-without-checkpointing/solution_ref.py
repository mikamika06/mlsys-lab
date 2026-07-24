def activation_memory_ratio(layer_sizes, checkpoint_every):
    """
    Return the ratio of full peak activation memory to checkpointed peak memory.
    
    Parameters
    ----------
    layer_sizes : list[int]
        Number of neurons in each layer (including input and output).
    checkpoint_every : int
        Store an activation every `checkpoint_every` layers.
    
    Returns
    -------
    float
        Ratio M_full / M_chkpt.
    """
    full = sum(sz * 8 for sz in layer_sizes)
    chkpt = sum(
        sz * 8
        for i, sz in enumerate(layer_sizes)
        if i % checkpoint_every == 0
    )
    return full / chkpt
