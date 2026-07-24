def per_gpu_bytes(phi: int, n_gpus: int, stage: int) -> int:
    """
    Return the number of bytes a single GPU must allocate for parameters,
    gradients and Adam optimizer state according to ZeRO memory accounting.
    
    Parameters
    ----------
    phi : int
        Number of model parameters (each fp16).
    n_gpus : int
        Total number of GPUs participating in training.
    stage : int
        ZeRO stage: 0, 1 or 2.
        
    Returns
    -------
    int
        Bytes per GPU.
    
    Raises
    ------
    ValueError
        If `stage` is not 0, 1 or 2.
    """
    if stage == 0:
        return 16 * phi
    elif stage == 1:
        return 4 * phi + (12 * phi) // n_gpus
    elif stage == 2:
        return (4 * phi) // n_gpus + 12 * phi
    else:
        raise ValueError("stage must be 0, 1 or 2")
