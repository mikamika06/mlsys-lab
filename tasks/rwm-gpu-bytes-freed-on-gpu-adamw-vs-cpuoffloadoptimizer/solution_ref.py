def gpu_bytes_freed(K: int, phi: int, offload_gradients: bool) -> int:
    """
    Compute the number of GPU bytes freed when moving from an on‑GPU optimizer
    (e.g. AdamW) to a CPU‑offloaded optimizer.

    Parameters
    ----------
    K : int
        Number of trainable parameters.
    phi : int
        Size in bytes of one parameter tensor.
    offload_gradients : bool
        If True, gradients are also moved to the CPU.

    Returns
    -------
    int
        Total number of bytes freed on the GPU.
    """
    return K * phi * (1 + 2 * int(offload_gradients))
