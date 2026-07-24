def compute_peak_activation_memory(L: int, S: int, H: int, k: int) -> tuple[int, int]:
    """
    Compute the memory footprint for full activation storage and for a checkpointing strategy.

    Parameters
    ----------
    L : int
        Number of layers in the network.
    S : int
        Batch size (number of samples processed simultaneously).
    H : int
        Size of each activation tensor per sample (e.g. hidden dimension).
    k : int
        Checkpoint interval – store activations every k layers.

    Returns
    -------
    tuple[int, int]
        (full_mem, ckpt_mem) where:
            full_mem  = L * S * H
            ckpt_mem  = (ceil(L / k) + 1) * S * H + k * S * H
    """
    # Full store memory
    full_mem = L * S * H

    # Number of checkpoints: ceil division plus the final boundary
    checkpoints = (L + k - 1) // k + 1

    # Peak memory during checkpointed execution
    ckpt_mem = checkpoints * S * H + k * S * H

    return full_mem, ckpt_mem
