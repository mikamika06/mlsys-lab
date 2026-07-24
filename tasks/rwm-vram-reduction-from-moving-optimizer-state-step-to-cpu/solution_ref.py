def vram_reduction_from_offload(
    n_params: int,
    param_bytes: int,
    grad_bytes: int,
    master_bytes: int,
    m_bytes: int,
    v_bytes: int,
    activation_bytes: int,
) -> float:
    """
    Fractional GPU memory reduction from offloading the Adam optimizer
    state (master weights + m + v) to CPU, given fixed activation memory
    that stays on the GPU regardless. See task.md for the closed form.
    """
    n = int(n_params)
    total_before = n * (param_bytes + grad_bytes + master_bytes + m_bytes + v_bytes) + activation_bytes
    offloaded = n * (master_bytes + m_bytes + v_bytes)
    return float(offloaded) / float(total_before)
