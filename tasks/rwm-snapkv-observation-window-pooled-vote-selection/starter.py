def snapkv_select(attn: list[list[list[float]]], window_size: int, kernel_size: int, capacity: int) -> tuple[list[int], int, float]:
    """SnapKV-style KV-cache eviction vote from the observation window.

    Parameters
    ----------
    attn : list[float], shape (H, window_size, L_prefix)
        Attention weights from the last `window_size` query positions (the
        "observation window") to the `L_prefix` prefill key positions that
        precede the window.
    window_size : int
    kernel_size : int
        Odd 1D average-pool kernel used to smooth the per-position vote.
    capacity : int
        Number of prefill positions to keep (clipped to L_prefix if larger).

    Returns
    -------
    selected_indices : list[float], int, ascending
        Indices into the L_prefix axis of the kept prefill positions.
    kept_total : int
        len(selected_indices) + window_size.
    compression_ratio : float
        kept_total / (L_prefix + window_size).
    """
    raise NotImplementedError('your code here')
