def _pool1d(scores: list[float], kernel_size: int) -> list[float]:
    """1D average pool, stride 1, zero-padded to the same output length
    (count_include_pad=True, matching torch.nn.functional.avg_pool1d)."""
    pad = kernel_size // 2
    L_prefix = len(scores)
    padded = [0.0] * (L_prefix + 2 * pad)
    for i in range(L_prefix):
        padded[pad + i] = scores[i]
    out = [0.0] * L_prefix
    for i in range(L_prefix):
        acc = 0.0
        for j in range(kernel_size):
            acc += padded[i + j]
        out[i] = acc / kernel_size
    return out


def snapkv_select(
    attn: list[list[list[float]]],
    window_size: int,
    kernel_size: int,
    capacity: int,
) -> tuple[list[int], int, float]:
    """SnapKV-style KV-cache eviction vote from the observation window.

    Parameters
    ----------
    attn : list[list[list[float]]]
        Attention weights of shape (H, window_size, L_prefix) from the last
        `window_size` query positions (the "observation window") to the
        `L_prefix` prefill key positions that precede the window.
    window_size : int
    kernel_size : int
        Odd 1D average-pool kernel used to smooth the per-position vote.
    capacity : int
        Number of prefill positions to keep (clipped to L_prefix if larger).

    Returns
    -------
    selected_indices : list[int]
        Indices into the L_prefix axis of the kept prefill positions, ascending sorted.
    kept_total : int
        len(selected_indices) + window_size.
    compression_ratio : float
        kept_total / (L_prefix + window_size).
    """
    H = len(attn)
    W = len(attn[0])
    L_prefix = len(attn[0][0])

    scores = [0.0] * L_prefix
    for h in range(H):
        for w in range(W):
            for l in range(L_prefix):
                scores[l] += attn[h][w][l]

    pooled = _pool1d(scores, kernel_size)

    k = min(int(capacity), L_prefix)
    order = sorted(range(L_prefix), key=lambda i: (-pooled[i], i))
    selected = sorted(order[:k])

    kept_total = k + window_size
    compression_ratio = kept_total / (L_prefix + window_size)

    return selected, kept_total, compression_ratio
