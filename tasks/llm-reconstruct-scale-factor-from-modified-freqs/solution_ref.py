def recover_scale_factor(orig_freqs: list[float],
                         mod_freqs: list[float]) -> float:
    """
    Recover the scaling factor s such that mod_freqs ≈ orig_freqs * s.
    The factor is computed as the mean of element‑wise ratios, which is
    accurate when all elements are scaled by a common constant up to small noise.
    """
    n = len(orig_freqs)
    total = 0.0
    for i in range(n):
        total += mod_freqs[i] / orig_freqs[i]
    return float(total / n)
