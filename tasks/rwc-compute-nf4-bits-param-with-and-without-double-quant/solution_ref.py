def compute_nf4_bits(
    weights,
    block_size,
    outer_block,
    inner_block
):
    """
    Compute NF4 bits per parameter with and without double quantization.

    Parameters
    ----------
    weights : np.ndarray
        The weight tensor to be quantized. Only its size is used.
    block_size : int
        Block size for the single‑level NF4 scheme.
    outer_block : int
        Outer block size for the double‑quant scheme.
    inner_block : int
        Inner block size for the double‑quant scheme.

    Returns
    -------
    tuple[float, float]
        (bits_no_double, bits_with_double)
    """
    n = weights.size

    # Single‑level NF4: 4 bits per value + 32‑bit scale per block.
    bits_no_double = 4 + 32 / block_size

    # Double‑quant: 4 bits per value, an 8‑bit inner scale and a 32‑bit outer scale.
    bits_with_double = 4 + 8 / outer_block + 32 / (outer_block * inner_block)

    return float(bits_no_double), float(bits_with_double)
