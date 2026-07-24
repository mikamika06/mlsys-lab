import numpy as np

FIRST_LEVEL_BLOCK_SIZE = 64  # NF4 weight-quantization block size these absmax values came from


def double_quantize_absmax(absmax, block_size=256):
    """QLoRA-style double quantization of a first-level absmax array.

    `absmax` holds one fp32 scale per NF4 weight-quantization block (block
    size 64). Storing every one of those scales as fp32 costs 32 bits per
    64 weight params -- 0.5 bits/param of pure overhead. Double
    quantization quantizes the absmax array itself to int8, in blocks of
    `block_size` (256), after subtracting the array's global mean:

        c1 = absmax - mean(absmax)
        for each block of `block_size` consecutive c1 values:
            scale = max(|block|) / 127
            codes = round(block / scale), clipped to [-127, 127], int8

    Parameters
    ----------
    absmax : np.ndarray, shape (N,)
        First-level per-block absmax values (all >= 0).
    block_size : int
        Second-level (double-quantization) block size.

    Returns
    -------
    codes : np.ndarray, int8, shape (N,)
    scales : np.ndarray, float64, shape (ceil(N/block_size),)
    mean : float
        Global mean subtracted before quantizing.
    recon : np.ndarray, float64, shape (N,)
        Reconstructed absmax: `codes * scale[block] + mean`.
    bits_saved_per_param : float
        Bits per original weight parameter saved by double quantization,
        i.e. the fp32-per-absmax overhead (32 / FIRST_LEVEL_BLOCK_SIZE)
        minus the double-quantized overhead
        ((8*N + 32*n_second_level_blocks + 32) / (N * FIRST_LEVEL_BLOCK_SIZE)).
    """
    absmax = np.asarray(absmax, dtype=np.float64)
    n = absmax.shape[0]
    mean = float(np.mean(absmax))
    centered = absmax - mean

    n_blocks2 = -(-n // block_size)  # ceil division
    codes = np.zeros(n, dtype=np.int8)
    scales = np.zeros(n_blocks2, dtype=np.float64)
    recon = np.zeros(n, dtype=np.float64)

    for b in range(n_blocks2):
        lo = b * block_size
        hi = min(lo + block_size, n)
        seg = centered[lo:hi]
        amax = float(np.max(np.abs(seg)))
        scale = amax / 127.0 if amax > 0 else 1.0
        c = np.clip(np.round(seg / scale), -127, 127).astype(np.int8)
        codes[lo:hi] = c
        scales[b] = scale
        recon[lo:hi] = c.astype(np.float64) * scale + mean

    original_bits = 32.0 * n
    new_bits = 8.0 * n + 32.0 * n_blocks2 + 32.0  # +32 for the stored global mean
    total_params = float(n) * FIRST_LEVEL_BLOCK_SIZE
    bits_saved_per_param = (original_bits - new_bits) / total_params

    return codes, scales, mean, recon, bits_saved_per_param
