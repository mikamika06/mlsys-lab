def nf4_quantize_indices(w: list[float], block_size: int=64) -> list[int]:
    """Quantize `w` to 4-bit NF4 codebook indices, block-normalized.

    w: 1-D float64 array, length a multiple of `block_size`.
    block_size: number of elements per normalization block (default 64).

    For each contiguous block of `block_size` elements:
      1. scale = max(abs(block))  (1.0 if the block is all zero)
      2. normalized = block / scale
      3. for each normalized value, pick the index (0..15) of the
         nearest NF4 codebook level:
         [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
          -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
          0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
          0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]

    Returns a 1-D int64 array of indices, same length as `w`.

    BUG: this implementation uses 16 EVENLY SPACED levels instead of the
    real NF4 quantile codebook above, and normalizes by a single GLOBAL
    absmax over the whole array instead of a separate absmax PER BLOCK
    -- both wrong for a real NF4 dequant path.
    """
    raise NotImplementedError('your code here')
