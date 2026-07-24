import numpy as np


def select_autotune_tile(M: int, N: int, K: int, num_SMs: int, candidates):
    """Pick the (BM, BN) tile that minimizes the wave-quantization cost.

    cost(BM, BN) = waves + waste / tile_area
      CTAs      = ceil(M/BM) * ceil(N/BN)
      waves     = ceil(CTAs / num_SMs)
      tile_area = (ceil(M/BM)*BM) * (ceil(N/BN)*BN)
      waste     = tile_area - M*N

    `K` scales useful and wasted FLOPs of every candidate equally, so it does
    not affect the argmin; it's accepted only for signature realism.

    Returns
    -------
    (best_idx, costs) : tuple[int, np.ndarray]
        `costs` has shape (len(candidates),); `best_idx` is the index of the
        smallest cost (ties resolved by taking the first occurrence, like
        `np.argmin`).
    """
    n = len(candidates)
    costs = np.empty(n, dtype=np.float64)
    for i, (BM, BN) in enumerate(candidates):
        ctas_m = -(-M // BM)
        ctas_n = -(-N // BN)
        CTAs = ctas_m * ctas_n
        waves = -(-CTAs // num_SMs)
        tile_area = (ctas_m * BM) * (ctas_n * BN)
        waste = tile_area - M * N
        costs[i] = waves + waste / tile_area

    best_idx = int(np.argmin(costs))
    return best_idx, costs
