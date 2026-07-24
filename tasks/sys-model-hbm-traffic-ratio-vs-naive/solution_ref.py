import math


def hbm_traffic(N: int, d: int, M: int, elem_bytes: int = 4) -> dict:
    """Model HBM (off-chip memory) traffic, in elements and bytes, for naive
    vs tiled (FlashAttention-style) single-head attention.

    N : sequence length.
    d : head dimension.
    M : on-chip SRAM capacity, in elements (not bytes).
    elem_bytes : bytes per element (4 for float32).

    Naive attention materializes the full (N, N) score matrix S = QK^T and
    the full (N, N) probability matrix P = softmax(S) in HBM:
      - read Q, K, V once each:        3*N*d
      - write S, then read S back:     2*N*N
      - write P, then read P back:     2*N*N
      - write O once:                  N*d
      total elements = 4*N*d + 4*N*N

    Tiled attention picks column-block size Bc = ceil(M / (4*d)) (clamped to
    [1, N]) and row-block size Br = min(Bc, d) (clamped to [1, N]), giving
    Tc = ceil(N / Bc) column tiles. K and V are each streamed from HBM to
    SRAM exactly once over the whole algorithm (2*N*d elements). Q and the
    running output accumulator O are re-streamed once per column tile (the
    outer loop), Q read Tc times and O read+written Tc times:
      total elements = 2*N*d + Tc*N*d + 2*Tc*N*d = 2*N*d + 3*Tc*N*d

    Returns
    -------
    dict with keys:
      naive_bytes, tiled_bytes : int -- total HBM bytes moved by each scheme.
      size_ratio : float -- tiled_bytes / naive_bytes.
    """
    naive_elems = 4 * N * d + 4 * N * N

    Bc = max(1, min(N, math.ceil(M / (4 * d))))
    Br = max(1, min(Bc, d))  # noqa: F841 -- kept for documentation of the model
    Tc = math.ceil(N / Bc)
    tiled_elems = 2 * N * d + 3 * Tc * N * d

    naive_bytes = naive_elems * elem_bytes
    tiled_bytes = tiled_elems * elem_bytes

    return {
        "naive_bytes": naive_bytes,
        "tiled_bytes": tiled_bytes,
        "size_ratio": tiled_bytes / naive_bytes,
    }
