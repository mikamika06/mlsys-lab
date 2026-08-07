def select_autotune_tile(M: int, N: int, K: int, num_SMs: int, candidates: list[tuple[int, int]]) -> tuple[int, list[float]]:
    """Pick the (BM, BN) tile that minimizes the wave-quantization cost.

    Parameters
    ----------
    M, N, K : int
        GEMM problem shape (C[M,N] = A[M,K] @ B[K,N]).
    num_SMs : int
        Number of streaming multiprocessors on the target GPU.
    candidates : Sequence[tuple[int, int]]
        Candidate (BM, BN) tile shapes to score.

    Returns
    -------
    (best_idx, costs) : tuple[int, list[float]]
        `costs` has shape (len(candidates),) — see task.md for the exact
        cost formula. `best_idx` is the index of the smallest cost.
    """
    raise NotImplementedError('your code here')
