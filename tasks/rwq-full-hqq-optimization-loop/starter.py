def hqq_optimize(W: list[float], scale: float, zero0: float, qmin: int, qmax: int, lp_norm: float, beta0: float, kappa: float, iters: int) -> tuple[list[int], float, list[float]]:
    """
    Run `iters` HQQ half-quadratic passes to refine the zero-point z
    (scale is fixed), then quantize W one final time with the converged z.
    Return (W_q, z, W_dequant). See task.md for the exact recursion,
    including the Lp shrink operator.
    """
    raise NotImplementedError('your code here')
