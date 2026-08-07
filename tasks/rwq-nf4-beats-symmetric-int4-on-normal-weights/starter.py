import math

def nf4_vs_int4_mse(w: list[float]) -> tuple[float, float]:
    """
    Return (mse_nf4, mse_int4): reconstruction MSE of w quantized with the
    fixed 16-level NF4 codebook (normalize by absmax, snap to nearest
    level, scale back), vs symmetric INT4 (scale = absmax/7, codes in
    [-8, 7]). See task.md for the exact NF4 codebook and formulas.
    """
    raise NotImplementedError('your code here')
