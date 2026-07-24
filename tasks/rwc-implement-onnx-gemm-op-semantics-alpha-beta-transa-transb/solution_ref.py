import numpy as np

def gemm(A: np.ndarray,
         B: np.ndarray,
         C: np.ndarray | None = None,
         alpha: float = 1.0,
         beta: float = 1.0,
         transA: bool = False,
         transB: bool = False) -> np.ndarray:
    A_mat = np.asarray(A, dtype=np.float64)
    B_mat = np.asarray(B, dtype=np.float64)
    if transA:
        A_mat = A_mat.T
    if transB:
        B_mat = B_mat.T

    Y = alpha * (A_mat @ B_mat)

    if C is not None:
        C_arr = np.asarray(C, dtype=np.float64)
        Y += beta * np.broadcast_to(C_arr, Y.shape)

    return Y
