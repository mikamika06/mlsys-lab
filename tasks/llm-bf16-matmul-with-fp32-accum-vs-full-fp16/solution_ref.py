import numpy as np

def compare_matmuls(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    Y_true = A @ B
    
    A_view = A.view(np.uint32)
    A_bf = (A_view + 0x7FFF + ((A_view >> 16) & 1)) & 0xFFFF0000
    A_bf = A_bf.view(np.float32)
    
    B_view = B.view(np.uint32)
    B_bf = (B_view + 0x7FFF + ((B_view >> 16) & 1)) & 0xFFFF0000
    B_bf = B_bf.view(np.float32)
    
    Y_bf = A_bf @ B_bf
    err_bf = float(np.max(np.abs(Y_bf - Y_true)))
    
    with np.errstate(over='ignore', invalid='ignore'):
        Y_fp = np.matmul(A.astype(np.float16), B.astype(np.float16)).astype(np.float32)
        err_fp = float(np.max(np.abs(Y_fp - Y_true)))
    
    return err_bf, err_fp
