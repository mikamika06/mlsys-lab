import numpy as np

def per_block_int8_matmul(A: np.ndarray, B: np.ndarray, block_size: int) -> np.ndarray:
    M, K = A.shape
    K2, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    for k in range(0, K, block_size):
        A_b = A[:, k:k+block_size]
        B_b = B[k:k+block_size, :]

        sA = np.max(np.abs(A_b), axis=1, keepdims=True) / 127.0
        sA = np.clip(sA, 1e-9, None)
        Aq = np.round(A_b / sA)

        sB = np.max(np.abs(B_b), axis=0, keepdims=True) / 127.0
        sB = np.clip(sB, 1e-9, None)
        Bq = np.round(B_b / sB)

        C += np.dot(Aq, Bq) * (sA @ sB)
    return C
