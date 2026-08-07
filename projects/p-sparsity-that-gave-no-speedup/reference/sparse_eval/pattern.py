import numpy as np


def verify_24_pattern(matrix: np.ndarray) -> bool:
    if matrix.ndim != 2 or matrix.shape[1] % 4 != 0:
        return False
    M, K = matrix.shape
    num_blocks = K // 4
    for r in range(M):
        for c in range(num_blocks):
            block = matrix[r, c * 4 : (c + 1) * 4]
            if np.count_nonzero(np.abs(block) > 1e-6) > 2:
                return False
    return True


def compress_24_matrix(matrix: np.ndarray) -> tuple:
    M, K = matrix.shape
    num_blocks = K // 4
    cw = np.zeros((M, num_blocks * 2), dtype=matrix.dtype)
    meta = np.zeros((M, num_blocks), dtype=np.uint8)
    for r in range(M):
        for c in range(num_blocks):
            block = matrix[r, c * 4 : (c + 1) * 4]
            nz = np.where(np.abs(block) > 1e-6)[0]
            if len(nz) == 0:
                idx0, idx1 = 0, 1
            elif len(nz) == 1:
                idx0 = nz[0]
                idx1 = (idx0 + 1) % 4
            else:
                idx0, idx1 = nz[0], nz[1]
            meta[r, c] = np.uint8(idx0 * 4 + idx1)
            cw[r, c * 2] = block[idx0]
            cw[r, c * 2 + 1] = block[idx1]
    return cw, meta


def decompress_24_matrix(compressed_weights: np.ndarray, metadata: np.ndarray, K: int) -> np.ndarray:
    M = compressed_weights.shape[0]
    num_blocks = K // 4
    out = np.zeros((M, K), dtype=compressed_weights.dtype)
    for r in range(M):
        for c in range(num_blocks):
            code = int(metadata[r, c])
            idx0 = code // 4
            idx1 = code % 4
            out[r, c * 4 + idx0] = compressed_weights[r, c * 2]
            out[r, c * 4 + idx1] = compressed_weights[r, c * 2 + 1]
    return out
