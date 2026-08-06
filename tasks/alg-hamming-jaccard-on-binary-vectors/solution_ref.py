import numpy as np

def hamming_and_jaccard(B):
    B = np.asarray(B, dtype=np.uint8)
    n, d = B.shape[0], B.shape[1]
    H = np.empty((n, n), dtype=np.int64)
    J = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            h_count = 0
            inter_count = 0
            union_count = 0
            for k in range(d):
                b_i = bool(B[i, k])
                b_j = bool(B[j, k])
                if b_i != b_j:
                    h_count += 1
                if b_i and b_j:
                    inter_count += 1
                if b_i or b_j:
                    union_count += 1
            H[i, j] = h_count
            if union_count == 0:
                J[i, j] = 1.0
            else:
                J[i, j] = float(inter_count) / float(union_count)
    return H, J
