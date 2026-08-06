import math
import numpy as np


def best_contiguous_n_block_to_drop(hidden_states: np.ndarray, n: int) -> tuple[int, float]:
    H = np.asarray(hidden_states, dtype=np.float64)
    B, L, D = H.shape

    scores = []
    for s in range(L - n):
        mean_acos = 0.0
        for b in range(B):
            dot_product = 0.0
            norm_a_sq = 0.0
            norm_b_sq = 0.0
            for d in range(D):
                val_a = H[b, s, d]
                val_b = H[b, s + n, d]
                dot_product += val_a * val_b
                norm_a_sq += val_a * val_a
                norm_b_sq += val_b * val_b
            
            denom = math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq)
            if denom == 0.0:
                cosine = 0.0
            else:
                cosine = dot_product / denom
            
            if cosine < -1.0:
                cosine = -1.0
            elif cosine > 1.0:
                cosine = 1.0
            
            mean_acos += math.acos(cosine)
        
        mean_acos /= B
        scores.append(mean_acos)

    best_idx = 0
    best_score = scores[0]
    for i in range(1, len(scores)):
        if scores[i] < best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx), float(best_score)
