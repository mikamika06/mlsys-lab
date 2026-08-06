import numpy as np
import math

def angular_distance_per_layer(states_a, states_b):
    result = {}
    for key in states_a:
        A = np.asarray(states_a[key], dtype=np.float64)
        B = np.asarray(states_b[key], dtype=np.float64)
        dist_list = []
        for i in range(A.shape[0]):
            dot = 0.0
            norm_a_sq = 0.0
            norm_b_sq = 0.0
            for j in range(A.shape[1]):
                val_a = A[i, j]
                val_b = B[i, j]
                dot += val_a * val_b
                norm_a_sq += val_a * val_a
                norm_b_sq += val_b * val_b
            normA = math.sqrt(norm_a_sq)
            normB = math.sqrt(norm_b_sq)
            cos = dot / (normA * normB)
            if cos < -1.0:
                cos_clipped = -1.0
            elif cos > 1.0:
                cos_clipped = 1.0
            else:
                cos_clipped = cos
            dist_val = math.acos(cos_clipped) / math.pi
            dist_list.append(dist_val)
        result[key] = np.asarray(dist_list, dtype=np.float64)
    return result
