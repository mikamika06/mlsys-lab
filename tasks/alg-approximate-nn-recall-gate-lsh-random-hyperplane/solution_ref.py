import numpy as np
import math

def lsh_recall(A: np.ndarray,
               Q: np.ndarray,
               k: int,
               t: int,
               seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    d = A.shape[1]
    m = Q.shape[0]
    planes = rng.standard_normal((t, d))

    data_bits = []
    for i in range(n):
        row_bits = []
        for j in range(t):
            dot_val = 0.0
            for dim in range(d):
                dot_val += float(A[i, dim]) * float(planes[j, dim])
            row_bits.append(dot_val >= 0.0)
        data_bits.append(row_bits)

    query_bits = []
    for i in range(m):
        row_bits = []
        for j in range(t):
            dot_val = 0.0
            for dim in range(d):
                dot_val += float(Q[i, dim]) * float(planes[j, dim])
            row_bits.append(dot_val >= 0.0)
        query_bits.append(row_bits)

    buckets = []
    for col in range(t):
        bucket = {}
        for idx in range(n):
            bit = data_bits[idx][col]
            key = int(bit)
            if key not in bucket:
                bucket[key] = []
            bucket[key].append(idx)
        buckets.append(bucket)

    recalls = []
    for qi in range(m):
        cand_set = set()
        for col in range(t):
            key = int(query_bits[qi][col])
            if key in buckets[col]:
                for item in buckets[col][key]:
                    cand_set.add(item)
        
        if not cand_set:
            cand_indices = list(range(n))
        else:
            cand_indices = list(cand_set)

        dists = []
        for idx in cand_indices:
            dist_sq = 0.0
            for dim in range(d):
                diff = float(A[idx, dim]) - float(Q[qi, dim])
                dist_sq += diff * diff
            dists.append(math.sqrt(dist_sq))

        topk_local_idx = sorted(range(len(dists)), key=lambda i: dists[i])[:k]
        approx_idx = [cand_indices[i] for i in topk_local_idx]

        all_dists = []
        for idx in range(n):
            dist_sq = 0.0
            for dim in range(d):
                diff = float(A[idx, dim]) - float(Q[qi, dim])
                dist_sq += diff * diff
            all_dists.append(math.sqrt(dist_sq))

        exact_idx = sorted(range(len(all_dists)), key=lambda i: all_dists[i])[:k]

        overlap = 0
        exact_set = set(exact_idx)
        for a_idx in set(approx_idx):
            if a_idx in exact_set:
                overlap += 1
                
        recall = float(overlap) / float(k)
        recalls.append(recall)

    sum_recalls = 0.0
    for r in recalls:
        sum_recalls += r
        
    return float(sum_recalls / len(recalls))
