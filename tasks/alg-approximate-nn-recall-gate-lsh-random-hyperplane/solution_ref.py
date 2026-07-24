import numpy as np

def lsh_recall(A: np.ndarray,
               Q: np.ndarray,
               k: int,
               t: int,
               seed: int) -> float:
    rng = np.random.default_rng(seed)
    d = A.shape[1]
    planes = rng.standard_normal((t, d))
    data_bits = (A @ planes.T) >= 0
    query_bits = (Q @ planes.T) >= 0

    buckets = []
    for col in range(t):
        bucket = {}
        bits = data_bits[:, col]
        for idx, bit in enumerate(bits):
            key = int(bit)
            bucket.setdefault(key, []).append(idx)
        buckets.append(bucket)

    recalls = []
    n = len(A)
    for qi in range(Q.shape[0]):
        cand_set = set()
        for col in range(t):
            key = int(query_bits[qi, col])
            cand_set.update(buckets[col].get(key, []))
        if not cand_set:
            cand_indices = np.arange(n)
        else:
            cand_indices = np.array(list(cand_set), dtype=int)

        cand_vecs = A[cand_indices]
        dists = np.linalg.norm(cand_vecs - Q[qi], axis=1)
        topk_local = np.argsort(dists)[:k]
        approx_idx = cand_indices[topk_local]

        all_dists = np.linalg.norm(A - Q[qi], axis=1)
        exact_idx = np.argpartition(all_dists, k)[:k]
        recall = len(set(approx_idx) & set(exact_idx)) / k
        recalls.append(recall)

    return float(np.mean(recalls))
