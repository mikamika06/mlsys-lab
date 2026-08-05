import numpy as np

def knn_majority_vote(Xtr: np.ndarray,
                      ytr: np.ndarray,
                      Xte: np.ndarray,
                      k: int) -> np.ndarray:
    """
    Brute‑force kNN classifier with majority vote and deterministic tie‑break.
    Parameters
    ----------
    Xtr : (n_train, d) array of training samples
    ytr : (n_train,) integer labels for the training samples
    Xte : (n_test, d) array of test samples to classify
    k   : number of nearest neighbours to consider

    Returns
    -------
    preds : (n_test,) array of predicted integer labels
    """
    n_train = Xtr.shape[0]
    n_test = Xte.shape[0]
    d = Xtr.shape[1]
    
    preds = []
    for i in range(n_test):
        dists = []
        for j in range(n_train):
            sq_dist = 0.0
            for f in range(d):
                diff = Xte[i, f] - Xtr[j, f]
                sq_dist += diff * diff
            dists.append(sq_dist)
        
        top_k_idx = []
        for _ in range(k):
            min_val = None
            min_idx = -1
            for j in range(n_train):
                is_used = False
                for used_idx in top_k_idx:
                    if j == used_idx:
                        is_used = True
                        break
                if not is_used:
                    if min_val is None or dists[j] < min_val:
                        min_val = dists[j]
                        min_idx = j
            top_k_idx.append(min_idx)
        
        counts = {}
        for idx in top_k_idx:
            lbl = ytr[idx]
            if lbl not in counts:
                counts[lbl] = 0
            counts[lbl] += 1
            
        max_c = -1
        for lbl, c in counts.items():
            if c > max_c:
                max_c = c
                
        best_lbl = None
        for lbl, c in counts.items():
            if c == max_c:
                if best_lbl is None or lbl < best_lbl:
                    best_lbl = lbl
                    
        preds.append(best_lbl)
        
    return np.array(preds, dtype=np.int64)
