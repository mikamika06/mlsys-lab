def knn_majority_vote(Xtr: list[list[float]],
                      ytr: list[int],
                      Xte: list[list[float]],
                      k: int) -> list[int]:
    """
    Brute‑force kNN classifier with majority vote and deterministic tie‑break.
    Parameters
    ----------
    Xtr : list of lists of training samples (n_train, d)
    ytr : list of integer labels for the training samples (n_train,)
    Xte : list of lists of test samples to classify (n_test, d)
    k   : number of nearest neighbours to consider

    Returns
    -------
    preds : list of predicted integer labels (n_test,)
    """
    n_train = len(Xtr)
    n_test = len(Xte)
    d = len(Xtr[0])

    preds = []
    for i in range(n_test):
        dists = []
        for j in range(n_train):
            sq_dist = 0.0
            for f in range(d):
                diff = Xte[i][f] - Xtr[j][f]
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

    return preds
