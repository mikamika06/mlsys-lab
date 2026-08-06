import numpy as np

def knn_regression_average(X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_query: np.ndarray,
                           k: int) -> np.ndarray:
    X_train_f = np.asarray(X_train, dtype=np.float64)
    y_train_f = np.asarray(y_train, dtype=np.float64)
    X_query_f = np.asarray(X_query, dtype=np.float64)
    n_train = X_train_f.shape[0]
    if k > n_train:
        raise ValueError("k cannot exceed number of training samples")
    
    n_query = X_query_f.shape[0]
    d = X_train_f.shape[1]
    
    preds = np.zeros(n_query, dtype=np.float64)
    
    for i in range(n_query):
        dists = []
        for j in range(n_train):
            d_sq = 0.0
            for l in range(d):
                diff = X_train_f[j, l] - X_query_f[i, l]
                d_sq += diff * diff
            dists.append([d_sq, j])
            
        for p in range(k):
            min_idx = p
            for q in range(p + 1, n_train):
                if dists[q][0] < dists[min_idx][0]:
                    min_idx = q
            
            temp = dists[p]
            dists[p] = dists[min_idx]
            dists[min_idx] = temp
            
        val_sum = 0.0
        for p in range(k):
            val_sum += y_train_f[dists[p][1]]
            
        preds[i] = val_sum / k
        
    return preds
