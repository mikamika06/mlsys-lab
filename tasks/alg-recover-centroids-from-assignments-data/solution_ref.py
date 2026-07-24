import numpy as np

def recover_centroids(X, labels):
    k = np.max(labels) + 1
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])
