import numpy as np
import scipy.ndimage as nd

def grade(sol, fx) -> dict:
    np.random.seed(42)
    N, D = 100, 5
    k = 4
    X = np.random.randn(N, D)
    labels = np.random.randint(0, k, size=N)
    
    # Run user solution
    user_centroids = sol.recover_centroids(X, labels)
    
    # Oracle: calculate reference using scipy
    ref_centroids = np.zeros((k, D))
    for d in range(D):
        ref_centroids[:, d] = nd.mean(X[:, d], labels=labels, index=np.arange(k))
    
    err = np.max(np.abs(user_centroids - ref_centroids))
    return {"max_abs_err": float(err)}
