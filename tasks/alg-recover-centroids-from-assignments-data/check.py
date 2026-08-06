import numpy as np
import scipy.ndimage as nd

def grade(sol, fx) -> dict:
    np.random.seed(42)
    N, D = 100, 5
    k = 4
    X_np = np.random.randn(N, D)
    labels_np = np.random.randint(0, k, size=N)

    X_list = X_np.tolist()
    labels_list = labels_np.tolist()

    # Run user solution
    user_centroids = sol.recover_centroids(X_list, labels_list)

    # Oracle: calculate reference using scipy
    ref_centroids = np.zeros((k, D))
    for d in range(D):
        ref_centroids[:, d] = nd.mean(X_np[:, d], labels=labels_np, index=np.arange(k))

    err = np.max(np.abs(np.array(user_centroids) - ref_centroids))
    return {"max_abs_err": float(err)}
