import numpy as np

def _reference_inertia_sequence(X, n_clusters, max_iter):
    rng = np.random.default_rng(0)
    centroids = X[rng.choice(len(X), size=n_clusters, replace=False)]
    seq = []
    for _ in range(max_iter):
        # assignment step
        diff = X[:, None, :] - centroids[None, :, :]
        d2 = np.sum(diff**2, axis=2)
        labels = np.argmin(d2, axis=1)

        # inertia
        inertia = np.sum((X - centroids[labels])**2)
        seq.append(float(inertia))

        # update centroids
        counts = np.bincount(labels, minlength=n_clusters)
        sums = np.array([np.bincount(labels, weights=X[:, d], minlength=n_clusters) for d in range(X.shape[1])]).T
        new_centroids = sums / counts[:, None]
        mask = counts == 0
        new_centroids[mask] = centroids[mask]

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return seq

def grade(sol, fx) -> dict:
    # Fixed dataset and parameters for grading
    X = np.array([[0., 0.], [1., 0.], [0., 2.], [3., 4.], [5., 6.]])
    n_clusters = 2
    max_iter = 10

    try:
        cand_seq = sol.inertia_sequence(X, n_clusters, max_iter)
    except Exception:
        return {"mse": 1e9, "monotonic": 0}

    ref_seq = _reference_inertia_sequence(X, n_clusters, max_iter)

    # Mean‑squared error (relative L2)
    if len(cand_seq) != len(ref_seq):
        mse_val = 1e9
    else:
        diff = np.array(cand_seq) - np.array(ref_seq)
        denom = np.linalg.norm(np.array(ref_seq)) + 1e-12
        mse_val = np.linalg.norm(diff) / denom

    # Monotonicity check (tolerance 1e-9)
    mono_flag = 1
    for a, b in zip(cand_seq[:-1], cand_seq[1:]):
        if b - a > 1e-9:
            mono_flag = 0
            break

    return {"mse": float(mse_val), "monotonic": float(mono_flag)}
