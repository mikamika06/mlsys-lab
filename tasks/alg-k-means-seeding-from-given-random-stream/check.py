import numpy as np

def _reference_kmeans_pp_seed(X, n_clusters, rng_stream):
    """
    Reference implementation of K-Means++ seeding that consumes rng_stream.
    """
    n_samples = X.shape[0]
    indices = np.empty(n_clusters, dtype=np.int64)

    # First center uniformly at random
    first_idx = int(np.floor(rng_stream[0] * n_samples))
    indices[0] = first_idx

    # Distances to nearest chosen center for each point
    dists = np.full(n_samples, np.inf, dtype=np.float64)
    # Update distances after first center
    diff = X - X[first_idx]
    dists = np.minimum(dists, np.sum(diff * diff, axis=1))

    for t in range(1, n_clusters):
        total = dists.sum()
        if total == 0.0:
            # All points are identical to chosen centers; pick uniformly
            idx = int(np.floor(rng_stream[t] * n_samples))
        else:
            cum = np.cumsum(dists) / total
            val = rng_stream[t]
            idx = np.searchsorted(cum, val)
        indices[t] = idx

        # Update distances with the new center
        diff = X - X[idx]
        dists = np.minimum(dists, np.sum(diff * diff, axis=1))

    return indices


def grade(sol, fx) -> dict:
    """
    Grader for kmeans_pp_seed. Generates random test cases and compares
    the student's output to a reference implementation.
    """
    rng = np.random.default_rng(0)
    ok = 1.0

    # Generate several test cases with varying sizes and cluster counts
    for _ in range(5):
        n_samples = rng.integers(10, 50)
        n_features = rng.integers(2, 6)
        k_clusters = rng.integers(2, min(n_samples, 8) + 1)

        X_np = rng.standard_normal((n_samples, n_features))
        rng_stream_np = rng.random(k_clusters)

        X_list = X_np.tolist()
        rng_stream_list = rng_stream_np.tolist()

        try:
            got = sol.kmeans_pp_seed(X_list, int(k_clusters), rng_stream_list)
            ref = _reference_kmeans_pp_seed(X_np, int(k_clusters), rng_stream_np)
        except Exception as e:
            ok = 0.0
            break

        if not isinstance(got, list) or not all(isinstance(x, int) for x in got):
            ok = 0.0
            break

        ref_list = ref.tolist()
        if got != ref_list:
            ok = 0.0
            break

    return {"exact_match": ok}
