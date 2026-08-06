import numpy as np
from mlsys.scorers import argmax_agreement

def _ref_knn(train_points, train_labels, grid_points, k):
    # Compute squared Euclidean distances
    dists = ((grid_points[:, None, :] - train_points[None, :, :]) ** 2).sum(axis=2)
    # Indices of the k nearest neighbours for each query point
    idx = np.argpartition(dists, kth=k-1, axis=1)[:, :k]
    neigh_labels = train_labels[idx]          # shape (M, k)
    # Majority vote with tie‑breaking to the smallest label
    counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=train_labels.max()+1),
        axis=1,
        arr=neigh_labels
    )
    preds = counts.argmax(axis=1)             # shape (M,)
    return preds

def grade(sol, fx) -> dict:
    try:
        # deterministic data for reproducibility
        rng = np.random.default_rng(0)
        N, d = 200, 2
        train_points_np = rng.uniform(-5, 5, size=(N, d))
        train_labels_np = rng.integers(0, 3, size=N)   # 3 classes

        # Create a regular grid covering the training data range
        x_min, x_max = train_points_np[:, 0].min(), train_points_np[:, 0].max()
        y_min, y_max = train_points_np[:, 1].min(), train_points_np[:, 1].max()
        xs = np.linspace(x_min, x_max, 50)
        ys = np.linspace(y_min, y_max, 50)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_points_np = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # Reference predictions
        ref_labels = _ref_knn(train_points_np, train_labels_np, grid_points_np, k=3)
        num_classes = int(train_labels_np.max()) + 1
        ref_logits = np.eye(num_classes)[ref_labels]   # one‑hot

        # Convert to plain lists for candidate solution
        train_points = train_points_np.tolist()
        train_labels = train_labels_np.tolist()
        grid_points = grid_points_np.tolist()

        # Candidate predictions from the solution
        cand_logits_raw = sol.knn_grid_labels(
            train_points, train_labels, grid_points, k=3
        )
        cand_logits = np.asarray(cand_logits_raw, dtype=np.float64)

        agreement = argmax_agreement(ref_logits, cand_logits)
    except Exception:
        return {"label_agreement": 0.0}

    return {"label_agreement": float(agreement)}
