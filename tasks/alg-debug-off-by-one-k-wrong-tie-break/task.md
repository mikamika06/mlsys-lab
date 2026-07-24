## Context

k‑Nearest Neighbours (kNN) is a non‑parametric classifier that assigns to a query point the most frequent label among its $k$ closest training points.  
The distance metric is usually Euclidean:
$$
d(\mathbf{x},\mathbf{y}) = \lVert \mathbf{x}-\mathbf{y}\rVert_2 .
$$

When several neighbours share the same minimal distance, or when the majority vote contains a tie, a deterministic rule must be applied.  A common convention is to choose the **smallest** label value.

A subtle implementation bug is an off‑by‑one error in the number of neighbours considered: using $k+1$ instead of $k$.  
Another frequent mistake is to break ties by selecting the largest label rather than the smallest.  
Both errors silently change the prediction on a handful of inputs but are hard to spot without a reference.

## Task

Implement `predict_knn(X_train, y_train, X_test, k)`:

```python
def predict_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    k: int
) -> np.ndarray:
    ...
```

* `X_train` is an $(m,d)$ array of training samples.  
* `y_train` is a length‑$m$ integer vector of labels (non‑negative, consecutive).  
* `X_test` is an $(n,d)$ array of query points.  
* `k` is the number of neighbours to consider ($1 \le k \le m$).

The function must return a 1‑D NumPy array of length $n$ containing the predicted labels for each test point.  
Use vectorised NumPy operations only; loops over training samples are allowed but should be avoided if possible.

## Example

```python
import numpy as np
X_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_train = np.array([1, 1, 0, 0])
X_test  = np.array([[0.5, 0.5]])
preds   = predict_knn(X_train, y_train, X_test, k=3)
print(preds)          # [1]
```

The query point is equidistant to all training points; the three nearest neighbours are the first three indices (due to stable sorting).  
Their labels are `[1, 1, 0]`, so the majority label is `1`.  The fourth neighbour would change the vote if an off‑by‑one error were present.

## What the gate checks

The grader evaluates your implementation against a reference kNN that uses:

* exactly $k$ neighbours,
* ties broken by choosing the **smallest** label value.

It then computes the fraction of test points for which your predictions match the reference.  
Your solution must achieve an agreement of `1.0` on all provided tests, including adversarial cases designed to expose off‑by‑one and tie‑breaking bugs.
