## Context

The $k$-Nearest Neighbors classifier predicts a label from the labels of the closest training examples. For a query point $x$, the distance to a training point $x_i$ can be measured with squared Euclidean distance:

$$d(x, x_i)^2 = \sum_{j=1}^{d}(x_j - x_{i,j})^2.$$

Leave-one-out (LOO) evaluation predicts the label of each training point while excluding that point from its own neighborhood. If the point itself is accidentally included, its distance is $0$ and it will usually dominate the neighbors, causing data leakage.

For a dataset $X \in \mathbb{R}^{n \times d}$ with labels $y$, the LOO prediction for row $i$ is computed using the $k$ closest rows from all rows except $i$. The predicted class is the majority label among those neighbors, with ties resolved by choosing the smallest class index.

## Task

Implement `loo_knn_predict(X, y, k, n_classes)`:

```python
def loo_knn_predict(X: list[list[float]], y: list[int], k: int, n_classes: int) -> list[int]:
    ...
```

The function receives:
- `X`: a floating point array of shape $(n, d)$ containing feature vectors.
- `y`: an integer array of shape $(n,)$ containing class labels in the range $[0, n\_classes)$.
- `k`: the number of neighbors to use.
- `n_classes`: the number of possible classes.

Return an integer list of shape $(n,)$ containing the LOO kNN predictions.

The implementation must exclude each sample from its own neighbor search. Use Python operations rather than fitting an external model.

## Example

```python

X = [[0.0, 0.0],
              [0.1, 0.0],
              [2.0, 2.0],
              [2.1, 2.0]]
y = [0, 0, 1, 1]

pred = loo_knn_predict(X, y, 1, 2)
# array([0, 0, 1, 1])
```

## What the gate checks

The gate computes a reference LOO kNN implementation directly from Python operations. It compares the submitted predictions against the oracle predictions using `argmax_agreement`.

A prediction array passes only when the fraction of matching predicted classes is $1.0$. Cases include duplicate points and values where accidentally keeping the zero self-distance changes the result.
