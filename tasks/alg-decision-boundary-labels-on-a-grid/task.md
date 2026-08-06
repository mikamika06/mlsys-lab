## Context

The k‑Nearest Neighbours (kNN) algorithm assigns to a query point $q$ the label that occurs most frequently among its $k$ closest training points in Euclidean space.  
For two points $x, y \in \mathbb{R}^d$ the squared distance is

$$
\lVert x-y\rVert^2 = \sum_{i=1}^{d}(x_i-y_i)^2 .
$$

When a dense grid of query points is labelled with kNN we obtain an explicit picture of the decision boundary that separates the classes.

## Task

Implement the function `knn_grid_labels`:

```python
def knn_grid_labels(
    train_points: list[list[float]],
    train_labels: list[int],
    grid_points: list[list[float]],
    k: int = 3
) -> list[list[float]]:
    ...
```

* `train_points` – shape `(N, d)` list of training samples.  
* `train_labels` – shape `(N,)` integer labels (0 … C‑1).  
* `grid_points` – shape `(M, d)` query points on a regular grid.  
* `k` – number of neighbours to consider (default 3).

The function must return an array of shape `(M, C)` containing one‑hot encoded predictions for every grid point. The implementation should be fully vectorised; no explicit Python loops over the queries are allowed.

## Example

```python
from your_module import knn_grid_labels

# Two training points in 2‑D
train_points = [[0, 0], [1, 1]]
train_labels = [0, 1]

# Three grid points
grid_points = [[0.2, 0.2],
                        [0.5, 0.5],
                        [0.8, 0.8]]

preds = knn_grid_labels(train_points, train_labels, grid_points, k=1)
print(preds)
# [[1., 0.]
#  [1., 0.]
#  [0., 1.]]
```

## What the gate checks

The grader computes a reference prediction using Python and compares it to your output with the `argmax_agreement` scorer from `arena.scorers`.  
Your solution must achieve an agreement of at least **99.9 %**:

$$
\frac{\texttt{\# correctly classified grid points}}{\text{total grid points}}
  \;\ge\; 0.999 .
$$

The gate also verifies that the returned array has the correct shape and dtype `float64`.  

A fully vectorised implementation will pass both correctness and performance checks.
