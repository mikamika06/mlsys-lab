## Context
The Mahalanobis distance is a measure of the distance between a point $x$ and a distribution, or between two points $x_i$ and $x_j$ given the covariance matrix $\Sigma$ of the space. Unlike Euclidean distance, it accounts for the correlations of the dataset and is scale-invariant.

The pairwise Mahalanobis distance between two vectors $x_i$ and $x_j$ given the inverse covariance matrix $\Sigma^{-1}$ is defined as:
$$ D_{ij} = \sqrt{(x_i - x_j)^T \Sigma^{-1} (x_i - x_j)} $$

When computing this for all pairs in a dataset $X \in \mathbb{R}^{n \times d}$, a naive loop is extremely slow. By expanding the quadratic form:
$$ (x_i - x_j)^T \Sigma^{-1} (x_i - x_j) = x_i^T \Sigma^{-1} x_i + x_j^T \Sigma^{-1} x_j - 2 x_i^T \Sigma^{-1} x_j $$
we can vectorize the computation using matrix multiplication.

## Task
Write a function `pairwise_mahalanobis(X: list[float], cov_inv: list[float]) -> list[float] that computes the pairwise Mahalanobis distance between all rows of $X$.

- `X`: An $n \times d$ matrix representing $n$ samples in $d$ dimensions.
- `cov_inv`: A $d \times d$ symmetric positive-definite matrix representing the inverse covariance ($\Sigma^{-1}$).

The function should return an $n \times n$ matrix where the $(i, j)$ entry is the Mahalanobis distance between $x_i$ and $x_j$.

## Example
```python

X = [[1.0, 2.0], [3.0, 4.0]]
cov_inv = [[1.0, 0.0], [0.0, 1.0]]  # Identity implies Euclidean distance

D = pairwise_mahalanobis(X, cov_inv)
# D[0, 1] will be sqrt((1-3)^2 + (2-4)^2) = sqrt(8) ~ 2.828
```

## What the gate checks
- The gate checks the maximum relative error (`rel_err`) between your distance matrix and a reference implementation using `scipy.spatial.distance.cdist`.
- `rel_err` must be $\le 10^{-7}$.
