## Context

The k‑Nearest‑Neighbors (kNN) algorithm predicts a target value for a query point by looking at the $k$ closest points in the training set. For regression we take the arithmetic mean of their labels. Let  
$$X_{\text{train}}\in\mathbb R^{n\times d}$$  
be the matrix of $n$ training samples, each with $d$ features, and let  
$$y_{\text{train}}\in\mathbb R^n$$  
contain the corresponding real‑valued targets. For a query point $x\in\mathbb R^d$, its squared Euclidean distance to a training sample $X_i$ is  
$$\lVert X_i - x\rVert^2 = \sum_{j=1}^d (X_{ij}-x_j)^2.$$  
The set of indices $\mathcal N_k(x)$ of the $k$ smallest distances yields the prediction  
$$\hat y(x) = \frac{1}{k}\sum_{i\in\mathcal N_k(x)} y_{\text{train},i}.$$

## Task

Implement `knn_regression_average`:

```python
def knn_regression_average(X_train: list[list[float]],
                           y_train: list[float],
                           X_query: list[list[float]],
                           k: int) -> list[float]:
    ...
```

The function must return a list of floats of shape `(m,)`, where `m` is the number of rows in `X_query`. Each entry should be the mean target value of the $k$ nearest neighbors from `X_train`. The implementation must use only vectorized Python operations; explicit Python loops over samples are disallowed. The result must have dtype `float64`.

If `k > X_train.shape[0]` a `ValueError` should be raised.

## Example

```python
X_train = [[0., 0.], [1., 0.], [0., 2.]]
y_train = [10., 20., 30.]
X_query = [[0.5, 0.5]]
preds = knn_regression_average(X_train, y_train, X_query, k=2)
print(preds)   # [15.]
```

## What the gate checks

The grader computes a reference prediction using Python and compares it to your output with the global relative L2 error  
$$\mathrm{rel\_err} = \frac{\lVert \hat y_{\text{ref}} - \hat y_{\text{cand}}\rVert}
{\lVert \hat y_{\text{ref}}\rVert + 10^{-12}}.$$  
The solution must satisfy $\mathrm{rel\_err}\le 1\times10^{-8}$ on a set of random test cases. Additionally, the output must be `float64` and have the correct shape.
