## Context

The Manhattan or L1 distance between two vectors $a,b\in \mathbb{R}^d$ is defined as

$$
\lVert a-b\rVert_1 = \sum_{i=1}^{d}\lvert a_i - b_i\rvert .
$$

For a set of points stored row‑wise in a 2‑D list $X\in\mathbb{R}^{n\times d}$, the pairwise distance matrix $D$ has entries
$D_{ij}=\lVert X_i-X_j\rVert_1$.  
A naive double loop costs $O(n^2d)$ Python bytecode events.  Python broadcasting allows us to compute all differences in a single vectorised expression and then sum over the feature axis, yielding an $O(nd^2)$‑time, $O(n^2)$‑space operation with only a handful of line events.

## Task

Implement `pairwise_l1_matrix(X, Y=None)`:

```python
def pairwise_l1_matrix(X: list[list[float]], Y: list[list[float]] | None = None) -> list[list[float]]:
    ...
```

The function accepts a 2‑D array $X$ and an optional second array $Y$.  
If `Y` is omitted it defaults to `X`.  The return value must be a list of shape
$(n,m)$ where $n=\text{len}(X)$ and $m=\text{len}(Y)$.  Each entry $(i,j)$ should contain the Manhattan distance between row $i$ of $X$ and row $j$ of $Y$.  
The implementation must use only vectorised Python operations; no explicit Python loops are allowed.  The result type must be `float64`.

## Example

```python
X = [[0, 1], [3, -2]]
D = pairwise_l1_matrix(X)
# [[0., 5.]
#  [5., 0.]]
```

## What the gate checks

Two metrics are evaluated:

* **mse** – the mean squared error between your output and a reference implementation based on Python broadcasting.  
  The gate requires `mse <= 1e-12`.

* **op_count** – the number of Python line events recorded by the tracer during execution.  
  Your solution must stay below 50 events; any explicit loop will exceed this limit.

The grader uses a deterministic random seed to generate test matrices, so your implementation will be evaluated on several independent cases.
