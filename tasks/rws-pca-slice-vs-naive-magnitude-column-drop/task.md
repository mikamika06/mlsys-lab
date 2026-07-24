## Context

"Reduce to $k$ dimensions" can mean very different things depending on
*how* you throw away information. The naive approach — keep the $k$
raw feature columns with the largest norm, zero out the rest — only
ever looks at each column in isolation and can't exploit correlations
between features. **PCA** instead finds the single best $k$-dimensional
*linear subspace* (any combination of the original columns, not just an
axis-aligned subset) to project onto. By the Eckart–Young–Mirsky
theorem, the truncated-SVD reconstruction is the **globally optimal**
rank-$k$ approximation of a matrix in Frobenius norm — so for the same
$k$, PCA's reconstruction MSE is *provably* never worse than any other
rank-$\le k$ reconstruction, including the naive column-drop.

### The two reconstructions

Given data $X \in \mathbb{R}^{n \times d}$ and target dimension $k$:

**PCA** — mean-center $X_c = X - \bar X$ (column-wise mean), take the
SVD $X_c = U\Sigma V^\top$, keep the first $k$ right-singular vectors
$V_k \in \mathbb{R}^{d\times k}$ (top-$k$ principal directions), project
and reconstruct:
$$
Z = X_c V_k, \qquad \widehat X_{pca} = Z V_k^\top + \bar X.
$$

**Naive** — rank columns by L2 norm $\lVert X_{:,j}\rVert_2$ descending,
keep the top $k$ columns unchanged, and zero every other column
entirely:
$$
\widehat X_{naive}[:, j] = \begin{cases} X[:, j] & j \text{ among the } k \text{ largest-norm columns} \\ 0 & \text{otherwise} \end{cases}
$$

Both report reconstruction MSE against the **original** $X$:
$\frac{1}{nd}\sum (X - \widehat X)^2$.

## Task

Implement:

```python
def pca_vs_naive_mse(X: np.ndarray, k: int) -> tuple[float, float]:
    ...
```

* `X` — `(n, d)` data matrix.
* `k` — target reduced dimension, `1 <= k < d`.

Return `(mse_pca, mse_naive)` as defined above.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(50, 10)) * rng.uniform(0.5, 2.0, size=(1, 10))
mse_pca, mse_naive = pca_vs_naive_mse(X, k=4)
assert mse_pca <= mse_naive + 1e-9   # Eckart-Young: PCA is the optimal rank-k approximation
```

## What the gate checks

* **max_abs_err** — both `mse_pca` and `mse_naive` must match a NumPy
  oracle implementing the exact formulas above to within $10^{-6}$
  absolute error, over several random `(X, k)` cases (fixed seed).
* **ordering_ok** — on every case, `mse_pca` must be `<= mse_naive`
  (within a `1e-9` numerical-tolerance slack) — a solution that swaps
  the two computations, or gets the PCA reconstruction wrong, will
  usually violate this.
