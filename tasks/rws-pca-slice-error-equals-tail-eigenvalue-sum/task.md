## Context

Compressing a data matrix $X \in \mathbb{R}^{n\times d}$ (rows = samples,
columns = features) by keeping only its top-$k$ principal directions means
projecting every row onto the $k$-dimensional subspace spanned by the
leading eigenvectors of the (uncentered) second-moment / Gram matrix

$$
G = X^T X \in \mathbb{R}^{d\times d}, \qquad G = V\Lambda V^T,
$$

with eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$ and
orthonormal eigenvectors as the columns of $V$. Let $Q_k \in
\mathbb{R}^{d\times k}$ be the first $k$ columns of $V$ (the top-$k$
eigenvectors). Projecting $X$ onto that subspace and mapping back to the
original $d$ coordinates gives $X Q_k Q_k^T$, and the reconstruction error

$$
\lVert X - X Q_k Q_k^T \rVert_F^2
$$

has a closed form that never requires explicitly forming the projection —
it equals the sum of the eigenvalues that were **dropped**:

$$
\lVert X - X Q_k Q_k^T \rVert_F^2 = \sum_{i=k+1}^{d} \lambda_i .
$$

This "slice error = tail eigenvalue sum" identity is the PCA/Eckart-Young
fact that makes it cheap to evaluate many candidate compression ratios $k$
without ever materializing the low-rank reconstruction: just look at how
much eigenvalue mass sits below the cut.

## Task

Implement `pca_slice_error`:

```python
def pca_slice_error(X: np.ndarray, k: int) -> float:
    ...
```

* `X` — 2-D `float` array of shape $(n,d)$. **Not** assumed to be
  mean-centered — use $X$ exactly as given, no centering.
* `k` — int, number of top eigenvectors to keep, $0 \le k \le d$.

Compute the projection error **directly**: form $G = X^T X$,
eigendecompose it, take $Q_k$ as the top-$k$ eigenvectors, project
$X_{\text{rec}} = X Q_k Q_k^T$, and return
$\lVert X - X_{\text{rec}} \rVert_F^2$ as a Python `float`. (Whether you
compute the Frobenius norm by explicitly forming $X_{\text{rec}}$ or by
some algebraically equivalent NumPy expression is up to you — the point is
that your number must equal the tail-eigenvalue-sum identity above, for
every $k$.)

## Example

$X = \begin{bmatrix}3 & 0\\ 0 & 1\end{bmatrix}$: $G = X^TX =
\begin{bmatrix}9&0\\0&1\end{bmatrix}$ has eigenvalues $\lambda_1=9,\
\lambda_2=1$. For $k=1$, $Q_1 = [1,0]^T$, $XQ_1Q_1^T =
\begin{bmatrix}3&0\\0&0\end{bmatrix}$, so the error is
$\lVert X - XQ_1Q_1^T\rVert_F^2 = 0^2+0^2+0^2+1^2 = 1$, matching
$\sum_{i=2}^2 \lambda_i = 1$.

## What the gate checks

**max_abs_err** — the grader loads a fixture matrix (`pca_x.npy`, an
80x12 matrix built from a fixed spectrum) with a keep-count
(`pca_k.npy`), plus several independently generated random matrices and
`k` values (including $k=0$ and $k=d$), computes $\sum_{i>k}\lambda_i$ with
`np.linalg.eigh(X.T @ X)` as the oracle, and checks
$|\,\hat E - \sum_{i>k}\lambda_i\,| \le 10^{-8}$ for every case. Forgetting
to use $X^T X$ (e.g. eigendecomposing $X$ itself, which isn't square),
projecting with the *bottom* eigenvectors instead of the top ones, or
mixing up $\lambda_i$ with $\sqrt{\lambda_i}$ will all produce a visibly
wrong error.
