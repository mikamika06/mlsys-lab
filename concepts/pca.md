---
title: "What is PCA?"
description: "PCA explained, with a measured explained-variance and reconstruction-error table you can reproduce, including what breaks when you skip centering, plus graded exercises."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is PCA?

PCA is a dimensionality-reduction technique that replaces a dataset's original,
correlated features with a small number of uncorrelated directions ranked by how much
variance each one explains. On the 200×8 correlated dataset measured below, the first
two of those eight directions already capture 91.94% of the total variance. The table
further down tracks that ratio and the reconstruction error it implies, component by
component, computed two independent ways that must agree — and a third way that quietly
does not.

## How it works

Start from a data matrix $X \in \mathbb{R}^{n \times d}$: $n$ rows of observations, $d$
columns of features. Subtract the column means so every feature is centered at zero —
this step is not cosmetic, and the last section of this page measures exactly what
happens if you skip it. Call the centered matrix $X_c$. Its covariance matrix
$C = X_c^\top X_c / (n-1)$ is a $d \times d$ symmetric matrix, and PCA is nothing more
than its eigendecomposition: $C = Q \Lambda Q^\top$, where the columns of $Q$ are
orthonormal directions (the principal components) and the diagonal of $\Lambda$ holds
the variance each direction accounts for, sorted largest first.

An equivalent, more numerically stable route skips forming $C$ at all. The singular
value decomposition $X_c = U \Sigma V^\top$ gives the same directions in the columns of
$V$, and the eigenvalues of $C$ are just $\sigma_i^2 / (n-1)$ for the singular values
$\sigma_i$. Forming $C$ explicitly squares the matrix's condition number, so on
ill-conditioned data the covariance route can diverge from the SVD route in a way that
is entirely avoidable — the subject of
[`alg-covariance-eig-vs-svd-on-ill-conditioned-x`](../tasks/alg-covariance-eig-vs-svd-on-ill-conditioned-x/task.md).
On well-conditioned data, as measured below, both routes agree to floating-point noise.

Keeping only the top $k$ columns of $Q$ (or $V$) and projecting $X_c$ onto them gives a
rank-$k$ approximation. By the Eckart–Young–Mirsky theorem this is the *provably best*
rank-$k$ approximation of the data in squared error — no other linear projection to $k$
dimensions can reconstruct the original matrix more accurately, which is what
[`rws-pca-slice-vs-naive-magnitude-column-drop`](../tasks/rws-pca-slice-vs-naive-magnitude-column-drop/task.md)
asks you to demonstrate against a naive column-dropping baseline. The reconstruction
error at a given $k$ also has a closed form — it equals the sum of the eigenvalues you
discarded, not something you need to recompute by actually reconstructing the matrix,
as in
[`rws-pca-slice-error-equals-tail-eigenvalue-sum`](../tasks/rws-pca-slice-error-equals-tail-eigenvalue-sum/task.md).

This is the same "count the thing exactly, everywhere" spirit as
[memory coalescing](memory-coalescing.md) counting transactions instead of timing a
kernel, or [false sharing](false-sharing.md) counting coherence invalidations instead of
timing a loop: PCA's eigenvalues and its reconstruction error are exact, deterministic
numbers on any machine, which is why every gate on this page is a tolerance on a number
rather than a speed threshold.

## Explained variance and reconstruction error, by component count

The table varies $k$, the number of components kept, on a fixed seeded 200×8 dataset
built from 3 latent factors plus noise and a large nonzero mean. For each $k$ it reports
the cumulative fraction of variance explained and the mean squared reconstruction error,
computed correctly (centered) and then again with centering skipped entirely.

| k | cum. explained variance | reconstruction MSE | cum. var. (no centering) | reconstruction MSE (no centering) |
|---|---|---|---|---|
| 1 | 0.771503 | 0.733227 | 0.896820 | **3.169172** |
| 2 | 0.919403 | 0.258630 | 0.977303 | 0.697146 |
| 3 | 0.995891 | 0.013184 | 0.992346 | 0.235080 |
| 4 | 0.996876 | 0.010025 | 0.999656 | 0.010571 |
| 5 | 0.997755 | 0.007205 | 0.999754 | 0.007569 |
| 6 | 0.998613 | 0.004452 | 0.999843 | 0.004816 |
| 7 | 0.999327 | 0.002161 | 0.999928 | 0.002212 |
| 8 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np
rng = np.random.default_rng(0)
n, d = 200, 8
base = rng.normal(size=(n, 3)) @ rng.normal(size=(3, d))
X = base + 0.15 * rng.normal(size=(n, d))
X += np.array([5., -3., 10., 0., 2., -8., 1., 4.])   # nonzero mean, on purpose
mean, Xc = X.mean(0), X - X.mean(0)

C = (Xc.T @ Xc) / (n - 1)                      # route 1: covariance eigendecomposition
eigvals, eigvecs = np.linalg.eigh(C)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

_, S, _ = np.linalg.svd(Xc, full_matrices=False)   # route 2: SVD, must agree
svd_vals = (S ** 2) / (n - 1)
agree = float(np.max(np.abs(eigvals - svd_vals)))
print(f"eig and svd variances agree to < 1e-12: {agree < 1e-12}")

G = (X.T @ X) / (n - 1)                        # route 3: skip centering entirely
eigvals_nc, eigvecs_nc = np.linalg.eigh(G)
o2 = np.argsort(eigvals_nc)[::-1]
eigvals_nc, eigvecs_nc = eigvals_nc[o2], eigvecs_nc[:, o2]

print("k  cum_var_ratio  recon_mse  cum_var_ratio_uncentered  recon_mse_uncentered")
for k in range(1, d + 1):
    Vk, Uk = eigvecs[:, :k], eigvecs_nc[:, :k]
    ratio = eigvals[:k].sum() / eigvals.sum()
    ratio_nc = eigvals_nc[:k].sum() / eigvals_nc.sum()
    mse = np.mean((X - ((Xc @ Vk) @ Vk.T + mean)) ** 2)
    mse_nc = np.mean((X - (X @ Uk) @ Uk.T) ** 2)
    print(f"{k}  {ratio:.6f}  {mse:.6f}  {ratio_nc:.6f}  {mse_nc:.6f}")
PY
```

The covariance route and the SVD route agree to better than `1e-12`, which is floating-point
noise, not a real disagreement — that is the first thing the snippet prints. The
centered columns read exactly as PCA promises: three real latent factors show up as
three components carrying almost all the variance, and MSE collapses toward zero well
before $k = d$. The uncentered columns are where it breaks. At $k=1$, skipping
centering *reports* a higher explained-variance ratio (0.896820 vs. 0.771503) while the
actual reconstruction is over four times worse (3.169172 vs. 0.733227) — because with a
mean of roughly $(5, -3, 10, 0, 2, -8, 1, 4)$, the single dominant "component" of the
uncentered Gram matrix is mostly pointing at the mean itself, not at the correlation
structure the latent factors created. The two views only converge once $k$ is large
enough that the offset gets absorbed along with everything else.

## Practise it

```bash
mlsys grade alg-pca-via-svd-center-decompose-project
```

[That task](../tasks/alg-pca-via-svd-center-decompose-project/task.md) gates
`pca_svd(X, k)` on `channel_rel_err <= 1e-6` against a centered-SVD reference, with
per-component sign alignment before scoring since a principal component's sign is not
unique. The shipped starter is the trap this page exists to warn about: it is easy to
write a `pca_svd` that runs the SVD correctly but forgets step 1, subtracting the column
mean, and it will fail the gate on any fixture whose mean is not already zero — the same
failure mode measured in the table above, just now enforced by a threshold instead of
read off a printout.

Building up the mechanism from both directions:
[read singular values as variance explained](../tasks/alg-read-singular-values-variance-explained/task.md),
[covariance eigendecomposition vs. SVD on ill-conditioned data](../tasks/alg-covariance-eig-vs-svd-on-ill-conditioned-x/task.md),
[the covariance matrix and its eigenbasis from scratch](../tasks/rws-activation-covariance-eigenbasis/task.md),
[slice error equals the tail eigenvalue sum](../tasks/rws-pca-slice-error-equals-tail-eigenvalue-sum/task.md),
[choosing k for a variance target](../tasks/rws-classify-k-for-a-variance-target/task.md),
[retained variance at a given slice level](../tasks/rws-retained-variance-for-a-slice-level/task.md),
and [PCA vs. naive magnitude column-drop](../tasks/rws-pca-slice-vs-naive-magnitude-column-drop/task.md),
which gates on both an error tolerance and an ordering check that PCA's error is never
worse.

## Common mistakes

- **Skipping centering.** As measured above, this doesn't just shift the numbers a
  little — at $k=1$ it inflates the reported explained-variance ratio from 0.7715 to
  0.8968 while making the actual reconstruction 4.3x worse, because the first
  "component" is mostly describing the mean, not the data's structure.
- **Forming the covariance matrix when the SVD would do.** $C = X_c^\top X_c$ squares
  the condition number of the data. On well-conditioned inputs this costs nothing
  visible; on ill-conditioned ones the eigendecomposition of $C$ can lose precision the
  SVD of $X_c$ never had, which is exactly the comparison in
  [`alg-covariance-eig-vs-svd-on-ill-conditioned-x`](../tasks/alg-covariance-eig-vs-svd-on-ill-conditioned-x/task.md).
- **Confusing eigenvalues with singular values.** The eigenvalues of $C$ are
  $\sigma_i^2/(n-1)$, not $\sigma_i$ itself. Using $\sigma_i$ directly in a
  variance-explained ratio silently changes every number without erroring anywhere.
- **Selecting components by column norm instead of by projection.** Keeping the $k$
  raw features with the largest individual norm and zeroing the rest is not PCA — it
  only ever looks at one column at a time and cannot exploit correlation between
  columns, which is why it is provably no better than PCA's reconstruction error and
  usually worse.
- **Treating sign as meaningful.** Eigenvectors and right-singular vectors are only
  defined up to a sign flip; a solution that differs from a reference by `-1` on some
  columns is not wrong, which is why the graders here align signs before scoring rather
  than comparing raw vectors.

## Where else to practise this

From the [full survey of what exists](../LANDSCAPE.md), this term is **crowded**:

- **[deep-ml.com](https://www.deep-ml.com/problems)** — the closest analog to this bank
  for this specific term: browser-based, actually auto-graded against hidden tests, PCA
  and SVD both listed in its catalog. Worth doing in parallel; it does not show you a
  measured centering-vs-not comparison, it only checks your final numbers.
- **[Machine Learning Specialization (Andrew Ng / DeepLearning.AI)](https://www.coursera.org/specializations/machine-learning-introduction)**
  — a named, auto-graded PCA lab exists in this course, in a guided fill-in-the-blank
  format rather than an open implement-from-spec one, and it is paid.
- **[CS231n Assignment 1](https://cs231n.github.io/assignments2026/assignment1/)** —
  does not cover PCA directly, but its "image features" part builds the same
  eigen-decomposition-of-a-Gram-matrix intuition this page uses, with a currently
  maintained 2026 edition and self-check cells in the notebook.
- **[Data Science from Scratch](https://github.com/joelgrus/data-science-from-scratch)**
  — walks through PCA in plain Python with no NumPy at all, which is a genuinely
  different and useful angle if you want to see every arithmetic step; it is
  narrative code-along, not graded.
- **[ddbourgin/numpy-ml](https://github.com/ddbourgin/numpy-ml)** — a documented
  reference implementation to compare your own against once you have one working;
  ships tests for the maintainer's own code, not for scoring yours.

## References

1. Jolliffe, I.T. & Cadima, J., *Principal component analysis: a review and recent
   developments*, Phil. Trans. R. Soc. A, 2016.
   https://royalsocietypublishing.org/doi/10.1098/rsta.2015.0202
2. Eckart, C. & Young, G., *The approximation of one matrix by another of lower rank*,
   Psychometrika, 1936 — the theorem behind PCA's optimality.
   https://link.springer.com/article/10.1007/BF02288367
3. scikit-learn, *Decomposing signals in components (PCA, ICA, ...)* — the covariance
   and SVD solvers side by side in a widely used implementation.
   https://scikit-learn.org/stable/modules/decomposition.html#pca
