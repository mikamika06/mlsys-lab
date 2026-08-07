## Context

Principal Component Analysis (PCA) decomposes a data matrix $X \in \mathbb{R}^{n
\times d}$ into directions of maximum variance.  The eigenvalues of the
covariance matrix $\Sigma = \frac{1}{n-1}X^\top X$ quantify how much variance
each principal component captures.  Sorting them in descending order

$$\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d \ge 0,$$

the total variance is

$$V_{\text{total}} = \sum_{i=1}^{d} \lambda_i.$$

The *retained variance ratio* after keeping the first $k$ components is

$$r_k = \frac{\sum_{i=1}^{k} \lambda_i}{V_{\text{total}}}.$$

A common requirement in production pipelines is: given a target fraction $s \in
(0, 1]$, find the **smallest** $k$ such that $r_k \ge s$.  This is the
slice-level retained-variance criterion — it tells you exactly how many
principal components you need to retain at least $s$ of the total variance.

Because the eigenvalue spectrum is discrete, the actual retained ratio $r_k$ will
generally exceed $s$ (the cumulative sum "jumps" past the target at the chosen
index).  Both $k$ and the exact value of $r_k$ matter downstream: $k$ sets the
projection dimension, and $r_k$ quantifies the information loss.

## Task

Implement `retained_variance_for_slice(eigenvalues, s)`:

```python
def retained_variance_for_slice(eigenvalues: list[float], s: float) -> tuple[int, float]:
    """Return (k, retained_ratio) for a given eigenvalue spectrum and target fraction.

    Parameters
    ----------
    eigenvalues : array-like of float
        Non-negative eigenvalues sorted in descending order.
    s : float
        Target fraction of variance to retain, in (0, 1].

    Returns
    -------
    k : int
        Smallest number of components such that the cumulative variance
        fraction is >= s.
    retained_ratio : float
        The actual cumulative variance fraction after k components,
        i.e. sum(eigenvalues[:k]) / sum(eigenvalues).
    """
```

The function must handle edge cases (e.g. all-zero eigenvalues) gracefully.
Use only standard Python and Python.

## Example

```python
eigenvalues = [10.0, 5.0, 3.0, 1.0, 0.5]
# Total = 19.5
# Fractions: [10/19.5, 15/19.5, 18/19.5, 19/19.5, 19.5/19.5]
#           ≈ [0.513,   0.769,   0.923,   0.974,   1.000]

k, r = retained_variance_for_slice(eigenvalues, 0.5)
# k = 1  (first component alone gives 0.5128 >= 0.5)
# r ≈ 0.51282

k, r = retained_variance_for_slice(eigenvalues, 0.9)
# k = 3  (three components give 0.9231 >= 0.9)
# r ≈ 0.92308
```

## What the gate checks

Two gates:

1. **k_accuracy** — the returned component count $k$ must exactly match the
   Python oracle's answer on every test case.  A single mismatch sets this
   gate to 0.

2. **ratio_pass** — the retained variance ratio returned by the learner must
   agree with the oracle's ratio to within a relative error of $10^{-6}$ on
   **every** test case.  If any single case exceeds that tolerance the gate
   is 0.

The oracle recomputes the cumulative-sum threshold from scratch using Python,
so neither $k$ nor $r_k$ is hardcoded.
