## Context

Every symmetric positive definite matrix $A \in \mathbb{R}^{n \times n}$ has a unique
factorisation

$$
A = L L^{\mathsf T},
$$

where $L$ is lower triangular with strictly positive diagonal. This is the **Cholesky
decomposition** — the workhorse behind Gaussian sampling, Kalman filters, natural
gradient / K-FAC preconditioners, and second-order optimisers, because once you have
$L$ a linear solve costs two triangular substitutions instead of a full LU.

The Cholesky–Banachiewicz recurrence builds $L$ column by column. Writing out the
$(i,j)$ entry of $LL^{\mathsf T}$ for $i \geq j$:

$$
A_{ij} = \sum_{p=1}^{j} L_{ip} L_{jp},
$$

and solving for the unknown entry gives

$$
L_{jj} = \sqrt{A_{jj} - \sum_{p=1}^{j-1} L_{jp}^2},
\qquad
L_{ij} = \frac{1}{L_{jj}}\left(A_{ij} - \sum_{p=1}^{j-1} L_{ip} L_{jp}\right), \quad i > j .
$$

Each column $j$ therefore needs only entries from columns $1 \dots j-1$, which are
already known. Note that the algorithm is self-checking: if the quantity under the
square root is not positive, $A$ was not positive definite.

## Task

Implement `cholesky_lower` in `solve.py`:

```python
def cholesky_lower(A: np.ndarray) -> np.ndarray:
    ...
```

* `A` — symmetric positive definite, shape $(n, n)$.
* Return `L` of shape $(n, n)$, dtype `float64`, with

  * the strict upper triangle **exactly zero**,
  * $L_{ii} > 0$ for every $i$,
  * $L L^{\mathsf T} = A$.

The positive-diagonal convention is what makes the factor unique, so there is exactly
one correct answer.

**`np.linalg.cholesky` (and `scipy.linalg.cholesky` / `cho_factor`) are off limits.**
The grader wraps them and records any call. Write the recurrence yourself; a per-column
loop with vectorised inner products is fine.

## Example

```python
import numpy as np

A = np.array([[4.0, 2.0],
              [2.0, 5.0]])

L = cholesky_lower(A)
# L -> [[2.0, 0.0],
#       [1.0, 2.0]]
# L @ L.T -> A
```

Check by hand: $L_{11} = \sqrt{4} = 2$, $L_{21} = 2/2 = 1$,
$L_{22} = \sqrt{5 - 1^2} = 2$.

## What the gate checks

Three matrices are graded: the hidden SPD fixture, a second SPD matrix the grader
generates itself with a seeded RNG, and a small hand-written $3 \times 3$ matrix. The
reference factor comes from `numpy.linalg.cholesky`, called **before** the library
routines are wrapped — nothing is hardcoded.

| metric | meaning | gate |
| --- | --- | --- |
| `recon_max_abs_err` | $\max_{ij} \lvert (LL^{\mathsf T} - A)_{ij} \rvert$ | $\leq 10^{-10}$ |
| `factor_max_abs_err` | $\max_{ij} \lvert (L - L_{\text{ref}})_{ij} \rvert$ | $\leq 10^{-10}$ |
| `upper_violation` | largest magnitude in the strict upper triangle of $L$ | $\leq 10^{-15}$ |
| `min_diag` | smallest diagonal entry of $L$ | $> 0$ |
| `builtin_used` | 1 if a built-in Cholesky was called, else 0 | $\leq 0$ |

Worst case over the three matrices is reported for each error metric. A symmetric
square root from `eigh` reconstructs $A$ but is not triangular, so it fails
`upper_violation`; a factor with a flipped sign on some column fails `min_diag` and
`factor_max_abs_err`.
