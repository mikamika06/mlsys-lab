## Context

A matrix $A \in \mathbb{R}^{n\times n}$ is **symmetric positive definite** (SPD)
when $A = A^{\mathsf T}$ and $x^{\mathsf T} A x > 0$ for every $x \neq 0$,
equivalently when every eigenvalue satisfies $\lambda_i > 0$.

The textbook way to *test* this is not to compute eigenvalues — it is to attempt a
Cholesky factorisation $A = L L^{\mathsf T}$ with $L$ lower triangular and
$L_{ii} > 0$. The factorisation is built column by column:

$$L_{jj} = \sqrt{A_{jj} - \sum_{k<j} L_{jk}^2},
\qquad
L_{ij} = \frac{A_{ij} - \sum_{k<j} L_{ik}L_{jk}}{L_{jj}} \quad (i > j).$$

Cholesky **is** the definiteness test: the quantity under the square root is the
$j$-th pivot, and $A$ is positive definite if and only if every pivot is strictly
positive. The moment a pivot is $\le 0$ the factorisation cannot continue — that
failure, not a `nan` from `sqrt`, is the verdict. It costs $\tfrac13 n^3$ flops
versus roughly $\tfrac{4}{3}\,n^3$ (and an iterative eigensolver) for
`eigvalsh`, which is why every sane covariance / kernel / preconditioner check
uses it.

Two failure modes are easy to miss:

* a matrix can have all-positive eigenvalues and still not be SPD because it is
  **not symmetric** (e.g. $\begin{psmallmatrix}2&1\\0&3\end{psmallmatrix}$ has
  eigenvalues $2,3$);
* a *negative pivot* must abort the factorisation. If you let `np.sqrt` of a
  negative number produce `nan` and keep going, the propagated `nan`s can be
  mistaken for a successful factorisation.

## Task

Implement two functions.

```python
def cholesky_spd(A: np.ndarray, sym_tol: float = 1e-10) -> np.ndarray | None: ...
def is_spd(A: np.ndarray, sym_tol: float = 1e-10) -> bool: ...
```

* `cholesky_spd` returns the lower-triangular Cholesky factor $L$ (with strictly
  positive diagonal, exact zeros above the diagonal, `float64`) such that
  $L L^{\mathsf T} = A$. It returns `None` — never a `nan`-filled array, never an
  exception — when `A` is non-square, when
  $\max_{ij}|A_{ij} - A_{ji}| > \texttt{sym\_tol}$, or when any pivot is $\le 0$.
* `is_spd` returns a Python `bool`: `True` exactly when `A` is symmetric within
  `sym_tol` and positive definite.

Write the factorisation yourself. `np.linalg.cholesky` raises `LinAlgError`
instead of reporting a verdict and does not check symmetry, so it is not a
drop-in answer.

## Example

```python
import numpy as np

A = np.array([[4.0, 1.0], [1.0, 3.0]])
L = cholesky_spd(A)
print(L)
# [[2.         0.        ]
#  [0.5        1.6583124 ]]
print(np.allclose(L @ L.T, A), is_spd(A))
# True True

B = np.array([[1.0, 2.0], [2.0, 1.0]])     # eigenvalues 3 and -1
print(cholesky_spd(B), is_spd(B))
# None False

C = np.array([[2.0, 1.0], [0.0, 3.0]])     # positive eigenvalues, NOT symmetric
print(cholesky_spd(C), is_spd(C))
# None False
```

## What the gate checks

The grader builds 14 deterministic matrices from `np.random.default_rng(0)` by
prescribing spectra through an orthogonal similarity $Q\,\mathrm{diag}(w)\,Q^{\mathsf T}$:
SPD ones (including a well-conditioned Gram matrix and a spectrum reaching down to
$10^{-3}$), symmetric non-SPD ones (indefinite, negative definite, all-zero, and one
whose smallest eigenvalue is $-10^{-6}$), and non-symmetric ones whose eigenvalues
are positive.

* `exact_match` — fraction of verdicts from your `is_spd` that agree with the
  **oracle**, which the grader computes independently from
  `np.linalg.eigvalsh` signs plus an explicit symmetry test. Must be `1.0`.
* `struct_ok` — must be `1.0`: `cholesky_spd` returns `None` for every non-SPD
  matrix, and for every SPD matrix returns an array that is exactly lower
  triangular with a strictly positive diagonal.
* `factor_max_abs_err` — for the SPD matrices, the largest absolute
  element-wise deviation of your $L$ from `np.linalg.cholesky(A)` (the Cholesky
  factor with positive diagonal is unique, so this comparison is well posed).
  Must be $\le 10^{-8}$.
