## Context

Knowledge distillation between models that share the exact same vocabulary (same index
$k$ always means the same token in both teacher and student) is usually done with
plain KL divergence, computed index-by-index:

$$
\mathrm{KL}(p_t \,\|\, p_s) = \sum_k p_{t,k} \log \frac{p_{t,k}}{p_{s,k}} .
$$

ULD (Universal Logit Distillation) was designed for the *harder* case where teacher and
student have **different** vocabularies, so index $k$ doesn't mean the same token in
both. It sidesteps the alignment problem by sorting each distribution's probabilities
independently and comparing the sorted sequences with a Wasserstein-1 (sorted-L1)
distance:

$$
\mathrm{ULD}(p_t, p_s) = \sum_k \left| p_t^{\downarrow}[k] - p_s^{\downarrow}[k] \right|,
$$

where $p^{\downarrow}$ denotes $p$'s entries sorted (ascending or descending — it does
not matter, since sorting both the same way cancels out). Even on a **shared**
vocabulary, where using index-aligned KL is unnecessary, ULD is still well defined; the
question is whether it behaves like a sensible loss there too. Both quantities are
genuine (pseudo-)divergences: KL divergence is non-negative by Gibbs' inequality, and
sorted-L1 is a true distance (an $\ell_1$ norm of a difference), so both satisfy

$$
\mathrm{KL}(p_t \,\|\, p_s) \ge 0, \qquad \mathrm{ULD}(p_t, p_s) \ge 0,
$$

with equality in both **exactly** when $p_s = p_t$.

## Task

Implement `uld_and_kl_along_sweep`:

```python
def uld_and_kl_along_sweep(
    p_teacher: np.ndarray, p_students: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

- `p_teacher`: `float64` array of shape `(V,)`, a probability distribution (positive,
  sums to 1).
- `p_students`: `float64` array of shape `(n, V)`, `n` probability distributions
  (each positive, sums to 1) forming a perturbation sweep away from `p_teacher`; one row
  is exactly equal to `p_teacher`.

For every row `p_students[i]`, compute `uld[i] = ULD(p_teacher, p_students[i])` and
`kl[i] = KL(p_teacher || p_students[i])` using the formulas above. Return
`(uld_values, kl_values)`, each a `float64` array of shape `(n,)`.

## Example

```python
import numpy as np

p_t = np.array([0.5, 0.3, 0.2])
p_students = np.array([
    [0.5, 0.3, 0.2],   # exact match -> both losses are 0
    [0.2, 0.3, 0.5],   # same multiset, different order -> ULD is 0, KL is NOT
])
uld, kl = uld_and_kl_along_sweep(p_t, p_students)
# uld ~= [0.0, 0.0]      (sorted distributions are identical in both rows)
# kl  ~= [0.0, >0.0]     (KL is index-aligned, so the reordering IS a difference)
```

## What the gate checks

The gate builds a NumPy oracle computing both losses directly, on a fixed teacher
distribution and a fixed perturbation sweep (row 0 is the exact `p_teacher` match; the
other rows are genuine perturbations along a fixed, mean-zero direction). It checks:

- `rel_err`: max relative error of your `uld_values`/`kl_values` versus the oracle's
  across the whole sweep, at most $10^{-6}$.
- `nonneg_ok`: every value in both arrays must be `>= 0` (must be `1.0`).
- `minimized_at_match`: both `uld_values` and `kl_values` must attain their minimum at
  row `0` — the exact `p_teacher == p_student` case (must be `1.0`).
