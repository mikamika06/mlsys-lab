## Context

Full fine-tuning produces a weight delta $W = W_{\text{finetuned}} -
W_{\text{base}} \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$.
LoRA instead trains a rank-$r$ factorization $A B \approx W$ directly
($A \in \mathbb{R}^{d_{\text{out}}\times r}$,
$B \in \mathbb{R}^{r\times d_{\text{in}}}$), which is only a good
approximation when $W$'s energy is concentrated in a few singular
directions. The **Eckart-Young theorem** says the best possible rank-$r$
approximation of $W$ in Frobenius norm is given by its truncated SVD
$W = U\Sigma V^\top$:

$$
A = U_{:,:r}\,\sqrt{\Sigma_{:r}}, \qquad
B = \sqrt{\Sigma_{:r}}\,V^\top_{:r,:}, \qquad
\lVert AB - W\rVert_F = \sqrt{\textstyle\sum_{i > r}\sigma_i^2}
$$

and no other rank-$r$ pair can do strictly better. This is the
theoretical ceiling any LoRA training run is implicitly trying (and, at
best, only asymptotically able) to reach.

## Task

Implement `lora_rank_r_approx`:

```python
def lora_rank_r_approx(W: np.ndarray, A0: np.ndarray, B0: np.ndarray) -> dict:
    ...
```

- `W`: `(d_out, d_in)`, the full delta.
- `A0`, `B0`: `(d_out, r)`, `(r, d_in)` — some OTHER rank-`r` factor pair
  to compare against (`r = A0.shape[1]`).
- Compute the optimal rank-`r` factors `A`, `B` via truncated SVD as
  above, and the Frobenius **relative** reconstruction errors
  $\lVert AB - W\rVert_F / \lVert W\rVert_F$ for your optimal pair and
  $\lVert A_0 B_0 - W\rVert_F / \lVert W\rVert_F$ for the given one.

Return a dict:
- `"A"`, `"B"`: the optimal rank-`r` factors.
- `"rel_err_optimal"`: your factors' relative error.
- `"rel_err_given"`: `(A0, B0)`'s relative error.

## Example

```python
import numpy as np

W = np.random.default_rng(0).standard_normal((20, 16))
A0 = np.random.default_rng(1).standard_normal((20, 3))
B0 = np.random.default_rng(2).standard_normal((3, 16))

out = lora_rank_r_approx(W, A0, B0)
# out["rel_err_optimal"] <= out["rel_err_given"]  -- Eckart-Young
# guarantees the truncated-SVD factors can never be beaten by any other
# rank-3 pair, including a randomly chosen one.
```

## What the gate checks

The grader loads a committed `W.npy` (a delta with fast singular-value
decay, as real fine-tuning deltas typically have) plus `A0.npy`/`B0.npy`
(an arbitrary, non-optimal rank-`r` pair), plus several additional
seeded random `(W, A0, B0)` triples, and computes the true Eckart-Young
bound and the given pair's error independently in NumPy — never calling
your function, never hardcoding an expected value. It also independently
reconstructs `A @ B` from your *returned* factors and checks that
matches your reported `rel_err_optimal` too, so hardcoding the right
number without actually computing the SVD does not pass.

- `rel_err` — the worst-case `scorers.rel_err` between your reported
  values (`rel_err_optimal`, `rel_err_given`, and the error recomputed
  from your own `A`, `B`) and the oracle's, across all cases. Must be
  `<= 1e-6`.
- `not_below_bound` — the fraction of cases where your
  `rel_err_optimal` is not smaller than the true Eckart-Young bound
  (down to a `1e-9` numerical tolerance). Must be `1.0` — the theorem
  makes beating that bound with any rank-`r` pair mathematically
  impossible, so a value below it means a bug, not a better answer.
