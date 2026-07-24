## Context

A KV cache holds $n$ cached keys/values $K, V \in \mathbb{R}^{n\times d}$. Under
a fixed token **budget** $m < n$, three production eviction policies decide
*which* $m$ tokens to keep:

- **Knorm** — key-norm pruning. Keys with a small $\lVert k_i\rVert_2$ tend
  to receive disproportionately high attention weight, so Knorm keeps the
  $m$ tokens with the **smallest** key norm:
  $$
  \text{score}_i = -\lVert k_i \rVert_2 .
  $$

- **H2O** (Heavy-Hitter Oracle) — always keeps a trailing **recent window**
  of `recent_window` tokens, plus the remaining budget filled with the
  tokens that accumulated the most attention mass across every query the
  model has already issued while this context was cached,
  $Q_{\text{hist}} \in \mathbb{R}^{T\times d}$:
  $$
  \text{score}_i = \sum_{t=1}^{T} \operatorname{softmax}\!\Big(\tfrac{Q_{\text{hist}} K^\top}{\sqrt d}\Big)_{t,i}.
  $$

- **SnapKV** — always keeps the trailing **observation window** of the last
  `snap_window` rows of $Q_{\text{hist}}$ (i.e. their corresponding $K/V$
  positions), plus the remaining budget filled with tokens scored using
  attention mass from *only* that window, smoothed with an odd-length
  average-pooling kernel (`pool_size`) over the token axis so that
  contiguous clusters of important tokens are favoured over isolated
  spikes.

## Task

Implement `compare_eviction_methods`:

```python
def compare_eviction_methods(
    K: np.ndarray, V: np.ndarray, Q_hist: np.ndarray, q_new: np.ndarray,
    budget: int, recent_window: int, snap_window: int, pool_size: int,
) -> dict:
    ...
```

- `K`, `V`: `(n, d)` cached keys/values.
- `Q_hist`: `(T, d)` queries already issued against the full context
  (used to score tokens for H2O / SnapKV). `snap_window <= T`.
- `q_new`: `(d,)` a fresh query attended *after* eviction, i.e. against
  only the `budget` kept tokens.
- `budget`, `recent_window`, `snap_window`: ints, with
  `recent_window <= budget` and `snap_window <= budget`.
- `pool_size`: an odd int, the SnapKV average-pooling kernel width. Pool
  with `mode="edge"` padding so the pooled score sequence has length $n$.

Compute each policy's kept index set (sorted, unique, ties broken by
lowest index first — i.e. a stable sort on score), then attend `q_new`
against the corresponding restricted `K`/`V` with standard scaled
dot-product attention, $\operatorname{softmax}(q K^\top/\sqrt d)\,V$.
Compare that to attending against the **full** (unevicted) $K, V$.

Return a dict with:

- `"knorm_error"`, `"h2o_error"`, `"snapkv_error"` — the max-abs-error
  between that policy's compressed-KV output and the full-attention
  output for `q_new`.
- `"overlap_knorm_h2o"`, `"overlap_knorm_snapkv"`, `"overlap_h2o_snapkv"`
  — the size of the intersection between the two policies' kept index
  sets (an `int`).

If `budget - recent_window <= 0` (or `budget - snap_window <= 0` for
SnapKV), that policy simply keeps the last `budget` tokens of its
always-kept window.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
n, d, T = 32, 8, 10
K = rng.standard_normal((n, d))
V = rng.standard_normal((n, d))
Q_hist = rng.standard_normal((T, d))
q_new = rng.standard_normal(d)

out = compare_eviction_methods(K, V, Q_hist, q_new,
                                budget=12, recent_window=4,
                                snap_window=4, pool_size=3)
# out.keys() == {"knorm_error", "h2o_error", "snapkv_error",
#                "overlap_knorm_h2o", "overlap_knorm_snapkv",
#                "overlap_h2o_snapkv"}
```

## What the gate checks

The grader builds several random `(K, V, Q_hist, q_new, budget,
recent_window, snap_window)` scenarios (seeded, `pool_size=3`) and computes
each policy's kept set and resulting attention error with an independent
NumPy oracle — never calling your function, never hardcoding an expected
number.

- `max_abs_err` — the largest absolute difference, across all scenarios
  and all three `*_error` fields, between your reported error and the
  oracle's. Must be `<= 1e-6`. Selecting the wrong tokens for any policy
  (e.g. keeping the *largest*-norm keys for Knorm, or scoring H2O with
  only the last query instead of the cumulative sum over `Q_hist`)
  changes that policy's attention output and therefore its error value.
- `overlap_exact_match` — the fraction of the three `overlap_*` counts,
  across all scenarios, that exactly match the oracle's kept-set
  intersection sizes. Must be `1.0`.
