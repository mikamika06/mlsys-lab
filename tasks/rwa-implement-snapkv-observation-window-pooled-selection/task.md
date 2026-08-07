## Context

SnapKV compresses a KV cache using only the attention pattern of the last
few queries — the **observation window** — under the assumption that
which prefix tokens matter is largely stable near generation time. Given
$H$ attention heads, cached keys/values $K, V \in \mathbb{R}^{H\times n\times d}$,
and the observation-window queries $Q_{\text{obs}} \in \mathbb{R}^{H\times w\times d}$
(the last $w$ queries issued while this context was cached), each head
independently:

1. Aggregates attention mass the window put on every position:
   $$
   \text{raw}_i = \sum_{t=1}^{w} \operatorname{softmax}\!\Big(\tfrac{Q_{\text{obs}}K^\top}{\sqrt d}\Big)_{t,i}
   $$
2. **Pools** the raw score along the token axis with an odd-width
   average-pooling kernel (`pool_size`, `mode="edge"` padding), so
   contiguous clusters of important tokens are favoured over isolated
   spikes — a single high-attention token surrounded by low-attention
   neighbors scores lower after pooling than a token embedded in a
   generally-important region.
3. Always keeps the window itself (the last $w$ positions), then fills
   the rest of the `budget` with the highest **pooled**-score positions
   outside the window.

Because scoring runs per head, **different heads may keep different
positions** — this is a real property of SnapKV, not an approximation.

## Task

Implement `snapkv_pooled_selection`:

```python
def snapkv_pooled_selection(K: list[list[list[float]]], V: list[list[list[float]]], Q_obs: list[list[list[float]]], Q_new: list[list[float]], budget: int, pool_size: int) -> dict:
    ...
```

- `K`, `V`: `(H, n, d)`.
- `Q_obs`: `(H, w, d)`, the observation-window queries per head
  (`w = Q_obs.shape[1] <= budget`).
- `Q_new`: `(H, d)`, a new query per head, attended *after* compression.
- `budget`: kept positions per head.
- `pool_size`: odd int, the average-pooling kernel width.

For each head, compute the kept index set as described above (ties in
the top-`k_extra` selection broken by smaller index, via a stable
descending sort), then attend `Q_new[h]` against the compressed
`K[h]`/`V[h]` (only the kept positions) with standard scaled dot-product
attention.

Return a dict:
- `"kept_idx"`: a list of `H` sorted int arrays/lists, one per head.
- `"output"`: `(H, d)`, the compressed-cache attention output per head.

## Example

```python

rng = random.Random(0)
H, n, d, w = 2, 30, 8, 4
K = rng.standard_normal((H, n, d))
V = rng.standard_normal((H, n, d))
Q_obs = rng.standard_normal((H, w, d))
Q_new = rng.standard_normal((H, d))

out = snapkv_pooled_selection(K, V, Q_obs, Q_new, budget=12, pool_size=3)
# out["kept_idx"][h] has length 12 for each head h (may differ across heads)
# out["output"].shape == (2, 8)
```

## What the gate checks

The grader builds several seeded multi-head scenarios and computes each
head's kept index set and compressed-cache output with an independent
Python oracle — never calling your function, never hardcoding an expected
value.

- `exact_match` — the fraction of (scenario, head) pairs whose kept index
  set exactly equals the oracle's. Must be `1.0`. Pooling in the wrong
  direction, scoring with only the last observation-window query instead
  of summing over all `w`, or forgetting to always retain the window
  itself all change the selected set.
- `max_abs_err` — the worst max-abs-error, across all scenarios and
  heads, between your `"output"` and the oracle's compressed-cache
  attention output. Must be `<= 1e-6`.
