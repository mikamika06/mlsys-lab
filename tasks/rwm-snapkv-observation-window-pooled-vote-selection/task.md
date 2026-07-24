## Context

SnapKV compresses a long prompt's KV cache before generation starts, using
the observation that the *last few* query positions of the prompt (the
"observation window") already reveal which earlier tokens the model is
about to keep attending to. Instead of keeping the whole prefix in the KV
cache, SnapKV:

1. Looks at the attention weights $A \in \mathbb{R}^{H \times W \times L}$
   from the $W$ window queries (heads $H$, prefill length $L$) to every
   prefill key position, and **votes** by summing over heads and the window:
   $$
   s_i = \sum_{h=1}^{H} \sum_{w=1}^{W} A_{h,w,i}, \qquad i = 1, \dots, L .
   $$
2. **Smooths** the vote with a 1D average pool of odd kernel size $k$
   (stride 1, zero-padded so the output has the same length $L$, padding
   counted in the average — exactly `torch.nn.functional.avg_pool1d` with
   `padding=k//2`):
   $$
   \hat{s}_i = \frac{1}{k} \sum_{j=-\lfloor k/2 \rfloor}^{\lfloor k/2 \rfloor} s_{i+j}
   \qquad (s_{i+j} := 0 \text{ outside } [1, L]) .
   $$
   Pooling clusters together tokens near an important one (e.g. a whole
   sentence), which raw per-token attention would miss.
3. Keeps the `capacity` prefill positions with the highest pooled vote
   $\hat{s}_i$ (ties broken by lower index), plus the observation window
   itself (which is never evicted).

## Task

Implement `snapkv_select` in `solve.py`:

```python
def snapkv_select(attn, window_size, kernel_size, capacity):
    ...
```

* `attn` — `float64` array of shape `(H, window_size, L_prefix)`: attention
  weights from the `window_size` observation-window queries to the
  `L_prefix` prefill keys preceding the window.
* `window_size` — $W$, also the number of always-kept window positions.
* `kernel_size` — odd int, the pooling kernel from step 2 above.
* `capacity` — number of prefill positions to keep; if `capacity >
  L_prefix`, keep all of them (clip to `L_prefix`).

Return a tuple `(selected_indices, kept_total, compression_ratio)`:

* `selected_indices` — `int` array, **ascending sorted**, the indices (into
  the `L_prefix` axis, `0`-based) of the kept prefill positions. Length is
  `min(capacity, L_prefix)`.
* `kept_total` — `int`, `len(selected_indices) + window_size`.
* `compression_ratio` — `float`, `kept_total / (L_prefix + window_size)`.

## Example

```python
import numpy as np

H, W, L = 2, 4, 10
attn = np.full((H, W, L), 1.0 / L)   # uniform attention
attn[:, :, 3] += 0.5                  # position 3 is clearly the most attended

selected, kept_total, ratio = snapkv_select(attn, W, kernel_size=3, capacity=2)
# selected includes 3 (and its pooled neighbour 2 or 4, whichever pools higher)
```

## What the gate checks

The grader builds its own NumPy oracle implementing the same
sum-over-heads-and-window vote, average pool, and top-`capacity` selection
(tie-broken by lower index). It runs both on several random attention
tensors (rows normalized to sum to 1, like real softmax attention) covering
a plain case, a `kernel_size=1` case (no smoothing), a `capacity` that
exceeds `L_prefix` (must clip to keep everything), and a `capacity` that
keeps almost nothing. `exact_match` requires `selected_indices`,
`kept_total`, and `compression_ratio` (within `1e-9`) to agree with the
oracle on every case.
