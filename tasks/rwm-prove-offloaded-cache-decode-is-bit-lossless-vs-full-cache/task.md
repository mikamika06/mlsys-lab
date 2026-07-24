## Context

Long-context decoding keeps a growing key/value (KV) cache for every past
token. When the cache is too large for GPU memory, production inference
engines *offload* it: past K/V pairs are moved to CPU host memory, and at
each decode step the needed range is gathered back before the attention
matmul runs. Offloading only changes **where** a value lives, never the
value itself, so the attention output must be bit-identical (up to
floating-point rounding) to the case where the entire cache always stayed
resident on the device.

For decode step $t$ of layer $l$, head $h$, with head dimension $d$, causal
single-query attention over everything cached so far ($t' \le t$) is

$$
\alpha_{t,t'} = \frac{q_t \cdot k_{t'}}{\sqrt d}, \qquad
w_{t,t'} = \operatorname{softmax}_{t' \le t}(\alpha_{t,t'}), \qquad
o_t = \sum_{t' \le t} w_{t,t'}\, v_{t'} .
$$

An offloaded implementation must reproduce $o_t$ exactly by gathering the
**entire** history $k_0, \dots, k_t$ and $v_0, \dots, v_t$ back from the
offload store at every step — not just the pair that was just pushed to it.

## Task

Implement `offloaded_decode_attention` in `solve.py`:

```python
def offloaded_decode_attention(Q, K_new, V_new):
    ...
```

* `Q`, `K_new`, `V_new` — `float64` arrays of shape `(L, T, H, d)`: `L`
  layers, `T` decode steps, `H` heads, `d` head dim. `K_new[l, t]` /
  `V_new[l, t]` is the new key/value pair produced at decode step `t` for
  layer `l` (independent per layer — layers do not share a cache).

Simulate each layer's KV cache as a CPU-resident **offload store**: at step
`t`, push `K_new[l, t]`, `V_new[l, t]` into it, then gather back the full
history `0..t` (every pair pushed so far, for that layer) and run causal
single-query attention with scale $1/\sqrt d$ to produce `out[l, t]`.

Return `out`, a `float64` array of shape `(L, T, H, d)`.

## Example

```python
import numpy as np

L, T, H, d = 2, 4, 1, 8
rng = np.random.default_rng(0)
Q = rng.standard_normal((L, T, H, d))
K = rng.standard_normal((L, T, H, d))
V = rng.standard_normal((L, T, H, d))

out = offloaded_decode_attention(Q, K, V)
# out[l, t] depends on K[l, 0:t+1] and V[l, 0:t+1] -- the WHOLE offloaded
# history up to and including step t, for every layer independently.
```

## What the gate checks

The grader builds a fully vectorised "everything stays on device" oracle:
for each layer and head it forms the whole `(T, T)` score matrix, applies a
causal mask, and computes softmax-weighted values directly with NumPy — no
incremental gather involved, but mathematically the same quantity a
correctly offloaded decode loop must produce at every step. It then calls
your `offloaded_decode_attention` on the same `Q, K_new, V_new` tensors
across several `(L, T, H, d)` configurations (including a single-step,
single-layer edge case) and takes `max_abs_err` over every layer, step,
head, and dimension. The threshold is a tight `1e-6`: offloading must be
lossless, not merely close. An implementation that only gathers the most
recently pushed pair (instead of the full accumulated history) matches only
at `t = 0` and diverges from step 1 onward.
