## Context

In autoregressive decoding, a transformer layer keeps a **KV-cache**: the key and
value vectors of every past token, so the next token can attend to them without
recomputing. The cache grows by one row per step, so its memory is $O(n)$ in the
sequence length $n$.

**Sliding-window attention** bounds this. Each query is only allowed to attend to
the most recent $W$ tokens, so we never need to store more than $W$ key/value
rows. The natural data structure is a **ring buffer** (circular buffer) of fixed
capacity $W$: token $t$ is written to physical slot

$$\mathrm{slot}(t) = t \bmod W,$$

which **overwrites (evicts) the oldest token** once the buffer is full. Memory is
now $O(W)$, independent of $n$.

At decode step $t$ the query attends over the tokens currently in the buffer,
i.e. indices $\max(0,\,t-W+1),\dots,t$ (there are $\min(t+1,\,W)$ of them). With
head dimension $d$, keys $K$, values $V$, and query $q_t = Q_t$, the windowed
output is scaled dot-product attention over that set:

$$o_t = \operatorname{softmax}\!\left(\frac{K_{\mathrm{win}(t)}\, q_t}{\sqrt{d}}\right)^{\!\top} V_{\mathrm{win}(t)} .$$

Because softmax is invariant to the order of the keys, the **physical** (scrambled)
order in the ring buffer does not change $o_t$ — as long as each key stays paired
with its own value.

## Task

Implement `windowed_ring_attention(Q, K, V, W)`:

```python
def windowed_ring_attention(Q, K, V, W):
    ...
    return out, Kbuf, Vbuf
```

- `Q`, `K` are `(n, d)` NumPy arrays, `V` is `(n, dv)`; `W` is the window size (int, `1 <= W <= n`).
- Stream the tokens `0..n-1`. Maintain a capacity-`W` ring buffer, writing token
  `t` into physical slot `t % W` and overwriting whatever was there.
- Return:
  - `out`: `(n, dv)` — the per-step sliding-window attention outputs $o_t$.
  - `Kbuf`: `(W, d)` — the **final** physical contents of the key ring buffer.
  - `Vbuf`: `(W, dv)` — the **final** physical contents of the value ring buffer.

Use $\sqrt{d}$ scaling and a numerically stable softmax (subtract the row max).

## Example

```python
import numpy as np
n, d, dv, W = 5, 2, 2, 3
Q = np.arange(n * d, dtype=float).reshape(n, d)
K = np.arange(n * d, dtype=float).reshape(n, d)
V = np.arange(n * dv, dtype=float).reshape(n, dv)

out, Kbuf, Vbuf = windowed_ring_attention(Q, K, V, W)
# out.shape == (5, 2)
# Step t=4 attends only to tokens {2, 3, 4} (the last W=3), not {0,1}.
# After streaming all 5 tokens the ring buffer (capacity 3) holds, by slot:
#   slot 0 -> token 3   (3 % 3 == 0, most recent writer of slot 0)
#   slot 1 -> token 4   (4 % 3 == 1)
#   slot 2 -> token 2   (2 % 3 == 2)
# so Kbuf == K[[3, 4, 2]] and Vbuf == V[[3, 4, 2]].
```

## What the gate checks

Two gates, evaluated on several random streams (all with $n > W$, so the buffer
fully wraps):

- `max_abs_err` $\le 10^{-5}$: the largest absolute difference between your `out`
  and a straightforward windowed-attention reference computed in chronological
  order. A full (non-evicting) cache attends to every past token and fails; an
  off-by-one window boundary fails.
- `buffer_max_abs_err` $\le 10^{-5}$: the largest absolute difference between your
  final `Kbuf`/`Vbuf` and the reference ring layout (token `t` at slot `t % W`,
  last writer wins). This forces genuine circular indexing, not just a sliced list.

Both references are computed with NumPy inside the grader — nothing is hardcoded.
