## Context

Tiled ("flash-style") attention processes keys/values in blocks instead of
materializing the full score matrix, using the **online-softmax**
recurrence. For query rows $Q$ against a key/value block $K_j, V_j$:

$$
S_j = \frac{Q K_j^\top}{\sqrt d}, \qquad
m_{\text{new}} = \max(m,\ \mathrm{rowmax}(S_j)),
$$
$$
P_j = e^{\,S_j - m_{\text{new}}}, \qquad
\ell \leftarrow \ell\, e^{\,m - m_{\text{new}}} + \mathrm{rowsum}(P_j), \qquad
O \leftarrow O\, e^{\,m - m_{\text{new}}} + P_j V_j, \qquad
m \leftarrow m_{\text{new}}.
$$

After the last block, $O \leftarrow O / \ell$. Two properties make this
correct and safe:

1. **Block-size invariance.** The rescale-by-$e^{m_{\text{old}}-m_{\text{new}}}$
   step means the running $(O, \ell, m)$ always represent the *exact*
   softmax statistics over every key seen so far — regardless of how keys
   are grouped into blocks. Whether you process one key at a time
   (`block_size=1`) or the whole sequence in a single block
   (`block_size=N`), the final output is mathematically identical (up to
   floating-point rounding) to the dense computation.
2. **Overflow safety.** Subtracting the running max *before* exponentiating
   keeps every exponent argument $\le 0$, so $e^{S_j-m_{\text{new}}}\in(0,1]$
   always. Skip that subtraction (the "naive" softmax
   $e^{S}/\sum e^{S}$) and a large enough score can send $e^{S}$ to
   `inf` in floating point — `float64` overflows past $e^{x}$ for
   $x \gtrsim 709.78$ — corrupting the result with `inf`/`nan`, even
   though the *true* softmax value would have been a well-defined number
   in $(0,1]$.

## Task

Implement two functions.

```python
def tiled_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    ...
```

- `Q`, `K`, `V`: 2-D arrays of shape `(N, d)`.
- `block_size`: tile size for the key/value sweep (does **not** need to
  evenly divide `N` — handle a smaller, ragged last block).
- Sweep `K`/`V` in blocks using the online-softmax recurrence above and
  return the `(N, d)` attention output. The result must be the same
  (within floating-point rounding) no matter what `block_size` is passed —
  `1`, some value that doesn't evenly divide `N`, or `N` itself.

```python
def softmax_stability_probe(scores: np.ndarray):
    ...
```

- `scores`: a 2-D array of raw (possibly very large-magnitude) attention
  scores.
- Compute `stable_out`: the row-wise softmax of `scores` using
  max-subtraction (must stay finite for any input).
- Compute `unstable_overflowed`: `True` if the *naive* softmax — `exp(scores)`
  normalized by its row sum, **without** subtracting the row max first —
  produces any non-finite value on this input; `False` if the naive path
  would also have stayed finite.
- Return `(stable_out, unstable_overflowed)`.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
Q = rng.standard_normal((64, 8))
K = rng.standard_normal((64, 8))
V = rng.standard_normal((64, 8))

out1 = tiled_attention(Q, K, V, block_size=1)
out64 = tiled_attention(Q, K, V, block_size=64)
np.allclose(out1, out64, atol=1e-4)   # True -- block size doesn't matter

scores = np.array([[1.0, 2.0, 3.0]])
stable_out, overflowed = softmax_stability_probe(scores)
print(overflowed)  # False -- ordinary scale, both paths would be fine

scores_big = np.array([[1000.0, 1.0, 2.0]])
stable_out, overflowed = softmax_stability_probe(scores_big)
print(stable_out)   # [[1., 0., 0.]] -- still exactly correct
print(overflowed)   # True -- exp(1000) overflows float64 unshifted
```

## What the gate checks

**Tiling.** Two `(Q, K, V)` cases from a seeded generator — one where
`block_size` values `{1, 8, 32, N}` all evenly divide `N`, one where they
don't (forcing a ragged last block) — are run through `tiled_attention`
for every listed block size and compared against a dense fp64
`softmax(QK^\top/\sqrt d)V` oracle (`max_abs_err`, threshold `1e-4`). Every
block size must agree with the dense oracle to that tolerance; getting the
rescale step wrong typically makes larger block counts (small
`block_size`) diverge from the `block_size=N` case even when both happen
to look locally plausible.

**Stability.** Two `scores` cases — one ordinary-scale (where the naive
path also stays finite) and one with a forced huge entry (where the naive
path provably overflows `float64`) — check `stable_out` against a real
max-subtracted softmax oracle (folded into the same `max_abs_err` gate)
and `unstable_overflowed` against the oracle's own naive-path finiteness
check (`exact_match`, must be `1.0` on both cases — so a solution that
always reports `True`, or always `False`, fails on whichever case
contradicts it).
