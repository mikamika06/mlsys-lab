## Context

Rotary position embeddings (RoPE) rotate each key/query vector by an angle
that depends on its **absolute sequence position**. For a $d$-dimensional
vector $x$ (with $d$ even) at position $t$, splitting into 2D pairs
$(x_{2i}, x_{2i+1})$ for $i = 0, \dots, d/2 - 1$:

$$
\theta_i = \text{base}^{-2i/d}, \qquad
\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix}
=
\begin{bmatrix} \cos(t\theta_i) & -\sin(t\theta_i) \\ \sin(t\theta_i) & \cos(t\theta_i) \end{bmatrix}
\begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix} .
$$

In autoregressive decoding, each new token's key is rotated with RoPE **once**,
at insertion time, and the rotated key is what gets stored in the KV cache.
On every later decode step, that cached key must be reused *as-is* — RoPE must
never be applied to it again. A common bug re-rotates the entire cache (or
re-derives it from scratch with an updated position) on every step, which
compounds the rotation on already-rotated entries and silently corrupts
attention as the sequence grows.

## Task

`decode_step(cache, k_raw, v, q_raw, pos)` runs one autoregressive decode step
for a single attention head:

- `cache` is a dict `{"k": ndarray (t, d), "v": ndarray (t, d)}` holding the
  **already RoPE-applied** keys and raw values cached from previous steps
  (`t == 0` on the first call).
- `k_raw`, `q_raw` are raw (pre-RoPE), un-rotated key/query vectors, shape
  `(d,)`, for the new token being decoded.
- `v` is the new token's value vector, shape `(d,)`.
- `pos` is the new token's absolute position (`int`, `0`-based).

Fix `decode_step` (provided below, and in `starter.py`) so it applies RoPE to
`k_raw` and `q_raw` **exactly once**, using the correct helper `rope`
(already correct — do not change it), appends the rotated key and the value
to the cache, and returns the causal scaled dot-product attention output over
everything cached so far:

```python
import numpy as np

def rope(x, positions, base: float = 10000.0):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    positions = np.atleast_1d(np.asarray(positions, dtype=np.float64))
    d = x.shape[-1]
    i = np.arange(d // 2)
    theta = base ** (-2.0 * i / d)
    angles = positions[:, None] * theta[None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    x_even, x_odd = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out

def decode_step(cache, k_raw, v, q_raw, pos):
    ...  # buggy — see starter.py
```

Return `(output, cache)` where `output` is a `(d,)` `float64` array and
`cache` is the updated dict (may be mutated in place).

## Example

Calling `decode_step` three times in a row with a fresh `cache = {"k":
np.zeros((0, d)), "v": np.zeros((0, d))}` at `pos = 0, 1, 2` must produce, at
each step, the same attention output as computing full causal attention
directly over the raw keys/queries rotated once each at their own absolute
position — never a drifting, cumulatively-rotated result.

## What the gate checks

The gate drives `decode_step` sequentially over several random sequences
(varying length and head dimension), keeping a single persistent `cache`
across calls — exactly like real autoregressive decoding. After every step it
independently computes the oracle output for that prefix: rotate the raw keys
`K[:pos+1]` once each at their own position with `rope`, rotate the query
once at `pos`, and run causal scaled dot-product attention directly with
NumPy. It compares your step's output against this oracle with

$$
\mathrm{rel\_err} = \frac{\lVert y - \hat{y} \rVert_2}{\lVert \hat{y} \rVert_2 + 10^{-12}}
$$

and reports the worst value seen over all steps and sequences. The result
must satisfy $\mathrm{rel\_err} \le 10^{-5}$. A `decode_step` that re-applies
RoPE to already-cached keys passes the very first step (nothing cached yet to
double-rotate) but drifts further from the oracle at every subsequent step,
failing the gate.
