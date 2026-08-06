## Context

The Rotary Position Embedding (RoPE) encodes token positions via a rotation matrix applied to pairs of dimensions. For a pair $(x_{2i}, x_{2i+1})$ at position $m$, the rotation is

$$\begin{pmatrix} x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i) \\ x_{2i+1} \cos(m\theta_i) + x_{2i} \sin(m\theta_i) \end{pmatrix}, \qquad  \theta_i = 10000^{-2i/d},$$

where $d$ is the head dimension (even).

When applied to a whole sequence, we must compute $\cos(m\theta_i)$ and $\sin(m\theta_i)$ for every position $m$ in $\{0, 1, \dots, S-1\}$ and every pair index $i$, then broadcast into the full tensor $x$ of shape `(batch, seq_len, num_heads, d)`.

## Task

Implement `apply_rope(x, pos)`:

```python

def apply_rope(x: list[list[list[list[float]]]], pos: list[int]) -> list[list[list[list[float]]]]:
    pass
```

- `x`: shape `(batch, seq_len, num_heads, head_dim)` with `head_dim` even.
- `pos`: integer array shape `(seq_len,)` containing the 0‑based position of each token.
- Returns a list of the same shape as `x`, with RoPE applied.


**Conditions**

- Use only fully vectorized Python operations (no Python `for` / `while` loops).
- The function must be written in pure Python with Python.


## Example

```python
import random

batch, seq_len, heads, dim = 2, 3, 4, 4
x = [[[[random.random() for _ in range(dim)] for _ in range(heads)] for _ in range(seq_len)] for _ in range(batch)]
pos = [0, 1, 2]                           # integer positions
out = apply_rope(x, pos)                             # shape (2, 3, 4, 4)

# check equivalence on first token (position 0, no rotation)
for b in range(batch):
    for h in range(heads):
        for d in range(dim):
            assert abs(out[b][0][h][d] - x[b][0][h][d]) < 1e-12
```

## What the gate checks

Two gates:

1. **max_abs_err**: the maximum absolute difference between your output and a Python‑based reference, computed across several random inputs. Must be ≤ $10^{-6}$.
2. **line_count**: the number of Python line events recorded during the function call (via `sys.settrace`). A properly vectorized solution will execute fewer than 40 line events; a solution that uses Python loops will exceed this threshold.
