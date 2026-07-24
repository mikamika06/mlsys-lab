## Context

Production compression pipelines stack multiple lossy transforms —
prune, then quantize. It's tempting to assume the compound error is
*roughly* the sum of the individual errors, but is that ever
**guaranteed**, not just empirically observed? Yes: if you define the
"quantization error" as the error quantization introduces **on top of
the already-pruned weights** (not on the original weights), the bound is
an exact consequence of the triangle inequality — no approximation, no
assumptions about the data.

### Setup

Given weights $W\in\mathbb{R}^{d_{out}\times d_{in}}$ and calibration
activations $X\in\mathbb{R}^{n\times d_{in}}$, define the linear layer
output $Y(W') = XW'^\top$.

1. **Prune**: zero the lowest-magnitude `sparsity` fraction of $W$'s
   entries, globally (rank all $|W_{ij}|$ ascending with a stable sort,
   zero the first $\lfloor \text{sparsity}\cdot d_{out}d_{in}\rceil$ of
   them) $\to W_p$.
2. **Quantize** the *pruned* weights: per-row symmetric round-to-nearest
   at `nbits` bits (scale $=\max_j|W_{p,i,j}|/(2^{b-1}-1)$ per row $i$,
   or $1$ if that row is all zero) $\to W_{pq}$.
3. **Relative output errors**, all normalized by the same
   $\lVert Y(W)\rVert_F$:
   $$
   e_{\text{prune}} = \frac{\lVert Y(W_p) - Y(W)\rVert_F}{\lVert Y(W)\rVert_F}, \qquad
   e_{\text{quant}} = \frac{\lVert Y(W_{pq}) - Y(W_p)\rVert_F}{\lVert Y(W)\rVert_F}, \qquad
   e_{\text{compound}} = \frac{\lVert Y(W_{pq}) - Y(W)\rVert_F}{\lVert Y(W)\rVert_F}
   $$

### Why the bound is exact

$Y$ is linear, so
$Y(W_{pq}) - Y(W) = \big(Y(W_{pq}) - Y(W_p)\big) + \big(Y(W_p) - Y(W)\big)$
exactly. Taking norms and applying the triangle inequality (then dividing
every term by the same positive constant $\lVert Y(W)\rVert_F$):

$$
e_{\text{compound}} \;\le\; e_{\text{prune}} + e_{\text{quant}}
$$

This holds for **any** $W$, $X$, sparsity, and bit width — it is not a
heuristic.

## Task

Implement `compound_error_bound`:

```python
def compound_error_bound(W: np.ndarray, X: np.ndarray, sparsity: float, nbits: int) -> tuple[float, float, float]:
    ...
```

* `W` — `(d_out, d_in)` weight matrix.
* `X` — `(n, d_in)` calibration activation matrix.
* `sparsity` — fraction of entries to prune, in `[0, 1)`.
* `nbits` — bit width for the post-prune quantization.

Return `(e_prune, e_quant, e_compound)`, computed exactly as defined
above.

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(30, 20))
W = rng.normal(size=(6, 20))
e_prune, e_quant, e_compound = compound_error_bound(W, X, sparsity=0.3, nbits=4)
assert e_compound <= e_prune + e_quant + 1e-9
```

## What the gate checks

* **errs_rel_err** — relative error between each of your three returned
  values and a NumPy oracle running the exact recipe above, over several
  random `(W, X, sparsity, nbits)` trials (worst of the three, worst of
  all trials).
* **bound_holds** — your own `e_compound` must never exceed your own
  `e_prune + e_quant` (up to floating-point slack), on every trial.
