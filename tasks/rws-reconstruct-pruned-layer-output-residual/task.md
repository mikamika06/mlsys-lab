## Context

**Wanda** ("Weights AND Activations") prunes each weight by how much it
actually matters to the layer's output, not just its own magnitude:
importance$_{ij} = |W_{ij}|\cdot\lVert X_{j,:}\rVert_2$ (weight magnitude
times the activation norm of the input feature it multiplies), pruned
per output row. Given a mask already computed this way, applying it and
measuring the damage is the first thing you'd do before deciding whether
that sparsity level is acceptable.

## Task

Implement `apply_wanda_mask`:

```python
def apply_wanda_mask(W: list[list[float]], M: list[list[float]], X: list[list[float]]):
    ...
```

* `W` — `(d_out, d_in)` weight matrix.
* `M` — `(d_out, d_in)` binary mask (`1` = kept, `0` = pruned).
* `X` — `(d_in, n)` calibration activations (input features as rows,
  samples as columns).

Return `(Y, R)`:

$$
Y = (W \odot M)\,X, \qquad R = WX - Y
$$

* `Y` — `(d_out, n)`, the pruned layer's output.
* `R` — `(d_out, n)`, the output residual introduced by pruning (the
  unpruned output minus the pruned output).

## Example

```python
W = [[1.0, 2.0], [3.0, 4.0]]
M = [[1.0, 0.0], [0.0, 1.0]]
X = [[1.0, 0.0], [0.0, 1.0]]
Y, R = apply_wanda_mask(W, M, X)
# Y = [[1,0],[0,4]], R = (W@X) - Y = [[0,2],[3,0]]
```

## What the gate checks

* **y_max_abs_err** — max-abs difference between your `Y` and a Python
  oracle computing `(W*M) @ X` on the fixed fixtures (`wanda_w.npy`,
  `wanda_m.npy`, `wanda_x.npy`).
* **r_max_abs_err** — max-abs difference between your `R` and the
  oracle's `W@X - Y`.
