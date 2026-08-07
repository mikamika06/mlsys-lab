## Context

Optimal Brain Surgeon (OBS) pruning removes a parameter while applying a second-order correction to reduce the reconstruction error. Given a weight matrix $W \in \mathbb{R}^{m \times n}$ and an inverse Hessian approximation $H^{-1} \in \mathbb{R}^{n \times n}$, pruning input column $q$ means setting that column to zero and compensating other columns.

The OBS update for column $j$ is

$$
W_{:,j} \leftarrow W_{:,j} - W_{:,q}\frac{[H^{-1}]_{qj}}{[H^{-1}]_{qq}},
$$

for all $j \neq q$, followed by

$$
W_{:,q} \leftarrow 0.
$$

The divisor $[H^{-1}]_{qq}$ is required because the Hessian curvature determines how much compensation is needed. Omitting it changes the magnitude of the correction, and changing the sign moves the update in the wrong direction.

## Task

Implement `obs_update(W, Hinv, q)`:

```python
def obs_update(W: list[list[float]], Hinv: list[list[float]], q: int) -> list[list[float]]:
    ...
```

The function receives a 2-D weight matrix `W`, a square inverse Hessian matrix `Hinv`, and the column index `q` to remove. Return a new list containing the OBS-corrected weights.

The input arrays must not be modified in-place. Use Python operations to compute the corrected matrix.

## Example

```python

W = [[2.0, 4.0], [1.0, 3.0]]
Hinv = [[2.0, 0.5], [0.5, 1.0]]

out = obs_update(W, Hinv, 1)

# column 1 is pruned and column 0 is corrected:
# [[0.0?]] is not the output shape example; the returned matrix has the
# same shape as W with the selected column set to zero.
```

## What the gate checks

The gate computes a Python oracle implementation of the OBS update and compares the submitted function against it. The reconstruction relative error

$$
\mathrm{rel\_err} =
\frac{\lVert W_{\mathrm{candidate}}-W_{\mathrm{oracle}}\rVert_2}
{\lVert W_{\mathrm{oracle}}\rVert_2 + 10^{-12}}
$$

must be below $10^{-5}$.

Implementations that omit the $[H^{-1}]_{qq}$ divisor or use the opposite update sign will produce a different reconstruction and fail the gate.
