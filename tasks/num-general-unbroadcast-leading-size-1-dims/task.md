## Context

When a tensor of shape $s$ is used in a NumPy broadcasting expression, it can
end up implicitly stretched to some larger shape $f$ (the "full" shape of the
expression's output). NumPy's broadcasting rule aligns the two shapes on the
**right** and does two things to go from $s$ to $f$:

1. It inserts brand-new **leading** axes on the left whenever $f$ has more
   dimensions than $s$ — $\mathrm{len}(f) - \mathrm{len}(s)$ of them.
2. For every axis that both shapes share (after right-alignment), if
   $s_i = 1$ but $f_i > 1$, that axis is **stretched**: the single value is
   conceptually repeated $f_i$ times.

During backpropagation you receive a gradient $g$ shaped like $f$ (the
*output* of the broadcast) and must produce a gradient shaped like $s$ (the
*input*) by summing away exactly the axes that were inserted or stretched —
this is the "unbroadcast" step.

## Task

Implement `unbroadcast`:

```python
def unbroadcast(grad, shape):
    """Reduce `grad` (which has the shape produced by broadcasting a tensor
    of shape `shape` up to grad.shape) back down to `shape`, by summing over
    every axis that broadcasting introduced or stretched. Returns a float64
    array with exactly `len(shape)` dimensions, equal to `shape`."""
```

* `grad` — a NumPy array with the "full", already-broadcasted shape.
* `shape` — a tuple (possibly empty, for a scalar) giving the original,
  pre-broadcast shape. `grad.shape` broadcasts from `shape` by ordinary
  NumPy rules.

Handle **both** kinds of axes uniformly and correctly, including cases with
several extra leading dimensions *and* several interior/trailing size-1
dimensions in the same call. The result must have exactly `len(shape)`
dimensions and shape equal to `shape` — not merely the same total size.

## Example

```python
import numpy as np
grad = np.arange(24.0).reshape(2, 3, 4)   # broadcasted shape (2, 3, 4)
out = unbroadcast(grad, (1, 4))           # original shape had a leading
                                           # broadcast dim AND a size-1 dim
# out.shape == (1, 4)
# out[0, j] == sum over i in 0..1, k in 0..2 of grad[i, k, j]
```

## What the gate checks

The gate computes the maximum absolute error

$$
\mathrm{max\_abs\_err} = \max \left| \text{unbroadcast}(g, s) - \text{oracle}(g, s) \right|
$$

against an independent NumPy reference, over eleven `(full_shape, shape)`
pairs covering: no broadcast at all, a single extra leading dimension,
multiple extra leading dimensions, interior and trailing size-1 dimensions,
combinations of leading dims with interior size-1 dims, and reduction all
the way down to a scalar. The result shape is also checked exactly. The
maximum error over all cases must satisfy $\mathrm{max\_abs\_err} \le 10^{-9}$.
