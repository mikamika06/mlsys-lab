## Context

QuaRot, SpinQuant, and related rotation-based quantization methods rely
on one algebraic fact: inserting an **orthogonal** rotation $Q$ into a
linear layer, in exactly the right place, does not change what the
layer computes. Because $Q^\top Q = I$, you can rotate the weight matrix
by $Q$ and rotate the incoming activation by $Q^\top$, and the two
rotations cancel out exactly:
$$
(W Q)(Q^\top x) = W (Q Q^\top) x = W x.
$$
This is what makes it safe to insert a fixed random Hadamard rotation
before quantizing — it changes the *numbers* (spreading outliers evenly
across channels, which is the whole point) without changing the
network's function, as long as the rotation is undone (or equivalently
baked into the next layer) consistently.

## Task

Implement:

```python
def rotate_and_matvec(W: np.ndarray, x: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

* `W` — `(d_out, d_in)` weight matrix.
* `x` — length-`d_in` input vector.
* `Q` — `(d_in, d_in)` orthogonal matrix ($Q^\top Q = I$).

Return `(W_rot, x_rot, y)`:

* `W_rot` — the rotated weights, $W Q$.
* `x_rot` — the rotated input, $Q^\top x$.
* `y` — the layer output computed **from the rotated pair**,
  $W_{rot}\, x_{rot}$ (which must numerically equal $Wx$).

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
W = rng.normal(size=(4, 6))
x = rng.normal(size=6)
A = rng.normal(size=(6, 6)); S = A + A.T
_, Q = np.linalg.eigh(S)   # eigenvectors of a symmetric matrix are orthonormal
W_rot, x_rot, y = rotate_and_matvec(W, x, Q)
assert np.allclose(y, W @ x)
```

## What the gate checks

* **max_abs_err** — the maximum absolute difference, across `W_rot`,
  `x_rot`, and `y`, between your outputs and a NumPy oracle computing
  `W @ Q`, `Q.T @ x`, and the original `W @ x` respectively, must be
  $\le 10^{-8}$, over several random `(W, x, Q)` cases (fixed seed, `Q`
  built from the eigenvectors of a random symmetric matrix).
