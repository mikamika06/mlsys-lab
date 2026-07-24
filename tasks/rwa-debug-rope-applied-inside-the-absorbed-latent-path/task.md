## Context

Multi-head latent attention implementations often compress the key/value path into a latent representation and absorb projection matrices into a smaller computation. The absorbed latent path is position-independent: applying a position-dependent rotation before or inside this path changes the representation and prevents the absorption from being equivalent.

Rotary positional embedding (RoPE) rotates pairs of channels. For a pair $(x_0, x_1)$ and angle $\theta$, the rotation is

$$
\begin{bmatrix}
x'_0 \\
x'_1
\end{bmatrix}
=
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) \\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
\begin{bmatrix}
x_0 \\
x_1
\end{bmatrix}.
$$

Because $\theta$ depends on token position, RoPE does not commute with a learned projection in general. The absorbed latent branch must remain unrotated, while a separate decoupled head can receive RoPE before being combined with the latent features.

## Task

Implement `mla_kv_features(z, head, w_latent, w_head, cos, sin)`.

The inputs are:

- `z`: latent token states with shape $(n, r)$.
- `head`: decoupled head states with shape $(n, h)$.
- `w_latent`: absorbed latent projection matrix with shape $(r, m)$.
- `w_head`: decoupled head projection matrix with shape $(h, h)$.
- `cos` and `sin`: RoPE values with shape $(n, h/2)$ for each token position.

Return a `float64` array with shape $(n, m+h)$ created by concatenating:

1. The absorbed latent branch `z @ w_latent` without RoPE.
2. The decoupled head branch `apply_rope(head @ w_head, cos, sin)`.

Do not apply RoPE to the absorbed latent branch.

## Example

```python
import numpy as np

z = np.array([[1.0, 2.0]])
head = np.array([[3.0, 4.0]])
w_latent = np.eye(2)
w_head = np.eye(2)
cos = np.array([[1.0]])
sin = np.array([[0.0]])

out = mla_kv_features(z, head, w_latent, w_head, cos, sin)
# The latent part stays [1, 2] and the head part is RoPE rotated.
```

## What the gate checks

The gate builds a NumPy oracle that performs the absorbed latent computation and applies RoPE only to the decoupled head branch. The returned array is compared with the oracle using maximum absolute error:

$$
\max_i |y_i - \hat{y}_i| \le 10^{-4}.
$$

Applying RoPE inside the absorbed latent path produces position-dependent errors and fails the numerical check.
