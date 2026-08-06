## Context

Rotary position embeddings (RoPE) rotate pairs of coordinates using frequencies. For a
pair of coordinates with inverse frequency $\omega$, the rotation angle at position
$p$ is

$$
\theta = p\omega .
$$

Context extension methods such as YaRN modify the inverse frequencies with a
dimension-dependent interpolation ramp. This task uses a simplified YaRN-style ramp.
For dimension index $i$, define

$$
r_i = \mathrm{clip}\left(\frac{i - \beta_{\mathrm{slow}}}
{\beta_{\mathrm{fast}}-\beta_{\mathrm{slow}}}, 0, 1\right),
$$

and interpolate the inverse frequency as

$$
\omega'_i = \frac{\omega_i}{1 + r_i(\mathrm{scale}-1)} .
$$

The adjusted frequencies are used for rotary embeddings. For a query or key vector,
each adjacent pair is rotated as

$$
\begin{bmatrix}
x'_0\\
x'_1
\end{bmatrix}
=
\begin{bmatrix}
\cos(\theta) & -\sin(\theta)\\
\sin(\theta) & \cos(\theta)
\end{bmatrix}
\begin{bmatrix}
x_0\\
x_1
\end{bmatrix}.
$$

Attention logits are then computed as a scaled dot product. The temperature factor
changes the normal attention scaling to

$$
\mathrm{scale}_{attn}=\frac{1}{\sqrt{d}\sqrt{t}},
$$

where $d$ is the hidden dimension and $t$ is the attention temperature.

## Task

Implement `yarn_ramp_temperature`:

```python
def yarn_ramp_temperature(
    q: list[list[float]],
    k: list[list[float]],
    inv_freq: list[float],
    positions: list[int],
    beta_fast: float,
    beta_slow: float,
    scale: float,
    temperature: float,
) -> list[list[float]]:
    ...
```

The inputs `q` and `k` have shape $(n,d)$ with even $d$. `inv_freq` contains
$d/2$ rotary inverse frequencies. Rotate every row of `q` and `k` using the YaRN
adjusted frequencies, then return the attention matrix with shape $(n,n)$:

$$
A_{ij} =
\frac{q_i^\top k_j}{\sqrt{d}\sqrt{t}} .
$$

Use Python operations. The returned array must be `float64`.

## Example

```python

q = [[1., 0., 0., 1.]]
k = [[1., 1., 0., 0.]]
freq = [1.0, 0.5]
pos = [2]

out = yarn_ramp_temperature(
    q, k, freq, pos,
    beta_fast=1.0,
    beta_slow=0.0,
    scale=2.0,
    temperature=1.0,
)
```

## What the gate checks

The gate computes the YaRN and rotary reference directly with Python and compares the
candidate output against that oracle. The maximum absolute element error

$$
\max_{i,j}|A_{ij}^{candidate}-A_{ij}^{reference}|
$$

must be less than $10^{-5}$.
