## Context

Asymmetric affine (min-max) quantization maps a real-valued group $x \in
\mathbb{R}^n$ onto integer codes $q \in \{0, 1, \dots, 2^{b}-1\}$ (for $b$
bits, unsigned range) using a scale $s$ and integer zero-point $z$:

$$
q_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x_i}{s}\right) + z,\; 0,\; 2^b-1\right), \qquad
\hat{x}_i = (q_i - z)\, s
$$

For this mapping to be **lossless at zero** (real value $0.0$ maps to an
exact integer code — important so that padding / ReLU-zeroed activations
don't accumulate error) and to *minimize the worst-case reconstruction
error* over the group, the standard closed-form derivation extends the
group's range to always include zero, then spreads the quantization levels
evenly across that extended range:

$$
x_{\min}' = \min(0, \min(x)), \qquad x_{\max}' = \max(0, \max(x))
$$

$$
s = \frac{x_{\max}' - x_{\min}'}{2^b - 1} \quad (\text{or } s=1 \text{ if } x_{\max}' = x_{\min}')
$$

$$
z = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{-x_{\min}'}{s}\right),\; 0,\; 2^b - 1\right)
$$

With this choice, every $x_i \in [x_{\min}', x_{\max}']$ lands inside the
representable range (no clipping), and the round-to-nearest step guarantees
$|\hat{x}_i - x_i| \le s/2$ for every element — i.e. $s$ is the smallest
scale that keeps the whole group representable, which minimizes the
worst-case (max-abs) reconstruction error achievable by *any* affine
`(s, z)` pair covering the group's range.

## Task

Implement `derive_affine_qparams`:

```python
def derive_affine_qparams(x: list[float], nbits: int) -> tuple[float, int]:
    ...
```

* `x` — 1-D float array: one quantization group.
* `nbits` — bit width $b$ (unsigned code range `0 .. 2**nbits - 1`).

Return `(scale, zero_point)` — `scale` a Python `float`, `zero_point` a
Python `int`, computed exactly as derived above.

## Example

```python
x = [-2.0, -1.0, 0.5, 3.0]
scale, zp = derive_affine_qparams(x, nbits=4)
# x_min' = -2.0, x_max' = 3.0, scale = 5.0/15 = 0.3333...
# zp = round(2.0/0.3333...) = 6
```

## What the gate checks

Gate **max_abs_err** re-derives `(scale, zero_point)` from the same min/max
formula with a Python oracle, quantizes-then-dequantizes `x` with **both**
your `(scale, zero_point)` and the oracle's, and reports the max-abs
difference between the two reconstructions across several random groups and
bit widths (including a degenerate constant-array group). A correct
derivation reproduces the oracle's reconstruction exactly (up to floating
point noise); any deviation in the formula (e.g. forgetting to include zero
in the range, or a symmetric instead of asymmetric mapping) shows up as a
nonzero max-abs error.
