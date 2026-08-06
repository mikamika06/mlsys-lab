## Context

Transformer blocks use a residual stream where each sublayer adds an update back to the current representation. The location of layer normalization changes the computation.

A post-norm block applies normalization after each residual addition:

$$
y = \mathrm{Norm}(x + F(x)).
$$

A pre-norm block applies normalization before the sublayer and keeps the residual stream unnormalized:

$$
y = x + F(\mathrm{Norm}(x)).
$$

For a transformer-like block with two residual sublayers, the pre-norm wiring is

$$
h_1 = x + A(\mathrm{Norm}(x)),
$$

$$
y = h_1 + M(\mathrm{Norm}(h_1)),
$$

where $A$ and $M$ are simplified attention and feed-forward transformations.

Layer normalization over the last dimension is

$$
\mathrm{Norm}(x)_i =
\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},
$$

where $\mu$ and $\sigma^2$ are computed from the vector elements.

## Task

Implement `transformer_block(x, w_attn, w_ff, gamma, beta)`.

The inputs are list:

- `x` has shape $(n, d)$ and is the residual stream.
- `w_attn` and `w_ff` are $(d, d)$ weight matrices.
- `gamma` and `beta` are layer normalization parameters of shape $(d,)$.

The function must return the output of the pre-norm block:

$$
h_1 = x + \mathrm{Norm}(x) W_{\mathrm{attn}},
$$

$$
y = h_1 + \mathrm{Norm}(h_1) W_{\mathrm{ff}}.
$$

Use Python operations and return a `float64` array.

## Example

```python

x = [[1.0, 2.0], [3.0, 5.0]]
w_attn = [[1.0 if i == j else 0.0 for j in range(2)] for i in range(2)]
w_ff = [[1.0 if i == j else 0.0 for j in range(2)] for i in range(2)]
gamma = [1.0] * 2
beta = [0.0] * 2

y = transformer_block(x, w_attn, w_ff, gamma, beta)
```

The output keeps the residual connections around the normalized sublayers rather than normalizing the residual stream after each addition.

## What the gate checks

The gate computes a Python reference implementation of the pre-norm block and compares the submitted function output against that oracle.

The maximum absolute error

$$
\max_i |y_i-\hat{y}_i|
$$

must be less than $10^{-5}$. Implementations that leave normalization after the residual addition produce a different residual stream and fail.
