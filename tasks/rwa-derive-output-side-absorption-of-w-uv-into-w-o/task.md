## Context

Some transformer architectures include a value up-projection before the output projection. Let the attention probabilities be $P \in \mathbb{R}^{n \times m}$ and the latent value cache be $c_V \in \mathbb{R}^{m \times d_{latent}}$.

The normal decode path first projects the latent values into the model space and then applies the output projection:

$$
Y = (P c_V) W_{UV}^T W_O^T .
$$

Because matrix multiplication is associative, the two learned projections can be folded together:

$$
Y = (P c_V)(W_O W_{UV})^T .
$$

The absorbed projection $W'_O = W_O W_{UV}$ lets the decoder apply one projection after reading the latent attention output.

## Task

Implement `absorb_w_uv(W_O, W_UV, P, c_V)`.

The function receives list:

- `W_O` with shape $(d_{model}, d_{up})$.
- `W_UV` with shape $(d_{up}, d_{latent})$.
- `P` with shape $(n, m)$.
- `c_V` with shape $(m, d_{latent})$.

Return the decoded output using output-side absorption:

$$
(Pc_V)(W_O W_{UV})^T.
$$

The returned array must be a list with floating point values. Do not compute the expanded up-projected value sequence.

## Example

```python

W_O = [[2.0, 1.0]]
W_UV = [[3.0], [4.0]]
P = [[1.0, 0.0]]
c_V = [[5.0], [6.0]]

Y = absorb_w_uv(W_O, W_UV, P, c_V)
# Y is [[58.]]
```

## What the gate checks

The gate creates random projection matrices and attention values. It computes the oracle output by applying the original path in float64:

$$
Y_{oracle}=(P c_V)W_{UV}^T W_O^T.
$$

The submitted implementation is compared against this oracle. The maximum absolute error must satisfy

$$
\max_i |Y_i-Y_{oracle,i}| < 10^{-4}.
$$
