## Context

Low-bit weight quantization maps floating point values to a small integer grid and then reconstructs them. A 4-bit quantizer has $2^4 = 16$ possible integer codes.

An affine asymmetric quantizer uses the observed minimum and maximum values. For a tensor $W$, the scale and zero point are

$$
s = \frac{W_{\max}-W_{\min}}{15},
$$

$$
z = \operatorname{round}\left(-\frac{W_{\min}}{s}\right).
$$

Each value is quantized with

$$
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{W}{s}+z\right), 0, 15\right),
$$

and reconstructed with

$$
\hat{W} = s(q-z).
$$

A symmetric absmax quantizer instead centers the grid around zero. Using signed int4 codes from $-8$ to $7$, it uses

$$
s = \frac{\max(|W|)}{7},
$$

then

$$
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{W}{s}\right), -8, 7\right),
$$

with reconstruction

$$
\hat{W} = s q.
$$

For skewed weights, the asymmetric grid can use the available levels more effectively because it does not force zero to be the midpoint of the range.

## Task

Implement `compare_int4_quantizers(W)`:

```python
def compare_int4_quantizers(W: list[float]) -> tuple[float, float, str]:
    ...
```

The input is a list of floating point weights. Compute the mean squared reconstruction error for both int4 schemes:

1. affine asymmetric min/max quantization
2. symmetric absmax quantization

Return:

- the affine reconstruction MSE as a Python `float`
- the symmetric reconstruction MSE as a Python `float`
- the name of the lower-error scheme, exactly `"affine"` or `"symmetric"`

Use Python operations for the calculations.

## Example

```python

W = [-1.0, -0.2, 0.1, 2.5]

affine_err, symmetric_err, best = compare_int4_quantizers(W)

# best is the scheme with the smaller reconstruction error
```

## What the gate checks

The gate builds skewed weight arrays and computes the two int4 reconstructions with an independent Python oracle. The returned two errors must match the oracle errors with relative error

$$
\frac{\lVert e_{\text{student}}-e_{\text{oracle}}\rVert_2}
{\lVert e_{\text{oracle}}\rVert_2 + 10^{-12}}
$$

below the required threshold.

The selected scheme must also match the oracle choice. A solution that always selects one quantizer fails because the best method depends on the weight distribution.
