## Context

Quantization maps floating point weights into a smaller numerical format. A production quantizer usually chooses a scale $s$ and stores values after mapping

$$
q = Q\left(\frac{x}{s}\right),
$$

where $Q$ rounds to the nearest representable value of the target format. Reconstruction is

$$
\hat{x} = s q .
$$

Different formats make different tradeoffs. FP8 E4M3 has more mantissa bits and therefore better precision near its range limit, while FP8 E5M2 has more exponent bits and can represent larger magnitudes. INT8 uses a uniform grid, which can be effective when values are distributed evenly but may lose precision on heavy-tailed weights with outliers.

For a tensor $W$, the scale can be selected per tensor. The quantization error is measured as

$$
\mathrm{rel\_err}(W,\hat{W}) =
\frac{\lVert \hat{W}-W\rVert_2}{\lVert W\rVert_2 + 10^{-12}} .
$$

## Task

Implement `compare_quant_formats(weights)`:

```python
def compare_quant_formats(weights: np.ndarray) -> dict:
    ...
```

The input is a one-dimensional NumPy array of floating point weights.

Return a dictionary with exactly these keys:

- `"e4m3_error"`: relative reconstruction error using FP8 E4M3.
- `"e5m2_error"`: relative reconstruction error using FP8 E5M2.
- `"int8_error"`: relative reconstruction error using signed INT8.
- `"best_format"`: the string name of the format with the smallest error. Use one of `"e4m3"`, `"e5m2"`, or `"int8"`.

Use the same per-tensor scale selection strategy for all formats: select the scale that minimizes the relative reconstruction error over the input tensor.

## Example

```python
import numpy as np

w = np.array([0.1, -0.2, 0.3, 12.0], dtype=np.float64)
result = compare_quant_formats(w)

# result contains:
# {
#   "e4m3_error": ...,
#   "e5m2_error": ...,
#   "int8_error": ...,
#   "best_format": ...
# }
```

## What the gate checks

The gate builds a NumPy oracle that implements the same quantization search for FP8 E4M3, FP8 E5M2, and INT8. It compares the three returned errors with the oracle values using relative error $\le 10^{-9}$ and checks that the reported lowest-error format matches the oracle choice.
