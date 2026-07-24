## Context

Floating point summation is sensitive to rounding error. In fp16, many small updates can disappear when they are added to a much larger partial sum because the spacing between representable numbers grows with magnitude.

For values $x_1, x_2, \dots, x_n$, the ideal sum is

$$
S = \sum_{i=1}^{n} x_i .
$$

A naive floating point accumulator repeatedly applies

$$
s \leftarrow \mathrm{round}(s + x_i),
$$

which can lose low-order bits. Kahan compensated summation keeps an additional correction term $c$ that tracks lost precision:

$$
y = x_i - c,
$$

$$
t = s + y,
$$

$$
c = (t - s) - y,
$$

$$
s = t.
$$

For mixed precision workloads, inputs may be stored in fp16 while accumulation uses a wider intermediate representation to reduce error.

## Task

Implement `kahan_sum_fp16(x)`:

```python
def kahan_sum_fp16(x: np.ndarray) -> float:
    ...
```

The function receives a one-dimensional NumPy array with dtype `float16`. Return a Python `float` containing the compensated sum of the values.

Use Kahan summation over the fp16 values. The accumulation and compensation variables should preserve enough precision to avoid the large errors caused by a naive fp16 accumulator.

## Example

```python
import numpy as np

x = np.array([0.1] * 5000, dtype=np.float16)
s = kahan_sum_fp16(x)

# The result is close to the fp64 reference:
# np.sum(x, dtype=np.float64)
```

## What the gate checks

The gate computes the reference result using NumPy fp64 summation:

$$
S_{\mathrm{ref}} = \mathrm{sum}(x, \mathrm{dtype}=\mathrm{float64}).
$$

It measures the relative error

$$
\mathrm{rel\_err} =
\frac{\lVert S - S_{\mathrm{ref}} \rVert}
{\lVert S_{\mathrm{ref}} \rVert + 10^{-12}}.
$$

The returned value must satisfy $\mathrm{rel\_err} \le 10^{-6}$ across several fp16 inputs where naive fp16 accumulation loses significant precision.
