## Context

Low-precision arrays save memory and improve throughput, but repeated accumulation can lose information because the accumulator cannot represent small updates after the running total becomes large.

For values $x_1, x_2, \dots, x_n$, the ordinary sum is

$$
s = \sum_{i=1}^{n} x_i .
$$

A naive low-precision implementation repeatedly rounds the intermediate value:

$$
s_{k+1} = \operatorname{round}(s_k + x_{k+1}) .
$$

Kahan summation keeps an additional compensation value $c$ that tracks lost low-order bits:

$$
y = x_k - c
$$

$$
t = s + y
$$

$$
c = (t - s) - y
$$

$$
s = t .
$$

The returned sum should be much closer to a high-precision oracle than summing directly in the input precision.

## Task

Implement `kahan_sum_fp16(a)`:

```python
def kahan_sum_fp16(a: np.ndarray) -> float:
    ...
```

The function receives a one-dimensional NumPy array with dtype `float16` and returns the sum as a Python `float`.

Use Kahan compensated summation with a `float32` accumulator and compensation variable. Do not convert the whole input array to `float64` before summing.

## Example

```python
import numpy as np

a = np.array([10000, 1, -10000, 1, 1], dtype=np.float16)
result = kahan_sum_fp16(a)
# result is close to 3.0
```

## What the gate checks

The gate computes a reference answer using NumPy's `float64` summation as the numerical oracle. It measures

$$
\mathrm{rel\_err} =
\frac{\lVert s_{\mathrm{candidate}} - s_{\mathrm{oracle}} \rVert}
{\lVert s_{\mathrm{oracle}} \rVert + 10^{-12}} .
$$

The reported `rel_err` must be at most $10^{-6}$ across several cancellation-heavy `float16` inputs. Direct `float16` accumulation loses too much information and does not pass.
