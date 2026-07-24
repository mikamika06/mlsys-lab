## Context

Mixed-precision machine learning often stores values in low precision formats such as
$fp16$ to reduce memory usage and increase throughput. Accumulation is a separate
choice: keeping the running sum in low precision can introduce repeated rounding
errors.

For vectors $a, b \in \mathbb{R}^{n}$, the dot product is

$$
a^\top b = \sum_{i=1}^{n} a_i b_i .
$$

An FP16 accumulation path rounds the partial sum after each update:

$$
s_{k+1} = \operatorname{round}_{fp16}(s_k + a_k b_k),
$$

while a mixed-precision path keeps the accumulator in FP32:

$$
s_{k+1} = \operatorname{round}_{fp32}(s_k + a_k b_k).
$$

Long reductions amplify the difference between these two strategies. A common neural
network optimization is to store activations or weights in lower precision while
using a wider accumulator for reductions.

## Task

Implement `fp32_dot_sum(a, b)`:

```python
def fp32_dot_sum(a: np.ndarray, b: np.ndarray) -> float:
    ...
```

The function receives two one-dimensional NumPy arrays. Compute the dot product with
inputs converted to `float32` and accumulate the running sum in `float32`. Return a
Python `float` containing the final accumulator value.

The implementation should not use an FP16 accumulator.

## Example

```python
import numpy as np

a = np.ones(10000, dtype=np.float16)
b = np.full(10000, 0.001, dtype=np.float16)

result = fp32_dot_sum(a, b)
# result is closer to the FP64 reference than an FP16 accumulation
```

## What the gate checks

The gate computes an FP64 NumPy dot product as the numerical oracle. The relative
error

$$
\mathrm{rel\_err} =
\frac{|x_{\mathrm{candidate}} - x_{\mathrm{oracle}}|}
{|x_{\mathrm{oracle}}| + 10^{-12}}
$$

must remain below the tolerance.

The gate also computes a real FP16 accumulation path using NumPy scalar rounding and
compares its error against the submitted implementation. The submitted implementation
must be substantially better than the FP16 accumulator, which prevents a solution that
only performs the reduction in FP16.
