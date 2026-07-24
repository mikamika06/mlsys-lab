## Context

Low-precision training formats such as fp16 have a smaller range and fewer representable values than float32. Tiny gradients can become zero when converted directly to fp16.

Loss scaling multiplies gradients before the low-precision conversion and divides them afterward. For an original gradient vector $g$, a scale factor $s$, and a simulated fp16 round trip, the process is

$$
\hat{g} = \mathrm{float16}(s g) / s .
$$

Without scaling, the conversion is

$$
g_{\mathrm{plain}} = \mathrm{float16}(g).
$$

For sufficiently small values, $g_{\mathrm{plain}}$ may contain zeros because they are below the fp16 representable range.

The reconstruction quality can be measured with relative error:

$$
\mathrm{rel\_err}(g, \hat{g}) =
\frac{\lVert \hat{g} - g \rVert_2}{\lVert g \rVert_2 + 10^{-12}} .
$$

## Task

Implement `loss_scale_round_trip(grad, scale)`:

```python
def loss_scale_round_trip(grad: np.ndarray, scale: float) -> np.ndarray:
    ...
```

The function receives a float32 NumPy array of gradients and a positive loss scale. Return a float32 array reconstructed by:

1. Multiplying the gradients by `scale`.
2. Converting the scaled values to NumPy `float16` to emulate low precision storage.
3. Converting back to float32.
4. Dividing by `scale`.

Do not simply cast the input to float16 without scaling. The returned array should preserve the tiny gradients as accurately as possible.

## Example

```python
import numpy as np

g = np.array([1e-8, -2e-8, 3e-7], dtype=np.float32)
out = loss_scale_round_trip(g, 65536.0)

# out remains close to g because scaling avoids fp16 underflow.
```

## What the gate checks

The gate computes the reference round trip using NumPy float16 conversion as the low-precision oracle. It measures the relative error $\mathrm{rel\_err}$ of the submitted result against the original gradients and requires

$$
\mathrm{rel\_err} < 10^{-3}.
$$

It also computes the error of the unscaled fp16 conversion and requires the submitted implementation to have strictly lower error. A direct cast to float16 loses the tiny gradients and does not pass.
