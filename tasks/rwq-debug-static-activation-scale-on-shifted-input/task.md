## Context

Activation quantization reduces memory bandwidth by representing floating point
activations with integer values. For symmetric int8 quantization, a tensor $x$
is mapped using a scale:

$$s = \frac{\max(|x|)}{127},$$

$$q = \mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right), -127, 127\right),$$

and reconstructed as

$$\hat{x} = q s.$$

A dynamic quantization path computes $s$ from the current input. This matters
when the input distribution changes. A scale computed from an earlier batch can
cause clipping or poor resolution on a shifted batch.

For a linear layer with weight matrix $W$ and bias $b$, the quantized activation
path is:

$$Y = W \hat{x} + b.$$

A production dynamic quantizer recomputes the activation scale for every input
tensor before performing the integer round-trip.

## Task

Implement `quantized_linear_dynamic`:

```python
def quantized_linear_dynamic(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...
```

The function receives a batch of activations `x` with shape $(n, d)$, weights
`W` with shape $(m, d)$, and bias `b` with shape $(m,)$.

Return the output matrix of shape $(n, m)$ after dynamically quantizing the
activation tensor and applying the linear layer. The activation scale must be
computed from the provided `x` on every call. Use NumPy operations only.

The function should return `float64` values.

## Example

```python
import numpy as np

x = np.array([[1.0, 2.0], [100.0, 50.0]])
W = np.array([[1.0, -1.0]])
b = np.array([0.0])

y = quantized_linear_dynamic(x, W, b)
```

The scale for this call is derived from `x` because the two rows have different
magnitudes. Reusing a scale from a previous batch is incorrect.

## What the gate checks

The gate compares the implementation against a NumPy oracle that performs the
dynamic int8 activation quantization algorithm on a distribution-shifted input.
The relative error

$$\mathrm{rel\_err} =
\frac{\lVert Y_{\mathrm{candidate}}-Y_{\mathrm{oracle}}\rVert}
{\lVert Y_{\mathrm{oracle}}\rVert + 10^{-12}}$$

must satisfy $\mathrm{rel\_err} \le 10^{-3}$.

A solution that caches the first input's activation scale fails because the
oracle recomputes the scale for each input tensor.
