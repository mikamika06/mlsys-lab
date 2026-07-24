## Context

In many neural‑network inference engines activations are quantized to unsigned 8‑bit integers (uint8). The mapping from a real value $x$ to an integer $q \in [0,255]$ is usually linear:
$$
q = \operatorname{round}\!\left(\frac{x - \text{min}}{\text{scale}}\right),
$$
where $\text{scale} > 0$ and $\text{min}$ is the smallest activation value in a given tensor. The *zero‑point* $z$ is defined such that $x=0$ maps to $q=z$, i.e.
$$
z = \operatorname{round}\!\left(-\frac{\text{min}}{\text{scale}}\right).
$$

When the range of activations varies from batch to batch, a *dynamic* quantization scheme recomputes $\text{scale}$ and $z$ for each batch. The most common choice is to use 255 discrete levels, so
$$
\text{scale} = \frac{\max - \min}{255}.
$$

If all activations are identical ($\max=\min$) the range collapses; in that case we conventionally set $\text{scale}=1.0$ and $z=128$, which places the constant value at the centre of the uint8 spectrum.

## Task

Implement a function

```python
def dynamic_activation_scale_zero_point(x: np.ndarray) -> tuple[float, int]:
    ...
```

that receives a 1‑D NumPy array `x` containing floating‑point activations and returns a pair `(scale, zero_point)` computed as described above. The returned scale must be a Python float (or numpy scalar) and the zero point an integer in the inclusive range `[0,255]`. Your implementation should work for any shape of `x`, but you may assume it is 1‑D.

## Example

```python
import numpy as np
x = np.array([-2.5, 0.0, 3.5])
scale, zp = dynamic_activation_scale_zero_point(x)
print(scale)   # 0.025
print(zp)      # 100
```

Here $\min=-2.5$, $\max=3.5$ so $\text{scale}=(3.5-(-2.5))/255 \approx 0.025$ and $z=\operatorname{round}(2.5/0.025)=100$.

## What the gate checks

Two metrics are evaluated:

* **`scale_rel_err`** – the relative L2 error between your returned scale and a NumPy reference implementation. The value must satisfy $\text{scale\_rel\_err} \le 10^{-8}$.
* **`zero_point_match`** – an exact integer equality test; it must be `1.0` when the zero point matches the reference, otherwise `0.0`.

The grader uses a deterministic random batch of activations to compute the reference values.
