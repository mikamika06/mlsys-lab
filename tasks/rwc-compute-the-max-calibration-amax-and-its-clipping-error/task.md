## Context

In quantization, the symmetric integer scale is often derived from a calibration step that records the maximum absolute activation value across many batches. Let $X$ denote a collection of floating‑point tensors collected during calibration. The **amax** used for an 8‑bit signed representation is defined as

$$
\mathrm{amax} = \max_{x \in X}\, |x|.
$$

With this amax we choose a linear scale factor

$$
s = \frac{\mathrm{amax}}{127},
$$

so that the integer range $[-127,\,127]$ covers all observed values. Each float $x$ is quantized to an integer

$$
q = \operatorname{round}\!\bigl(\tfrac{x}{s}\bigr),
$$

and reconstructed as $\hat x = q\, s$. The **clipping error** measures how much the reconstruction deviates from the original. A convenient metric is the mean‑squared‑error (MSE)

$$
\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N} (\hat x_i - x_i)^2,
$$

where $N$ is the total number of elements in all calibration batches.

## Task

Implement a function that, given a list of NumPy arrays representing calibration batches, returns a tuple `(amax, mse)` where:

* `amax` is the maximum absolute value seen across all batches.
* `mse` is the reconstruction MSE after quantizing with scale `s = amax / 127`.

```python
def calibrate_max_and_error(batches: list[np.ndarray]) -> tuple[float, float]:
    ...
```

The implementation must use only NumPy operations; no explicit Python loops over individual elements. The returned values should be of type `float` (Python float).

## Example

```python
import numpy as np

b1 = np.array([[0.5, -1.2], [3.0, 0.0]])
b2 = np.array([[-0.7, 2.4], [1.1, -3.3]])

amax, mse = calibrate_max_and_error([b1, b2])
print(amax)   # 3.3
print(mse)    # ≈ 0.000... (depends on quantization)
```

## What the gate checks

The grader recomputes the reference `amax` and `mse` from the same batches using a NumPy‑only oracle. It then compares your outputs to the oracle with a relative error metric:

$$
\mathrm{rel\_err} = \max\!\left(
    \frac{|\,\texttt{student\_amax} - \texttt{oracle\_amax}\,|}{\texttt{oracle\_amax}},
    \frac{|\,\texttt{student\_mse}   - \texttt{oracle\_mse}\,  }{\texttt{oracle\_mse}}
\right).
$$

The solution must satisfy $\mathrm{rel\_err}\le10^{-9}$.
