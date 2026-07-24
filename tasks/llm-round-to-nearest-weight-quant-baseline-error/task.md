## Context

In many large‑scale language models, the weight tensors of linear layers are often quantized to reduce memory and compute requirements. A common approach is **round‑to‑nearest (RTN)** symmetric quantization: each floating‑point value $w$ is mapped to an integer $q$ by dividing by a scale factor $\alpha$, rounding to the nearest integer, and clipping to the representable range.

For an $n$‑bit signed representation the range is
$$\mathcal{Q} = \{-2^{\,n-1},\dots,-1,0,1,\dots,2^{\,n-1}-1\}.$$
The scale factor $\alpha$ is chosen so that the largest magnitude weight maps to the maximum representable integer:
$$\alpha = \frac{\max_{i}|w_i|}{2^{\,n-1}-1}.$$

After quantization we can recover a floating‑point approximation by multiplying back with $\alpha$. The quality of this approximation is measured by the **relative error**
$$
\mathrm{rel\_err}(W,\hat W) = \frac{\lVert \hat W - W\rVert}{\lVert W\rVert},
$$
where $W$ are the original weights and $\hat W$ their de‑quantized counterpart.

## Task

Implement a function `round_to_nearest` that performs symmetric RTN quantization of a NumPy array:

```python
def round_to_nearest(W: np.ndarray, num_bits: int) -> np.ndarray:
    ...
```

* `W` is an arbitrary‑shaped NumPy array of type `float32`.
* `num_bits` is the number of signed bits to use (e.g. 8 or 16).
* The function must return an integer array with dtype `np.int8` if `num_bits <= 8`, otherwise `np.int16`.  
  The returned values must lie in the range $\mathcal{Q}$ defined above.
* Internally compute the scale factor as described, round to nearest integer, clip to the representable range, and cast to the correct dtype.

The function should be fully vectorized; no explicit Python loops are allowed.

## Example

```python
import numpy as np
W = np.array([[0.1, -2.3], [4.5, 0.0]], dtype=np.float32)
q8 = round_to_nearest(W, 8)   # int8 array
# q8 might be:
# array([[-1, -127],
#        [ 127,    0]], dtype=int8)

# De‑quantize to check error
scale = np.max(np.abs(W)) / (2**(8-1)-1)
W_hat = q8.astype(np.float32) * scale
```

## What the gate checks

The grader computes the relative error between the original weights and the de‑quantized approximation using a reference implementation. The candidate must achieve $\mathrm{rel\_err} \le 0.05$ on a fixed random tensor of shape $(5,7)$ with $8$‑bit quantization.
