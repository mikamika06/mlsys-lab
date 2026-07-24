## Context

Uniform affine quantization represents a weight tensor $W$ using a scale $s$ and a zero-point $z$:

$$
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{W}{s}\right)+z,\ 0,\ 2^b-1\right),
$$

where $b$ is the number of bits. The reconstructed tensor is

$$
W_q = s(q-z).
$$

In HQQ-style optimization, a debugging pass may keep the scale fixed and optimize only the zero-point. The scale is not updated during this phase because changing it alters the quantization grid instead of correcting the affine offset.

For a fixed scale $s$, the objective is to minimize reconstruction error:

$$
\min_z \lVert W - s(\operatorname{clip}(\operatorname{round}(W/s)+z,0,2^b-1)-z)\rVert^2 .
$$

A common bug is swapping the optimized variable: the loop updates $s$ while leaving $z$ unchanged. This task asks you to correct that update direction.

## Task

Implement `optimize_zero_point(W, scale, bits, iters)`.

The function receives a NumPy array `W`, a fixed positive scalar `scale`, an integer bit width `bits`, and an iteration count. It must return:

```python
W_q, z
```

where `W_q` is the reconstructed quantized tensor and `z` is the optimized integer zero-point.

Use the following algorithm:

1. Initialize $z=0$.
2. For each iteration, evaluate candidate zero-points in the range $[z-2,z+2]$.
3. Select the candidate with the smallest squared reconstruction error while keeping `scale` fixed.
4. Use the final $z$ to compute and return $W_q$.

The returned `W_q` must be `float64`.

## Example

```python
import numpy as np

W = np.array([-1.0, 0.2, 1.5, 3.0])
W_q, z = optimize_zero_point(W, 0.5, 3, 5)

# z is chosen by minimizing reconstruction error with scale=0.5 fixed.
# W_q contains the affine quantized reconstruction.
```

## What the gate checks

The gate computes an independent NumPy oracle implementing the fixed-scale zero-point optimization loop. The returned reconstruction is compared with the oracle using

$$
\max_i |(W_q)_i - (W_{q,\mathrm{oracle}})_i|.
$$

The metric `max_abs_err` must be at most $10^{-6}$. A solution that updates the scale instead of the zero-point produces a different quantization grid and fails this check.
