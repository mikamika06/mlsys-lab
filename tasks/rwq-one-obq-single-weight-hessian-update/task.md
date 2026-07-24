## Context

Optimal Brain Quantization (OBQ) chooses a single weight to quantize while compensating
for the induced error using an approximate inverse Hessian. For a weight vector
$w \in \mathbb{R}^n$, a candidate quantized value $q_i$ for index $i$ has an
estimated local loss increase:

$$
\Delta_i = \frac{(q_i - w_i)^2}{H^{-1}_{ii}},
$$

where $H^{-1}$ is the inverse Hessian approximation.

After selecting the weight with the smallest OBS cost, the remaining weights are
updated to compensate for the quantization error. If index $k$ is selected and
$q_k$ is the chosen quantized value, the error is

$$
e = q_k - w_k .
$$

The closed-form OBS update is

$$
w_j \leftarrow w_j - e \frac{H^{-1}_{jk}}{H^{-1}_{kk}}
$$

for all $j \ne k$, while the selected element becomes $w_k \leftarrow q_k$.

## Task

Implement `obq_single_weight_update(w, Hinv, grid)`.

Inputs:

- `w`: a 1-D NumPy array containing the floating point weights.
- `Hinv`: a square NumPy array containing the inverse Hessian approximation.
- `grid`: a 1-D NumPy array of allowed quantization values.

The function must:

1. Find the nearest grid value for every weight.
2. Compute the OBS cost for each possible single-weight quantization.
3. Select the index with the smallest cost.
4. Apply exactly one OBQ Hessian compensation update.
5. Return the updated weight vector as a NumPy array of type `float64`.

No loops over alternative update rules are needed. Use the formulas above directly.

## Example

```python
import numpy as np

w = np.array([0.8, -1.2, 0.3])
Hinv = np.array([
    [2.0, 0.1, 0.0],
    [0.1, 1.0, 0.2],
    [0.0, 0.2, 3.0],
])
grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

out = obq_single_weight_update(w, Hinv, grid)
```

The function returns the compensated vector after quantizing exactly one weight.

## What the gate checks

The gate computes an independent NumPy oracle that performs the OBQ selection and
Hessian update from the equations above. The returned vector must have

$$
\max_j |w^{candidate}_j - w^{oracle}_j| < 10^{-8}.
$$

The check uses multiple deterministic vectors, Hessian inverses, and int4-style
quantization grids.
