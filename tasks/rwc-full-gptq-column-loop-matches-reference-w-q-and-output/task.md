## Context

GPTQ is a post-training quantization method that reduces the precision of a weight matrix while compensating for the effect of quantization errors. For a linear layer with weights $W \in \mathbb{R}^{m \times n}$ and calibration inputs $X \in \mathbb{R}^{n \times s}$, the layer output is

$$Y = WX.$$

The algorithm estimates curvature from the calibration data using a Hessian-like matrix

$$H = XX^\top,$$

then applies damping before computing an inverse curvature matrix. The quantization process moves through columns from left to right. When a column is quantized, the introduced error is propagated into future columns using the inverse Hessian information.

For a column $i$, if the current working column is $w_i$ and its quantized version is $q_i$, the error is

$$e_i = q_i - w_i.$$

The future columns are updated using the inverse Hessian row:

$$W_{:,j} \leftarrow W_{:,j} - e_i \frac{H^{-1}_{ij}}{H^{-1}_{ii}}, \quad j > i.$$

This lazy error compensation is the core of the GPTQ column loop.

## Task

Implement `gptq_quantize(W, X, bits=3, group_size=2, damp=0.01)`.

The function receives a floating point weight matrix `W` with shape $(m,n)$ and calibration matrix `X` with shape $(n,s)`. It must return a tuple `(W_q, Y_q)`.

Requirements:

- Use a left-to-right GPTQ column loop.
- Compute the curvature matrix from `X`.
- Apply diagonal damping using `damp`.
- Quantize columns using symmetric integer quantization with `bits` bits.
- Use groups of input columns with shared per-output-channel scales. A group containing columns $a$ through $b$ uses one scale vector computed from that slice of the working weights.
- Propagate quantization error to later columns using the inverse Hessian update.
- `W_q` must contain the final quantized weights as `float64`.
- `Y_q` must equal `W_q @ X`.

The returned tuple must have:

```python
def gptq_quantize(
    W: np.ndarray,
    X: np.ndarray,
    bits: int = 3,
    group_size: int = 2,
    damp: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

## Example

```python
import numpy as np

W = np.array([[0.8, -0.3], [0.2, 0.9]], dtype=np.float64)
X = np.array([[1.0, 0.5], [-0.5, 1.0]], dtype=np.float64)

W_q, Y_q = gptq_quantize(W, X)
```

The returned matrix `Y_q` is the quantized layer output:

$$Y_q = W_q X.$$

## What the gate checks

The gate computes a NumPy GPTQ oracle independently. It compares the student's quantized weight matrix against the oracle using relative error

$$\mathrm{rel\_err} = \frac{\lVert A-B\rVert_2}{\lVert A\rVert_2 + 10^{-12}}.$$

It also compares the reconstructed output $W_qX$. A solution that skips Hessian-based error propagation, quantizes without grouping, or only performs independent rounding will fail the numerical checks.
