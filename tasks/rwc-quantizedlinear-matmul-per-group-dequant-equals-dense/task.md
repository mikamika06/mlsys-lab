## Context

Production inference libraries often store neural network weights in quantized form to reduce memory usage. A common format groups consecutive output channels and stores each group with its own scale and zero point.

For a weight matrix $W \in \mathbb{R}^{m \times k}$, the dense linear operation is

$$
Y = X W^\top ,
$$

where $X \in \mathbb{R}^{n \times k}$ contains input activations and $Y \in \mathbb{R}^{n \times m}$ contains output activations.

In grouped affine quantization, an integer weight matrix $Q$ is converted back to approximate floating point weights using

$$
W_{ij} = (Q_{ij} - z_g) s_g ,
$$

where $g$ is the group containing output row $i$, $s_g$ is the group's scale, and $z_g$ is its zero point. The quantized linear layer first reconstructs the grouped weights and then performs the dense matrix multiplication.

## Task

Implement `quantized_linear(X, Q, scales, zeros, group_size)`:

```python
def quantized_linear(X, Q, scales, zeros, group_size):
    ...
```

The arguments are:

- `X`: a NumPy array of shape $(n, k)$ with floating point activations.
- `Q`: an integer NumPy array of shape $(m, k)$ containing quantized weights.
- `scales`: a floating point array of length $\lceil m / \text{group\_size} \rceil$.
- `zeros`: an integer array of the same length as `scales`.
- `group_size`: the number of output rows sharing one scale and zero point.

Return the floating point output array of shape $(n, m)$.

The function must implement grouped dequantization and the matrix multiplication. It may use NumPy operations.

## Example

```python
import numpy as np

X = np.array([[1.0, 2.0]])
Q = np.array([[2, 4], [5, 7]], dtype=np.int8)
scales = np.array([0.5, 0.25])
zeros = np.array([0, 1])
Y = quantized_linear(X, Q, scales, zeros, 1)

# Equivalent dense computation:
# W = [[1.0, 2.0], [1.0, 1.5]]
# Y = [[5.0, 4.0]]
```

## What the gate checks

The gate creates real quantized weight matrices and activation matrices, then computes the oracle result by dequantizing the grouped weights with NumPy and multiplying them densely.

The returned value is compared with the oracle using maximum absolute error:

$$
\max_{i,j} |Y_{ij}^{candidate} - Y_{ij}^{oracle}|.
$$

The result must satisfy the numerical tolerance. Implementations that ignore group scales, use incorrect zero point handling, or apply scales to the wrong rows will fail.
