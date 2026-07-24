## Context

Layer Normalization (LayerNorm) normalizes the activations of a layer across the feature dimension.  
For an input tensor $X \in \mathbb{R}^{B\times F}$, where $B$ is batch size and $F$ is the number of features, LayerNorm computes

$$
\mu_b = \frac{1}{F}\sum_{f=1}^F X_{b,f}, \qquad
\sigma_b^2 = \frac{1}{F}\sum_{f=1}^F (X_{b,f}-\mu_b)^2,
$$

and outputs

$$
Y_{b,f} = \gamma_f\,\frac{X_{b,f}-\mu_b}{\sqrt{\sigma_b^2+\varepsilon}} + \beta_f .
$$

In many inference pipelines the activations are quantized to 8‑bit signed integers ($-128$ to $127$) to reduce memory bandwidth.  
A *quantization‑aware* implementation must therefore accept an int8 input, dequantize it internally (here we use a scale of $1.0$ for simplicity), perform the normalisation in floating point, and return a high‑precision output.

## Task

Implement the function

```python
def layernorm_int8(x: np.ndarray,
                   gamma: np.ndarray,
                   beta: np.ndarray,
                   eps: float = 1e-5) -> np.ndarray:
    ...
```

* `x` is a 2‑D NumPy array of dtype `np.int8` with shape `(B, F)`.
* `gamma` and `beta` are 1‑D arrays of length `F` (or scalars that broadcast).
* The function must return a NumPy array of dtype `float64` containing the LayerNorm output.
* Use vectorised NumPy operations only; no explicit Python loops.

## Example

```python
import numpy as np
x = np.array([[0, 1, -1],
              [2, -3, 4]], dtype=np.int8)
gamma = np.ones(3, dtype=np.float64)
beta = np.zeros(3, dtype=np.float64)

y = layernorm_int8(x, gamma, beta)
print(y)
# [[ 0.          1.         -1.        ]
#  [ 2.          -3.           4.        ]]
```

(The example uses a trivial scale of $1$ and identity parameters; the output is identical to the input.)

## What the gate checks

The grader computes a reference LayerNorm using floating‑point dequantisation:

$$
X_{\text{float}} = X \times 1.0,
$$

then applies the standard formula above.  
It compares your implementation against this reference with the scorer `rel_err`.  
Your solution must achieve a relative error $\le 10^{-6}$ on all random test cases.
