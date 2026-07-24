## Context

GPTQ (Generalized Post‑Training Quantization) is a column‑wise quantization scheme that iteratively updates each weight column while compensating for the error introduced by the previous columns.  
Let $W \in \mathbb{R}^{m\times n}$ be a weight matrix and $X \in \mathbb{R}^{b\times n}$ be a batch of activations.  The Hessian approximation used in GPTQ is

$$H = \frac{1}{b}\, X^\top X,$$

which captures the second‑order sensitivity of the loss to each column.

For a column $j$ we choose a scale factor $s_j>0$ and integer codes $c_{:,j}\in\{-127,\dots,127\}$ such that

$$W_{\cdot,j} \approx s_j\, c_{\cdot,j}.$$

The residual $\varepsilon_j = W_{\cdot,j}-s_j\,c_{\cdot,j}$ is then propagated to the remaining columns $k>j$ by adding

$$\Delta W_{\cdot,k}\;=\;\varepsilon_j \,\frac{H_{jk}}{H_{jj}},$$

which is a first‑order correction that keeps the overall reconstruction error small.

## Task

Implement the function

```python
def gptq_quantize(W: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ...
```

* `W` – 2‑D NumPy array of shape `(m, n)` with dtype `float32`.  
* `X` – 2‑D NumPy array of shape `(b, n)` with dtype `float32`.

The function must return a tuple

1. `codes` – integer codes of shape `(m, n)`, dtype `int8`; each entry is in the range `[-127, 127]`.  
2. `scales` – scale factors of shape `(n,)`, dtype `float64`.

The algorithm should follow the GPTQ procedure described above: compute the Hessian $H$, quantize each column sequentially, propagate the residual to all later columns using the factor $\frac{H_{jk}}{H_{jj}}$, and use a per‑column scale that maps the maximum absolute weight to `127`.

## Example

```python
import numpy as np
W = np.array([[0.5, -1.2, 3.4],
              [1.0,  0.0, -0.7]], dtype=np.float32)
X = np.random.randn(10, 3).astype(np.float32)

codes, scales = gptq_quantize(W, X)
print(codes)   # shape (2,3), int8
print(scales)  # shape (3,), float64
```

The output will contain integer codes in `[-127,127]` and corresponding scale factors.

## What the gate checks

The grader computes a reference implementation of GPTQ using NumPy.  
For each test case it reconstructs the weight matrix from the candidate’s `(codes, scales)`:

$$\hat W_{\text{cand}} = \operatorname{diag}(\text{scales})\, \text{codes},$$

and compares it to the reference reconstruction $\hat W_{\text{ref}}$ using the global relative L2 error

$$
\mathrm{rel\_err}
= \frac{\|\hat W_{\text{cand}}-\hat W_{\text{ref}}\|_F}
       {\|\hat W_{\text{ref}}\|_F + 10^{-12}}.
$$

The candidate must achieve $\mathrm{rel\_err}\le 1\times10^{-6}$ on all test cases.
