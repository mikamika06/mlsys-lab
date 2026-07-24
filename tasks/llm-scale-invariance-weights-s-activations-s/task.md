## Context

In many quantization schemes for large language models, a *scale invariance* property is exploited.  
Let $W \in \mathbb{R}^{n\times d}$ be a weight matrix and $X \in \mathbb{R}^{d\times m}$ an activation matrix.  
For any non‑zero scalar $s$, the product

$$
W X = (W\, s)\,\bigl(X / s\bigr)
$$

holds exactly in real arithmetic. This identity is at the heart of Activation‑Aware Weight Quantization (AWQ), where weights are scaled by a factor $s$ and activations are divided by the same factor before matrix multiplication, allowing the quantized product to match the original.

## Task

Implement `scale_invariant_product(W, X, s)` that returns a tuple `(orig, scaled)`:

* `orig` – the standard matrix product $W X$,
* `scaled` – the product after scaling weights by $s$ and activations by $1/s$, i.e. $(W\, s)\,(X / s)$.

The function must use only vectorised NumPy operations; no explicit Python loops are allowed.  The returned arrays should be of type `float64`.

```python
def scale_invariant_product(W: np.ndarray, X: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    ...
```

## Example

```python
import numpy as np
W = np.array([[1., 2.], [3., 4.]])
X = np.array([[5., 6.], [7., 8.]])
s = 0.5
orig, scaled = scale_invariant_product(W, X, s)
print(orig)
# [[19. 22.]
#  [43. 50.]]
print(scaled)
# [[19. 22.]
#  [43. 50.]]
```

## What the gate checks

The grader computes the *maximum absolute error* between `orig` and `scaled` produced by your implementation and the reference values computed with NumPy.  
The solution must satisfy

$$
\max_{i,j}\bigl|\, \text{orig}_{ij} - (W X)_{ij}\,\bigr|
   + \max_{i,j}\bigl|\, \text{scaled}_{ij} - ((W s)(X / s))_{ij}\,\bigr|
   \;\leq\; 10^{-6}.
$$

A correct implementation will produce an error of exactly $0$ for real‑valued inputs.
