## Context

Scaled dot‑product attention is a core building block of transformer models.  
Given query, key and value tensors $Q \in \mathbb{R}^{B\times N_q\times d_k}$,
$K \in \mathbb{R}^{B\times N_k\times d_k}$ and
$V \in \mathbb{R}^{B\times N_k\times d_v}$, the attention output is

$$
\operatorname{Attention}(Q,K,V)
  = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V .
$$

In production systems it is common to offload the large key/value tensors to CPU memory or disk, then reload them when needed.  The offloading step must be *lossless*: after a round‑trip the attention result should match that of an in‑memory computation up to machine precision.

## Task

Implement the function `offload_attention`:

```python
def offload_attention(q: np.ndarray,
                      k: np.ndarray,
                      v: np.ndarray) -> np.ndarray:
    ...
```

The function must:

1. Serialize the key and value tensors to a bytes buffer (e.g., using `numpy.savez_compressed`) and then reload them.
2. Compute scaled dot‑product attention on the reloaded tensors, exactly as in the formula above.
3. Return the resulting tensor with dtype `float64`.

The function should work for arbitrary batch size $B$, query length $N_q$, key/value length $N_k$ and feature dimensions $d_k$, $d_v$.  No Python loops are required; vectorised NumPy operations are sufficient.

## Example

```python
import numpy as np
q = np.array([[[1., 0.], [0., 1.]]])          # shape (1,2,2)
k = np.array([[[1., 0.], [0., 1.], [1., 1.]]])# shape (1,3,2)
v = np.array([[[1., 0.], [0., 1.], [1., 1.]]])# shape (1,3,2)

out = offload_attention(q, k, v)
print(out.shape)   # (1, 2, 2)
```

The output should match the result of a direct NumPy implementation of attention.

## What the gate checks

The grader computes an oracle attention matrix using `float64` arithmetic and compares it to the student's output with the scorer `max_abs_err`.  
The solution must satisfy

$$
\mathrm{max\_abs\_err} \le 10^{-6}.
$$

Any loss of precision (e.g., due to incorrect dtype handling or missing scaling) will cause the gate to fail.
