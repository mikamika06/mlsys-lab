## Context

Scaled dot‑product attention is the core of transformer models.  
Given queries $Q \in \mathbb{R}^{B\times H\times T_q\times d_k}$, keys $K \in \mathbb{R}^{B\times H\times T_k\times d_k}$ and values $V \in \mathbb{R}^{B\times H\times T_k\times d_v}$ the attention logits are

$$
\text{logits} = \frac{QK^\top}{\sqrt{d_k}} .
$$

An optional *causal* mask forces each position to attend only to earlier positions by adding $-\infty$ above the main diagonal.  
A float mask can be added element‑wise to the logits, and a boolean mask can zero out selected entries by setting them to $-\infty$.  
After masking, a row‑wise softmax produces attention weights $\alpha$, which are then used to aggregate values:

$$
\alpha = \operatorname{softmax}(\text{logits}), \qquad
O = \alpha V .
$$

The implementation must faithfully reproduce these semantics using only NumPy vectorised operations.

## Task

Implement the function

```python
def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    causal: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
```

* `Q`, `K`, `V` have matching batch and head dimensions.  
* `mask` may be `None`, a float array broadcastable to the logits shape, or a boolean array of the same shape.  
* If `causal=True` a causal mask is applied in addition to any provided `mask`.  
The function should return a tuple `(output, weights)` where `output` has the same shape as the values aggregated and `weights` contains the softmax probabilities.

All computations must be performed with NumPy only; no Python loops are allowed. The result must use `float64`.

## Example

```python
import numpy as np

Q = np.array([[[[1., 0.], [0., 1.]]]])   # shape (1,1,2,2)
K = Q.copy()
V = np.array([[[[1., 2.], [3., 4.]]]])

out, w = scaled_dot_product_attention(Q, K, V)

print(out)   # [[[[2.5, 3.5],[2.5, 3.5]]]]
print(w)     # [[[[0.5, 0.5],[0.5, 0.5]]]]
```

## What the gate checks

The grader computes a NumPy reference implementation for several random test cases and compares your output against it using the metric `max_abs_err`.  
Both the returned attention weights and the aggregated values must satisfy

$$
\max_{i,j} |\, \text{your}_{ij} - \text{reference}_{ij}\,| \le 10^{-5}.
$$

If either error exceeds this threshold the solution fails. No timing or line‑count checks are performed.
