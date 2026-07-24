## Context

The standard softmax function  

$$
\sigma(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)}
$$

can overflow for large $x_i$. The numerically stable variant subtracts the maximum:

$$
m = \max_j x_j,\qquad 
p_i = \exp(x_i - m),\qquad 
\sigma(x_i) = \frac{p_i}{\sum_j p_j}.
$$

For very long sequences (e.g., transformer attention scores), waiting for the full sum before normalising forces storing the entire exponentiated vector in high-bandwidth memory. Flash Attention uses an *online blockwise recurrence*:

Initialize $m_0 = -\infty$, $d_0 = 0$.  
For a new block $x^{(k)}$ with $m^{(k)} = \max x^{(k)}$:

$$
m_{\text{new}} = \max(m_{\text{prev}}, m^{(k)}),
$$

$$
d_{\text{new}} = d_{\text{prev}} \cdot \exp(m_{\text{prev}} - m_{\text{new}}) + \sum_{i} \exp(x_i^{(k)} - m_{\text{new}}).
$$

After processing all blocks, the final softmax for any element $x_i$ is  

$$
\sigma(x_i) = \frac{\exp(x_i - m_{\text{final}})}{d_{\text{final}}}.
$$

This recurrence is algebraically identical to the one‑shot version, but it can be executed in an online fashion, never materialising the full sum before partial contributions are known.

## Task

Implement `blockwise_softmax(logits, block_size)`:

```python
def blockwise_softmax(logits: np.ndarray, block_size: int) -> np.ndarray:
```

The function receives a 1D array of logits and a positive integer `block_size`.  
It computes the softmax of the entire array using the online blockwise recurrence described above. The result must be an array of the same shape as `logits` containing the softmax probabilities.

Processing must proceed in contiguous, non‑overlapping blocks of size `block_size` (the last block may be shorter). All operations must be vectorised at the block level; Python‑level loops over individual elements within a block are forbidden.

## Example

```python
import numpy as np
logits = np.array([2.0, -1.0, 0.0, 3.0])
probs = blockwise_softmax(logits, 2)
# Reference (stable one‑shot softmax):
# m = 3.0
# sum_exp = exp(-1) + exp(-4) + exp(-3) + exp(0) ≈ 1.435982
# probs ≈ [0.256200, 0.012729, 0.034661, 0.696411]
```

The returned `probs` should match the reference within $1 \times 10^{-6}$ (maximum absolute per‑element difference).

## What the gate checks

The grader computes the stable one‑shot softmax on the full array as the reference, then compares against the student output using the maximum absolute element‑wise difference:

$$
\text{max\_abs\_err} = \max_i |\hat{p}_i - p_i|.
$$

The gate passes when $\text{max\_abs\_err} \le 10^{-6}$ across all test cases. The grader itself computes the reference with a real NumPy call — no values are hard‑coded.
