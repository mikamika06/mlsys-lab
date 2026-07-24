## Context

Scaled dot-product attention computes a weighted combination of value vectors from query and key vectors. For queries $Q \in \mathbb{R}^{n \times d}$ and keys $K \in \mathbb{R}^{n \times d}$, the logits are

$$
L = \frac{QK^\top}{\sqrt{d}} .
$$

An additive attention bias $B \in \mathbb{R}^{n \times n}$ modifies the logits before normalization:

$$
P = \operatorname{softmax}(L + B).
$$

The attention output is

$$
O = PV,
$$

where $V \in \mathbb{R}^{n \times d_v}$.

Materializing $L$ and $P$ requires storing $O(n^2)$ temporary tensors. Production implementations avoid this by processing rows in tiles. For a query tile, the algorithm streams over key/value tiles, keeps running softmax normalization statistics, and accumulates the output without creating the full attention matrix.

For a row of logits $x$, the numerically stable softmax uses

$$
m = \max_i x_i,\qquad
p_i = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}}.
$$

The tiled algorithm extends this idea by combining partial maxima and partial sums as more key tiles are processed.

## Task

Implement `mem_efficient_attention(Q, K, V, attn_bias, block_size=64)`:

```python
def mem_efficient_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    attn_bias: np.ndarray,
    block_size: int = 64,
) -> np.ndarray:
    ...
```

The inputs are float arrays with shapes:

- `Q`: $(n, d)$
- `K`: $(n, d)$
- `V`: $(n, d_v)$
- `attn_bias`: $(n, n)$

Return the attention output with shape $(n, d_v)$.

Compute the same result as dense attention with

$$
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}} + B\right)V
$$

in float64 arithmetic. The implementation must not create an intermediate $n \times n$ logits or probability matrix. Use tiled processing over the key dimension.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0, 0.0], [0.0, 3.0]])
B = np.zeros((2, 2))

O = mem_efficient_attention(Q, K, V, B, block_size=1)
```

The result matches the dense attention calculation while avoiding a full attention matrix.

## What the gate checks

The numeric gate computes a NumPy float64 oracle from dense attention with the additive bias included and requires the implementation output to satisfy

$$
\max_i |O_i-\hat{O}_i| < 10^{-5}.
$$

The memory gate runs the implementation on a larger input and checks that it does not allocate a full extra $n \times n$ float64 attention buffer. The provided bias matrix is allowed as input storage; temporary dense logits or probabilities are not.
