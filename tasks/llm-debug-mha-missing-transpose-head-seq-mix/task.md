## Context

Multi-head attention splits the model dimension into multiple heads. For an input
tensor $X \in \mathbb{R}^{B \times S \times E}$, where $B$ is the batch size,
$S$ is the sequence length, and $E$ is the embedding size, the projected queries,
keys, and values are reshaped into heads.

With $H$ heads and head dimension $d$, the embedding size satisfies
$E = H d$. The attention computation for each head is

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

The tensor layout after splitting heads should be

$$
(B,S,E) \rightarrow (B,S,H,d) \rightarrow (B,H,S,d).
$$

The transpose is required because attention is computed independently for each
head across the sequence dimension. A common bug is reshaping directly to
$(B,H,S,d)$ without moving the axes. This silently mixes the head and sequence
dimensions and produces incorrect outputs while keeping all tensor shapes valid.

## Task

Implement `mha_forward(X, Wq, Wk, Wv, Wo, num_heads)`.

The function receives:

- `X`, a NumPy array with shape $(B,S,E)$.
- `Wq`, `Wk`, and `Wv`, projection matrices with shape $(E,E)$.
- `Wo`, an output projection matrix with shape $(E,E)$.
- `num_heads`, the number of attention heads $H$.

Return the output tensor with shape $(B,S,E)$.

Use the standard multi-head attention layout:

1. Project the input with $Wq$, $Wk`, and $Wv`.
2. Split the embedding dimension into heads.
3. Transpose from $(B,S,H,d)$ to $(B,H,S,d)$ before computing attention.
4. Apply scaled dot-product attention.
5. Merge heads back to $(B,S,E)$ and apply $Wo$.

The implementation may use NumPy operations only.

## Example

```python
import numpy as np

X = np.zeros((2, 4, 8))
W = np.eye(8)

Y = mha_forward(X, W, W, W, W, 2)
# Y has shape (2, 4, 8)
# All values are zero.
```

## What the gate checks

The gate computes a NumPy reference implementation of multi-head attention and
compares the submitted output using maximum absolute error.

The returned value must satisfy

$$
\max_i |y_i-\hat{y}_i| < 10^{-5}.
$$

The tests include multiple batches, sequence lengths, and head counts so a
missing transpose that mixes the head and sequence axes does not pass.
