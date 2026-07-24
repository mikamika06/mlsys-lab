## Context

Attention computes a weighted combination of value vectors. Given queries $Q \in \mathbb{R}^{n \times d}$, keys $K \in \mathbb{R}^{n \times d}$, and values $V \in \mathbb{R}^{n \times m}$, the output is

$$
O = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

The matrix of attention scores can be large, so implementations often process keys and values in tiles instead of materializing the full $n \times n$ score matrix.

A streaming softmax keeps a running maximum $M$, normalizer $L$, and accumulated output $O$. When a new score block is processed, the old and new contributions are rescaled so the final result is independent of the tile size. For a score vector $x$,

$$
\operatorname{softmax}(x)_i = \frac{e^{x_i-\max(x)}}{\sum_j e^{x_j-\max(x)}} .
$$

Correct online softmax must produce the same result whether the key/value sequence is processed one element at a time or as one complete block.

## Task

Implement `streaming_attention(Q, K, V, block_size)`.

The function takes:

- `Q`: a NumPy array of shape $(n, d)$.
- `K`: a NumPy array of shape $(n, d)$.
- `V`: a NumPy array of shape $(n, m)$.
- `block_size`: positive integer tile size.

Return the attention output as a NumPy array of shape $(n, m)$ with dtype `float64`.

Use a streaming softmax approach over key/value tiles. The output should not depend on the chosen `block_size`. The implementation may use NumPy matrix operations inside each tile, but should not construct the complete attention matrix for all $n$ keys.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0], [4.0]])

out = streaming_attention(Q, K, V, 1)
# out is approximately:
# [[2.75447669],
#  [3.24552331]]
```

## What the gate checks

The gate computes a NumPy full-matrix softmax attention oracle and compares it with the submitted implementation.

It runs the implementation twice on the same inputs: once with `block_size = 1` and once with `block_size = n`. The reported metric is the largest absolute difference across these results and the oracle:

$$
\max |O_{\mathrm{candidate}} - O_{\mathrm{oracle}}| .
$$

The gate requires this value to be at most $10^{-7}$. A solution whose result changes with tile size fails because it does not implement numerically stable streaming softmax.
