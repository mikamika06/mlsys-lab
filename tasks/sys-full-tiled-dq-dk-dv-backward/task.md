## Context

FlashAttention computes attention without materializing the full attention matrix. In the forward pass, scaled dot-product attention is

$$
S = \frac{QK^\top}{\sqrt{d}},
$$

$$
P_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})},
$$

and

$$
O = PV.
$$

The log-sum-exp values from the forward pass are

$$
LSE_i = \log \sum_j \exp(S_{ij}).
$$

The backward pass can recompute probability tiles from $Q$, $K$, and $LSE$. Given an output gradient $dO$, first compute

$$
D_i = \sum_j dO_{ij} O_{ij}.
$$

For a probability tile $P$, the gradient is

$$
dS = P \odot (dO V^\top - D),
$$

where $D$ is broadcast across the columns. The parameter gradients are

$$
dQ = \frac{dS K}{\sqrt{d}},
$$

$$
dK = \frac{dS^\top Q}{\sqrt{d}},
$$

$$
dV = P^\top dO.
$$

A tiled implementation processes blocks of rows and columns instead of creating the complete $n \times n$ attention matrix.

## Task

Implement `flash_backward`:

```python
def flash_backward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], O: list[list[float]], LSE: list[float], dO: list[list[float]], tile_size: int=32) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    ...
```

The inputs are list:

- `Q` has shape $(n, d)$.
- `K` has shape $(n, d)$.
- `V` has shape $(n, dv)$.
- `O` has shape $(n, dv)$.
- `LSE` has shape $(n,)$.
- `dO` has shape $(n, dv)$.

Return a tuple `(dQ, dK, dV)` with the same shapes as `Q`, `K`, and `V`. The implementation should recompute attention probabilities from `Q`, `K`, and `LSE`, and accumulate the three gradients using tiles controlled by `tile_size`.

The returned arrays must use floating point values.

## Example

```python

Q = [[1.0, 0.0], [0.0, 1.0]]
K = [[1.0, 0.0], [0.0, 1.0]]
V = [[2.0], [3.0]]
S = [[sum(q * k for q, k in zip(row_q, col_k)) / (2.0 0.5) for col_k in zip(*K)] for row_q in Q]
LSE = [math.log(sum(math.exp(x) for x in row)) for row in S]
O = [[sum(math.exp(S[i][j] - LSE[i]) * V[j][k] for j in range(len(V))) for k in range(len(V[0]))] for i in range(len(S))]
dO = [[1.0 for _ in row] for row in O]

dQ, dK, dV = flash_backward(Q, K, V, O, LSE, dO)
```

The result should match the gradients from the mathematical backward equations.

## What the gate checks

The gate builds random small attention problems, computes the reference gradients with a Python implementation of the same backward equations, and compares all returned tensors.

The reported metric is

$$
\max(|dQ-dQ_{ref}|, |dK-dK_{ref}|, |dV-dV_{ref}|).
$$

The value must be below $10^{-4}$. A solution that uses an incorrect softmax reconstruction, misses the $D_i$ correction term, or produces incomplete tiled accumulation will fail.
