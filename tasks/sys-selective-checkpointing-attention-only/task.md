## Context

Activation checkpointing trades computation for memory. Instead of storing every intermediate tensor from a forward pass, a backward pass can recompute selected parts of the graph.

For scaled dot-product attention, the forward computation is

$$
S = \frac{QK^\top}{\sqrt{d}},
$$

$$
P_{ij} = \frac{e^{S_{ij}}}{\sum_j e^{S_{ij}}},
$$

$$
O = PV,
$$

where $Q \in \mathbb{R}^{n \times d}$, $K \in \mathbb{R}^{m \times d}$, and $V \in \mathbb{R}^{m \times d}$.

A full checkpoint stores intermediates such as the attention scores $S$ and probabilities $P$. Selective checkpointing stores only the tensors needed to reconstruct the attention operation and recomputes the softmax path during backward. The mathematical goal is to preserve the same gradients while reducing saved activation memory.

The loss used by the grader is

$$
L = \sum_{i,j} O_{ij}G_{ij},
$$

where $G$ is the upstream gradient. The required outputs are the gradients

$$
\frac{\partial L}{\partial Q},\quad
\frac{\partial L}{\partial K},\quad
\frac{\partial L}{\partial V}.
$$

## Task

Implement:

```python
def attention_checkpoint(Q, K, V, G):
    ...
```

The inputs are `float64` NumPy arrays:

- `Q` has shape `(n, d)`
- `K` has shape `(m, d)`
- `V` has shape `(m, d)`
- `G` has shape `(n, d)`

Return a tuple:

```python
(dQ, dK, dV, reported_memory)
```

where the first three values are `float64` arrays with the same shapes as `Q`, `K`, and `V`. `reported_memory` is an integer number of bytes representing the activation memory used by your checkpoint strategy.

The implementation should perform attention-only selective checkpointing. Do not store the full attention matrix as a saved activation. Recompute attention intermediates when needed for gradients.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0, 3.0], [4.0, 5.0]])
G = np.ones((2, 2))

dQ, dK, dV, memory = attention_checkpoint(Q, K, V, G)

# dQ, dK, and dV are attention gradients.
# memory reports the checkpoint activation bytes.
```

## What the gate checks

The grader builds a NumPy oracle for the attention loss and computes reference gradients with central finite differences:

$$
\frac{\partial L}{\partial x}
\approx
\frac{L(x+h)-L(x-h)}{2h}.
$$

The returned gradients must have maximum absolute error at most $10^{-5}$ compared with the oracle.

The memory gate computes the activation bytes required by a full attention checkpoint and requires the reported checkpoint memory to be lower. A solution that computes correct gradients but stores the complete attention probability matrix does not pass.
