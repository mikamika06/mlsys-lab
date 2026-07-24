## Context

Standard multi-head attention (MHA) splits queries, keys, and values into $h$
independent heads.  For query head $i$:

$$\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_i \, K_i^\top}{\sqrt{d_k}}\right) V_i$$

where $Q_i, K_i, V_i \in \mathbb{R}^{S \times d_k}$ and $d_k$ is the per-head
dimension.

Multi-query attention (MQA) replaces all $h$ KV heads with a **single** shared
KV head.  Every query head attends to the same keys and values:

$$\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_i \, K_1^\top}{\sqrt{d_k}}\right) V_1
\qquad \forall\; i \in \{1, \ldots, h\}$$

In tensor form the inputs are

$$Q \in \mathbb{R}^{B \times h \times S \times d_k}, \quad
  K \in \mathbb{R}^{B \times 1 \times S \times d_k}, \quad
  V \in \mathbb{R}^{B \times 1 \times S \times d_k}$$

and NumPy broadcasting handles the head dimension automatically — no explicit
expansion is required.

MQA reduces the KV cache by a factor of $h$ during autoregressive inference.
The trade-off is a small quality loss for significant memory and bandwidth
savings.

## Task

Implement `mha_single_kv_head(Q, K, V)`:

```python
import numpy as np

def mha_single_kv_head(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention with a single shared KV head."""
    ...
```

Inputs are all `float64` NumPy arrays:

| Argument | Shape | Description |
|----------|-------|-------------|
| `Q` | $(B, h, S, d_k)$ | Queries with $h$ heads |
| `K` | $(B, 1, S, d_k)$ | Keys with a single head |
| `V` | $(B, 1, S, d_k)$ | Values with a single head |

Requirements:

1. Compute the attention score matrix $\frac{QK^\top}{\sqrt{d_k}}$ using
   `np.matmul` (or equivalent) so that the single KV head broadcasts across
   all $h$ query heads.
2. Apply numerically stable softmax over the last axis (subtract the row-max
   before exponentiation).
3. Multiply the attention weights by $V$, again relying on broadcasting.
4. Return the result as a `float64` array of shape $(B, h, S, d_k)$.

Do **not** manually expand $K$ or $V$ to $h$ copies; use broadcasting.

## Example

```python
import numpy as np

rng = np.random.RandomState(0)
B, h, S, d = 2, 4, 8, 16
Q = rng.randn(B, h, S, d)
K = rng.randn(B, 1, S, d)
V = rng.randn(B, 1, S, d)

out = mha_single_kv_head(Q, K, V)
assert out.shape == (2, 4, 8, 16)
```

## What the gate checks

One gate: **`max_abs_err`**.

The grader constructs several `(B, h, S, d_k)` test cases with fixed random
seeds.  It computes a reference answer by explicitly repeating $K$ and $V$
along axis 1 to $h$ copies, then running ordinary scaled dot-product
attention.  Your output is compared element-wise against this reference.

The gate passes when the maximum absolute element-wise difference satisfies

$$\max_{b,i,s,j} \bigl| \mathrm{out}_{b,i,s,j} - \mathrm{ref}_{b,i,s,j} \bigr| < 10^{-5}.$$

Both a broadcasting implementation and an explicit-expansion implementation
produce identical numerical results and pass the gate.
