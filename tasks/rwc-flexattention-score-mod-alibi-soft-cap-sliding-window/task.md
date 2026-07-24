## Context

**FlexAttention** is a generalization of scaled dot-product attention where a user-provided `score_mod` function is applied to the raw attention scores before the softmax:

$$\text{out} = \text{softmax}\!\bigl(\text{score\_mod}(S / \sqrt{d})\bigr)\, V, \quad S_{ij} = q_i \cdot k_j$$

Three common score modifications:

1. **ALiBi** (Attention with Linear Biases): adds a position-dependent linear bias $-m \cdot |i - j|$ where $m$ is a per-head slope. This encodes relative position without learned embeddings.

2. **Soft-cap**: applies a tanh saturation $c \cdot \tanh(s / c)$ to prevent score explosion, used in Gemma 2 / Gemini.

3. **Sliding-window mask**: sets $s_{ij} = -\infty$ when $|i - j| > w$, restricting attention to a local window of radius $w$.

## Task

Implement `flex_attention(Q, K, V, score_mod)`:

```python
def flex_attention(Q, K, V, score_mod):
    ...
```

- `Q`, `K`, `V`: 2-D float32 NumPy arrays of shape $(N, d)$.
- `score_mod`: a callable with signature `score_mod(scores, query_idx, key_idx)` where:
  - `scores`: float64 array of shape $(N, N)$ — the raw scaled scores $QK^\top / \sqrt{d}$.
  - `query_idx`: integer row indices array of shape $(N, 1)$.
  - `key_idx`: integer column indices array of shape $(1, N)$.
  - Returns the modified score matrix of the same shape.
- Returns: float32 array of shape $(N, d)$.

Apply `score_mod` to the full scaled score matrix, then softmax row-wise, then multiply by $V$.

## Example

```python
import numpy as np
N, d = 4, 8
Q = np.random.randn(N, d).astype(np.float32)
K = np.random.randn(N, d).astype(np.float32)
V = np.random.randn(N, d).astype(np.float32)

# ALiBi with slope 0.5
def alibi_mod(scores, qi, ki): return scores - 0.5 * np.abs(qi - ki)

out = flex_attention(Q, K, V, alibi_mod)
# out.shape == (4, 8)
```

## What the gate checks

The grader tests three `score_mod` variants — ALiBi, tanh soft-cap, and sliding-window mask — against a NumPy reference. The **max_abs_err** across all variants must be $\le 10^{-5}$.
