## Context

In linear-recurrence attention models (e.g., Mamba-2, GLA, DeltaNet), the recurrence operates over a **latent** representation that carries positional-agnostic semantic content. Because a rotary position embedding (RoPE) is position-dependent, it cannot be absorbed into the latent without breaking the linear recurrence structure.

The solution is a **decoupled rope head**: a small, separate head that receives standard RoPE rotations and is concatenated to the latent only at scoring time.

Let $q_{\text{rope}},\, k_{\text{rope}} \in \mathbb{R}^{B \times H \times N \times D_r}$ be the query and key rope-head tensors after RoPE, where $D_r$ is the rope-head dimension (commonly 64). Let $q_{\text{lat}},\, k_{\text{lat}} \in \mathbb{R}^{B \times H \times N \times D_l}$ be the latent tensors without RoPE. Here $B$ is the batch size, $H$ the number of heads, and $N$ the sequence length.

The final attention score between query position $i$ and key position $j$ is computed on the concatenated representation:

$$
Q = [q_{\text{lat}},\; q_{\text{rope}}] \in \mathbb{R}^{B \times H \times N \times D},
\qquad
K = [k_{\text{lat}},\; k_{\text{rope}}] \in \mathbb{R}^{B \times H \times N \times D},
$$

where $D = D_l + D_r$ and $[\,\cdot\,,\,\cdot\,]$ denotes concatenation along the last (head-dimension) axis. The scaled dot-product score matrix is

$$
S_{b,h,i,j} = \frac{1}{\sqrt{D}} \sum_{d=1}^{D} Q_{b,h,i,d} \; K_{b,h,j,d},
$$

followed by a softmax over the key dimension $j$:

$$
\hat{S}_{b,h,i,j} = \frac{\exp(S_{b,h,i,j})}{\sum_{j'} \exp(S_{b,h,i,j'})}.
$$

## Task

Implement `decoupled_rope_score(q_lat, k_lat, q_rope, k_rope)`:

```python
def decoupled_rope_score(
    q_lat: np.ndarray,   # shape (B, H, N, D_l)
    k_lat: np.ndarray,   # shape (B, H, N, D_l)
    q_rope: np.ndarray,  # shape (B, H, N, D_r)
    k_rope: np.ndarray,  # shape (B, H, N, D_r)
) -> np.ndarray:         # shape (B, H, N, N)
```

Steps:

1. Concatenate `q_lat` and `q_rope` along the last axis → `Q` of shape `(B, H, N, D)`.
2. Concatenate `k_lat` and `k_rope` likewise → `K`.
3. Compute the scaled dot-product scores $S = Q\,K^\top / \sqrt{D}$ (using `np.matmul`, which broadcasts over the leading batch and head dims).
4. Apply softmax over the last (key) axis.

Use vectorised NumPy only — no Python `for` loops. The output dtype must be `float64`.

## Example

```python
import numpy as np
B, H, N, D_l, D_r = 2, 3, 5, 64, 32
q_lat  = np.random.randn(B, H, N, D_l)
k_lat  = np.random.randn(B, H, N, D_l)
q_rope = np.random.randn(B, H, N, D_r)
k_rope = np.random.randn(B, H, N, D_r)

out = decoupled_rope_score(q_lat, k_lat, q_rope, k_rope)
# out.shape -> (2, 3, 5, 5)
```

## What the gate checks

One gate: **max_abs_err** $\le 10^{-4}$. The grader computes a NumPy reference oracle implementing the identical algorithm in `float64` and compares element-wise. Any correct vectorised implementation will produce a max absolute error well below $10^{-12}$; the $10^{-4}$ threshold leaves ample margin. A non-vectorised solution that loops in Python is also accepted as long as the result is numerically correct.
