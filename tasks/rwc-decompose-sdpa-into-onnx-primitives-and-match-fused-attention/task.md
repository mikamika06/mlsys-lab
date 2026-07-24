## Context

Scaled Dot-Product Attention (SDPA) is the core computation inside every
Transformer layer.  Given query, key and value matrices
$Q \in \mathbb{R}^{L_q \times d_k}$,
$K \in \mathbb{R}^{L_k \times d_k}$,
$V \in \mathbb{R}^{L_k \times d_v}$, the attention output is

$$\text{SDPA}(Q, K, V) \;=\; \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

where $M$ is an optional additive mask — positions that should be blocked
receive large negative values such as $-\infty$.

Modern runtimes (ONNX Runtime, TensorRT) fuse this entire expression into a
single CUDA kernel.  ONNX opset 22+ introduces a native
`ScaledDotProductAttention` node, but graphs targeting older opsets must
decompose the computation into primitive ONNX ops that every runtime already
supports:

1. **MatMul** — raw scores $S = QK^\top$, shape $(L_q, L_k)$
2. **Mul** — scale the logits $S \leftarrow S / \sqrt{d_k}$
3. **Add** — apply the mask $S \leftarrow S + M$ (skip when mask is absent)
4. **Softmax** — compute attention weights
   $W = \text{softmax}(S, \text{axis}=-1)$
5. **MatMul** — context output $\text{out} = WV$, shape $(L_q, d_v)$

Each step maps 1 : 1 to an ONNX op (`MatMul`, `Mul`, `Add`, `Softmax`).
The decomposed sub-graph must produce results that are numerically
indistinguishable from the fused kernel.

## Task

Implement `decompose_sdpa`:

```python
import numpy as np

def decompose_sdpa(
    Q: np.ndarray,                        # (..., Lq, Dk)  float64
    K: np.ndarray,                        # (..., Lk, Dk)  float64
    V: np.ndarray,                        # (..., Lk, Dv)  float64
    mask: np.ndarray | None = None,       # broadcastable to (..., Lq, Lk)
    scale: float | None = None,           # default 1/√Dk
) -> np.ndarray:
    ...
```

The function must perform the five ONNX-compatible primitive steps listed above
using only `numpy`.  It must handle arbitrary leading batch dimensions via
NumPy broadcasting.  When `scale` is `None`, compute it as
$1 / \sqrt{D_k}$ where $D_k = Q.shape[-1]$.  When `mask` is `None`, skip
the Add step entirely.  The output shape must be `(..., Lq, Dv)` with dtype
`float64`.

Numerically stable softmax: subtract the row maximum before exponentiating
to avoid overflow.

## Example

```python
import numpy as np
rng = np.random.default_rng(42)
Q = rng.standard_normal((3, 8))
K = rng.standard_normal((5, 8))
V = rng.standard_normal((5, 6))

# Full (non-causal) attention, scale = 1/sqrt(8)
out = decompose_sdpa(Q, K, V)
print(out.shape)  # (3, 6)

# Causal mask — future positions get −∞
Lq, Lk = 3, 5
causal = np.triu(np.full((Lq, Lk), -np.inf), k=1)
out_causal = decompose_sdpa(Q, K, V, mask=causal)
print(out_causal.shape)  # (3, 6)
```

## What the gate checks

The grader builds a NumPy oracle that evaluates the canonical
formula $\text{softmax}(QK^\top / \sqrt{d_k} + M)\,V$ with numerically
stable softmax and compares its output to yours via `max_abs_err`.

Six scenarios are evaluated:

1. **Square, no mask** — $(L_q = L_k)$, default scale
2. **Explicit additive mask** — select entries set to $-\infty$
3. **Causal mask** — lower-triangular with $-\infty$ above the diagonal, $L_q = L_k$
4. **Causal mask, rectangular** — $L_q \neq L_k$ (cross-attention style)
5. **Custom scale** — caller supplies a non-default `scale` value
6. **Batched 3-D inputs** — leading batch dimension, non-square causal mask

The gate passes when the worst-case `max_abs_err` across all six scenarios is
$\le 10^{-6}$.
