## Context

In transformer models the self‑attention mechanism uses a *key–value* cache to avoid recomputing all keys and values when decoding token by token.  
For each input token $x_t$ we compute

$$Q_t = x_t W_q,\qquad K_t = x_t W_k,\qquad V_t = x_t W_v,$$

where $W_q,W_k,W_v\in\mathbb{R}^{d_{\text{in}}\times d_k}$ are learned weight matrices.  
At decoding step $t$ the attention output is

$$
O_t \;=\;\sum_{i=0}^t \alpha_{ti}\,V_i,
\qquad
\alpha_{ti}
  = \frac{\exp\!\bigl(\frac{Q_t K_i^\top}{\sqrt{d_k}}\bigr)}
         {\displaystyle\sum_{j=0}^{t}\exp\!\bigl(\frac{Q_t K_j^\top}{\sqrt{d_k}}\bigr)} .
$$

The *KV cache* stores all $K_i$ and $V_i$ seen so far.  Incremental decoding appends the new key/value pair to this cache and then computes the attention output for the current token using **all** keys up to that point.

## Task

Implement a function with the following signature:

```python
def incremental_decode(embeddings: np.ndarray,
                       Wq: np.ndarray,
                       Wk: np.ndarray,
                       Wv: np.ndarray) -> np.ndarray:
    ...
```

* `embeddings` is an $(n, d_{\text{in}})$ array of token embeddings.  
* The function must return a float64 array of shape $(n, d_v)$ containing the incremental attention outputs $O_0,\dots,O_{n-1}$.

The implementation should maintain a cache of keys and values as tokens are processed.  Do **not** recompute all keys/values from scratch at each step; instead reuse the cached tensors.

## Example

```python
import numpy as np

rng = np.random.default_rng(42)
embeddings = rng.standard_normal((3, 4))
Wq = rng.standard_normal((4, 2))
Wk = rng.standard_normal((4, 2))
Wv = rng.standard_normal((4, 3))

outputs = incremental_decode(embeddings, Wq, Wk, Wv)
print(outputs.shape)   # (3, 3)
```

## What the gate checks

The grader computes a reference implementation that performs the full pre‑fill attention in one pass.  
Your output must match this reference within a maximum absolute error of $10^{-5}$:

$$\max_{i,j}\bigl|\,O^{\text{your}}_{ij}-O^{\ref}_{ij}\bigr|\;\le 1\times10^{-5}.$$

The gate metric is named `max_abs_err`.  
If the error exceeds this threshold, the solution fails.
