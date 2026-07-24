## Context

In multi‑query attention (MQA) the number of key/value heads can be smaller than the number of query heads.  
When there is a single KV head ($n_{\text{kv}}=1$), that head’s key $K\in\mathbb{R}^{d_k}$ and value $V\in\mathbb{R}^{d_v}$ are shared across all query heads.  

For a batch of queries arranged as
$$Q \in \mathbb{R}^{n_q\times h\times d_k},$$
the attention scores for head $i$ are computed by the dot product
$$s_{qi} = \frac{Q_{q i}\cdot K}{\sqrt{d_k}},$$
followed by a softmax over the single key dimension.  
Because there is only one key, $\operatorname{softmax}(s_{qi})=1$ for every query and head, so the output of the attention layer is simply the broadcast value $V$, repeated for each query.

The goal of this task is to implement a function that performs exactly this operation in a fully vectorised way.

## Task

Implement `mqa_single_kv_broadcast(Q, K, V)`:

```python
def mqa_single_kv_broadcast(Q: np.ndarray,
                            K: np.ndarray,
                            V: np.ndarray) -> np.ndarray:
    ...
```

* `Q` has shape `(n_q, h, d_k)`
* `K` has shape `(1, d_k)`
* `V` has shape `(1, d_v)`

The function must return an array of shape `(n_q, h, d_v)` containing the attention output for every query and head.  
All computations should be performed with NumPy only; no explicit Python loops are allowed.

## Example

```python
import numpy as np

Q = np.array([[[1., 0.], [0., 1.]],
              [[-1., 2.], [3., -4.]]])          # shape (2, 2, 2)
K = np.array([[0.5, 0.5]])                       # shape (1, 2)
V = np.array([[1., 2.]])                         # shape (1, 2)

out = mqa_single_kv_broadcast(Q, K, V)
print(out.shape)   # (2, 2, 2)
print(out)
```

Output:

```
[[[1. 2.]
  [1. 2.]]

 [[1. 2.]
  [1. 2.]]]
```

The value `V` is broadcast to every query and head.

## What the gate checks

* **Metric**: `max_abs_err` – the maximum absolute difference between your output and a NumPy oracle.
* **Threshold**: $10^{-5}$, i.e. $\displaystyle \max_{i,j,k}\lvert\hat{y}_{ijk}-y_{ijk}\rvert \le 10^{-5}$.

The oracle implements the same mathematical definition as above; it is recomputed for each test case so no hard‑coded expected values are used.  A correct implementation will therefore pass all gates, while a broken one (e.g., missing the $\sqrt{d_k}$ scaling or failing to broadcast across heads) will fail.
