## Context

Long-context attention systems often keep a small set of initial "sink" tokens together with a recent sliding window when the available key/value cache budget is limited.

Given query, key, and value matrices $Q, K, V \in \mathbb{R}^{n \times d}$, full attention output is

$$
O = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

A compressed attention cache with budget $B=k+w$ keeps the first $k$ tokens as sinks and the last $w$ tokens as a recent window. For a chosen split, the retained indices are

$$
S_k = \{0,1,\dots,k-1\} \cup \{n-w,n-w+1,\dots,n-1\}.
$$

The approximate output is computed by attending only over the retained keys and values:

$$
\hat{O}_k =
\operatorname{softmax}\left(\frac{QK_{S_k}^{\top}}{\sqrt{d}}\right)V_{S_k}.
$$

The split should minimize the difference from full attention. The oracle evaluates every possible split and selects

$$
k^* = \arg\min_{1 \le k < B} \lVert O-\hat{O}_k\rVert_F^2 .
$$

## Task

Implement `optimize_sink_window_split(Q, K, V, B)`.

The function receives three NumPy arrays of shape $(n,d)$ and an integer cache budget $B$ where $1 < B < n$. It must return the integer sink size $k$ that gives the lowest squared Frobenius error against full attention when the retained cache size is fixed to $B$.

The implementation must:

1. Compute the full attention output in `float64`.
2. Evaluate every candidate split $k \in [1,B-1]$.
3. Return the index $k$ with the smallest attention output error.
4. Use the same attention definition as the context, including the scale factor $\sqrt{d}$.

No approximation or heuristic search is allowed.

## Example

```python
import numpy as np

Q = np.array([[1., 0.], [0., 1.], [1., 1.], [0., -1.]])
K = Q.copy()
V = np.arange(8, dtype=np.float64).reshape(4, 2)

k = optimize_sink_window_split(Q, K, V, 3)
# k is the best number of sink tokens among:
# k=1, window=2
# k=2, window=1
```

## What the gate checks

The gate computes a NumPy reference implementation that performs the full attention calculation, sweeps all valid sink sizes, and selects the true minimum-error split. The returned `k` must exactly match this oracle-selected argmin index on multiple attention inputs.
