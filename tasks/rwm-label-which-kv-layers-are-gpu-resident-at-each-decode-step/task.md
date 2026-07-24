## Context

In transformer decoding, key‑value (KV) tensors for each layer are often kept in GPU memory only while the layer is being processed or immediately after. A common strategy is *double buffering*: at any decode step the GPU holds the KV of the current layer and the next layer that will be needed in the following step. If we denote by $L$ the total number of layers and by $T$ the number of decoding steps, then during step $t$ (starting from $0$) the model attends to layer
$$i(t)=t \bmod L.$$
The resident set is therefore
$$\mathcal{R}(t)=\bigl\{\, i(t),\; (i(t)+1)\bmod L \,\bigr\}.$$

We want a function that, given $L$ and $T$, returns an $(L\times T)$ matrix $M$ where
$$M_{ij}= \begin{cases}
1 & \text{if layer } i \in \mathcal{R}(j),\\[4pt]
0 & \text{otherwise.}
\end{cases}$$

## Task

Implement `label_gpu_residency(L, T)`:

```python
def label_gpu_residency(L: int, T: int) -> np.ndarray:
    ...
```

It should return a NumPy array of shape `(L, T)` with integer values `0` or `1`. The function must run in $O(LT)$ time and use only NumPy operations; no explicit Python loops over layers or steps.

## Example

```python
import numpy as np
M = label_gpu_residency(4, 6)
print(M)
# [[1 0 0 0 0 1]
#  [1 1 0 0 0 0]
#  [0 1 1 0 0 0]
#  [0 0 1 1 1 0]]
```

Here $L=4$ layers and $T=6$ decode steps. At step $t=0$ the current layer is $0$, so layers $0$ and $1$ are resident; at step $t=5$ the current layer is $1$, so layers $1$ and $2$ are resident, etc.

## What the gate checks

The grader computes a reference matrix using the exact double‑buffer schedule described above and compares it elementwise with your output. The metric `exact_match` must equal `1.0`; any mismatch yields `0.0`.
