## Context

Attention mechanisms often restrict which keys each query can see. A sliding-window
causal attention mask allows a query position $q$ to attend only to recent previous
positions $k$.

A production mask can combine multiple constraints. For a document-packed batch,
tokens from different documents must not attend to each other. The allowed index
set is:

$$
M_{q,k} =
(k \le q) \land (q-k < w) \land (\mathrm{doc\_id}[q] = \mathrm{doc\_id}[k]),
$$

where $w$ is the sliding window size.

Given query vectors $Q$, key vectors $K$, and value vectors $V$, masked attention is
computed from

$$
S = \frac{QK^\top}{\sqrt{d}},
$$

then scores where $M_{q,k}$ is false are replaced with $-\infty$. The output is

$$
O = \mathrm{softmax}(S)V.
$$

The mask is commonly implemented as a compact `mask_mod` function because it can
be combined with other attention rules without materializing unnecessary masks.

## Task

Implement `sliding_window_document_attention(Q, K, V, doc_ids, window)`.

Arguments:

- `Q`: a NumPy array of shape $(n,d)$ containing query vectors.
- `K`: a NumPy array of shape $(n,d)$ containing key vectors.
- `V`: a NumPy array of shape $(n,m)$ containing value vectors.
- `doc_ids`: a 1-D integer NumPy array of length $n$ assigning each token to a document.
- `window`: a positive integer sliding-window size.

Return a tuple `(output, mask)`.

`output` must be the masked attention result with dtype `float64` and shape
$(n,m)`. `mask` must be a boolean array of shape $(n,n)$ where `mask[q,k]` is
true exactly when key position $k$ is visible to query position $q`.

Use NumPy operations to implement the mask and attention computation.

## Example

```python
import numpy as np

Q = np.array([[1., 0.], [0., 1.], [1., 1.]])
K = Q.copy()
V = np.array([[1., 2.], [3., 4.], [5., 6.]])
doc_ids = np.array([0, 0, 1])

out, mask = sliding_window_document_attention(Q, K, V, doc_ids, 2)

# mask allows:
# query 0 -> key 0
# query 1 -> keys 0, 1
# query 2 -> key 2 only because it is a different document
```

## What the gate checks

The gate builds a NumPy oracle that creates the causal sliding-window document mask,
applies it to attention logits, and computes the result in `float64`.

`mask_match` must be exactly $1.0$, meaning the returned boolean mask has the same
index set as the oracle. `max_abs_err` is the maximum absolute difference between
the returned output and the oracle output and must satisfy
$\mathrm{max\_abs\_err} \le 10^{-5}$.
