## Context

Scaled dot‑product attention is defined for a batch of queries $Q \in \mathbb{R}^{B\times N_q\times d_k}$, keys $K \in \mathbb{R}^{B\times N_k\times d_k}$ and values $V \in \mathbb{R}^{B\times N_v\times d_v}$.  
The attention scores are

$$
S = \frac{Q\,K^\top}{\sqrt{d_k}}\;,
$$

and the softmax is applied over the last axis of $S$ to obtain weights $W$.  The output is then

$$
O = W\,V .
$$

Grouped‑query attention (GQA) partitions the queries into groups, each group attending only to a subset of key/value pairs.  
When the number of key/value pairs per head equals one ($n_{\text{kv}}=1$), every query in a group shares the same single key/value pair – this is precisely multi‑query attention (MQA).  In that limit the computation reduces to standard scaled dot‑product attention, but we still expose it as a separate function for clarity.

## Task

Implement `gqa_limit_nkv_1(Q, K, V)`:

```python
def gqa_limit_nkv_1(
    Q: list[list[list[float]]],
    K: list[list[list[float]]],
    V: list[list[list[float]]],
) -> list[list[list[float]]]:
    ...
```

The function receives three 3‑D list of shapes  
$Q \in \mathbb{R}^{B\times N_q\times d_k}$, $K \in \mathbb{R}^{B\times N_k\times d_k}$ and $V \in \mathbb{R}^{B\times N_v\times d_v}$.  
It must return the attention output of shape $(B,\,N_q,\,d_v)$ computed exactly as described in the context section.  Use only Python operations; no explicit Python loops.

## Example

```python
Q = [[[1.,0.],[0.,1.]]]          # shape (1,2,2)
K = [[[1.,0.],[0.,1.]]]          # shape (1,2,2)
V = [[[1.,2.],[3.,4.]]]          # shape (1,2,2)

O = gqa_limit_nkv_1(Q, K, V)
print(O)  # [[[1.6604769013466862, 2.6604769013466862], [2.3395230986533138, 3.3395230986533138]]]
```

## What the gate checks

The grader generates several random test cases and compares your output to a reference implementation computed on‑the‑fly.  
It reports the maximum absolute error

$$
\max_{i,j,k} |\,O_{\text{cand}}(i,j,k) - O_{\text{ref}}(i,j,k)\,|
$$

and requires this value to be at most $10^{-6}$.
