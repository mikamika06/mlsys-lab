## Context

In disaggregated inference, a prefill worker computes the key and value cache for a sequence, serializes that KV state, and transfers it to a decode worker. The decode worker reconstructs the cache and computes attention for new query tokens.

For a query matrix $Q \in \mathbb{R}^{m \times d}$, key matrix $K \in \mathbb{R}^{n \times d}$, and value matrix $V \in \mathbb{R}^{n \times h}$, scaled dot-product attention is

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

The prefill stage produces $K$ and $V$. A production system must preserve these tensors across a serialization boundary without changing the numerical result. The decode stage then loads the KV state and continues attention computation.

## Task

Implement the two functions:

```python
def serialize_kv(K: list[list[float]], V: list[list[float]]) -> bytes:
    ...

def decode_from_kv(Q: list[float], payload: bytes) -> list[float]:
    ...
```

`serialize_kv` receives the prefill worker's key cache `K` with shape $(n,d)$ and value cache `V` with shape $(n,h)$ and returns a byte payload containing enough information to reconstruct both arrays.

`decode_from_kv` receives decode queries `Q` with shape $(m,d)$ and the serialized payload. It must reconstruct the KV cache and return the attention output with shape $(m,h)$.

Use Python operations for the attention computation. The reconstructed output must match a fused fp64 attention computation. The serialization format is internal to your implementation, but it must round-trip exactly.

## Example

```python

K = [[1.0, 0.0], [0.0, 1.0]]
V = [[2.0], [4.0]]
Q = [[1.0, 0.0]]

payload = serialize_kv(K, V)
out = decode_from_kv(Q, payload)

# out is the attention result over the two cached tokens.
```

## What the gate checks

The gate computes a Python fp64 oracle for the fused prefill-plus-decode attention path:

$$
Y = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

It then compares the candidate output after a serialize and reload boundary using maximum absolute error. The metric `max_abs_err` must be less than $10^{-5}$.
