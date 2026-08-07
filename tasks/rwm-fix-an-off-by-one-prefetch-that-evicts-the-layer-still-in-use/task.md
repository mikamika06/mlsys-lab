## Context

Large transformer inference systems may keep only a small number of layer weights resident in memory. A scheduler uses a cache with a limited number of slots and prefetches future layers while computing the current layer.

For attention with query matrix $Q$, key matrix $K$, and value matrix $V$, the output is

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

A double-buffer scheduler has two cache slots. The important ordering constraint is that the current layer's attention must read its key and value matrices before the slot is reused for a prefetched layer.

An incorrect scheduler performs the eviction first:

$$
\text{evict layer } i \rightarrow \text{load layer } i+1 \rightarrow \text{compute layer } i .
$$

The correct order is:

$$
\text{compute layer } i \rightarrow \text{evict slot} \rightarrow \text{load layer } i+1 .
$$

## Task

Implement `scheduled_attention(layers, Qs, Ks, Vs)`.

`layers[i]` is a tuple `(K_i, V_i)` containing the key and value matrices for layer $i$. `Qs[i]` is the query matrix for layer $i$.

Return a list of attention outputs, one for each layer. The implementation should simulate a two-slot cache, but it must preserve the rule that a layer is consumed before its cache slot is overwritten.

The `Ks` and `Vs` arguments are unused and are provided only to match a production-style scheduler interface.

The returned arrays must be `float64` list.

## Example

```python

layers = [
    ([[1.0, 0.0]], [[2.0, 3.0]]),
    ([[0.0, 1.0]], [[4.0, 5.0]]),
]

Qs = [
    [[1.0, 0.0]],
    [[0.0, 1.0]],
]

out = scheduled_attention(layers, Qs, None, None)
```

The result contains the attention output for layer $0$ followed by the output for layer $1$.

## What the gate checks

The gate computes a Python oracle that has access to every layer and applies the attention equation directly.

The returned values are compared using

$$
\max_{i,j}|A_{ij}-\hat{A}_{ij}|.
$$

The error must be at most $10^{-6}$. A scheduler that overwrites the active cache slot before computing the current layer will use the wrong layer weights and fail.
