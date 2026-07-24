## Context

Paged key/value caches store transformer attention states in fixed-size blocks. A
token at absolute position $p$ is stored using a block index and an offset inside
that block.

For a block size $B$, the correct mapping is

$$
\mathrm{block\_idx} = \left\lfloor \frac{p}{B} \right\rfloor,\qquad
\mathrm{block\_offset} = p \bmod B .
$$

An off-by-one error in the block index changes which key and value vectors are
loaded. Attention then computes over incorrect tokens even though the attention
formula itself is unchanged.

For a query vector $q$ and key/value matrices $K,V$, scaled dot-product
attention is

$$
\mathrm{Attention}(q,K,V)
=
\mathrm{softmax}\left(\frac{qK^T}{\sqrt{d}}\right)V .
$$

A production implementation must gather the correct $K,V$ entries from the
paged cache before applying this operation.

## Task

Implement `slot_attention(K_cache, V_cache, Q, positions, B)`.

Arguments:

- `K_cache` is a NumPy array of shape $(N_b, B, H, D)$ containing key vectors
  stored by block.
- `V_cache` has the same shape and contains value vectors.
- `Q` has shape $(T, H, D)$ and contains one query per token position.
- `positions` is a length-$T$ integer array containing absolute token positions.
- `B` is the block size.

The function must:

1. Map each position $p$ to its correct block index and block offset.
2. Gather the corresponding keys and values from the cache in position order.
3. Compute attention for each query over all gathered tokens.
4. Return a NumPy array of shape $(T, H, D)$ with `float64` values.

The head dimension is $D$. The attention scale is $1/\sqrt{D}$.

## Example

```python
import numpy as np

# K_cache and V_cache contain block-packed tokens.
# slot_attention returns the same result as if the tokens were stored contiguously.
out = slot_attention(K_cache, V_cache, Q, positions, B)
```

## What the gate checks

The gate builds paged caches from contiguous NumPy arrays and computes a
reference implementation that directly applies attention to the contiguous
token sequence in `float64`.

The returned output is compared with the oracle using

$$
\max_{i,j,k} |y_{i,j,k} - \hat{y}_{i,j,k}|.
$$

The error must be less than $10^{-6}$. A slot mapping with
$\mathrm{block\_idx}=p\mathbin{//}(B-1)$ selects incorrect cache entries and
does not pass.
