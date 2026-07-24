## Context

Autoregressive language models generate one token at a time. During decoding, the attention keys and values from previous tokens are stored in a KV-cache so they do not need to be recomputed.

A key cache can be represented as a tensor $K \in \mathbb{R}^{T \times d}$, where $T$ is the maximum sequence length and $d$ is the head dimension. When a new token arrives at position $p$, its key vector $k_p$ must be written into row $p$:

$$
K[p, :] = k_p .
$$

The same rule applies to the value cache $V$. If the write uses the wrong offset, an earlier token can be overwritten and the attention computation will use incorrect history.

## Task

Implement `write_kv_cache(cache_k, cache_v, new_k, new_v, position)`.

The function receives:

- `cache_k`: a NumPy array of shape $(T, d)$ containing cached keys.
- `cache_v`: a NumPy array of shape $(T, d)$ containing cached values.
- `new_k`: a NumPy array of shape $(d,)$ containing the new key.
- `new_v`: a NumPy array of shape $(d,)$ containing the new value.
- `position`: a zero-based integer token position.

Return a tuple `(updated_k, updated_v)` where the rows at `position` contain `new_k` and `new_v`. Do not change other rows.

## Example

```python
import numpy as np

cache_k = np.zeros((4, 3))
cache_v = np.zeros((4, 3))

new_k = np.array([1.0, 2.0, 3.0])
new_v = np.array([4.0, 5.0, 6.0])

updated_k, updated_v = write_kv_cache(
    cache_k, cache_v, new_k, new_v, 2
)

# updated_k[2] is [1.0, 2.0, 3.0]
# updated_v[2] is [4.0, 5.0, 6.0]
```

## What the gate checks

The gate builds the expected cache update using a NumPy oracle that writes the new token at the requested zero-based position. The returned key and value caches are compared against this reference using the maximum absolute error:

$$
\mathrm{max\_abs\_err} = \max_i |x_i - \hat{x}_i|.
$$

The value must be below $10^{-5}$. Writing to `position - 1` or any other offset fails because it overwrites the wrong token.
