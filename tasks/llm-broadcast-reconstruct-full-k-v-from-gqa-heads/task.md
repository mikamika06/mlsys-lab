## Context

Grouped-query attention (GQA) stores fewer key and value heads than query heads.
A query head group shares the same key and value head. If there are
$n_q$ query heads and $n_{kv}$ stored key/value heads, the replication factor is

$$r = \frac{n_q}{n_{kv}}.$$

The compressed representation stores

$$K_{kv}, V_{kv} \in \mathbb{R}^{B \times n_{kv} \times S \times d}.$$

To run the attention computation as if it were standard multi-head attention (MHA),
the key and value tensors are broadcast back to

$$K_{mha}, V_{mha} \in \mathbb{R}^{B \times n_q \times S \times d}.$$

The reconstruction repeats each key/value head $r$ times in head order:

$$
K_{mha}[:, i r:(i+1)r, :, :] = K_{kv}[:, i:i+1, :, :]
$$

for each stored head index $i$.

## Task

Implement `expand_gqa_kv(kv, num_query_heads)`:

```python
def expand_gqa_kv(kv: np.ndarray, num_query_heads: int) -> np.ndarray:
    ...
```

The input `kv` is a NumPy array with shape $(B, n_{kv}, S, d)$ containing grouped key
or value heads. Return an array with shape
$(B, n_q, S, d)$ where every stored head has been repeated enough times to match
the query head count.

Use NumPy operations only. The output must preserve the input values and use
`float32` output dtype.

You may assume that $n_q$ is divisible by $n_{kv}$.

## Example

```python
import numpy as np

kv = np.array(
    [
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ]
    ],
    dtype=np.float32,
)

full = expand_gqa_kv(kv, 4)

# full.shape == (1, 4, 1, 2)
# heads are:
# [[1, 2], [1, 2], [3, 4], [3, 4]]
```

## What the gate checks

The gate compares the implementation against a NumPy oracle that reconstructs the
full MHA-equivalent tensor using exact head repetition. The maximum absolute error

$$
\max_i |x_i - \hat{x}_i|
$$

must be below $10^{-6}$.
