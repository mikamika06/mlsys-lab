## Context

Attention layers score keys using a query vector. For a query $q$ and key $k_j$,
the scaled dot-product pre-softmax score is

$$
s_j = \frac{q^\top k_j}{\sqrt{d}} .
$$

In some systems the future query is uncertain. A useful approximation models
future queries as a Gaussian distribution

$$
q \sim \mathcal{N}(\mu, \Sigma),
$$

where $\mu$ and $\Sigma$ are estimated from observed query vectors.

For a fixed key $k_j$, the random score is also Gaussian:

$$
s_j \sim \mathcal{N}\left(
\frac{\mu^\top k_j}{\sqrt{d}},
\frac{k_j^\top \Sigma k_j}{d}
\right).
$$

The expected exponential score is

$$
\mathbb{E}[\exp(s_j)] =
\exp\left(
\frac{\mu^\top k_j}{\sqrt{d}}
+
\frac{1}{2}\frac{k_j^\top \Sigma k_j}{d}
\right).
$$

Because the exponential is monotonic, the quantity

$$
e_j =
\frac{\mu^\top k_j}{\sqrt{d}}
+
\frac{1}{2}\frac{k_j^\top \Sigma k_j}{d}
$$

can be used as an expected-attention ranking score.

## Task

Implement `expected_attention_scores(queries, keys, top_k)`:

```python
def expected_attention_scores(
    queries: np.ndarray,
    keys: np.ndarray,
    top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The input `queries` has shape $(n, d)$ and contains observed query vectors.
The input `keys` has shape $(m, d)$ and contains candidate attention keys.

Estimate $\mu$ and $\Sigma$ from the rows of `queries`, compute the expected
score $e_j$ for every key, and return:

1. A float64 NumPy array of shape $(m,)$ containing the expected scores.
2. An integer NumPy array containing the indices of the `top_k` keys sorted from
   highest expected score to lowest expected score.

Use NumPy operations for the matrix calculations.

## Example

```python
import numpy as np

queries = np.array([
    [1.0, 0.0],
    [0.0, 2.0],
    [1.0, 1.0],
])

keys = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
])

scores, indices = expected_attention_scores(queries, keys, 1)

# indices is [1] because the second key receives the larger
# Gaussian variance correction.
```

## What the gate checks

The gate builds several query and key matrices and computes the oracle result
from the Gaussian mean and covariance equations directly with NumPy.

The `rel_err` metric compares the returned expected scores with the oracle scores
and must satisfy $\mathrm{rel\_err} \le 10^{-4}$.

The `selected_exact` metric requires the returned top-key indices to exactly
match the oracle ranking.
