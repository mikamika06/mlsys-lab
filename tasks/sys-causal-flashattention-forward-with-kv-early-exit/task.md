## Context

FlashAttention computes attention without materializing the full attention matrix. It processes query and key/value blocks and maintains numerically stable softmax statistics while streaming through key/value tiles.

For a query vector $q_i$ and key/value matrices $K, V$, causal attention only allows positions $j \le i$:

$$
S_{ij} =
\begin{cases}
q_i K_j^\top / \sqrt{d}, & j \le i \\
-\infty, & j > i .
\end{cases}
$$

The output is

$$
O_i = \sum_j \mathrm{softmax}(S_i)_j V_j .
$$

The log-sum-exp value is

$$
\mathrm{LSE}_i = \log \sum_j \exp(S_{ij}),
$$

which is useful for backward passes and numerical verification.

A tiled implementation divides the sequence into query tiles and key/value tiles. For a causal mask, a query tile starting at row $r$ does not need to inspect key/value tiles whose starting column is greater than the last valid causal column. These future tiles can be skipped before any score computation.

## Task

Implement:

```python
def causal_flash_attention_forward(Q, K, V, tile_size=2):
    ...
```

Inputs are NumPy arrays:

- `Q` has shape $(n, d)$.
- `K` has shape $(n, d)$.
- `V` has shape $(n, dv)$.
- `tile_size` is a positive integer.

Return a tuple:

```python
(output, lse)
```

where:

- `output` has shape $(n, dv)$ and contains causal attention results.
- `lse` has shape $(n,)$ and contains the row-wise log-sum-exp values.

Use tiled processing. Do not compute score blocks for key/value tiles that are entirely above the causal diagonal. The implementation should keep intermediate score storage tile-sized rather than constructing the full $n \times n$ attention matrix.

The module may define a helper named `_score_kv_tile`. The grader uses tracing to verify that future KV tiles are never passed to this scoring step.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0],
              [0.0, 1.0]])
K = Q.copy()
V = np.array([[2.0], [4.0]])

out, lse = causal_flash_attention_forward(Q, K, V, tile_size=1)

# out contains:
# row 0: attention only over value 0
# row 1: attention over values 0 and 1
```

## What the gate checks

The numeric gate builds a dense causal NumPy reference implementation and compares the returned output and log-sum-exp values. The maximum absolute error must satisfy $\mathrm{max\_abs\_err} < 10^{-5}$.

The execution gate uses `sys.settrace` to observe calls to `_score_kv_tile`. It verifies that key/value tiles above the causal diagonal are skipped and never scored. A dense implementation that computes every query/key tile fails this gate even if its numerical result is correct.
