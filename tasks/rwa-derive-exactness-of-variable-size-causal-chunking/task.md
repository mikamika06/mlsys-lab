## Context

Causal attention computes each query output from itself and all keys before it in the
sequence. For queries $Q \in \mathbb{R}^{n \times d}$, keys $K \in \mathbb{R}^{n \times d}$,
and values $V \in \mathbb{R}^{n \times m}$, the causal attention output is

$$
O_i = \sum_{j=0}^{i} \frac{\exp(Q_i K_j^\top / \sqrt{d})}{\sum_{t=0}^{i}\exp(Q_i K_t^\top / \sqrt{d})} V_j .
$$

Large inference systems often process the sequence in chunks. A later query chunk can
reuse key-value blocks from all earlier chunks while applying the causal mask inside
the current chunk. The chunked computation is exact because the softmax is still
computed over the same set of visible keys as full prefill.

For a query chunk ending at position $e$, the visible key range is
$0 \ldots e$. Combining all previous key-value blocks with the current chunk's
causal mask produces the same result as evaluating the full causal attention matrix.

## Task

Implement `causal_chunk_attention(Q, K, V, chunks)`.

The inputs are list:
- `Q` has shape $(n, d)$.
- `K` has shape $(n, d)$.
- `V` has shape $(n, m)$.
- `chunks` is a list of positive integers whose sum is $n$.

Return an array of shape $(n, m)$ containing the attention output. Compute the
attention in chunks, where each chunk of queries attends to all keys from previous
chunks and the visible prefix of the current chunk. The output must be numerically
equivalent to full causal attention.

Use Python operations. The implementation should support different chunk sizes for the
same sequence length.

## Example

```python

Q = [[1., 0.], [0., 1.], [1., 1.]]
K = Q.copy()
V = [[1.], [2.], [3.]]

out = causal_chunk_attention(Q, K, V, [1, 2])
```

The first query attends only to the first value. The final two queries attend to the
complete visible prefix according to the causal rule.

## What the gate checks

The gate builds a Python fp64 full causal-attention oracle. It runs the submitted
chunked implementation on several variable chunk schedules and computes the largest
absolute difference

$$
\max_{i,j} |O^{chunk}_{ij} - O^{oracle}_{ij}|.
$$

The result must satisfy `max_abs_err <= 1e-5` for every tested schedule. A solution
that only attends inside each chunk fails because later chunks must include all earlier
key-value states.
