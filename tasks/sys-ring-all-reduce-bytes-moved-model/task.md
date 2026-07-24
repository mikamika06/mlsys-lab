## Context

In distributed data-parallel training, $N$ ranks each compute gradients on a local data partition and must aggregate them. The **ring all-reduce** arranges the ranks in a logical ring and completes the reduction in two phases:

1. **Scatter-reduce** ($N - 1$ steps): each rank sends a chunk of size $\frac{B}{N}$ bytes to its clockwise neighbor and receives a chunk from its counter-clockwise neighbor. After $N - 1$ steps every rank holds the fully reduced tensor.

2. **All-gather** ($N - 1$ steps): each rank broadcasts its slice of the reduced result using the same ring pattern, again sending $\frac{B}{N}$ bytes per step.

Here $B$ is the per-rank data size in bytes. In each phase a rank transmits $(N - 1)$ chunks of $\frac{B}{N}$ bytes, so the total **send volume per rank** is

$$\text{bytes\_sent} = 2 \cdot \frac{N - 1}{N} \cdot B .$$

The receive volume is symmetric, giving a total of $4 \cdot \frac{N - 1}{N} \cdot B$ bytes moved (send + receive), but the standard metric reported in the systems literature is the send volume above.

Three observations:

- When $N = 1$ there is no neighbor; $\text{bytes\_sent} = 0$.
- As $N \to \infty$ the fraction $\frac{N-1}{N} \to 1$, so per-rank communication approaches $2B$ regardless of cluster size.
- The algorithm is **bandwidth-optimal**: no all-reduce scheme can move fewer than $2 \cdot \frac{N-1}{N} \cdot B$ bytes per rank.

## Task

Implement `ring_allreduce_bytes_moved(n_ranks, data_bytes)`:

```python
def ring_allreduce_bytes_moved(n_ranks: int, data_bytes: int) -> float:
    """Return per-rank send volume in bytes for a ring all-reduce."""
    ...
```

- `n_ranks`: number of ranks in the ring ($N \ge 1$).
- `data_bytes`: per-rank data buffer size in bytes ($B \ge 0$).
- Returns a `float` equal to $2 \cdot \frac{N - 1}{N} \cdot B$.

Do **not** call any external libraries; the formula is pure arithmetic.

## Example

```python
ring_allreduce_bytes_moved(1, 1024)   # 0.0   — single rank, no communication
ring_allreduce_bytes_moved(2, 1024)   # 1024.0
ring_allreduce_bytes_moved(4, 1000)   # 1500.0
ring_allreduce_bytes_moved(8, 4096)   # 7168.0
```

## What the gate checks

The grader evaluates the formula $2 \cdot \frac{N - 1}{N} \cdot B$ directly for several $(N, B)$ pairs — including edge cases $N = 1$, $B = 0$, and a large $N$ — and compares each result against the student's return value. It passes only if every case matches within a relative tolerance of $10^{-6}$. No expected values are hardcoded; the grader recomputes the formula itself.
